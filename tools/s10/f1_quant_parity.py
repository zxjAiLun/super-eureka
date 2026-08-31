"""S10-F1 B5 parity: raw-residual and composed-eval, Python vs Rust.

Two exact-integer layers (1000 deterministic validation positions):

  1. raw residual parity: the engine's `nnue-v2q-probe-batch` raw_output
     on the v2 material-residual artifact must equal a Python integer
     reference implementation of the SAME quantized forward pass
     (bit-exact: same i16/i32 weights, same shifts, same roundings);
  2. composed parity: material_cp_stm + residual_cp computed in Python
     (using the canonical values cross-checked against
     `bench material-batch`) must equal the engine-side composition the
     search will produce (`material_cp_stm` is a Rust integer too).

Also reports FP32-vs-quantized error statistics on the residual output
(mean / p95 / p99 / max) and the composed validation MAE, which is the
number the runtime actually delivers.

Usage:
    python tools/s10/f1_quant_parity.py \
        --checkpoint data/s10/f1/seed-20260820/checkpoint_v2_s20260820.pt \
        --quantized data/s10/f1/seed-20260820/nnue-v2-q01-material.bin \
        --dataset data/s10/s10-eval-v1-300k01 \
        --engine target/release/eureka \
        --out results/s10/s10-f1-quant-parity.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FT_SHIFT = 12
DENSE_W_SHIFT = 12
DENSE_Z_SHIFT = 12
QA = 1 << FT_SHIFT
TARGET_SCALE = 1000.0


def load_quantized(path: Path):
    """Parse the v2 EUNN2Q01 artifact in Python (weights + target mode)."""
    import numpy as np

    data = path.read_bytes()
    assert data[0:8] == b"EUNN2Q01", "bad magic"
    version = int.from_bytes(data[8:12], "little")
    inputs = int.from_bytes(data[12:16], "little")
    ft_width = int.from_bytes(data[16:20], "little")
    mode = int.from_bytes(data[40:44], "little")
    assert version == 2, f"expected v2, got {version}"
    assert mode == 1, f"expected target_mode=material_residual, got {mode}"
    assert inputs == 22528 and ft_width == 128

    off = 108
    ft_w = np.frombuffer(data, dtype="<i2", count=inputs * ft_width, offset=off)
    off += ft_w.nbytes
    ft_b = np.frombuffer(data, dtype="<i4", count=ft_width, offset=off)
    off += ft_b.nbytes
    l1_w = np.frombuffer(data, dtype="<i2", count=32 * 256, offset=off)
    off += l1_w.nbytes
    l1_b = np.frombuffer(data, dtype="<i4", count=32, offset=off)
    off += l1_b.nbytes
    l2_w = np.frombuffer(data, dtype="<i2", count=32 * 32, offset=off)
    off += l2_w.nbytes
    l2_b = np.frombuffer(data, dtype="<i4", count=32, offset=off)
    off += l2_b.nbytes
    out_w = np.frombuffer(data, dtype="<i2", count=32, offset=off)
    off += out_w.nbytes
    out_b = np.frombuffer(data, dtype="<i4", count=1, offset=off)
    return {
        "ft_w": ft_w.reshape(inputs, ft_width),
        "ft_b": ft_b,
        "l1_w": l1_w.reshape(32, 256),
        "l1_b": l1_b,
        "l2_w": l2_w.reshape(32, 32),
        "l2_b": l2_b,
        "out_w": out_w,
        "out_b": out_b,
    }


def shift_round(z: int, shift: int) -> int:
    denom = 1 << shift
    if z >= 0:
        return (z + denom // 2) // denom
    return -((-z + denom // 2) // denom)


def clamp_i(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def py_quant_forward(q, features_stm: list[int], features_nstm: list[int]) -> int:
    """Integer reference mirroring the Rust head exactly:
    out is NOT shifted (the cp conversion happens outside)."""
    ft_w = q["ft_w"]
    own = q["ft_b"].astype(object).copy()
    for idx in features_stm:
        own = own + ft_w[idx].astype(object)
    opp = q["ft_b"].astype(object).copy()
    for idx in features_nstm:
        opp = opp + ft_w[idx].astype(object)

    acts = [0] * 256
    for i in range(128):
        acts[i] = clamp_i(int(own[i]), 0, QA)
        acts[128 + i] = clamp_i(int(opp[i]), 0, QA)

    a1 = [0] * 32
    for o in range(32):
        z = int(q["l1_b"][o])
        row = q["l1_w"][o]
        for i in range(256):
            z += int(row[i]) * acts[i]
        a1[o] = clamp_i(shift_round(z, DENSE_Z_SHIFT), 0, QA)

    a2 = [0] * 32
    for o in range(32):
        z = int(q["l2_b"][o])
        row = q["l2_w"][o]
        for i in range(32):
            z += int(row[i]) * a1[i]
        a2[o] = clamp_i(shift_round(z, DENSE_Z_SHIFT), 0, QA)

    z_out = int(q["out_b"][0])
    for i in range(32):
        z_out += int(q["out_w"][i]) * a2[i]
    # Rust head: shift_round(z_out, DENSE_Z_SHIFT) — raw is in post-shift
    # units; cp = raw * 1000 / 2^FT_SHIFT.
    return shift_round(z_out, DENSE_Z_SHIFT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026083102)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from tools.s10.train_nnue import (
        EncodedSplit, NNUE_INPUTS_V2, NnueModel,
        export_features_from_engine, material_cp_stm_python,
    )

    records = []
    for line in (args.dataset / "part-0000.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            records.append(json.loads(line))
    labels = {}
    for line in (args.dataset / "labels.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    validation = [
        r for r in records
        if r["split"] == "validation"
        and labels[r["position_id"]].get("teacher_cp_stm") is not None
    ]
    sample = random.Random(args.seed).sample(validation, args.n)

    # features + material via the engine (single source of truth)
    exported = export_features_from_engine(args.engine, sample, "v2")
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("".join(f"{r['position_id']}|{r['fen']}\n" for r in sample))
        batch = fh.name
    try:
        proc = subprocess.run(
            [str(args.engine), "bench", "material-batch", "--batch", batch],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    finally:
        Path(batch).unlink(missing_ok=True)
    material = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            rec = json.loads(line)
            material[rec["position_id"]] = rec["material_cp_stm"]
    for r in sample:
        assert material[r["position_id"]] == material_cp_stm_python(r["fen"])

    # Rust raw residual outputs
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("".join(f"{r['position_id']}|{r['fen']}\n" for r in sample))
        qbatch = fh.name
    try:
        proc = subprocess.run(
            [str(args.engine), "bench", "nnue-v2q-probe-batch",
             "--model", str(args.quantized), "--batch", qbatch],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    finally:
        Path(qbatch).unlink(missing_ok=True)
    rust_raw = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            rec = json.loads(line)
            rust_raw[rec["position_id"]] = rec["raw_output"]

    # Layer 1: Python integer reference of the SAME quantized network
    q = load_quantized(args.quantized)
    raw_mismatches = 0
    composed_mismatches = 0
    residual_cp_errors = []
    composed_mae_errors = []
    for r in sample:
        pid = r["position_id"]
        exp = exported[pid]
        stm_white = r["fen"].split()[1] == "w"
        stm_feats = exp["white"] if stm_white else exp["black"]
        nstm_feats = exp["black"] if stm_white else exp["white"]
        py_raw = py_quant_forward(q, stm_feats, nstm_feats)
        rv = rust_raw[pid]
        if py_raw != rv:
            raw_mismatches += 1
            if raw_mismatches <= 3:
                print(f"RAW MISMATCH {pid}: python={py_raw} rust={rv}")
        # composed parity: material + residual in integer cp
        py_composed = material[pid] + py_raw * 1000 // (1 << FT_SHIFT)
        rust_composed = material[pid] + rv * 1000 // (1 << FT_SHIFT)
        if py_composed != rust_composed:
            composed_mismatches += 1

    # FP32 vs quantized residual error + composed validation MAE
    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    items = []
    for r in sample:
        pid = r["position_id"]
        exp = exported[pid]
        stm_white = r["fen"].split()[1] == "w"
        items.append({
            "stm": exp["white"] if stm_white else exp["black"],
            "nstm": exp["black"] if stm_white else exp["white"],
        })
    enc = EncodedSplit([
        {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items
    ])
    with torch.no_grad():
        fp32_scaled = model(
            enc.stm_indices, enc.stm_offsets,
            enc.nstm_indices, enc.nstm_offsets,
        ).numpy()
    for r, it, fs in zip(sample, items, fp32_scaled):
        pid = r["position_id"]
        fp32_cp = float(fs) * TARGET_SCALE
        q_cp = rust_raw[pid] * TARGET_SCALE / (1 << FT_SHIFT)
        residual_cp_errors.append(abs(fp32_cp - q_cp))
        teacher = float(labels[pid]["teacher_cp_stm"])
        composed = material[pid] + q_cp
        composed_mae_errors.append(abs(composed - teacher))

    errs = sorted(residual_cp_errors)
    n = len(errs)

    def pct(p):
        return errs[min(n - 1, int(round(p / 100.0 * (n - 1))))]

    report = {
        "schema_version": 1,
        "quantized_artifact": str(args.quantized),
        "n": args.n,
        "seed": args.seed,
        "layer1_raw_residual_parity": {
            "mismatches": raw_mismatches,
            "pass": raw_mismatches == 0,
        },
        "layer2_composed_parity": {
            "mismatches": composed_mismatches,
            "pass": composed_mismatches == 0,
        },
        "fp32_vs_quantized_residual_cp": {
            "mean": round(sum(errs) / n, 3),
            "p95": round(pct(95), 3),
            "p99": round(pct(99), 3),
            "max": round(errs[-1], 3),
        },
        "composed_validation_mae_cp": round(
            sum(composed_mae_errors) / len(composed_mae_errors), 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    ok = raw_mismatches == 0 and composed_mismatches == 0
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
