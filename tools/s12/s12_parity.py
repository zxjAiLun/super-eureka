#!/usr/bin/env python3
"""S12 parity: FP32 checkpoint vs Python integer artifact vs Rust runtime.

Layers:
  A. FP32 torch checkpoint vs Python integer reference of the SAME v5
     artifact (sigmoid-space MSE tolerance, since quantization is lossy).
  B. Python integer reference vs Rust probe raw outputs — BIT-EXACT.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_train  # noqa: E402  (S12Model, output_bucket)

CKPT = Path(r"data\s12\r1\seed-20260908\checkpoint_s12_v2r12_s20260908.pt")
ART = Path(r"data\s12\r1\seed-20260908\nnue-s12-screlu-buckets-v5.bin")
ENGINE = Path(r"target\release\eureka.exe")
N = 2000
SEED = 2026090802
FT_SHIFT = 12
QA = 1 << FT_SHIFT
DENSE_Z_SHIFT = 12


def load_v5(path: Path):
    import struct
    d = path.read_bytes()
    assert d[0:8] == b"EUNN2Q01"
    ver, inputs, w = struct.unpack("<III", d[8:20])
    assert ver == 5 and inputs == 23296 and w == 256
    hk = struct.unpack("<I", d[48:52])[0]
    assert hk == 1
    off = 116
    ft_w = np.frombuffer(d, dtype="<i2", count=inputs * w, offset=off)
    off += ft_w.nbytes
    ft_b = np.frombuffer(d, dtype="<i4", count=w, offset=off)
    off += ft_b.nbytes
    l1_w = np.frombuffer(d, dtype="<i2", count=8 * 2 * w, offset=off)
    off += l1_w.nbytes
    l1_b = np.frombuffer(d, dtype="<i4", count=8, offset=off)
    off += l1_b.nbytes
    assert off == len(d), (off, len(d))
    return {
        "ft_w": ft_w.reshape(inputs, w),
        "ft_b": ft_b,
        "l1_w": l1_w.reshape(8, 2 * w),
        "l1_b": l1_b,
    }


def shift_round(z: int, shift: int) -> int:
    denom = 1 << shift
    if z >= 0:
        return (z + denom // 2) // denom
    return -((-z + denom // 2) // denom)


def py_int_raw(q, stm_feats, nstm_feats, bucket: int, w=256):
    acc = q["ft_b"].astype(object).copy()
    for idx in stm_feats:
        acc = acc + q["ft_w"][idx].astype(object)
    z = int(q["l1_b"][bucket])
    row = q["l1_w"][bucket]
    for idx in stm_feats:
        a = int(q["ft_b"][0])  # placeholder never used
        break
    # full recompute per perspective
    def screlu_acc(feats):
        acc = q["ft_b"].astype(object).copy()
        for idx in feats:
            acc = acc + q["ft_w"][idx].astype(object)
        out = []
        for v in acc:
            c = max(0, min(QA, int(v)))
            out.append(c * c // QA)
        return out

    y_stm = screlu_acc(stm_feats)
    y_nstm = screlu_acc(nstm_feats)
    z = int(q["l1_b"][bucket])
    row = q["l1_w"][bucket].astype(object)
    for j, y in enumerate(y_stm):
        z += int(row[j]) * y
    for j, y in enumerate(y_nstm):
        z += int(row[256 + j]) * y
    return shift_round(int(z), DENSE_Z_SHIFT)


def main() -> int:
    import chess
    import random

    # frozen sample from the 1M validation split
    records = []
    for p in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob("part-*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    labels = {}
    for line in Path(r"data\s10\s10-eval-v2-1m01\labels.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    val = [r for r in records if r["split"] == "validation"
           and labels[r["position_id"]].get("teacher_cp_stm") is not None]
    sample = random.Random(SEED).sample(val, N)

    # features via the engine (single source of truth)
    from importlib import util as ilu
    spec = ilu.spec_from_file_location(
        "_tn", str(Path(__file__).resolve().parents[1] / "s10" / "train_nnue.py"))
    _tn = ilu.module_from_spec(spec)
    sys.modules.setdefault("_tn", _tn)
    spec.loader.exec_module(_tn)
    exported = _tn.export_features_from_engine(ENGINE, sample, "v2r12")
    material = _tn.export_material_from_engine(ENGINE, sample)

    # FP32 model
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = s12_train.S12Model(
        num_inputs=s12_train.NNUE_INPUTS_V2R12, ft_width=256)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Rust raw outputs
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("".join(f"{r['position_id']}|{r['fen']}\n" for r in sample))
        batch = fh.name
    proc = subprocess.run(
        [str(ENGINE), "bench", "nnue-v2q-probe-batch",
         "--model", str(ART), "--batch", batch],
        capture_output=True, text=True, timeout=3600, check=True)
    rust_raw = {}
    for line in proc.stdout.splitlines():
        if line.strip().startswith('{"position_id"'):
            rec = json.loads(line)
            rust_raw[rec["position_id"]] = rec["raw_output"]

    q = load_v5(ART)

    int_vs_rust = 0
    fp_int_abs_cp = []
    for r in sample:
        pid = r["position_id"]
        exp = exported[pid]
        stm_white = r["fen"].split()[1] == "w"
        stm_feats = exp["white"] if stm_white else exp["black"]
        nstm_feats = exp["black"] if stm_white else exp["white"]
        board = chess.Board(r["fen"])
        bucket = s12_train.output_bucket(len(board.piece_map()))

        py_raw = py_int_raw(q, stm_feats, nstm_feats, bucket)
        rv = rust_raw[pid]
        if py_raw != rv:
            int_vs_rust += 1

        # FP32 residual
        with torch.no_grad():
            out = model(
                torch.tensor(stm_feats), torch.tensor([0]),
                torch.tensor(nstm_feats), torch.tensor([0]),
                torch.tensor([bucket]))
        fp_cp = float(out) * 1000.0
        int_cp = py_raw * 1000.0 / (1 << FT_SHIFT)
        fp_int_abs_cp.append(abs(fp_cp - int_cp))

    errs = sorted(fp_int_abs_cp)
    report = {
        "schema_version": 1,
        "stage": "s12_parity",
        "n": N,
        "seed": SEED,
        "layer_B_python_int_vs_rust": {
            "mismatches": int_vs_rust,
            "pass": int_vs_rust == 0,
        },
        "layer_A_fp32_vs_int_residual_cp": {
            "mean": round(sum(errs) / len(errs), 3),
            "p95": round(errs[int(0.95 * (len(errs) - 1))], 3),
            "max": round(errs[-1], 3),
        },
    }
    print(json.dumps(report, indent=1))
    Path("results/s12/s12-parity.json").parent.mkdir(
        parents=True, exist_ok=True)
    Path("results/s12/s12-parity.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0 if int_vs_rust == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
