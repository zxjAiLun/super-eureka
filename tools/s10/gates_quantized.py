#!/usr/bin/env python3
"""S10-B5 gate harness.

Gate 1: Python integer reference vs Rust integer runtime — bit-exact raw
         integer output on >= 1000 positions.
Gate 2: quantized vs frozen FP32 — error distribution on >= 10000
         non-holdout positions (mean <= 1cp, p95 <= 2cp, p99 <= 4cp).
Gate 3: validation teacher-MAE delta (quantized vs FP32) <= +1cp.

Outputs results/s10/s10-b5-gates.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.integer_reference import IntegerNnueV2
from tools.s10.train_nnue import (
    NNUE_INPUTS_V2, TARGET_SCALE, EncodedSplit, NnueModel,
    export_features_from_engine, load_dataset,
)

FROZEN_CHECKPOINT = Path(
    "data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt")
FROZEN_CHECKPOINT_SHA = (
    "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7")
FROZEN_Q_ARTIFACT = Path("data/s10/b3/seed-20260818/nnue-v2-q01.bin")
FROZEN_Q_ARTIFACT_SHA = (
    "b51a79b19999aeed974c2279eef60b01f890248c7d006cbe3d504cc7c0f28b9a")
ENGINE = Path("target/release/eureka")

GATE1_POSITIONS = 1000
GATE2_POSITIONS = 10000
GATE2_MEAN_CP = 1.0
GATE2_P95_CP = 2.0
GATE2_P99_CP = 4.0
GATE3_DELTA_CP = 1.0


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rust_batch(batch_text: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="s10-b5-rust-") as tmp:
        bp = Path(tmp) / "batch.txt"
        bp.write_text(batch_text, encoding="utf-8")
        proc = subprocess.run(
            [str(ENGINE), "bench", "nnue-v2q-probe-batch",
             "--model", str(FROZEN_Q_ARTIFACT), "--batch", str(bp)],
            capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise SystemExit(
                f"FATAL: rust exit {proc.returncode}: {proc.stderr[:500]}")
    return [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]


def main() -> int:
    # Frozen identities.
    if sha_of(FROZEN_CHECKPOINT) != FROZEN_CHECKPOINT_SHA:
        print("FATAL: checkpoint SHA mismatch")
        return 4
    if sha_of(FROZEN_Q_ARTIFACT) != FROZEN_Q_ARTIFACT_SHA:
        print("FATAL: quantized artifact SHA mismatch")
        return 4

    iref = IntegerNnueV2.from_frozen_checkpoint()

    ckpt = torch.load(FROZEN_CHECKPOINT, map_location="cpu",
                      weights_only=False)
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = load_dataset(Path("data/s10/s10-eval-v1-300k01"))
    records = ds["records"]

    # ---------------- Corpus: first GATE2_POSITIONS usable non-holdout -----
    corpus = []
    for r in records:
        if len(corpus) >= GATE2_POSITIONS:
            break
        if r["split"] == "holdout":
            continue
        if ds["labels"][r["position_id"]].get("teacher_cp_stm") is None:
            continue
        corpus.append(r)

    exported = export_features_from_engine(ENGINE, corpus, "v2")
    items = []
    for r in corpus:
        exp = exported[r["position_id"]]
        stm_is_white = r["fen"].split()[1] == "w"
        items.append({
            "record": r,
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
        })

    # FP32 reference predictions (batched).
    enc = EncodedSplit([
        {"stm": it["stm"], "nstm": it["nstm"],
         "target_scaled": 0.0, "target_cp": 0.0} for it in items])
    with torch.no_grad():
        fp_preds = model(
            enc.stm_indices, enc.stm_offsets,
            enc.nstm_indices, enc.nstm_offsets,
        ).numpy() * TARGET_SCALE

    # Python integer reference predictions.
    py_raw = [iref.evaluate_raw(it["stm"], it["nstm"]) for it in items]
    py_cp = np.array([raw / 4096.0 * 1000.0 for raw in py_raw])

    # ---------------- Gate 1: Python vs Rust bit-exact ---------------------
    g1_items = items[:GATE1_POSITIONS]
    batch_text = "".join(
        f"{it['record']['position_id']}|{it['record']['fen']}\n"
        for it in g1_items)
    rust_out = rust_batch(batch_text)
    rust_ids = [o["position_id"] for o in rust_out]
    if rust_ids != [it["record"]["position_id"] for it in g1_items]:
        print("FATAL: rust order/id mismatch")
        return 4
    rust_raw = [o["raw_output"] for o in rust_out]
    g1_mismatches = sum(
        1 for i in range(GATE1_POSITIONS) if rust_raw[i] != py_raw[i])
    gate1_pass = g1_mismatches == 0

    # ---------------- Gate 2: quantized vs FP32 ----------------------------
    errs = np.abs(py_cp[:GATE2_POSITIONS] - fp_preds[:GATE2_POSITIONS])
    gate2 = {
        "positions": int(len(errs)),
        "mean_abs_error_cp": float(errs.mean()),
        "p50_abs_error_cp": float(np.percentile(errs, 50)),
        "p95_abs_error_cp": float(np.percentile(errs, 95)),
        "p99_abs_error_cp": float(np.percentile(errs, 99)),
        "max_abs_error_cp": float(errs.max()),
    }
    gate2_pass = (gate2["mean_abs_error_cp"] <= GATE2_MEAN_CP
                  and gate2["p95_abs_error_cp"] <= GATE2_P95_CP
                  and gate2["p99_abs_error_cp"] <= GATE2_P99_CP)

    # ---------------- Gate 3: validation teacher-MAE delta -----------------
    val_records = [r for r in records if r["split"] == "validation"
                   and ds["labels"][r["position_id"]].get("teacher_cp_stm")
                   is not None]
    val_exported = export_features_from_engine(ENGINE, val_records, "v2")
    val_items = []
    for r in val_records:
        exp = val_exported[r["position_id"]]
        stm_is_white = r["fen"].split()[1] == "w"
        target_cp = float(ds["labels"][r["position_id"]]["teacher_cp_stm"])
        target_cp = max(-2000.0, min(2000.0, target_cp))
        val_items.append({
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
            "target_cp": target_cp,
            "target_scaled": target_cp / 1000.0,
        })
    venc = EncodedSplit(val_items)
    with torch.no_grad():
        vpreds = model(
            venc.stm_indices, venc.stm_offsets,
            venc.nstm_indices, venc.nstm_offsets,
        ).numpy() * TARGET_SCALE
    fp_val_mae = float(np.abs(vpreds - venc.raw_cps.numpy()).mean())
    q_val_pred = np.array([
        iref.evaluate_raw(it["stm"], it["nstm"]) / 4096.0 * 1000.0
        for it in val_items])
    q_val_mae = float(np.abs(q_val_pred - venc.raw_cps.numpy()).mean())
    gate3 = {
        "validation_positions": len(val_items),
        "fp32_validation_mae_cp": fp_val_mae,
        "quantized_validation_mae_cp": q_val_mae,
        "delta_mae_cp": q_val_mae - fp_val_mae,
    }
    gate3_pass = gate3["delta_mae_cp"] <= GATE3_DELTA_CP

    result = {
        "stage": "s10_b5_gates",
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA,
        "quantized_artifact_sha256": FROZEN_Q_ARTIFACT_SHA,
        "engine_sha256": sha_of(ENGINE),
        "holdout_used": False,
        "gate1_python_rust_bit_exact": {
            "positions": GATE1_POSITIONS,
            "mismatches": g1_mismatches,
            "passed": gate1_pass,
        },
        "gate2_quantized_vs_fp32": {**gate2, "passed": gate2_pass},
        "gate3_validation_mae_delta": {**gate3, "passed": gate3_pass},
        "gates": {
            "gate2_thresholds": {"mean": GATE2_MEAN_CP, "p95": GATE2_P95_CP,
                                 "p99": GATE2_P99_CP},
            "gate3_threshold_delta_cp": GATE3_DELTA_CP,
        },
        "all_passed": gate1_pass and gate2_pass and gate3_pass,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
