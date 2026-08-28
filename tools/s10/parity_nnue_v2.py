#!/usr/bin/env python3
"""S10-B4 parity harness: Python FP32 reference vs Rust full-refresh V2.

Builds a parity corpus from the frozen dataset (train + validation splits
ONLY — never holdout), runs the Python NnueModel forward pass on the frozen
checkpoint, runs `bench nnue-v2-probe-batch` on the exported EUNN2F32
artifact, and compares every prediction.

Gate: max_abs_error_cp <= MAX_ABS_ERROR_CP (default 1e-2, i.e. far below
1 cp — float operation-ordering noise only).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tools.s10.train_nnue import (
    NNUE_INPUTS_V2,
    EncodedSplit,
    NnueModel,
    load_dataset,
)

MAX_ABS_ERROR_CP = 1e-2
CORPUS_SIZE = 1000
FROZEN_CHECKPOINT = Path(
    "data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt")
FROZEN_CHECKPOINT_SHA = (
    "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7")
FROZEN_ARTIFACT = Path("data/s10/b3/seed-20260818/nnue-v2-f32.bin")
FROZEN_ARTIFACT_SHA = (
    "9bf7adddf7b3b44affa5e26d2276b13d74566191a4eb4d0090fbde5a7afbc9fc")
ENGINE = Path("target/release/eureka")
ENGINE_SHA = (
    "f005811bda2c73f8833915787dc7fcc777b8a6b84803f752ff357c5d60bfb41b")


def main() -> int:
    # 0. Verify frozen identities (checkpoint, artifact, engine).
    ckpt_sha = hashlib.sha256(FROZEN_CHECKPOINT.read_bytes()).hexdigest()
    if ckpt_sha != FROZEN_CHECKPOINT_SHA:
        print(f"FATAL: checkpoint SHA {ckpt_sha} != frozen")
        return 4
    art_sha = hashlib.sha256(FROZEN_ARTIFACT.read_bytes()).hexdigest()
    if art_sha != FROZEN_ARTIFACT_SHA:
        print(f"FATAL: artifact SHA {art_sha} != frozen")
        return 4
    eng_sha = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    if eng_sha != ENGINE_SHA:
        print(f"FATAL: engine SHA {eng_sha} != frozen")
        return 4

    # 1. Load frozen dataset (fail-closed checks inside).
    ds = load_dataset(Path("data/s10/s10-eval-v1-300k01"))
    records = ds["records"]

    # 2. Deterministic parity corpus: first N records with split != holdout
    #    and usable teacher_cp_stm, in dataset order.
    corpus = []
    for r in records:
        if len(corpus) >= CORPUS_SIZE:
            break
        if r["split"] == "holdout":
            continue
        cp = ds["labels"][r["position_id"]].get("teacher_cp_stm")
        if cp is None:
            continue
        corpus.append(r)
    if len(corpus) < CORPUS_SIZE:
        print(f"FATAL: corpus too small ({len(corpus)})")
        return 4

    # 3. Python reference forward pass on the frozen checkpoint.
    ckpt = torch.load(FROZEN_CHECKPOINT, map_location="cpu", weights_only=False)
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Feature export via the engine exporter (single source of truth).
    from tools.s10.train_nnue import export_features_from_engine
    exported = export_features_from_engine(ENGINE, corpus, "v2")

    items = []
    for r in corpus:
        exp = exported[r["position_id"]]
        stm_is_white = r["fen"].split()[1] == "w"
        items.append({
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
        })

    enc = EncodedSplit([
        {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items
    ])
    with torch.no_grad():
        preds = model(
            enc.stm_indices, enc.stm_offsets,
            enc.nstm_indices, enc.nstm_offsets,
        ).numpy()
    py_cp = preds * 1000.0

    # 4. Rust full-refresh inference over the same corpus.
    with tempfile.TemporaryDirectory(prefix="s10-b4-parity-") as tmp:
        batch_path = Path(tmp) / "batch.txt"
        batch_path.write_text(
            "".join(f"{r['position_id']}|{r['fen']}\n" for r in corpus),
            encoding="utf-8")
        proc = subprocess.run(
            [str(ENGINE), "bench", "nnue-v2-probe-batch",
             "--model", str(FROZEN_ARTIFACT),
             "--batch", str(batch_path)],
            capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            print(f"FATAL: rust batch exit {proc.returncode}: "
                  f"{proc.stderr[:500]}")
            return 4

    rust_cp = []
    rust_ids = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rust_ids.append(rec["position_id"])
        rust_cp.append(rec["prediction_cp"])

    if rust_ids != [r["position_id"] for r in corpus]:
        print("FATAL: rust output order/ids mismatch")
        return 4
    if len(rust_cp) != len(py_cp):
        print(f"FATAL: count mismatch {len(rust_cp)} != {len(py_cp)}")
        return 4

    # 5. Compare.
    errors = [abs(float(a) - float(b)) for a, b in zip(py_cp, rust_cp)]
    max_err = float(max(errors))
    mean_err = float(sum(errors) / len(errors))

    split_counts = {}
    for r in corpus:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    result = {
        "stage": "s10_b4_parity",
        "checkpoint_sha256": ckpt_sha,
        "artifact_sha256": art_sha,
        "engine_sha256": eng_sha,
        "corpus_size": len(corpus),
        "corpus_splits": split_counts,
        "corpus_source": "first 1000 non-holdout usable records of "
                         "s10-eval-v1-300k01, dataset order",
        "holdout_positions_used": 0,
        "max_abs_error_cp": max_err,
        "mean_abs_error_cp": mean_err,
        "gate_max_abs_error_cp": MAX_ABS_ERROR_CP,
        "passed": max_err <= MAX_ABS_ERROR_CP,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
