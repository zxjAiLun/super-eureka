#!/usr/bin/env python3
"""S10-B5-A: calibration and provenance telemetry for the FROZEN v3 scheme.

Reports (a) FP32 parameter/activation telemetry over N train positions
(never holdout) and (b) the actual quantized arrays, shifts, and PROVEN
overflow bounds of the frozen EUNN2Q01 v3 scheme.

The scheme constants and quantized arrays are imported from
tools/s10/export_quantized.py — this module does NOT maintain a second set
of magic numbers. The final artifact must match
results/s10/s10-b5-quantized-layout.json field by field (enforced by
tools/s10/test_b5_consistency.py).

Output: results/s10/s10-b5-calibration.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.export_quantized import (
    DENSE_W_SHIFT, DENSE_Z_SHIFT, FT_SHIFT, QA,
    build_quantized_arrays, load_frozen_sd,
)
from tools.s10.train_nnue import (
    NNUE_INPUTS_V2, NnueModel, export_features_from_engine, load_dataset,
)

FROZEN_CHECKPOINT = Path(
    "data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt")
FROZEN_CHECKPOINT_SHA = (
    "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7")
ENGINE = Path("target/release/eureka")
TELEMETRY_POSITIONS = 10000
MAX_FEATURES_PER_PERSPECTIVE = 31  # startpos; 32 pieces minus own king


def tensor_stats(t: torch.Tensor) -> dict:
    t = t.detach().numpy().astype(np.float64)
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "absmax": float(np.abs(t).max()),
        "p999_abs": float(np.percentile(np.abs(t), 99.9)),
    }


def main() -> int:
    ckpt_sha = hashlib.sha256(FROZEN_CHECKPOINT.read_bytes()).hexdigest()
    if ckpt_sha != FROZEN_CHECKPOINT_SHA:
        print(f"FATAL: checkpoint SHA {ckpt_sha} != frozen")
        return 4

    sd, _ = load_frozen_sd()
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(sd)
    model.eval()

    # ---- (a) FP32 parameter statistics ---------------------------------------
    param_stats = {k: tensor_stats(v) for k, v in {
        "ft_weight": sd["ft_weights.weight"],
        "ft_bias": sd["ft_bias"],
        "l1_weight": sd["l1.weight"],
        "l1_bias": sd["l1.bias"],
        "l2_weight": sd["l2.weight"],
        "l2_bias": sd["l2.bias"],
        "out_weight": sd["out.weight"],
        "out_bias": sd["out.bias"],
    }.items()}

    # ---- FP32 activation telemetry over train positions ----------------------
    ds = load_dataset(Path("data/s10/s10-eval-v1-300k01"))
    corpus = []
    for r in ds["records"]:
        if len(corpus) >= TELEMETRY_POSITIONS:
            break
        if r["split"] != "train":
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
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
        })

    acc_pre = []
    l1_pre, l2_pre = [], []
    outputs = []
    CHUNK = 1000
    done = 0
    for c0 in range(0, len(items), CHUNK):
        chunk = items[c0:c0 + CHUNK]
        from tools.s10.train_nnue import EncodedSplit
        enc = EncodedSplit([
            {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in chunk])
        with torch.no_grad():
            stm_acc = (
                torch.nn.functional.embedding_bag(
                    enc.stm_indices, model.ft_weights.weight,
                    enc.stm_offsets, mode="sum")
                + model.ft_bias)
            nstm_acc = (
                torch.nn.functional.embedding_bag(
                    enc.nstm_indices, model.ft_weights.weight,
                    enc.nstm_offsets, mode="sum")
                + model.ft_bias)
            acc_pre.append(stm_acc.abs().max().item())
            acc_pre.append(nstm_acc.abs().max().item())
            clipped = torch.clamp(
                torch.cat([stm_acc, nstm_acc], dim=1), 0.0, 1.0)
            z1 = model.l1(clipped)
            l1_pre.append(z1.abs().max().item())
            a1 = torch.clamp(z1, 0.0, 1.0)
            z2 = model.l2(a1)
            l2_pre.append(z2.abs().max().item())
            a2 = torch.clamp(z2, 0.0, 1.0)
            outputs.append(model.out(a2).abs().max().item())
        done += len(chunk)

    telemetry = {
        "positions": done,
        "acc_pre_clip_absmax": max(acc_pre),
        "l1_preact_absmax": max(l1_pre),
        "l2_preact_absmax": max(l2_pre),
        "output_absmax": max(outputs),
    }

    # ---- (b) frozen v3 quantized arrays + proven bounds ----------------------
    q = build_quantized_arrays(sd)
    bounds = q["bounds"]

    result = {
        "stage": "s10_b5_calibration",
        "scheme_version": 3,
        "checkpoint_sha256": ckpt_sha,
        "telemetry_positions": telemetry["positions"],
        "holdout_used": False,
        "parameter_stats": param_stats,
        "activation_telemetry": telemetry,
        "scheme": {
            "ft": {
                "container": "i16",
                "shift": FT_SHIFT,
            },
            "dense": {
                "container": "i16",
                "weight_shift": DENSE_W_SHIFT,
                "input_requantization": False,
            },
            "bias_shift": DENSE_W_SHIFT + FT_SHIFT,
            "dense_z_shift": DENSE_Z_SHIFT,
            "qa": QA,
            "rounding": "quantize round-half-away; shift round-half-away; "
                        "integer ClippedReLU clamp(0, QA)",
            "mac": "z_int = q_b + sum(q_w * a) at accumulator precision",
        },
        "proven_overflow_bounds": bounds,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
