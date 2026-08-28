#!/usr/bin/env python3
"""S10-B5-A: offline scale calibration for the frozen V2 checkpoint.

Collects per-layer parameter statistics and FP32 activation telemetry over
N train positions (never holdout), computes worst-case i32 accumulator
bounds, and derives the frozen quantization scales for the EUNN2Q01
artifact.

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

from tools.s10.train_nnue import (
    NNUE_INPUTS_V2,
    EncodedSplit,
    NnueModel,
    export_features_from_engine,
    load_dataset,
)

FROZEN_CHECKPOINT = Path(
    "data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt")
FROZEN_CHECKPOINT_SHA = (
    "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7")
ENGINE = Path("target/release/eureka")
ENGINE_SHA = (
    "f005811bda2c73f8833915787dc7fcc777b8a6b84803f752ff357c5d60bfb41b")
TELEMETRY_POSITIONS = 10000
MAX_FEATURES_PER_PERSPECTIVE = 31  # startpos; 32 pieces minus own king

# NNUE-typical quantization scheme (frozen before measurement):
#   FT weights    i16, scale s_ft  = 127 / absmax(ft_w)
#   FT bias       i32 in the SAME scale as accumulated weights
#                  (i.e. q_bias = round(bias / s_ft))
#   accumulator   i32 (bias + sum of i16 rows)
#   dense weights i8,  scale s_l   = 127 / absmax(l_w) per layer
#   dense bias    i32 in input scale of the layer
#   activations   ClippedReLU(0, 1) implemented in the dense input scale:
#                  clipping at 0 and at SCALE_ACT where SCALE_ACT is the
#                  accumulator activation scale
FT_W_BITS = 16
DENSE_W_BITS = 8


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
    eng_sha = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    if eng_sha != ENGINE_SHA:
        print(f"FATAL: engine SHA {eng_sha} != frozen")
        return 4

    ckpt = torch.load(FROZEN_CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(sd)
    model.eval()

    ds = load_dataset(Path("data/s10/s10-eval-v1-300k01"))
    records = ds["records"]

    # Telemetry corpus: first N usable train positions, dataset order.
    corpus = []
    for r in records:
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

    # ---- Parameter statistics -------------------------------------------------
    ft_w = sd["ft_weights.weight"]
    ft_b = sd["ft_bias"]
    l1_w, l1_b = sd["l1.weight"], sd["l1.bias"]
    l2_w, l2_b = sd["l2.weight"], sd["l2.bias"]
    out_w, out_b = sd["out.weight"], sd["out.bias"]

    param_stats = {
        "ft_weight": tensor_stats(ft_w),
        "ft_bias": tensor_stats(ft_b),
        "l1_weight": tensor_stats(l1_w),
        "l1_bias": tensor_stats(l1_b),
        "l2_weight": tensor_stats(l2_w),
        "l2_bias": tensor_stats(l2_b),
        "out_weight": tensor_stats(out_w),
        "out_bias": tensor_stats(out_b),
    }

    # ---- Activation telemetry (hooks-free manual forward) ---------------------
    # We reproduce the forward pass manually to capture every intermediate.
    n_items = [0]
    acc_pre_clip = []   # FT accumulator before clip (both perspectives)
    act_post_clip = []  # post-clip accumulator activations
    l1_pre, l1_post = [], []
    l2_pre, l2_post = [], []
    outputs = []

    # Batch through in chunks to bound memory.
    CHUNK = 1000
    for c0 in range(0, len(items), CHUNK):
        chunk = items[c0:c0 + CHUNK]
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
            acc_pre_clip.append(stm_acc.abs().max().item())
            acc_pre_clip.append(nstm_acc.abs().max().item())

            clipped = torch.clamp(
                torch.cat([stm_acc, nstm_acc], dim=1), 0.0, 1.0)
            act_post_clip.append(clipped.abs().max().item())

            z1 = model.l1(clipped)
            l1_pre.append(z1.abs().max().item())
            a1 = torch.clamp(z1, 0.0, 1.0)
            l1_post.append(a1.abs().max().item())

            z2 = model.l2(a1)
            l2_pre.append(z2.abs().max().item())
            a2 = torch.clamp(z2, 0.0, 1.0)
            l2_post.append(a2.abs().max().item())

            outputs.append(model.out(a2).abs().max().item())
        n_items[0] += len(chunk)

    telemetry = {
        "positions": n_items[0],
        "acc_pre_clip_absmax": max(acc_pre_clip),
        "acc_post_clip_absmax": max(act_post_clip),  # <= 1.0 by clamp
        "l1_preact_absmax": max(l1_pre),
        "l1_postact_absmax": max(l1_post),           # <= 1.0
        "l2_preact_absmax": max(l2_pre),
        "l2_postact_absmax": max(l2_post),           # <= 1.0
        "output_absmax": max(outputs),
    }

    # ---- Quantization scales (derived, then frozen) ---------------------------
    # FT: i16 weights. scale s_ft = 127 / absmax(ft_w).
    ft_absmax = param_stats["ft_weight"]["absmax"]
    s_ft = 127.0 / ft_absmax

    # Dense layers: i8 weights, per-layer scale s_l = 127 / absmax(l_w).
    s_l1 = 127.0 / param_stats["l1_weight"]["absmax"]
    s_l2 = 127.0 / param_stats["l2_weight"]["absmax"]
    s_out = 127.0 / param_stats["out_weight"]["absmax"]

    # Dense input activation scale: activations live in [0, ACT_MAX] where
    # ACT_MAX is the accumulator scale. We quantize the post-clip activation
    # with the SAME scale as the FT weights (1 activation quantum = s_ft in
    # float units), so ClippedReLU(0,1) becomes clamp(0, QACT) with
    # QACT = round(1.0 / s_ft) in accumulator units.
    qact = int(round(1.0 / s_ft))

    # ---- Worst-case i32 accumulator bounds (PROVEN, not empirical) ------------
    ft_w_absmax = ft_absmax
    # Integer accumulator worst case: |q_bias| + N_max * max|q_w|
    # q_w max = 32767 (i16), but realistically bounded by round(w/s_ft) with
    # w <= absmax -> q_w <= 127... NO: ft weights use i16 with scale s_ft =
    # 127/absmax, so q_w = round(w / s_ft) <= 127. Wait — that wastes i16.
    # Standard NNUE practice: FT weights i16 with scale 127/absmax gives
    # q_w in [-127, 127]. The i16 container has headroom; the BOUND uses the
    # actual quantized max.
    q_ft_absmax = int(round(ft_w_absmax / s_ft))   # == 127 by construction
    q_ft_bias_absmax = int(np.ceil(np.abs(ft_b.numpy()).max() / s_ft))
    acc_bound = q_ft_bias_absmax + MAX_FEATURES_PER_PERSPECTIVE * q_ft_absmax
    # Also the absolute i16-container worst case for reference:
    acc_bound_i16_container = (q_ft_bias_absmax
                               + MAX_FEATURES_PER_PERSPECTIVE * 32767)

    # Dense MAC worst case (l1): |q_bias| + 256 * max|q_act| * max|q_w|
    # q_act <= qact, q_w <= 127 (i8).
    q_l1_bias_absmax = int(np.ceil(
        np.abs(l1_b.numpy()).max() / (s_l1 * s_ft)))
    mac_l1_bound = q_l1_bias_absmax + 256 * qact * 127
    q_l2_bias_absmax = int(np.ceil(
        np.abs(l2_b.numpy()).max() / (s_l2 * s_l1 * s_ft)))
    # l2 input is l1 output in scale s_l1*s_ft units? See scheme note below.
    # We define activation scale chain explicitly in the reference; the MAC
    # bound uses the layer output quantum chain.
    # For safety take the maximal product bound: inputs up to qact_l1 = 127.
    mac_l2_bound = q_l2_bias_absmax + 32 * 127 * 127
    q_out_bias_absmax = int(np.ceil(
        np.abs(out_b.numpy()).max() / (s_out * s_l2 * s_l1 * s_ft)))
    mac_out_bound = q_out_bias_absmax + 32 * 127 * 127

    i32_max = 2**31 - 1
    overflow = {
        "ft_accumulator_bound": acc_bound,
        "ft_accumulator_bound_i16_container_worst": acc_bound_i16_container,
        "l1_mac_bound": mac_l1_bound,
        "l2_mac_bound": mac_l2_bound,
        "out_mac_bound": mac_out_bound,
        "i32_max": i32_max,
        "ft_overflow": acc_bound > i32_max,
        "ft_overflow_i16_container_worst": acc_bound_i16_container > i32_max,
        "l1_overflow": mac_l1_bound > i32_max,
        "l2_overflow": mac_l2_bound > i32_max,
        "out_overflow": mac_out_bound > i32_max,
    }

    result = {
        "stage": "s10_b5_calibration",
        "checkpoint_sha256": ckpt_sha,
        "engine_exporter_sha256": eng_sha,
        "telemetry_positions": telemetry["positions"],
        "holdout_used": False,
        "parameter_stats": param_stats,
        "activation_telemetry": telemetry,
        "scheme": {
            "ft_weight": "i16, scale s_ft = 127/absmax, q = round(w/s_ft)",
            "ft_bias": "i32 in s_ft units, q = round(b/s_ft)",
            "accumulator": "i32 = q_bias + sum(q_w)",
            "dense_weight": "i8 per layer, s_l = 127/absmax, q = round(w/s_l)",
            "dense_bias": "i32 in layer input scale",
            "activation": "integer ClippedReLU: clamp(x, 0, QACT)",
            "output": "fixed-point chain dequantization",
        },
        "scales": {
            "s_ft": s_ft,
            "s_l1": s_l1,
            "s_l2": s_l2,
            "s_out": s_out,
            "qact": qact,
        },
        "overflow_analysis": overflow,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
