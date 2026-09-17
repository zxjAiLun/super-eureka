#!/usr/bin/env python3
"""S16 export: merge the training-time shared table, THEN quantize once.

Exports an S16 checkpoint to the SAME v5 artifact format as S12/S14/S15, so the
Rust loader, search, and incremental update are all unchanged.

Merge rule (owner-frozen):
    merged[:22528] = W + repeat(F, 32)      # F has 704 rows, 32 repeats
    merged[22528:] = R12                    # untouched

ORDERING IS THE POINT: the merge happens in FLOAT, and quantization runs ONCE
on the merged table. Quantizing W and F separately and adding the integers
would round twice and produce a different model than training evaluated.

Range policy: `s12_export.quantize_i16` silently clips to [-32768, 32767]. For
this exporter that would hide a genuine merge blow-up, so we check the merged
float table's quantized range explicitly and FAIL LOUDLY instead.

Usage:
  python tools/s16/s16_export.py \
      --checkpoint data/s16/run/checkpoint_s16_v2r12_s20260915.pt \
      --out data/s16/run/nnue-s16-shared-v5.bin \
      --layout data/s16/run/nnue-s16-shared-v5.layout.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"tools\s12")
import s12_export as E  # noqa: E402  (frozen header/payload + quantizers)

V2_BASE = 22528
SHARED_PERIOD = 704
INPUTS = E.INPUTS          # 23296
FT_WIDTH = E.FT_WIDTH      # 256
I16_MIN, I16_MAX = -32768, 32767


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", type=Path, default=None)
    args = ap.parse_args()

    ckpt_bytes = args.checkpoint.read_bytes()
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    summary = ckpt["summary"]

    if summary.get("schema") != "s16-shared-factoriser":
        raise SystemExit(
            f"FAIL CLOSED: not an S16 checkpoint: {summary.get('schema')!r}")

    for key in ("ft_weights.weight", "ft_shared", "ft_bias", "l1.weight",
                "l1.bias"):
        if key not in sd:
            raise SystemExit(f"FAIL CLOSED: checkpoint missing {key}")

    W = sd["ft_weights.weight"]
    F = sd["ft_shared"]
    if int(W.shape[0]) != INPUTS:
        raise SystemExit(f"FAIL CLOSED: ft_weights rows {W.shape[0]} != {INPUTS}")
    if int(F.shape[0]) != SHARED_PERIOD + 1:
        raise SystemExit(
            f"FAIL CLOSED: ft_shared rows {F.shape[0]} != {SHARED_PERIOD + 1} "
            f"(704 real + 1 sentinel)")
    if int(W.shape[1]) != FT_WIDTH or int(F.shape[1]) != FT_WIDTH:
        raise SystemExit("FAIL CLOSED: FT width mismatch")

    # --- the sentinel row must be exactly zero; it exists only so R12
    #     indices can be routed to "no shared contribution".
    sent = F[SHARED_PERIOD]
    if not bool(torch.all(sent == 0)):
        raise SystemExit(
            "FAIL CLOSED: ft_shared sentinel row is non-zero; R12 rows would "
            "receive a shared contribution they were never trained with")

    # --- merge in FLOAT (single quantization, after merge)
    merged = W.clone()
    reps = V2_BASE // SHARED_PERIOD                     # 32
    merged[:V2_BASE] = merged[:V2_BASE] + F[:SHARED_PERIOD].repeat(reps, 1)

    # --- explicit range check (do NOT let clip() hide a blow-up)
    scaled = merged.detach().numpy().astype(np.float64) * (1 << E.FT_SHIFT)
    lo, hi = float(scaled.min()), float(scaled.max())
    if lo < I16_MIN or hi > I16_MAX:
        raise SystemExit(
            f"PIPELINE_FAILURE: merged FT table out of int16 range "
            f"after scaling by 2^{E.FT_SHIFT}: min={lo:.1f} max={hi:.1f} "
            f"(allowed [{I16_MIN}, {I16_MAX}]). Refusing to silently clip.")

    ft_w_q = E.quantize_i16(merged, 1 << E.FT_SHIFT)
    # post-condition: nothing was clipped away
    if ft_w_q.min() == I16_MIN or ft_w_q.max() == I16_MAX:
        raise SystemExit(
            "PIPELINE_FAILURE: quantized merged table hit an int16 rail "
            f"(min={ft_w_q.min()} max={ft_w_q.max()}); merge range check and "
            "quantizer disagree")

    ft_b_q = E.quantize_i32(sd["ft_bias"], 1 << E.FT_SHIFT)
    l1w = sd["l1.weight"]
    if tuple(l1w.shape) != (E.NUM_BUCKETS, 2 * FT_WIDTH):
        raise SystemExit(f"FAIL CLOSED: l1.weight shape {tuple(l1w.shape)}")
    l1_w_q = E.quantize_i16(l1w, 1 << E.DENSE_W_SHIFT)
    l1_b_q = E.quantize_i32(sd["l1.bias"],
                            1 << (E.DENSE_W_SHIFT + E.FT_SHIFT))

    ft_bound = int(np.abs(ft_b_q).max()) + E.N_FEATURES_MAX * int(
        np.abs(ft_w_q).max())
    if ft_bound > (1 << 31) - 1:
        raise SystemExit(f"PIPELINE_FAILURE: ft bound {ft_bound}")
    z_bound = (int(np.abs(l1_b_q).max())
               + 2 * FT_WIDTH * int(np.abs(l1_w_q).max()) * E.QA)
    if z_bound >> E.DENSE_Z_SHIFT > (1 << 31) - 1:
        raise SystemExit(
            f"PIPELINE_FAILURE: output bound {z_bound >> E.DENSE_Z_SHIFT}")

    header = bytearray()
    header += E.MAGIC
    header += struct.pack(
        "<IIIfIIIIIII",
        E.VERSION, E.INPUTS, E.FT_WIDTH, E.TARGET_SCALE,
        E.FT_SHIFT, E.DENSE_W_SHIFT, E.DENSE_Z_SHIFT, E.QA,
        E.TARGET_MODE_MATERIAL_RESIDUAL, E.FEATURE_SET_V2R12,
        E.HEAD_KIND_SCRELU_BUCKETS,
    )
    header += bytes.fromhex(ckpt_sha)
    header += bytes.fromhex(ckpt_sha)
    assert len(header) == 8 + 4 * 11 + 64, len(header)

    payload = bytearray()
    payload += ft_w_q.tobytes()
    payload += ft_b_q.tobytes()
    payload += l1_w_q.tobytes()
    payload += l1_b_q.tobytes()

    blob = bytes(header) + bytes(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    out_sha = hashlib.sha256(blob).hexdigest()

    info = {
        "artifact_path": str(args.out),
        "artifact_sha256": out_sha,
        "total_bytes": len(blob),
        "magic": E.MAGIC.decode(),
        "format_version": E.VERSION,
        "feature_set": "v2r12",
        "head_kind": "screlu-8-material-buckets",
        "target_mode": "material-residual",
        "inputs": E.INPUTS,
        "ft_width": FT_WIDTH,
        "shifts": {"ft": E.FT_SHIFT, "dense_w": E.DENSE_W_SHIFT,
                   "dense_z": E.DENSE_Z_SHIFT},
        "qa": E.QA,
        "n_features_max_per_perspective": E.N_FEATURES_MAX,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": ckpt_sha,
        "s16_merge": {
            "rule": ("merged[:22528] = W + repeat(F,32); merged[22528:] = R12; "
                     "merge in float, quantize once"),
            "shared_period": SHARED_PERIOD,
            "shared_repeats": reps,
            "v2_rows": V2_BASE,
            "r12_rows": INPUTS - V2_BASE,
            "merged_scaled_min": round(lo, 3),
            "merged_scaled_max": round(hi, 3),
            "clip_applied": False,
        },
        "proven_bounds": {
            "ft_accumulator_bound": ft_bound,
            "l1_z_bound": z_bound,
            "l1_z_bound_shifted": z_bound >> E.DENSE_Z_SHIFT,
        },
        "quantized_layer_sizes": {
            "ft_weights_bytes": ft_w_q.nbytes,
            "ft_bias_bytes": ft_b_q.nbytes,
            "l1_weight_bytes": l1_w_q.nbytes,
            "l1_bias_bytes": l1_b_q.nbytes,
        },
    }
    print(f"artifact {args.out} sha256={out_sha} bytes={len(blob)}")
    print(f"merged float range (x2^{E.FT_SHIFT}): "
          f"[{lo:.1f}, {hi:.1f}]  -> int16 clean, no clipping")
    if args.layout is not None:
        args.layout.parent.mkdir(parents=True, exist_ok=True)
        args.layout.write_text(json.dumps(info, indent=2) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
