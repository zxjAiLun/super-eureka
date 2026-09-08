#!/usr/bin/env python3
"""S12 exporter: S12 checkpoint -> EUNN2Q01 v5 artifact (quantized).

v5 header (120 bytes) = v4 (112) + `head_kind` u32:
    magic[8] "EUNN2Q01"
    u32 version = 5
    u32 inputs = 23296 (V2R12)
    u32 ft_width = 256
    f32 target_scale = 1000.0
    u32 ft_shift = 12, dense_w_shift = 12, dense_z_shift = 12, qa = 4096
    u32 target_mode = 1 (material_residual)
    u32 feature_set = 1 (V2R12)
    u32 head_kind = 1 (SCReLU + linear(2W, 8) material-count buckets)
    32B source_fp32_artifact_sha256 (checkpoint file SHA here: the
         checkpoint IS the fp32 source of truth; no intermediate f32 bin)
    32B source_checkpoint_sha256
Payload:
    ft_weights i16[23296][256]  (@2^12)
    ft_bias    i32[256]         (@2^12)
    l1_weight  i16[8][512]      (@2^12, row-major per bucket)
    l1_bias    i32[8]           (@2^24)

Integer semantics (runtime contract):
    A = bias + sum(ft rows)                      (A units, i32, bound 61*32767+|b|)
    y = clamp(A,0,4096)^2 / 4096                 (SCReLU, [0,4096], exact int div)
    z = l1_bias[bucket] + sum_j q_w[bucket][j]*y[j]   (i64 accumulate)
    raw = round_half_away(z >> 12)
    eval_cp = material_cp_stm + raw * 1000 / 2^12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"EUNN2Q01"
VERSION = 5
FEATURE_SET_V2R12 = 1
HEAD_KIND_SCRELU_BUCKETS = 1
TARGET_MODE_MATERIAL_RESIDUAL = 1
INPUTS = 23296
FT_WIDTH = 256
TARGET_SCALE = 1000.0
FT_SHIFT = 12
DENSE_W_SHIFT = 12
DENSE_Z_SHIFT = 12
QA = 1 << FT_SHIFT
NUM_BUCKETS = 8
N_FEATURES_MAX = 61  # V2R12 proven per-perspective bound


def round_half_away(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_i16(t: torch.Tensor, scale: float) -> np.ndarray:
    q = round_half_away(t.detach().numpy() * scale)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.int16)


def quantize_i32(t: torch.Tensor, scale: float) -> np.ndarray:
    q = round_half_away(t.detach().numpy() * scale)
    q = q.astype(np.int64)
    if np.abs(q).max() >= (1 << 31):
        raise SystemExit("PIPELINE_FAILURE: i32 bias overflow")
    return q.astype(np.int32)


def export(ckpt_path: Path, out_path: Path) -> dict:
    ckpt_bytes = ckpt_path.read_bytes()
    ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    summary = ckpt["summary"]
    assert summary["schema"] == "s12-bullet-recipe", \
        f"not an S12 checkpoint: {summary.get('schema')}"
    assert int(sd["ft_bias"].shape[0]) == FT_WIDTH
    assert int(sd["ft_weights.weight"].shape[0]) == INPUTS
    l1w = sd["l1.weight"]                       # [8, 512]
    assert tuple(l1w.shape) == (NUM_BUCKETS, 2 * FT_WIDTH), l1w.shape
    l1b = sd["l1.bias"]                         # [8]

    ft_w_q = quantize_i16(sd["ft_weights.weight"], 1 << FT_SHIFT)
    ft_b_q = quantize_i32(sd["ft_bias"], 1 << FT_SHIFT)
    l1_w_q = quantize_i16(l1w, 1 << DENSE_W_SHIFT)
    l1_b_q = quantize_i32(l1b, 1 << (DENSE_W_SHIFT + FT_SHIFT))

    # proven MAC bounds (fail closed)
    ft_bound = int(np.abs(ft_b_q).max()) + N_FEATURES_MAX * int(np.abs(ft_w_q).max())
    if ft_bound > (1 << 31) - 1:
        raise SystemExit(f"PIPELINE_FAILURE: ft bound {ft_bound}")
    # L1 MAC accumulates in i64: 512 * 32768 * 4096 + 2^31 << 2^63 (by
    # construction); the i32 check only applies to the shifted OUTPUT.
    z_bound = int(np.abs(l1_b_q).max()) + 2 * FT_WIDTH * int(np.abs(l1_w_q).max()) * QA
    if z_bound >> DENSE_Z_SHIFT > (1 << 31) - 1:
        raise SystemExit(f"PIPELINE_FAILURE: output bound {z_bound >> DENSE_Z_SHIFT}")

    header = bytearray()
    header += MAGIC
    header += struct.pack(
        "<IIIfIIIIIII",
        VERSION, INPUTS, FT_WIDTH, TARGET_SCALE,
        FT_SHIFT, DENSE_W_SHIFT, DENSE_Z_SHIFT, QA,
        TARGET_MODE_MATERIAL_RESIDUAL, FEATURE_SET_V2R12,
        HEAD_KIND_SCRELU_BUCKETS,
    )
    header += bytes.fromhex(ckpt_sha)      # fp32 source = the checkpoint
    header += bytes.fromhex(ckpt_sha)
    assert len(header) == 8 + 4 * 11 + 64, len(header)

    payload = bytearray()
    payload += ft_w_q.tobytes()
    payload += ft_b_q.tobytes()
    payload += l1_w_q.tobytes()
    payload += l1_b_q.tobytes()

    blob = bytes(header) + bytes(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)

    return {
        "artifact_path": str(out_path),
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "total_bytes": len(blob),
        "magic": MAGIC.decode(),
        "format_version": VERSION,
        "feature_set": "v2r12",
        "head_kind": "screlu-8-material-buckets",
        "target_mode": "material-residual",
        "inputs": INPUTS,
        "ft_width": FT_WIDTH,
        "shifts": {"ft": FT_SHIFT, "dense_w": DENSE_W_SHIFT,
                   "dense_z": DENSE_Z_SHIFT},
        "qa": QA,
        "n_features_max_per_perspective": N_FEATURES_MAX,
        "source_checkpoint": str(ckpt_path),
        "source_checkpoint_sha256": ckpt_sha,
        "proven_bounds": {
            "ft_accumulator_bound": ft_bound,
            "l1_z_bound": z_bound,
            "l1_z_bound_shifted": z_bound >> DENSE_Z_SHIFT,
        },
        "quantized_layer_sizes": {
            "ft_weights_bytes": ft_w_q.nbytes,
            "ft_bias_bytes": ft_b_q.nbytes,
            "l1_weight_bytes": l1_w_q.nbytes,
            "l1_bias_bytes": l1_b_q.nbytes,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", type=Path, default=None)
    args = ap.parse_args()
    info = export(args.checkpoint, args.out)
    if args.layout is not None:
        args.layout.parent.mkdir(parents=True, exist_ok=True)
        args.layout.write_text(json.dumps(info, indent=2) + "\n",
                               encoding="utf-8")
    print(f"artifact {args.out} sha256={info['artifact_sha256']} "
          f"bytes={info['total_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
