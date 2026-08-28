#!/usr/bin/env python3
"""S10-B5-B: export EUNN2Q01 quantized artifact (power-of-two shift scheme).

Frozen quantization scheme (v3, all power-of-two shifts, no float ops at
inference time):

  FT_SHIFT = 12
    q_ft_w  = round(w * 2^12), i16   (absmax 0.441 -> 1808, headroom 18x)
    q_ft_b  = round(b * 2^12), i32
    acc     = q_ft_b + sum(q_ft_w rows), i32
              PROVEN bound: |q_b|max + 31 * 32767 = 1,016,721 << 2^31

  Dense layers (l1, l2, out) share one pattern (NO input requantization:
  the MAC consumes the accumulator-precision activation directly):
    input a     : integer ClippedReLU output in A units, a in [0, QA=4096]
    weights     : q_w = round(w * 2^12), i16
                  (l1 absmax 0.338 -> 1387; l2 0.724 -> 2967; out 0.543 -> 2224)
    bias        : q_b = round(b * 2^24), i32
                  (term q_w*a = w*a_float*2^24, so z_int is in float*2^24
                   units; bias matches that scale)
    z_int       = q_b + sum(q_w * a), i32
                  PROVEN bounds (empirical q_w maxima; see layout JSON):
                    l1: |q_b|max + 256 * 1387 * 4096 ≈ 1.45e9  (1.5x margin)
                    l2:  |q_b|max +  32 * 2967 * 4096 ≈ 3.9e8  (5.5x margin)
                    out: |q_b|max +  32 * 2224 * 4096 ≈ 2.9e8  (7.4x margin)
    a_next      = clamp(shift_round(z_int, 12), 0, 4096)   (back to A units:
                  z_int >> 12 = z_float * 2^12 = A units)

  Output:
    raw   = shift_round(z_out_int, 12)   (A units)
    cp    = raw / 2^FT_SHIFT * 1000.0    (single final float conversion)

  Rounding rules (FROZEN):
    weight/bias quantization: round-half-away-from-zero
    z shift:                 arithmetic right shift, round half away from zero
    saturation:              clamp to container / [0, QA] only

Artifact layout (EUNN2Q01 v1, little-endian):
    magic[8]              "EUNN2Q01"
    version        u32    1
    inputs         u32    22528
    ft_width       u32    128
    target_scale   f32    1000.0
    ft_shift       u32    12
    dense_w_shift  u32    12     (q_w = round(w * 2^dense_w_shift))
    dense_z_shift  u32    12     (z_int >> dense_z_shift -> A units)
    qa             u32    4096   (integer ClippedReLU upper bound, A units)
    reserved       u32    0
    source_fp32_artifact_sha256   32 bytes
    source_checkpoint_sha256      32 bytes
    ft_weights     i16[22528][128]
    ft_bias        i32[128]
    l1_weight      i16[32][256]
    l1_bias        i32[32]
    l2_weight      i16[32][32]
    l2_bias        i32[32]
    out_weight     i16[1][32]
    out_bias       i32[1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MAGIC = b"EUNN2Q01"
FORMAT_VERSION = 1
INPUTS = 22528
FT_WIDTH = 128
TARGET_SCALE = 1000.0

FT_SHIFT = 12
DENSE_W_SHIFT = 12
DENSE_Z_SHIFT = 12
QA = 1 << FT_SHIFT  # 4096: integer ClippedReLU upper bound in A units

FROZEN_CHECKPOINT = Path(
    "data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt")
FROZEN_CHECKPOINT_SHA = (
    "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7")
FROZEN_FP32_ARTIFACT = Path("data/s10/b3/seed-20260818/nnue-v2-f32.bin")
FROZEN_FP32_ARTIFACT_SHA = (
    "9bf7adddf7b3b44affa5e26d2276b13d74566191a4eb4d0090fbde5a7afbc9fc")


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


def build_quantized_arrays(sd: dict) -> dict:
    ft_w_q = quantize_i16(sd["ft_weights.weight"], 1 << FT_SHIFT)
    ft_b_q = quantize_i32(sd["ft_bias"], 1 << FT_SHIFT)

    # Dense weights: i16 in 2^16 scale.
    l1_w_q = quantize_i16(sd["l1.weight"], 1 << DENSE_W_SHIFT)
    l2_w_q = quantize_i16(sd["l2.weight"], 1 << DENSE_W_SHIFT)
    out_w_q = quantize_i16(sd["out.weight"], 1 << DENSE_W_SHIFT)

    # Dense biases: z_int units = float * 2^(W_SHIFT - IN_SHIFT + FT_SHIFT)
    # (term q_w*d = w*a_float*2^24, so bias lives in 2^24 scale)
    bias_scale = 1 << (DENSE_W_SHIFT + FT_SHIFT)  # 2^24
    l1_b_q = quantize_i32(sd["l1.bias"], bias_scale)
    l2_b_q = quantize_i32(sd["l2.bias"], bias_scale)
    out_b_q = quantize_i32(sd["out.bias"], bias_scale)

    # PROVEN overflow bounds (worst case, not empirical).
    n_features_max = 31
    ft_bound = int(np.abs(ft_b_q).max()) + n_features_max * 32767
    q_w_absmax_l1 = int(np.abs(l1_w_q).max())
    q_w_absmax_l2 = int(np.abs(l2_w_q).max())
    q_w_absmax_out = int(np.abs(out_w_q).max())
    l1_bound = int(np.abs(l1_b_q).max()) + 256 * q_w_absmax_l1 * QA
    l2_bound = int(np.abs(l2_b_q).max()) + 32 * q_w_absmax_l2 * QA
    out_bound = int(np.abs(out_b_q).max()) + 32 * q_w_absmax_out * QA

    bounds = {
        "ft_accumulator_bound": ft_bound,
        "l1_mac_bound": l1_bound,
        "l2_mac_bound": l2_bound,
        "out_mac_bound": out_bound,
        "i32_max": (1 << 31) - 1,
        "ft_overflow": ft_bound > (1 << 31) - 1,
        "l1_overflow": l1_bound > (1 << 31) - 1,
        "l2_overflow": l2_bound > (1 << 31) - 1,
        "out_overflow": out_bound > (1 << 31) - 1,
    }
    if any(v for k, v in bounds.items() if k.endswith("_overflow")):
        raise SystemExit(f"PIPELINE_FAILURE: proven overflow {bounds}")

    return {
        "ft_w": ft_w_q, "ft_b": ft_b_q,
        "l1_w": l1_w_q, "l1_b": l1_b_q,
        "l2_w": l2_w_q, "l2_b": l2_b_q,
        "out_w": out_w_q, "out_b": out_b_q,
        "bounds": bounds,
    }


def artifact_bytes(q: dict, src_fp32_sha: str, src_ckpt_sha: str) -> bytes:
    import struct
    header = bytearray()
    header += MAGIC
    header += struct.pack(
        "<IIIfIIIII", FORMAT_VERSION, INPUTS, FT_WIDTH, TARGET_SCALE,
        FT_SHIFT, DENSE_W_SHIFT, DENSE_Z_SHIFT, QA,
        0)  # last u32 reserved (0)
    header += bytes.fromhex(src_fp32_sha)
    header += bytes.fromhex(src_ckpt_sha)
    payload = bytearray()
    payload += q["ft_w"].tobytes()
    payload += q["ft_b"].tobytes()
    payload += q["l1_w"].tobytes()
    payload += q["l1_b"].tobytes()
    payload += q["l2_w"].tobytes()
    payload += q["l2_b"].tobytes()
    payload += q["out_w"].tobytes()
    payload += q["out_b"].tobytes()
    return bytes(header) + bytes(payload)


def load_frozen_sd() -> tuple[dict, str]:
    data = FROZEN_CHECKPOINT.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != FROZEN_CHECKPOINT_SHA:
        raise SystemExit(f"PIPELINE_FAILURE: checkpoint SHA {sha} != frozen")
    ckpt = torch.load(FROZEN_CHECKPOINT, map_location="cpu",
                      weights_only=False)
    return ckpt["model_state_dict"], sha


def export(out_path: Path) -> dict:
    sd, ckpt_sha = load_frozen_sd()
    fp32_sha = hashlib.sha256(FROZEN_FP32_ARTIFACT.read_bytes()).hexdigest()
    if fp32_sha != FROZEN_FP32_ARTIFACT_SHA:
        raise SystemExit("PIPELINE_FAILURE: FP32 artifact SHA mismatch")
    q = build_quantized_arrays(sd)
    blob = artifact_bytes(q, fp32_sha, ckpt_sha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    return {
        "artifact_path": str(out_path),
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "total_bytes": len(blob),
        "magic": MAGIC.decode(),
        "format_version": FORMAT_VERSION,
        "inputs": INPUTS,
        "ft_width": FT_WIDTH,
        "shifts": {
            "ft": FT_SHIFT,
                        "dense_w": DENSE_W_SHIFT,
            "dense_z": DENSE_Z_SHIFT,
        },
        "qa": QA,
        "source_fp32_artifact_sha256": fp32_sha,
        "source_checkpoint_sha256": ckpt_sha,
        "proven_overflow_bounds": q["bounds"],
        "quantized_layer_sizes": {
            "ft_weights_bytes": q["ft_w"].nbytes,
            "ft_bias_bytes": q["ft_b"].nbytes,
            "l1_weight_bytes": q["l1_w"].nbytes,
            "l1_bias_bytes": q["l1_b"].nbytes,
            "l2_weight_bytes": q["l2_w"].nbytes,
            "l2_bias_bytes": q["l2_b"].nbytes,
            "out_weight_bytes": q["out_w"].nbytes,
            "out_bias_bytes": q["out_b"].nbytes,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("data/s10/b3/seed-20260818/nnue-v2-q01.bin"))
    ap.add_argument("--layout", type=Path, default=None)
    args = ap.parse_args()
    info = export(args.out)
    if args.layout is not None:
        args.layout.parent.mkdir(parents=True, exist_ok=True)
        args.layout.write_text(json.dumps(info, indent=2) + "\n",
                               encoding="utf-8")
    print(f"artifact {args.out} sha256={info['artifact_sha256']} "
          f"bytes={info['total_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
