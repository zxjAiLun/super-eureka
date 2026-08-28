#!/usr/bin/env python3
"""S10-B4 export: B3 production checkpoint -> deterministic FP32 artifact.

Format `EUNN2F32` (bench-only parity bridge, NOT a production model format):

Layout (little-endian, no padding):
    magic[8]            "EUNN2F32"
    version      u32    1
    inputs       u32    22528 (V2)
    ft_width     u32    128
    target_scale f32    1000.0
    checkpoint_sha256   32 raw bytes
    ft_weights          float32[22528][128]  (input-major)
    ft_bias             float32[128]
    l1_weight           float32[32][256]    (row-major, out x in)
    l1_bias             float32[32]
    l2_weight           float32[32][32]
    l2_bias             float32[32]
    out_weight          float32[1][32]
    out_bias            float32[1]

Fail-closed: checkpoint SHA256 must match the frozen B3 selection, tensor
keys/shapes/dtypes must match the production V2 architecture, and every
payload float must be finite.
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

MAGIC = b"EUNN2F32"
FORMAT_VERSION = 1
INPUTS = 22528
FT_WIDTH = 128
TARGET_SCALE = 1000.0

# Frozen S10-B3 production candidate (results/s10/s10-b3-selection.json)
EXPECTED_CHECKPOINT_SHA = (
    "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7")
EXPECTED_SEED = 20260818
EXPECTED_DATASET_SHA = (
    "503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3")
EXPECTED_LABELS_SHA = (
    "bcd49da1ece75a15591e135d5bcf6d036608b1759d6a00e639f3e344e516116f")

HEADER_BYTES = 8 + 4 + 4 + 4 + 4 + 32
STATE_SHAPES = {
    "ft_weights.weight": (INPUTS, FT_WIDTH),
    "ft_bias": (FT_WIDTH,),
    "l1.weight": (32, FT_WIDTH * 2),
    "l1.bias": (32,),
    "l2.weight": (32, 32),
    "l2.bias": (32,),
    "out.weight": (1, 32),
    "out.bias": (1,),
}


def tensor_f32_le_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.float32:
        raise SystemExit(f"PIPELINE_FAILURE: dtype {tensor.dtype} != float32")
    if tensor.device.type != "cpu":
        raise SystemExit(f"PIPELINE_FAILURE: device {tensor.device} != cpu")
    if not tensor.is_contiguous():
        raise SystemExit("PIPELINE_FAILURE: tensor is not contiguous")
    if not torch.isfinite(tensor).all():
        raise SystemExit("PIPELINE_FAILURE: non-finite value in tensor")
    array = tensor.detach().numpy()
    return np.ascontiguousarray(array, dtype="<f4").tobytes()


def validate_checkpoint(path: Path) -> dict:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != EXPECTED_CHECKPOINT_SHA:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint SHA256 {sha} != frozen B3 "
            f"selection {EXPECTED_CHECKPOINT_SHA}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    summary = ckpt["summary"]
    checks = {
        "seed": summary["seed"] == EXPECTED_SEED,
        "feature_set": summary["feature_set"] == "v2",
        "dataset_sha256": summary["dataset_sha256"] == EXPECTED_DATASET_SHA,
        "labels_sha256": summary["labels_sha256"] == EXPECTED_LABELS_SHA,
        "holdout_observed": summary.get("holdout_observed") is False,
        "num_inputs": summary["architecture"]["num_inputs"] == INPUTS,
        "ft_width": summary["architecture"]["ft_width"] == FT_WIDTH,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"PIPELINE_FAILURE: summary mismatch: {bad}")
    sd = ckpt["model_state_dict"]
    if set(sd) != set(STATE_SHAPES):
        raise SystemExit(
            f"PIPELINE_FAILURE: state-dict keys {sorted(sd)} != "
            f"{sorted(STATE_SHAPES)}")
    for key, shape in STATE_SHAPES.items():
        if tuple(sd[key].shape) != shape:
            raise SystemExit(
                f"PIPELINE_FAILURE: tensor {key} shape {tuple(sd[key].shape)}"
                f" != {shape}")
        if not torch.isfinite(sd[key]).all():
            raise SystemExit(f"PIPELINE_FAILURE: non-finite value in {key}")
    return {"sha": sha, "state_dict": sd, "summary": summary}


def export_artifact(ckpt_path: Path, out_path: Path) -> dict:
    info = validate_checkpoint(ckpt_path)
    sd = info["state_dict"]

    ft_weights = sd["ft_weights.weight"]            # [22528, 128] input-major
    ft_bias = sd["ft_bias"].contiguous()
    l1_weight = sd["l1.weight"].contiguous()        # [32, 256] row-major
    l1_bias = sd["l1.bias"].contiguous()
    l2_weight = sd["l2.weight"].contiguous()
    l2_bias = sd["l2.bias"].contiguous()
    out_weight = sd["out.weight"].contiguous()
    out_bias = sd["out.bias"].contiguous()

    header = bytearray()
    header += MAGIC
    header += __import__("struct").pack(
        "<IIIf", FORMAT_VERSION, INPUTS, FT_WIDTH, TARGET_SCALE)
    header += bytes.fromhex(info["sha"])

    payload = bytearray()
    for tensor in (ft_weights, ft_bias, l1_weight, l1_bias,
                   l2_weight, l2_bias, out_weight, out_bias):
        payload += tensor_f32_le_bytes(tensor)

    blob = bytes(header) + bytes(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    artifact_sha = hashlib.sha256(blob).hexdigest()

    return {
        "artifact_path": str(out_path),
        "artifact_sha256": artifact_sha,
        "total_bytes": len(blob),
        "magic": MAGIC.decode(),
        "format_version": FORMAT_VERSION,
        "inputs": INPUTS,
        "ft_width": FT_WIDTH,
        "target_scale": TARGET_SCALE,
        "checkpoint_sha256": info["sha"],
        "seed": EXPECTED_SEED,
        "format": "EUNN2F32 (S10-B4 bench-only parity bridge)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", type=Path, default=None)
    args = ap.parse_args()
    info = export_artifact(args.checkpoint.resolve(), args.out.resolve())
    if args.layout is not None:
        args.layout.parent.mkdir(parents=True, exist_ok=True)
        args.layout.write_text(json.dumps(info, indent=2) + "\n",
                               encoding="utf-8")
    print(f"artifact {args.out} sha256={info['artifact_sha256']} "
          f"bytes={info['total_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
