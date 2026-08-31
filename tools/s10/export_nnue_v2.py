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


def validate_checkpoint(path: Path, expected_sha: str | None,
                        expected_seed: int | None,
                        expected_dataset_sha: str | None,
                        expected_labels_sha: str | None,
                        expected_target_mode: str | None) -> dict:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint SHA256 {sha} != frozen "
            f"selection {expected_sha}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    summary = ckpt["summary"]
    checks = {}
    if expected_seed is not None:
        checks["seed"] = summary["seed"] == expected_seed
    checks["feature_set"] = summary["feature_set"] == "v2"
    if expected_dataset_sha is not None:
        checks["dataset_sha256"] = (
            summary["dataset_sha256"] == expected_dataset_sha)
    if expected_labels_sha is not None:
        checks["labels_sha256"] = summary["labels_sha256"] == expected_labels_sha
    # holdout_observed is informational: for S10-F1 the winner's weights are
    # selected purely on validation, and the single winner-only holdout
    # evaluation happens BEFORE export without touching any parameter.
    checks["num_inputs"] = summary["architecture"]["num_inputs"] == INPUTS
    checks["ft_width"] = summary["architecture"]["ft_width"] == FT_WIDTH
    if expected_target_mode is not None:
        checks["target_mode"] = (
            summary.get("target_mode") == expected_target_mode)
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


def export_artifact(ckpt_path: Path, out_path: Path,
                    expected_sha: str | None = None,
                    expected_seed: int | None = None,
                    expected_dataset_sha: str | None = None,
                    expected_labels_sha: str | None = None,
                    expected_target_mode: str | None = None) -> dict:
    info = validate_checkpoint(
        ckpt_path, expected_sha, expected_seed, expected_dataset_sha,
        expected_labels_sha, expected_target_mode)
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
        "seed": info["summary"]["seed"],
        "target_mode": info["summary"].get("target_mode"),
        "format": "EUNN2F32 (S10-B4 bench-only parity bridge)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", type=Path, default=None)
    ap.add_argument("--checkpoint-sha", default=None,
                    help="fail-closed checkpoint SHA (omit to use any)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset-sha", default=None)
    ap.add_argument("--labels-sha", default=None)
    ap.add_argument("--target-mode", choices=["cp", "material-residual"],
                    default=None,
                    help="require this target_mode in the checkpoint summary")
    args = ap.parse_args()
    info = export_artifact(
        args.checkpoint.resolve(), args.out.resolve(),
        expected_sha=args.checkpoint_sha,
        expected_seed=args.seed,
        expected_dataset_sha=args.dataset_sha,
        expected_labels_sha=args.labels_sha,
        expected_target_mode=args.target_mode,
    )
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
