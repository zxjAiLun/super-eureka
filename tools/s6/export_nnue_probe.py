#!/usr/bin/env python3
"""S6-N2 probe artifact exporter: checkpoint -> local float32 .bin artifact.

The artifact format is EXPLICITLY named "S6-N2 probe format" — it is a
bench-only local bridge, NOT a generic/production NNUE model format.

Layout (little-endian, no padding):
    magic[8]            "EUNN1F32"
    version      u32    1
    inputs       u32    40960
    width        u32    32
    target_scale f32    1000.0
    checkpoint_sha256   32 raw bytes
    features.weight     float32[40960][32]   (transposed input-major)
    acc_bias            float32[32]
    head.weight         float32[64]          (own 32, opponent 32 order)
    head.bias           float32[1]

All payload values must be finite; any NaN/Inf, missing key, shape mismatch,
or checkpoint SHA mismatch fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"EUNN1F32"
FORMAT_VERSION = 1
EXPECTED_CHECKPOINT_SHA = "6bfdba6d7d9cc034d55d8bfe433ebb3b0d6f48d78afa2351f3ef465ac9003a66"
EXPECTED_INPUTS = 40960
EXPECTED_WIDTH = 32
EXPECTED_BEST_EPOCH = 3
EXPECTED_DATASET_SHA = "3a3483fd46fd5a570c4c62b7d93378efc80eafbab43ec155db5ac5894fbc6a9d"
EXPECTED_LABELS_SHA = "78dd8d52a34d1dd10a5d09cb3295be8f3a91a495d808fbd8b0cb68d31d668aa5"
TARGET_SCALE = 1000.0

HEADER_BYTES = 8 + 4 + 4 + 4 + 4 + 32
PAYLOAD_BYTES = (EXPECTED_INPUTS * EXPECTED_WIDTH * 4
                 + EXPECTED_WIDTH * 4
                 + 64 * 4
                 + 1 * 4)
TOTAL_BYTES = HEADER_BYTES + PAYLOAD_BYTES


def checkpoint_metadata_ok(ckpt: dict) -> None:
    """Architecture / tensor shape / finiteness validation (SHA-independent)."""
    arch = ckpt.get("architecture", {})
    checks = {
        "inputs": arch.get("inputs") == EXPECTED_INPUTS,
        "width": arch.get("width") == EXPECTED_WIDTH,
        "best_epoch": ckpt.get("best_epoch") == EXPECTED_BEST_EPOCH,
        "dataset_sha": ckpt.get("dataset_sha256") == EXPECTED_DATASET_SHA,
        "labels_sha": ckpt.get("labels_sha256") == EXPECTED_LABELS_SHA,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise SystemExit(f"PIPELINE_FAILURE: checkpoint metadata mismatch: {bad}")
    sd = ckpt["state_dict"]
    expected_shapes = {
        "features.weight": (EXPECTED_WIDTH, EXPECTED_INPUTS),
        "acc_bias": (EXPECTED_WIDTH,),
        "head.weight": (1, EXPECTED_WIDTH * 2),
        "head.bias": (1,),
    }
    if set(sd) != set(expected_shapes):
        raise SystemExit(
            f"PIPELINE_FAILURE: tensor keys {sorted(sd)} != "
            f"{sorted(expected_shapes)}")
    for key, shape in expected_shapes.items():
        if tuple(sd[key].shape) != shape:
            raise SystemExit(
                f"PIPELINE_FAILURE: tensor {key} shape {tuple(sd[key].shape)} "
                f"!= {shape}")
    for key, tensor in sd.items():
        if not torch.isfinite(tensor).all():
            raise SystemExit(f"PIPELINE_FAILURE: non-finite value in {key}")


def validate_checkpoint(path: Path) -> dict:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != EXPECTED_CHECKPOINT_SHA:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint SHA256 {sha} != expected "
            f"{EXPECTED_CHECKPOINT_SHA}")
    ckpt = torch.load(path, weights_only=True)
    checkpoint_metadata_ok(ckpt)
    return {"sha": sha, "state_dict": ckpt["state_dict"], "checkpoint": ckpt}


def build_artifact_bytes(features_t: torch.Tensor, acc_bias: torch.Tensor,
                         head_weight: torch.Tensor, head_bias: torch.Tensor,
                         checkpoint_sha: str) -> bytes:
    """Assemble the little-endian S6-N2 probe artifact (no SHA validation)."""
    header = bytearray()
    header += MAGIC
    header += struct.pack("<IIIf", FORMAT_VERSION, EXPECTED_INPUTS,
                          EXPECTED_WIDTH, TARGET_SCALE)
    header += bytes.fromhex(checkpoint_sha)

    payload = bytearray()
    payload += features_t.numpy().tobytes()
    payload += acc_bias.numpy().tobytes()
    payload += head_weight.numpy().tobytes()
    payload += head_bias.numpy().tobytes()

    blob = bytes(header) + bytes(payload)
    if len(blob) != TOTAL_BYTES:
        raise SystemExit(
            f"PIPELINE_FAILURE: artifact length {len(blob)} != expected {TOTAL_BYTES}")
    return blob


def export_artifact(ckpt_path: Path, out_path: Path) -> dict:
    info = validate_checkpoint(ckpt_path)
    sd = info["state_dict"]

    features_t = sd["features.weight"].t().contiguous()  # [40960][32]
    acc_bias = sd["acc_bias"].contiguous()
    head_weight = sd["head.weight"].reshape(-1).contiguous()  # 64, own-then-opp
    head_bias = sd["head.bias"].reshape(-1).contiguous()  # 1

    blob = build_artifact_bytes(features_t, acc_bias, head_weight, head_bias,
                                info["sha"])
    out_path.write_bytes(blob)
    artifact_sha = hashlib.sha256(blob).hexdigest()

    offsets = {
        "header": 0,
        "features_weight": HEADER_BYTES,
        "acc_bias": HEADER_BYTES + EXPECTED_INPUTS * EXPECTED_WIDTH * 4,
        "head_weight": HEADER_BYTES + EXPECTED_INPUTS * EXPECTED_WIDTH * 4
                       + EXPECTED_WIDTH * 4,
        "head_bias": HEADER_BYTES + EXPECTED_INPUTS * EXPECTED_WIDTH * 4
                     + EXPECTED_WIDTH * 4 + 64 * 4,
        "end": len(blob),
    }
    return {
        "artifact_path": str(out_path),
        "artifact_sha256": artifact_sha,
        "total_bytes": len(blob),
        "header_bytes": HEADER_BYTES,
        "payload_bytes": PAYLOAD_BYTES,
        "offsets": offsets,
        "checkpoint_sha256": info["sha"],
        "format": "S6-N2 probe format (bench-only, not production)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("data/s6/models/s6-n1-probe.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/s6/models/s6-n1-probe-f32.bin"))
    ap.add_argument("--layout", type=Path, default=None)
    args = ap.parse_args()
    info = export_artifact(args.checkpoint.resolve(), args.out)
    if args.layout is not None:
        args.layout.parent.mkdir(parents=True, exist_ok=True)
        args.layout.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
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
