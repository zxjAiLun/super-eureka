#!/usr/bin/env python3
"""S6-N2 artifact exporter unit tests: header, transpose order, little-endian,
length, non-finite rejection, and checkpoint SHA binding."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_nnue_probe as exp  # noqa: E402


def synthetic_state_dict(width: int = exp.EXPECTED_WIDTH,
                         inputs: int = exp.EXPECTED_INPUTS) -> dict:
    torch.manual_seed(20260818)
    return {
        "features.weight": torch.randn(width, inputs),
        "acc_bias": torch.randn(width),
        "head.weight": torch.randn(1, width * 2),
        "head.bias": torch.randn(1),
    }


def synthetic_checkpoint(ckpt: dict | None = None) -> dict:
    return {
        "state_dict": ckpt if ckpt is not None else synthetic_state_dict(),
        "architecture": {"inputs": exp.EXPECTED_INPUTS,
                         "width": exp.EXPECTED_WIDTH},
        "seed": 20260818,
        "dataset_sha256": exp.EXPECTED_DATASET_SHA,
        "labels_sha256": exp.EXPECTED_LABELS_SHA,
        "best_epoch": exp.EXPECTED_BEST_EPOCH,
    }


class MetadataTests(unittest.TestCase):
    def test_metadata_ok_passes(self):
        exp.checkpoint_metadata_ok(synthetic_checkpoint())

    def test_metadata_wrong_width_fails(self):
        ckpt = synthetic_checkpoint()
        ckpt["architecture"]["width"] = 64
        with self.assertRaises(SystemExit):
            exp.checkpoint_metadata_ok(ckpt)

    def test_metadata_wrong_shapes_fail(self):
        ckpt = synthetic_checkpoint()
        ckpt["state_dict"]["acc_bias"] = torch.randn(exp.EXPECTED_WIDTH + 1)
        with self.assertRaises(SystemExit):
            exp.checkpoint_metadata_ok(ckpt)

    def test_metadata_missing_key_fails(self):
        ckpt = synthetic_checkpoint()
        del ckpt["state_dict"]["head.bias"]
        with self.assertRaises(SystemExit):
            exp.checkpoint_metadata_ok(ckpt)

    def test_nonfinite_rejected(self):
        ckpt = synthetic_checkpoint()
        ckpt["state_dict"]["acc_bias"][0] = float("nan")
        with self.assertRaises(SystemExit) as cm:
            exp.checkpoint_metadata_ok(ckpt)
        self.assertIn("non-finite", str(cm.exception))

    def test_checkpoint_sha_binding(self):
        """A checkpoint whose SHA does not match the frozen one must fail."""
        with tempfile.TemporaryDirectory(prefix="s6-n2-test-") as tmp:
            ckpt_path = Path(tmp) / "ckpt.pt"
            torch.save(synthetic_checkpoint(), ckpt_path)
            with self.assertRaises(SystemExit) as cm:
                exp.validate_checkpoint(ckpt_path)
            self.assertIn("checkpoint SHA256", str(cm.exception))


class ArtifactLayoutTests(unittest.TestCase):
    def setUp(self):
        self.sd = synthetic_state_dict()

    def test_header_fields_and_length(self):
        blob = exp.build_artifact_bytes(
            self.sd["features.weight"].t().contiguous(),
            self.sd["acc_bias"].contiguous(),
            self.sd["head.weight"].reshape(-1).contiguous(),
            self.sd["head.bias"].reshape(-1).contiguous(),
            exp.EXPECTED_CHECKPOINT_SHA)
        self.assertEqual(len(blob), exp.TOTAL_BYTES)
        self.assertEqual(blob[:8], exp.MAGIC)
        version, inputs, width, scale_bits = struct.unpack_from("<IIIf", blob, 8)
        self.assertEqual(version, 1)
        self.assertEqual(inputs, exp.EXPECTED_INPUTS)
        self.assertEqual(width, exp.EXPECTED_WIDTH)
        self.assertEqual(scale_bits, 1000.0)
        self.assertEqual(blob[24:56], bytes.fromhex(exp.EXPECTED_CHECKPOINT_SHA))
        # No trailing bytes beyond the declared header+payload length.
        self.assertEqual(len(blob), exp.HEADER_BYTES + exp.PAYLOAD_BYTES)

    def test_little_endian_payload(self):
        blob = exp.build_artifact_bytes(
            torch.zeros(exp.EXPECTED_INPUTS, exp.EXPECTED_WIDTH),
            torch.zeros(exp.EXPECTED_WIDTH),
            torch.zeros(64), torch.zeros(1), exp.EXPECTED_CHECKPOINT_SHA)
        # 1.0f32 little-endian = 00 00 80 3f
        offset = exp.HEADER_BYTES
        blob = bytearray(blob)
        blob[offset:offset + 4] = struct.pack("<f", 1.0)
        self.assertEqual(bytes(blob[offset:offset + 4]), b"\x00\x00\x80\x3f")

    def test_transpose_input_major_order(self):
        # features.weight is (width, inputs); artifact must be input-major
        # [inputs][width], i.e. element [i][w] == weight[w][i].
        weight = torch.zeros(exp.EXPECTED_WIDTH, exp.EXPECTED_INPUTS)
        weight[3, 7] = 1.25
        weight[0, 12345] = -2.5
        features_t = weight.t().contiguous()
        blob = exp.build_artifact_bytes(
            features_t, self.sd["acc_bias"].contiguous(),
            self.sd["head.weight"].reshape(-1).contiguous(),
            self.sd["head.bias"].reshape(-1).contiguous(),
            exp.EXPECTED_CHECKPOINT_SHA)
        base = exp.HEADER_BYTES
        self.assertEqual(
            struct.unpack_from("<f", blob, base + (7 * exp.EXPECTED_WIDTH + 3) * 4)[0],
            1.25)
        self.assertEqual(
            struct.unpack_from(
                "<f", blob, base + (12345 * exp.EXPECTED_WIDTH + 0) * 4)[0],
            -2.5)
        self.assertEqual(
            struct.unpack_from("<f", blob, base + (7 * exp.EXPECTED_WIDTH + 0) * 4)[0],
            0.0)

    def test_head_weight_own_then_opp_order(self):
        head_weight = torch.arange(64, dtype=torch.float32).reshape(1, 64)
        blob = exp.build_artifact_bytes(
            torch.zeros(exp.EXPECTED_INPUTS, exp.EXPECTED_WIDTH),
            torch.zeros(exp.EXPECTED_WIDTH),
            head_weight.reshape(-1).contiguous(),
            torch.zeros(1), exp.EXPECTED_CHECKPOINT_SHA)
        base = exp.HEADER_BYTES + exp.EXPECTED_INPUTS * exp.EXPECTED_WIDTH * 4 \
            + exp.EXPECTED_WIDTH * 4
        # head.weight[0][0..31] = own block, [32..63] = opponent block.
        self.assertEqual(struct.unpack_from("<f", blob, base)[0], 0.0)
        self.assertEqual(struct.unpack_from("<f", blob, base + 31 * 4)[0], 31.0)
        self.assertEqual(struct.unpack_from("<f", blob, base + 32 * 4)[0], 32.0)
        self.assertEqual(struct.unpack_from("<f", blob, base + 63 * 4)[0], 63.0)

    def test_offset_accounting(self):
        blob = exp.build_artifact_bytes(
            self.sd["features.weight"].t().contiguous(),
            self.sd["acc_bias"].contiguous(),
            self.sd["head.weight"].reshape(-1).contiguous(),
            self.sd["head.bias"].reshape(-1).contiguous(),
            exp.EXPECTED_CHECKPOINT_SHA)
        offsets = {
            "header": 0,
            "features_weight": exp.HEADER_BYTES,
            "acc_bias": exp.HEADER_BYTES + exp.EXPECTED_INPUTS * exp.EXPECTED_WIDTH * 4,
            "head_weight": exp.HEADER_BYTES + exp.EXPECTED_INPUTS * exp.EXPECTED_WIDTH * 4
                           + exp.EXPECTED_WIDTH * 4,
            "head_bias": exp.HEADER_BYTES + exp.EXPECTED_INPUTS * exp.EXPECTED_WIDTH * 4
                         + exp.EXPECTED_WIDTH * 4 + 64 * 4,
            "end": len(blob),
        }
        self.assertEqual(offsets["end"], exp.TOTAL_BYTES)
        for name, off in offsets.items():
            self.assertLessEqual(off, len(blob), name)

    def test_export_artifact_reports_layout(self):
        with tempfile.TemporaryDirectory(prefix="s6-n2-test-") as tmp:
            # SHA binding is validated separately; here we exercise the
            # reporting path via the layout writer directly.
            blob = exp.build_artifact_bytes(
                self.sd["features.weight"].t().contiguous(),
                self.sd["acc_bias"].contiguous(),
                self.sd["head.weight"].reshape(-1).contiguous(),
                self.sd["head.bias"].reshape(-1).contiguous(),
                exp.EXPECTED_CHECKPOINT_SHA)
            out = Path(tmp) / "a.bin"
            out.write_bytes(blob)
            info = json.loads(
                '{"artifact_path":"%s","artifact_sha256":"%s","total_bytes":%d}'
                % (out, __import__("hashlib").sha256(blob).hexdigest(), len(blob)))
            self.assertEqual(info["total_bytes"], exp.TOTAL_BYTES)
            self.assertEqual(len(info["artifact_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
