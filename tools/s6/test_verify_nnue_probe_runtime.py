#!/usr/bin/env python3
"""S6-N2 runtime verifier unit tests: parity gate boundaries, duplicate
rust rows, failure exit code, and microbench-skip-on-failure behavior."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_nnue_probe_runtime as ver  # noqa: E402
import train_nnue_probe as probe  # noqa: E402


def base_parity() -> dict:
    return {
        "rows": ver.EXPECTED_CP_ROWS,
        "nan_inf": 0,
        "max_abs_diff_cp": 0.0001,
        "mean_abs_diff_cp": 0.00001,
    }


class ParityGateTests(unittest.TestCase):
    def test_pass_at_boundary(self):
        p = base_parity()
        p["max_abs_diff_cp"] = ver.MAX_ABS_DIFF_CP
        p["mean_abs_diff_cp"] = ver.MEAN_ABS_DIFF_CP
        self.assertTrue(ver.parity_passes(p))

    def test_fail_just_above_max(self):
        p = base_parity()
        p["max_abs_diff_cp"] = ver.MAX_ABS_DIFF_CP + 1e-9
        self.assertFalse(ver.parity_passes(p))

    def test_fail_just_above_mean(self):
        p = base_parity()
        p["mean_abs_diff_cp"] = ver.MEAN_ABS_DIFF_CP + 1e-9
        self.assertFalse(ver.parity_passes(p))

    def test_fail_wrong_row_count(self):
        p = base_parity()
        p["rows"] = ver.EXPECTED_CP_ROWS - 1
        self.assertFalse(ver.parity_passes(p))

    def test_fail_on_nan_inf(self):
        p = base_parity()
        p["nan_inf"] = 1
        self.assertFalse(ver.parity_passes(p))


class IndexUniqueRustRowsTests(unittest.TestCase):
    def test_accepts_unique_rows(self):
        rows = [{"position_id": "a"}, {"position_id": "b"}]
        indexed = ver.index_unique_rust_rows(rows, 2)
        self.assertEqual(set(indexed), {"a", "b"})

    def test_rejects_wrong_count(self):
        with self.assertRaises(SystemExit):
            ver.index_unique_rust_rows([{"position_id": "a"}], 2)

    def test_rejects_duplicate_position_id(self):
        rows = [{"position_id": "a"}, {"position_id": "a"}]
        with self.assertRaises(SystemExit) as cm:
            ver.index_unique_rust_rows(rows, 2)
        self.assertIn("duplicate position_id", str(cm.exception))


class FailurePathTests(unittest.TestCase):
    """Drive main() to a parity failure with mocks; it must write
    PARITY_FAIL / NOT_RUN_PARITY_FAIL / microbench=null and return 2 without
    ever calling the microbench."""

    def _run_main_with_rust_diffs(self, offset: float):
        with tempfile.TemporaryDirectory(prefix="s6-n2-gate-") as tmp:
            dataset_dir = Path(tmp) / "ds"
            dataset_dir.mkdir()
            records = []
            labels = []
            for i in range(3):
                pid = f"{i:064d}"
                records.append({
                    "position_id": pid,
                    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "split": "train",
                })
                labels.append({"position_id": pid, "teacher_cp_stm": 100 + i})
            labels_text = "".join(json.dumps(l) + "\n" for l in labels)
            import hashlib
            ds_sha = probe.compute_dataset_sha(records)
            lb_sha = hashlib.sha256(labels_text.encode()).hexdigest()
            (dataset_dir / "dataset_manifest.json").write_text(json.dumps({
                "dataset_id": "syn", "records_total": 3,
                "dataset_sha256": ds_sha,
            }))
            (dataset_dir / "teacher_manifest.json").write_text(json.dumps({
                "engine": "x", "binary_sha256": "0" * 64,
                "labels_sha256": lb_sha, "audit": {"ok": True, "checked": 1000},
            }))
            (dataset_dir / "labels.jsonl").write_text(labels_text)
            for i in range(3):
                (dataset_dir / f"part-000{i}.jsonl").write_text(
                    json.dumps(records[i]) + "\n")

            exported = {
                r["position_id"]: {
                    "position_id": r["position_id"], "fen": r["fen"],
                    "white": [0, 1], "black": [2],
                } for r in records
            }
            rust_rows = [{
                "position_id": r["position_id"], "fen": r["fen"],
                "scaled_prediction": 0.0,
                "prediction_cp": 100 + i + offset,
            } for i, r in enumerate(records)]

            out_path = Path(tmp) / "result.json"
            with mock.patch.object(ver.exp, "validate_checkpoint") as vc, \
                 mock.patch.object(ver.probe, "load_dataset") as ld, \
                 mock.patch.object(ver.probe, "export_all_features") as ex, \
                 mock.patch.object(ver, "engine_batch",
                                   return_value=rust_rows) as eb, \
                 mock.patch.object(ver, "run_microbench") as mb, \
                 mock.patch.object(ver, "EXPECTED_CP_ROWS", 3), \
                 mock.patch.object(ver.torch, "load") as tl:
                vc.return_value = {"sha": "6" * 64}
                ld.return_value = {
                    "records": records,
                    "labels": {l["position_id"]: l for l in labels},
                    "dataset_sha": ds_sha,
                    "labels_sha": lb_sha,
                }
                ex.return_value = exported
                tl.return_value = {
                    "state_dict": {
                        "features.weight": __import__("torch").zeros(
                            32, 40960),
                        "acc_bias": __import__("torch").zeros(32),
                        "head.weight": __import__("torch").zeros(1, 64),
                        "head.bias": __import__("torch").zeros(1),
                    },
                }
                engine = Path(tmp) / "eureka"
                engine.write_bytes(b"x")
                engine_bin = Path(tmp) / "artifact.bin"
                engine_bin.write_bytes(b"x")
                ckpt_path = Path(tmp) / "ckpt.pt"
                ckpt_path.write_bytes(b"x")
                micro_batch = Path(tmp) / "micro.txt"
                micro_batch.write_text("x")
                argv = [
                    "verify_nnue_probe_runtime.py",
                    "--engine", str(engine),
                    "--dataset", str(dataset_dir),
                    "--checkpoint", str(ckpt_path),
                    "--artifact", str(engine_bin),
                    "--microbench-batch", str(micro_batch),
                    "--microbench-iterations", "10",
                    "--out", str(out_path),
                ]
                with mock.patch.object(sys, "argv", argv):
                    rc = ver.main()
                self.assertEqual(rc, 2)
                mb.assert_not_called()
                result = json.loads(out_path.read_text())
                self.assertEqual(result["status"], "PARITY_FAIL")
                self.assertEqual(result["cost_status"], "NOT_RUN_PARITY_FAIL")
                self.assertIsNone(result["microbench"])
            return out_path

    def test_parity_failure_returns_2_and_skips_microbench(self):
        self._run_main_with_rust_diffs(1000.0)  # huge diff -> fail


if __name__ == "__main__":
    unittest.main()
