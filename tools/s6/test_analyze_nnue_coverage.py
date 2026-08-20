#!/usr/bin/env python3
"""analyze_nnue_coverage gate tests: low+zero from per_phase, family shares,
SHA / verify failures, and DATA_PILOT_FAIL exiting nonzero after writing."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_nnue_coverage as anc  # noqa: E402


def base_analysis(records: int = 12000,
                  families: dict | None = None,
                  phases: dict | None = None) -> dict:
    return {
        "dataset_sha256": "a" * 64,
        "records_total": records,
        "per_family": families if families is not None else {
            "arena": 5000, "lichess-standard-rated-v1": 7000},
        "per_phase": phases if phases is not None else {
            "high": 5000, "mid": 5600, "low": 1200, "zero": 200},
        "per_split": {"train": 9600, "validation": 1200, "holdout": 1200},
        "coverage": {
            "train": {"union_unique": 7000},
            "validation": {"positions": 1200},
            "holdout": {"positions": 1200},
        },
    }


class EvaluateGatesTests(unittest.TestCase):
    def test_pass_when_all_conditions_met(self):
        a = base_analysis()
        g = anc.evaluate_gates(a, rebuilt_sha=a["dataset_sha256"], verify_rc=0)
        self.assertTrue(g["pass"])
        # low+zero share = (1200+200)/12000 = 11.7% >= 10%
        self.assertGreaterEqual(g["facts"]["low_plus_zero_share"], 0.10)

    def test_low_plus_zero_below_10_percent_fails(self):
        a = base_analysis(phases={"high": 10000, "mid": 1000,
                                  "low": 500, "zero": 0})
        g = anc.evaluate_gates(a, rebuilt_sha=a["dataset_sha256"], verify_rc=0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["checks"]["low+zero >= 10%"])
        self.assertLess(g["facts"]["low_plus_zero_share"], 0.10)

    def test_family_share_out_of_range_fails(self):
        a = base_analysis(families={"arena": 11000,
                                    "lichess-standard-rated-v1": 1000})
        g = anc.evaluate_gates(a, rebuilt_sha=a["dataset_sha256"], verify_rc=0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["checks"][
            "each family share in [0.30, 0.70]"])

    def test_missing_family_fails(self):
        a = base_analysis(families={"arena": 12000})
        g = anc.evaluate_gates(a, rebuilt_sha=a["dataset_sha256"], verify_rc=0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["checks"][
            "families include arena + lichess-standard-rated-v1"])

    def test_train_union_below_6500_fails(self):
        a = base_analysis()
        a["coverage"]["train"]["union_unique"] = 6000
        g = anc.evaluate_gates(a, rebuilt_sha=a["dataset_sha256"], verify_rc=0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["checks"]["train feature union >= 6500"])

    def test_rebuilt_sha_mismatch_fails(self):
        a = base_analysis()
        g = anc.evaluate_gates(a, rebuilt_sha="different", verify_rc=0)
        self.assertFalse(g["pass"])
        self.assertFalse(g["checks"][
            "second rebuild dataset_sha256 identical"])

    def test_verify_failure_fails(self):
        a = base_analysis()
        g = anc.evaluate_gates(a, rebuilt_sha="abc", verify_rc=1)
        self.assertFalse(g["pass"])
        self.assertFalse(g["checks"][
            "verify_dataset --allow-unlabeled pass"])


class MainFailureExitTests(unittest.TestCase):
    def test_data_pilot_fail_writes_result_and_returns_2(self):
        """DATA_PILOT_FAIL must write the results file and return 2."""
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            out = tmp / "result.json"
            argv = ["analyze_nnue_coverage.py", "--engine", "eureka",
                    "--dataset", "ds", "--sources", "s1", "--gate",
                    "--rebuilt-sha", "a" * 64, "--verify-rc", "0",
                    "--out", str(out)]
            # Force a failing gate via a bad per_phase distribution.
            with mock.patch.object(anc, "analyze") as analyze, \
                 mock.patch.object(sys, "argv", argv):
                analyze.return_value = base_analysis(
                    phases={"high": 11000, "mid": 500,
                            "low": 300, "zero": 0})
                rc = anc.main()
            self.assertEqual(rc, 2)
            result = json.loads(out.read_text())
            self.assertEqual(result["status"], "DATA_PILOT_FAIL")

    def test_data_pilot_pass_returns_0(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            out = tmp / "result.json"
            argv = ["analyze_nnue_coverage.py", "--engine", "eureka",
                    "--dataset", "ds", "--sources", "s1", "--gate",
                    "--rebuilt-sha", "a" * 64, "--verify-rc", "0",
                    "--out", str(out)]
            with mock.patch.object(anc, "analyze") as analyze, \
                 mock.patch.object(sys, "argv", argv):
                analyze.return_value = base_analysis()
                rc = anc.main()
            self.assertEqual(rc, 0)
            result = json.loads(out.read_text())
            self.assertEqual(result["status"], "DATA_PILOT_PASS")


if __name__ == "__main__":
    unittest.main()
