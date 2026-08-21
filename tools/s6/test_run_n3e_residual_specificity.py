#!/usr/bin/env python3
"""Tests for the frozen S6-N3E specificity gates.

The N3E verdict decides whether NNUE runtime work is authorized at all, so the
gate arithmetic is tested directly against stubs: a cheap calibrator that
already explains the gain must produce CHEAP_CALIBRATION_SUFFICIENT, and only a
genuinely position-specific hybrid may pass.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import residual_calibration as calib  # noqa: E402
import residual_probe as residual  # noqa: E402
import run_n3e_residual_specificity as n3e  # noqa: E402


def split_view(calibrator_mae: float, hybrid_mae: float,
               calibrator_rmse: float = 220.0, hybrid_rmse: float = 210.0,
               classical_mae: float = 240.0) -> dict:
    return {
        "n": 6979,
        "raw_classical": {"clipped_mae_cp": classical_mae,
                          "clipped_rmse_cp": 260.0},
        "selected_calibrator": {
            "name": "global_bias", "parameter_count": 1,
            "metrics": {"clipped_mae_cp": calibrator_mae,
                        "clipped_rmse_cp": calibrator_rmse}},
        "nnue_hybrid": {"clipped_mae_cp": hybrid_mae,
                        "clipped_rmse_cp": hybrid_rmse},
        "hybrid_vs_selected_calibrator": {
            "mae_delta_cp": round(hybrid_mae - calibrator_mae, 6),
            "mae_improvement_fraction": round(
                (calibrator_mae - hybrid_mae) / calibrator_mae, 6)
            if calibrator_mae else None,
            "rmse_delta_cp": round(hybrid_rmse - calibrator_rmse, 6)},
    }


def bootstrap_stub(ci_upper: float, ci_lower: float = -9.0) -> dict:
    return {"n": 6979, "mean_delta_cp": -5.0, "median_delta_cp": -4.0,
            "positions_improved": 4000, "positions_worsened": 2979,
            "positions_unchanged": 0,
            "bootstrap_seed": residual.BOOTSTRAP_SEED,
            "bootstrap_resamples": residual.BOOTSTRAP_RESAMPLES,
            "ci_lower_cp": ci_lower, "ci_upper_cp": ci_upper}


def group_stub(entries: dict) -> dict:
    out = {}
    for name, value in entries.items():
        if value is None:
            out[name] = {"n": 0}
            continue
        n, calibrator_mae, hybrid_mae = value
        out[name] = {
            "n": n,
            # grouped_comparison puts the baseline in "classical" and the
            # candidate in "residual"; here they are calibrator and hybrid.
            "classical": {"clipped_mae_cp": calibrator_mae},
            "residual": {"clipped_mae_cp": hybrid_mae},
        }
    return out


HEALTHY_GROUPS = {
    "phase": group_stub({"high": (2784, 130.0, 122.0),
                         "mid": (2872, 180.0, 172.0),
                         "low": (1245, 165.0, 162.0),
                         "zero": (78, 400.0, 900.0)}),
    "abs_cp": group_stub({"0-100": (2507, 90.0, 86.0),
                          "100-300": (2020, 148.0, 140.0),
                          "300-1000": (2443, 245.0, 233.0),
                          "1000-inf": (9, 800.0, 2000.0)}),
}

HEALTHY_INTEGRITY = {
    "provenance_ok": True, "engine_binding_ok": True, "n3c_cache_ok": True,
    "confirmation_cache_ok": True, "checkpoint_sha_ok": True,
    "eligible_set_matches_n3d_ok": True,
    "nnue_retrained": False, "engine_invoked_for_base_eval": False,
}


def gates_for(validation=None, holdout=None, confirmation=None,
              bootstrap=None, by_group=None, integrity=None, finite=None):
    return n3e.evaluate_gates(
        validation or split_view(160.0, 150.0),
        holdout or split_view(160.0, 150.0),
        confirmation or split_view(160.0, 150.0),
        bootstrap or bootstrap_stub(-2.0),
        by_group or HEALTHY_GROUPS,
        integrity or dict(HEALTHY_INTEGRITY),
        finite if finite is not None else [])


class FrozenConstantTests(unittest.TestCase):
    def test_gates_and_identities_are_the_contract_values(self):
        self.assertEqual(n3e.MIN_CONFIRMATION_MAE_IMPROVEMENT_FRACTION, 0.02)
        self.assertEqual(n3e.CI_UPPER_MUST_BE_BELOW_CP, 0.0)
        self.assertEqual(n3e.MAX_GROUP_REGRESSION_FRACTION, 0.02)
        self.assertEqual(n3e.MIN_GROUP_N, 200)
        self.assertEqual(n3e.PASS_STATUS, "POSITION_SPECIFIC_GAIN_SUPPORTED")
        self.assertEqual(n3e.PASS_AUTHORIZATION,
                         "BENCH_ONLY_RUNTIME_AUTHORIZED")
        self.assertEqual(n3e.FAIL_STATUS, "CHEAP_CALIBRATION_SUFFICIENT")
        self.assertEqual(n3e.FAIL_AUTHORIZATION,
                         "NNUE_RUNTIME_NOT_AUTHORIZED")
        self.assertEqual(n3e.PRODUCTION_BASELINE, "bde9085")
        self.assertEqual(
            n3e.EXPECTED_CANONICAL_CHECKPOINT_SHA256,
            "5033d47cb101d96057e13aae9d3819d48fa8079e90bda8eae8cd935ac1006c55")
        self.assertEqual(n3e.EXPECTED_N3C_CACHE_SHA256,
                         "c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727")
        self.assertEqual(n3e.CITED_GLOBAL_SHIFT_CP, 41.3827)


class GateArithmeticTests(unittest.TestCase):
    def test_healthy_position_specific_result_passes_all_seven(self):
        result = gates_for()
        self.assertTrue(result["all_passed"])
        self.assertEqual(len(result["gates"]), 7)

    def test_gate1_rejects_hybrid_worse_on_validation(self):
        result = gates_for(validation=split_view(150.0, 151.0))
        self.assertFalse(result["gates"][0]["passed"])
        self.assertFalse(result["all_passed"])

    def test_gate1_rejects_hybrid_worse_on_holdout(self):
        result = gates_for(holdout=split_view(150.0, 150.5))
        self.assertFalse(result["gates"][0]["passed"])

    def test_gate1_allows_exact_tie(self):
        result = gates_for(validation=split_view(150.0, 150.0),
                           holdout=split_view(150.0, 150.0))
        self.assertTrue(result["gates"][0]["passed"])

    def test_gate2_needs_two_percent_over_the_calibrator(self):
        under = gates_for(confirmation=split_view(100.0, 98.1))
        self.assertFalse(under["gates"][1]["passed"])
        exact = gates_for(confirmation=split_view(100.0, 98.0))
        self.assertTrue(exact["gates"][1]["passed"])

    def test_cheap_calibration_sufficient_when_hybrid_matches_calibrator(self):
        """The core null result: hybrid ~= calibrator must NOT authorize."""
        result = gates_for(confirmation=split_view(153.9, 153.8))
        self.assertFalse(result["gates"][1]["passed"])
        self.assertFalse(result["all_passed"])

    def test_gate3_requires_ci_upper_strictly_below_zero(self):
        self.assertFalse(gates_for(bootstrap=bootstrap_stub(0.0))
                         ["gates"][2]["passed"])
        self.assertTrue(gates_for(bootstrap=bootstrap_stub(-1e-6))
                        ["gates"][2]["passed"])
        self.assertFalse(gates_for(bootstrap=bootstrap_stub(2.0))
                         ["gates"][2]["passed"])

    def test_gate3_records_the_paired_definition(self):
        gate = gates_for()["gates"][2]
        self.assertEqual(gate["definition"],
                         "abs(hybrid - teacher) - abs(calibrator - teacher)")
        self.assertEqual(gate["bootstrap_seed"], 20260821)
        self.assertEqual(gate["bootstrap_resamples"], 10000)

    def test_gate4_rejects_worse_rmse(self):
        result = gates_for(confirmation=split_view(160.0, 150.0,
                                                  calibrator_rmse=200.0,
                                                  hybrid_rmse=201.0))
        self.assertFalse(result["gates"][3]["passed"])

    def test_gate4_allows_equal_rmse(self):
        result = gates_for(confirmation=split_view(160.0, 150.0,
                                                  calibrator_rmse=200.0,
                                                  hybrid_rmse=200.0))
        self.assertTrue(result["gates"][3]["passed"])

    def test_gate5_skips_small_buckets_and_catches_regression(self):
        gate = gates_for()["gates"][4]
        self.assertFalse(gate["groups"]["zero"]["evaluated"])
        self.assertEqual(gate["evaluated_groups"], 3)
        groups = dict(HEALTHY_GROUPS)
        groups["phase"] = group_stub({"high": (2784, 100.0, 103.0),
                                      "mid": None, "low": None, "zero": None})
        bad = gates_for(by_group=groups)
        self.assertFalse(bad["gates"][4]["groups"]["high"]["passed"])
        self.assertFalse(bad["all_passed"])

    def test_gate5_boundary_is_exactly_two_percent(self):
        groups = dict(HEALTHY_GROUPS)
        groups["phase"] = group_stub({"high": (2784, 100.0, 102.0),
                                      "mid": None, "low": None, "zero": None})
        self.assertTrue(gates_for(by_group=groups)
                        ["gates"][4]["groups"]["high"]["passed"])
        groups["phase"] = group_stub({"high": (2784, 100.0, 102.01),
                                      "mid": None, "low": None, "zero": None})
        self.assertFalse(gates_for(by_group=groups)
                         ["gates"][4]["groups"]["high"]["passed"])

    def test_gate6_catches_abs_cp_bucket_regression(self):
        groups = dict(HEALTHY_GROUPS)
        groups["abs_cp"] = group_stub({"0-100": (2507, 90.0, 120.0),
                                       "100-300": None, "300-1000": None,
                                       "1000-inf": None})
        result = gates_for(by_group=groups)
        self.assertFalse(result["gates"][5]["groups"]["0-100"]["passed"])

    def test_gate7_fails_on_each_integrity_break(self):
        for key in ("provenance_ok", "engine_binding_ok", "n3c_cache_ok",
                    "confirmation_cache_ok", "checkpoint_sha_ok",
                    "eligible_set_matches_n3d_ok"):
            integrity = dict(HEALTHY_INTEGRITY)
            integrity[key] = False
            with self.subTest(key=key):
                self.assertFalse(gates_for(integrity=integrity)
                                 ["gates"][6]["passed"])

    def test_gate7_fails_on_non_finite_metric(self):
        result = gates_for(finite=["/splits/n3d_confirmation/nnue_hybrid"])
        self.assertFalse(result["gates"][6]["passed"])
        self.assertFalse(result["all_passed"])


class StatsTests(unittest.TestCase):
    def test_prediction_stats_reports_required_percentiles(self):
        stats = n3e.prediction_stats([float(i) for i in range(101)])
        for key in ("n", "min", "max", "mean", "std", "p10", "p50", "p90"):
            self.assertIn(key, stats)
        self.assertEqual(stats["n"], 101)
        self.assertEqual(stats["p50"], 50.0)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 100.0)

    def test_constant_correction_has_zero_std(self):
        """The signature of the cheap-global-shift explanation."""
        stats = n3e.prediction_stats([41.3827] * 500)
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["mean"], 41.3827)
        self.assertEqual(stats["p10"], stats["p90"])

    def test_prediction_stats_rejects_non_finite(self):
        with self.assertRaises(SystemExit):
            n3e.prediction_stats([1.0, float("inf")])
        with self.assertRaises(SystemExit):
            n3e.prediction_stats([])

    def test_pearson_is_none_for_a_constant_series(self):
        self.assertIsNone(n3e.pearson([5.0] * 10, [float(i) for i in range(10)]))

    def test_pearson_detects_perfect_relationships(self):
        xs = [float(i) for i in range(50)]
        self.assertEqual(n3e.pearson(xs, xs), 1.0)
        self.assertEqual(n3e.pearson(xs, [-v for v in xs]), -1.0)

    def test_pearson_length_mismatch_fails_closed(self):
        with self.assertRaises(SystemExit):
            n3e.pearson([1.0, 2.0], [1.0])

    def test_all_finite_locates_bad_paths(self):
        bad = n3e.all_finite({"a": {"b": [1.0, float("nan")]}, "c": 2.0})
        self.assertEqual(bad, ["/a/b[1]"])
        self.assertEqual(n3e.all_finite({"a": [1.0], "b": {"c": 2.0}}), [])


class CacheReuseTests(unittest.TestCase):
    def test_missing_cache_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            n3e.load_validated_cache(Path("/nonexistent/cache.json"), {},
                                     "e" * 64, "c" * 64, "N3C")
        self.assertIn("classical cache missing", str(cm.exception))

    def test_cache_sha_mismatch_fails_closed(self):
        with mock.patch.object(n3e.diag, "validate_classical_cache",
                               return_value={"sha256": "0" * 64}), \
             mock.patch.object(Path, "is_file", return_value=True):
            with self.assertRaises(SystemExit) as cm:
                n3e.load_validated_cache(Path("cache.json"), {}, "e" * 64,
                                         "c" * 64, "N3C")
        self.assertIn("N3C cache sha256", str(cm.exception))

    def test_matching_cache_sha_is_returned(self):
        payload = {"sha256": "c" * 64, "values": {}, "header": {}}
        with mock.patch.object(n3e.diag, "validate_classical_cache",
                               return_value=payload), \
             mock.patch.object(Path, "is_file", return_value=True):
            got = n3e.load_validated_cache(Path("cache.json"), {}, "e" * 64,
                                           "c" * 64, "N3C")
        self.assertIs(got, payload)


class EvaluateSplitSetTests(unittest.TestCase):
    """End-to-end on a tiny synthetic split, using the real calibrators."""

    def _fixture(self, shift: float):
        rows = []
        for i in range(240):
            phase_name = calib.PHASE_BUCKETS[i % 4]
            base = -300.0 + (i % 31) * 20.0
            rows.append((f"p{i:05d}", base, base + shift, phase_name))
        phase_value = {"high": 20, "mid": 12, "low": 4, "zero": 0}
        split = {
            "pids": [r[0] for r in rows],
            "raw_target_cp": [r[2] for r in rows],
            "phases": [phase_value[r[3]] for r in rows],
            "source_families": ["lichess-standard-rated-v1"] * len(rows),
        }
        classical = {r[0]: r[1] for r in rows}
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        selection = calib.validation_select_calibrator(fitted, data)
        return split, classical, fitted, selection

    def test_pure_constant_world_lets_the_calibrator_capture_the_gain(self):
        """If the world IS a constant shift, a 1-parameter fit captures ~all."""
        split, classical, fitted, selection = self._fixture(41.3827)
        classical_cp = [classical[pid] for pid in split["pids"]]
        # A hybrid that only adds the same constant.
        hybrid_cp = [v + 41.3827 for v in classical_cp]
        report = n3e.evaluate_split_set("t", split, classical_cp, hybrid_cp,
                                        fitted, selection["selected"],
                                        classical)
        captured = report["gain_decomposition"][
            "fraction_of_nnue_gain_captured_by_calibrator"]
        self.assertIsNotNone(captured)
        self.assertGreater(captured, 0.99)
        self.assertEqual(report["correction_prediction_stats"]["std"], 0.0)
        self.assertAlmostEqual(
            report["correction_prediction_stats"]["mean"], 41.3827, places=3)
        self.assertIsNone(report["correction_vs_classical_pearson"])

    def test_field_names_follow_the_contract(self):
        split, classical, fitted, selection = self._fixture(20.0)
        classical_cp = [classical[pid] for pid in split["pids"]]
        hybrid_cp = [v + 20.0 for v in classical_cp]
        report = n3e.evaluate_split_set("t", split, classical_cp, hybrid_cp,
                                        fitted, selection["selected"],
                                        classical)
        self.assertIn("classical_prediction_stats", report)
        self.assertIn("correction_prediction_stats", report)
        self.assertIn("hybrid_prediction_stats", report)
        self.assertNotIn("residual_prediction_stats", report)
        self.assertEqual(set(report["calibrator_candidates"]),
                         set(calib.CANDIDATE_ORDER))

    def test_position_specific_hybrid_beats_the_calibrator(self):
        """A hybrid that tracks the true per-position residual must win."""
        rows = []
        phase_value = {"high": 20, "mid": 12, "low": 4, "zero": 0}
        for i in range(400):
            phase_name = calib.PHASE_BUCKETS[i % 4]
            base = -300.0 + (i % 31) * 20.0
            # Correction depends on a feature no calibrator can see.
            wobble = 120.0 if (i % 7) < 3 else -90.0
            rows.append((f"p{i:05d}", base, base + wobble, phase_name))
        split = {
            "pids": [r[0] for r in rows],
            "raw_target_cp": [r[2] for r in rows],
            "phases": [phase_value[r[3]] for r in rows],
        }
        classical = {r[0]: r[1] for r in rows}
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        selection = calib.validation_select_calibrator(fitted, data)
        classical_cp = [classical[pid] for pid in split["pids"]]
        hybrid_cp = [r[2] for r in rows]  # oracle hybrid
        report = n3e.evaluate_split_set("t", split, classical_cp, hybrid_cp,
                                        fitted, selection["selected"],
                                        classical)
        self.assertEqual(report["nnue_hybrid"]["clipped_mae_cp"], 0.0)
        self.assertGreater(
            report["selected_calibrator"]["metrics"]["clipped_mae_cp"], 50.0)
        captured = report["gain_decomposition"][
            "fraction_of_nnue_gain_captured_by_calibrator"]
        self.assertLess(captured, 0.5)
        self.assertGreater(report["correction_prediction_stats"]["std"], 50.0)


if __name__ == "__main__":
    unittest.main()
