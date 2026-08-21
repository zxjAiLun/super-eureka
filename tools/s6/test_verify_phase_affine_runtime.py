#!/usr/bin/env python3
"""S6-C1 verifier gates: label/count alignment must fail before inference.

`teacher_cp_stm != null` decides which positions are compared and therefore the
reported MAE/RMSE, so a swapped labels file would change the measurement while
every dataset and cache SHA still matched. These tests pin that the alignment
check fails closed, and that it happens before the engine is ever invoked.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_phase_affine_runtime as verify  # noqa: E402

N3B_LABELS = "e6f036f426db8a5fffc6c28baa6ae5333b0fe441bd9eec13f56d4dda989896d9"
CONFIRM_LABELS = "e1c25844fd785d46625f6a2a24edaa1a2e8fbd2863f57edfc3f3769723e8edfb"
ELIGIBLE = 6979


def n3e_stub(n3b_labels: str = N3B_LABELS,
             confirm_labels: str = CONFIRM_LABELS,
             eligible: int = ELIGIBLE) -> dict:
    return {
        "bindings": {
            "n3b_labels_sha256": n3b_labels,
            "confirmation_labels_sha256": confirm_labels,
        },
        "splits": {"n3d_confirmation": {"n": eligible}},
    }


def loaded(labels_sha: str) -> dict:
    return {"data": {"labels_sha": labels_sha}}


def rows(count: int) -> list[dict]:
    return [{"position_id": f"p{i}", "fen": "fen"} for i in range(count)]


class LabelAlignmentTests(unittest.TestCase):
    def test_matching_labels_and_count_bind_successfully(self):
        binding = verify.bind_to_n3e_labels(
            n3e_stub(), loaded(N3B_LABELS), loaded(CONFIRM_LABELS),
            rows(ELIGIBLE))
        self.assertTrue(binding["aligned_with_n3e_result"])
        self.assertEqual(binding["n3b_labels_sha256"], N3B_LABELS)
        self.assertEqual(binding["confirmation_labels_sha256"], CONFIRM_LABELS)
        self.assertEqual(binding["confirmation_eligible_positions"], ELIGIBLE)

    def test_n3b_labels_mismatch_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            verify.bind_to_n3e_labels(
                n3e_stub(), loaded("0" * 64), loaded(CONFIRM_LABELS),
                rows(ELIGIBLE))
        self.assertIn("N3B labels sha", str(cm.exception))

    def test_confirmation_labels_mismatch_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            verify.bind_to_n3e_labels(
                n3e_stub(), loaded(N3B_LABELS), loaded("0" * 64),
                rows(ELIGIBLE))
        self.assertIn("confirmation labels sha", str(cm.exception))

    def test_eligible_count_mismatch_fails_closed(self):
        """Catches a divergent null-CP pattern even if both file hashes matched."""
        with self.assertRaises(SystemExit) as cm:
            verify.bind_to_n3e_labels(
                n3e_stub(), loaded(N3B_LABELS), loaded(CONFIRM_LABELS),
                rows(ELIGIBLE - 1))
        self.assertIn("confirmation eligible positions", str(cm.exception))

    def test_every_mismatch_is_rejected_before_the_engine_is_invoked(self):
        """No failing alignment may reach `rust_batch`."""
        cases = [
            (loaded("0" * 64), loaded(CONFIRM_LABELS), ELIGIBLE),
            (loaded(N3B_LABELS), loaded("0" * 64), ELIGIBLE),
            (loaded(N3B_LABELS), loaded(CONFIRM_LABELS), ELIGIBLE - 1),
        ]
        for n3b, confirm, eligible in cases:
            with self.subTest(n3b=n3b, confirm=confirm, eligible=eligible):
                with mock.patch.object(verify, "rust_batch") as engine:
                    with self.assertRaises(SystemExit):
                        verify.bind_to_n3e_labels(n3e_stub(), n3b, confirm,
                                                  rows(eligible))
                engine.assert_not_called()

    def test_alignment_runs_before_any_comparison_in_main(self):
        """Structural: the binding call must precede the first compare_split."""
        source = inspect.getsource(verify.main)
        bind_at = source.index("bind_to_n3e_labels(")
        compare_at = source.index("compare_split(")
        self.assertLess(bind_at, compare_at,
                        "labels must be bound before any Rust inference")


class MicrobenchContractTests(unittest.TestCase):
    def test_feasibility_gate_is_the_authorized_value(self):
        self.assertEqual(verify.MAX_MICROBENCH_RATIO, 1.10)

    def test_quantization_and_drift_budgets_are_the_authorized_values(self):
        self.assertEqual(verify.MAX_ABS_QUANTIZATION_CP, 0.51)
        self.assertEqual(verify.MAX_MEAN_QUANTIZATION_CP, 0.26)
        self.assertEqual(verify.MAX_METRIC_DRIFT_CP, 0.05)

    def test_microbench_requires_warmup_and_alternation_fields(self):
        """The hardened bench must report how it controlled for order bias."""
        report = {"positions": 3, "iterations": 10, "rounds": 5,
                  "warmup_per_round": True, "order_alternated": True,
                  "round_order": ["base_first", "candidate_first"],
                  "base_median_ns": 100, "candidate_median_ns": 105,
                  "base_ns_per_eval": 1.0, "candidate_ns_per_eval": 1.05,
                  "ratio": 1.05}
        with mock.patch.object(verify.subprocess, "run",
                               return_value=mock.Mock(
                                   returncode=0,
                                   stdout=verify.json.dumps(report),
                                   stderr="")):
            got = verify.microbench(Path("engine"), rows(3), 10, 5)
        self.assertTrue(got["warmup_per_round"])
        self.assertTrue(got["order_alternated"])
        self.assertTrue(got["feasible"])

    def test_microbench_ratio_above_gate_is_infeasible(self):
        report = {"ratio": 1.2, "warmup_per_round": True,
                  "order_alternated": True}
        with mock.patch.object(verify.subprocess, "run",
                               return_value=mock.Mock(
                                   returncode=0,
                                   stdout=verify.json.dumps(report),
                                   stderr="")):
            got = verify.microbench(Path("engine"), rows(3), 10, 5)
        self.assertFalse(got["feasible"])

    def test_microbench_nonzero_rc_fails_closed(self):
        with mock.patch.object(verify.subprocess, "run",
                               return_value=mock.Mock(
                                   returncode=1, stdout="", stderr="boom")):
            with self.assertRaises(SystemExit):
                verify.microbench(Path("engine"), rows(3), 10, 5)


class ConstantDerivationTests(unittest.TestCase):
    def test_constants_must_derive_from_the_selected_calibrator(self):
        n3e = {
            "calibration": {
                "selection": {"selected": "phase_affine"},
                "selected_parameters": {
                    "name": "phase_affine",
                    "phase_order": ["high", "mid", "low", "zero"],
                    "parameters": {
                        "u": [-0.2613823636, -0.1061941736, -0.0117859984,
                              1.5102040829],
                        "b": [0.036717418, 0.0503747198, 0.0217028031,
                              -0.000464891],
                    },
                },
            }
        }
        constants = verify.verify_constants(n3e)
        self.assertEqual(tuple(constants["factor"]), verify.FACTOR)
        self.assertEqual(tuple(constants["bias_scaled_cp"]),
                         verify.BIAS_SCALED_CP)

    def test_a_retuned_calibrator_fails_closed(self):
        n3e = {
            "calibration": {
                "selection": {"selected": "phase_affine"},
                "selected_parameters": {
                    "name": "phase_affine",
                    "phase_order": ["high", "mid", "low", "zero"],
                    "parameters": {
                        "u": [-0.25, -0.1, -0.01, 1.5],
                        "b": [0.03, 0.05, 0.02, -0.0004],
                    },
                },
            }
        }
        with self.assertRaises(SystemExit) as cm:
            verify.verify_constants(n3e)
        self.assertIn("derived factor constants", str(cm.exception))

    def test_a_different_selected_calibrator_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            verify.verify_constants(
                {"calibration": {"selection": {"selected": "global_bias"}}})
        self.assertIn("N3E selected calibrator", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
