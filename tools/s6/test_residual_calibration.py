#!/usr/bin/env python3
"""Tests for the S6-N3E cheap calibration ladder.

The point of these calibrators is to be a HONEST null hypothesis, so the tests
check that each candidate can actually recover a correction it should be able
to recover, that fitting is deterministic and RNG-free, that selection only
ever looks at validation, and that ties resolve toward fewer parameters.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import residual_calibration as calib  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

PHASE_VALUE = {"high": 20, "mid": 12, "low": 4, "zero": 0}


def make_split(rows: list[tuple[str, float, float, str]]) -> tuple[dict, dict]:
    """rows: (pid, base_cp, teacher_cp, phase_name) -> (split, classical)."""
    split = {
        "pids": [r[0] for r in rows],
        "raw_target_cp": [r[2] for r in rows],
        "phases": [PHASE_VALUE[r[3]] for r in rows],
    }
    classical = {r[0]: r[1] for r in rows}
    return split, classical


def constant_shift_rows(n: int, shift: float, phases=("high", "mid", "low",
                                                     "zero")) -> list:
    """teacher = base + shift everywhere; a pure global-bias world."""
    rows = []
    for i in range(n):
        phase = phases[i % len(phases)]
        base = -400.0 + (i % 37) * 25.0
        rows.append((f"p{i:05d}", base, base + shift, phase))
    return rows


class InputTests(unittest.TestCase):
    def test_inputs_are_scaled_and_clipped_like_the_frozen_target(self):
        split, classical = make_split([("a", 100.0, 500.0, "high"),
                                       ("b", 0.0, 9000.0, "zero")])
        data = calib.calibration_inputs(split, classical)
        self.assertAlmostEqual(data["x"][0].item(), 0.1, places=12)
        self.assertAlmostEqual(data["z"][0].item(), 0.4, places=12)
        # residual 9000 clips to +2000 then scales to 2.0
        self.assertAlmostEqual(data["z"][1].item(), 2.0, places=12)
        self.assertEqual(data["n"], 2)

    def test_missing_classical_entry_fails_closed(self):
        split, classical = make_split([("a", 1.0, 2.0, "high")])
        with self.assertRaises(SystemExit) as cm:
            calib.calibration_inputs(split, {})
        self.assertIn("classical cache missing", str(cm.exception))

    def test_missing_phases_fails_closed(self):
        split, classical = make_split([("a", 1.0, 2.0, "high")])
        del split["phases"]
        with self.assertRaises(SystemExit) as cm:
            calib.calibration_inputs(split, classical)
        self.assertIn("no phases", str(cm.exception))

    def test_missing_phase_bucket_fails_closed(self):
        split, classical = make_split(
            [(f"p{i}", 10.0, 20.0, "high") for i in range(8)])
        data = calib.calibration_inputs(split, classical)
        with self.assertRaises(SystemExit) as cm:
            calib.require_all_phases(data, "train")
        self.assertIn("missing phase bucket", str(cm.exception))

    def test_all_phases_present_reports_counts(self):
        split, classical = make_split(constant_shift_rows(40, 25.0))
        data = calib.calibration_inputs(split, classical)
        counts = calib.require_all_phases(data, "train")
        self.assertEqual(set(counts), set(calib.PHASE_BUCKETS))
        self.assertTrue(all(v > 0 for v in counts.values()))


class FitTests(unittest.TestCase):
    def test_identity_has_no_parameters_and_predicts_classical(self):
        split, classical = make_split(constant_shift_rows(40, 30.0))
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        self.assertEqual(fitted["identity"]["parameters"], {})
        self.assertEqual(fitted["identity"]["parameter_count"], 0)
        predicted = calib.predict_calibrator(fitted["identity"], split,
                                             classical)
        for value, pid in zip(predicted, split["pids"]):
            self.assertAlmostEqual(value, classical[pid], places=6)

    def test_global_bias_recovers_a_constant_shift(self):
        split, classical = make_split(constant_shift_rows(200, 41.38))
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        bias_cp = fitted["global_bias"]["parameters"]["b"][0] * 1000.0
        self.assertAlmostEqual(bias_cp, 41.38, places=2)
        predicted = calib.predict_calibrator(fitted["global_bias"], split,
                                             classical)
        for value, pid in zip(predicted, split["pids"]):
            self.assertAlmostEqual(value, classical[pid] + 41.38, places=2)

    def test_global_affine_recovers_slope_and_intercept(self):
        rows = []
        for i in range(300):
            phase = calib.PHASE_BUCKETS[i % 4]
            base = -600.0 + i * 4.0
            # correction_cp = 0.25 * base + 30
            rows.append((f"p{i:05d}", base, base + 0.25 * base + 30.0, phase))
        split, classical = make_split(rows)
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        u = fitted["global_affine"]["parameters"]["u"][0]
        b_cp = fitted["global_affine"]["parameters"]["b"][0] * 1000.0
        self.assertAlmostEqual(u, 0.25, places=2)
        self.assertAlmostEqual(b_cp, 30.0, places=1)

    def test_phase_bias_recovers_per_phase_constants(self):
        wanted = {"high": 10.0, "mid": -20.0, "low": 55.0, "zero": 5.0}
        rows = []
        for i in range(400):
            phase = calib.PHASE_BUCKETS[i % 4]
            base = -300.0 + (i % 41) * 15.0
            rows.append((f"p{i:05d}", base, base + wanted[phase], phase))
        split, classical = make_split(rows)
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        for index, phase in enumerate(calib.PHASE_BUCKETS):
            got = fitted["phase_bias"]["parameters"]["b"][index] * 1000.0
            self.assertAlmostEqual(got, wanted[phase], places=1)
        self.assertEqual(fitted["phase_bias"]["phase_order"],
                         list(calib.PHASE_BUCKETS))

    def test_parameter_counts_are_the_frozen_ladder(self):
        self.assertEqual(calib.CANDIDATE_PARAMETER_COUNT,
                         {"identity": 0, "global_bias": 1, "global_affine": 2,
                          "phase_bias": 4, "phase_affine": 8})
        self.assertEqual(calib.CANDIDATE_ORDER,
                         ("identity", "global_bias", "global_affine",
                          "phase_bias", "phase_affine"))

    def test_fitting_is_deterministic_across_repeats(self):
        split, classical = make_split(constant_shift_rows(150, 33.0))
        data = calib.calibration_inputs(split, classical)
        first = calib.fit_calibrators(data)
        second = calib.fit_calibrators(data)
        for name in calib.CANDIDATE_ORDER:
            self.assertEqual(first[name]["parameters"],
                             second[name]["parameters"], name)
            self.assertEqual(first[name]["train_smooth_l1"],
                             second[name]["train_smooth_l1"], name)

    def test_fitting_does_not_consume_global_rng(self):
        """Deterministic protocol: the torch RNG state must be untouched."""
        split, classical = make_split(constant_shift_rows(120, 20.0))
        data = calib.calibration_inputs(split, classical)
        torch.manual_seed(12345)
        before = torch.get_rng_state()
        calib.fit_calibrators(data)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_richer_candidates_never_fit_train_worse(self):
        split, classical = make_split(constant_shift_rows(200, 41.38))
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        losses = [fitted[name]["train_smooth_l1"]
                  for name in calib.CANDIDATE_ORDER]
        for richer, cheaper in zip(losses[1:], losses[:-1]):
            self.assertLessEqual(richer, cheaper + 1e-9)

    def test_uses_float64(self):
        self.assertIs(calib.DTYPE, torch.float64)
        split, classical = make_split(constant_shift_rows(40, 10.0))
        data = calib.calibration_inputs(split, classical)
        self.assertIs(data["x"].dtype, torch.float64)


class SelectionTests(unittest.TestCase):
    def _fit(self, rows):
        split, classical = make_split(rows)
        data = calib.calibration_inputs(split, classical)
        return calib.fit_calibrators(data), data, split, classical

    def test_constant_world_selects_global_bias_not_something_richer(self):
        fitted, data, _, _ = self._fit(constant_shift_rows(300, 41.38))
        selection = calib.validation_select_calibrator(fitted, data)
        self.assertEqual(selection["selected"], "global_bias")
        self.assertEqual(selection["selected_parameter_count"], 1)
        self.assertFalse(selection["holdout_or_confirmation_used"])

    def test_zero_correction_world_selects_identity(self):
        fitted, data, _, _ = self._fit(constant_shift_rows(300, 0.0))
        selection = calib.validation_select_calibrator(fitted, data)
        self.assertEqual(selection["selected"], "identity")
        self.assertEqual(selection["selected_parameter_count"], 0)

    def test_per_phase_world_selects_a_phase_candidate(self):
        wanted = {"high": 120.0, "mid": -150.0, "low": 200.0, "zero": -80.0}
        rows = []
        for i in range(400):
            phase = calib.PHASE_BUCKETS[i % 4]
            base = -300.0 + (i % 41) * 15.0
            rows.append((f"p{i:05d}", base, base + wanted[phase], phase))
        fitted, data, _, _ = self._fit(rows)
        selection = calib.validation_select_calibrator(fitted, data)
        self.assertIn(selection["selected"], ("phase_bias", "phase_affine"))

    def test_tie_resolves_toward_fewer_parameters(self):
        fitted, data, _, _ = self._fit(constant_shift_rows(80, 12.0))
        # Force an exact tie across the whole ladder.
        for name in calib.CANDIDATE_ORDER:
            fitted[name] = dict(fitted[name])
        selection = calib.validation_select_calibrator(
            fitted, data)
        losses = selection["validation_smooth_l1"]
        tied = [n for n in calib.CANDIDATE_ORDER
                if abs(losses[n] - min(losses.values())) <= calib.TIE_TOLERANCE]
        self.assertEqual(selection["selected"], tied[0],
                         "cheapest tied candidate must win")

    def test_marginal_improvement_below_tolerance_does_not_displace(self):
        class Stub(dict):
            pass
        fitted = {name: {"name": name,
                         "parameter_count": calib.CANDIDATE_PARAMETER_COUNT[name],
                         "parameters": {}} for name in calib.CANDIDATE_ORDER}
        base = 0.5

        def fake_loss(cal, data, _base=base):
            offsets = {"identity": 0.0, "global_bias": -1e-12,
                       "global_affine": -2e-12, "phase_bias": -3e-12,
                       "phase_affine": -4e-12}
            return _base + offsets[cal["name"]]
        original = calib.candidate_loss
        calib.candidate_loss = fake_loss
        try:
            split, classical = make_split(constant_shift_rows(40, 5.0))
            data = calib.calibration_inputs(split, classical)
            selection = calib.validation_select_calibrator(fitted, data)
        finally:
            calib.candidate_loss = original
        self.assertEqual(selection["selected"], "identity")

    def test_real_improvement_above_tolerance_does_displace(self):
        fitted = {name: {"name": name,
                         "parameter_count": calib.CANDIDATE_PARAMETER_COUNT[name],
                         "parameters": {}} for name in calib.CANDIDATE_ORDER}

        def fake_loss(cal, data):
            return {"identity": 0.5, "global_bias": 0.4, "global_affine": 0.4,
                    "phase_bias": 0.4, "phase_affine": 0.4}[cal["name"]]
        original = calib.candidate_loss
        calib.candidate_loss = fake_loss
        try:
            split, classical = make_split(constant_shift_rows(40, 5.0))
            data = calib.calibration_inputs(split, classical)
            selection = calib.validation_select_calibrator(fitted, data)
        finally:
            calib.candidate_loss = original
        self.assertEqual(selection["selected"], "global_bias")

    def test_missing_fitted_candidate_fails_closed(self):
        fitted, data, _, _ = self._fit(constant_shift_rows(80, 12.0))
        del fitted["phase_affine"]
        with self.assertRaises(SystemExit) as cm:
            calib.validation_select_calibrator(fitted, data)
        self.assertIn("was not fitted", str(cm.exception))

    def test_selection_records_every_candidate_loss(self):
        fitted, data, _, _ = self._fit(constant_shift_rows(120, 18.0))
        selection = calib.validation_select_calibrator(fitted, data)
        self.assertEqual(set(selection["validation_smooth_l1"]),
                         set(calib.CANDIDATE_ORDER))


class PredictionTests(unittest.TestCase):
    def test_correction_cp_and_prediction_are_consistent(self):
        split, classical = make_split(constant_shift_rows(120, 27.0))
        data = calib.calibration_inputs(split, classical)
        fitted = calib.fit_calibrators(data)
        for name in calib.CANDIDATE_ORDER:
            corrections = calib.correction_cp(fitted[name], split, classical)
            predictions = calib.predict_calibrator(fitted[name], split,
                                                   classical)
            for pid, correction, prediction in zip(split["pids"], corrections,
                                                   predictions):
                self.assertAlmostEqual(prediction,
                                       classical[pid] + correction, places=6)

    def test_unknown_calibrator_fails_closed(self):
        split, classical = make_split(constant_shift_rows(8, 1.0))
        with self.assertRaises(SystemExit) as cm:
            calib.predict_calibrator({"name": "magic", "parameters": {}},
                                     split, classical)
        self.assertIn("unknown calibrator", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
