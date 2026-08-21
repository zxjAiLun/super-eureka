#!/usr/bin/env python3
"""Focused tests for the frozen S6-N3D confirmation gates.

These assert the gate ARITHMETIC and the fail-closed paths without touching
the real dataset, so a gate can never silently loosen. The thresholds
themselves are frozen constants and are asserted literally.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lichess_select as ls  # noqa: E402
import residual_probe as residual  # noqa: E402
import run_n3d_residual_confirmation as n3d  # noqa: E402


def overall_stub(classical_mae: float, residual_mae: float,
                 classical_rmse: float = 200.0,
                 residual_rmse: float = 190.0) -> dict:
    return {
        "n": 6000,
        "classical": {"clipped_mae_cp": classical_mae,
                      "clipped_rmse_cp": classical_rmse},
        "residual": {"clipped_mae_cp": residual_mae,
                     "clipped_rmse_cp": residual_rmse},
        "mae_delta_cp": round(residual_mae - classical_mae, 6),
        "mae_improvement_fraction": round(
            (classical_mae - residual_mae) / classical_mae, 6),
    }


def bootstrap_stub(ci_upper: float, ci_lower: float = -12.0) -> dict:
    return {
        "n": 6000, "mean_delta_cp": -8.0, "median_delta_cp": -3.0,
        "positions_improved": 4000, "positions_worsened": 2000,
        "positions_unchanged": 0,
        "bootstrap_seed": residual.BOOTSTRAP_SEED,
        "bootstrap_resamples": residual.BOOTSTRAP_RESAMPLES,
        "ci_lower_cp": ci_lower, "ci_upper_cp": ci_upper,
    }


def group_stub(entries: dict) -> dict:
    """entries: name -> (n, classical_mae, residual_mae)."""
    out = {}
    for name, value in entries.items():
        if value is None:
            out[name] = {"n": 0}
            continue
        n, classical_mae, residual_mae = value
        out[name] = {
            "n": n,
            "classical": {"clipped_mae_cp": classical_mae,
                          "clipped_rmse_cp": classical_mae * 1.4},
            "residual": {"clipped_mae_cp": residual_mae,
                         "clipped_rmse_cp": residual_mae * 1.4},
            "paired_delta": {"n": n,
                             "mean_delta_cp": residual_mae - classical_mae,
                             "median_delta_cp": 0.0},
        }
    return out


def identity_stub(passed: bool = True) -> dict:
    return {
        "passed": passed,
        "gates": {
            "game_fingerprint_intersection_zero": passed,
            "eligible_usable_at_least_5000": passed,
            "retained_fraction_at_least_0.90": passed,
        },
    }


def roundtrip_stub(passed: bool = True) -> dict:
    return {"passed": passed, "state_exact_equal": passed,
            "prediction_max_abs_delta_scaled": 0.0}


def gates_for(overall, bootstrap, by_group, identity=None, cache_ok=True,
              roundtrip=None, finite=None) -> dict:
    return n3d.evaluate_gates(
        overall, bootstrap, by_group,
        identity if identity is not None else identity_stub(),
        cache_ok,
        roundtrip if roundtrip is not None else roundtrip_stub(),
        finite if finite is not None else [])


HEALTHY_GROUPS = {
    "phase": group_stub({"high": (2000, 170.0, 160.0),
                         "mid": (2500, 165.0, 158.0),
                         "low": (1200, 150.0, 148.0),
                         "zero": (150, 400.0, 900.0)}),
    "abs_cp": group_stub({"0-100": (3000, 90.0, 85.0),
                          "100-300": (2000, 170.0, 165.0),
                          "300-1000": (900, 300.0, 290.0),
                          "1000-inf": (100, 800.0, 2000.0)}),
}


class FrozenThresholdTests(unittest.TestCase):
    def test_thresholds_are_the_authorized_values(self):
        self.assertEqual(n3d.MIN_MAE_IMPROVEMENT_FRACTION, 0.02)
        self.assertEqual(n3d.CI_UPPER_MUST_BE_BELOW_CP, 0.0)
        self.assertEqual(n3d.MAX_GROUP_REGRESSION_FRACTION, 0.02)
        self.assertEqual(n3d.MIN_GROUP_N, 200)
        self.assertEqual(n3d.MIN_ELIGIBLE_USABLE_POSITIONS, 5000)
        self.assertEqual(n3d.MIN_RETAINED_FRACTION, 0.90)
        self.assertEqual(n3d.CONFIRM_SEED, 20260821)
        self.assertEqual(n3d.PASS_STATUS, "RESIDUAL_CONFIRMATION_PASS")
        self.assertEqual(n3d.FAIL_STATUS, "CONFIRMATION_FAIL")
        self.assertEqual(n3d.PASS_VERDICT,
                         "RESIDUAL_CONFIRMED_AWAITING_RUNTIME_REVIEW")

    def test_authorized_enlargement_identities_are_fixed(self):
        self.assertEqual(n3d.CONFIRM_SELECTED_GAMES, 1400)
        self.assertEqual(n3d.SUPERSEDED_ATTEMPT_GAMES, 1000)
        self.assertEqual(n3d.CONFIRM_SOURCE_ID,
                         "lichess-standard-rated-confirm-v1-g1400")
        self.assertEqual(n3d.CONFIRM_DATASET_ID,
                         "s6-eval-v1-residual-confirm01-g1400")
        self.assertEqual(n3d.CONFIRM_SOURCE_FAMILY,
                         "lichess-standard-rated-v1")
        self.assertEqual(n3d.SAMPLE_SIZE_ADJUSTMENT_REASON,
                         "pre-metric construction precondition shortfall")
        # The enlargement changed data volume only; the seed is untouched.
        self.assertEqual(n3d.CONFIRM_SEED, 20260821)


class GateArithmeticTests(unittest.TestCase):
    def test_all_gates_pass_on_a_healthy_result(self):
        result = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                           HEALTHY_GROUPS)
        self.assertTrue(result["all_passed"])
        self.assertEqual(len(result["gates"]), 6)

    def test_gate1_needs_at_least_two_percent(self):
        just_under = gates_for(overall_stub(100.0, 98.1), bootstrap_stub(-4.0),
                               HEALTHY_GROUPS)
        self.assertFalse(just_under["gates"][0]["passed"])
        self.assertFalse(just_under["all_passed"])
        exactly = gates_for(overall_stub(100.0, 98.0), bootstrap_stub(-4.0),
                            HEALTHY_GROUPS)
        self.assertTrue(exactly["gates"][0]["passed"])

    def test_gate1_fails_when_residual_is_worse(self):
        result = gates_for(overall_stub(100.0, 105.0), bootstrap_stub(-4.0),
                           HEALTHY_GROUPS)
        self.assertFalse(result["gates"][0]["passed"])

    def test_gate2_requires_ci_upper_strictly_below_zero(self):
        zero = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(0.0),
                         HEALTHY_GROUPS)
        self.assertFalse(zero["gates"][1]["passed"])
        negative = gates_for(overall_stub(170.0, 160.0),
                             bootstrap_stub(-0.0001), HEALTHY_GROUPS)
        self.assertTrue(negative["gates"][1]["passed"])
        positive = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(1.5),
                             HEALTHY_GROUPS)
        self.assertFalse(positive["gates"][1]["passed"])

    def test_gate2_records_the_frozen_bootstrap_protocol(self):
        result = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                           HEALTHY_GROUPS)
        gate = result["gates"][1]
        self.assertEqual(gate["bootstrap_seed"], 20260821)
        self.assertEqual(gate["bootstrap_resamples"], 10000)

    def test_gate3_allows_equal_rmse_but_not_worse(self):
        equal = gates_for(overall_stub(170.0, 160.0, 200.0, 200.0),
                          bootstrap_stub(-4.0), HEALTHY_GROUPS)
        self.assertTrue(equal["gates"][2]["passed"])
        worse = gates_for(overall_stub(170.0, 160.0, 200.0, 200.01),
                          bootstrap_stub(-4.0), HEALTHY_GROUPS)
        self.assertFalse(worse["gates"][2]["passed"])
        self.assertFalse(worse["all_passed"])

    def test_gate4_ignores_small_buckets_and_catches_big_ones(self):
        # 'zero' regresses catastrophically but has n=150 < 200: not evaluated.
        healthy = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                            HEALTHY_GROUPS)
        gate = healthy["gates"][3]
        self.assertFalse(gate["groups"]["zero"]["evaluated"])
        self.assertEqual(gate["evaluated_groups"], 3)
        self.assertTrue(gate["passed"])

        groups = dict(HEALTHY_GROUPS)
        groups["phase"] = group_stub({"high": (2000, 170.0, 160.0),
                                      "mid": (2500, 100.0, 103.0),
                                      "low": (1200, 150.0, 148.0),
                                      "zero": None})
        regressed = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                              groups)
        self.assertFalse(regressed["gates"][3]["groups"]["mid"]["passed"])
        self.assertFalse(regressed["all_passed"])

    def test_gate4_boundary_is_exactly_two_percent(self):
        groups = dict(HEALTHY_GROUPS)
        groups["phase"] = group_stub({"high": (2000, 100.0, 102.0),
                                      "mid": None, "low": None, "zero": None})
        at_limit = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                             groups)
        self.assertTrue(at_limit["gates"][3]["groups"]["high"]["passed"])

        groups["phase"] = group_stub({"high": (2000, 100.0, 102.01),
                                      "mid": None, "low": None, "zero": None})
        over_limit = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                               groups)
        self.assertFalse(over_limit["gates"][3]["groups"]["high"]["passed"])

    def test_gate5_catches_abs_cp_bucket_regression(self):
        groups = dict(HEALTHY_GROUPS)
        groups["abs_cp"] = group_stub({"0-100": (3000, 90.0, 120.0),
                                       "100-300": None, "300-1000": None,
                                       "1000-inf": None})
        result = gates_for(overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                           groups)
        self.assertFalse(result["gates"][4]["groups"]["0-100"]["passed"])
        self.assertFalse(result["all_passed"])

    def test_gate6_fails_on_non_finite_roundtrip_identity_or_cache(self):
        base = (overall_stub(170.0, 160.0), bootstrap_stub(-4.0),
                HEALTHY_GROUPS)
        self.assertFalse(
            gates_for(*base, finite=["/confirmation/overall"])
            ["gates"][5]["passed"])
        self.assertFalse(
            gates_for(*base, roundtrip=roundtrip_stub(False))
            ["gates"][5]["passed"])
        self.assertFalse(
            gates_for(*base, identity=identity_stub(False))
            ["gates"][5]["passed"])
        self.assertFalse(
            gates_for(*base, cache_ok=False)["gates"][5]["passed"])


class FiniteScanTests(unittest.TestCase):
    def test_all_finite_detects_nan_and_inf_by_path(self):
        payload = {"a": {"b": [1.0, float("nan")]},
                   "c": float("inf"), "d": 3.0, "e": "text", "f": 7}
        bad = n3d.all_finite(payload)
        self.assertIn("/a/b[1]", bad)
        self.assertIn("/c", bad)
        self.assertEqual(len(bad), 2)

    def test_all_finite_accepts_clean_payload(self):
        self.assertEqual(n3d.all_finite({"a": [1.0, 2.0], "b": {"c": 3.0}}), [])


class TeacherGateTests(unittest.TestCase):
    def _manifest(self, **overrides) -> dict:
        manifest = {
            "verified_binary_sha256": n3d.EXPECTED_TEACHER_BINARY_SHA256,
            "nodes": 16384,
            "options": {"Threads": "1", "Hash": "64", "MultiPV": "1",
                        "UCI_ShowWDL": "true"},
            "uci_id_name": "Stockfish 18",
            "labels_sha256": "f" * 64,
            "labeled_positions": 7000,
            "audit": {"ok": True, "checked": 1000, "mismatches": [],
                      "mode": "fresh-second-pass", "sample_count": 1000,
                      "sample_position_id_sha256": "s" * 64},
        }
        manifest.update(overrides)
        return manifest

    def test_healthy_manifest_passes(self):
        report = n3d.verify_teacher(self._manifest())
        self.assertEqual(report["audit_checked"], 1000)
        self.assertEqual(report["audit_mismatches"], 0)
        self.assertEqual(report["audit_mode"], "fresh-second-pass")

    def test_wrong_binary_sha_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_teacher(self._manifest(verified_binary_sha256="0" * 64))
        self.assertIn("teacher binary sha256", str(cm.exception))

    def test_wrong_nodes_or_options_fail_closed(self):
        with self.assertRaises(SystemExit):
            n3d.verify_teacher(self._manifest(nodes=32768))
        with self.assertRaises(SystemExit):
            n3d.verify_teacher(self._manifest(
                options={"Threads": "4", "Hash": "64", "MultiPV": "1",
                         "UCI_ShowWDL": "true"}))

    def test_vs_stored_audit_mode_is_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_teacher(self._manifest(
                audit={"ok": True, "checked": 1000, "mismatches": [],
                       "mode": "vs-stored"}))
        self.assertIn("teacher audit mode", str(cm.exception))

    def test_short_audit_and_mismatches_fail_closed(self):
        with self.assertRaises(SystemExit):
            n3d.verify_teacher(self._manifest(
                audit={"ok": True, "checked": 999, "mismatches": [],
                       "mode": "fresh-second-pass"}))
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_teacher(self._manifest(
                audit={"ok": True, "checked": 1000,
                       "mismatches": [{"position_id": "x"}],
                       "mode": "fresh-second-pass"}))
        self.assertIn("mismatches", str(cm.exception))


class ConfirmSourceGateTests(unittest.TestCase):
    def _write(self, tmp: Path, **overrides) -> Path:
        import json
        source_dir = tmp / "confirm"
        source_dir.mkdir(parents=True, exist_ok=True)
        pgn = source_dir / f"{n3d.CONFIRM_SOURCE_ID}.pgn"
        pgn.write_text("[Event \"x\"]\n\n1. e4 e5 *\n\n", encoding="utf-8")
        manifest = {
            "source_id": n3d.CONFIRM_SOURCE_ID,
            "source_family": n3d.CONFIRM_SOURCE_FAMILY,
            "selection_seed": n3d.CONFIRM_SEED,
            "official_sha256": {"2026-07": n3d.EXPECTED_ARCHIVE_SHA256},
            "fingerprint_intersection": 0,
            "exclude_fingerprint_count": 2000,
            "exclude_fingerprints_sha256": "e" * 64,
            "selected_fingerprints_sha256": "s" * 64,
            "selected_fingerprint_count": n3d.CONFIRM_SELECTED_GAMES,
            "games_selected": n3d.CONFIRM_SELECTED_GAMES,
            "script_sha256": "c" * 64,
            "pgn_sha256": residual.sha256_file(pgn),
        }
        manifest.update(overrides)
        (source_dir / "source-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return source_dir

    def test_healthy_source_passes(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            info = n3d.verify_confirm_source(self._write(Path(tmp)))
            self.assertEqual(info["selection_seed"], 20260821)
            self.assertEqual(info["fingerprint_intersection"], 0)
            self.assertEqual(info["games_selected"], 1400)

    def test_superseded_game_count_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            source = self._write(Path(tmp), games_selected=1000)
            with self.assertRaises(SystemExit) as cm:
                n3d.verify_confirm_source(source)
            self.assertIn("confirm selected games", str(cm.exception))

    def test_wrong_seed_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            source = self._write(Path(tmp), selection_seed=20260812)
            with self.assertRaises(SystemExit) as cm:
                n3d.verify_confirm_source(source)
            self.assertIn("confirm seed", str(cm.exception))

    def test_nonzero_intersection_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            source = self._write(Path(tmp), fingerprint_intersection=1)
            with self.assertRaises(SystemExit) as cm:
                n3d.verify_confirm_source(source)
            self.assertIn("intersection", str(cm.exception))

    def test_wrong_archive_sha_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            source = self._write(Path(tmp),
                                 official_sha256={"2026-07": "0" * 64})
            with self.assertRaises(SystemExit) as cm:
                n3d.verify_confirm_source(source)
            self.assertIn("archive SHA", str(cm.exception))

    def test_empty_exclude_set_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            source = self._write(Path(tmp), exclude_fingerprint_count=0)
            with self.assertRaises(SystemExit) as cm:
                n3d.verify_confirm_source(source)
            self.assertIn("no exclude fingerprints", str(cm.exception))

    def test_pgn_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-src-") as tmp:
            source = self._write(Path(tmp), pgn_sha256="0" * 64)
            with self.assertRaises(SystemExit) as cm:
                n3d.verify_confirm_source(source)
            self.assertIn("confirm PGN sha256", str(cm.exception))


class IdentityAuditTests(unittest.TestCase):
    def _confirm(self, count: int, null_count: int = 0) -> dict:
        records = [{"position_id": f"c{i:06d}", "source_game_id": f"g{i}",
                    "source_id": n3d.CONFIRM_SOURCE_ID, "phase": 18,
                    "fen": "fen", "split": "holdout"}
                   for i in range(count)]
        labels = {row["position_id"]:
                  {"teacher_cp_stm": None if i < null_count else 10}
                  for i, row in enumerate(records)}
        return {"records": records, "data": {"labels": labels}}

    def _audit(self, confirm, n3b_records, confirm_keys, n3b_keys):
        with mock.patch.object(n3d, "pgn_fingerprints",
                               return_value=confirm_keys), \
             mock.patch.object(n3d, "n3b_source_fingerprints",
                               return_value=(n3b_keys,
                                             {"per_source_games": {},
                                              "unique_fingerprints":
                                                  len(n3b_keys),
                                              "total_games": len(n3b_keys)})):
            return n3d.identity_audit(confirm, n3b_records, Path("x.pgn"),
                                      [Path("srcs")])

    def test_disjoint_and_large_enough_passes(self):
        audit, eligible = self._audit(
            self._confirm(6000), [{"position_id": "n1"}],
            {"key-a": 1}, {"key-b"})
        self.assertTrue(audit["passed"])
        self.assertEqual(len(eligible), 6000)
        self.assertEqual(audit["positions"]["retained_fraction"], 1.0)
        self.assertEqual(audit["game_fingerprints"]["intersection"], 0)
        self.assertFalse(audit["selection_uses_labels_or_predictions"])
        self.assertFalse(
            audit["confirmation_used_for_training_or_early_stopping"])

    def test_game_fingerprint_overlap_fails_the_gate(self):
        audit, _ = self._audit(self._confirm(6000), [{"position_id": "n1"}],
                               {"shared": 1}, {"shared"})
        self.assertFalse(audit["passed"])
        self.assertFalse(
            audit["gates"]["game_fingerprint_intersection_zero"])
        self.assertEqual(audit["game_fingerprints"]["intersection"], 1)

    def test_too_few_eligible_positions_fails_the_gate(self):
        audit, eligible = self._audit(
            self._confirm(4999), [{"position_id": "n1"}],
            {"key-a": 1}, {"key-b"})
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["gates"]["eligible_usable_at_least_5000"])
        self.assertEqual(len(eligible), 4999)

    def test_position_overlap_is_excluded_and_lowers_retention(self):
        confirm = self._confirm(6000)
        overlapping = [{"position_id": f"c{i:06d}"} for i in range(1000)]
        audit, eligible = self._audit(confirm, overlapping,
                                      {"key-a": 1}, {"key-b"})
        self.assertEqual(audit["positions"]["excluded_positions"], 1000)
        self.assertEqual(len(eligible), 5000)
        self.assertAlmostEqual(
            audit["positions"]["retained_fraction"], 5000 / 6000, places=6)
        self.assertFalse(audit["gates"]["retained_fraction_at_least_0.90"])
        self.assertFalse(audit["passed"])
        self.assertNotIn("c000000",
                         {row["position_id"] for row in eligible})

    def test_null_cp_rows_are_not_counted_as_usable(self):
        audit, eligible = self._audit(
            self._confirm(6000, null_count=500), [{"position_id": "n1"}],
            {"key-a": 1}, {"key-b"})
        self.assertEqual(audit["positions"]["raw_records"], 6000)
        self.assertEqual(audit["positions"]["usable_records"], 5500)
        self.assertEqual(audit["positions"]["null_cp_records"], 500)
        self.assertEqual(len(eligible), 5500)


class EnlargementVerificationTests(unittest.TestCase):
    """The authorized 1000 -> 1400 enlargement must be provably the same draw."""

    DATASET_SHA = "d" * 64

    def _attempt(self, label: str, games: int, keys: list[str], status: str,
                 **overrides) -> dict:
        record = {
            "attempt_label": label,
            "status": status,
            "confirmation_metrics_observed": False,
            "part5_evaluation_run": False,
            "selection": {
                "selected_games": games,
                "selection_seed": n3d.CONFIRM_SEED,
                "exclude_fingerprints_sha256": "e" * 64,
                "exclude_fingerprint_count": 2000,
                "archive_official_sha256": {
                    "2026-07": n3d.EXPECTED_ARCHIVE_SHA256},
                "fingerprint_definition": {"fields": ["initial_fen", "result",
                                                      "moves"]},
            },
            "hashes": {"dataset_sha256": self.DATASET_SHA},
            "ordered_game_fingerprints": keys,
            "ordered_game_fingerprint_count": len(keys),
            "game_fingerprints_sha256": "f" * 64,
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(record.get(key), dict):
                record[key] = {**record[key], **value}
            else:
                record[key] = value
        return record

    def _pair(self, old_keys=None, new_keys=None, **kwargs):
        old_keys = old_keys or [f"old{i:04d}" for i in range(1000)]
        if new_keys is None:
            # Interleave the 400 additions AFTER index 747, mirroring how the
            # 1000-game run saturated its long stratum and started dropping
            # long candidates the 1400-game run keeps.
            new_keys = (old_keys[:748]
                        + [f"new{i:04d}" for i in range(400)]
                        + old_keys[748:])
        old = self._attempt("g1000", 1000, old_keys,
                            "CONSTRUCTION_INSUFFICIENT")
        new = self._attempt("g1400", 1400, new_keys,
                            "CONSTRUCTION_SUFFICIENT", **kwargs)
        return [old, new]

    def test_interleaved_superset_passes(self):
        result = n3d.verify_enlargement(self._pair(), self.DATASET_SHA)
        self.assertTrue(result["passed"])
        self.assertEqual(result["retained_games"], 1000)
        self.assertEqual(result["added_games"], 400)
        self.assertTrue(result["checks"]["superseded_is_ordered_subsequence"])
        self.assertFalse(result["contiguous_prefix_identity_expected"])
        self.assertFalse(result["metrics_observed_before_adjustment"])
        self.assertEqual(result["reason"],
                         "pre-metric construction precondition shortfall")

    def test_contiguous_prefix_also_passes(self):
        """A contiguous prefix is a special case of subsequence containment."""
        old_keys = [f"old{i:04d}" for i in range(1000)]
        new_keys = old_keys + [f"new{i:04d}" for i in range(400)]
        result = n3d.verify_enlargement(
            self._pair(old_keys, new_keys), self.DATASET_SHA)
        self.assertTrue(result["passed"])

    def test_dropping_a_superseded_game_fails_closed(self):
        old_keys = [f"old{i:04d}" for i in range(1000)]
        new_keys = (old_keys[:500] + old_keys[501:]
                    + [f"new{i:04d}" for i in range(401)])
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(self._pair(old_keys, new_keys),
                                   self.DATASET_SHA)
        self.assertIn("enlargement verification failed", str(cm.exception))

    def test_reordering_superseded_games_fails_closed(self):
        old_keys = [f"old{i:04d}" for i in range(1000)]
        swapped = list(old_keys)
        swapped[10], swapped[900] = swapped[900], swapped[10]
        new_keys = swapped + [f"new{i:04d}" for i in range(400)]
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(self._pair(old_keys, new_keys),
                                   self.DATASET_SHA)
        self.assertIn("ordered_subsequence", str(cm.exception))

    def test_changed_exclude_set_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(
                self._pair(selection={"exclude_fingerprints_sha256": "0" * 64}),
                self.DATASET_SHA)
        self.assertIn("exclude fingerprint set sha", str(cm.exception))

    def test_changed_archive_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(
                self._pair(selection={
                    "archive_official_sha256": {"2026-08": "0" * 64}}),
                self.DATASET_SHA)
        self.assertIn("archive official sha", str(cm.exception))

    def test_changed_fingerprint_definition_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(
                self._pair(selection={
                    "fingerprint_definition": {"fields": ["moves"]}}),
                self.DATASET_SHA)
        self.assertIn("fingerprint definition", str(cm.exception))

    def test_wrong_game_count_fails_closed(self):
        attempts = self._pair()
        attempts[1]["selection"]["selected_games"] = 1600
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(attempts, self.DATASET_SHA)
        self.assertIn("confirmation attempt selected games",
                      str(cm.exception))

    def test_dataset_sha_must_match_the_evaluated_dataset(self):
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(self._pair(), "9" * 64)
        self.assertIn("confirmation attempt dataset sha", str(cm.exception))

    def test_insufficient_final_attempt_fails_closed(self):
        attempts = self._pair()
        attempts[1]["status"] = "CONSTRUCTION_INSUFFICIENT"
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(attempts, self.DATASET_SHA)
        self.assertIn("CONSTRUCTION_SUFFICIENT", str(cm.exception))

    def test_missing_superseded_attempt_fails_closed(self):
        attempts = self._pair()
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement([attempts[1]], self.DATASET_SHA)
        self.assertIn("missing superseded attempt", str(cm.exception))

    def test_superseded_attempt_must_not_claim_sufficiency(self):
        attempts = self._pair()
        attempts[0]["status"] = "CONSTRUCTION_SUFFICIENT"
        with self.assertRaises(SystemExit) as cm:
            n3d.verify_enlargement(attempts, self.DATASET_SHA)
        self.assertIn("CONSTRUCTION_INSUFFICIENT", str(cm.exception))


class ConstructionRecordLoadTests(unittest.TestCase):
    def _write(self, tmp: Path, name: str, **overrides) -> Path:
        import json
        record = {"attempt_label": name,
                  "confirmation_metrics_observed": False,
                  "part5_evaluation_run": False}
        record.update(overrides)
        path = tmp / f"{name}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def test_records_are_loaded_with_their_own_sha(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-rec-") as tmp:
            tmp = Path(tmp)
            paths = [self._write(tmp, "g1000"), self._write(tmp, "g1400")]
            attempts = n3d.load_construction_attempts(paths)
            self.assertEqual([a["attempt_label"] for a in attempts],
                             ["g1000", "g1400"])
            for attempt, path in zip(attempts, paths):
                self.assertEqual(attempt["record_sha256"],
                                 residual.sha256_file(path))

    def test_missing_record_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-rec-") as tmp:
            with self.assertRaises(SystemExit) as cm:
                n3d.load_construction_attempts([Path(tmp) / "nope.json"])
            self.assertIn("construction record missing", str(cm.exception))

    def test_record_claiming_observed_metrics_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-rec-") as tmp:
            path = self._write(Path(tmp), "g1400",
                               confirmation_metrics_observed=True)
            with self.assertRaises(SystemExit) as cm:
                n3d.load_construction_attempts([path])
            self.assertIn("metrics were observed", str(cm.exception))

    def test_record_claiming_part5_already_ran_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-rec-") as tmp:
            path = self._write(Path(tmp), "g1400", part5_evaluation_run=True)
            with self.assertRaises(SystemExit) as cm:
                n3d.load_construction_attempts([path])
            self.assertIn("part 5 already ran", str(cm.exception))

    def test_duplicate_labels_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-rec-") as tmp:
            tmp = Path(tmp)
            first = self._write(tmp, "g1400")
            second = tmp / "copy.json"
            second.write_text(first.read_text(), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                n3d.load_construction_attempts([first, second])
            self.assertIn("duplicate construction attempt labels",
                          str(cm.exception))


class SubsequenceHelperTests(unittest.TestCase):
    def test_ordered_subsequence_semantics(self):
        self.assertTrue(ls.is_ordered_subsequence(["a", "c"],
                                                  ["a", "b", "c", "d"]))
        self.assertTrue(ls.is_ordered_subsequence([], ["a"]))
        self.assertTrue(ls.is_ordered_subsequence(["a", "b"], ["a", "b"]))
        self.assertFalse(ls.is_ordered_subsequence(["c", "a"],
                                                   ["a", "b", "c"]))
        self.assertFalse(ls.is_ordered_subsequence(["a", "a"],
                                                   ["a", "b", "c"]))
        self.assertFalse(ls.is_ordered_subsequence(["z"], ["a", "b"]))


class RoundtripTests(unittest.TestCase):
    def test_roundtrip_passes_for_a_real_disk_save_and_reload(self):
        model = residual.build_residual_model()
        split = {
            "white": [torch.tensor([1, 2]), torch.tensor([3])],
            "black": [torch.tensor([4]), torch.tensor([5, 6])],
            "stm_white": torch.tensor([True, False]),
            "target": torch.tensor([0.1, -0.2]),
            "raw_target_cp": [100.0, -200.0],
            "fens": ["f1", "f2"], "pids": ["p1", "p2"],
        }
        with tempfile.TemporaryDirectory(prefix="s6-n3d-rt-") as tmp:
            result = n3d.checkpoint_roundtrip(model, split, Path(tmp))
        self.assertTrue(result["passed"])
        self.assertTrue(result["state_exact_equal"])
        self.assertEqual(result["prediction_max_abs_delta_scaled"], 0.0)
        self.assertTrue(math.isfinite(result["loss_before"]))


if __name__ == "__main__":
    unittest.main()
