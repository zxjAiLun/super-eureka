#!/usr/bin/env python3
"""Focused tests for the S6-N3C diagnostics gates and bookkeeping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_n3c_probe_diagnostics as diag  # noqa: E402
import train_nnue_probe as probe  # noqa: E402


def row(pid: str, game: str = "g", source: str = "s", phase: int = 18) -> dict:
    return {
        "position_id": pid,
        "source_game_id": game,
        "source_id": source,
        "phase": phase,
        "fen": "fen",
        "split": "holdout",
    }


class StatisticsTests(unittest.TestCase):
    def test_target_statistics_preserves_nulls_and_buckets(self):
        records = [row("a"), row("b"), row("c"), row("d")]
        labels = {
            "a": {"teacher_cp_stm": None},
            "b": {"teacher_cp_stm": 50},
            "c": {"teacher_cp_stm": -400},
            "d": {"teacher_cp_stm": 1500},
        }
        stats = diag.target_statistics(records, labels)
        self.assertEqual(stats["raw_records"], 4)
        self.assertEqual(stats["usable_records"], 3)
        self.assertEqual(stats["null_cp_records"], 1)
        self.assertEqual(stats["abs_cp_buckets"]["0-100"]["n"], 1)
        self.assertEqual(stats["abs_cp_buckets"]["300-1000"]["n"], 1)
        self.assertEqual(stats["abs_cp_buckets"]["1000-inf"]["n"], 1)

    def test_feature_statistics_counts_activation_frequency(self):
        exported = {
            "a": {"white": [1, 2], "black": [3]},
            "b": {"white": [1], "black": [4, 4]},
        }
        stats = diag.feature_statistics(exported, [row("a"), row("b")])
        self.assertEqual(stats["union_unique"], 4)
        self.assertEqual(stats["total_activations"], 6)
        self.assertEqual(stats["singleton_features"], 2)
        self.assertEqual(stats["features_with_activation_le5"], 4)


class IdentityAndGateTests(unittest.TestCase):
    def test_identity_filter_excludes_only_position_ids(self):
        rows = [row("a", "eval-a"), row("b", "eval-b")]
        filtered, audit = diag.identity_filter(
            rows, {"n1": [row("a", "train-a")]}, "test")
        self.assertEqual([r["position_id"] for r in filtered], ["b"])
        self.assertEqual(audit["excluded_positions"], 1)
        self.assertFalse(audit["selection_uses_labels_or_predictions"])

    def test_identity_filter_fails_on_source_game_overlap(self):
        with self.assertRaises(SystemExit) as cm:
            diag.identity_filter(
                [row("a", "same-game")],
                {"n1": [row("b", "same-game")]}, "test")
        self.assertIn("source_game_id overlap", str(cm.exception))

    def test_clear_improvement_requires_five_percent_and_two_seeds(self):
        good = diag.clear_improvement([(90.0, 100.0), (92.0, 100.0),
                                       (101.0, 100.0)])
        self.assertTrue(good["at_least_5_percent_better"])
        self.assertTrue(good["at_least_2_of_3_seeds_better"])

        bad = diag.clear_improvement([(96.0, 100.0), (97.0, 100.0),
                                      (98.0, 100.0)])
        self.assertFalse(bad["at_least_5_percent_better"])


class CheckpointAndResidualTests(unittest.TestCase):
    def test_state_comparison_reports_tensor_difference(self):
        left = {"x": torch.tensor([1.0, 2.0])}
        right = {"x": torch.tensor([1.0, 3.0])}
        result = diag.state_comparison(left, right)
        self.assertFalse(result["exact_equal"])
        self.assertEqual(result["max_abs_delta"], 1.0)

    def test_residual_split_uses_clipped_scaled_residual_target(self):
        prepared = {
            "white": [torch.tensor([1])],
            "black": [torch.tensor([2])],
            "stm_white": torch.tensor([True]),
            "target": torch.tensor([0.5]),
            "raw_target_cp": [500.0],
            "fens": ["fen"],
            "pids": ["p"],
            "source_ids": ["s"],
            "source_game_ids": ["g"],
            "phases": [18],
            "source_families": ["arena"],
        }
        result = diag.residual_split(prepared, {"p": 100.0})
        self.assertEqual(result["residual_raw_target_cp"], [400.0])
        self.assertAlmostEqual(result["target"].item(), 0.4, places=6)

    def test_model_builder_is_cpu_only(self):
        model = diag.make_model(4, 20260818)
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
