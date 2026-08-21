#!/usr/bin/env python3
"""Focused tests for the S6-N3D residual library.

Covers the paired-delta / clipped-metric identity, bootstrap determinism,
bucket boundaries, grouped reporting, canonical-checkpoint metadata gates, and
the script-provenance fail-closed paths.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import residual_probe as residual  # noqa: E402
import train_nnue_probe as probe  # noqa: E402


def split_stub(pids, raw_target_cp, phases=None, families=None) -> dict:
    n = len(pids)
    return {
        "white": [torch.tensor([1]) for _ in range(n)],
        "black": [torch.tensor([2]) for _ in range(n)],
        "stm_white": torch.tensor([True] * n),
        "target": torch.zeros(n),
        "raw_target_cp": list(raw_target_cp),
        "fens": ["fen"] * n,
        "pids": list(pids),
        "source_ids": ["s"] * n,
        "source_game_ids": ["g"] * n,
        "phases": list(phases or [18] * n),
        "source_families": list(families or ["lichess-standard-rated-v1"] * n),
    }


class DeltaTests(unittest.TestCase):
    def test_delta_sign_follows_who_is_closer(self):
        deltas = residual.abs_error_deltas(
            residual_cp=[110.0, 300.0],
            classical_cp=[150.0, 200.0],
            raw_target_cp=[100.0, 250.0])
        self.assertEqual(deltas, [-40.0, 0.0])

    def test_delta_uses_the_clipped_pair_on_both_sides(self):
        # Raw target 5000 clips to 2000; a 9000 cp prediction clips to 2000 and
        # therefore scores a perfect 0 error, exactly as clipped_metrics does.
        deltas = residual.abs_error_deltas([9000.0], [1000.0], [5000.0])
        self.assertEqual(deltas, [-1000.0])

    def test_mean_delta_equals_clipped_mae_difference(self):
        residual_cp = [10.0, -250.0, 3000.0, 40.0]
        classical_cp = [90.0, -100.0, 500.0, 45.0]
        target = [0.0, -300.0, 2500.0, 60.0]
        deltas = residual.abs_error_deltas(residual_cp, classical_cp, target)
        mean_delta = sum(deltas) / len(deltas)
        residual_mae = probe.clipped_metrics(
            residual_cp, target)["clipped_mae_cp"]
        classical_mae = probe.clipped_metrics(
            classical_cp, target)["clipped_mae_cp"]
        self.assertAlmostEqual(mean_delta, residual_mae - classical_mae,
                               places=3)

    def test_length_mismatch_fails_closed(self):
        with self.assertRaises(SystemExit):
            residual.abs_error_deltas([1.0], [1.0, 2.0], [1.0])


class BootstrapTests(unittest.TestCase):
    def test_same_seed_is_reproducible_and_seed_matters(self):
        deltas = [float(-10 + (i % 7)) for i in range(400)]
        first = residual.paired_bootstrap(deltas, seed=20260821, resamples=200)
        again = residual.paired_bootstrap(deltas, seed=20260821, resamples=200)
        other = residual.paired_bootstrap(deltas, seed=20260822, resamples=200)
        self.assertEqual(first["ci_lower_cp"], again["ci_lower_cp"])
        self.assertEqual(first["ci_upper_cp"], again["ci_upper_cp"])
        self.assertNotEqual(first["ci_lower_cp"], other["ci_lower_cp"])

    def test_ci_brackets_the_observed_mean(self):
        deltas = [float(-10 + (i % 5)) for i in range(500)]
        result = residual.paired_bootstrap(deltas, resamples=500)
        self.assertLessEqual(result["ci_lower_cp"], result["mean_delta_cp"])
        self.assertGreaterEqual(result["ci_upper_cp"], result["mean_delta_cp"])

    def test_uniformly_negative_deltas_give_ci_upper_below_zero(self):
        result = residual.paired_bootstrap([-5.0] * 300, resamples=300)
        self.assertLess(result["ci_upper_cp"], 0.0)
        self.assertEqual(result["positions_improved"], 300)
        self.assertEqual(result["positions_worsened"], 0)

    def test_uniformly_positive_deltas_give_ci_upper_above_zero(self):
        result = residual.paired_bootstrap([5.0] * 300, resamples=300)
        self.assertGreater(result["ci_upper_cp"], 0.0)
        self.assertEqual(result["positions_worsened"], 300)

    def test_defaults_are_the_frozen_protocol(self):
        self.assertEqual(residual.BOOTSTRAP_SEED, 20260821)
        self.assertEqual(residual.BOOTSTRAP_RESAMPLES, 10000)
        self.assertEqual(residual.CI_PERCENTILES, (2.5, 97.5))

    def test_empty_or_nonpositive_fails_closed(self):
        with self.assertRaises(SystemExit):
            residual.paired_bootstrap([])
        with self.assertRaises(SystemExit):
            residual.paired_bootstrap([1.0], resamples=0)

    def test_non_finite_delta_fails_closed(self):
        with self.assertRaises(SystemExit):
            residual.paired_bootstrap([1.0, float("nan")])


class BucketTests(unittest.TestCase):
    def test_abs_cp_bucket_boundaries_match_clipped_metrics(self):
        self.assertEqual(residual.abs_cp_bucket(0.0), "0-100")
        self.assertEqual(residual.abs_cp_bucket(99.9), "0-100")
        self.assertEqual(residual.abs_cp_bucket(100.0), "100-300")
        self.assertEqual(residual.abs_cp_bucket(-299.9), "100-300")
        self.assertEqual(residual.abs_cp_bucket(300.0), "300-1000")
        self.assertEqual(residual.abs_cp_bucket(1000.0), "1000-inf")
        self.assertEqual(residual.abs_cp_bucket(-9999.0), "1000-inf")

    def test_bucket_assignment_agrees_with_probe_bucket_counts(self):
        targets = [0.0, 50.0, -120.0, 400.0, -1500.0, 2500.0]
        metrics = probe.clipped_metrics([0.0] * len(targets), targets)
        counted: dict[str, int] = {}
        for target in targets:
            key = residual.abs_cp_bucket(target)
            counted[key] = counted.get(key, 0) + 1
        for key, data in metrics["buckets"].items():
            self.assertEqual(data["n"], counted.get(key, 0), key)


class ComparisonTests(unittest.TestCase):
    def test_comparison_reports_improvement_fraction_and_deltas(self):
        result = residual.comparison([100.0, 100.0], [200.0, 200.0],
                                     [100.0, 100.0])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["classical"]["clipped_mae_cp"], 100.0)
        self.assertEqual(result["residual"]["clipped_mae_cp"], 0.0)
        self.assertEqual(result["mae_improvement_fraction"], 1.0)
        self.assertEqual(result["mae_delta_cp"], -100.0)
        self.assertEqual(result["paired_delta"]["mean_delta_cp"], -100.0)

    def test_grouped_comparison_reports_empty_groups_as_zero(self):
        split = split_stub(["a", "b"], [50.0, 50.0], phases=[20, 20])
        grouped = residual.grouped_comparison(
            split, [50.0, 50.0], [80.0, 80.0])
        self.assertEqual(grouped["phase"]["high"]["n"], 2)
        # 'mid', 'low' and 'zero' have no rows and must still be reported.
        for empty in ("mid", "low", "zero"):
            self.assertEqual(grouped["phase"][empty], {"n": 0})
        self.assertEqual(grouped["abs_cp"]["0-100"]["n"], 2)
        self.assertEqual(grouped["abs_cp"]["1000-inf"], {"n": 0})
        self.assertEqual(
            grouped["family"]["lichess-standard-rated-v1"]["n"], 2)

    def test_grouped_comparison_splits_by_abs_cp_bucket(self):
        split = split_stub(["a", "b", "c"], [10.0, 500.0, 5000.0])
        grouped = residual.grouped_comparison(
            split, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(grouped["abs_cp"]["0-100"]["n"], 1)
        self.assertEqual(grouped["abs_cp"]["300-1000"]["n"], 1)
        self.assertEqual(grouped["abs_cp"]["1000-inf"]["n"], 1)

    def test_unknown_group_kind_fails_closed(self):
        split = split_stub(["a"], [10.0])
        with self.assertRaises(SystemExit):
            residual.grouped_comparison(split, [0.0], [0.0], kinds=("nope",))


class CanonicalCheckpointTests(unittest.TestCase):
    def _payload(self, **overrides) -> dict:
        model = residual.build_residual_model()
        payload = residual.canonical_checkpoint_payload(
            model, width=residual.RESIDUAL_WIDTH, seed=residual.RESIDUAL_SEED,
            dataset_sha256="d" * 64, labels_sha256="l" * 64,
            classical_cache_sha256="c" * 64, engine_binary_sha256="e" * 64,
            best_epoch=1, best_val_loss=0.117962,
            trainer_git_sha="g" * 40, trainer_blob_sha256="b" * 64,
            dataset_id="s6-eval-v1-multisource-pilot01")
        payload.update(overrides)
        return payload

    def test_roundtrip_preserves_required_metadata(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(), path)
            model, metadata = residual.load_canonical_checkpoint(path)
            self.assertEqual(metadata["target_mode"], "classical_residual")
            self.assertEqual(metadata["width"], 16)
            self.assertEqual(metadata["seed"], 20260818)
            self.assertEqual(metadata["best_epoch"], 1)
            self.assertEqual(metadata["classical_cache_sha256"], "c" * 64)
            self.assertEqual(metadata["engine_binary_sha256"], "e" * 64)
            self.assertEqual(metadata["trainer_git_sha"], "g" * 40)
            self.assertNotIn("state_dict", metadata)
            self.assertEqual(model.head.out_features, 1)
            self.assertEqual(model.features.out_features, 16)

    def test_missing_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            payload = self._payload()
            del payload["classical_cache_sha256"]
            torch.save(payload, path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("missing metadata", str(cm.exception))

    def test_wrong_target_mode_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(target_mode="absolute"), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("target_mode", str(cm.exception))

    def test_wrong_seed_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(seed=20260819), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("checkpoint seed 20260819", str(cm.exception))

    def test_expected_sha256_mismatch_fails_before_deserialization(self):
        """A substituted file must be rejected without ever being unpickled."""
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(), path)
            with mock.patch.object(residual.torch, "load") as loader:
                with self.assertRaises(SystemExit) as cm:
                    residual.load_canonical_checkpoint(
                        path, expected_sha256="0" * 64)
            loader.assert_not_called()
            self.assertIn("checkpoint sha256", str(cm.exception))

    def test_expected_sha256_match_is_accepted_and_recorded(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(), path)
            actual = residual.sha256_file(path)
            _, metadata = residual.load_canonical_checkpoint(
                path, expected_sha256=actual)
            self.assertEqual(metadata["checkpoint_sha256"], actual)

    def test_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(Path(tmp) / "nope.pt")
            self.assertIn("checkpoint missing", str(cm.exception))

    def test_wrong_target_formula_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(
                target_formula="teacher_cp_stm / 1000"), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("target_formula", str(cm.exception))

    def test_wrong_inference_formula_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(
                inference_formula="residual * 1000"), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("inference_formula", str(cm.exception))

    def test_wrong_target_scale_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(target_scale=100.0), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("target_scale", str(cm.exception))

    def test_wrong_clip_cp_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(clip_cp=600.0), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("clip_cp", str(cm.exception))

    def test_wrong_activation_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            payload = self._payload()
            payload["architecture"] = dict(payload["architecture"],
                                           activation="tanh")
            torch.save(payload, path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("architecture", str(cm.exception))

    def test_wrong_head_wiring_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            payload = self._payload()
            payload["architecture"] = dict(payload["architecture"],
                                           head="width->1 linear")
            torch.save(payload, path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("architecture", str(cm.exception))

    def test_wrong_inputs_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            payload = self._payload()
            payload["architecture"] = dict(payload["architecture"],
                                           inputs=20480)
            torch.save(payload, path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("architecture", str(cm.exception))

    def test_extra_architecture_field_fails_closed(self):
        """Exact-equality means a smuggled extra field is also rejected."""
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            payload = self._payload()
            payload["architecture"] = dict(payload["architecture"],
                                           quantized=True)
            torch.save(payload, path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("architecture", str(cm.exception))

    def test_non_dict_architecture_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            torch.save(self._payload(architecture="40960x16"), path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            self.assertIn("architecture not a dict", str(cm.exception))

    def test_expected_architecture_matches_the_frozen_contract(self):
        self.assertEqual(residual.EXPECTED_ARCHITECTURE, {
            "inputs": 40960, "width": 16, "activation": "relu",
            "head": "concat(own,opp) width*2->1 linear"})

    def test_required_keys_cover_the_contract_fields(self):
        for key in ("target_formula", "inference_formula", "clip_cp",
                    "target_scale"):
            self.assertIn(key, residual.REQUIRED_CHECKPOINT_KEYS)

    def test_width_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-ckpt-") as tmp:
            path = Path(tmp) / "canonical.pt"
            payload = self._payload()
            payload["architecture"] = dict(payload["architecture"], width=8)
            torch.save(payload, path)
            with self.assertRaises(SystemExit) as cm:
                residual.load_canonical_checkpoint(path)
            # width now fails via exact architecture comparison
            self.assertIn("architecture", str(cm.exception))
            self.assertIn("'width': 8", str(cm.exception))


class ProvenanceTests(unittest.TestCase):
    """script_provenance must reject a dirty tree and a stale committed blob.

    `probe.subprocess` and `residual.subprocess` are the same module object, so
    one patch covers both the trainer bind and the own-blob comparison. The
    `git show HEAD:<path>` handler serves the REAL file bytes unless the test
    overrides a specific path, so only the intended check can fail.
    """

    REPO = Path(residual.__file__).resolve().parents[2]
    SCRIPT = Path(residual.__file__).resolve()

    def _git(self, diff_rc: int = 0, override: dict[str, bytes] | None = None):
        repo = self.REPO

        def side_effect(cmd, *args, **kwargs):
            parts = [str(part) for part in cmd]
            if "diff" in parts:
                return mock.Mock(returncode=diff_rc, stdout=b"")
            if "rev-parse" in parts:
                return mock.Mock(returncode=0, stdout="a" * 40 + "\n")
            for part in parts:
                if part.startswith("HEAD:"):
                    relative = part.split(":", 1)[1]
                    if override and relative in override:
                        return mock.Mock(returncode=0,
                                         stdout=override[relative])
                    return mock.Mock(
                        returncode=0,
                        stdout=(repo / relative).read_bytes())
            raise AssertionError(f"unexpected git call {parts}")
        return side_effect

    def test_dirty_worktree_fails_closed(self):
        with mock.patch.object(probe.subprocess, "run",
                               side_effect=self._git(diff_rc=1)):
            with self.assertRaises(SystemExit) as cm:
                residual.script_provenance(self.REPO, self.SCRIPT)
        self.assertIn("not clean", str(cm.exception))

    def test_stale_trainer_blob_fails_closed(self):
        with mock.patch.object(probe.subprocess, "run", side_effect=self._git(
                override={"tools/s6/train_nnue_probe.py": b"stale trainer"})):
            with self.assertRaises(SystemExit) as cm:
                residual.script_provenance(self.REPO, self.SCRIPT)
        self.assertIn("disk trainer differs from HEAD blob", str(cm.exception))

    def test_stale_own_script_blob_fails_closed(self):
        with mock.patch.object(probe.subprocess, "run", side_effect=self._git(
                override={"tools/s6/residual_probe.py": b"stale script"})):
            with self.assertRaises(SystemExit) as cm:
                residual.script_provenance(self.REPO, self.SCRIPT)
        message = str(cm.exception)
        self.assertIn("residual_probe.py differs from HEAD blob", message)

    def test_matching_blobs_bind_the_run(self):
        with mock.patch.object(probe.subprocess, "run",
                               side_effect=self._git()):
            bound = residual.script_provenance(self.REPO, self.SCRIPT)
        self.assertTrue(bound["run_started_clean"])
        self.assertEqual(bound["script_relpath"],
                         "tools/s6/residual_probe.py")
        self.assertEqual(bound["script_sha256"],
                         bound["committed_script_blob_sha256"])
        self.assertEqual(bound["trainer_script_sha256"],
                         bound["committed_trainer_blob_sha256"])


if __name__ == "__main__":
    unittest.main()
