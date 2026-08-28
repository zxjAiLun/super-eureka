import hashlib
import json
import unittest
from pathlib import Path

from tools.s10.eval_nnue_checkpoint import verify_selection_binding

REAL_SELECTION = {
    "dataset_sha256": "503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3",
    "labels_sha256": "bcd49da1ece75a15591e135d5bcf6d036608b1759d6a00e639f3e344e516116f",
    "feature_set": "v2",
    "environment": {
        "engine_exporter_sha256": "d6649098fe5f47b335454a54cddb57c64afcda157d20cf8a279e4c2f23d02abe",
    },
    "seeds": {
        "20260818": {
            "checkpoint_sha256": "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7",
        },
        "20260819": {
            "checkpoint_sha256": "5b40ffc2489b8698afab28125165c2d0e15498e7e6e06d661bb0e228aa8aa973",
        },
    },
    "selection_summary": {
        "selected_seed": 20260818,
        "selected_checkpoint_sha256": "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7",
    },
}

CKPT_SHA = "d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7"
DATASET_SHA = REAL_SELECTION["dataset_sha256"]
LABELS_SHA = REAL_SELECTION["labels_sha256"]
EXPORTER_SHA = REAL_SELECTION["environment"]["engine_exporter_sha256"]

CKPT_SUMMARY = {"seed": 20260818, "feature_set": "v2"}


class TestSelectionBinding(unittest.TestCase):
    def _check(self, selection=None, ckpt_sha=CKPT_SHA, summary=None,
               dataset_sha=DATASET_SHA, labels_sha=LABELS_SHA,
               exporter_sha=EXPORTER_SHA):
        verify_selection_binding(
            json.loads(json.dumps(selection or REAL_SELECTION)),
            ckpt_sha=ckpt_sha,
            ckpt_summary=summary if summary is not None
            else json.loads(json.dumps(CKPT_SUMMARY)),
            dataset_sha=dataset_sha,
            labels_sha=labels_sha,
            exporter_sha=exporter_sha,
        )

    def test_correct_frozen_selection_passes(self):
        self._check()

    def test_wrong_checkpoint_sha_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._check(ckpt_sha="a" * 64)

    def test_unselected_seed_checkpoint_fails_closed(self):
        # seed 20260819 checkpoint evaluated although selection froze 20260818
        with self.assertRaises(SystemExit):
            self._check(
                ckpt_sha="5b40ffc2489b8698afab28125165c2d0e15498e7e6e06d661bb0e228aa8aa973",
                summary={"seed": 20260819, "feature_set": "v2"},
            )

    def test_unknown_seed_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._check(summary={"seed": 12345, "feature_set": "v2"})

    def test_wrong_dataset_sha_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._check(dataset_sha="b" * 64)

    def test_wrong_labels_sha_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._check(labels_sha="c" * 64)

    def test_wrong_exporter_sha_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._check(exporter_sha="d" * 64)

    def test_wrong_feature_set_fails_closed(self):
        with self.assertRaises(SystemExit):
            self._check(summary={"seed": 20260818, "feature_set": "v1"})

    def test_tampered_selected_sha_in_selection_fails_closed(self):
        sel = json.loads(json.dumps(REAL_SELECTION))
        sel["selection_summary"]["selected_checkpoint_sha256"] = "e" * 64
        with self.assertRaises(SystemExit):
            self._check(selection=sel)


class TestBlindnessClarificationArtifact(unittest.TestCase):
    def test_artifact_fields_present_and_consistent(self):
        path = Path(__file__).resolve().parents[2] / \
            "results/s10/s10-b3-holdout-blindness-clarification.json"
        art = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(art["holdout_performance_observed_before_selection"])
        self.assertFalse(art["holdout_features_exported_before_selection"])
        self.assertFalse(art["holdout_predictions_observed_before_selection"])
        self.assertFalse(art["holdout_loss_observed_before_selection"])
        self.assertFalse(art["holdout_mae_observed_before_selection"])
        self.assertTrue(art["holdout_label_type_metadata_observed_before_selection"])
        self.assertFalse(art["selection_used_holdout_performance"])
        self.assertEqual(art["holdout_usable_cp_count_observed"], 29229)
        self.assertEqual(art["holdout_mate_only_count_observed"], 771)


if __name__ == "__main__":
    unittest.main()
