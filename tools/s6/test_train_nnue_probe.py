#!/usr/bin/env python3
"""S6-N1 trainer unit tests: data join fail-closed, coverage, metrics,
export parsing, and a small synthetic forward/train smoke run."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_nnue_probe as probe  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MID_FEN = "r1bq1rk1/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 6 6"


def make_synthetic_dataset() -> dict:
    """Temporary dataset dir with two records and correct hashes."""
    records = [
        {"position_id": "a" * 64, "fen": START_FEN, "split": "train",
         "canonical_fen4": START_FEN.split()[0:4].__str__()},
        {"position_id": "b" * 64, "fen": MID_FEN, "split": "validation",
         "canonical_fen4": MID_FEN.split()[0:4].__str__()},
    ]
    labels = [
        {"position_id": "a" * 64, "teacher_cp_stm": 120, "teacher_bestmove": "b1c3"},
        {"position_id": "b" * 64, "teacher_cp_stm": None, "teacher_bestmove": None},
    ]
    tmp = tempfile.TemporaryDirectory(prefix="s6-n1-test-")
    d = Path(tmp.name)
    (d / "part-0000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    labels_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels)
    (d / "labels.jsonl").write_text(labels_text, encoding="utf-8")
    dataset_sha = probe.compute_dataset_sha(records)
    labels_sha = hashlib.sha256(labels_text.encode("utf-8")).hexdigest()
    (d / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "synthetic", "records_total": 2,
        "dataset_sha256": dataset_sha,
    }), encoding="utf-8")
    (d / "teacher_manifest.json").write_text(json.dumps({
        "engine": "Stockfish 18", "binary_sha256": "0" * 64,
        "labels_sha256": labels_sha,
        "audit": {"ok": True, "checked": 1000},
    }), encoding="utf-8")
    return {"tmp": tmp, "dir": d, "records": records, "labels": labels,
            "dataset_sha": dataset_sha, "labels_sha": labels_sha}


class DatasetJoinTests(unittest.TestCase):
    def test_load_synthetic_dataset_ok(self):
        ctx = make_synthetic_dataset()
        try:
            loaded = probe.load_dataset(ctx["dir"])
            self.assertEqual(len(loaded["records"]), 2)
            self.assertEqual(loaded["dataset_sha"], ctx["dataset_sha"])
            self.assertEqual(loaded["labels_sha"], ctx["labels_sha"])
        finally:
            ctx["tmp"].cleanup()

    def test_fail_closed_missing_label(self):
        ctx = make_synthetic_dataset()
        try:
            (ctx["dir"] / "labels.jsonl").write_text(
                json.dumps(ctx["labels"][0]) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                probe.load_dataset(ctx["dir"])
            self.assertIn("without labels", str(cm.exception))
        finally:
            ctx["tmp"].cleanup()

    def test_fail_closed_extra_label(self):
        ctx = make_synthetic_dataset()
        try:
            extra = dict(ctx["labels"][0])
            extra["position_id"] = "c" * 64
            with open(ctx["dir"] / "labels.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(extra) + "\n")
            with self.assertRaises(SystemExit) as cm:
                probe.load_dataset(ctx["dir"])
            self.assertIn("labels without records", str(cm.exception))
        finally:
            ctx["tmp"].cleanup()

    def test_fail_closed_duplicate_position_id(self):
        ctx = make_synthetic_dataset()
        try:
            dup = dict(ctx["records"][0])
            with open(ctx["dir"] / "part-0000.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dup) + "\n")
            with self.assertRaises(SystemExit) as cm:
                probe.load_dataset(ctx["dir"])
            self.assertIn("duplicate position_id", str(cm.exception))
        finally:
            ctx["tmp"].cleanup()

    def test_fail_closed_manifest_hash_mismatch(self):
        ctx = make_synthetic_dataset()
        try:
            manifest = json.loads(
                (ctx["dir"] / "dataset_manifest.json").read_text(encoding="utf-8"))
            manifest["dataset_sha256"] = "f" * 64
            (ctx["dir"] / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                probe.load_dataset(ctx["dir"])
            self.assertIn("dataset_sha256 mismatch", str(cm.exception))
        finally:
            ctx["tmp"].cleanup()

    def test_fail_closed_duplicate_label_position_id(self):
        ctx = make_synthetic_dataset()
        try:
            with open(ctx["dir"] / "labels.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ctx["labels"][0]) + "\n")
            with self.assertRaises(SystemExit) as cm:
                probe.load_dataset(ctx["dir"])
            self.assertIn("duplicate position_id in labels", str(cm.exception))
        finally:
            ctx["tmp"].cleanup()


class ExportParsingTests(unittest.TestCase):
    def test_parse_export_line_ok(self):
        line = json.dumps({
            "position_id": "a" * 64, "fen": START_FEN,
            "white": [1, 2, 3], "black": [4, 5],
        })
        rec = probe.parse_export_line(line)
        self.assertEqual(rec["white"], [1, 2, 3])

    def test_parse_export_line_missing_field_fails(self):
        line = json.dumps({"position_id": "a" * 64, "fen": START_FEN, "white": [1]})
        with self.assertRaises(SystemExit):
            probe.parse_export_line(line)

    def test_parse_export_line_missing_id_fails(self):
        line = json.dumps({"fen": START_FEN, "white": [1], "black": [2]})
        with self.assertRaises(SystemExit):
            probe.parse_export_line(line)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.exported = {
            "a" * 64: {"position_id": "a" * 64, "fen": START_FEN,
                       "white": [0, 1, 2], "black": [3, 4]},
            "b" * 64: {"position_id": "b" * 64, "fen": MID_FEN,
                       "white": [0, 5], "black": [6]},
        }
        self.records = [
            {"position_id": "a" * 64, "fen": START_FEN},
            {"position_id": "b" * 64, "fen": MID_FEN},
        ]

    def test_train_union_counts(self):
        cov = probe.coverage_for_split(self.exported, self.records, None)
        self.assertEqual(cov["white_unique"], 4)   # {0,1,2,5}
        self.assertEqual(cov["black_unique"], 3)   # {3,4,6}
        self.assertEqual(cov["union_unique"], 7)   # {0..6}
        self.assertEqual(cov["total_activations"], 8)

    def test_unseen_rate(self):
        train_union = {0, 1, 2, 3, 4}
        cov = probe.coverage_for_split(self.exported, self.records, train_union)
        self.assertEqual(cov["unseen_activations"], 2)  # 5, 6
        self.assertEqual(cov["positions_with_unseen"], 1)

    def test_unseen_counted_per_activation_across_perspectives(self):
        exported = {"c": {"position_id": "c", "fen": START_FEN,
                          "white": [5, 5], "black": [5]}}
        records = [{"position_id": "c", "fen": START_FEN}]
        cov = probe.coverage_for_split(exported, records, {0, 1, 2, 3, 4})
        # The same unseen index 5 appears twice in white and once in black:
        # three activations, one position.
        self.assertEqual(cov["total_activations"], 3)
        self.assertEqual(cov["unseen_activations"], 3)
        self.assertEqual(cov["unseen_white_activations"], 2)
        self.assertEqual(cov["unseen_black_activations"], 1)
        self.assertEqual(cov["unseen_union_unique"], 1)
        self.assertEqual(cov["positions_with_unseen"], 1)

    def test_null_cp_row_excluded_from_usable_coverage(self):
        exported = {
            "a" * 64: {"position_id": "a" * 64, "fen": START_FEN,
                       "white": [0, 1], "black": [2]},
            "b" * 64: {"position_id": "b" * 64, "fen": MID_FEN,
                       "white": [3], "black": [4]},
        }
        records = [
            {"position_id": "a" * 64, "fen": START_FEN},
            {"position_id": "b" * 64, "fen": MID_FEN},
        ]
        labels = {
            "a" * 64: {"teacher_cp_stm": 120},
            "b" * 64: {"teacher_cp_stm": None},
        }
        prepared = probe.prepare_split(exported, records, labels)
        self.assertEqual(len(prepared["target"]), 1)
        usable = [r for r in records
                  if labels[r["position_id"]]["teacher_cp_stm"] is not None]
        self.assertEqual(len(usable), 1)
        cov = probe.coverage_for_split(exported, usable, None)
        self.assertEqual(cov["positions"], 1)
        self.assertEqual(cov["total_activations"], 3)

    def test_train_activation_frequency(self):
        freq = probe.train_activation_frequency(self.exported, self.records)
        # indices: white {0:2, 1:1, 2:1, 5:1}, black {3:1, 4:1, 6:1}
        self.assertEqual(freq["total_activations"], 8)
        self.assertEqual(freq["observed_unique_features"], 7)
        self.assertEqual(freq["unobserved_features"],
                         probe.NNUE_INPUTS - 7)
        self.assertEqual(freq["singleton_features"], 6)
        self.assertEqual(freq["features_with_activation_le5"], 7)
        self.assertEqual(freq["median_activations_per_feature"], 1)


class MetricsTests(unittest.TestCase):
    def test_clipped_metrics_and_buckets(self):
        pred = [0.0, 0.0, 0.0, 0.0]
        target = [50.0, 200.0, 600.0, 5000.0]
        m = probe.clipped_metrics(pred, target)
        self.assertEqual(m["n"], 4)
        self.assertEqual(m["raw_mae_cp"], 1462.5)  # 50+200+600+5000 / 4
        # clipped: 50, 200, 600, 2000 -> mean 712.5
        self.assertEqual(m["clipped_mae_cp"], 712.5)
        self.assertEqual(m["buckets"]["0-100"]["n"], 1)
        self.assertEqual(m["buckets"]["100-300"]["n"], 1)
        self.assertEqual(m["buckets"]["300-1000"]["n"], 1)
        self.assertEqual(m["buckets"]["1000-inf"]["n"], 1)

    def test_pred_stats(self):
        s = probe.pred_stats([1.0, 2.0, 3.0])
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 3.0)
        self.assertEqual(s["mean"], 2.0)

    def test_raw_target_preserved_and_clip_metrics_distinguish(self):
        exported = {"a" * 64: {"position_id": "a" * 64, "fen": START_FEN,
                               "white": [0], "black": [1]}}
        records = [{"position_id": "a" * 64, "fen": START_FEN, "split": "train"}]
        labels = {"a" * 64: {"position_id": "a" * 64, "teacher_cp_stm": 5000}}
        split = probe.prepare_split(exported, records, labels)
        # Raw target survives; training target is clipped and scaled.
        self.assertEqual(split["raw_target_cp"], [5000.0])
        self.assertEqual(split["target"].tolist(), [2.0])
        m = probe.clipped_metrics([0.0], [5000.0])
        self.assertEqual(m["raw_mae_cp"], 5000.0)
        self.assertEqual(m["clipped_mae_cp"], 2000.0)
        self.assertNotEqual(m["raw_mae_cp"], m["clipped_mae_cp"])


class ModelTests(unittest.TestCase):
    def test_synthetic_forward_no_nan(self):
        model = probe.build_model(seed=probe.SEED)
        own = [torch.tensor([0, 1, 2], dtype=torch.long)]
        opp = [torch.tensor([3, 4], dtype=torch.long)]
        out = model.forward(own, opp)
        self.assertEqual(out.shape, (1,))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_synthetic_train_loss_decreases(self):
        torch.manual_seed(probe.SEED)
        model = probe.build_model(seed=probe.SEED)
        n = 64
        rng = torch.Generator().manual_seed(probe.SEED)
        indices = torch.randint(0, probe.NNUE_INPUTS, (n * 20,),
                                generator=rng).tolist()
        split = {
            "white": [torch.tensor(indices[i * 20:(i + 1) * 20], dtype=torch.long)
                      for i in range(n)],
            "black": [torch.tensor([0, 1], dtype=torch.long)] * n,
            "stm_white": torch.tensor([i % 2 == 0 for i in range(n)]),
            "target": torch.randn(n, generator=rng) * 0.5,
            "fens": [START_FEN] * n, "pids": [f"p{i}" for i in range(n)],
        }
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        criterion = torch.nn.SmoothL1Loss(beta=probe.LOSS_BETA)
        losses = []
        for _ in range(50):
            own, opp = probe.stm_ordered(split, list(range(n)))
            opt.zero_grad()
            loss = criterion(model.forward(own, opp), split["target"])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertFalse(any(math.isnan(l) for l in losses))
        self.assertLess(losses[-1], losses[0])

    def test_stm_ordering(self):
        split = {
            "white": [torch.tensor([1]), torch.tensor([2])],
            "black": [torch.tensor([3]), torch.tensor([4])],
            "stm_white": torch.tensor([True, False]),
            "target": torch.zeros(2), "fens": ["x"] * 2, "pids": ["p", "q"],
        }
        own, opp = probe.stm_ordered(split, [0, 1])
        self.assertEqual(own[0].item(), 1)  # row 0 stm white -> white
        self.assertEqual(own[1].item(), 4)  # row 1 stm black -> black
        self.assertEqual(opp[0].item(), 3)
        self.assertEqual(opp[1].item(), 2)


def make_probe_split(n: int = 32, seed: int = probe.SEED) -> dict:
    """Small deterministic split for best-state tests."""
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 1000, (n * 20,), generator=rng).tolist()
    return {
        "white": [torch.tensor(indices[i * 20:(i + 1) * 20], dtype=torch.long)
                  for i in range(n)],
        "black": [torch.tensor([0, 1], dtype=torch.long)] * n,
        "stm_white": torch.tensor([i % 2 == 0 for i in range(n)]),
        "target": torch.randn(n, generator=rng) * 0.5,
        "raw_target_cp": [0.0] * n,
        "fens": [START_FEN] * n, "pids": [f"p{i}" for i in range(n)],
    }


class BestStateTests(unittest.TestCase):
    def test_model_after_train_probe_matches_best_val_loss(self):
        model = probe.build_model(seed=probe.SEED)
        train = make_probe_split(32, seed=probe.SEED)
        val = make_probe_split(24, seed=probe.SEED + 1)
        training = probe.train_probe(model, train, val)
        self.assertTrue(training["best_state_restored"])
        self.assertEqual(training["restored_validation_loss"],
                         training["best_val_loss"])
        _, val_loss = probe.evaluate_split(model, val)
        self.assertAlmostEqual(val_loss, training["best_val_loss"], places=6,
                               msg="model must hold the BEST state, not the final epoch")
        # The final-epoch state is (with high probability) a different loss;
        # at minimum the best epoch must be well-defined.
        self.assertGreaterEqual(training["best_epoch"], 1)

    def test_checkpoint_round_trip_keeps_best_validation_loss(self):
        model = probe.build_model(seed=probe.SEED)
        train = make_probe_split(32, seed=probe.SEED)
        val = make_probe_split(24, seed=probe.SEED + 1)
        training = probe.train_probe(model, train, val)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            torch.save({
                "state_dict": model.state_dict(),
                "architecture": {"inputs": probe.NNUE_INPUTS,
                                 "width": probe.WIDTH},
                "seed": probe.SEED,
                "dataset_sha256": "d" * 64,
                "labels_sha256": "l" * 64,
                "best_epoch": training["best_epoch"],
                "best_val_loss": training["best_val_loss"],
            }, path)
            loaded = probe.build_model(seed=probe.SEED)
            ckpt = torch.load(path, weights_only=True)
            loaded.load_state_dict(ckpt["state_dict"])
            _, roundtrip = probe.evaluate_split(loaded, val)
            self.assertAlmostEqual(roundtrip, training["best_val_loss"], places=6)
            self.assertEqual(ckpt["best_epoch"], training["best_epoch"])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
