#!/usr/bin/env python3
"""label_teacher audit tests: fresh-second-pass uses a NEW Teacher instance,
second-pass mismatch fails without publish, duplicate stored labels fail,
and a successful publish passes verify_dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import label_teacher as lt  # noqa: E402
import verify_dataset as vd  # noqa: E402

PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7), "zero": (0, 0)}


def phase_of(board: chess.Board) -> int:
    total = 0
    for _, p in board.piece_map().items():
        total += PHASE_WEIGHTS.get(p.piece_type, 0)
    return min(24, total)


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


def random_midgame(seed: int) -> str:
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(8):
        legal = list(board.legal_moves)
        if not legal:
            return None
        board.push(rng.choice(legal))
    if board.is_game_over(claim_draw=False) or board.is_check() \
            or not any(board.legal_moves) or not board.is_valid():
        return None
    return board.fen()


def make_valid_dataset(tmp: Path, n: int = 1000) -> Path:
    d = tmp / "ds"
    d.mkdir()
    records = []
    seed = 0
    while len(records) < n:
        fen = random_midgame(seed)
        seed += 1
        if fen is None:
            continue
        board = chess.Board(fen)
        fen4 = " ".join(fen.split(" ")[:4])
        records.append({
            "position_id": hashlib.sha256(fen4.encode()).hexdigest(),
            "fen": fen,
            "canonical_fen4": fen4,
            "phase": phase_of(board),
            "source_game_id": f"g{len(records)}",
            "split": "train",
        })
    (d / "part-0000.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in records), encoding="utf-8")
    canonical = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                        for r in records)
    buckets = {"high": 0, "mid": 0, "low": 0, "zero": 0}
    for r in records:
        buckets[bucket_of(r["phase"])] += 1
    (d / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "syn", "records_total": len(records),
        "dataset_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "phase_buckets": buckets,
    }))
    return d


class FakeTeacher:
    """Deterministic fake; audit instances are tracked. MISMATCH makes the
    SECOND teacher instance return different labels."""

    instances = 0
    MISMATCH = False
    SCORE_BASE = 100

    def __init__(self, wsl: bool = True, binary=None,
                 expected_binary_sha256: str | None = None):
        type(self).instances += 1
        self.seq = type(self).instances
        self.uci_id_name = "FakeStockfish"
        self.uci_id_author = "fake"
        self.uci_options = {"Threads": "1"}
        self.verified_binary_sha256 = "0" * 64
        self.closed = False

    def label(self, fen: str) -> dict:
        offset = 999 if (type(self).MISMATCH and self.seq == 2) else 0
        return {
            "teacher_cp_stm": type(self).SCORE_BASE + offset,
            "teacher_mate": None,
            "teacher_bestmove": "e2e4",
            "teacher_wdl_stm": [990, 10, 0],
            "nodes": lt.TEACHER_NODES,
        }

    def close(self):
        self.closed = True


def run_main(dataset: Path, n: int | None = None) -> int:
    argv = ["label_teacher.py", "--dataset", str(dataset), "--native"]
    if n is not None:
        argv += ["--audit-n", str(n)]
    with mock.patch.object(sys, "argv", argv), \
         mock.patch.object(lt, "Teacher", FakeTeacher):
        return lt.main()


class LabelTeacherAuditTests(unittest.TestCase):
    def setUp(self):
        FakeTeacher.instances = 0
        FakeTeacher.MISMATCH = False

    def test_fresh_second_pass_uses_new_teacher_instance(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3b-") as tmp:
            d = make_valid_dataset(Path(tmp), n=8)
            rc = run_main(d)
            self.assertEqual(rc, 0)
            # original + audit second pass = 2 instances
            self.assertEqual(FakeTeacher.instances, 2)
            manifest = json.loads(
                (d / "teacher_manifest.json").read_text())
            self.assertEqual(manifest["audit"]["mode"],
                             "fresh-second-pass")
            self.assertEqual(manifest["audit"]["checked"], 8)
            self.assertTrue(manifest["audit"]["ok"])
            self.assertTrue((d / "labels.jsonl").is_file())

    def test_second_pass_mismatch_fails_without_publish(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3b-") as tmp:
            d = make_valid_dataset(Path(tmp), n=8)
            FakeTeacher.MISMATCH = True
            rc = run_main(d)
            self.assertEqual(rc, 3)
            self.assertFalse((d / "labels.jsonl").exists())
            self.assertFalse((d / "teacher_manifest.json").exists())

    def test_duplicate_stored_label_fails(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3b-") as tmp:
            d = make_valid_dataset(Path(tmp), n=8)
            pid = "a" * 64
            (d / "labels.jsonl").write_text(
                json.dumps({"position_id": pid}) + "\n" +
                json.dumps({"position_id": pid}) + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                run_main(d)
            self.assertFalse((d / "teacher_manifest.json").exists())

    def test_vs_stored_mode_and_publish_passes_verify(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3b-") as tmp:
            d = make_valid_dataset(Path(tmp), n=1000)
            rc = run_main(d)
            self.assertEqual(rc, 0)
            manifest = json.loads(
                (d / "teacher_manifest.json").read_text())
            self.assertEqual(manifest["audit"]["mode"],
                             "fresh-second-pass")
            self.assertEqual(manifest["audit"]["checked"], 1000)
            self.assertTrue(manifest["audit"]["ok"])
            self.assertEqual(len(manifest["audit"]
                                 ["sample_position_id_sha256"]), 64)
            # verify_dataset (labeled mode, no --allow-unlabeled) passes
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=False))
            self.assertEqual(rc, 0)

    def test_vs_stored_mode_used_when_labels_exist(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3b-") as tmp:
            d = make_valid_dataset(Path(tmp), n=8)
            # First pass publishes labels; second run must use vs-stored.
            rc = run_main(d)
            self.assertEqual(rc, 0)
            FakeTeacher.instances = 0
            rc = run_main(d)
            self.assertEqual(rc, 0)
            # vs-stored re-labels with a FRESH teacher too
            self.assertEqual(FakeTeacher.instances, 1)
            manifest = json.loads(
                (d / "teacher_manifest.json").read_text())
            self.assertEqual(manifest["audit"]["mode"], "vs-stored")

    def test_sample_position_id_sha_deterministic(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3b-") as tmp:
            d = make_valid_dataset(Path(tmp), n=8)
            run_main(d)
            m1 = json.loads((d / "teacher_manifest.json").read_text())
            run_main(d)  # same records -> same sample sha
            m2 = json.loads((d / "teacher_manifest.json").read_text())
            self.assertEqual(
                m1["audit"]["sample_position_id_sha256"],
                m2["audit"]["sample_position_id_sha256"])


if __name__ == "__main__":
    unittest.main()
