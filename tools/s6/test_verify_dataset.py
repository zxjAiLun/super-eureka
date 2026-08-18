#!/usr/bin/env python3
"""verify_dataset tests: --allow-unlabeled skips ONLY labels/teacher checks;
all other integrity checks stay enforced."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_dataset as vd  # noqa: E402

PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7), "zero": (0, 0)}


def phase_of(board: chess.Board) -> int:
    total = 0
    for _, piece in board.piece_map().items():
        total += PHASE_WEIGHTS.get(piece.piece_type, 0)
    return min(24, total)


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


def make_record(fen: str, split: str, game_id: str) -> dict:
    board = chess.Board(fen)
    fen4 = " ".join(board.fen().split(" ")[:4])
    return {
        "position_id": hashlib.sha256(fen4.encode("utf-8")).hexdigest(),
        "fen": board.fen(),
        "canonical_fen4": fen4,
        "phase": phase_of(board),
        "source_game_id": game_id,
        "split": split,
    }


def make_dataset_dir(tmp: Path, with_labels: bool,
                     corrupt_canonical: bool = False) -> Path:
    d = tmp / "ds"
    d.mkdir()
    records = [
        make_record("r1bq1rk1/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/"
                    "PPP2PPP/RNBQ1RK1 w - - 6 6", "train", "g1"),
        make_record("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/"
                    "PPPBBPPP/R3K2R w KQkq - 0 1", "validation", "g2"),
    ]
    for i, r in enumerate(records):
        (d / f"part-000{i}.jsonl").write_text(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8")
    canonical = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                        for r in records)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if corrupt_canonical:
        sha = "f" * 64
    buckets = {"high": 0, "mid": 0, "low": 0, "zero": 0}
    for r in records:
        buckets[bucket_of(r["phase"])] += 1
    (d / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "syn", "records_total": 2, "dataset_sha256": sha,
        "phase_buckets": buckets,
    }))
    if with_labels:
        labels = [
            {"position_id": records[0]["position_id"], "teacher_cp_stm": 100},
            {"position_id": records[1]["position_id"], "teacher_cp_stm": -50},
        ]
        text = "".join(json.dumps(l) + "\n" for l in labels)
        (d / "labels.jsonl").write_text(text, encoding="utf-8")
        (d / "teacher_manifest.json").write_text(json.dumps({
            "engine": "Stockfish 18", "binary_sha256": "0" * 64,
            "labels_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "audit": {"ok": True, "checked": 1000},
        }))
    return d


class VerifyAllowUnlabeledTests(unittest.TestCase):
    def test_without_labels_fails_without_flag(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            d = make_dataset_dir(Path(tmp), with_labels=False)
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=False))
            self.assertNotEqual(rc, 0)

    def test_without_labels_passes_with_flag(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            d = make_dataset_dir(Path(tmp), with_labels=False)
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=True))
            self.assertEqual(rc, 0)

    def test_labeled_dataset_passes_with_flag(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            d = make_dataset_dir(Path(tmp), with_labels=True)
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=True))
            self.assertEqual(rc, 0)

    def test_other_checks_not_degraded_by_flag(self):
        # Corrupted canonical hash must still fail even with --allow-unlabeled.
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            d = make_dataset_dir(Path(tmp), with_labels=False,
                                 corrupt_canonical=True)
            rc = vd.verify(argparse.Namespace(dataset=str(d),
                                              allow_unlabeled=True))
            self.assertNotEqual(rc, 0)

    def test_legacy_namespace_without_flag_attribute_defaults_off(self):
        # build_dataset calls verify(Namespace(dataset=...)) without the flag.
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            d = make_dataset_dir(Path(tmp), with_labels=False)
            rc = vd.verify(argparse.Namespace(dataset=str(d)))
            self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
