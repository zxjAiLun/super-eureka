#!/usr/bin/env python3
"""S6-N3A build_dataset tests: dual manifest schema, duplicate source ids,
family-mix gate without publish, final-mode staging, and atomic publish."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as bd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def make_pgn(seed: int = 0, result: str = "1-0", elo: int = 2500) -> chess.pgn.Game:
    """Deterministic random legal game with >= MIN_PLY plies (distinct seeds
    yield distinct games -> distinct position_ids across sources)."""
    import random
    rng = random.Random(seed)
    board = chess.Board()
    moves: list[chess.Move] = []
    for _ in range(30):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(claim_draw=False):
            break
        move = rng.choice(legal)
        board.push(move)
        moves.append(move)
    assert len(moves) >= 12, f"seed {seed} produced a too-short game"
    game = chess.pgn.Game()
    game.headers["Event"] = "synthetic"
    game.headers["Result"] = result
    game.headers["White"] = "A"
    game.headers["Black"] = "B"
    game.headers["WhiteElo"] = str(elo)
    game.headers["BlackElo"] = str(elo)
    game.headers["TimeControl"] = "600+5"
    node = game
    for m in moves:
        node = node.add_main_variation(m)
    return game


def pgn_bytes(game: chess.pgn.Game) -> bytes:
    exporter = chess.pgn.StringExporter(headers=True, variations=False,
                                        comments=False)
    return game.accept(exporter).encode("utf-8")


def write_pgn(path: Path, game: chess.pgn.Game) -> str:
    data = pgn_bytes(game)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_pgns(path: Path, games: list[chess.pgn.Game]) -> str:
    # FRESH exporter per game: StringExporter accumulates its internal buffer
    # across accept() calls, so reusing one instance would balloon the output
    # (n games -> sum(1..n) records) and skew family pools.
    def export(g):
        exporter = chess.pgn.StringExporter(headers=True, variations=False,
                                            comments=False)
        return g.accept(exporter) + "\n\n"
    text = "".join(export(g) for g in games)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_aggregate_dir(tmp: Path, name: str, games: list[chess.pgn.Game],
                      family: str = "arena") -> Path:
    d = tmp / name
    d.mkdir()
    manifest = {}
    for i, game in enumerate(games):
        key = f"{name}-{i}"
        sha = write_pgn(d / f"{key}.pgn", game)
        manifest[key] = {
            "source_id": f"{name}-{i}",
            "source_family": family,
            "sha256": sha,
        }
    (d / "source_manifest.json").write_text(json.dumps(manifest, indent=1))
    return d


def make_local_dir(tmp: Path, name: str, game) -> Path:
    d = tmp / name
    d.mkdir()
    games = game if isinstance(game, list) else [game]
    sha = write_pgns(d / "lichess-standard-rated-v1.pgn", games)
    (d / "source-manifest.json").write_text(json.dumps({
        "source_family": "lichess-standard-rated-v1",
        "source_id": "lichess-standard-rated-v1",
        "pgn_sha256": sha,
    }, indent=1))
    return d


def run_build(sources: list[Path], out: Path, dataset_id: str, **kw):
    defaults = dict(sources=[str(s) for s in sources], out=str(out),
                    dataset_id=dataset_id, sampling_version=2,
                    final_mode=False, enforce_family_mix=False)
    defaults.update(kw)
    return bd.build(argparse.Namespace(**defaults))


class CatalogLoaderTests(unittest.TestCase):
    def test_dual_manifest_schema(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            agg = make_aggregate_dir(tmp, "arena", [make_pgn(1), make_pgn(2)])
            lic = make_local_dir(tmp, "lichess", make_pgn(3))
            catalog = bd.load_source_catalog([agg, lic])
            self.assertEqual(len(catalog), 3)
            self.assertEqual(
                catalog["lichess-standard-rated-v1"]["source_family"],
                "lichess-standard-rated-v1")
            self.assertEqual(catalog["arena-0"]["source_family"], "arena")

    def test_duplicate_source_id_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            a = make_aggregate_dir(tmp, "a", [make_pgn()])
            b = make_aggregate_dir(tmp, "b", [make_pgn()])
            # force the same source_id across catalogs
            manifest = json.loads(
                (b / "source_manifest.json").read_text())
            key = next(iter(manifest))
            manifest[key]["source_id"] = "a-0"
            (b / "source_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(SystemExit) as cm:
                bd.load_source_catalog([a, b])
            self.assertIn("duplicate source_id", str(cm.exception))

    def test_duplicate_source_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            a = make_aggregate_dir(tmp, "a", [make_pgn(1)])
            b = make_aggregate_dir(tmp, "b", [make_pgn(2)])
            # Same KEY across catalogs (different source_id) must fail.
            manifest = json.loads(
                (b / "source_manifest.json").read_text())
            key = next(iter(manifest))
            manifest["a-0"] = manifest.pop(key)
            (b / "source_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(SystemExit) as cm:
                bd.load_source_catalog([a, b])
            self.assertIn("duplicate source key", str(cm.exception))

    def test_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            a = make_aggregate_dir(tmp, "a", [make_pgn()])
            manifest = json.loads(
                (a / "source_manifest.json").read_text())
            key = next(iter(manifest))
            manifest[key]["sha256"] = "f" * 64
            (a / "source_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(SystemExit) as cm:
                bd.load_source_catalog([a])
            self.assertIn("SHA mismatch", str(cm.exception))


class BuildPublishTests(unittest.TestCase):
    def test_multisource_build_publishes_atomically(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            agg = make_aggregate_dir(tmp, "arena", [make_pgn(1), make_pgn(2)])
            lic = make_local_dir(tmp, "lichess", make_pgn(3))
            out = tmp / "out"
            rc = run_build([agg, lic], out, "s6-test-pilot01")
            self.assertEqual(rc, 0)
            dataset = out / "s6-test-pilot01"
            self.assertTrue(dataset.is_dir())
            self.assertTrue((dataset / "dataset_manifest.json").is_file())
            self.assertFalse((out / ".staging-s6-test-pilot01").exists())
            manifest = json.loads(
                (dataset / "dataset_manifest.json").read_text())
            self.assertGreater(manifest["records_total"], 0)
            families = manifest["source_families"]
            self.assertIn("arena", families)
            self.assertIn("lichess-standard-rated-v1", families)
            self.assertNotEqual(manifest["not_final_reason"], "FINAL gate not enabled")

    def test_family_mix_gate_does_not_publish(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            agg = make_aggregate_dir(tmp, "arena", [make_pgn()] * 2)
            out = tmp / "out"
            rc = run_build([agg], out, "s6-test-single-family",
                           enforce_family_mix=True)
            self.assertEqual(rc, 4)
            self.assertFalse((out / "s6-test-single-family").exists())

    def test_existing_out_dir_refused(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            agg = make_aggregate_dir(tmp, "arena", [make_pgn()] * 2)
            out = tmp / "out"
            target = out / "s6-test-exists"
            target.mkdir(parents=True)
            rc = run_build([agg], out, "s6-test-exists")
            self.assertEqual(rc, 3)

    def test_final_mode_staging_never_creates_out_dir_early(self):
        """Regression: final mode used to mkdir(out_dir) at entry, so the
        'target exists' gate always fired; now failure leaves no out_dir."""
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            agg = make_aggregate_dir(tmp, "arena", [make_pgn()] * 2)
            lic = make_local_dir(tmp, "lichess", make_pgn())
            out = tmp / "out"
            rc = run_build([agg, lic], out, "s6-eval-v1-core-300k",
                           final_mode=True, sampling_version=2)
            self.assertEqual(rc, 5)  # pool shortfall on tiny data
            self.assertFalse((out / "s6-eval-v1-core-300k").exists(),
                             "final failure must not leave a target dir")

    def test_final_mode_success_path_publishes_via_staged_allow_unlabeled(self):
        """Reach the FINAL success path: downsample to exact targets, verify
        the STAGED dataset with allow_unlabeled=True, then atomically publish.

        The FINAL scale constants are patched to clean multiples so a small
        2-family corpus can satisfy the exact per-split/phase targets; the
        staged verify call is captured to prove allow_unlabeled=True."""
        import random as _rng
        import chess as _chess

        def make_long_pgn(seed: int) -> _chess.pgn.Game:
            rng = _rng.Random(seed)
            board = _chess.Board()
            moves = []
            for _ in range(60):
                legal = [m for m in board.legal_moves
                         if not board.is_game_over(claim_draw=False)]
                cands = []
                for m in legal:
                    board.push(m)
                    if not board.is_repetition(3) \
                            and not board.is_game_over(claim_draw=False):
                        cands.append(m)
                    board.pop()
                if not cands:
                    break
                move = rng.choice(cands)
                board.push(move)
                moves.append(move)
            game = _chess.pgn.Game()
            game.headers["Result"] = "1-0"
            game.headers["White"] = "A"
            game.headers["Black"] = "B"
            game.headers["WhiteElo"] = "2500"
            game.headers["BlackElo"] = "2500"
            node = game
            for m in moves:
                node = node.add_main_variation(m)
            return game

        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            # Fixed seed ranges chosen so the position_id-ordered final
            # downsample yields a deterministic, gate-passing family mix
            # (largest share 60% here).
            arena = [make_long_pgn(i) for i in range(100)]
            lichess = [make_long_pgn(5000 + i) for i in range(100)]
            agg = make_aggregate_dir(tmp, "arena", arena)
            lic = make_local_dir(tmp, "lichess", lichess)
            out = tmp / "out"
            verify_calls = []
            orig = bd.TARGET, dict(bd.FINAL_SPLIT_TARGETS), \
                dict(bd.FINAL_PHASE_TARGETS), bd.MIN_FAMILIES
            try:
                bd.TARGET = 40
                bd.FINAL_SPLIT_TARGETS = {"train": 32, "validation": 4,
                                          "holdout": 4}
                bd.FINAL_PHASE_TARGETS = {"high": 20, "mid": 20,
                                          "low": 0, "zero": 0}
                with mock.patch.object(bd, "_verify_staged") as vs:
                    vs.return_value = 0
                    rc = run_build([agg, lic], out, "s6-eval-v1-core-300k",
                                   final_mode=True, sampling_version=2)
                    self.assertEqual(rc, 0)
                    verify_calls = [c.args[0] for c in vs.call_args_list]
            finally:
                (bd.TARGET, bd.FINAL_SPLIT_TARGETS, bd.FINAL_PHASE_TARGETS,
                 bd.MIN_FAMILIES) = orig
            dataset = out / "s6-eval-v1-core-300k"
            self.assertTrue(dataset.is_dir())
            self.assertFalse((out / ".staging-s6-eval-v1-core-300k").exists())
            manifest = json.loads(
                (dataset / "dataset_manifest.json").read_text())
            self.assertTrue(manifest["final"])
            self.assertEqual(manifest["records_total"], 40)
            # The FINAL success path must call _verify_staged exactly once on
            # the staging directory (that helper hard-codes
            # allow_unlabeled=True; it is the only label-less verify path).
            self.assertEqual(len(verify_calls), 1)
            self.assertEqual(str(verify_calls[0]),
                             str(out / "s6-eval-v1-core-300k.staging"))

    def test_verify_staged_uses_allow_unlabeled(self):
        """_verify_staged() verifies an unlabeled staged dataset successfully
        (labels/teacher checks skipped), proving the FINAL pre-label staging
        verify is safe."""
        import verify_dataset as vd_mod
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            agg = make_aggregate_dir(tmp, "arena", [make_pgn(1), make_pgn(2)])
            out = tmp / "out"
            rc = run_build([agg], out, "unlabeled-staged",
                           sampling_version=2)
            self.assertEqual(rc, 0)
            staged = out / "unlabeled-staged"
            # Direct: _verify_staged must succeed on an unlabeled dataset.
            self.assertEqual(bd._verify_staged(staged), 0)
            # And the same dataset WITHOUT allow_unlabeled must fail.
            self.assertNotEqual(
                vd_mod.verify(argparse.Namespace(dataset=str(staged))), 0)


if __name__ == "__main__":
    unittest.main()
