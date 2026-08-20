#!/usr/bin/env python3
"""lichess_select tests: stream SHA mismatch fails closed without publish;
insufficient selection fails closed without publish (both via --local)."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chess
import chess.pgn
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lichess_select as ls  # noqa: E402


def make_game(seed: int = 0) -> chess.pgn.Game:
    """Deterministic ~50-ply game avoiding 3-fold repetition and mates."""
    import random
    rng = random.Random(seed)
    board = chess.Board()
    moves: list[chess.Move] = []
    for _ in range(50):
        candidates = []
        for m in board.legal_moves:
            board.push(m)
            if not board.is_repetition(3) \
                    and not board.is_game_over(claim_draw=False):
                candidates.append(m)
            board.pop()
        if not candidates:
            break
        move = rng.choice(candidates)
        board.push(move)
        moves.append(move)
    assert len(moves) >= 40, f"seed {seed} produced {len(moves)} plies"
    game = chess.pgn.Game()
    game.headers["Event"] = "Rated Standard game"
    game.headers["Site"] = f"https://lichess.org/{seed:06d}"
    game.headers["Result"] = "1-0"
    game.headers["White"] = "W"
    game.headers["Black"] = "B"
    game.headers["WhiteElo"] = "2500"
    game.headers["BlackElo"] = "2500"
    game.headers["TimeControl"] = "600+5"
    node = game
    for m in moves:
        node = node.add_main_variation(m)
    return game


def make_archive_pgn(n_games: int, seed_base: int = 0) -> bytes:
    """Deterministic zstd archive of `n_games` distinct games.

    MEMORY-BOUNDED: each game is exported and released immediately; only the
    accumulated PGN text (~2 KB/game) and the compressed archive are kept, so
    multi-MB test archives never materialize thousands of Game objects."""
    parts = []
    for i in range(n_games):
        # FRESH exporter per game (StringExporter accumulates across calls).
        exporter = chess.pgn.StringExporter(headers=True, variations=False,
                                            comments=False)
        game = make_game(seed_base + i)
        parts.append(game.accept(exporter) + "\n\n")
    return zstandard.ZstdCompressor().compress("".join(parts).encode("utf-8"))


def fake_urlopen_side_effect(sha256sums_text: str, archive_bytes: bytes):
    def side_effect(url, *args, **kwargs):
        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        if "sha256sums.txt" in url:
            return FakeResp(sha256sums_text.encode())
        return FakeResp(archive_bytes)
    return side_effect


class SmallChunkFakeResp(io.BytesIO):
    """Fake response that serves at most MAX_CHUNK bytes per read() call, so
    the decompressor can only ever consume part of the archive before
    selection stops; the final drain must hash the rest."""

    MAX_CHUNK = 64 * 1024
    instances: list["SmallChunkFakeResp"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.served = 0
        type(self).instances.append(self)

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size is None or size < 0:
            size = self.MAX_CHUNK
        size = min(size, self.MAX_CHUNK)
        data = super().read(size)
        if data:
            self.served += len(data)
        return data


def small_chunk_side_effect(sha256sums_text: str, archive_bytes: bytes):
    SmallChunkFakeResp.instances = []

    def side_effect(url, *args, **kwargs):
        if "sha256sums.txt" in url:
            return SmallChunkFakeResp(sha256sums_text.encode())
        return SmallChunkFakeResp(archive_bytes)
    return side_effect


class LichessSelectTests(unittest.TestCase):
    def _run(self, tmp: Path, archive: bytes, official_sha: str,
             games_per_month: int):
        sha_text = f"{official_sha}  lichess_db_standard_rated_2026-01.pgn.zst\n"
        archive_path = tmp / "month.zst"
        archive_path.write_bytes(archive)
        out = tmp / "out"
        argv = ["lichess_select.py", "--months", "2026-01",
                "--games-per-month", str(games_per_month),
                "--seed", "20260812", "--out", str(out),
                "--local", str(archive_path)]
        with mock.patch.object(ls.urllib.request, "urlopen",
                               side_effect=fake_urlopen_side_effect(
                                   sha_text, archive)) as _m, \
             mock.patch.object(sys, "argv", argv):
            return ls.main(), out

    def test_checksum_mismatch_fails_closed_no_publish(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            archive = make_archive_pgn(1, seed_base=1)
            rc, out = self._run(Path(tmp), archive, "f" * 64, 2000)
            self.assertEqual(rc, 5)
            self.assertFalse(out.exists(), "mismatch must not publish")

    def test_insufficient_selection_no_publish(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            archive = make_archive_pgn(1, seed_base=1)
            actual = hashlib.sha256(archive).hexdigest()
            rc, out = self._run(Path(tmp), archive, actual, 2000)
            self.assertEqual(rc, 4)
            self.assertFalse(out.exists(), "short selection must not publish")
            self.assertFalse(
                (Path(str(out) + ".staging") / "source-manifest.json").is_file()
                if False else (out.with_name(out.name + ".staging")
                               / "source-manifest.json").exists(),
                "staging manifest must not be written on failure")

    def test_success_publishes_with_manifest(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            archive = make_archive_pgn(200)
            actual = hashlib.sha256(archive).hexdigest()
            rc, out = self._run(tmp, archive, actual, 1)
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            manifest = json.loads(
                (out / "source-manifest.json").read_text())
            self.assertEqual(manifest["source_family"],
                             "lichess-standard-rated-v1")
            self.assertEqual(manifest["official_sha256"]["2026-01"], actual)
            self.assertTrue(manifest["script_sha256"])
            self.assertGreaterEqual(manifest["games_selected"], 1)

    def test_doc_filters_match_passes(self):
        # passes() must NOT filter on Event (TimeControl is the speed gate).
        game = make_game(1)
        game.headers["Event"] = "Rated Rapid tournament something"
        self.assertTrue(ls.passes(game))
        game.headers["TimeControl"] = "30+0"
        self.assertFalse(ls.passes(game))

    def test_open_month_stream_drain_completes_hash(self):
        """Direct drain contract: after one game is read, the compressed
        archive is largely UNREAD and the hash incomplete; draining through
        the HashingReader completes the official SHA.

        Memory-light: a ~40 KiB archive with a 4 KiB per-read cap keeps the
        whole test at a few MB. The old implementation did not return the
        HashingReader (it returned the bare hasher), so this drain path
        cannot even be expressed with the old API: the test fails
        structurally against it."""
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            SmallChunkFakeResp.MAX_CHUNK = 4096
            # Large enough to exceed the decompressor's read-ahead window so
            # the majority of the archive stays unread after one game.
            archive = make_archive_pgn(4000)
            official = hashlib.sha256(archive).hexdigest()
            sha_text = f"{official}  lichess_db_standard_rated_2026-01.pgn.zst\n"
            with mock.patch.object(
                    ls.urllib.request, "urlopen",
                    side_effect=small_chunk_side_effect(sha_text, archive)):
                reader, hashing_reader, raw, official2, display = \
                    ls.open_month_stream("2026-01", None)
                self.assertEqual(official2, official)
                text = io.TextIOWrapper(reader, encoding="utf-8",
                                        errors="replace")
                game = ls.chess.pgn.read_game(text)
                self.assertIsNotNone(game)
                archive_resp = SmallChunkFakeResp.instances[-1]
                served_before_drain = archive_resp.served
                self.assertGreater(served_before_drain, 0)
                self.assertLess(served_before_drain, len(archive),
                                "selection read must leave archive bytes "
                                "unread")
                self.assertNotEqual(hashing_reader.hexdigest(), official,
                                    "hash must be incomplete before drain")
                while hashing_reader.read(1 << 20):
                    pass
                self.assertEqual(hashing_reader.hexdigest(), official)
                text.close()
                reader.close()
                raw.close()

    def test_small_chunk_stream_end_to_end_sha_passes(self):
        """Full main() flow over a small-chunk fake source: the recorded
        official SHA equals the complete archive SHA."""
        with tempfile.TemporaryDirectory(prefix="s6-n3a-") as tmp:
            tmp = Path(tmp)
            # Large enough to exceed the decompressor's read-ahead window so
            # the majority of the archive stays unread after one game.
            archive = make_archive_pgn(4000)
            official = hashlib.sha256(archive).hexdigest()
            sha_text = f"{official}  lichess_db_standard_rated_2026-01.pgn.zst\n"
            out = tmp / "out"
            argv = ["lichess_select.py", "--months", "2026-01",
                    "--games-per-month", "1", "--seed", "20260812",
                    "--out", str(out)]
            with mock.patch.object(
                    ls.urllib.request, "urlopen",
                    side_effect=small_chunk_side_effect(sha_text, archive)) as _m, \
                 mock.patch.object(sys, "argv", argv):
                rc = ls.main()
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            manifest = json.loads(
                (out / "source-manifest.json").read_text())
            self.assertEqual(manifest["official_sha256"]["2026-01"], official)

    def test_hashing_reader_read_zero_is_probe_semantics(self):
        reader = ls.HashingReader(io.BytesIO(b"abc"), hashlib.sha256())
        self.assertEqual(reader.read(0), b"")
        self.assertEqual(reader.hexdigest(),
                         hashlib.sha256().hexdigest())


if __name__ == "__main__":
    unittest.main()
