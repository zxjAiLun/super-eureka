#!/usr/bin/env python3
"""lichess_select tests: stream SHA mismatch fails closed without publish;
insufficient selection fails closed without publish (both via --local);
S6-N3D game-fingerprint identity, --exclude-pgn skipping, duplicate-candidate
rejection, full-drain SHA, and per-game exporter isolation."""

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


def accepted_indices(seed: int, count: int, limit: int = 5000) -> list[int]:
    """Game indices whose Site URL passes the frozen accept-byte rule.

    Computed from the URL alone (no Game objects), mirroring main()'s rule.
    """
    found: list[int] = []
    for i in range(limit):
        url = f"https://lichess.org/{i:06d}"
        digest = hashlib.sha256(f"{url}:{seed}".encode("utf-8")).digest()
        if digest[0] < ls.ACCEPT_BYTE:
            found.append(i)
            if len(found) == count:
                return found
    raise AssertionError(f"only found {len(found)} accepted indices")


def archive_from_games(games: list[chess.pgn.Game]) -> bytes:
    parts = []
    for game in games:
        exporter = chess.pgn.StringExporter(columns=None)
        parts.append(game.accept(exporter) + "\n\n")
    return zstandard.ZstdCompressor().compress("".join(parts).encode("utf-8"))


class FingerprintTests(unittest.TestCase):
    def test_same_movetext_fingerprints_identically(self):
        """Identity is movetext + result + initial FEN, nothing else."""
        left = make_game(7)
        right = make_game(7)
        right.headers["Site"] = "https://lichess.org/DIFFERENT"
        right.headers["White"] = "someone-else"
        right.headers["WhiteElo"] = "2999"
        right.headers["UTCTime"] = "23:59:59"
        self.assertEqual(ls.game_fingerprint(left), ls.game_fingerprint(right))

    def test_fingerprint_survives_pgn_export_and_reparse(self):
        game = make_game(11)
        exporter = chess.pgn.StringExporter(columns=None)
        reparsed = chess.pgn.read_game(io.StringIO(game.accept(exporter)))
        self.assertEqual(ls.game_fingerprint(game),
                         ls.game_fingerprint(reparsed))

    def test_fingerprint_changes_with_result_moves_and_initial_fen(self):
        base = make_game(7)
        baseline = ls.game_fingerprint(base)

        other_result = make_game(7)
        other_result.headers["Result"] = "0-1"
        self.assertNotEqual(baseline, ls.game_fingerprint(other_result))

        other_moves = make_game(8)
        self.assertNotEqual(baseline, ls.game_fingerprint(other_moves))

        with_fen = make_game(7)
        with_fen.headers["FEN"] = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        self.assertNotEqual(baseline, ls.game_fingerprint(with_fen))

    def test_missing_fen_header_uses_standard_initial_position(self):
        game = make_game(7)
        self.assertNotIn("FEN", game.headers)
        explicit = make_game(7)
        explicit.headers["FEN"] = chess.STARTING_FEN
        self.assertEqual(ls.game_fingerprint(game),
                         ls.game_fingerprint(explicit))

    def test_fingerprint_set_sha_is_order_independent(self):
        left = ls.fingerprint_set_sha256({"b" * 64, "a" * 64})
        right = ls.fingerprint_set_sha256({"a" * 64, "b" * 64})
        self.assertEqual(left, right)
        self.assertEqual(ls.fingerprint_set_sha256(set()),
                         hashlib.sha256(b"").hexdigest())

    def test_write_game_does_not_accumulate_across_games(self):
        """A reused StringExporter would repeat the first game in the second."""
        buffer = io.StringIO()
        ls.write_game(make_game(1), buffer)
        ls.write_game(make_game(2), buffer)
        buffer.seek(0)
        first = chess.pgn.read_game(buffer)
        second = chess.pgn.read_game(buffer)
        third = chess.pgn.read_game(buffer)
        self.assertIsNone(third, "exactly two games must be written")
        self.assertEqual(ls.game_fingerprint(first),
                         ls.game_fingerprint(make_game(1)))
        self.assertEqual(ls.game_fingerprint(second),
                         ls.game_fingerprint(make_game(2)))


class ExcludeAndDuplicateTests(unittest.TestCase):
    def _select(self, tmp: Path, archive: bytes, games_per_month: int,
                out_name: str, exclude: list[Path] | None = None,
                source_id: str = "lichess-standard-rated-confirm-v1"):
        official = hashlib.sha256(archive).hexdigest()
        sha_text = f"{official}  lichess_db_standard_rated_2026-01.pgn.zst\n"
        archive_path = tmp / f"{out_name}.zst"
        archive_path.write_bytes(archive)
        out = tmp / out_name
        argv = ["lichess_select.py", "--months", "2026-01",
                "--games-per-month", str(games_per_month),
                "--seed", "20260812", "--out", str(out),
                "--source-id", source_id,
                "--source-family", "lichess-standard-rated-v1",
                "--local", str(archive_path)]
        for path in exclude or []:
            argv += ["--exclude-pgn", str(path)]
        with mock.patch.object(ls.urllib.request, "urlopen",
                               side_effect=fake_urlopen_side_effect(
                                   sha_text, archive)), \
             mock.patch.object(sys, "argv", argv):
            rc = ls.main()
        return rc, out, source_id

    def test_excluded_game_is_never_selected(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-") as tmp:
            tmp = Path(tmp)
            indices = accepted_indices(20260812, 2)
            games = [make_game(i) for i in indices]
            archive = archive_from_games(games)

            rc, out, source_id = self._select(tmp, archive, 1, "first")
            self.assertEqual(rc, 0)
            first_pgn = out / f"{source_id}.pgn"
            first_manifest = json.loads(
                (out / "source-manifest.json").read_text())
            self.assertEqual(first_manifest["exclude_fingerprint_count"], 0)
            with open(first_pgn, encoding="utf-8") as fh:
                first_fingerprint = ls.game_fingerprint(
                    chess.pgn.read_game(fh))

            rc, out2, _ = self._select(tmp, archive, 1, "second",
                                       exclude=[first_pgn])
            self.assertEqual(rc, 0)
            manifest = json.loads((out2 / "source-manifest.json").read_text())
            self.assertEqual(manifest["exclude_fingerprint_count"], 1)
            self.assertEqual(manifest["fingerprint_intersection"], 0)
            self.assertGreaterEqual(manifest["excluded_candidates_skipped"], 1)
            with open(out2 / f"{source_id}.pgn", encoding="utf-8") as fh:
                second_fingerprint = ls.game_fingerprint(
                    chess.pgn.read_game(fh))
            self.assertNotEqual(first_fingerprint, second_fingerprint)
            self.assertEqual(
                manifest["exclude_fingerprints_sha256"],
                ls.fingerprint_set_sha256({first_fingerprint}))

    def test_duplicate_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-") as tmp:
            tmp = Path(tmp)
            first, second = accepted_indices(20260812, 2)
            # The same game three times, then a distinct accepted game.
            games = [make_game(first), make_game(first), make_game(first),
                     make_game(second)]
            rc, out, source_id = self._select(
                tmp, archive_from_games(games), 2, "dupes")
            self.assertEqual(rc, 0)
            manifest = json.loads((out / "source-manifest.json").read_text())
            self.assertEqual(manifest["games_selected"], 2)
            self.assertEqual(manifest["selected_fingerprint_count"], 2)
            self.assertEqual(manifest["duplicate_candidates_rejected"], 2)
            fingerprints = []
            with open(out / f"{source_id}.pgn", encoding="utf-8") as fh:
                while (game := chess.pgn.read_game(fh)) is not None:
                    fingerprints.append(ls.game_fingerprint(game))
            self.assertEqual(len(fingerprints), 2)
            self.assertEqual(len(set(fingerprints)), 2)

    def test_source_id_names_the_output_pgn_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-") as tmp:
            tmp = Path(tmp)
            games = [make_game(i) for i in accepted_indices(20260812, 1)]
            rc, out, source_id = self._select(
                tmp, archive_from_games(games), 1, "named")
            self.assertEqual(rc, 0)
            self.assertTrue((out / f"{source_id}.pgn").is_file())
            manifest = json.loads((out / "source-manifest.json").read_text())
            self.assertEqual(manifest["source_id"], source_id)
            self.assertEqual(manifest["source_family"],
                             "lichess-standard-rated-v1")
            self.assertEqual(manifest["fingerprint"]["fields"],
                             list(ls.FINGERPRINT_FIELDS))

    def test_missing_exclude_pgn_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-") as tmp:
            tmp = Path(tmp)
            with self.assertRaises(SystemExit) as cm:
                ls.load_exclude_fingerprints([tmp / "nope.pgn"])
            self.assertIn("--exclude-pgn missing", str(cm.exception))

    def test_empty_exclude_pgn_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-") as tmp:
            empty = Path(tmp) / "empty.pgn"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit) as cm:
                ls.load_exclude_fingerprints([empty])
            self.assertIn("has no games", str(cm.exception))

    def test_exclude_fingerprints_are_read_from_all_games(self):
        with tempfile.TemporaryDirectory(prefix="s6-n3d-") as tmp:
            path = Path(tmp) / "used.pgn"
            with open(path, "w", encoding="utf-8") as fh:
                ls.write_game(make_game(3), fh)
                ls.write_game(make_game(4), fh)
            keys, sources = ls.load_exclude_fingerprints([path])
            self.assertEqual(keys, {ls.game_fingerprint(make_game(3)),
                                    ls.game_fingerprint(make_game(4))})
            self.assertEqual(sources[0]["games"], 2)
            self.assertEqual(
                sources[0]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
