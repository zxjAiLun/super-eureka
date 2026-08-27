import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.pgn
import zstandard as zstd

from tools.s10.extract_broadcast import extract_broadcasts
from tools.s10.source_identity import game_fingerprint
from tools.s6.build_dataset import load_source_catalog


def create_synthetic_zst_archive(games: list[chess.pgn.Game]) -> tuple[bytes, str]:
    """Create in-memory compressed .pgn.zst archive and return (bytes, sha256)."""
    text_io = io.StringIO()
    for g in games:
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        text_io.write(g.accept(exporter) + "\n\n")
    raw_bytes = text_io.getvalue().encode("utf-8")

    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(raw_bytes)
    sha = hashlib.sha256(compressed).hexdigest()
    return compressed, sha


def make_distinct_game(ply_count: int, seed_id: int, result: str = "1-0") -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = f"Event_{seed_id}"
    game.headers["Site"] = f"https://lichess.org/test/{seed_id}"
    game.headers["White"] = f"PlayerW_{seed_id}"
    game.headers["Black"] = f"PlayerB_{seed_id}"
    game.headers["Result"] = result

    node = game
    board = game.board()
    # Play deterministic, legal, unique move paths
    for ply in range(ply_count):
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal:
            break
        idx = (seed_id * 17 + ply * 31) % len(legal)
        m = legal[idx]
        node = node.add_variation(m)
        board.push(m)
    return game


class TestExtractBroadcast(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test-broadcast-"))

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def _assert_no_staging_dirs(self):
        staging_dirs = list(self.tmp_dir.glob(".staging-*"))
        self.assertEqual(staging_dirs, [], f"Leftover staging directories found: {staging_dirs}")

    def test_checksum_pass_and_mismatch_fail(self):
        # 10 short games (45 plies) + 6 long games (85 plies)
        games = [make_distinct_game(45, i) for i in range(10)] + [
            make_distinct_game(85, i + 50) for i in range(6)
        ]
        compressed, actual_sha = create_synthetic_zst_archive(games)
        archive_file = self.tmp_dir / "lichess_db_broadcast_2026-07.pgn.zst"
        archive_file.write_bytes(compressed)

        out_pass = self.tmp_dir / "broadcast_pass"
        # Test Checksum PASS: request 6 games (2 long, 4 short)
        manifest = extract_broadcasts(
            months=["2026-07"],
            games_per_month=6,
            seed=20260827,
            source_id="test-pass",
            source_family="lichess-broadcast",
            out_dir=out_pass,
            local_archives={"2026-07": archive_file},
            mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": actual_sha},
        )
        self.assertEqual(manifest["games_total"], 6)
        self.assertTrue((out_pass / "source-manifest.json").exists())
        self._assert_no_staging_dirs()

        # Test Checksum Mismatch FAIL
        out_fail = self.tmp_dir / "broadcast_fail"
        with self.assertRaises(SystemExit) as cm:
            extract_broadcasts(
                months=["2026-07"],
                games_per_month=6,
                seed=20260827,
                source_id="test-fail",
                source_family="lichess-broadcast",
                out_dir=out_fail,
                local_archives={"2026-07": archive_file},
                mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": "0" * 64},
            )
        self.assertIn("FAIL CLOSED: archive SHA-256 mismatch", str(cm.exception))
        self.assertFalse(out_fail.exists())
        self._assert_no_staging_dirs()

    def test_existing_target_refuses_overwrite(self):
        out_dir = self.tmp_dir / "existing_source"
        out_dir.mkdir()
        with self.assertRaises(SystemExit) as cm:
            extract_broadcasts(
                months=["2026-07"],
                games_per_month=2,
                seed=20260827,
                source_id="test-existing",
                source_family="lichess-broadcast",
                out_dir=out_dir,
                local_archives={},
                mock_checksums={},
            )
        self.assertIn("already exists", str(cm.exception))
        self._assert_no_staging_dirs()

    def test_deterministic_top_k_and_archive_order_independence(self):
        games = [make_distinct_game(45, i) for i in range(15)] + [
            make_distinct_game(85, i + 100) for i in range(10)
        ]
        compressed_1, sha_1 = create_synthetic_zst_archive(games)
        compressed_2, sha_2 = create_synthetic_zst_archive(list(reversed(games)))

        arc_1 = self.tmp_dir / "arc1.pgn.zst"
        arc_2 = self.tmp_dir / "arc2.pgn.zst"
        arc_1.write_bytes(compressed_1)
        arc_2.write_bytes(compressed_2)

        out_1 = self.tmp_dir / "out1"
        out_2 = self.tmp_dir / "out2"

        m1 = extract_broadcasts(
            months=["2026-07"],
            games_per_month=6,
            seed=20260827,
            source_id="test-order-1",
            source_family="lichess-broadcast",
            out_dir=out_1,
            local_archives={"2026-07": arc_1},
            mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": sha_1},
        )
        m2 = extract_broadcasts(
            months=["2026-07"],
            games_per_month=6,
            seed=20260827,
            source_id="test-order-2",
            source_family="lichess-broadcast",
            out_dir=out_2,
            local_archives={"2026-07": arc_2},
            mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": sha_2},
        )

        # Output PGN games must be identically selected and ordered
        pgn1 = (out_1 / "test-order-1.pgn").read_text()
        pgn2 = (out_2 / "test-order-2.pgn").read_text()
        self.assertEqual(pgn1, pgn2)
        self.assertEqual(m1["selected_fingerprints_sha256"], m2["selected_fingerprints_sha256"])
        self._assert_no_staging_dirs()

    def test_same_month_duplicate_rejection(self):
        # Archive contains duplicate copies of game 0 and game 1
        g0 = make_distinct_game(45, 0)
        g1 = make_distinct_game(85, 50)
        games = [
            g0,
            g0,  # Duplicate of g0
            g1,
            g1,  # Duplicate of g1
            make_distinct_game(45, 2),
            make_distinct_game(45, 3),
            make_distinct_game(45, 4),
            make_distinct_game(85, 52),
            make_distinct_game(85, 53),
        ]
        compressed, sha = create_synthetic_zst_archive(games)
        arc = self.tmp_dir / "arc_dup.pgn.zst"
        arc.write_bytes(compressed)

        out_dir = self.tmp_dir / "out_dup"
        manifest = extract_broadcasts(
            months=["2026-07"],
            games_per_month=4,
            seed=20260827,
            source_id="test-dup",
            source_family="lichess-broadcast",
            out_dir=out_dir,
            local_archives={"2026-07": arc},
            mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": sha},
        )

        # Output PGN game count must equal 4 and match selected_fingerprints_count
        actual_games = 0
        with open(out_dir / "test-dup.pgn", "r", encoding="utf-8") as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                actual_games += 1

        self.assertEqual(actual_games, 4)
        self.assertEqual(manifest["games_total"], 4)
        self.assertEqual(manifest["selected_fingerprints_count"], 4)
        self.assertEqual(manifest["duplicate_candidates_rejected_total"], 2)
        self._assert_no_staging_dirs()

    def test_duplicate_and_exclude_rejection(self):
        game_to_exclude = make_distinct_game(45, 999)
        excluded_pgn = self.tmp_dir / "excluded.pgn"
        with open(excluded_pgn, "w") as f:
            exporter = chess.pgn.StringExporter(headers=True)
            f.write(game_to_exclude.accept(exporter) + "\n\n")

        games = [game_to_exclude] + [make_distinct_game(45, i) for i in range(10)] + [
            make_distinct_game(85, i + 50) for i in range(5)
        ]
        compressed, sha = create_synthetic_zst_archive(games)
        arc = self.tmp_dir / "arc.pgn.zst"
        arc.write_bytes(compressed)

        out_dir = self.tmp_dir / "out_exclude"
        manifest = extract_broadcasts(
            months=["2026-07"],
            games_per_month=4,
            seed=20260827,
            source_id="test-exclude",
            source_family="lichess-broadcast",
            out_dir=out_dir,
            local_archives={"2026-07": arc},
            mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": sha},
            exclude_pgns=[excluded_pgn],
        )
        self.assertEqual(manifest["fingerprint_intersection_count"], 0)
        self.assertEqual(manifest["excluded_fingerprints_count"], 1)
        self.assertEqual(manifest["excluded_candidates_rejected_total"], 1)
        self._assert_no_staging_dirs()

    def test_missing_exclude_pgn_fails_closed(self):
        with self.assertRaises(SystemExit) as cm:
            extract_broadcasts(
                months=["2026-07"],
                games_per_month=2,
                seed=20260827,
                source_id="test-missing-exclude",
                source_family="lichess-broadcast",
                out_dir=self.tmp_dir / "out_missing",
                exclude_pgns=[self.tmp_dir / "non_existent.pgn"],
            )
        self.assertIn("exclusion PGN not found", str(cm.exception))
        self._assert_no_staging_dirs()

    def test_insufficient_quota_fails_closed(self):
        games = [make_distinct_game(45, i) for i in range(2)]
        compressed, sha = create_synthetic_zst_archive(games)
        arc = self.tmp_dir / "arc.pgn.zst"
        arc.write_bytes(compressed)

        with self.assertRaises(SystemExit) as cm:
            extract_broadcasts(
                months=["2026-07"],
                games_per_month=5,
                seed=20260827,
                source_id="test-insufficient",
                source_family="lichess-broadcast",
                out_dir=self.tmp_dir / "out_insufficient",
                local_archives={"2026-07": arc},
                mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": sha},
            )
        self.assertIn("insufficient candidates", str(cm.exception))
        self._assert_no_staging_dirs()

    def test_manifest_accepted_by_build_dataset_load_source_catalog(self):
        games = [make_distinct_game(45, i) for i in range(5)] + [
            make_distinct_game(85, i + 50) for i in range(3)
        ]
        compressed, sha = create_synthetic_zst_archive(games)
        arc = self.tmp_dir / "arc.pgn.zst"
        arc.write_bytes(compressed)

        out_dir = self.tmp_dir / "catalog_test"
        extract_broadcasts(
            months=["2026-07"],
            games_per_month=3,
            seed=20260827,
            source_id="test-catalog",
            source_family="lichess-broadcast",
            out_dir=out_dir,
            local_archives={"2026-07": arc},
            mock_checksums={"lichess_db_broadcast_2026-07.pgn.zst": sha},
        )

        catalog = load_source_catalog([out_dir])
        self.assertIn("test-catalog", catalog)
        self.assertEqual(catalog["test-catalog"]["source_family"], "lichess-broadcast")
        self._assert_no_staging_dirs()


if __name__ == "__main__":
    unittest.main()
