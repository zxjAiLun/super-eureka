import argparse
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.pgn

from tools.s10.analyze_source_pool import profile_pool
from tools.s6.build_dataset import build, load_source_catalog, sha256_bytes


def make_test_game(ply_count: int, seed_id: int, result: str = "1-0") -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = f"Event_{seed_id}"
    game.headers["Site"] = f"https://lichess.org/test/{seed_id}"
    game.headers["White"] = f"PlayerW_{seed_id}"
    game.headers["Black"] = f"PlayerB_{seed_id}"
    game.headers["Result"] = result

    node = game
    board = game.board()
    for ply in range(ply_count):
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        if not legal:
            break
        idx = (seed_id * 13 + ply * 29) % len(legal)
        m = legal[idx]
        node = node.add_variation(m)
        board.push(m)
    return game


def create_synthetic_source(
    target_dir: Path,
    source_id: str,
    source_family: str,
    game_count: int,
    start_seed: int,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = target_dir / f"{source_id}.pgn"

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    with open(pgn_path, "w", encoding="utf-8") as f:
        for i in range(game_count):
            g = make_test_game(50 + (i % 40), start_seed + i)
            f.write(g.accept(exporter) + "\n\n")

    pgn_sha = sha256_bytes(pgn_path.read_bytes())
    manifest = {
        "source_id": source_id,
        "source_family": source_family,
        "pgn_sha256": pgn_sha,
        "games_total": game_count,
    }
    (target_dir / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return target_dir


class TestAnalyzeSourcePoolParity(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test-parity-"))

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_profiler_parity_with_builder_non_final(self):
        # Create 2 distinct sources across 2 families
        src1 = create_synthetic_source(
            self.tmp_dir / "src1",
            source_id="synth-broadcast",
            source_family="lichess-broadcast",
            game_count=20,
            start_seed=100,
        )
        src2 = create_synthetic_source(
            self.tmp_dir / "src2",
            source_id="synth-standard",
            source_family="lichess-standard-rated-v1",
            game_count=20,
            start_seed=500,
        )

        # 1. Run profiler
        report = profile_pool([src1, src2], target_n=300000, final_mode=True)

        # 2. Run builder in standard mode (sampling-version 2)
        out_dataset = self.tmp_dir / "out_dataset"
        build_args = argparse.Namespace(
            sources=[str(src1), str(src2)],
            sampling_version=2,
            final_mode=False,
            enforce_family_mix=False,
            dataset_id="test-parity-ds",
            out=str(self.tmp_dir),
        )
        rc = build(build_args)
        self.assertEqual(rc, 0)

        # Read builder stats
        manifest_file = self.tmp_dir / "test-parity-ds" / "dataset_manifest.json"
        builder_stats = json.loads(manifest_file.read_text(encoding="utf-8"))

        # 3. Assert exact parity between Tier 1/2 in Profiler and Builder
        self.assertEqual(
            report["tier1_raw_post_dedup"]["duplicates_removed"],
            builder_stats["duplicates_removed"],
        )
        self.assertEqual(
            report["tier2_stratified_pre_final"]["stratified_total"],
            builder_stats["records_total"],
        )
        self.assertEqual(
            report["tier2_stratified_pre_final"]["pre_final_families"],
            builder_stats["source_families"],
        )
        self.assertAlmostEqual(
            report["tier2_stratified_pre_final"]["pre_final_largest_share"],
            builder_stats["largest_family_share"],
            places=6,
        )

    def test_profiler_detects_300k_shortfall_accurately(self):
        src1 = create_synthetic_source(
            self.tmp_dir / "src_small",
            source_id="small-broadcast",
            source_family="lichess-broadcast",
            game_count=5,
            start_seed=10,
        )
        report = profile_pool([src1], target_n=300000, final_mode=True, enforce_family_mix=True)
        self.assertFalse(report["is_feasible"])
        self.assertGreater(report["tier2_stratified_pre_final"]["shortfalls_count"], 0)
        self.assertFalse(report["tier2_stratified_pre_final"]["pre_final_family_pass"])


if __name__ == "__main__":
    unittest.main()
