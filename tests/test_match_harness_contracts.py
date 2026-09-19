"""Fast harness tripwires; no models, builds or games required."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import chess

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S15 = load("s15_screen", "tools/s15/s15_screen.py")
S17 = load("s17_match", "tools/s17/s17_ab_match.py")


class EvaluatorContract(unittest.TestCase):
    def test_explicit_nnue_in_both_game_commands(self):
        cmd = S15.build_command(Path("book.epd"), Path("match.pgn"), 4)
        self.assertEqual(cmd.count("arg=--evaluation"), 2)
        self.assertEqual(cmd.count("arg=nnue"), 2)
        self.assertEqual(cmd[cmd.index("-rounds") + 1], "256")
        self.assertEqual(cmd[cmd.index("-repeat") + 1], "2")

    def test_real_handshake_readback_contract(self):
        good = ("info string profile current-final\ninfo string eval nnue-v2q\n"
                "info string network nnue-v2q\ninfo string evalfile model.bin\nuciok\nreadyok\n")
        variants = [(good, 0, True),
                    (good.replace("eval nnue-v2q", "eval handcrafted"), 0, False),
                    (good.replace("network nnue-v2q", "network none"), 0, False),
                    (good.replace("model.bin", "wrong.bin"), 0, False),
                    ("", 0, False), (good, 2, False)]
        for module, verify in [(S15, S15.verify_evaluator), (S17, S17.verify_arm)]:
            for stdout, rc, passes in variants:
                with self.subTest(module=module.__name__, rc=rc, stdout=stdout):
                    result = subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")
                    with patch.object(module.subprocess, "run", return_value=result) as run:
                        if passes:
                            self.assertEqual(verify(Path("engine.exe"), Path("model.bin"), "test")["eval"], "nnue-v2q")
                        else:
                            with self.assertRaises(SystemExit):
                                verify(Path("engine.exe"), Path("model.bin"), "test")
                        command = run.call_args.args[0]
                        self.assertEqual(command[command.index("--evaluation") + 1], "nnue")
                        self.assertIn("--nnue-model", command)


class RunnerCorrectnessRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

    def test_s15_pairing_by_fen_survives_shuffle(self):
        """P1-1: cutechess concurrency can write games out of launch order.
        FEN-based pair grouping must compute the exact pentanomial regardless of PGN ordering,
        whereas naive consecutive pair grouping produces an incorrect pentanomial."""
        fen1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        fen2 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"

        def make_game(round_id, white, black, result, fen):
            return (f'[Event "test"]\n[Round "{round_id}"]\n[White "{white}"]\n'
                    f'[Black "{black}"]\n[Result "{result}"]\n[SetUp "1"]\n[FEN "{fen}"]\n\n'
                    f'1... e5 2. Nf3 {result}\n\n')

        # Opening 1: Cand wins both (WW, pair score 2.0 -> penta index 4)
        g1a = make_game(1, "Cand", "Base", "1-0", fen1)
        g1b = make_game(2, "Base", "Cand", "0-1", fen1)
        # Opening 2: Cand loses both (LL, pair score 0.0 -> penta index 0)
        g2a = make_game(3, "Cand", "Base", "0-1", fen2)
        g2b = make_game(4, "Base", "Cand", "1-0", fen2)

        # Interleave games in PGN: [g1a (1.0), g2a (0.0), g1b (1.0), g2b (0.0)]
        # Naive consecutive pairing would see (g1a, g2a) -> 1.0 (idx 2) and (g1b, g2b) -> 1.0 (idx 2) => [0, 0, 2, 0, 0]
        # True FEN-grouped pairing sees fen1 -> 2.0 (idx 4) and fen2 -> 0.0 (idx 0) => [1, 0, 0, 0, 1]
        shuffled_pgn = self.tmp_path / "shuffled.pgn"
        shuffled_pgn.write_text(g1a + g2a + g1b + g2b, encoding="utf-8")

        stats = S15.parse_pgn_and_stats(shuffled_pgn, "Cand", "Base", expected_fens=[fen1, fen2])
        self.assertEqual(stats["pentanomial"]["counts"], [1, 0, 0, 0, 1])

        # Paired Elo CI also reflects the true pair scores [1.0, 0.0] (non-zero variance)
        elo, ci = S15.paired_elo_ci(stats["pair_evidence"], 2.0, 4)
        self.assertEqual(elo, 0.0)
        self.assertIsNotNone(ci)
        self.assertGreater(ci, 0.0)

    def test_incomplete_match_fails_closed(self):
        """P1-2: Incomplete matches (e.g. 137/256 games across 128 distinct openings)
        must fail closed due to game count / opening set mismatch, never generating a report."""
        openings = S17.read_openings()
        self.assertEqual(len(openings), 128)
        normalized = [" ".join(f.split()[:4]) + " 0 1" for f in openings]
        self.assertEqual(len(set(normalized)), 128)

        # Helper to generate 137 games for specified arm names across 128 openings
        def make_137_pgn(arm_a, arm_b, target_path):
            pgn_games = []
            for i in range(137):
                op_idx = i // 2
                fen = normalized[op_idx]
                is_white = (i % 2 == 0)
                white = arm_a if is_white else arm_b
                black = arm_b if is_white else arm_a
                b = chess.Board(fen)
                m = list(b.legal_moves)[0].uci()
                pgn_games.append(f'[Event "test"]\n[Round "{i+1}"]\n[White "{white}"]\n'
                                 f'[Black "{black}"]\n[Result "1/2-1/2"]\n[SetUp "1"]\n'
                                 f'[FEN "{fen}"]\n\n1. {m} 1/2-1/2\n\n')
            target_path.write_text("".join(pgn_games), encoding="utf-8")

        # Test S15 runner
        pgn_s15 = self.tmp_path / "incomplete_137_s15.pgn"
        make_137_pgn("Cand", "Base", pgn_s15)
        with self.assertRaises(ValueError) as ctx:
            S15.parse_pgn_and_stats(pgn_s15, "Cand", "Base", expected_fens=openings)
        self.assertIn("opening set or game count does not match protocol", str(ctx.exception))

        # Test S17 runner
        pgn_s17 = self.tmp_path / "incomplete_137_s17.pgn"
        make_137_pgn("A-avx2", "B-scalar", pgn_s17)
        with self.assertRaises(ValueError) as ctx:
            S17.parse_pgn_and_stats(pgn_s17, "A-avx2", "B-scalar", expected_fens=openings)
        self.assertIn("opening set or game count does not match protocol", str(ctx.exception))

        # Even without expected_fens, opening 68 has only 1 game -> pair incomplete
        with self.assertRaises(ValueError) as ctx:
            S15.parse_pgn_and_stats(pgn_s15, "Cand", "Base", expected_fens=None)
        self.assertIn("opening is not a complete color-swapped pair", str(ctx.exception))

    def test_s17_binary_source_not_bound_to_repo_head(self):
        """P2 & P1: S17 report commit must come from binary UCI handshake provenance,
        NOT from git rev-parse HEAD at script runtime. Mismatched arms and partial
        provenance conflicts must fail closed."""
        # Case A: Binary handshake captures source SHA
        handshake_output = (
            "info string build eureka-v0.2.0-custom\n"
            "info string source 315534991628ce084c34392d6abb98d71f3f591a\n"
            "info string profile current-final\n"
            "info string eval nnue-v2q\n"
            "info string network nnue-v2q\n"
            "info string evalfile model.bin\n"
            "uciok\nreadyok\n"
        )
        with patch.object(S17.subprocess, "run",
                          return_value=subprocess.CompletedProcess([], 0, stdout=handshake_output, stderr="")):
            hs = S17.handshake(Path("engine.exe"), Path("model.bin"))
            self.assertEqual(hs.get("source"), "315534991628ce084c34392d6abb98d71f3f591a")

        # Case B: Matching binary sources resolve without calling git rev-parse
        hs_a = {"source": "315534991628ce084c34392d6abb98d71f3f591a"}
        hs_b = {"source": "315534991628ce084c34392d6abb98d71f3f591a"}
        commit, prov = S17.resolve_source_commit(hs_a, hs_b)
        self.assertEqual(commit, "315534991628ce084c34392d6abb98d71f3f591a")
        self.assertEqual(prov, "binary_uci_handshake")

        # Case C: Mismatched binary sources fail closed
        hs_bad = {"source": "mismatched_commit"}
        with self.assertRaises(SystemExit) as ctx:
            S17.resolve_source_commit(hs_a, hs_bad)
        self.assertIn("mismatch", str(ctx.exception))

        # Case D (P1 fix): Partial provenance - one binary reports source, other does not
        hs_one = {"source": "known_commit_x"}
        # Subcase D1: Caller supplies conflicting commit Y -> MUST FAIL CLOSED
        with self.assertRaises(SystemExit) as ctx:
            S17.resolve_source_commit(hs_one, {}, expected_commit="conflicting_commit_y")
        self.assertIn("does not match", str(ctx.exception))

        # Subcase D2: Caller supplies no commit -> MUST FAIL CLOSED
        with self.assertRaises(SystemExit) as ctx:
            S17.resolve_source_commit(hs_one, {})
        self.assertIn("only one binary reported source", str(ctx.exception))

        # Subcase D3: Caller supplies matching commit X -> verified
        commit, prov = S17.resolve_source_commit(hs_one, {}, expected_commit="known_commit_x")
        self.assertEqual(commit, "known_commit_x")
        self.assertEqual(prov, "caller_supplied_verified_by_one_binary")

        # Case E: Missing binary sources require --source-commit
        with self.assertRaises(SystemExit) as ctx:
            S17.resolve_source_commit({}, {})
        self.assertIn("--source-commit is required", str(ctx.exception))

        commit, prov = S17.resolve_source_commit({}, {}, expected_commit="manual_sha")
        self.assertEqual(commit, "manual_sha")
        self.assertEqual(prov, "caller_supplied")


if __name__ == "__main__":
    unittest.main()
