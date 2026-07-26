import math
from pathlib import Path
import sys
import tempfile
import unittest

import chess

from tournament import (
    DEFAULT_OPENINGS,
    EngineProtocolError,
    EngineSession,
    EngineTimeout,
    PAIR_CATEGORIES,
    PentanomialState,
    TournamentError,
    _terminal_result,
    build_schedule,
    candidate_result_for_game,
    load_openings,
    pair_category,
    pair_score_from_results,
    parse_bestmove_line,
    play_game,
    Opening,
    ScheduledGame,
    score_statistics,
    sha256_file,
    validate_bestmove,
)


class TournamentUnitTests(unittest.TestCase):
    @staticmethod
    def fake_engine(mode: str, delay: float = 0.0) -> list[str]:
        script = f"""
import sys
import time
mode = {mode!r}
delay = {delay!r}
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name E1Fake', flush=True)
        print('id author E1Tests', flush=True)
        print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('go'):
        if mode == 'slow':
            time.sleep(delay)
        print('bestmove a2a3', flush=True)
    elif line == 'quit':
        break
"""
        return [sys.executable, "-u", "-c", script]

    @staticmethod
    def legal_move_engine(delay: float = 0.0) -> list[str]:
        script = f"""
import chess
import sys
import time
delay = {delay!r}
board = chess.Board()
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name E1LegalFake', flush=True)
        print('id author E1Tests', flush=True)
        print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('position'):
        board = chess.Board()
        fields = line.split()
        if 'moves' in fields:
            for move_uci in fields[fields.index('moves') + 1:]:
                board.push_uci(move_uci)
    elif line.startswith('go'):
        time.sleep(delay)
        print('bestmove ' + next(iter(board.legal_moves)).uci(), flush=True)
    elif line == 'quit':
        break
"""
        return [sys.executable, "-u", "-c", script]

    def test_uci_handshake_and_id_and_bestmove(self):
        with tempfile.TemporaryDirectory() as temp:
            stderr_path = Path(temp) / "fake.stderr.log"
            with EngineSession(
                self.fake_engine("normal"), "fake", 16, stderr_path
            ) as engine:
                engine.position([])
                move, elapsed_ms = engine.go_movetime(20, 20)
            self.assertEqual(move, "a2a3")
            self.assertGreaterEqual(elapsed_ms, 0.0)
            self.assertEqual(engine.uci_name, "E1Fake")
            self.assertEqual(engine.uci_author, "E1Tests")
            self.assertTrue(stderr_path.exists())

    def test_host_deadline_accepts_on_time_and_rejects_late_legal_move(self):
        with EngineSession(self.fake_engine("slow", 0.01), "fast", 16) as engine:
            engine.position([])
            self.assertEqual(engine.go_movetime(20, 40)[0], "a2a3")

        with self.assertRaises(EngineTimeout) as context:
            with EngineSession(self.fake_engine("slow", 0.08), "slow", 16) as engine:
                engine.position([])
                engine.go_movetime(20, 20)
        self.assertGreater(context.exception.elapsed_ms, 40.0)

    def test_play_game_records_timeout_loss_for_both_baseline_colours(self):
        opening = Opening("test", ())
        with tempfile.TemporaryDirectory() as temp:
            diagnostics = Path(temp)
            for engine_a_white, expected_result, expected_color in (
                (True, "0-1", "white"),
                (False, "1-0", "black"),
            ):
                record = play_game(
                    ScheduledGame(1, 1, opening, engine_a_white),
                    self.legal_move_engine(0.08),
                    self.legal_move_engine(),
                    "baseline",
                    "candidate",
                    16,
                    5,
                    5,
                    40,
                    diagnostics,
                )
                self.assertEqual(record.result, expected_result)
                self.assertEqual(record.reason, "timeout")
                self.assertEqual(record.timeout_engine, "baseline")
                self.assertEqual(record.timeout_color, expected_color)
                self.assertIsNone(record.error)

    def test_bestmove_parser_and_legality_errors(self):
        self.assertEqual(parse_bestmove_line("bestmove a2a3 ponder a7a6"), "a2a3")
        with self.assertRaises(EngineProtocolError):
            parse_bestmove_line("bestmove")
        with self.assertRaises(TournamentError):
            validate_bestmove(chess.Board(), "a1a8")

    def test_bestmove_0000_only_accepts_terminal_position(self):
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_checkmate())
        self.assertIsNone(validate_bestmove(board, "0000"))
        self.assertEqual(_terminal_result(board), ("1-0", "checkmate"))
        with self.assertRaises(EngineProtocolError):
            validate_bestmove(chess.Board(), "0000")

    def test_candidate_perspective_maps_both_colours(self):
        # A is baseline. When A is white, candidate B wins 0-1; when A is
        # black, candidate B wins 1-0.
        self.assertEqual(candidate_result_for_game("0-1", True), "win")
        self.assertEqual(candidate_result_for_game("1-0", True), "loss")
        self.assertEqual(candidate_result_for_game("1-0", False), "win")
        self.assertEqual(candidate_result_for_game("0-1", False), "loss")
        self.assertEqual(candidate_result_for_game("1/2-1/2", True), "draw")

    def test_opening_suite_has_32_legal_lines(self):
        openings = load_openings(DEFAULT_OPENINGS)
        self.assertEqual(len(openings), 32)
        self.assertEqual(len({opening.identifier for opening in openings}), 32)
        for opening in openings:
            board = chess.Board()
            for move_uci in opening.moves:
                move = chess.Move.from_uci(move_uci)
                self.assertIn(move, board.legal_moves, opening.identifier)
                board.push(move)

    def test_schedule_requires_even_games_and_pairs_colours(self):
        openings = load_openings(DEFAULT_OPENINGS)
        with self.assertRaises(TournamentError):
            build_schedule(openings, 63, seed=7)
        schedule = build_schedule(openings, 64, seed=7)
        self.assertEqual(sum(game.engine_a_white for game in schedule), 32)
        self.assertEqual(len({game.opening.identifier for game in schedule}), 32)
        for index in range(0, 64, 2):
            first, second = schedule[index : index + 2]
            self.assertEqual(first.pair, second.pair)
            self.assertEqual(first.opening, second.opening)
            self.assertTrue(first.engine_a_white)
            self.assertFalse(second.engine_a_white)

    def test_pentanomial_counts_all_categories_and_is_order_invariant(self):
        state = PentanomialState(elo1=100.0)
        for total in (0.0, 0.5, 1.0, 1.5, 2.0):
            state.update_pair(total)
        self.assertEqual(state.pairs_completed, 5)
        self.assertEqual(state.counts, {category: 1 for category in PAIR_CATEGORIES})
        self.assertEqual(pair_category(pair_score_from_results("win", "loss")), "1.0")
        self.assertEqual(pair_category(pair_score_from_results("loss", "win")), "1.0")

    def test_pentanomial_golden_sequence_llr(self):
        state = PentanomialState(elo1=5.0, draw_rate=0.5)
        for total in (0.0, 0.5, 1.0, 1.5, 2.0):
            state.update_pair(total)
        self.assertAlmostEqual(state.llr, -0.00069032272312676, places=14)
        self.assertIsNone(state.decision)

    def test_sprt_pass_and_reject_are_candidate_direction(self):
        passing = PentanomialState(elo1=400.0)
        while passing.decision is None:
            passing.update_pair(2.0)
        self.assertEqual(passing.decision, "PASS")
        self.assertGreaterEqual(passing.llr, passing.upper_bound)

        rejecting = PentanomialState(elo1=400.0)
        while rejecting.decision is None:
            rejecting.update_pair(0.0)
        self.assertEqual(rejecting.decision, "REJECTED")
        self.assertLessEqual(rejecting.llr, rejecting.lower_bound)

    def test_pair_aware_ci_is_finite_ordered_and_narrows(self):
        for totals in ([0.0], [2.0], [1.0] * 4, [0.0] * 20, [2.0] * 20):
            stats = score_statistics(totals)
            low, high = stats["candidate_elo_ci95"]
            self.assertTrue(math.isfinite(low))
            self.assertTrue(math.isfinite(high))
            self.assertLessEqual(low, high)
        draws = score_statistics([1.0] * 4)
        more_draws = score_statistics([1.0] * 40)
        self.assertLess(
            more_draws["candidate_elo_ci95"][1] - more_draws["candidate_elo_ci95"][0],
            draws["candidate_elo_ci95"][1] - draws["candidate_elo_ci95"][0],
        )
        self.assertLessEqual(
            score_statistics([1.0] * 20)["candidate_elo_ci95"][0], 0.0
        )
        self.assertGreaterEqual(
            score_statistics([1.0] * 20)["candidate_elo_ci95"][1], 0.0
        )

    def test_sha256_file_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample"
            path.write_bytes(b"e1")
            self.assertEqual(
                sha256_file(path),
                "8b5cc4df7eec7d32a7814eca4af047ae33b2d52342667715682e19c25b0b9faa",
            )


if __name__ == "__main__":
    unittest.main()
