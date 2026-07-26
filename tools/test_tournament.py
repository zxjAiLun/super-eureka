import math
import sys
import unittest

import chess

from tournament import (
    DEFAULT_OPENINGS,
    EngineProtocolError,
    EngineSession,
    EngineTimeout,
    SprtState,
    build_schedule,
    load_openings,
    parse_bestmove_line,
    pgn_for_record,
    score_statistics,
    validate_bestmove,
)
from tournament import GameRecord, TournamentError, _terminal_result


class TournamentUnitTests(unittest.TestCase):
    @staticmethod
    def fake_engine(mode: str) -> list[str]:
        script = f"""
import sys
import time
mode = {mode!r}
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name E1Fake', flush=True)
        print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('go'):
        if mode == 'timeout':
            time.sleep(0.5)
        elif mode == 'bestmove':
            print('bestmove a2a3', flush=True)
    elif line == 'quit':
        break
"""
        return [sys.executable, "-u", "-c", script]

    def test_uci_handshake_and_bestmove(self):
        with EngineSession(self.fake_engine("bestmove"), "fake", 16) as engine:
            engine.position([])
            move, elapsed_ms = engine.go_movetime(1)
        self.assertEqual(move, "a2a3")
        self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_uci_timeout_is_distinguished(self):
        with self.assertRaises(EngineTimeout):
            with EngineSession(
                self.fake_engine("timeout"), "fake", 16, response_timeout=0.05
            ) as engine:
                engine.position([])
                engine.go_movetime(1)

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
        result, reason = _terminal_result(board)
        self.assertEqual((result, reason), ("1-0", "checkmate"))
        with self.assertRaises(EngineProtocolError):
            validate_bestmove(chess.Board(), "0000")

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

    def test_schedule_balances_colors_and_is_seeded(self):
        openings = load_openings(DEFAULT_OPENINGS)
        first = build_schedule(openings, 64, seed=7)
        second = build_schedule(openings, 64, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(sum(game.engine_a_white for game in first), 32)
        self.assertEqual(len({game.opening.identifier for game in first}), 32)
        for index in range(0, 64, 2):
            self.assertEqual(first[index].opening, first[index + 1].opening)
            self.assertNotEqual(first[index].engine_a_white, first[index + 1].engine_a_white)

    def test_sprt_reaches_pass_and_reject_boundaries(self):
        passing = SprtState()
        while passing.decision is None:
            passing.update("win")
        self.assertEqual(passing.decision, "PASS")
        self.assertGreaterEqual(passing.llr, passing.upper_bound)

        rejecting = SprtState()
        while rejecting.decision is None:
            rejecting.update("loss")
        self.assertEqual(rejecting.decision, "REJECTED")
        self.assertLessEqual(rejecting.llr, rejecting.lower_bound)

    def test_sprt_draw_has_half_score_contribution(self):
        state = SprtState()
        state.update("draw")
        p0 = 1.0 / (1.0 + 10.0 ** (-0.0 / 400.0))
        p1 = 1.0 / (1.0 + 10.0 ** (-5.0 / 400.0))
        expected = 0.5 * math.log(p1 / p0) + 0.5 * math.log(
            (1.0 - p1) / (1.0 - p0)
        )
        self.assertAlmostEqual(state.llr, expected)
        self.assertEqual(state.draws, 1)

    def test_score_statistics_reports_ci(self):
        stats = score_statistics(6, 2, 2)
        self.assertEqual(stats["games"], 10)
        self.assertAlmostEqual(stats["score"], 0.7)
        self.assertIsNotNone(stats["elo"])
        self.assertEqual(len(stats["elo_ci95"]), 2)

        draws_only = score_statistics(0, 64, 0)
        self.assertLess(draws_only["elo_ci95"][0], 0.0)
        self.assertGreater(draws_only["elo_ci95"][1], 0.0)

    def test_pgn_replays_opening_and_played_moves(self):
        record = GameRecord(
            number=1,
            opening="test",
            engine_a_white=True,
            result="1/2-1/2",
            reason="max-plies",
            plies=4,
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            elapsed_ms={"white": 1.0, "black": 1.0},
        )
        text = pgn_for_record(record, "A", "B")
        self.assertIn('[Result "1/2-1/2"]', text)
        self.assertIn("1. e4 e5 2. Nf3 Nc6", text)


if __name__ == "__main__":
    unittest.main()
