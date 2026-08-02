import unittest
from pathlib import Path
import tempfile

import chess
import chess.pgn

from verify_d114_match import (
    BASELINE_LABEL,
    CANDIDATE_LABEL,
    D114VerificationError,
    VerifiedGame,
    candidate_score,
    compare_existing_summary,
    descriptive_statistics,
    position_key,
    validate_game,
    verify_pairs,
)


CHECKMATED_BLACK = "7k/5Q2/7K/8/8/8/8/8 b - - 0 1"


def terminal_game(white: str, black: str, result: str) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers.update(
        {
            "FEN": CHECKMATED_BLACK,
            "SetUp": "1",
            "White": white,
            "Black": black,
            "Result": result,
        }
    )
    return game


class D114VerifierTests(unittest.TestCase):
    def test_candidate_score_uses_candidate_perspective(self):
        self.assertEqual(candidate_score("1-0", CANDIDATE_LABEL), 1.0)
        self.assertEqual(candidate_score("0-1", BASELINE_LABEL), 1.0)
        self.assertEqual(candidate_score("1/2-1/2", CANDIDATE_LABEL), 0.5)

    def test_pair_requires_color_swap(self):
        games = [
            VerifiedGame(1, CHECKMATED_BLACK, CANDIDATE_LABEL, BASELINE_LABEL, "1-0", 1.0, True, "checkmate"),
            VerifiedGame(2, CHECKMATED_BLACK, BASELINE_LABEL, CANDIDATE_LABEL, "1-0", 0.0, True, "checkmate"),
        ]
        verify_pairs(games, [CHECKMATED_BLACK], 2, require_sequential_order=True)

    def test_pair_with_missing_game_is_rejected(self):
        game = VerifiedGame(1, CHECKMATED_BLACK, CANDIDATE_LABEL, BASELINE_LABEL, "1-0", 1.0, True, "checkmate")
        with self.assertRaises(D114VerificationError):
            verify_pairs([game], [CHECKMATED_BLACK], 2, require_sequential_order=True)

    def test_illegal_pgn_move_is_rejected(self):
        game = terminal_game(CANDIDATE_LABEL, BASELINE_LABEL, "1-0")
        game.board().push(chess.Move.null())
        game.add_main_variation(chess.Move.from_uci("a1a8"))
        with self.assertRaises(D114VerificationError):
            validate_game(game, 1, {position_key(CHECKMATED_BLACK)}, {1: ("1-0", "checkmate")})

    def test_pentanomial_and_descriptive_stats_are_candidate_oriented(self):
        games = [
            VerifiedGame(1, CHECKMATED_BLACK, CANDIDATE_LABEL, BASELINE_LABEL, "1-0", 1.0, True, "checkmate"),
            VerifiedGame(2, CHECKMATED_BLACK, BASELINE_LABEL, CANDIDATE_LABEL, "1-0", 0.0, True, "checkmate"),
        ]
        summary = descriptive_statistics(games, "Smoke")
        self.assertEqual(summary["candidate_wins"], 1)
        self.assertEqual(summary["candidate_losses"], 1)
        self.assertEqual(summary["pentanomial_candidate_points_0_to_2"], [0, 0, 1, 0, 0])

    def test_existing_summary_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            path.write_text('{"games": 1}\n', encoding="utf-8")
            with self.assertRaises(D114VerificationError):
                compare_existing_summary(path, {"games": 2})


if __name__ == "__main__":
    unittest.main()
