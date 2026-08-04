import tempfile
import unittest
from pathlib import Path

import verify_s3_final_match as verifier


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def game_text(white, black, result, moves):
    return (
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        f'[Result "{result}"]\n'
        f'[FEN "{STARTPOS}"]\n'
        '[SetUp "1"]\n'
        '[TimeControl "10+0.1"]\n\n'
        f"{moves} {result}\n\n"
    )


class S3FinalMatchVerifierTests(unittest.TestCase):
    def write_inputs(self, directory, pgn):
        pgn_path = directory / "match.pgn"
        opening_path = directory / "openings.epd"
        stdout_path = directory / "stdout.log"
        stderr_path = directory / "stderr.log"
        engine_path = directory / "engine.exe"
        manager_path = directory / "cutechess.exe"
        pgn_path.write_text(pgn, encoding="utf-8")
        opening_path.write_text(STARTPOS + "\n", encoding="utf-8")
        stdout_path.write_text(
            "Started game 1 of 2\n"
            "Finished game 1 (CurrentFinal vs Current): 1-0\n"
            "Started game 2 of 2\n"
            "Finished game 2 (Current vs CurrentFinal): 1/2-1/2\n"
            "Finished match\n",
            encoding="utf-8",
        )
        stderr_path.write_text(verifier.EXPECTED_STDERR + "\n", encoding="utf-8")
        engine_path.write_bytes(b"engine")
        manager_path.write_bytes(b"manager")
        return pgn_path, opening_path, stdout_path, stderr_path, engine_path, manager_path

    def test_accepts_swapped_pair_and_scores_from_candidate_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = self.write_inputs(
                directory,
                game_text("CurrentFinal", "Current", "1-0", "1. e4 e5 2. Nf3 Nc6")
                + game_text("Current", "CurrentFinal", "1/2-1/2", "1. d4 d5 2. c4 e6"),
            )
            summary = verifier.verify_match(*inputs[:4], expected_game_count=2)
            self.assertEqual(summary["candidate_wins"], 1)
            self.assertEqual(summary["draws"], 1)
            self.assertEqual(summary["candidate_score_percent"], 75.0)
            self.assertEqual(summary["candidate_color_counts"], {"White": 1, "Black": 1})

    def test_rejects_pair_without_color_swap(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = self.write_inputs(
                directory,
                game_text("CurrentFinal", "Current", "1-0", "1. e4 e5 2. Nf3 Nc6")
                + game_text("CurrentFinal", "Current", "1-0", "1. d4 d5 2. c4 e6"),
            )
            with self.assertRaises(ValueError):
                verifier.verify_match(*inputs[:4], expected_game_count=2)

    def test_quick_screen_exit_code_rejects_below_threshold(self):
        self.assertEqual(
            verifier.quick_screen_exit_code({"candidate_score_percent": 59.5}),
            3,
        )

    def test_quick_screen_exit_code_accepts_threshold_boundary(self):
        self.assertEqual(
            verifier.quick_screen_exit_code({"candidate_score_percent": 60.0}),
            0,
        )


if __name__ == "__main__":
    unittest.main()
