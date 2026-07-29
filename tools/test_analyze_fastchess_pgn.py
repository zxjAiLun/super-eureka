import io
import json
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn

from analyze_fastchess_pgn import (
    analyze_game,
    analyze_pgn,
    parse_comment,
    passed_pawns,
    write_report,
)


class FastchessPgnAnalysisTests(unittest.TestCase):
    def test_search_comment_parses_eval_and_uci_info(self):
        score, info = parse_comment(
            '[%eval -0.14] -0.50/4 0.380s, n=11252, sd=0, '
            'nps=146129, hashfull=0, pv="e2e4 e8g8"'
        )
        self.assertEqual(score.raw, "-0.14")
        self.assertEqual(info.depth, 4)
        self.assertEqual(info.nodes, 11252)
        self.assertEqual(info.pv, ["e2e4", "e8g8"])

    def test_optional_fastchess_timeleft_and_latency_are_captured(self):
        _, info = parse_comment(
            '-0.50/4 0.380s, tl=1200, lat=-2, n=10, sd=4, '
            'nps=25, hashfull=0, pv="e5"'
        )
        self.assertEqual(info.time_left, 1200.0)
        self.assertEqual(info.latency, -2.0)

    def test_mate_scores_are_signed_and_comparable(self):
        _, info = parse_comment(
            '[%eval #-3] +M5/5 0.100s, n=1, sd=5, nps=10, hashfull=0, pv="e2e4"'
        )
        self.assertEqual(info.score.mate, 5)
        self.assertEqual(info.score.comparable_cp, 99995)

    def test_book_comment_has_eval_but_no_search_info(self):
        score, info = parse_comment('[%eval +0.31] book')
        self.assertEqual(score.cp, 31)
        self.assertIsNone(info)

    def test_black_mover_loss_is_normalized_to_mover_perspective(self):
        game = chess.pgn.read_game(
            io.StringIO(
                '[Event "fixture"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n'
                '1. e4 {[%eval +0.20] +0.20/3 0.100s, n=10, sd=3, nps=25, hashfull=0, pv="e4"} e5 '
                '{[%eval +2.00] +2.00/4 0.400s, n=10, sd=4, nps=25, hashfull=0, pv="e5"} '
                '2. Nf3 {[%eval +0.10] +0.10/3 0.100s, n=10, sd=3, nps=25, hashfull=0, pv="g1f3"} *\n'
            )
        )
        records, counts = analyze_game(game, 1, 150, 4, 0.35)
        self.assertEqual(counts["search_infos"], 3)
        black_record = next(record for record in records if record["mover"] == "black")
        self.assertEqual(black_record["eval_before_cp_mover"], 200)
        self.assertEqual(black_record["eval_after_cp_mover"], -10)
        self.assertEqual(black_record["eval_loss_cp"], 210)
        self.assertTrue(black_record["flags"]["horizon_time_blunder"])

    def test_passed_pawn_and_report_outputs_are_machine_readable(self):
        board = chess.Board("8/k7/4P3/8/8/8/8/4K3 w - - 0 1")
        self.assertEqual(passed_pawns(board)[0]["square"], "e6")
        pgn = Path(tempfile.gettempdir()) / "fastchess-postmortem-fixture.pgn"
        pgn.write_text(
            '[Event "fixture"]\n[SetUp "1"]\n'
            '[FEN "8/k7/4P3/8/8/8/8/4K3 w - - 0 1"]\n'
            '[Result "*"]\n\n1. e7 {[%eval +0.10] 0.10/2 '
            '0.100s, n=1, sd=2, nps=10, hashfull=0, pv="e7"} *\n',
            encoding="utf-8",
        )
        try:
            records, summary = analyze_pgn(pgn)
            with tempfile.TemporaryDirectory() as output:
                write_report(Path(output), records, summary, 5)
                loaded = json.loads((Path(output) / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(loaded["games"], 1)
                self.assertTrue((Path(output) / "candidates.jsonl").is_file())
                self.assertTrue((Path(output) / "report.md").is_file())
        finally:
            pgn.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
