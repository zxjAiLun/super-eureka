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
    initial_time_ms_from_time_control,
    parse_comment,
    passed_pawns,
    write_report,
)


class FastchessPgnAnalysisTests(unittest.TestCase):
    def test_time_control_initial_clock_parser(self):
        self.assertEqual(initial_time_ms_from_time_control("10+0.1"), 10000.0)
        self.assertEqual(initial_time_ms_from_time_control("2:00+1"), 120000.0)
        self.assertIsNone(initial_time_ms_from_time_control("unknown"))

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
            '-0.50/4 0.380s, tl=0.500s, lat=-0.002s, n=10, sd=4, '
            'nps=25, hashfull=0, pv="e5"'
        )
        self.assertEqual(info.time_left_ms, 500.0)
        self.assertEqual(info.fastchess_latency_delta_ms, -2.0)

    def test_unitless_clock_diagnostics_are_not_assumed_to_be_seconds(self):
        _, info = parse_comment(
            '-0.50/4 0.380s, tl=500, lat=-2, n=10, sd=4, '
            'nps=25, hashfull=0, pv="e5"'
        )
        self.assertIsNone(info.time_left_ms)
        self.assertIsNone(info.fastchess_latency_delta_ms)

    def test_old_pgn_has_unknown_clock_pressure_but_think_time_diagnostics(self):
        _, info = parse_comment(
            '-0.50/4 0.380s, n=10, sd=4, nps=25, hashfull=0, pv="e5"'
        )
        self.assertIsNone(info.time_left_ms)
        game = chess.pgn.read_game(
            io.StringIO(
                '[Event "fixture"]\n[Result "*"]\n\n'
                '1. e4 {[%eval +0.20] +0.20/4 0.380s, n=10, sd=4, nps=25, '
                'hashfull=0, pv="e4"} e5 '
                '{[%eval +2.00] +2.00/4 0.100s, n=10, sd=4, nps=25, '
                'hashfull=0, pv="e5"} *\n'
            )
        )
        records, _ = analyze_game(game, 1, 150, 4, 0.35)
        self.assertEqual(records[0]["flags"]["time_pressure"], None)
        self.assertTrue(records[0]["flags"]["long_think"])
        self.assertFalse(records[0]["flags"]["short_think"])

    def test_timeleft_sets_clock_pressure_and_ratio(self):
        game = chess.pgn.read_game(
            io.StringIO(
                '[Event "fixture"]\n[Result "*"]\n\n'
                '1. e4 {[%eval +0.20] +0.20/4 0.100s, tl=0.500s, n=10, sd=4, '
                'nps=25, hashfull=0, pv="e4"} e5 '
                '{[%eval +2.00] +2.00/4 0.100s, tl=2.000s, n=10, sd=4, '
                'nps=25, hashfull=0, pv="e5"} *\n'
            )
        )
        records, _ = analyze_game(
            game, 1, 150, 4, 0.35, time_left_threshold_ms=1000, initial_time_ms=10000
        )
        self.assertTrue(records[0]["flags"]["time_pressure"])
        self.assertEqual(records[0]["time_left_ms"], 500.0)
        self.assertEqual(records[0]["time_left_ratio"], 0.05)

    def test_think_time_threshold_has_no_overlap(self):
        game = chess.pgn.read_game(
            io.StringIO(
                '[Event "fixture"]\n[Result "*"]\n\n'
                '1. e4 {[%eval +0.20] +0.20/4 0.350s, n=10, sd=4, '
                'nps=25, hashfull=0, pv="e4"} e5 '
                '{[%eval +2.00] +2.00/4 0.100s, n=10, sd=4, '
                'nps=25, hashfull=0, pv="e5"} *\n'
            )
        )
        records, _ = analyze_game(game, 1, 150, 4, 0.35)
        self.assertFalse(records[0]["flags"]["short_think"])
        self.assertTrue(records[0]["flags"]["long_think"])

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
