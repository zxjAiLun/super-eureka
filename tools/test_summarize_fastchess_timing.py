import json
from pathlib import Path
import tempfile
import unittest

from summarize_fastchess_timing import summarize


class FastchessTimingSummaryTests(unittest.TestCase):
    def test_groups_search_and_clock_telemetry_by_engine(self):
        pgn = (
            '[Event "fixture"]\n[White "Aspiration"]\n[Black "Current"]\n'
            '[Result "*"]\n\n'
            '1. e4 {+0.20/4 0.100s, tl=0.500s, latency=-0.002s, n=10, '
            'sd=4, nps=100, hashfull=0, pv="e2e4"} e5 '
            '{+0.10/5 0.200s, tl=2.000s, latency=0.003s, n=20, sd=5, '
            'nps=100, hashfull=0, pv="e7e5"} *\n'
        )
        with tempfile.TemporaryDirectory() as temp:
            pgn_path = Path(temp) / "games.pgn"
            pgn_path.write_text(pgn, encoding="utf-8")
            report = summarize(pgn_path)
        self.assertEqual(report["games"], 1)
        self.assertEqual(report["parse_errors"], 0)
        aspiration = report["engines"]["Aspiration"]
        current = report["engines"]["Current"]
        self.assertEqual(aspiration["time_left_ms_min"], 500.0)
        self.assertEqual(aspiration["time_pressure_moves"], 1)
        self.assertEqual(current["time_left_ms_min"], 2000.0)
        self.assertEqual(current["time_pressure_moves"], 0)
        self.assertEqual(aspiration["fastchess_latency_delta_ms_p50"], -2.0)


if __name__ == "__main__":
    unittest.main()
