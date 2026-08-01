import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from run_depth_uplift import run_depth_uplift


class DepthUpliftTests(unittest.TestCase):
    @staticmethod
    def _fake_engine(root: Path) -> Path:
        script = root / "fake_depth_engine.py"
        script.write_text(
            textwrap.dedent(
                """
                import sys

                profile = sys.argv[sys.argv.index('--profile') + 1]
                depth = 4 if profile == 'current-qsearch-pruning' else 3

                for raw in sys.stdin:
                    line = raw.strip()
                    if line == 'uci':
                        print('id name D12Fake', flush=True)
                        print('id author tests', flush=True)
                        print('info string search profile ' + profile, flush=True)
                        print('uciok', flush=True)
                    elif line == 'isready':
                        print('readyok', flush=True)
                    elif line.startswith('go'):
                        for current in range(1, depth + 1):
                            print(
                                f'info depth {current} score cp 0 nodes {current * 100} '
                                f'time {current} nps {current * 100} pv e2e4',
                                flush=True,
                            )
                        print('bestmove e2e4', flush=True)
                    elif line == 'quit':
                        break
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return script

    def test_fixed_time_schedule_and_depth_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = self._fake_engine(root)
            with patch.dict(os.environ, {}, clear=False):
                report = run_depth_uplift(
                    script,
                    movetime_ms=10,
                    repeats=2,
                    timeout_s=1.0,
                    fixture_ids={"startpos"},
                )
        self.assertEqual(report["measurement_status"], "PASS")
        self.assertEqual(report["searches_expected"], 4)
        self.assertEqual(report["searches_completed"], 4)
        self.assertEqual([row["order"] for row in report["rows"]], [
            "baseline candidate",
            "baseline candidate",
            "candidate baseline",
            "candidate baseline",
        ])
        summary = report["fixture_summary"][0]
        self.assertEqual(summary["candidate_minus_baseline_depth_median"], 1)
        self.assertIsNone(report["rows"][0]["qsearch_nodes"])
        self.assertTrue(report["resources"]["fresh_process_per_search"])

    def test_unknown_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_depth_uplift(Path(temp) / "missing.exe", fixture_ids={"unknown"})


if __name__ == "__main__":
    unittest.main()
