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
                depth = 4 if profile in {'current-qsearch-pruning', 'current-lmr'} else 3

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
                        reductions = 0 if profile == 'current' else 10
                        researches = 0 if profile == 'current' else 2
                        print(
                            'info string search stats '
                            f'lmr_reductions={reductions} lmr_researches={researches}',
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
        self.assertEqual(summary["equal_depth_paired_samples"], 0)
        self.assertIsNone(report["rows"][0]["qsearch_nodes"])
        self.assertEqual(report["rows"][0]["time_to_depth_ms"], 3)
        self.assertEqual(report["rows"][0]["lmr_reductions"], 0)
        self.assertEqual(report["rows"][1]["lmr_researches"], 2)
        self.assertEqual(report["rows"][1]["lmr_research_rate"], 0.2)
        self.assertTrue(report["resources"]["fresh_process_per_search"])

    def test_unknown_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_depth_uplift(Path(temp) / "missing.exe", fixture_ids={"unknown"})

    def test_gate_decision_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_depth_uplift(Path(temp) / "missing.exe", fixture_ids={"startpos"}, gate_decision="maybe")

    def test_profiles_are_explicit_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run_depth_uplift(
                self._fake_engine(Path(temp)),
                movetime_ms=10,
                repeats=1,
                timeout_s=1.0,
                fixture_ids={"startpos"},
                baseline_profile="current",
                candidate_profile="current-lmr",
            )
        self.assertEqual(
            report["profiles"],
            {"baseline": "current", "candidate": "current-lmr"},
        )
        self.assertEqual(report["fixture_summary"][0]["candidate_minus_baseline_depth_median"], 1)

    def test_equal_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_depth_uplift(
                    Path(temp) / "missing.exe",
                    fixture_ids={"startpos"},
                    baseline_profile="current",
                    candidate_profile="current",
                )

    def test_fixture_summaries_do_not_mix_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            report = run_depth_uplift(
                self._fake_engine(Path(temp)),
                movetime_ms=10,
                repeats=1,
                timeout_s=1.0,
                fixture_ids={"startpos", "queen-win"},
            )
        self.assertEqual(len(report["fixture_summary"]), 2)
        self.assertEqual(
            {item["fixture"] for item in report["fixture_summary"]},
            {"startpos", "queen-win"},
        )


if __name__ == "__main__":
    unittest.main()
