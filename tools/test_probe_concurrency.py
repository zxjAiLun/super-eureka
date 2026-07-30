from pathlib import Path
import tempfile
import unittest

from probe_concurrency import (
    empirical_probe,
    _parse_info,
    probe,
    recommend_concurrency,
)


class ConcurrencyProbeTests(unittest.TestCase):
    def test_topology_mode_is_observational(self):
        report = probe()
        self.assertEqual(report["mode"], "topology")
        self.assertFalse(report["formal_match_started"])
        self.assertEqual(report["engine_thread_model"], "single-threaded")

    def test_info_parser_extracts_search_measurements(self):
        parsed = _parse_info(
            'info depth 6 score cp 20 nodes 1000000 time 8000 nps 125000 pv="e2e4"'
        )
        self.assertEqual(parsed["depth"], 6)
        self.assertEqual(parsed["nodes"], 1_000_000)
        self.assertEqual(parsed["time"], 8000)
        self.assertEqual(parsed["nps"], 125000)

    def test_recommendation_maximizes_aggregate_throughput_with_safety_rules(self):
        points = [
            {
                "concurrency": 1,
                "workers_failed": 0,
                "median_worker_nps": 100.0,
                "aggregate_nps": 100.0,
                "median_duration_ms": 1000.0,
                "p95_duration_ms": 1000.0,
            },
            {
                "concurrency": 2,
                "workers_failed": 0,
                "median_worker_nps": 85.0,
                "aggregate_nps": 150.0,
                "median_duration_ms": 1000.0,
                "p95_duration_ms": 1300.0,
            },
            {
                "concurrency": 4,
                "workers_failed": 0,
                "median_worker_nps": 60.0,
                "aggregate_nps": 190.0,
                "median_duration_ms": 1000.0,
                "p95_duration_ms": 1000.0,
            },
            {
                "concurrency": 8,
                "workers_failed": 1,
                "median_worker_nps": 95.0,
                "aggregate_nps": 250.0,
                "median_duration_ms": 1000.0,
                "p95_duration_ms": 1000.0,
            },
        ]
        recommendation, baseline = recommend_concurrency(points)
        self.assertEqual(recommendation, 4)
        self.assertEqual(baseline, 100.0)
        self.assertTrue(points[1]["eligible"])
        self.assertTrue(points[2]["eligible"])
        self.assertFalse(points[3]["eligible"])
        self.assertEqual(points[2]["relative_worker_speed"], 0.6)
        self.assertEqual(points[2]["aggregate_throughput_ratio"], 1.9)

    def test_empirical_probe_runs_parallel_uci_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root / "fake_engine.py"
            fake.write_text(
                "import sys\n"
                "for raw in sys.stdin:\n"
                "    line = raw.strip()\n"
                "    if line == 'uci':\n"
                "        print('id name fake', flush=True)\n"
                "        print('id author test', flush=True)\n"
                "        print('uciok', flush=True)\n"
                "    elif line == 'isready':\n"
                "        print('readyok', flush=True)\n"
                "    elif line.startswith('go nodes'):\n"
                "        nodes = int(line.split()[-1])\n"
                "        # Simulate a node cap interrupting an unfinished next iteration:\n"
                "        # the last complete info line under-reports the requested work.\n"
                "        reported = nodes // 2\n"
                "        print(f'info depth 4 nodes {reported} time 10 nps {reported * 100}', flush=True)\n"
                "        print('bestmove e2e4', flush=True)\n"
                "    elif line == 'quit':\n"
                "        break\n",
                encoding="utf-8",
            )
            report = empirical_probe(
                fake,
                concurrency_points=(1, 2),
                fixtures=("open-tactical",),
                nodes=1000,
                repeat=1,
                warmup=0,
                timeout_s=5,
            )
            self.assertEqual(report["mode"], "empirical")
            self.assertIn(report["recommended_fastchess_concurrency"], (1, 2))
            self.assertEqual([point["workers_failed"] for point in report["points"]], [0, 0])
            self.assertEqual(report["points"][1]["workers_completed"], 2)
            self.assertEqual(report["points"][0]["worker_results"][0]["work_nodes"], 1000)
            self.assertEqual(report["points"][0]["worker_results"][0]["reported_info_nodes"], [500])
            self.assertEqual(report["points"][0]["aggregate_work_nodes"], 1000)
            self.assertGreater(report["points"][1]["aggregate_throughput_ratio"], 0.0)
            self.assertFalse(report["formal_match_started"])


if __name__ == "__main__":
    unittest.main()
