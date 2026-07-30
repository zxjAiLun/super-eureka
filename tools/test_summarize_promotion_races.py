import json
from pathlib import Path
import tempfile
import unittest

from summarize_promotion_races import select_records, write_report


class PromotionRaceSummaryTests(unittest.TestCase):
    def test_filters_non_mate_races_and_renames_historical_latency(self):
        records = [
            {
                "eval_loss_cp": 220,
                "latency_ms": None,
                "flags": {"promotion_race": True, "mate_transition": False},
                "pv": ["e7e8q"],
            },
            {
                "eval_loss_cp": 100000,
                "latency_ms": None,
                "flags": {"promotion_race": True, "mate_transition": True},
            },
            {
                "eval_loss_cp": 120,
                "flags": {"promotion_race": True, "mate_transition": False},
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "candidates.jsonl"
            source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            selected, count = select_records(source)
            self.assertEqual(count, 3)
            self.assertEqual(len(selected), 1)
            self.assertIn("fastchess_latency_delta_ms", selected[0])
            self.assertNotIn("latency_ms", selected[0])
            summary = write_report(source, root / "report")
            self.assertEqual(summary["selected_records"], 1)
            self.assertTrue((root / "report" / "promotion-races.jsonl").is_file())
            self.assertIn("e7e8q", (root / "report" / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
