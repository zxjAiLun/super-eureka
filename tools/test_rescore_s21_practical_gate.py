import json
import tempfile
import unittest
from pathlib import Path

from rescore_s21_practical_gate import (
    EXPECTED_GROUPS,
    EXPECTED_PROFILES,
    EXPECTED_ROW_COUNT,
    EXPECTED_TIME_LIMITS,
    build_summary,
    load_practical_gate,
)


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "s2.1" / "practical-gate.json"


class S21TeacherRescoreTests(unittest.TestCase):
    def test_saved_gate_has_exact_contract(self):
        data = load_practical_gate(INPUT)
        self.assertEqual(len(data["rows"]), EXPECTED_ROW_COUNT)
        self.assertEqual(
            {row["engine_profile"] for row in data["rows"]}, set(EXPECTED_PROFILES)
        )
        self.assertEqual(
            {row["group"] for row in data["rows"]}, set(EXPECTED_GROUPS)
        )
        self.assertEqual(
            {int(row["time_limit_ms"]) for row in data["rows"]},
            set(EXPECTED_TIME_LIMITS),
        )

    def test_loader_rejects_row_count_drift(self):
        data = json.loads(INPUT.read_text(encoding="utf-8"))
        data["rows"] = data["rows"][:-1]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gate.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_practical_gate(path)

    def test_summary_reports_profile_time_group_and_pair_winners(self):
        rows = [
            {
                "position_id": "p1",
                "group": "control",
                "time_limit_ms": 1000,
                "engine_profile": "current",
                "played_move": "e2e4",
                "centipawn_loss": 80,
                "best_move_agreement": False,
                "classification": "mistake",
                "mate_swing": False,
                "mate_outcome": False,
                "mate_category": None,
            },
            {
                "position_id": "p1",
                "group": "control",
                "time_limit_ms": 1000,
                "engine_profile": "current-threat-aware",
                "played_move": "d2d4",
                "centipawn_loss": 20,
                "best_move_agreement": True,
                "classification": "normal",
                "mate_swing": False,
                "mate_outcome": False,
                "mate_category": None,
            },
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["profile_time_group"]["current"]["1000"]["control"]["mistake"], 1)
        comparison = summary["paired_comparison"]
        self.assertEqual(comparison["paired_position_time_groups"], 1)
        self.assertEqual(comparison["candidate_lower_cpl"], 1)
        self.assertEqual(comparison["different_move_groups"], 1)
        self.assertEqual(comparison["per_position_winners"][0]["winner"], "current-threat-aware")


if __name__ == "__main__":
    unittest.main()
