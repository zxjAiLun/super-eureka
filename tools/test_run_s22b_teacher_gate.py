import json
import tempfile
import unittest
from pathlib import Path

from analyze_d114_stockfish import ParsedScore, SearchResult
import run_s22b_teacher_gate as s22b


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "s2.2a" / "integrated-fixed-time.json"
MANIFEST = ROOT / "tests" / "data" / "s2.2a-integrated-positions.json"


class S22bTeacherGateTests(unittest.TestCase):
    def test_source_provenance_and_mapping_are_exact(self):
        data, rows, positions = s22b.load_source_artifact(SOURCE, MANIFEST)
        self.assertEqual(data["git_sha"], s22b.SOURCE_GATE_GIT_SHA)
        self.assertEqual(len(rows), 150)
        self.assertEqual(len(positions), 25)
        self.assertEqual(
            {(row["position_id"], int(row["time_limit_ms"]), row["engine_profile"]) for row in rows},
            {
                (position_id, time_ms, profile)
                for position_id in positions
                for time_ms in s22b.EXPECTED_TIME_MS
                for profile in s22b.PROFILES
            },
        )

    def test_source_provenance_rejects_git_sha_drift(self):
        data = json.loads(SOURCE.read_text(encoding="utf-8"))
        data["git_sha"] = "wrong"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                s22b.load_source_artifact(path, MANIFEST)

    def test_mate_categories_include_severe_outcomes(self):
        self.assertEqual(
            s22b.mate_category_s22b(ParsedScore("mate", 7), ParsedScore("cp", 300)),
            "missed_winning_mate",
        )
        self.assertEqual(
            s22b.mate_category_s22b(ParsedScore("cp", 300), ParsedScore("mate", -4)),
            "entered_losing_mate",
        )
        self.assertEqual(
            s22b.mate_category_s22b(ParsedScore("mate", -7), ParsedScore("mate", -3)),
            "losing_mate_accelerated",
        )

    def test_raw_negative_cpl_is_not_clamped(self):
        metrics = s22b.teacher_metrics(
            SearchResult("e2e4", ParsedScore("cp", 100), 10, ()),
            SearchResult("e2e4", ParsedScore("cp", 105), 10, ()),
            "e2e4",
        )
        self.assertEqual(metrics["centipawn_loss_raw"], -5)
        self.assertTrue(metrics["negative_cpl_anomaly"])

    def test_decision_requires_target_group_signals(self):
        stats = {
            "moves": 75,
            "cp_scored_moves": 75,
            "median_cpl": 10,
            "p90_cpl": 20,
            "harmful_mate": 0,
            "missed_winning_mate": 0,
            "entered_losing_mate": 0,
            "cpl_300_plus": 0,
        }
        groups = {
            group: {
                "net_positive_signal": False,
                "candidate_harmful_mate_not_more": True,
                "profiles": {
                    s22b.PROFILE_CURRENT: {"cpl_300_plus": 0},
                    s22b.PROFILE_CANDIDATE: {"cpl_300_plus": 0},
                },
            }
            for group in s22b.EXPECTED_GROUPS
        }
        pairs = {"candidate_wins": 0, "current_wins": 0}
        decision = s22b.evaluate_decision(
            {s22b.PROFILE_CURRENT: stats, s22b.PROFILE_CANDIDATE: stats},
            {profile: [] for profile in s22b.PROFILES},
            pairs,
            groups,
        )
        self.assertEqual(decision["status"], "FAIL")
        self.assertIn("development-coordination has no net positive signal", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
