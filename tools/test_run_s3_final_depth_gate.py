import unittest

import run_s3_final_depth_gate as s3


def synthetic_rows(candidate_depths=(7, 6), activity=True):
    positions = s3.load_manifest(s3.Path("tests/data/s2.2a-integrated-positions.json"))
    rows = []
    for time_index, time_ms in enumerate(s3.EXPECTED_TIME_MS):
        for position_index, position in enumerate(positions):
            order = s3.rotated_profile_order(time_index, position_index)
            for profile in order:
                counters = {name: 0 for name in s3.FEATURE_COUNTERS}
                if profile == s3.PROFILE_CANDIDATE and activity:
                    for name in s3.AUTHORIZED_ACTIVITY_COUNTERS:
                        counters[name] = 1
                    counters["qsearch_see_pruned"] = 1
                row = {
                    "position_id": position["id"],
                    "group": position["group"],
                    "fen": position["fen"],
                    "time_limit_ms": time_ms,
                    "time_index": time_index,
                    "position_index": position_index,
                    "run_order": len(rows),
                    "rotated_profile_order": list(order),
                    "engine_profile": profile,
                    "completed_depth": (
                        candidate_depths[time_index]
                        if profile == s3.PROFILE_CANDIDATE
                        else 6
                    ),
                    "nodes": 100,
                    "qsearch_nodes": 80,
                    "eval_calls": 90,
                    "nps": 100,
                    "elapsed_ms": time_ms,
                    "bestmove": "e2e4",
                    "pv": ["e2e4", "e7e5"],
                    "score": {"kind": "cp", "value": 0},
                    "metrics": {name: 0 for name in s3.SEARCH_METRICS},
                    "counters": counters,
                }
                rows.append(row)
    return rows


class S3FinalDepthGateTests(unittest.TestCase):
    def test_manifest_and_matrix_are_exactly_pinned(self):
        positions = s3.load_manifest(
            s3.Path("tests/data/s2.2a-integrated-positions.json")
        )
        self.assertEqual(len(positions), 25)
        rows = synthetic_rows()
        s3.validate_rows(rows)
        self.assertEqual(len(rows), 100)

    def test_pass_requires_one_full_median_ply_and_activity(self):
        decision = s3.evaluate_gate(synthetic_rows())
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["reasons"], [])
        self.assertEqual(decision["criteria"]["median_depth_uplift_tiers"], ["1000"])
        self.assertTrue(decision["criteria"]["authorized_feature_activity_nonzero"])

    def test_missing_uplift_is_a_hard_failure(self):
        decision = s3.evaluate_gate(synthetic_rows(candidate_depths=(6, 6)))
        self.assertEqual(decision["status"], "FAIL")
        self.assertIn("no fixed-time median depth uplift", " ".join(decision["reasons"]))

    def test_counter_isolation_and_activity_are_hard_gates(self):
        rows = synthetic_rows(activity=False)
        candidate = next(
            row for row in rows if row["engine_profile"] == s3.PROFILE_CANDIDATE
        )
        candidate["counters"]["root_reorders"] = 1
        decision = s3.evaluate_gate(rows)
        self.assertEqual(decision["status"], "FAIL")
        self.assertFalse(decision["criteria"]["candidate_unauthorized_counters_zero"])
        self.assertFalse(decision["criteria"]["authorized_feature_activity_nonzero"])

    def test_stable_two_depth_regression_is_a_hard_failure(self):
        rows = synthetic_rows()
        for row in rows:
            if row["position_id"] == "s21-attack-345-78" and row["engine_profile"] == s3.PROFILE_CANDIDATE:
                row["completed_depth"] = 4
        decision = s3.evaluate_gate(rows)
        self.assertEqual(decision["status"], "FAIL")
        self.assertIn("at least two depths behind", " ".join(decision["reasons"]))

    def test_pv_validation_rejects_illegal_or_mismatched_lines(self):
        startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        s3.validate_bestmove_and_pv(startpos, "e2e4", ["e2e4", "e7e5"])
        with self.assertRaises(ValueError):
            s3.validate_bestmove_and_pv(startpos, "e2e4", ["d2d4"])
        with self.assertRaises(ValueError):
            s3.validate_bestmove_and_pv(startpos, "e2e5", ["e2e5"])


if __name__ == "__main__":
    unittest.main()
