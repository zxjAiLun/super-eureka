import unittest

import run_s22a_fixed_time_gate as s22a


def synthetic_rows(candidate_depth=6, candidate_nps=100, control_candidate="g8f8"):
    positions = s22a.load_manifest(
        s22a.Path("tests/data/s2.2a-integrated-positions.json")
    )
    rows = []
    for time_index, time_ms in enumerate(s22a.EXPECTED_TIME_MS):
        for position_index, position in enumerate(positions):
            order = s22a.rotated_profile_order(time_index, position_index)
            for profile in order:
                is_candidate = profile == s22a.PROFILE_CANDIDATE
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
                    "completed_depth": candidate_depth if is_candidate else 6,
                    "nodes": 100,
                    "qsearch_nodes": 80,
                    "eval_calls": 90,
                    "nps": candidate_nps if is_candidate else 100,
                    "elapsed_ms": time_ms,
                    "bestmove": (
                        control_candidate
                        if position["id"] == "s21-control-237-32" and is_candidate
                        else "g8f8"
                    ),
                    "score": {"kind": "cp", "value": 0},
                    "pv": ["g8f8"],
                    "metrics": {name: 0 for name in s22a.SEARCH_METRICS},
                    "counters": {name: 0 for name in s22a.FEATURE_COUNTERS},
                }
                rows.append(row)
    return rows


class S22aFixedTimeGateTests(unittest.TestCase):
    def test_manifest_is_exactly_pinned(self):
        positions = s22a.load_manifest(
            s22a.Path("tests/data/s2.2a-integrated-positions.json")
        )
        self.assertEqual(len(positions), 25)
        self.assertEqual(
            {position["group"] for position in positions},
            set(s22a.EXPECTED_GROUPS),
        )
        for group in s22a.EXPECTED_GROUPS:
            self.assertEqual(sum(position["group"] == group for position in positions), 5)
        self.assertEqual(
            sum(position["id"].startswith("s22a-") for position in positions),
            16,
        )

    def test_rotation_and_exact_150_row_matrix(self):
        rows = synthetic_rows()
        s22a.validate_rows(rows)
        self.assertEqual(len(rows), 150)
        self.assertEqual(
            s22a.rotated_profile_order(0, 0),
            (s22a.PROFILE_CURRENT, s22a.PROFILE_CANDIDATE),
        )
        self.assertEqual(
            s22a.rotated_profile_order(0, 1),
            (s22a.PROFILE_CANDIDATE, s22a.PROFILE_CURRENT),
        )

    def test_pass_has_no_quality_flag(self):
        decision = s22a.evaluate_gate(synthetic_rows())
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["quality_flags"], [])
        self.assertTrue(all(decision["criteria"]["candidate_median_depth_within_one"].values()))
        self.assertTrue(all(decision["criteria"]["candidate_median_nps_at_least_0_65"].values()))

    def test_quality_flag_does_not_become_hard_failure(self):
        decision = s22a.evaluate_gate(synthetic_rows(control_candidate="g8h8"))
        self.assertEqual(decision["status"], "PASS_WITH_QUALITY_FLAGS")
        self.assertEqual(len(decision["quality_flags"]), 3)
        self.assertEqual(decision["reasons"], [])

    def test_depth_nps_counter_and_stable_regression_fail(self):
        rows = synthetic_rows(candidate_depth=4, candidate_nps=50)
        for row in rows:
            if row["position_id"] == "s21-control-237-32" and row["engine_profile"] == s22a.PROFILE_CANDIDATE:
                row["counters"]["root_reorders"] = 1
        decision = s22a.evaluate_gate(rows)
        self.assertEqual(decision["status"], "FAIL")
        self.assertFalse(all(decision["criteria"]["candidate_median_depth_within_one"].values()))
        self.assertFalse(all(decision["criteria"]["candidate_median_nps_at_least_0_65"].values()))
        self.assertFalse(decision["criteria"]["no_stable_two_plus_depth_regression"])
        self.assertFalse(decision["criteria"]["feature_counters_all_zero"])


if __name__ == "__main__":
    unittest.main()
