import unittest

import run_s21d_fixed_time_gate as s21d


def synthetic_rows():
    rows = []
    for time_index, time_ms in enumerate(s21d.EXPECTED_TIME_MS):
        for position_index in range(s21d.EXPECTED_POSITION_COUNT):
            order = s21d.rotated_profile_order(time_index, position_index)
            position_id = (
                "s21-control-237-32"
                if position_index == 0
                else f"position-{position_index}"
            )
            for profile in order:
                rows.append(
                    {
                        "position_id": position_id,
                        "time_limit_ms": time_ms,
                        "time_index": time_index,
                        "position_index": position_index,
                        "run_order": len(rows),
                        "rotated_profile_order": list(order),
                        "engine_profile": profile,
                        "completed_depth": 6,
                        "nodes": 100,
                        "elapsed_ms": time_ms,
                        "nps": 100,
                        "bestmove": "g8f8",
                        "source_teacher_agreement": False,
                        "counters": {name: 0 for name in s21d.FEATURE_COUNTERS},
                    }
                )
    return rows


class S21dFixedTimeGateTests(unittest.TestCase):
    def test_profile_order_reverses_by_time_and_position(self):
        self.assertEqual(
            s21d.rotated_profile_order(0, 0),
            (s21d.PROFILE_CURRENT, s21d.PROFILE_EVAL_ONLY),
        )
        self.assertEqual(
            s21d.rotated_profile_order(0, 1),
            (s21d.PROFILE_EVAL_ONLY, s21d.PROFILE_CURRENT),
        )
        self.assertEqual(
            s21d.rotated_profile_order(1, 0),
            (s21d.PROFILE_EVAL_ONLY, s21d.PROFILE_CURRENT),
        )

    def test_exact_matrix_and_zero_counter_gate(self):
        rows = synthetic_rows()
        s21d.validate_rows(rows)
        decision = s21d.evaluate_gate(rows)
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["criteria"]["feature_counters_all_zero"], True)
        self.assertEqual(decision["criteria"]["control_237_safe"], True)

    def test_counter_and_control_regressions_fail(self):
        rows = synthetic_rows()
        rows[0]["counters"]["root_reorders"] = 1
        for row in rows:
            if (
                row["position_id"] == "s21-control-237-32"
                and row["time_limit_ms"] == 3000
                and row["engine_profile"] == s21d.PROFILE_EVAL_ONLY
            ):
                row["bestmove"] = "g8h8"
            if (
                row["position_id"] == "s21-control-237-32"
                and row["time_limit_ms"] == 3000
                and row["engine_profile"] == s21d.PROFILE_CURRENT
            ):
                row["bestmove"] = "g8f8"
        decision = s21d.evaluate_gate(rows)
        self.assertEqual(decision["status"], "FAIL")
        self.assertFalse(decision["criteria"]["feature_counters_all_zero"])
        self.assertFalse(decision["criteria"]["control_237_safe"])


if __name__ == "__main__":
    unittest.main()
