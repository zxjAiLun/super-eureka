import unittest

import run_s21c_shared_core_attribution as s21c


class S21cSharedCoreAttributionTests(unittest.TestCase):
    def test_profile_order_rotates_by_position(self):
        base = s21c.ATTRIBUTION_PROFILES
        self.assertEqual(s21c.rotated_profile_order(0), base)
        self.assertEqual(
            s21c.rotated_profile_order(1),
            base[1:] + base[:1],
        )
        self.assertEqual(
            s21c.rotated_profile_order(4),
            base,
        )
        self.assertEqual(
            set(s21c.rotated_profile_order(8)),
            set(base),
        )

    def test_summary_requires_and_records_rotated_profile_order(self):
        rows = []
        for position_index in range(s21c.EXPECTED_POSITION_COUNT):
            position_id = f"position-{position_index}"
            for profile in s21c.rotated_profile_order(position_index):
                rows.append(
                    {
                        "position_id": position_id,
                        "engine_profile": profile,
                        "time_limit_ms": s21c.EXPECTED_TIME_MS,
                        "position_index": position_index,
                        "run_order": len(rows),
                        "completed_depth": 5,
                        "nodes": 100,
                        "qsearch_nodes": 80,
                        "elapsed_ms": 3000,
                        "nps": 33,
                        "bestmove": "a2a3",
                        "score": {"kind": "cp", "value": 0},
                        "source_teacher_agreement": profile == s21c.PROFILE_CURRENT,
                        "component_counters": {
                            name: 0 for name in s21c.ATTRIBUTION_COUNTERS
                        },
                    }
                )

        summary = s21c.summarize(rows)
        self.assertEqual(summary["profiles"][s21c.PROFILE_CURRENT]["positions"], 9)
        self.assertEqual(
            summary["profiles"][s21c.PROFILE_CURRENT]["source_teacher_agreement"],
            9,
        )
        self.assertEqual(
            summary["execution_orders"]["position-1"],
            list(s21c.rotated_profile_order(1)),
        )


if __name__ == "__main__":
    unittest.main()
