import unittest

from run_s21b_cost_attribution import (
    ATTRIBUTION_PROFILES,
    EXPECTED_TIME_MS,
    PROFILE_EVAL_ORDER,
    PROFILE_NO_QCHECKS,
    summarize,
    validate_variants,
)


def fake_row(position_id: str, profile: str) -> dict:
    return {
        "position_id": position_id,
        "group": "control",
        "time_limit_ms": EXPECTED_TIME_MS,
        "engine_profile": profile,
        "completed_depth": 4,
        "nodes": 1000,
        "qsearch_nodes": 700,
        "elapsed_ms": 3000,
        "nps": 333,
        "bestmove": "e2e4",
        "score": {"kind": "cp", "value": 10},
        "component_counters": {
            "check_extensions": 1,
            "single_evasion_extensions": 2,
            "qsearch_check_moves": 3,
            "threat_ordered_moves": 4,
            "root_reorders": 5,
        },
    }


class S21bCostAttributionTests(unittest.TestCase):
    def test_variant_set_is_explicit(self):
        rows = [
            fake_row(f"p{index}", profile)
            for index in range(9)
            for profile in ATTRIBUTION_PROFILES
        ]
        validate_variants(rows)
        summary = summarize(rows)
        self.assertEqual(summary["profiles"][PROFILE_NO_QCHECKS]["positions"], 9)
        self.assertEqual(summary["profiles"][PROFILE_EVAL_ORDER]["sum_counters"]["root_reorders"], 45)

    def test_time_limit_is_fixed(self):
        rows = [
            fake_row(f"p{index}", profile)
            for index in range(9)
            for profile in ATTRIBUTION_PROFILES
        ]
        rows[0]["time_limit_ms"] = 1000
        with self.assertRaises(ValueError):
            validate_variants(rows)


if __name__ == "__main__":
    unittest.main()
