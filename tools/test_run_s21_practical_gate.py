import json
import unittest
from pathlib import Path

from run_s21_practical_gate import (
    EXPECTED_GROUPS,
    PROFILE_CANDIDATE,
    PROFILE_CURRENT,
    load_manifest,
    parse_info_depth_lines,
    parse_key_values,
    parse_score,
)


class S21PracticalGateTests(unittest.TestCase):
    def test_manifest_has_three_fixed_diagnostic_groups(self):
        manifest = (
            Path(__file__).resolve().parent.parent
            / "tests"
            / "data"
            / "s2.1-practical-positions.json"
        )
        positions = load_manifest(manifest.resolve())
        self.assertEqual(len(positions), 9)
        self.assertEqual(
            {position["group"] for position in positions}, set(EXPECTED_GROUPS)
        )
        for group in EXPECTED_GROUPS:
            self.assertEqual(
                sum(position["group"] == group for position in positions), 3
            )

    def test_parse_bench_result_preserves_profile_counters_and_pv(self):
        line = (
            'bench_result suite=profile fixture=custom mode=cold '
            'profile=current-threat-aware repeat=1 limit=movetime_ms:100 '
            'score=cp:-380 bestmove=c7c6 completed_depth=2 stopped=true '
            'nodes=8710 elapsed_us=98013 nps=88865 pv="c7c6 g5f6" '
            'elapsed_ms=98 check_extensions=624 single_evasion_extensions=86 '
            'qsearch_check_moves=5068'
        )
        fields = parse_key_values(line)
        self.assertEqual(fields["profile"], PROFILE_CANDIDATE)
        self.assertEqual(parse_score(fields["score"]), {"kind": "cp", "value": -380})
        self.assertEqual(fields["pv"], "c7c6 g5f6")
        self.assertEqual(int(fields["check_extensions"]), 624)
        self.assertEqual(int(fields["single_evasion_extensions"]), 86)
        self.assertEqual(int(fields["qsearch_check_moves"]), 5068)

    def test_parse_info_depth_keeps_score_and_pv_by_iteration(self):
        stdout = "\n".join(
            [
                "info depth 1 score cp -64 nodes 419 time 3 nps 139666 pv c7c6",
                "info depth 2 score mate 3 nodes 2547 time 23 nps 110739 pv c7c6 g5f6",
            ]
        )
        iterations = parse_info_depth_lines(stdout)
        self.assertEqual([item["depth"] for item in iterations], [1, 2])
        self.assertEqual(iterations[0]["score"], {"kind": "cp", "value": -64})
        self.assertEqual(iterations[1]["score"], {"kind": "mate", "value": 3})
        self.assertEqual(iterations[1]["pv"], ["c7c6", "g5f6"])
        self.assertEqual(PROFILE_CURRENT, "current")


if __name__ == "__main__":
    unittest.main()
