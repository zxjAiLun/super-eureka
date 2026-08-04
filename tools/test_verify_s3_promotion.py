import tempfile
import unittest
from pathlib import Path

import verify_s3_promotion as promotion


class S3PromotionVerifierTests(unittest.TestCase):
    def test_candidate_result_uses_candidate_color(self):
        self.assertEqual(promotion.candidate_result("White", "1-0"), "win")
        self.assertEqual(promotion.candidate_result("Black", "1-0"), "loss")
        self.assertEqual(promotion.candidate_result("Black", "0-1"), "win")
        self.assertEqual(promotion.candidate_result("White", "1/2-1/2"), "draw")

    def test_manager_log_accepts_early_h1_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            stdout = directory / "manager.stdout.log"
            stderr = directory / "manager.stderr.log"
            stdout.write_text(
                "\n".join(
                    [
                        "Started game 1 of 1000 (CurrentFinal vs Current)",
                        "Finished game 1 (CurrentFinal vs Current): 1-0 {White mates}",
                        "Started game 2 of 1000 (Current vs CurrentFinal)",
                        "Finished game 2 (Current vs CurrentFinal): 1/2-1/2 {Draw}",
                        "Score of CurrentFinal vs Current: 1 - 0 - 1  [0.750] 2",
                        "SPRT: llr 3.01, lbound -2.94, ubound 2.94 - H1 was accepted",
                        "Finished match",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            stderr.write_text(promotion.EXPECTED_STDERR + "\n", encoding="utf-8")
            summary = promotion.validate_manager_logs(stdout, stderr, 2)
            self.assertEqual(summary["sprt_decision"], "H1_ACCEPTED")
            self.assertEqual(summary["manager_score"]["candidate_wins"], 1)
            self.assertEqual(summary["manager_score"]["draws"], 1)


if __name__ == "__main__":
    unittest.main()
