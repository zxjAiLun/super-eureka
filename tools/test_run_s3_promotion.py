import tempfile
import unittest
from pathlib import Path

import run_s3_promotion as promotion


class S3PromotionTests(unittest.TestCase):
    def test_selection_manifest_is_a_pinned_disjoint_500_pair_slice(self):
        selection, source, selected = promotion.load_selection(promotion.DEFAULT_SELECTION)
        self.assertEqual(selection["suite_id"], "s3-promotion-openings-v1")
        self.assertEqual(len(selected), 500)
        self.assertEqual(selection["selection"]["source_line_start"], 1000)
        self.assertEqual(selection["selection"]["source_line_end"], 1499)
        self.assertEqual(source.name, "d1.14-openings-v1.epd")

    def test_build_command_is_candidate_first_and_uses_declared_sprt(self):
        command = promotion.build_command(
            Path("manager.exe"), Path("engine.exe"), Path("openings.epd"), Path("out")
        )
        self.assertLess(command.index("name=CurrentFinal"), command.index("name=Current"))
        self.assertIn("arg=current-final", command)
        self.assertIn("arg=current", command)
        self.assertIn("elo0=20", command)
        self.assertIn("elo1=60", command)
        self.assertIn("alpha=0.05", command)
        self.assertIn("beta=0.05", command)
        self.assertIn("-rounds", command)
        self.assertEqual(command[command.index("-rounds") + 1], "1000")
        self.assertEqual(command[command.index("-concurrency") + 1], "1")

    def test_runtime_opening_output_is_hashable_and_exact(self):
        selection, _, selected = promotion.load_selection(promotion.DEFAULT_SELECTION)
        self.assertEqual(selection["selection"]["count"], promotion.PAIRS)
        with tempfile.TemporaryDirectory() as temporary:
            runtime, digest = promotion.write_runtime_openings(Path(temporary), selected)
            self.assertTrue(runtime.is_file())
            self.assertEqual(digest, promotion.sha256_file(runtime))
            self.assertEqual(runtime.read_text(encoding="utf-8").splitlines(), selected)


if __name__ == "__main__":
    unittest.main()
