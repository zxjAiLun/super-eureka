import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

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

    def test_dry_run_does_not_create_requested_formal_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            requested_output = Path(temporary) / "formal-run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(promotion.REPO_ROOT / "tools" / "run_s3_promotion.py"),
                    "--dry-run",
                    "--output-dir",
                    str(requested_output),
                ],
                cwd=promotion.REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if promotion.git_output("rev-parse", "HEAD") != promotion.FROZEN_ENGINE_GIT_SHA:
                # Once production promotion advances the source beyond the
                # historical S3 run, the frozen launcher must fail closed
                # rather than silently prepare a different experiment.
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("source differs from frozen S3-FINAL", result.stderr)
                self.assertFalse(requested_output.exists())
                return
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(requested_output.exists())
            self.assertIn('"status": "DRY_RUN"', result.stdout)

    def test_nonempty_formal_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "formal-run"
            output_dir.mkdir()
            (output_dir / "marker.txt").write_text("real match artifact\n", encoding="utf-8")
            with self.assertRaises(promotion.PromotionError):
                promotion.refuse_reused_output(output_dir)


if __name__ == "__main__":
    unittest.main()
