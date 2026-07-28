import json
from pathlib import Path
import tempfile
import unittest

from run_fastchess import (
    build_fastchess_command,
    classify_result,
    ensure_book,
    run,
    sha384_sri,
    parser,
)


class FastchessWrapperTests(unittest.TestCase):
    @staticmethod
    def profile():
        return {
            "baseline": {"label": "Current", "search_profile": "current"},
            "candidate": {
                "label": "Aspiration",
                "search_profile": "current-aspiration",
            },
            "book_format": "pgn",
            "book": "fixture",
            "opening_order": "random",
            "opening_plies": 16,
            "seed": 7,
            "time_control": "10+0.1",
            "hash_mb": 64,
            "threads": 1,
            "rounds": 200,
            "concurrency": 1,
            "sprt": {"elo0": 0, "elo1": 5, "alpha": 0.05, "beta": 0.05, "model": "logistic"},
        }

    def test_command_keeps_profiles_and_fastchess_owns_games(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = build_fastchess_command(
                root / "fastchess.exe",
                root / "engine.exe",
                root / "engine.exe",
                self.profile(),
                root / "book.pgn",
                root / "results",
                "sprt",
            )

            self.assertIn("args=--profile current", command)
            self.assertIn("args=--profile current-aspiration", command)
            self.assertIn("option.Hash=64", command)
            self.assertIn("-repeat", command)
            self.assertIn("-sprt", command)
            self.assertNotIn("-games", command)
            self.assertNotIn("-concurrency 1", command)

    def test_classify_sprt_boundaries_without_reimplementing_statistics(self):
        self.assertEqual(
            classify_result("SPRT: H1 was accepted", "", 0, "sprt"),
            ("PASS", "fastchess-sprt"),
        )
        self.assertEqual(
            classify_result("SPRT: H0 was rejected", "", 0, "sprt"),
            ("PASS", "fastchess-sprt"),
        )
        self.assertEqual(
            classify_result("SPRT: H1 was rejected", "", 0, "sprt"),
            ("REJECTED", "fastchess-sprt"),
        )
        self.assertEqual(
            classify_result("", "", 0, "sprt"),
            ("INCONCLUSIVE", "fastchess-sprt-no-boundary"),
        )
        self.assertEqual(
            classify_result("", "fatal", 1, "sprt"),
            ("REJECTED", "fastchess-exit-1"),
        )
        self.assertEqual(
            classify_result("Finished match", "", 0, "fixed"),
            ("INCONCLUSIVE", "fixed-games-no-sprt-decision"),
        )

    def test_book_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            book = root / "book.pgn"
            book.write_bytes(b"[Event \"fixture\"]\n\n1. e4 *\n")
            expected = sha384_sri(book)
            manifest = root / "books.json"
            manifest.write_text(
                json.dumps(
                    {
                        "books": {
                            "fixture": {
                                "format": "pgn",
                                "content_filename": "book.pgn",
                                "archive_url": "https://example.invalid/book.zip",
                                "content_sha384_base64": expected,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = ensure_book(manifest, "fixture", book, root / "cache", False)
            self.assertTrue(result["verified"])
            self.assertEqual(result["actual_content_sha384_base64"], expected)

            book.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                ensure_book(manifest, "fixture", book, root / "cache", False)

    def test_dry_run_manifest_records_commands_and_effective_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = root / "engine.exe"
            engine.write_bytes(b"engine fixture")
            book = root / "book.pgn"
            book.write_bytes(b"[Event \"fixture\"]\n\n1. e4 *\n")
            book_manifest = root / "books.json"
            book_manifest.write_text(
                json.dumps(
                    {
                        "books": {
                            "fixture": {
                                "format": "pgn",
                                "content_filename": "book.pgn",
                                "archive_url": "https://example.invalid/book.zip",
                                "content_sha384_base64": sha384_sri(book),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            profile_config = root / "profiles.json"
            profile_config.write_text(
                json.dumps({"profiles": {"fixture": self.profile()}}),
                encoding="utf-8",
            )
            args = parser().parse_args(
                [
                    "--engine-a",
                    str(engine),
                    "--engine-b",
                    str(engine),
                    "--sha-a",
                    "base",
                    "--sha-b",
                    "candidate",
                    "--profile-name",
                    "fixture",
                    "--profile-config",
                    str(profile_config),
                    "--book-manifest",
                    str(book_manifest),
                    "--book-path",
                    str(book),
                    "--output-dir",
                    str(root / "results"),
                    "--dry-run",
                ]
            )
            manifest = run(args)

            self.assertEqual(manifest["status"], "DRY_RUN")
            self.assertEqual(manifest["engine_a_baseline"]["search_profile"], "current")
            self.assertEqual(
                manifest["engine_b_candidate"]["search_profile"], "current-aspiration"
            )
            self.assertEqual(manifest["opening_book"]["verified"], True)
            self.assertIn("args=--profile current", manifest["command"])
            self.assertIn("args=--profile current-aspiration", manifest["command"])
            self.assertTrue((root / "results" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
