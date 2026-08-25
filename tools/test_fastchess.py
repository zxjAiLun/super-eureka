import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from run_fastchess import (
    FastchessError,
    build_fastchess_command,
    classify_result,
    ensure_book,
    _download_book,
    probe_engine_identity,
    run,
    sha384_sri,
    sha384_sri_bytes,
    normalize_line_endings,
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

    def test_active_2m1s_profile_is_candidate_first_ready(self):
        config = json.loads(
            (Path(__file__).with_name("fastchess_profiles.json")).read_text(
                encoding="utf-8"
            )
        )
        profile = config["profiles"]["s2-current-vs-aspiration-2m1s"]
        self.assertEqual(profile["status"], "active")
        self.assertEqual(profile["time_control"], "2:00+1")
        self.assertEqual(profile["concurrency"], 1)
        self.assertEqual(profile["rounds"], 1000)
        self.assertEqual(profile["baseline"]["search_profile"], "current")
        self.assertEqual(
            profile["candidate"]["search_profile"], "current-aspiration"
        )
        self.assertEqual(profile["seed"], 2026073001)

    def test_command_keeps_profiles_and_fastchess_owns_games(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = build_fastchess_command(
                root / "fastchess.exe",
                root / "engine.exe",
                root / "engine.exe",
                self.profile(),
                root / "book.pgn",
                "epd",
                root / "results",
                "sprt",
            )

            self.assertIn("args=--profile current", command)
            self.assertIn("args=--profile current-aspiration", command)
            self.assertIn("option.Hash=64", command)
            self.assertIn("format=epd", command)
            self.assertIn("append=false", command)
            self.assertIn("timeleft=true", command)
            self.assertIn("latency=true", command)
            self.assertIn("-repeat", command)
            self.assertIn("-sprt", command)
            self.assertNotIn("-games", command)
            self.assertNotIn("-concurrency 1", command)
            first_engine = command.index("-engine")
            second_engine = command.index("-engine", first_engine + 1)
            first_block = command[first_engine:second_engine]
            second_block = command[second_engine:]
            self.assertIn("name=Aspiration", first_block)
            self.assertIn("args=--profile current-aspiration", first_block)
            self.assertIn("name=Current", second_block)
            self.assertIn("args=--profile current", second_block)

    def test_uci_identity_probe_records_reported_profile(self):
        # Cross-platform release binary name (.exe only on Windows).
        engine = Path("target/release") / (
            "eureka.exe" if os.name == "nt" else "eureka"
        )
        self.assertTrue(
            engine.is_file(), f"release engine missing: {engine}"
        )
        identity = probe_engine_identity([
            str(engine), "--profile", "current-aspiration"
        ])
        self.assertEqual(identity["reported_search_profile"], "current-aspiration")
        self.assertTrue(identity["id_name"].startswith("Eureka"))
        self.assertEqual(identity["id_author"], "zxjAiLun")

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
            ("INCONCLUSIVE", "fastchess-exit-1"),
        )
        self.assertEqual(
            classify_result("Finished match", "", 0, "fixed"),
            ("NOT_APPLICABLE", "fixed-games-no-sprt-decision"),
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

    def test_download_verifies_before_replacing_existing_book(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "book.zip"
            extracted = b"line one\r\nline two\rline three\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/book.pgn", extracted)
            destination = root / "cache" / "book.pgn"
            destination.parent.mkdir()
            destination.write_bytes(b"known-good")
            entry = {
                "archive_url": archive_path.as_uri(),
                "archive_filename": archive_path.name,
                "content_filename": "book.pgn",
                "content_sha384_base64": sha384_sri_bytes(b"wrong"),
            }
            with self.assertRaisesRegex(FastchessError, "hash mismatch"):
                _download_book(entry, destination)
            self.assertEqual(destination.read_bytes(), b"known-good")

    def test_download_accepts_normalized_lone_cr_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "book.zip"
            extracted = b"line one\r\nline two\rline three\n"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("book.pgn", extracted)
            destination = root / "cache" / "book.pgn"
            entry = {
                "archive_url": archive_path.as_uri(),
                "archive_filename": archive_path.name,
                "content_filename": "book.pgn",
                "content_sha384_base64": sha384_sri_bytes(extracted),
                "upstream_normalized_sri": sha384_sri_bytes(
                    normalize_line_endings(extracted)
                ),
            }
            _download_book(entry, destination)
            self.assertEqual(destination.read_bytes(), extracted)

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

            self.assertEqual(manifest["execution_status"], "PREPARED")
            self.assertEqual(manifest["decision"], "NOT_APPLICABLE")
            self.assertEqual(manifest["sprt_subject"], "engine_b_candidate")
            self.assertEqual(manifest["fastchess_player_1"], "engine_b_candidate")
            self.assertEqual(manifest["fastchess_player_2"], "engine_a_baseline")
            self.assertEqual(manifest["games_max"], 400)
            self.assertIsNone(manifest["games_completed"])
            self.assertIsNone(manifest["stopped_early"])
            self.assertEqual(manifest["engine_thread_model"], "single-threaded")
            self.assertEqual(manifest["engine_a_baseline"]["search_profile"], "current")
            self.assertEqual(
                manifest["engine_b_candidate"]["search_profile"], "current-aspiration"
            )
            self.assertEqual(manifest["opening_book"]["verified"], True)
            self.assertIn("args=--profile current", manifest["command"])
            self.assertIn("args=--profile current-aspiration", manifest["command"])
            self.assertTrue((root / "results" / "manifest.json").is_file())

    def test_non_active_profile_is_fail_closed_but_dry_run_can_inspect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "profiles.json"
            profile = self.profile()
            profile["status"] = "historical-cancelled"
            config.write_text(json.dumps({"profiles": {"old": profile}}), encoding="utf-8")
            args = parser().parse_args([
                "--engine-a", str(root / "a.exe"), "--engine-b", str(root / "b.exe"),
                "--sha-a", "a", "--sha-b", "b", "--profile-name", "old",
                "--profile-config", str(config), "--output-dir", str(root / "run"),
            ])
            with self.assertRaisesRegex(FastchessError, "not active"):
                run(args)
            self.assertFalse((root / "run").exists())

    def test_reused_output_dir_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = root / "engine.exe"
            engine.write_bytes(b"engine fixture")
            book = root / "book.pgn"
            book.write_bytes(b"[Event \"fixture\"]\n\n1. e4 *\n")
            book_manifest = root / "books.json"
            book_manifest.write_text(
                json.dumps(
                    {"books": {"fixture": {
                        "format": "pgn",
                        "content_filename": "book.pgn",
                        "archive_url": "https://example.invalid/book.zip",
                        "content_sha384_base64": sha384_sri(book),
                    }}}
                ),
                encoding="utf-8",
            )
            profile_config = root / "profiles.json"
            profile_config.write_text(
                json.dumps({"profiles": {"fixture": self.profile()}}), encoding="utf-8"
            )
            output_dir = root / "results"
            output_dir.mkdir()
            (output_dir / "games.pgn").write_text("existing", encoding="utf-8")
            args = parser().parse_args([
                "--engine-a", str(engine), "--engine-b", str(engine),
                "--sha-a", "base", "--sha-b", "candidate",
                "--profile-name", "fixture", "--profile-config", str(profile_config),
                "--book-manifest", str(book_manifest), "--book-path", str(book),
                "--output-dir", str(output_dir), "--dry-run",
            ])
            with self.assertRaisesRegex(FastchessError, "already contains"):
                run(args)

    def test_missing_fastchess_is_launch_failure_not_candidate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = root / "engine.exe"
            engine.write_bytes(b"engine fixture")
            book = root / "book.pgn"
            book.write_bytes(b"[Event \"fixture\"]\n\n1. e4 *\n")
            book_manifest = root / "books.json"
            book_manifest.write_text(
                json.dumps({"books": {"fixture": {
                    "format": "pgn", "content_filename": "book.pgn",
                    "archive_url": "https://example.invalid/book.zip",
                    "content_sha384_base64": sha384_sri(book),
                }}}), encoding="utf-8"
            )
            profile_config = root / "profiles.json"
            profile_config.write_text(
                json.dumps({"profiles": {"fixture": self.profile()}}), encoding="utf-8"
            )
            args = parser().parse_args([
                "--fastchess", str(root / "missing-fastchess.exe"),
                "--engine-a", str(engine), "--engine-b", str(engine),
                "--sha-a", "base", "--sha-b", "candidate",
                "--profile-name", "fixture", "--profile-config", str(profile_config),
                "--book-manifest", str(book_manifest), "--book-path", str(book),
                "--output-dir", str(root / "results"),
            ])
            manifest = run(args)
            self.assertEqual(manifest["execution_status"], "LAUNCH_FAIL")
            self.assertEqual(manifest["decision"], "INCONCLUSIVE")
            self.assertNotEqual(manifest["decision"], "REJECTED")
            self.assertEqual(manifest["games_completed"], None)


if __name__ == "__main__":
    unittest.main()
