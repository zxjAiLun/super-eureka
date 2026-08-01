import os
import hashlib
import json
from pathlib import Path
import textwrap
import tempfile
import unittest
from unittest.mock import patch

import chess

from build_external_validation_v1 import DEFAULT_CACHE, verify_snapshot

from run_external_validation import (
    DEFAULT_BOOKS_MANIFEST,
    DEFAULT_CORPUS,
    DEFAULT_PROJECT_CORPUS,
    DEFAULT_SOURCES,
    CorpusIntegrityError,
    Case,
    EngineFailure,
    EngineSession,
    build_parser,
    load_corpus,
    load_project_corpus,
    parse_info,
    score_rank,
    Score,
)


class ExternalValidationTests(unittest.TestCase):
    @staticmethod
    def _fake_engine(root: Path) -> Path:
        script = root / "fake_external_engine.py"
        script.write_text(
            textwrap.dedent(
                """
                import os
                import sys
                import time

                profile = sys.argv[sys.argv.index("--profile") + 1]
                mode = os.environ.get("D111_FAKE_MODE", "normal")
                reported = "wrong-profile" if mode == "profile-mismatch" else profile

                def emit(line):
                    print(line, flush=True)

                for raw in sys.stdin:
                    line = raw.strip()
                    if line == "uci":
                        emit("id name D111Fake")
                        emit("id author tests")
                        emit("info string search profile " + reported)
                        emit("uciok")
                    elif line == "isready":
                        emit("readyok")
                    elif line.startswith("go"):
                        if mode == "timeout":
                            print("timeout diagnostic", file=sys.stderr, flush=True)
                            while True:
                                emit("info depth 1 score cp 0 pv e2e4")
                                time.sleep(0.01)
                        if mode == "short-depth":
                            emit("info depth 0 score cp 0 pv e2e4")
                        elif mode == "empty-pv":
                            emit("info depth 1 score cp 0")
                        elif mode == "pv-mismatch":
                            emit("info depth 1 score cp 0 pv d2d4")
                        else:
                            emit("info depth 1 score cp 0 pv e2e4")
                        if mode == "illegal":
                            emit("bestmove a1a8")
                        else:
                            emit("bestmove e2e4")
                        if mode == "abnormal-exit":
                            print("child panic", file=sys.stderr, flush=True)
                            raise SystemExit(17)
                    elif line == "quit":
                        break
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return script

    def _run_fake_search(self, root: Path, mode: str, timeout_s: float = 0.5):
        script = self._fake_engine(root)
        case = Case(
            "fake-startpos",
            "closedpos",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "fake",
            1,
            1,
        )
        with patch.dict(os.environ, {"D111_FAKE_MODE": mode}):
            with EngineSession(script, "current", 16, timeout_s) as session:
                session.handshake()
                return session.search(case)

    def test_pinned_corpus_and_provenance(self):
        cases, metadata = load_corpus(DEFAULT_CORPUS, DEFAULT_SOURCES, DEFAULT_BOOKS_MANIFEST)
        self.assertEqual(len(cases), 32)
        self.assertEqual(
            metadata["group_counts"],
            {"closedpos": 8, "stalemate-stress": 8, "endgames-a": 8, "endgames-cdb": 8},
        )
        self.assertEqual(
            metadata["snapshot_sha256"],
            hashlib.sha256(DEFAULT_CORPUS.read_bytes()).hexdigest(),
        )
        for source in metadata["source_refs"].values():
            self.assertEqual(source["source_repository"], "https://github.com/official-stockfish/books")
            self.assertEqual(source["license"], "CC0-1.0")
            self.assertTrue(source["archive_url"].endswith(".epd.zip"))
            self.assertTrue(source["raw_content_sha384_base64"])

    def test_snapshot_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            copied = root / "external.epd"
            copied.write_bytes(DEFAULT_CORPUS.read_bytes() + b"\n")
            metadata = json.loads(DEFAULT_SOURCES.read_text(encoding="utf-8"))
            sources = root / "sources.json"
            sources.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(CorpusIntegrityError):
                load_corpus(copied, sources, DEFAULT_BOOKS_MANIFEST)

    def test_info_score_and_rank_are_typed(self):
        parsed = parse_info("info depth 4 score mate 3 pv e2e4 e7e5")
        self.assertEqual(parsed, (4, Score("mate", 3), ("e2e4", "e7e5")))
        self.assertGreater(score_rank(Score("mate", 2)), score_rank(Score("cp", 250)))
        self.assertGreater(score_rank(Score("cp", 350)), score_rank(Score("cp", -200)))

    def test_external_fens_have_legal_move_sets(self):
        cases, _metadata = load_corpus(DEFAULT_CORPUS, DEFAULT_SOURCES, DEFAULT_BOOKS_MANIFEST)
        self.assertTrue(all(any(chess.Board(case.fen).legal_moves) for case in cases))

    def test_project_corpus_loader_and_depth_override_cli(self):
        cases, metadata = load_project_corpus(DEFAULT_PROJECT_CORPUS)
        self.assertEqual(len(cases), 23)
        self.assertEqual(metadata["corpus_id"], "d1.10-project-curated-v2")
        self.assertIn("d10-promotion-chain-white", {case.case_id for case in cases})
        args = build_parser().parse_args(
            [
                "--engine",
                "engine",
                "--corpus-kind",
                "project",
                "--depth-override",
                "7",
            ]
        )
        self.assertEqual(args.corpus_kind, "project")
        self.assertEqual(args.depth_override, 7)

    def test_source_lines_reproduce_snapshot_when_cache_is_prepared(self):
        if not all(DEFAULT_CACHE.joinpath(name).is_file() for name in (
            "closedpos.epd",
            "stalemates_200d30_v1.epd",
            "endgames.epd",
            "endgames_cdb95105.epd",
        )):
            self.skipTest("pinned local book cache is not prepared")
        result = verify_snapshot()
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["case_count"], 32)

    def test_abnormal_exit_after_legal_bestmove_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(EngineFailure) as context:
                self._run_fake_search(Path(temp), "abnormal-exit")
        self.assertIn("code 17", str(context.exception))
        self.assertIn("child panic", str(context.exception))

    def test_profile_mismatch_fails_before_search(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(EngineFailure) as context:
                self._run_fake_search(Path(temp), "profile-mismatch")
        self.assertIn("missing profile identity", str(context.exception))

    def test_search_result_contract_failures_are_rejected(self):
        for mode in ("illegal", "empty-pv", "pv-mismatch", "short-depth"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(EngineFailure):
                    self._run_fake_search(Path(temp), mode)

    def test_timeout_preserves_stderr_diagnostics_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(EngineFailure) as context:
                self._run_fake_search(Path(temp), "timeout", timeout_s=0.08)
        self.assertIn("total deadline expired", str(context.exception))
        self.assertIn("timeout diagnostic", str(context.exception))


if __name__ == "__main__":
    unittest.main()
