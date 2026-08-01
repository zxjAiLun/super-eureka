import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import chess

from run_external_validation import (
    DEFAULT_BOOKS_MANIFEST,
    DEFAULT_CORPUS,
    DEFAULT_SOURCES,
    CorpusIntegrityError,
    load_corpus,
    parse_info,
    score_rank,
    Score,
)


class ExternalValidationTests(unittest.TestCase):
    def test_pinned_corpus_and_provenance(self):
        cases, metadata = load_corpus(DEFAULT_CORPUS, DEFAULT_SOURCES, DEFAULT_BOOKS_MANIFEST)
        self.assertEqual(len(cases), 32)
        self.assertEqual(metadata["group_counts"], {"tactical": 8, "mate": 8, "promotion": 8, "endgame": 8})
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


if __name__ == "__main__":
    unittest.main()
