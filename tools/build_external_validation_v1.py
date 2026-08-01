"""Verify the committed D1.11 snapshot against pinned local source files.

This tool is deliberately offline. It does not download anything and is not
part of the runtime UCI runner. The caller must first prepare the four source
EPD files under ``books/cache``; their raw SHA-384 values are checked against
``books/manifest.json`` before source-line selection is compared with the
committed D1.11 snapshot.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import chess

from run_external_validation import (
    DEFAULT_BOOKS_MANIFEST,
    DEFAULT_CORPUS,
    DEFAULT_SOURCES,
    CorpusIntegrityError,
    _load_json,
    load_corpus,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "books" / "cache"
EXPECTED_GROUP_SOURCES = {
    "closedpos": "stockfish-closedpos",
    "stalemate-stress": "stockfish-stalemates-200d30-v1",
    "endgames-a": "stockfish-endgames",
    "endgames-cdb": "stockfish-endgames-cdb95105",
}
EXPECTED_SOURCE_LINES = {
    "closedpos": [1, 2, 3, 4, 5, 6, 7, 8],
    "stalemate-stress": [3, 4, 6, 7, 9, 10, 11, 12],
    "endgames-a": [1, 2, 3, 4, 5, 6, 7, 8],
    "endgames-cdb": [2, 3, 5, 8, 9, 13, 20, 23],
}
CASES_PER_GROUP = 8


class SourceVerificationError(RuntimeError):
    """A local source file does not reproduce the committed snapshot."""


def _raw_sha384_base64(data: bytes) -> str:
    return base64.b64encode(hashlib.sha384(data).digest()).decode("ascii")


def _source_rows(path: Path, source_lines: list[int]) -> list[tuple[int, str]]:
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise SourceVerificationError(f"cannot read source file {path}: {exc}") from exc
    rows: list[tuple[int, str]] = []
    for line_number in source_lines:
        if line_number <= 0 or line_number > len(raw_lines):
            raise SourceVerificationError(f"{path}: source line {line_number} is out of range")
        raw_line = raw_lines[line_number - 1]
        try:
            text = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise SourceVerificationError(f"{path}:{line_number}: not UTF-8: {exc}") from exc
        if not text or text.startswith("#"):
            raise SourceVerificationError(f"{path}:{line_number}: selected line is empty/comment")
        try:
            board = chess.Board(text)
        except ValueError:
            raise SourceVerificationError(f"{path}:{line_number}: selected line is not a FEN")
        if board.is_valid():
            rows.append((line_number, text))
        else:
            raise SourceVerificationError(f"{path}:{line_number}: selected FEN is invalid")
    if len(rows) != CASES_PER_GROUP:
        raise SourceVerificationError(
            f"{path}: expected {CASES_PER_GROUP} valid FEN rows, got {len(rows)}"
        )
    return rows


def verify_snapshot(
    corpus_path: Path = DEFAULT_CORPUS,
    sources_path: Path = DEFAULT_SOURCES,
    books_manifest_path: Path = DEFAULT_BOOKS_MANIFEST,
    cache_dir: Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    try:
        cases, _metadata = load_corpus(corpus_path, sources_path, books_manifest_path)
    except CorpusIntegrityError as exc:
        raise SourceVerificationError(str(exc)) from exc

    sources = _load_json(sources_path)
    selection = sources.get("selection")
    if not isinstance(selection, dict) or selection.get("groups") != EXPECTED_GROUP_SOURCES:
        raise SourceVerificationError("selection groups do not match D1.11 v1 rules")
    if selection.get("cases") != 32 or selection.get("depth") != 4:
        raise SourceVerificationError("selection cardinality/depth metadata changed")
    if selection.get("source_lines") != EXPECTED_SOURCE_LINES:
        raise SourceVerificationError("selection source lines do not match D1.11 v1 rules")

    books_manifest = _load_json(books_manifest_path)
    books = books_manifest.get("books")
    if not isinstance(books, dict):
        raise SourceVerificationError("books manifest has no books object")

    verified_sources: dict[str, dict[str, Any]] = {}
    expected_rows: list[tuple[str, str, str, str, int, int]] = []
    for group, book_id in EXPECTED_GROUP_SOURCES.items():
        entry = books.get(book_id)
        if not isinstance(entry, dict):
            raise SourceVerificationError(f"missing book manifest entry: {book_id}")
        content_filename = entry.get("content_filename")
        expected_hash = entry.get("raw_content_sha384_base64")
        if not isinstance(content_filename, str) or not isinstance(expected_hash, str):
            raise SourceVerificationError(f"incomplete hash metadata for {book_id}")
        source_path = cache_dir / content_filename
        try:
            raw = source_path.read_bytes()
        except OSError as exc:
            raise SourceVerificationError(
                f"missing local source {source_path}; run prepare_books.py first: {exc}"
            ) from exc
        actual_hash = _raw_sha384_base64(raw)
        if actual_hash != expected_hash:
            raise SourceVerificationError(
                f"raw SHA-384 mismatch for {book_id}: expected {expected_hash}, got {actual_hash}"
            )
        source_rows = _source_rows(source_path, EXPECTED_SOURCE_LINES[group])
        for index, (source_line, fen) in enumerate(source_rows, 1):
            expected_rows.append(
                (f"d11-{group}-{index:04d}", group, fen, book_id, source_line, 4)
            )
        verified_sources[book_id] = {
            "path": str(source_path.resolve()),
            "raw_content_sha384_base64": actual_hash,
            "source_lines": [line for line, _fen in source_rows],
        }

    actual_rows = [
        (case.case_id, case.group, case.fen, case.source_book_id, case.source_line, case.depth)
        for case in cases
    ]
    if actual_rows != expected_rows:
        raise SourceVerificationError(
            "committed snapshot differs from deterministic source extraction"
        )

    return {
        "decision": "PASS",
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "case_count": len(actual_rows),
        "group_counts": {group: CASES_PER_GROUP for group in EXPECTED_GROUP_SOURCES},
        "sources": verified_sources,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--books-manifest", type=Path, default=DEFAULT_BOOKS_MANIFEST)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_snapshot(
            corpus_path=args.corpus,
            sources_path=args.sources,
            books_manifest_path=args.books_manifest,
            cache_dir=args.cache_dir,
        )
    except (SourceVerificationError, OSError, ValueError) as exc:
        print(json.dumps({"decision": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
