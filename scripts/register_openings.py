#!/usr/bin/env python3
"""Validate and register an opening set (section 8 / P4.F1 Phase C).

Usage:
    python scripts/register_openings.py <openings.file> <manifest.json>

Supports two formats (from the manifest ``format`` key):
- ``epd``: every line parses as a non-terminal EPD position; unique keys.
- ``pgn``: a multi-game book (e.g. the official Stockfish 8moves_v3 suite);
  every game must parse and have a legal mainline; the position count is the
  number of games.

The file SHA-256 and position count must match the manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make ``arena`` importable when run from a source checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from chessarena.config import get_settings  # noqa: E402
from chessarena.db import make_engine, make_session_factory  # noqa: E402
from chessarena.models import OpeningSet  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha384_base64(path: Path) -> str:
    h = hashlib.sha384()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def _count_pgn_games(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            list(game.mainline_moves())  # must be legal/parsable
            count += 1
    return count


def read_positions(path: Path) -> list[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return lines


def validate(lines: list[str], expected_sha: str, expected_count: int) -> None:
    if len(lines) != expected_count:
        sys.exit(
            f"error: position count {len(lines)} does not match manifest "
            f"{expected_count}"
        )
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        fen = line.split(";")[0].strip()
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            sys.exit(f"error: line {idx + 1} unparsable: {line!r} ({exc})")
        if not list(board.legal_moves):
            sys.exit(f"error: line {idx + 1} is a terminal position: {line!r}")
        key = board.fen()
        if key in seen:
            sys.exit(f"error: duplicate position key at line {idx + 1}")
        seen.add(key)


def _adapt_catalog_manifest(catalog_path: Path, book_id: str,
                            opening_file: Path) -> dict:
    """Adapt the official books catalog (books/manifest.json) into an Arena
    opening-set manifest (P4.F1 P1-2).

    Maps: book-id -> opening_set_id, format, expected_positions -> count,
    expected_plies -> default_plies, and preserves source provenance.  The
    actual file is verified against the catalog-pinned SHA-384 before any
    Arena SHA-256 is recorded.
    """
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = (catalog.get("books") or {}).get(book_id)
    if entry is None:
        sys.exit(f"error: catalog has no book {book_id!r}")

    pinned384 = entry.get("content_sha384_base64")
    if pinned384:
        actual384 = sha384_base64(opening_file)
        if actual384 != pinned384:
            sys.exit(
                "error: SHA-384 mismatch against official catalog "
                f"(pinned {pinned384}, actual {actual384})"
            )

    fmt = entry["format"]
    if fmt not in ("epd", "pgn"):
        sys.exit(f"error: unsupported format {fmt!r}")
    default_plies = entry.get("expected_plies") if fmt == "pgn" else None
    return {
        "schema_version": 1,
        "opening_set_id": book_id,
        "format": fmt,
        "count": entry["expected_positions"],
        "sha256": sha256_file(opening_file),
        "unique_position_keys": None,
        "non_terminal": None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "default_plies": default_plies,
        "source_repository": entry.get("source_repository"),
        "source_ref": entry.get("source_ref"),
        "license": entry.get("license"),
        "content_sha384_base64": pinned384,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opening_file", type=Path)
    parser.add_argument("manifest_file", type=Path, nargs="?",
                        help="legacy Arena opening-set manifest (JSON)")
    parser.add_argument(
        "--catalog", type=Path,
        help="official book catalog (books/manifest.json) to adapt from",
    )
    parser.add_argument(
        "--book-id",
        help="catalog book id to register (required with --catalog)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow re-registering an existing opening set",
    )
    args = parser.parse_args()

    opening_file = args.opening_file.resolve()
    if not opening_file.exists():
        sys.exit(f"error: opening file not found: {opening_file}")

    if args.catalog:
        manifest = _adapt_catalog_manifest(args.catalog, args.book_id,
                                           opening_file)
    else:
        if args.manifest_file is None:
            sys.exit("error: provide a legacy manifest file or --catalog")
        manifest_file = args.manifest_file.resolve()
        if not manifest_file.exists():
            sys.exit(f"error: manifest not found: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for key in ("schema_version", "opening_set_id", "format", "count",
                    "sha256", "unique_position_keys", "non_terminal",
                    "created_utc"):
            if key not in manifest:
                sys.exit(f"error: manifest missing key {key!r}")

    fmt = manifest["format"]
    if fmt not in ("epd", "pgn"):
        sys.exit(f"error: unsupported format {fmt!r}")

    actual_sha = sha256_file(opening_file)
    if actual_sha != manifest["sha256"]:
        sys.exit(
            f"error: SHA mismatch: manifest {manifest['sha256']} actual {actual_sha}"
        )

    if fmt == "pgn":
        count = _count_pgn_games(opening_file)
        if count != manifest["count"]:
            sys.exit(
                f"error: PGN game count {count} does not match manifest "
                f"{manifest['count']}"
            )
    else:
        lines = read_positions(opening_file)
        validate(lines, manifest["sha256"], manifest["count"])
        count = len(lines)

    settings = get_settings()
    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        existing = (
            session.query(OpeningSet)
            .filter(OpeningSet.opening_set_id == manifest["opening_set_id"])
            .first()
        )
        if existing is not None and not args.overwrite:
            sys.exit(
                f"error: opening set {manifest['opening_set_id']} already "
                "registered (use --overwrite to force)"
            )
        if existing is None:
            source = manifest.get("source") or "{} @ {} ({})".format(
                manifest.get("source_repository") or "?",
                manifest.get("source_ref") or "?",
                manifest.get("license") or "?",
            )
            existing = OpeningSet(
                opening_set_id=manifest["opening_set_id"],
                file_path=str(opening_file),
                sha256=actual_sha,
                position_count=count,
                format=fmt,
                source=source,
                manifest=manifest,
                enabled=True,
            )
            session.add(existing)
        else:
            existing.file_path = str(opening_file)
            existing.sha256 = actual_sha
            existing.position_count = count
            existing.format = fmt
            existing.source = manifest.get("source")
            existing.manifest = manifest
            existing.enabled = True
        session.commit()
    print(f"registered opening set {manifest['opening_set_id']}: {count} positions ({fmt})")
    print(f"  sha256: {actual_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
