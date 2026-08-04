#!/usr/bin/env python3
"""Validate and register an opening set (section 8).

Usage:
    python scripts/register_openings.py <openings.epd> <manifest.json>

Validation performed with python-chess:
- every line parses as an EPD position,
- every position has at least one legal move (non-terminal),
- position keys are unique,
- the file SHA-256 and position count match the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make ``arena`` importable when run from a source checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess  # noqa: E402

from chessarena.config import get_settings  # noqa: E402
from chessarena.db import make_engine, make_session_factory  # noqa: E402
from chessarena.models import OpeningSet  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epd_file", type=Path)
    parser.add_argument("manifest_file", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow re-registering an existing opening set",
    )
    args = parser.parse_args()

    epd_file = args.epd_file.resolve()
    manifest_file = args.manifest_file.resolve()
    if not epd_file.exists():
        sys.exit(f"error: opening file not found: {epd_file}")
    if not manifest_file.exists():
        sys.exit(f"error: manifest not found: {manifest_file}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for key in ("schema_version", "opening_set_id", "format", "count", "sha256",
                "unique_position_keys", "non_terminal", "created_utc"):
        if key not in manifest:
            sys.exit(f"error: manifest missing key {key!r}")

    actual_sha = sha256_file(epd_file)
    if actual_sha != manifest["sha256"]:
        sys.exit(
            f"error: SHA mismatch: manifest {manifest['sha256']} actual {actual_sha}"
        )

    lines = read_positions(epd_file)
    validate(lines, manifest["sha256"], manifest["count"])

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
            existing = OpeningSet(
                opening_set_id=manifest["opening_set_id"],
                file_path=str(epd_file),
                sha256=actual_sha,
                position_count=len(lines),
                manifest=manifest,
                enabled=True,
            )
            session.add(existing)
        else:
            existing.file_path = str(epd_file)
            existing.sha256 = actual_sha
            existing.position_count = len(lines)
            existing.manifest = manifest
            existing.enabled = True
        session.commit()
    print(f"registered opening set {manifest['opening_set_id']}: {len(lines)} positions")
    print(f"  sha256: {actual_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
