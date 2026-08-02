#!/usr/bin/env python3
"""Prepare and verify the pinned D1.14 opening suite.

The upstream UHO file is deliberately not committed.  Preparation validates
the complete extracted source, removes duplicate complete FENs, selects a
deterministic fixed-size reservoir, shuffles the selected positions with a
fixed seed, and records all provenance and hashes in a committed metadata
file.  ``--verify`` only needs the committed output and metadata, while
``--source`` additionally rechecks the upstream extracted file hash.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import chess


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "data" / "openings" / "d1.14-openings-v1.epd"
DEFAULT_METADATA = REPO_ROOT / "tests" / "data" / "openings" / "d1.14-openings-v1.json"
DEFAULT_COUNT = 4_852
DEFAULT_SMOKE_COUNT = 20
DEFAULT_SEED = 20260802
SOURCE_REPOSITORY = "https://github.com/official-stockfish/books"
SOURCE_REF = "65815ccdbc7727cd4f6aee252ba8f67fb740e92f"
SOURCE_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/official-stockfish/books/"
    f"{SOURCE_REF}/UHO_Lichess_4852_v1.epd.zip"
)
SOURCE_ARCHIVE_SHA256 = "4e298f11e8acfa106babe02968f2e61582145e7874c59284690b20b9650e0e07"
SOURCE_CONTENT_FILENAME = "UHO_Lichess_4852_v1.epd"
SOURCE_LICENSE = "CC0-1.0"


class OpeningError(RuntimeError):
    """Raised when the pinned opening contract is not satisfied."""


@dataclass(frozen=True)
class OpeningRecord:
    fen: str
    source_line: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OpeningError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_stored_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def canonical_fen(raw: str, source_line: int) -> str:
    fields = raw.strip().split()
    if len(fields) < 6:
        raise OpeningError(f"source line {source_line}: expected a complete six-field FEN")
    fen = " ".join(fields[:6])
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise OpeningError(f"source line {source_line}: invalid FEN: {fen}") from exc
    if not board.is_valid():
        raise OpeningError(f"source line {source_line}: FEN is not a legal position: {fen}")
    if board.is_game_over(claim_draw=False):
        raise OpeningError(f"source line {source_line}: opening position is already terminal: {fen}")
    # Keep the complete FEN contract, including an upstream en-passant field.
    return board.fen(en_passant="fen")


def read_source(path: Path) -> tuple[list[OpeningRecord], str, int]:
    digest = hashlib.sha256()
    unique: dict[str, int] = {}
    physical_lines = 0
    try:
        with path.open("rb") as source:
            for physical_lines, raw_line in enumerate(source, start=1):
                digest.update(raw_line)
                text = raw_line.decode("utf-8").strip()
                if not text or text.startswith("#"):
                    continue
                fen = canonical_fen(text, physical_lines)
                unique.setdefault(fen, physical_lines)
    except (OSError, UnicodeDecodeError) as exc:
        raise OpeningError(f"cannot parse source {path}: {exc}") from exc
    records = [OpeningRecord(fen, line) for fen, line in unique.items()]
    return records, digest.hexdigest(), physical_lines


def select_records(records: list[OpeningRecord], count: int, seed: int) -> list[OpeningRecord]:
    if count <= 0:
        raise OpeningError("selection count must be positive")
    if len(records) < count:
        raise OpeningError(f"source has only {len(records)} unique positions; need {count}")
    # Reservoir sampling avoids retaining a second list of all source records.
    rng = random.Random(seed)
    selected: list[OpeningRecord] = []
    for index, record in enumerate(records, start=1):
        if len(selected) < count:
            selected.append(record)
            continue
        replacement = rng.randrange(index)
        if replacement < count:
            selected[replacement] = record
    rng.shuffle(selected)
    return selected


def write_output(path: Path, records: Iterable[OpeningRecord]) -> bytes:
    data = "".join(f"{record.fen}\n" for record in records).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def metadata_for(
    output_path: Path,
    metadata_path: Path,
    selected: list[OpeningRecord],
    source_sha256: str,
    source_line_count: int,
    source_unique_count: int,
    output_data: bytes,
    seed: int,
    count: int,
    smoke_count: int,
    retrieved_utc: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite_id": "d1.14-openings-v1",
        "format": "epd",
        "selection": {
            "count": count,
            "smoke_count": smoke_count,
            "seed": seed,
            "method": "reservoir_sample_then_seeded_shuffle",
            "order": "committed_file_order",
            "selected_source_lines": [record.source_line for record in selected],
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "ref": SOURCE_REF,
            "archive_url": SOURCE_ARCHIVE_URL,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "content_filename": SOURCE_CONTENT_FILENAME,
            "content_sha256": source_sha256,
            "physical_line_count": source_line_count,
            "unique_position_count": source_unique_count,
            "license": SOURCE_LICENSE,
            "retrieved_utc": retrieved_utc,
        },
        "output": {
            "path": stored_path(output_path),
            "sha256": sha256_bytes(output_data),
            "byte_length": len(output_data),
            "position_count": len(selected),
        },
        "metadata_path": stored_path(metadata_path),
    }


def write_metadata(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, object]:
    if args.source is None:
        raise OpeningError("--source is required when preparing the suite")
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    records, source_sha256, physical_lines = read_source(source)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256.lower():
        raise OpeningError(
            f"source SHA-256 mismatch: expected {args.expected_source_sha256.lower()}, got {source_sha256}"
        )
    selected = select_records(records, args.count, args.seed)
    output_data = write_output(output, selected)
    value = metadata_for(
        output,
        metadata,
        selected,
        source_sha256,
        physical_lines,
        len(records),
        output_data,
        args.seed,
        args.count,
        args.smoke_count,
        datetime.now(timezone.utc).isoformat(),
    )
    write_metadata(metadata, value)
    return value


def load_metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpeningError(f"cannot read metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OpeningError("opening metadata root must be an object")
    return value


def verify(args: argparse.Namespace) -> dict[str, object]:
    metadata_path = args.metadata.expanduser().resolve()
    value = load_metadata(metadata_path)
    selection = value.get("selection")
    source = value.get("source")
    output_info = value.get("output")
    if not isinstance(selection, dict) or not isinstance(source, dict) or not isinstance(output_info, dict):
        raise OpeningError("metadata lacks selection, source, or output objects")
    output_path = resolve_stored_path(output_info.get("path"))
    if not output_path.is_file():
        raise OpeningError(f"committed opening output is missing: {output_path}")
    actual_sha256 = sha256_file(output_path)
    if actual_sha256 != output_info.get("sha256"):
        raise OpeningError(f"opening output SHA-256 mismatch: expected {output_info.get('sha256')}, got {actual_sha256}")
    lines = [line.strip() for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != output_info.get("position_count") or len(lines) != selection.get("count"):
        raise OpeningError("opening output count does not match metadata")
    if len(set(lines)) != len(lines):
        raise OpeningError("opening output contains duplicate positions")
    for line_number, fen in enumerate(lines, start=1):
        canonical_fen(fen, line_number)
    selected_lines = selection.get("selected_source_lines")
    if not isinstance(selected_lines, list) or len(selected_lines) != len(lines):
        raise OpeningError("selected source-line audit list does not match output")
    if args.source is not None:
        records, source_sha256, physical_lines = read_source(args.source.expanduser().resolve())
        if source_sha256 != source.get("content_sha256"):
            raise OpeningError(
                f"source SHA-256 mismatch: expected {source.get('content_sha256')}, got {source_sha256}"
            )
        if physical_lines != source.get("physical_line_count") or len(records) != source.get("unique_position_count"):
            raise OpeningError("source line or unique-position count differs from metadata")
    return {
        "status": "PASS",
        "output": str(output_path),
        "positions": len(lines),
        "source_rechecked": args.source is not None,
        "output_sha256": actual_sha256,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--verify", action="store_true", help="verify committed output and metadata")
    command.add_argument("--source", type=Path, default=None, help="extracted upstream EPD source")
    command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    command.add_argument("--count", type=int, default=DEFAULT_COUNT)
    command.add_argument("--smoke-count", type=int, default=DEFAULT_SMOKE_COUNT)
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--expected-source-sha256", default=None)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        result = verify(args) if args.verify else prepare(args)
    except OpeningError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
