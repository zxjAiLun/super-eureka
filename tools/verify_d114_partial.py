#!/usr/bin/env python3
"""Verify the completed pairs of an intentionally aborted D1.14 match.

This is deliberately separate from ``verify_d114_match.py``.  The normal
verifier remains strict and requires the configured game limit; this tool
accepts a runtime-aborted artifact, validates every PGN game that was actually
written, and computes paired statistics only from complete opening pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from verify_d114_match import (
    D114VerificationError,
    descriptive_statistics,
    load_games,
    load_opening_lines,
    parse_manager_reasons,
    position_key,
    read_json,
    require,
    verify_binary_artifact,
    verify_opening_hash_artifact,
    verify_pairs,
)


def complete_pair_counts(raw_games: int) -> tuple[int, int, int]:
    """Return raw games, complete games, and complete opening pairs."""
    require(raw_games >= 0, "raw game count cannot be negative")
    complete_games = raw_games - (raw_games % 2)
    return raw_games, complete_games, complete_games // 2


def verify_partial(run_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    require(run_dir.is_dir(), f"run directory does not exist: {run_dir}")
    manifest = read_json(run_dir / "manifest.json")
    require(manifest.get("milestone") == "D1.14", "not a D1.14 manifest")
    require(
        manifest.get("status") in {"INTEGRITY_FAIL", "MANAGER_COMPLETED", "VERIFIED"},
        "partial verification requires a completed or aborted manager artifact",
    )
    require(manifest.get("decision") in {"INCONCLUSIVE", "PENDING_VERIFICATION", "NOT_STARTED"},
            "partial artifact already carries a terminal SPRT decision")
    match = manifest.get("match")
    require(isinstance(match, dict), "manifest match section is missing")
    require(match.get("hash_mb") == 16, "D1.14 Hash must be 16 MB")
    require(match.get("threads") == 1, "D1.14 engine threads must be 1")
    verify_opening_hash_artifact(manifest, run_dir)
    expected_fens = load_opening_lines(manifest, run_dir)
    verify_binary_artifact(manifest, run_dir)
    reasons = parse_manager_reasons(run_dir)
    games = load_games(run_dir, expected_fens, reasons)

    raw_games = len(games)
    require(raw_games > 0, "partial artifact contains no completed games")
    _raw_games, complete_game_count, complete_pairs = complete_pair_counts(raw_games)
    require(complete_game_count > 0, "partial artifact contains no complete opening pair")
    complete_games = games[:complete_game_count]
    verify_pairs(
        complete_games,
        expected_fens[:complete_pairs],
        complete_game_count,
        require_sequential_order=match.get("concurrency") == 1,
    )

    unpaired_games = games[complete_game_count:]
    if unpaired_games:
        require(len(unpaired_games) == 1, "partial artifact has more than one unpaired game")
        require(
            position_key(unpaired_games[0].fen) == position_key(expected_fens[complete_pairs]),
            "unpaired game is not the next opening after the complete pairs",
        )

    summary = descriptive_statistics(complete_games, "D1.14 aborted runtime pilot")
    summary.update(
        {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "raw_games": raw_games,
            "complete_pair_games": complete_game_count,
            "complete_opening_pairs": complete_pairs,
            "unpaired_games": len(unpaired_games),
            "status": "ABORTED_FOR_RUNTIME",
            "decision": "INCONCLUSIVE",
            "sprt_decision": "NO_DECISION",
            "integrity_status": "PASS_FOR_COMPLETED_GAMES",
            "pgn_status": "PARTIAL_VALIDATED",
            "profile_status": "PASS",
            "opening_pair_status": "PASS_FOR_COMPLETE_PAIRS",
            "raw_pgn_retained": True,
        }
    )
    if output_path is None:
        output_path = run_dir / "partial-verification.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("run_dir", type=Path)
    command.add_argument("--output", type=Path, default=None)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        summary = verify_partial(args.run_dir, args.output)
    except D114VerificationError as exc:
        print(f"INTEGRITY_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
