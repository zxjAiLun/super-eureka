#!/usr/bin/env python3
"""Rescore the saved S2.1 practical-gate moves with Stockfish.

The practical gate already contains the moves selected by ``Current`` and
``current-threat-aware`` at 1, 3 and 10 seconds.  This tool does not rerun
either engine.  It gives every saved row two fresh, equal-resource Stockfish
searches:

* unrestricted search for the teacher's best move;
* ``searchmoves <played_move>`` for the move selected by the engine.

Only centipawn loss and explicit mate outcomes are used.  WDL is neither
requested nor interpreted, and the result is diagnostic evidence rather than
an Elo or promotion decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable

import chess

from analyze_d114_stockfish import (
    CANDIDATE_LABEL as D114_CANDIDATE_LABEL,
    BASELINE_LABEL as D114_BASELINE_LABEL,
    ParsedScore,
    SearchResult,
    StockfishAnalysisError,
    StockfishSession,
    classify_cpl,
    percentile,
    score_metrics,
)


PROFILE_CURRENT = "current"
PROFILE_CANDIDATE = "current-threat-aware"
EXPECTED_PROFILES = (PROFILE_CURRENT, PROFILE_CANDIDATE)
EXPECTED_GROUPS = ("king-danger", "defensive-resource", "control")
EXPECTED_POSITION_COUNT = 9
EXPECTED_TIME_LIMITS = (1000, 3000, 10000)
EXPECTED_ROW_COUNT = 54

# The imported D1.14 module uses human-readable labels for its own PGN
# records.  Keep the S2.1 labels explicit here so the output cannot silently
# acquire the wrong comparison role.
PROFILE_LABELS = {
    PROFILE_CURRENT: "Current",
    PROFILE_CANDIDATE: "CurrentThreatAware",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_practical_gate(path: Path) -> dict[str, Any]:
    """Load and strictly validate the saved 54-row practical artifact."""

    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "practical gate artifact must be an object")
    require(
        data.get("status") == "DIAGNOSTIC_ONLY_NO_DECISION",
        "input artifact must be the diagnostic-only S2.1 gate",
    )
    rows = data.get("rows")
    require(isinstance(rows, list), "practical gate artifact rows must be a list")
    require(
        len(rows) == EXPECTED_ROW_COUNT,
        f"S2.1 practical gate must contain exactly {EXPECTED_ROW_COUNT} rows",
    )

    identities: set[tuple[str, int, str]] = set()
    positions: dict[str, str] = {}
    groups: set[str] = set()
    profiles: set[str] = set()
    limits: set[int] = set()
    for index, row in enumerate(rows, start=1):
        require(isinstance(row, dict), f"row {index} must be an object")
        for field in (
            "position_id",
            "group",
            "fen",
            "time_limit_ms",
            "engine_profile",
            "bestmove",
        ):
            require(field in row, f"row {index} is missing {field}")
        position_id = str(row["position_id"])
        group = str(row["group"])
        profile = str(row["engine_profile"])
        time_limit_ms = int(row["time_limit_ms"])
        fen = str(row["fen"])
        key = (position_id, time_limit_ms, profile)
        require(key not in identities, f"duplicate S2.1 row identity: {key}")
        identities.add(key)
        if position_id in positions:
            require(
                positions[position_id] == fen,
                f"position {position_id} changes FEN across rows",
            )
        positions[position_id] = fen
        groups.add(group)
        profiles.add(profile)
        limits.add(time_limit_ms)
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"invalid FEN in row {index}: {fen}") from exc
        move_text = str(row["bestmove"])
        require(move_text != "0000", f"non-terminal S2.1 row has bestmove 0000: {key}")
        try:
            move = chess.Move.from_uci(move_text)
        except ValueError as exc:
            raise ValueError(f"invalid bestmove in row {index}: {move_text}") from exc
        require(board.is_legal(move), f"illegal selected move in row {index}: {key}")

    require(
        len(positions) == EXPECTED_POSITION_COUNT,
        f"S2.1 practical gate must contain {EXPECTED_POSITION_COUNT} positions",
    )
    require(groups == set(EXPECTED_GROUPS), f"unexpected S2.1 groups: {groups}")
    require(profiles == set(EXPECTED_PROFILES), f"unexpected S2.1 profiles: {profiles}")
    require(limits == set(EXPECTED_TIME_LIMITS), f"unexpected time limits: {limits}")
    for profile in EXPECTED_PROFILES:
        require(
            sum(row["engine_profile"] == profile for row in rows) == 27,
            f"profile {profile} must contain 27 rows",
        )
    for limit in EXPECTED_TIME_LIMITS:
        require(
            sum(int(row["time_limit_ms"]) == limit for row in rows) == 18,
            f"time limit {limit} must contain 18 rows",
        )
    for group in EXPECTED_GROUPS:
        require(
            sum(row["group"] == group for row in rows) == 18,
            f"group {group} must contain 18 rows",
        )
    return data


def _validate_teacher_bestmove(board: chess.Board, result: SearchResult, fen: str) -> None:
    require(result.best_move != "0000", f"Stockfish returned 0000 for non-terminal {fen}")
    try:
        move = chess.Move.from_uci(result.best_move)
    except ValueError as exc:
        raise StockfishAnalysisError(
            f"Stockfish returned malformed bestmove {result.best_move!r}"
        ) from exc
    require(board.is_legal(move), f"Stockfish returned illegal bestmove {result.best_move} for {fen}")


def rescore_rows(
    rows: list[dict[str, Any]],
    executable: Path,
    nodes: int,
    hash_mb: int,
    threads: int,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run exactly two fresh teacher searches for every practical-gate row."""

    analyzed: list[dict[str, Any]] = []
    identity: list[str] = []
    with StockfishSession(executable, hash_mb, threads, timeout_s) as stockfish:
        identity = list(stockfish.identity)
        for index, row in enumerate(rows, start=1):
            fen = str(row["fen"])
            board = chess.Board(fen)
            played_move = str(row["bestmove"])
            move = chess.Move.from_uci(played_move)
            require(board.is_legal(move), f"saved engine move is illegal: {fen} {played_move}")

            # StockfishSession.search sends ucinewgame, Clear Hash and waits
            # for readyok before every one of these calls.  Consequently the
            # unrestricted and forced searches have equal, cold TT state.
            best = stockfish.search(fen, nodes)
            _validate_teacher_bestmove(board, best, fen)
            played = stockfish.search(fen, nodes, played_move)
            require(
                played.best_move == played_move,
                f"searchmoves was not honoured for {fen}: requested {played_move}, "
                f"got {played.best_move}",
            )

            result = dict(row)
            result["played_move"] = played_move
            result.update(score_metrics(best, played, played_move))
            result["teacher_nodes_per_search"] = nodes
            result["teacher_hash_mb"] = hash_mb
            result["teacher_threads"] = threads
            result["teacher_wdl_enabled"] = False
            result["teacher_searches_cleared"] = True
            result["teacher_search_pair"] = 2 * index - 1, 2 * index
            analyzed.append(result)
            if index == 1 or index % 9 == 0 or index == len(rows):
                print(f"rescored {index}/{len(rows)} rows ({2 * index} searches)", flush=True)
    return analyzed, identity


def _stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    cpls = [int(record["centipawn_loss"]) for record in values if record.get("centipawn_loss") is not None]
    count = len(values)
    cp_count = len(cpls)
    classifications = {name: sum(record.get("classification") == name for record in values) for name in (
        "normal", "inaccuracy", "mistake", "blunder"
    )}
    mate_categories = {
        category: sum(record.get("mate_category") == category for record in values)
        for category in sorted({
            record["mate_category"]
            for record in values
            if record.get("mate_category") is not None
        })
    }
    return {
        "moves": count,
        "cp_scored_moves": cp_count,
        "mean_cpl": statistics.mean(cpls) if cpls else None,
        "median_cpl": statistics.median(cpls) if cpls else None,
        "p75_cpl": percentile(cpls, 0.75),
        "p90_cpl": percentile(cpls, 0.90),
        "p95_cpl": percentile(cpls, 0.95),
        "inaccuracy": classifications["inaccuracy"],
        "mistake": classifications["mistake"],
        "blunder": classifications["blunder"],
        "cpl_100_plus": sum(value >= 100 for value in cpls),
        "cpl_300_plus": sum(value >= 300 for value in cpls),
        "cpl_500_plus": sum(value >= 500 for value in cpls),
        "mate_swing": sum(bool(record.get("mate_swing")) for record in values),
        "mate_outcomes": sum(bool(record.get("mate_outcome")) for record in values),
        "harmful_mate_categories": {
            category: count
            for category, count in mate_categories.items()
            if category in {"winning_mate_delayed", "losing_mate_accelerated"}
        },
        "mate_categories": mate_categories,
        "best_move_matches": sum(bool(record.get("best_move_agreement")) for record in values),
        "best_move_match_rate": (
            sum(bool(record.get("best_move_agreement")) for record in values) / count
            if count
            else None
        ),
        "classification_rates": {
            name: classifications[name] / cp_count if cp_count else None
            for name in ("inaccuracy", "mistake", "blunder")
        },
    }


def _pair_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["position_id"]), int(row["time_limit_ms"]))][
            str(row["engine_profile"])
        ] = row

    candidate_lower = baseline_lower = equal = cp_comparable = 0
    candidate_harmful_mate = baseline_harmful_mate = 0
    same_move = different_move = 0
    winners: list[dict[str, Any]] = []
    for key in sorted(grouped):
        pair = grouped[key]
        baseline = pair.get(PROFILE_CURRENT)
        candidate = pair.get(PROFILE_CANDIDATE)
        if baseline is None or candidate is None:
            continue
        if baseline["played_move"] == candidate["played_move"]:
            same_move += 1
        else:
            different_move += 1
        winner = "incomparable"
        baseline_loss = baseline.get("centipawn_loss")
        candidate_loss = candidate.get("centipawn_loss")
        if baseline_loss is not None and candidate_loss is not None:
            cp_comparable += 1
            if candidate_loss < baseline_loss:
                candidate_lower += 1
                winner = PROFILE_CANDIDATE
            elif candidate_loss > baseline_loss:
                baseline_lower += 1
                winner = PROFILE_CURRENT
            else:
                equal += 1
                winner = "equal"
        else:
            candidate_harmful = bool(candidate.get("mate_swing"))
            baseline_harmful = bool(baseline.get("mate_swing"))
            candidate_harmful_mate += int(candidate_harmful)
            baseline_harmful_mate += int(baseline_harmful)
            if candidate_harmful != baseline_harmful:
                winner = PROFILE_CURRENT if candidate_harmful else PROFILE_CANDIDATE
            elif candidate.get("mate_category") == baseline.get("mate_category"):
                equal += 1
                winner = "equal"
        winners.append({
            "position_id": key[0],
            "time_limit_ms": key[1],
            "winner": winner,
            "same_played_move": baseline["played_move"] == candidate["played_move"],
            "baseline_played_move": baseline["played_move"],
            "candidate_played_move": candidate["played_move"],
            "baseline_cpl": baseline_loss,
            "candidate_cpl": candidate_loss,
            "baseline_mate_category": baseline.get("mate_category"),
            "candidate_mate_category": candidate.get("mate_category"),
        })
    return {
        "paired_position_time_groups": len(winners),
        "same_move_groups": same_move,
        "different_move_groups": different_move,
        "cp_comparable_groups": cp_comparable,
        "candidate_lower_cpl": candidate_lower,
        "baseline_lower_cpl": baseline_lower,
        "equal_quality_groups": equal,
        "candidate_harmful_mate_groups": candidate_harmful_mate,
        "baseline_harmful_mate_groups": baseline_harmful_mate,
        "per_position_winners": winners,
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile = {
        profile: _stats(row for row in rows if row["engine_profile"] == profile)
        for profile in EXPECTED_PROFILES
    }
    by_profile_time_group: dict[str, dict[str, dict[str, Any]]] = {}
    for profile in EXPECTED_PROFILES:
        by_profile_time_group[profile] = {}
        for time_limit in EXPECTED_TIME_LIMITS:
            by_profile_time_group[profile][str(time_limit)] = {
                group: _stats(
                    row
                    for row in rows
                    if row["engine_profile"] == profile
                    and int(row["time_limit_ms"]) == time_limit
                    and row["group"] == group
                )
                for group in EXPECTED_GROUPS
            }
    return {
        "profiles": by_profile,
        "profile_time_group": by_profile_time_group,
        "paired_comparison": _pair_comparison(rows),
    }


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--stockfish", type=Path, required=True, help="Stockfish 18 executable"
    )
    command.add_argument(
        "--input",
        type=Path,
        default=Path("results/s2.1/practical-gate.json"),
        help="saved S2.1 practical-gate JSON",
    )
    command.add_argument(
        "--output",
        type=Path,
        default=Path("results/s2.1/teacher-rescore-500k.json"),
        help="teacher rescore JSON output",
    )
    command.add_argument("--nodes", type=int, default=500_000)
    command.add_argument("--hash-mb", type=int, default=16)
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--timeout-s", type=float, default=120.0)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.nodes <= 0 or args.hash_mb <= 0 or args.threads <= 0 or args.timeout_s <= 0:
        print("nodes, hash, threads, and timeout must be positive", file=sys.stderr)
        return 2
    try:
        input_path = args.input.expanduser().resolve()
        executable = args.stockfish.expanduser().resolve()
        data = load_practical_gate(input_path)
        repo = Path(__file__).resolve().parent.parent
        rows, identity = rescore_rows(
            list(data["rows"]), executable, args.nodes, args.hash_mb, args.threads, args.timeout_s
        )
        payload = {
            "schema_version": 1,
            "status": "TEACHER_RESCORE_COMPLETE",
            "analysis": "S2.1 practical position Stockfish rescore",
            "score_model": "centipawn-loss and explicit mate outcomes",
            "wdl_enabled": False,
            "wdl_used": False,
            "input_artifact": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "git_sha": data.get("git_sha"),
                "rows": len(data["rows"]),
            },
            "runner_git_sha": git_sha(repo),
            "stockfish": {
                "path": str(executable),
                "sha256": sha256_file(executable),
                "identity": identity,
                "nodes_per_search": args.nodes,
                "hash_mb": args.hash_mb,
                "threads": args.threads,
                "searches_per_row": 2,
                "total_searches": len(rows) * 2,
                "clear_hash_before_each_search": True,
                "searches": ["unrestricted", "searchmoves played_move"],
            },
            "rows": rows,
            "summary": build_summary(rows),
            "interpretation": (
                "Diagnostic CP/mate evidence only; not an Elo estimate, not a WDL "
                "result, and not a Current-promotion decision."
            ),
        }
        write_output(args.output.expanduser().resolve(), payload)
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, StockfishAnalysisError, subprocess.SubprocessError) as exc:
        print(f"S21_TEACHER_RESCORE_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
