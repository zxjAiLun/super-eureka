#!/usr/bin/env python3
"""Run the S2.2b fixed-node Stockfish CP/mate teacher gate.

The input is the local S2.2a fixed-time artifact.  Its provenance is pinned
to the reviewed gate commit, engine hash, and manifest hash before any
Stockfish process is started.  Unrestricted Stockfish searches are run once
per FEN; forced searches are run once per unique ``(FEN, selected move)``.

This tool produces a self-contained, committed artifact.  It is a quality
gate for the integrated evaluation candidate, not a game, Elo, or SPRT tool.
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
    ParsedScore,
    SearchResult,
    StockfishAnalysisError,
    StockfishSession,
    classify_cpl,
    percentile,
)
from run_s22a_fixed_time_gate import (
    EXPECTED_GROUPS,
    EXPECTED_POSITION_COUNT,
    EXPECTED_RETAINED_S21_IDS,
    EXPECTED_TIME_MS,
    PROFILE_CANDIDATE,
    PROFILE_CURRENT,
    PROFILES,
    load_manifest,
)


SOURCE_GATE_GIT_SHA = "cbc1077a093ef078c39f427cd72f7a0fdcd34d30"
SOURCE_ENGINE_SHA256 = "77a10653f0238b5f224c7ffc29b8bd31043f8b0c5dc41a49d5ec60538a31186c"
SOURCE_MANIFEST_SHA256 = "036701d2a72c2b2efa24acdc6044663477caac8b71f081aca8a2650d61db8319"
EXPECTED_SOURCE_ROWS = 150
EXPECTED_PROFILE_ROWS = 75
EXPECTED_TIME_ROWS = 50
TEACHER_NODES = 500_000
TEACHER_HASH_MB = 16
TEACHER_THREADS = 1
TEACHER_OPTIONS = {"MultiPV": "1", "UCI_ShowWDL": "false"}
HARMFUL_MATE_CATEGORIES = {
    "winning_mate_delayed",
    "losing_mate_accelerated",
}
SEVERE_MATE_CATEGORIES = {
    "missed_winning_mate",
    "entered_losing_mate",
}
TARGET_GROUPS = (
    "development-coordination",
    "pawn-structure-pawn-push",
    "rook-activity",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
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


class S22bStockfishSession(StockfishSession):
    """StockfishSession with the S2.2b options explicitly locked."""

    def __enter__(self) -> "S22bStockfishSession":
        super().__enter__()
        self._send("setoption name MultiPV value 1")
        self._send("setoption name UCI_ShowWDL value false")
        self._ready()
        return self


def load_source_artifact(
    path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "S2.2a source artifact must be an object")
    require(data.get("decision", {}).get("status") == "PASS", "S2.2a source gate is not PASS")
    require(data.get("git_sha") == SOURCE_GATE_GIT_SHA, "S2.2a source git SHA mismatch")
    require(len(data.get("rows", [])) == EXPECTED_SOURCE_ROWS, "S2.2a source row count mismatch")
    require(
        data.get("engine", {}).get("sha256") == SOURCE_ENGINE_SHA256,
        "S2.2a source engine SHA-256 mismatch",
    )
    require(
        data.get("manifest", {}).get("sha256") == SOURCE_MANIFEST_SHA256,
        "S2.2a source manifest SHA-256 mismatch",
    )
    require(tuple(data.get("limits_ms", ())) == EXPECTED_TIME_MS, "S2.2a source time limits mismatch")
    require(
        tuple(data.get("engine", {}).get("profiles", ())) == PROFILES,
        "S2.2a source profiles mismatch",
    )
    require(
        data.get("tt") == {"mode": "cold", "hash_mb": 16, "threads": 1},
        "S2.2a source TT configuration mismatch",
    )

    manifest_bytes_sha = sha256_file(manifest_path)
    require(manifest_bytes_sha == SOURCE_MANIFEST_SHA256, "local manifest SHA-256 mismatch")
    positions = load_manifest(manifest_path)
    require(len(positions) == EXPECTED_POSITION_COUNT, "S2.2a manifest position count mismatch")
    by_id = {str(position["id"]): position for position in positions}
    require(len(by_id) == EXPECTED_POSITION_COUNT, "S2.2a manifest IDs are not unique")
    require(
        set(by_id) >= EXPECTED_RETAINED_S21_IDS,
        "S2.2a manifest retained-position IDs are incomplete",
    )

    expected_identities = {
        (position_id, time_ms, profile)
        for position_id in by_id
        for time_ms in EXPECTED_TIME_MS
        for profile in PROFILES
    }
    actual_identities: set[tuple[str, int, str]] = set()
    for row_index, row in enumerate(data["rows"], start=1):
        require(isinstance(row, dict), f"source row {row_index} is not an object")
        for field in ("position_id", "group", "fen", "time_limit_ms", "engine_profile", "bestmove"):
            require(field in row, f"source row {row_index} missing {field}")
        position_id = str(row["position_id"])
        time_ms = int(row["time_limit_ms"])
        profile = str(row["engine_profile"])
        identity = (position_id, time_ms, profile)
        require(identity not in actual_identities, f"duplicate source row identity: {identity}")
        actual_identities.add(identity)
        require(position_id in by_id, f"source row {row_index} has unknown position {position_id}")
        position = by_id[position_id]
        require(row["fen"] == position["fen"], f"source row {row_index} FEN mismatch")
        require(row["group"] == position["group"], f"source row {row_index} group mismatch")
        require(time_ms in EXPECTED_TIME_MS, f"source row {row_index} time limit mismatch")
        require(profile in PROFILES, f"source row {row_index} profile mismatch")
        board = chess.Board(str(row["fen"]))
        move_text = str(row["bestmove"])
        require(move_text != "0000", f"source row {row_index} has terminal bestmove")
        move = chess.Move.from_uci(move_text)
        require(board.is_legal(move), f"source row {row_index} selected move is illegal")
        counters = row.get("counters", {})
        require(
            all(int(value) == 0 for value in counters.values()),
            f"source row {row_index} contains nonzero feature counters",
        )
    require(actual_identities == expected_identities, "S2.2a source identities are incomplete")
    return data, list(data["rows"]), by_id


def score_payload(result: SearchResult) -> dict[str, Any]:
    return {
        "best_move": result.best_move,
        "score": result.score.raw,
        "score_kind": result.score.kind,
        "score_value": result.score.value,
        "depth": result.depth,
        "pv": list(result.pv),
    }


def validate_bestmove(board: chess.Board, result: SearchResult, context: str) -> None:
    require(result.best_move != "0000", f"Stockfish returned 0000 for {context}")
    try:
        move = chess.Move.from_uci(result.best_move)
    except ValueError as exc:
        raise StockfishAnalysisError(
            f"Stockfish returned malformed bestmove {result.best_move!r} for {context}"
        ) from exc
    require(board.is_legal(move), f"Stockfish returned illegal bestmove {result.best_move} for {context}")


def mate_category_s22b(best: ParsedScore, forced: ParsedScore) -> str | None:
    if best.kind == "mate" and best.value > 0:
        if forced.kind != "mate" or forced.value <= 0:
            return "missed_winning_mate"
        if forced.value > best.value:
            return "winning_mate_delayed"
        if forced.value < best.value:
            return "winning_mate_accelerated"
        return "allowed_mate"
    if best.kind == "mate" and best.value < 0:
        if forced.kind != "mate" or forced.value >= 0:
            return "losing_mate_delayed"
        if forced.value > best.value:
            return "losing_mate_accelerated"
        if forced.value < best.value:
            return "losing_mate_delayed"
        return "allowed_mate"
    if forced.kind == "mate" and forced.value > 0:
        return "winning_mate_accelerated"
    if forced.kind == "mate" and forced.value < 0:
        return "entered_losing_mate"
    return None


def teacher_metrics(best: SearchResult, forced: SearchResult, selected_move: str) -> dict[str, Any]:
    category = mate_category_s22b(best.score, forced.score)
    raw_cpl = (
        best.score.value - forced.score.value
        if best.score.kind == "cp" and forced.score.kind == "cp"
        else None
    )
    return {
        "best_move": best.best_move,
        "best_score": best.score.raw,
        "best_score_kind": best.score.kind,
        "best_score_value": best.score.value,
        "best_depth": best.depth,
        "best_pv": list(best.pv),
        "forced_move": forced.best_move,
        "forced_score": forced.score.raw,
        "forced_score_kind": forced.score.kind,
        "forced_score_value": forced.score.value,
        "forced_depth": forced.depth,
        "forced_pv": list(forced.pv),
        "best_move_agreement": best.best_move == selected_move,
        "centipawn_loss_raw": raw_cpl,
        "negative_cpl_anomaly": raw_cpl is not None and raw_cpl < 0,
        "classification": classify_cpl(raw_cpl) if category is None else (
            "mate-swing" if category in HARMFUL_MATE_CATEGORIES else "mate-outcome"
        ),
        "mate_category": category,
        "harmful_mate": category in HARMFUL_MATE_CATEGORIES,
        "missed_winning_mate": category == "missed_winning_mate",
        "entered_losing_mate": category == "entered_losing_mate",
    }


def source_mapping(row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_row_index": row_index,
        "position_id": row["position_id"],
        "group": row["group"],
        "fen": row["fen"],
        "time_limit_ms": int(row["time_limit_ms"]),
        "engine_profile": row["engine_profile"],
        "selected_move": row["bestmove"],
        "source_score": row.get("score"),
        "source_completed_depth": row.get("completed_depth"),
        "source_pv": row.get("pv", []),
        "source_nodes": row.get("nodes"),
        "source_nps": row.get("nps"),
    }


def run_teacher(
    rows: list[dict[str, Any]],
    executable: Path,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    positions: dict[str, dict[str, Any]] = {}
    for row in rows:
        positions.setdefault(str(row["fen"]), row)
    ordered_positions = [positions[fen] for fen in dict.fromkeys(str(row["fen"]) for row in rows)]
    forced_targets = list(
        dict.fromkeys((str(row["fen"]), str(row["bestmove"])) for row in rows)
    )
    unrestricted: dict[str, SearchResult] = {}
    forced: dict[tuple[str, str], SearchResult] = {}
    with S22bStockfishSession(
        executable,
        TEACHER_HASH_MB,
        TEACHER_THREADS,
        timeout_s,
    ) as stockfish:
        identity = list(stockfish.identity)
        for index, row in enumerate(ordered_positions, start=1):
            fen = str(row["fen"])
            board = chess.Board(fen)
            print(f"teacher unrestricted {index}/{len(ordered_positions)} {row['position_id']}", flush=True)
            result = stockfish.search(fen, TEACHER_NODES)
            validate_bestmove(board, result, f"unrestricted {fen}")
            unrestricted[fen] = result
        for index, (fen, move_text) in enumerate(forced_targets, start=1):
            board = chess.Board(fen)
            print(f"teacher forced {index}/{len(forced_targets)} move={move_text}", flush=True)
            result = stockfish.search(fen, TEACHER_NODES, move_text)
            require(result.best_move == move_text, f"searchmoves was not honoured for {fen}: {move_text}")
            validate_bestmove(board, result, f"forced {fen} {move_text}")
            forced[(fen, move_text)] = result

    analyzed: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        fen = str(row["fen"])
        selected_move = str(row["bestmove"])
        best = unrestricted[fen]
        forced_result = forced[(fen, selected_move)]
        mapping.append(source_mapping(row_index, row))
        result = source_mapping(row_index, row)
        result.update(teacher_metrics(best, forced_result, selected_move))
        result["unrestricted_key"] = fen
        result["forced_key"] = {"fen": fen, "move": selected_move}
        analyzed.append(result)
    unrestricted_payload = [
        {"fen": fen, "result": score_payload(result)}
        for fen, result in unrestricted.items()
    ]
    forced_payload = [
        {"fen": fen, "move": move, "result": score_payload(result)}
        for (fen, move), result in forced.items()
    ]
    return analyzed, mapping, unrestricted_payload + forced_payload, identity


def _stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    cpls = [int(row["centipawn_loss_raw"]) for row in values if row.get("centipawn_loss_raw") is not None]
    categories = {
        category: sum(row.get("mate_category") == category for row in values)
        for category in sorted({row.get("mate_category") for row in values if row.get("mate_category")})
    }
    return {
        "moves": len(values),
        "cp_scored_moves": len(cpls),
        "mean_cpl": statistics.mean(cpls) if cpls else None,
        "median_cpl": statistics.median(cpls) if cpls else None,
        "p90_cpl": percentile(cpls, 0.90),
        "p95_cpl": percentile(cpls, 0.95),
        "cpl_300_plus": sum(value >= 300 for value in cpls),
        "cpl_500_plus": sum(value >= 500 for value in cpls),
        "negative_cpl_anomalies": sum(value < 0 for value in cpls),
        "harmful_mate": sum(bool(row.get("harmful_mate")) for row in values),
        "missed_winning_mate": sum(bool(row.get("missed_winning_mate")) for row in values),
        "entered_losing_mate": sum(bool(row.get("entered_losing_mate")) for row in values),
        "mate_categories": categories,
        "best_move_matches": sum(bool(row.get("best_move_agreement")) for row in values),
        "best_move_match_rate": (
            sum(bool(row.get("best_move_agreement")) for row in values) / len(values)
            if values
            else None
        ),
    }


def _mate_pair_winner(current: dict[str, Any], candidate: dict[str, Any]) -> str:
    current_harmful = bool(current.get("harmful_mate"))
    candidate_harmful = bool(candidate.get("harmful_mate"))
    if current_harmful != candidate_harmful:
        return PROFILE_CANDIDATE if not candidate_harmful else PROFILE_CURRENT
    current_severe = bool(current.get("missed_winning_mate") or current.get("entered_losing_mate"))
    candidate_severe = bool(candidate.get("missed_winning_mate") or candidate.get("entered_losing_mate"))
    if current_severe != candidate_severe:
        return PROFILE_CANDIDATE if not candidate_severe else PROFILE_CURRENT
    return "equal"


def pair_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[(str(row["position_id"]), int(row["time_limit_ms"]))][str(row["engine_profile"])] = row
    comparisons: list[dict[str, Any]] = []
    counts = {
        "candidate_wins": 0,
        "current_wins": 0,
        "equal": 0,
        "incomparable": 0,
        "different_move_groups": 0,
        "same_move_groups": 0,
        "directly_comparable_groups": 0,
    }
    for (position_id, time_ms), pair in sorted(grouped.items()):
        current = pair.get(PROFILE_CURRENT)
        candidate = pair.get(PROFILE_CANDIDATE)
        if current is None or candidate is None:
            continue
        same_move = current["selected_move"] == candidate["selected_move"]
        counts["same_move_groups" if same_move else "different_move_groups"] += 1
        winner = "same-move" if same_move else "incomparable"
        comparison_kind = "same-move" if same_move else "incomparable"
        if not same_move:
            current_cpl = current.get("centipawn_loss_raw")
            candidate_cpl = candidate.get("centipawn_loss_raw")
            if current_cpl is not None and candidate_cpl is not None:
                comparison_kind = "cp"
                if candidate_cpl < current_cpl:
                    winner = PROFILE_CANDIDATE
                elif candidate_cpl > current_cpl:
                    winner = PROFILE_CURRENT
                else:
                    winner = "equal"
            elif current.get("mate_category") is not None and candidate.get("mate_category") is not None:
                comparison_kind = "mate"
                winner = _mate_pair_winner(current, candidate)
            counts["directly_comparable_groups"] += int(winner in {PROFILE_CURRENT, PROFILE_CANDIDATE, "equal"})
        counts[{
            PROFILE_CANDIDATE: "candidate_wins",
            PROFILE_CURRENT: "current_wins",
            "equal": "equal",
        }.get(winner, "incomparable")] += 1
        comparisons.append({
            "position_id": position_id,
            "group": current["group"],
            "time_limit_ms": time_ms,
            "same_move": same_move,
            "current_move": current["selected_move"],
            "candidate_move": candidate["selected_move"],
            "comparison_kind": comparison_kind,
            "winner": winner,
            "current_cpl": current.get("centipawn_loss_raw"),
            "candidate_cpl": candidate.get("centipawn_loss_raw"),
            "current_mate_category": current.get("mate_category"),
            "candidate_mate_category": candidate.get("mate_category"),
        })
    return {**counts, "comparisons": comparisons}


def position_level_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile_position: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_profile_position[str(row["engine_profile"])][str(row["position_id"])].append(row)
    output: dict[str, list[dict[str, Any]]] = {}
    for profile in PROFILES:
        entries = []
        for position_id in sorted(by_profile_position[profile]):
            values = sorted(by_profile_position[profile][position_id], key=lambda row: int(row["time_limit_ms"]))
            cpls = [int(row["centipawn_loss_raw"]) for row in values if row.get("centipawn_loss_raw") is not None]
            entries.append({
                "position_id": position_id,
                "group": values[0]["group"],
                "moves_by_time": {str(row["time_limit_ms"]): row["selected_move"] for row in values},
                "unique_selected_moves": sorted({row["selected_move"] for row in values}),
                "stable_selected_move": len({row["selected_move"] for row in values}) == 1,
                "worst_cpl": max(cpls) if cpls else None,
                "median_cpl": statistics.median(cpls) if cpls else None,
                "harmful_mate": any(row.get("harmful_mate") for row in values),
                "missed_winning_mate": any(row.get("missed_winning_mate") for row in values),
                "entered_losing_mate": any(row.get("entered_losing_mate") for row in values),
            })
        output[profile] = entries
    return output


def group_summary(records: list[dict[str, Any]], pairs: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in EXPECTED_GROUPS:
        group_records = [row for row in records if row["group"] == group]
        profile_stats = {
            profile: _stats(row for row in group_records if row["engine_profile"] == profile)
            for profile in PROFILES
        }
        group_pairs = [row for row in pairs["comparisons"] if row["group"] == group and not row["same_move"]]
        candidate_wins = sum(row["winner"] == PROFILE_CANDIDATE for row in group_pairs)
        current_wins = sum(row["winner"] == PROFILE_CURRENT for row in group_pairs)
        candidate = profile_stats[PROFILE_CANDIDATE]
        current = profile_stats[PROFILE_CURRENT]
        median_non_worse = (
            candidate["median_cpl"] is not None
            and current["median_cpl"] is not None
            and candidate["median_cpl"] <= current["median_cpl"]
        )
        no_added_harmful = candidate["harmful_mate"] <= current["harmful_mate"]
        output[group] = {
            "profiles": profile_stats,
            "different_move_pair_count": len(group_pairs),
            "candidate_wins": candidate_wins,
            "current_wins": current_wins,
            "candidate_median_cpl_not_more": median_non_worse,
            "candidate_harmful_mate_not_more": no_added_harmful,
            "net_positive_signal": (
                candidate_wins > current_wins and median_non_worse and no_added_harmful
            ),
        }
    return output


def evaluate_decision(
    decision_weighted: dict[str, dict[str, Any]],
    position_level: dict[str, list[dict[str, Any]]],
    pairs: dict[str, Any],
    groups: dict[str, Any],
) -> dict[str, Any]:
    current = decision_weighted[PROFILE_CURRENT]
    candidate = decision_weighted[PROFILE_CANDIDATE]
    reasons: list[str] = []
    criteria = {
        "harmful_mate_not_more": candidate["harmful_mate"] <= current["harmful_mate"],
        "missed_winning_mate_not_more": candidate["missed_winning_mate"] <= current["missed_winning_mate"],
        "entered_losing_mate_not_more": candidate["entered_losing_mate"] <= current["entered_losing_mate"],
        "decision_weighted_median_cpl_not_more": (
            candidate["median_cpl"] is not None
            and current["median_cpl"] is not None
            and candidate["median_cpl"] <= current["median_cpl"]
        ),
        "decision_weighted_p90_cpl_not_more": (
            candidate["p90_cpl"] is not None
            and current["p90_cpl"] is not None
            and candidate["p90_cpl"] <= current["p90_cpl"]
        ),
        "different_move_candidate_wins_more": pairs["candidate_wins"] > pairs["current_wins"],
        "neutral_no_added_harmful_mate": groups["neutral-control"]["candidate_harmful_mate_not_more"],
        "neutral_no_added_300_cpl": (
            groups["neutral-control"]["profiles"][PROFILE_CANDIDATE]["cpl_300_plus"]
            <= groups["neutral-control"]["profiles"][PROFILE_CURRENT]["cpl_300_plus"]
        ),
        "development_coordination_net_positive": groups["development-coordination"]["net_positive_signal"],
        "one_pawn_or_rook_net_positive": (
            groups["pawn-structure-pawn-push"]["net_positive_signal"]
            or groups["rook-activity"]["net_positive_signal"]
        ),
    }
    messages = {
        "harmful_mate_not_more": "candidate has more harmful mate outcomes",
        "missed_winning_mate_not_more": "candidate misses more winning mates",
        "entered_losing_mate_not_more": "candidate enters more losing mates",
        "decision_weighted_median_cpl_not_more": "candidate median CPL is higher",
        "decision_weighted_p90_cpl_not_more": "candidate P90 CPL is higher",
        "different_move_candidate_wins_more": "candidate does not win more directly comparable different-move pairs",
        "neutral_no_added_harmful_mate": "neutral-control adds harmful mate outcomes",
        "neutral_no_added_300_cpl": "neutral-control adds 300+ CPL outcomes",
        "development_coordination_net_positive": "development-coordination has no net positive signal",
        "one_pawn_or_rook_net_positive": "neither pawn-structure nor rook-activity has a net positive signal",
    }
    reasons.extend(messages[name] for name, passed in criteria.items() if not passed)
    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "criteria": criteria,
        "candidate_minus_current": {
            "median_cpl": (
                candidate["median_cpl"] - current["median_cpl"]
                if candidate["median_cpl"] is not None and current["median_cpl"] is not None
                else None
            ),
            "p90_cpl": (
                candidate["p90_cpl"] - current["p90_cpl"]
                if candidate["p90_cpl"] is not None and current["p90_cpl"] is not None
                else None
            ),
            "harmful_mate": candidate["harmful_mate"] - current["harmful_mate"],
            "missed_winning_mate": candidate["missed_winning_mate"] - current["missed_winning_mate"],
            "entered_losing_mate": candidate["entered_losing_mate"] - current["entered_losing_mate"],
        },
        "position_level_counts": {
            profile: len(position_level[profile]) for profile in PROFILES
        },
    }


def build_payload(
    source_path: Path,
    source_data: dict[str, Any],
    manifest_path: Path,
    executable: Path,
    analyzed: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
    search_payload: list[dict[str, Any]],
    identity: list[str],
) -> dict[str, Any]:
    decision_weighted = {
        profile: _stats(row for row in analyzed if row["engine_profile"] == profile)
        for profile in PROFILES
    }
    pairs = pair_comparison(analyzed)
    positions = position_level_summary(analyzed)
    groups = group_summary(analyzed, pairs)
    decision = evaluate_decision(decision_weighted, positions, pairs, groups)
    unrestricted = [item for item in search_payload if "move" not in item]
    forced = [item for item in search_payload if "move" in item]
    return {
        "schema_version": 1,
        "status": "S2.2B_TEACHER_GATE_COMPLETE",
        "analysis": "S2.2b integrated evaluation Stockfish CP/mate teacher gate",
        "score_model": "raw centipawn loss plus explicit mate outcomes",
        "wdl_enabled": False,
        "wdl_used": False,
        "source_gate": {
            "artifact_path": str(source_path),
            "artifact_sha256": sha256_file(source_path),
            "git_sha": source_data["git_sha"],
            "engine_sha256": source_data["engine"]["sha256"],
            "manifest_sha256": source_data["manifest"]["sha256"],
            "rows": len(mapping),
            "positions": EXPECTED_POSITION_COUNT,
            "profiles": list(PROFILES),
            "time_limits_ms": list(EXPECTED_TIME_MS),
            "manifest_path": str(manifest_path),
            "source_decision_mapping": mapping,
        },
        "stockfish": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "identity": identity,
            "nodes_per_search": TEACHER_NODES,
            "hash_mb": TEACHER_HASH_MB,
            "threads": TEACHER_THREADS,
            "multi_pv": 1,
            "uci_show_wdl": False,
            "options": TEACHER_OPTIONS,
            "clear_hash_before_each_search": True,
            "unrestricted_searches": len(unrestricted),
            "forced_searches": len(forced),
            "total_searches": len(search_payload),
        },
        "searches": {
            "unrestricted": unrestricted,
            "forced": forced,
        },
        "rows": analyzed,
        "summary": {
            "decision_weighted": decision_weighted,
            "position_level": positions,
            "group": groups,
            "pair_comparison": pairs,
        },
        "decision": decision,
        "interpretation": [
            "This is a fixed-node Stockfish CP/mate teacher gate, not an Elo/SPRT result.",
            "Current remains unchanged; current-eval2 remains candidate-only.",
            "WDL was disabled and no WDL-derived metric was computed.",
            "Games, Elo, and SPRT were not run.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--stockfish",
        type=Path,
        required=True,
        help="Stockfish 18 executable",
    )
    command.add_argument(
        "--input",
        type=Path,
        default=Path("results/s2.2a/integrated-fixed-time.json"),
    )
    command.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/data/s2.2a-integrated-positions.json"),
    )
    command.add_argument(
        "--output",
        type=Path,
        default=Path("results/s2.2b/teacher-rescore-500k.json"),
    )
    command.add_argument("--timeout-s", type=float, default=180.0)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.timeout_s <= 0:
        print("timeout must be positive", file=sys.stderr)
        return 2
    try:
        source_path = args.input.expanduser().resolve()
        manifest_path = args.manifest.expanduser().resolve()
        executable = args.stockfish.expanduser().resolve()
        require(source_path.is_file(), f"source artifact does not exist: {source_path}")
        require(manifest_path.is_file(), f"manifest does not exist: {manifest_path}")
        require(executable.is_file(), f"Stockfish does not exist: {executable}")
        source_data, rows, _ = load_source_artifact(source_path, manifest_path)
        repo = Path(__file__).resolve().parent.parent
        analyzed, mapping, search_payload, identity = run_teacher(rows, executable, args.timeout_s)
        payload = build_payload(
            source_path,
            source_data,
            manifest_path,
            executable,
            analyzed,
            mapping,
            search_payload,
            identity,
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload["decision"], indent=2, sort_keys=True))
        print(f"wrote {output}")
        return 0 if payload["decision"]["status"] == "PASS" else 3
    except (OSError, ValueError, StockfishAnalysisError, subprocess.SubprocessError) as exc:
        print(f"S22B_TEACHER_GATE_FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
