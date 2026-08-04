#!/usr/bin/env python3
"""Run the S3-FINAL fixed-time depth gate.

This is a bounded search-efficiency gate, not an Elo, Stockfish, or game
result.  It compares the approved ``current`` profile with the explicitly
selected ``current-final`` combination on the committed S2.2a position set.
Every row starts a fresh process with a cold 16 MB TT and one thread.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import chess

from run_s22a_fixed_time_gate import (
    git_sha,
    parse_info_depth_lines,
    parse_key_values,
    parse_score,
    probe_uci,
    sha256_file,
)


PROFILE_CURRENT = "current"
PROFILE_CANDIDATE = "current-final"
PROFILES = (PROFILE_CURRENT, PROFILE_CANDIDATE)
EXPECTED_GROUPS = (
    "development-coordination",
    "pawn-structure-pawn-push",
    "rook-activity",
    "king-safety-defensive-resource",
    "neutral-control",
)
EXPECTED_TIME_MS = (1000, 3000)
EXPECTED_POSITION_COUNT = 25
EXPECTED_ROW_COUNT = EXPECTED_POSITION_COUNT * len(PROFILES) * len(EXPECTED_TIME_MS)

SEARCH_METRICS = (
    "completed_depth",
    "nodes",
    "qsearch_nodes",
    "eval_calls",
    "legal_move_generations",
    "pseudo_moves",
    "legal_moves",
    "make_moves",
    "unmake_moves",
    "tt_probes",
    "tt_hits",
    "tt_cutoffs",
    "tt_stores",
)

FEATURE_COUNTERS = (
    "see_calls",
    "see_pruned",
    "qsearch_see_tests",
    "qsearch_see_pruned",
    "qsearch_see_fail_open_promotions",
    "qsearch_checking_captures_kept",
    "qsearch_promotions_kept",
    "qsearch_en_passant_kept",
    "check_extensions",
    "single_evasion_extensions",
    "qsearch_check_moves",
    "threat_ordered_moves",
    "root_reorders",
    "aspiration_retries",
    "aspiration_fail_low",
    "aspiration_fail_high",
    "lmr_reductions",
    "lmr_researches",
    "null_move_attempts",
    "null_move_fail_highs",
    "null_move_researches",
    "futility_pruned",
)

# These counters describe the candidate features inherited by CurrentFinal.
# An attempt/reduction/test counter is required to be non-zero somewhere in
# the matrix; a fail-high, re-search, or actual prune is not guaranteed by a
# fixed corpus and is therefore reported but not made a hard activity gate.
AUTHORIZED_ACTIVITY_COUNTERS = (
    "aspiration_retries",
    "lmr_reductions",
    "null_move_attempts",
    "futility_pruned",
    "qsearch_see_tests",
)

# Current must not inherit any of the candidate search features.
CURRENT_MUST_ZERO_COUNTERS = (
    "see_calls",
    "see_pruned",
    "qsearch_see_tests",
    "qsearch_see_pruned",
    "qsearch_see_fail_open_promotions",
    "aspiration_retries",
    "aspiration_fail_low",
    "aspiration_fail_high",
    "lmr_reductions",
    "lmr_researches",
    "null_move_attempts",
    "null_move_fail_highs",
    "null_move_researches",
    "futility_pruned",
)

# CurrentFinal deliberately does not enable the old SEE ordering candidate or
# any threat/forcing/root-reorder path.  Specialized qsearch movegen counters
# remain valid for both profiles and are not included here.
CANDIDATE_MUST_ZERO_COUNTERS = (
    "see_calls",
    "see_pruned",
    "check_extensions",
    "single_evasion_extensions",
    "qsearch_check_moves",
    "threat_ordered_moves",
    "root_reorders",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rotated_profile_order(time_index: int, position_index: int) -> tuple[str, ...]:
    if (time_index + position_index) % 2 == 0:
        return PROFILES
    return PROFILES[::-1]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = data.get("positions")
    require(
        isinstance(positions, list) and len(positions) == EXPECTED_POSITION_COUNT,
        f"S3-FINAL manifest must contain exactly {EXPECTED_POSITION_COUNT} positions",
    )
    require(tuple(data.get("groups", ())) == EXPECTED_GROUPS, "S3-FINAL group order is not pinned")
    ids = [item.get("id") for item in positions]
    require(all(isinstance(item, str) for item in ids), "position IDs must be strings")
    require(len(set(ids)) == EXPECTED_POSITION_COUNT, "position IDs must be unique")
    fens: set[str] = set()
    for item in positions:
        require(isinstance(item.get("fen"), str) and item["fen"].strip(), "position FEN is missing")
        require(item["fen"] not in fens, f"duplicate position FEN: {item['id']}")
        fens.add(item["fen"])
        require(isinstance(item.get("group"), str), f"position group is missing: {item['id']}")
        require(item["group"] in EXPECTED_GROUPS, f"unknown position group: {item['id']}")
    for group in EXPECTED_GROUPS:
        require(
            sum(item["group"] == group for item in positions) == 5,
            f"S3-FINAL group {group} must contain 5 positions",
        )
    return positions


def validate_bestmove_and_pv(fen: str, bestmove: str, pv: list[str]) -> None:
    board = chess.Board(fen)
    if bestmove == "0000":
        require(not any(board.legal_moves), "bestmove 0000 is only valid for a terminal position")
        require(not pv, "terminal bestmove must have an empty PV")
        return
    require(pv, "non-terminal search must report a non-empty PV")
    require(pv[0] == bestmove, "PV first move must equal bestmove")
    for index, token in enumerate(pv):
        try:
            move = chess.Move.from_uci(token)
        except ValueError as exc:
            raise ValueError(f"invalid PV move {token!r} at ply {index}") from exc
        require(board.is_legal(move), f"illegal PV move {token!r} at ply {index}")
        board.push(move)


def run_search(
    engine: Path,
    profile: str,
    position: dict[str, Any],
    time_ms: int,
    repo: Path,
) -> dict[str, Any]:
    argv = [
        str(engine),
        "bench",
        "profile",
        "--mode",
        "cold",
        "--movetime",
        str(time_ms),
        "--profile",
        profile,
        "--fen",
        position["fen"],
    ]
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=max(30, time_ms / 1000 + 20),
        check=False,
    )
    wall_time_ms = round((time.monotonic() - started) * 1000, 3)
    result_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("bench_result ")
    ]
    if completed.returncode != 0 or len(result_lines) != 1 or completed.stderr:
        raise RuntimeError(
            "S3-FINAL search failed: "
            + json.dumps(
                {
                    "argv": argv,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr,
                    "stdout_tail": completed.stdout.splitlines()[-20:],
                },
                ensure_ascii=False,
            )
        )

    fields = parse_key_values(result_lines[0])
    required = {
        "profile",
        "score",
        "bestmove",
        "pv",
        "elapsed_ms",
        "nps",
        *SEARCH_METRICS,
        *FEATURE_COUNTERS,
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise RuntimeError(f"S3-FINAL bench result missing fields {missing}: {result_lines[0]}")
    require(fields["profile"] == profile, f"profile mismatch: expected {profile}, got {fields['profile']}")

    numeric_fields = {
        key: int(fields[key])
        for key in ("elapsed_ms", "nps", *SEARCH_METRICS, *FEATURE_COUNTERS)
    }
    pv = fields["pv"].split() if fields["pv"] else []
    validate_bestmove_and_pv(position["fen"], fields["bestmove"], pv)
    require(numeric_fields["completed_depth"] > 0, "non-terminal S3-FINAL position completed no depth")
    return {
        "position_id": position["id"],
        "group": position["group"],
        "fen": position["fen"],
        "time_limit_ms": time_ms,
        "engine_profile": profile,
        "engine_path": str(engine),
        "argv": argv,
        "returncode": completed.returncode,
        "wall_time_ms": wall_time_ms,
        "score": parse_score(fields["score"]),
        "bestmove": fields["bestmove"],
        "completed_depth": numeric_fields["completed_depth"],
        "nodes": numeric_fields["nodes"],
        "qsearch_nodes": numeric_fields["qsearch_nodes"],
        "eval_calls": numeric_fields["eval_calls"],
        "nps": numeric_fields["nps"],
        "elapsed_ms": numeric_fields["elapsed_ms"],
        "pv": pv,
        "score_by_depth": parse_info_depth_lines(completed.stdout),
        "metrics": {key: numeric_fields[key] for key in SEARCH_METRICS},
        "counters": {key: numeric_fields[key] for key in FEATURE_COUNTERS},
        "raw_result": result_lines[0],
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == EXPECTED_ROW_COUNT, f"S3-FINAL must contain {EXPECTED_ROW_COUNT} rows")
    identities = {
        (str(row["position_id"]), int(row["time_limit_ms"]), str(row["engine_profile"]))
        for row in rows
    }
    require(len(identities) == EXPECTED_ROW_COUNT, "duplicate S3-FINAL position/time/profile row")
    for time_index, time_ms in enumerate(EXPECTED_TIME_MS):
        for position_index, position_id in enumerate(
            sorted({str(row["position_id"]) for row in rows})
        ):
            pair = [
                row
                for row in rows
                if int(row["time_index"]) == time_index
                and int(row["time_limit_ms"]) == time_ms
                and str(row["position_id"]) == position_id
            ]
            require(len(pair) == 2, f"missing pair for {position_id} at {time_ms}ms")
            actual_order = tuple(
                row["engine_profile"] for row in sorted(pair, key=lambda item: int(item["run_order"]))
            )
            expected_order = rotated_profile_order(time_index, int(pair[0]["position_index"]))
            require(actual_order == expected_order, f"profile order mismatch for {position_id} at {time_ms}ms")


def _median(values: list[int]) -> float:
    require(values, "cannot compute a median of an empty result set")
    return float(statistics.median(values))


def _profile_summary(rows: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    selected = [row for row in rows if row["engine_profile"] == profile]
    return {
        "rows": len(selected),
        "median_completed_depth": _median([int(row["completed_depth"]) for row in selected]),
        "median_nodes": _median([int(row["nodes"]) for row in selected]),
        "median_qsearch_nodes": _median([int(row["qsearch_nodes"]) for row in selected]),
        "median_elapsed_ms": _median([int(row["elapsed_ms"]) for row in selected]),
        "median_nps": _median([int(row["nps"]) for row in selected]),
        "counter_sums": {
            name: sum(int(row["counters"][name]) for row in selected)
            for name in FEATURE_COUNTERS
        },
    }


def summary_by_time(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_time: dict[str, Any] = {}
    pairs: list[dict[str, Any]] = []
    for time_ms in EXPECTED_TIME_MS:
        selected = [row for row in rows if int(row["time_limit_ms"]) == time_ms]
        by_time[str(time_ms)] = {
            "profiles": {
                profile: _profile_summary(selected, profile) for profile in PROFILES
            }
        }
        for position_id in sorted({str(row["position_id"]) for row in selected}):
            pair = {
                row["engine_profile"]: row
                for row in selected
                if str(row["position_id"]) == position_id
            }
            if set(pair) != set(PROFILES):
                continue
            pairs.append(
                {
                    "time_limit_ms": time_ms,
                    "position_id": position_id,
                    "current_depth": int(pair[PROFILE_CURRENT]["completed_depth"]),
                    "candidate_depth": int(pair[PROFILE_CANDIDATE]["completed_depth"]),
                    "depth_delta": int(pair[PROFILE_CANDIDATE]["completed_depth"])
                    - int(pair[PROFILE_CURRENT]["completed_depth"]),
                    "current_bestmove": pair[PROFILE_CURRENT]["bestmove"],
                    "candidate_bestmove": pair[PROFILE_CANDIDATE]["bestmove"],
                }
            )
    cross_time: dict[str, list[int]] = {}
    position_ids = sorted({str(row["position_id"]) for row in rows})
    for position_id in position_ids:
        deltas = []
        for time_ms in EXPECTED_TIME_MS:
            pair = {
                row["engine_profile"]: row
                for row in rows
                if str(row["position_id"]) == position_id
                and int(row["time_limit_ms"]) == time_ms
            }
            deltas.append(
                int(pair[PROFILE_CANDIDATE]["completed_depth"])
                - int(pair[PROFILE_CURRENT]["completed_depth"])
            )
        cross_time[position_id] = deltas
    return {
        "by_time_ms": by_time,
        "pairs": pairs,
        "cross_time_depth_delta_by_position": cross_time,
        "stable_two_plus_regression_positions": [
            position_id
            for position_id, deltas in cross_time.items()
            if all(delta <= -2 for delta in deltas)
        ],
    }


def counter_violations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current = []
    candidate = []
    for row in rows:
        nonzero_current = {
            name: int(row["counters"][name])
            for name in CURRENT_MUST_ZERO_COUNTERS
            if row["engine_profile"] == PROFILE_CURRENT and int(row["counters"][name]) != 0
        }
        nonzero_candidate = {
            name: int(row["counters"][name])
            for name in CANDIDATE_MUST_ZERO_COUNTERS
            if row["engine_profile"] == PROFILE_CANDIDATE and int(row["counters"][name]) != 0
        }
        if nonzero_current:
            current.append({"position_id": row["position_id"], "time_limit_ms": row["time_limit_ms"], "counters": nonzero_current})
        if nonzero_candidate:
            candidate.append({"position_id": row["position_id"], "time_limit_ms": row["time_limit_ms"], "counters": nonzero_candidate})
    return {"current": current, "candidate": candidate}


def evaluate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_rows(rows)
    summary = summary_by_time(rows)
    reasons: list[str] = []
    median_depths: dict[str, dict[str, float]] = {}
    median_not_below: dict[str, bool] = {}
    for time_ms in EXPECTED_TIME_MS:
        profiles = summary["by_time_ms"][str(time_ms)]["profiles"]
        current_depth = profiles[PROFILE_CURRENT]["median_completed_depth"]
        candidate_depth = profiles[PROFILE_CANDIDATE]["median_completed_depth"]
        median_depths[str(time_ms)] = {
            PROFILE_CURRENT: current_depth,
            PROFILE_CANDIDATE: candidate_depth,
        }
        median_not_below[str(time_ms)] = candidate_depth >= current_depth
        if not median_not_below[str(time_ms)]:
            reasons.append(f"candidate median depth is below Current at {time_ms}ms")

    uplift_tiers = [
        str(time_ms)
        for time_ms in EXPECTED_TIME_MS
        if median_depths[str(time_ms)][PROFILE_CANDIDATE]
        >= median_depths[str(time_ms)][PROFILE_CURRENT] + 1
    ]
    if not uplift_tiers:
        reasons.append("CurrentFinal has no fixed-time median depth uplift of one full ply")

    stable_regressions = summary["stable_two_plus_regression_positions"]
    if stable_regressions:
        reasons.append("a position is at least two depths behind at both time tiers")

    violations = counter_violations(rows)
    if violations["current"]:
        reasons.append("Current has non-zero candidate-feature counters")
    if violations["candidate"]:
        reasons.append("CurrentFinal has non-zero unauthorized counters")

    candidate_sums = summary["by_time_ms"][str(EXPECTED_TIME_MS[0])]["profiles"][PROFILE_CANDIDATE]["counter_sums"].copy()
    for time_ms in EXPECTED_TIME_MS[1:]:
        sums = summary["by_time_ms"][str(time_ms)]["profiles"][PROFILE_CANDIDATE]["counter_sums"]
        for name in FEATURE_COUNTERS:
            candidate_sums[name] += sums[name]
    missing_activity = {
        name: candidate_sums[name]
        for name in AUTHORIZED_ACTIVITY_COUNTERS
        if candidate_sums[name] <= 0
    }
    if missing_activity:
        reasons.append("CurrentFinal enabled-feature counters did not all fire")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "criteria": {
            "candidate_median_not_below_current": median_not_below,
            "median_depth_uplift_tiers": uplift_tiers,
            "no_stable_two_plus_regression": not stable_regressions,
            "current_candidate_counters_zero": not violations["current"],
            "candidate_unauthorized_counters_zero": not violations["candidate"],
            "authorized_feature_activity_nonzero": not missing_activity,
        },
        "median_depths": median_depths,
        "stable_two_plus_regression_positions": stable_regressions,
        "counter_violations": violations,
        "missing_authorized_activity": missing_activity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("tests/data/s2.2a-integrated-positions.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/s3-final/depth-gate.json")
    )
    parser.add_argument("--time-ms", type=int, nargs="+", default=list(EXPECTED_TIME_MS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    engine = args.engine.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not engine.is_file() or not manifest.is_file():
        print("engine and manifest must point to files", file=sys.stderr)
        return 2
    if tuple(args.time_ms) != EXPECTED_TIME_MS:
        print("S3-FINAL time set is fixed at 1000 3000 ms", file=sys.stderr)
        return 2

    try:
        positions = load_manifest(manifest)
        probes = {profile: probe_uci(engine, profile) for profile in PROFILES}
        rows: list[dict[str, Any]] = []
        for time_index, time_ms in enumerate(EXPECTED_TIME_MS):
            for position_index, position in enumerate(positions):
                order = rotated_profile_order(time_index, position_index)
                for profile in order:
                    print(
                        f"s3-final time_ms={time_ms} position={position['id']} "
                        f"profile={profile} order={order}",
                        flush=True,
                    )
                    row = run_search(engine, profile, position, time_ms, repo)
                    row["time_index"] = time_index
                    row["position_index"] = position_index
                    row["run_order"] = len(rows)
                    row["rotated_profile_order"] = list(order)
                    rows.append(row)
        validate_rows(rows)
        decision = evaluate_gate(rows)
        artifact = {
            "schema_version": 1,
            "status": "GATE_RESULT",
            "analysis": "S3-FINAL integrated selective-search fixed-time depth gate",
            "git_sha": git_sha(repo),
            "manifest": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "position_count": len(positions),
                "groups": {group: 5 for group in EXPECTED_GROUPS},
            },
            "engine": {
                "path": str(engine),
                "sha256": sha256_file(engine),
                "profiles": list(PROFILES),
                "uci_probes": probes,
                "same_binary_for_both_profiles": True,
            },
            "limits_ms": list(EXPECTED_TIME_MS),
            "matrix": {
                "row_count": len(rows),
                "fresh_process_per_row": True,
                "tt_mode": "cold",
                "hash_mb": 16,
                "threads": 1,
                "rotation": "reverse profile order when (time_index + position_index) is odd",
            },
            "counter_contract": {
                "authorized_activity": list(AUTHORIZED_ACTIVITY_COUNTERS),
                "current_must_zero": list(CURRENT_MUST_ZERO_COUNTERS),
                "candidate_must_zero": list(CANDIDATE_MUST_ZERO_COUNTERS),
            },
            "rows": rows,
            "summary": summary_by_time(rows),
            "decision": decision,
            "interpretation": [
                "This is a fixed-time search-depth gate, not an Elo/SPRT or game result.",
                "Current remains unchanged; current-final is candidate-only.",
                "No Stockfish, opening match, or E2/threat-aware evaluation is run by this tool.",
                "A PASS authorizes only the separately specified 100-game quick screen.",
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"decision={decision['status']}")
        print(f"wrote {output}")
        return 0 if decision["status"] == "PASS" else 3
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s3-final_error {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
