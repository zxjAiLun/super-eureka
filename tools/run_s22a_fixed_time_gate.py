#!/usr/bin/env python3
"""Run the S2.2a integrated positional evaluation fixed-time gate.

This gate compares only the approved production ``current`` profile with the
bench-only ``current-eval2`` profile.  It runs the same release binary in a
fresh process for every position/profile/time combination, with a cold 16 MB
TT and one search thread.  The result is a performance and isolation gate;
it does not run Stockfish, games, Elo, or SPRT.
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

from run_s21_practical_gate import (
    git_sha,
    parse_info_depth_lines,
    parse_key_values,
    parse_score,
    probe_uci,
    sha256_file,
)


PROFILE_CURRENT = "current"
PROFILE_CANDIDATE = "current-eval2"
PROFILES = (PROFILE_CURRENT, PROFILE_CANDIDATE)
EXPECTED_GROUPS = (
    "development-coordination",
    "pawn-structure-pawn-push",
    "rook-activity",
    "king-safety-defensive-resource",
    "neutral-control",
)
EXPECTED_TIME_MS = (1000, 3000, 10000)
EXPECTED_POSITION_COUNT = 25
EXPECTED_ROW_COUNT = EXPECTED_POSITION_COUNT * len(PROFILES) * len(EXPECTED_TIME_MS)
EXPECTED_RETAINED_S21_IDS = {
    "s21-attack-345-78",
    "s21-attack-616-31",
    "s21-attack-621-38",
    "s21-defense-607-19",
    "s21-defense-293-39",
    "s21-defense-143-7",
    "s21-control-559-9",
    "s21-control-237-32",
    "s21-control-467-16",
}
EXPECTED_AUTHORED_COUNT = 16

# These counters must remain zero for both profiles.  Evaluation terms are
# intentionally not counters: changing score/PV is the candidate's purpose.
FEATURE_COUNTERS = (
    "aspiration_retries",
    "aspiration_fail_low",
    "aspiration_fail_high",
    "lmr_reductions",
    "lmr_researches",
    "null_move_attempts",
    "null_move_fail_highs",
    "null_move_researches",
    "futility_pruned",
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
)

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
        f"S2.2a manifest must contain exactly {EXPECTED_POSITION_COUNT} positions",
    )
    require(tuple(data.get("groups", ())) == EXPECTED_GROUPS, "S2.2a group order is not pinned")

    ids = [item.get("id") for item in positions]
    require(all(isinstance(item, str) for item in ids), "S2.2a IDs must be strings")
    require(len(set(ids)) == EXPECTED_POSITION_COUNT, "S2.2a position IDs must be unique")
    require(set(ids[:9]) == EXPECTED_RETAINED_S21_IDS, "the first nine positions must retain S2.1 IDs")
    authored_ids = {item for item in ids if item.startswith("s22a-")}
    require(len(authored_ids) == EXPECTED_AUTHORED_COUNT, "S2.2a authored position count is not 16")
    require(
        set(ids) == EXPECTED_RETAINED_S21_IDS | authored_ids,
        "S2.2a contains an unexpected position ID",
    )

    require(
        {item.get("group") for item in positions} == set(EXPECTED_GROUPS),
        "S2.2a groups are incomplete",
    )
    for group in EXPECTED_GROUPS:
        count = sum(item.get("group") == group for item in positions)
        require(count == 5, f"S2.2a group {group} must contain 5 positions, got {count}")
    fens = set()
    for item in positions:
        require(isinstance(item.get("fen"), str) and item["fen"].strip(), "position FEN is missing")
        require(item["fen"] not in fens, f"duplicate S2.2a FEN: {item['id']}")
        fens.add(item["fen"])
        require(isinstance(item.get("source"), str) and item["source"], "position source is missing")
        require(
            isinstance(item.get("diagnostic_purpose"), str) and item["diagnostic_purpose"],
            "position diagnostic_purpose is missing",
        )
    return positions


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
            "S2.2a search failed: "
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
        raise RuntimeError(f"S2.2a bench result missing fields {missing}: {result_lines[0]}")
    if fields["profile"] != profile:
        raise RuntimeError(
            f"S2.2a profile mismatch: expected {profile}, got {fields['profile']}"
        )

    numeric_fields = {
        key: int(fields[key])
        for key in ("elapsed_ms", "nps", *SEARCH_METRICS, *FEATURE_COUNTERS)
    }
    return {
        "position_id": position["id"],
        "group": position["group"],
        "fen": position["fen"],
        "source": position["source"],
        "diagnostic_purpose": position["diagnostic_purpose"],
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
        "pv": fields["pv"].split() if fields["pv"] else [],
        "score_by_depth": parse_info_depth_lines(completed.stdout),
        "metrics": {key: numeric_fields[key] for key in SEARCH_METRICS},
        "counters": {key: numeric_fields[key] for key in FEATURE_COUNTERS},
        "raw_result": result_lines[0],
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == EXPECTED_ROW_COUNT, f"S2.2a must contain {EXPECTED_ROW_COUNT} rows")
    identities = {
        (str(row["position_id"]), int(row["time_limit_ms"]), str(row["engine_profile"]))
        for row in rows
    }
    require(len(identities) == EXPECTED_ROW_COUNT, "duplicate S2.2a position/time/profile row")
    require(
        {str(row["position_id"]) for row in rows} == EXPECTED_RETAINED_S21_IDS
        | {str(row["position_id"]) for row in rows if str(row["position_id"]).startswith("s22a-")},
        "S2.2a row position set is incomplete",
    )
    position_indices = {
        str(row["position_id"]): int(row["position_index"]) for row in rows
    }
    for time_index, time_ms in enumerate(EXPECTED_TIME_MS):
        for position_id in sorted(position_indices):
            position_index = position_indices[position_id]
            group = [
                row
                for row in rows
                if int(row["time_index"]) == time_index
                and int(row["time_limit_ms"]) == time_ms
                and str(row["position_id"]) == position_id
            ]
            require(len(group) == 2, f"missing S2.2a pair for {position_id} at {time_ms}ms")
            actual_order = tuple(
                row["engine_profile"] for row in sorted(group, key=lambda item: int(item["run_order"]))
            )
            expected_order = rotated_profile_order(time_index, position_index)
            require(
                actual_order == expected_order,
                f"position={position_id} time={time_ms} did not use rotated order",
            )


def median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _profile_summary(rows: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    selected = [row for row in rows if row["engine_profile"] == profile]
    return {
        "positions": len(selected),
        "median_completed_depth": median([int(row["completed_depth"]) for row in selected]),
        "median_nps": median([int(row["nps"]) for row in selected]),
        "median_nodes": median([int(row["nodes"]) for row in selected]),
        "median_qsearch_nodes": median([int(row["qsearch_nodes"]) for row in selected]),
        "median_eval_calls": median([int(row["eval_calls"]) for row in selected]),
        "median_elapsed_ms": median([int(row["elapsed_ms"]) for row in selected]),
        "counter_sums": {
            name: sum(int(row["counters"][name]) for row in selected)
            for name in FEATURE_COUNTERS
        },
    }


def _summary_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = {profile: _profile_summary(rows, profile) for profile in PROFILES}
    pairs = []
    for position_id in sorted({str(row["position_id"]) for row in rows}):
        pair = {
            row["engine_profile"]: row
            for row in rows
            if str(row["position_id"]) == position_id
        }
        if set(pair) == set(PROFILES):
            delta = int(pair[PROFILE_CANDIDATE]["completed_depth"]) - int(
                pair[PROFILE_CURRENT]["completed_depth"]
            )
            pairs.append(
                {
                    "position_id": position_id,
                    "current_depth": pair[PROFILE_CURRENT]["completed_depth"],
                    "candidate_depth": pair[PROFILE_CANDIDATE]["completed_depth"],
                    "depth_delta": delta,
                    "current_bestmove": pair[PROFILE_CURRENT]["bestmove"],
                    "candidate_bestmove": pair[PROFILE_CANDIDATE]["bestmove"],
                }
            )
    deltas = [int(pair["depth_delta"]) for pair in pairs]
    return {
        "pair_count": len(pairs),
        "candidate_deeper": sum(delta > 0 for delta in deltas),
        "equal": sum(delta == 0 for delta in deltas),
        "current_deeper": sum(delta < 0 for delta in deltas),
        "pairwise_depth_deltas": deltas,
        "profiles": profiles,
        "pairs": pairs,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_rows(rows)
    by_time = {
        str(time_ms): _summary_for_rows(
            [row for row in rows if int(row["time_limit_ms"]) == time_ms]
        )
        for time_ms in EXPECTED_TIME_MS
    }
    by_group = {
        group: {
            str(time_ms): _summary_for_rows(
                [
                    row
                    for row in rows
                    if row["group"] == group and int(row["time_limit_ms"]) == time_ms
                ]
            )
            for time_ms in EXPECTED_TIME_MS
        }
        for group in EXPECTED_GROUPS
    }
    cross_time: dict[str, list[int]] = {}
    for position_id in sorted({str(row["position_id"]) for row in rows}):
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
    all_pairs = _summary_for_rows(rows)
    return {
        "by_time_ms": by_time,
        "by_group": by_group,
        "all_pairs": {
            "count": len(all_pairs["pairs"]),
            "candidate_deeper": all_pairs["candidate_deeper"],
            "equal": all_pairs["equal"],
            "current_deeper": all_pairs["current_deeper"],
            "pairwise_depth_deltas": all_pairs["pairwise_depth_deltas"],
        },
        "cross_time_by_position": cross_time,
        "stable_two_plus_regression_positions": [
            position_id
            for position_id, deltas in cross_time.items()
            if all(delta <= -2 for delta in deltas)
        ],
    }


def counter_violations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for row in rows:
        nonzero = {
            name: int(row["counters"][name])
            for name in FEATURE_COUNTERS
            if int(row["counters"][name]) != 0
        }
        if nonzero:
            violations.append(
                {
                    "position_id": row["position_id"],
                    "time_limit_ms": row["time_limit_ms"],
                    "profile": row["engine_profile"],
                    "counters": nonzero,
                }
            )
    return violations


def focus_quality_flags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    for time_ms in EXPECTED_TIME_MS:
        pair = {
            row["engine_profile"]: row
            for row in rows
            if row["position_id"] == "s21-control-237-32"
            and int(row["time_limit_ms"]) == time_ms
        }
        if set(pair) == set(PROFILES):
            if (
                pair[PROFILE_CURRENT]["bestmove"] == "g8f8"
                and pair[PROFILE_CANDIDATE]["bestmove"] == "g8h8"
            ):
                flags.append(
                    {
                        "position_id": "s21-control-237-32",
                        "time_limit_ms": time_ms,
                        "current": "g8f8",
                        "candidate": "g8h8",
                        "kind": "diagnostic_quality_flag",
                    }
                )
    return flags


def evaluate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    reasons: list[str] = []
    depth_ok: dict[str, bool] = {}
    nps_ok: dict[str, bool] = {}
    nps_ratios: dict[str, float | None] = {}
    for time_ms in EXPECTED_TIME_MS:
        current = summary["by_time_ms"][str(time_ms)]["profiles"][PROFILE_CURRENT]
        candidate = summary["by_time_ms"][str(time_ms)]["profiles"][PROFILE_CANDIDATE]
        depth_ok[str(time_ms)] = candidate["median_completed_depth"] >= current[
            "median_completed_depth"
        ] - 1
        if not depth_ok[str(time_ms)]:
            reasons.append(f"candidate median depth is more than one lower at {time_ms}ms")
        ratio = candidate["median_nps"] / current["median_nps"] if current["median_nps"] else None
        nps_ratios[str(time_ms)] = ratio
        nps_ok[str(time_ms)] = ratio is not None and ratio >= 0.65
        if not nps_ok[str(time_ms)]:
            reasons.append(f"candidate median NPS is below 0.65x Current at {time_ms}ms")

    stable_two_plus = summary["stable_two_plus_regression_positions"]
    if stable_two_plus:
        reasons.append("a position has a stable two-or-more-depth regression")
    violations = counter_violations(rows)
    if violations:
        reasons.append("feature-isolation counters are non-zero")

    quality_flags = focus_quality_flags(rows)
    status = "FAIL" if reasons else ("PASS_WITH_QUALITY_FLAGS" if quality_flags else "PASS")
    return {
        "status": status,
        "reasons": reasons,
        "quality_flags": quality_flags,
        "criteria": {
            "candidate_median_depth_within_one": depth_ok,
            "candidate_median_nps_at_least_0_65": nps_ok,
            "median_nps_ratios": nps_ratios,
            "no_stable_two_plus_depth_regression": not stable_two_plus,
            "feature_counters_all_zero": not violations,
        },
        "counter_violations": violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("tests/data/s2.2a-integrated-positions.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2.2a/integrated-fixed-time.json")
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
        print("S2.2a time set is fixed at 1000 3000 10000 ms", file=sys.stderr)
        return 2

    try:
        positions = load_manifest(manifest)
        probes = {
            profile: probe_uci(engine, profile)
            for profile in PROFILES
        }
        rows: list[dict[str, Any]] = []
        for time_index, time_ms in enumerate(EXPECTED_TIME_MS):
            for position_index, position in enumerate(positions):
                order = rotated_profile_order(time_index, position_index)
                for profile in order:
                    print(
                        f"s2.2a time_ms={time_ms} position={position['id']} "
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
            "analysis": "S2.2a integrated positional evaluation fixed-time gate",
            "git_sha": git_sha(repo),
            "manifest": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "position_count": len(positions),
                "groups": {group: 5 for group in EXPECTED_GROUPS},
                "retained_s21_position_count": len(EXPECTED_RETAINED_S21_IDS),
                "authored_position_count": EXPECTED_AUTHORED_COUNT,
            },
            "engine": {
                "path": str(engine),
                "sha256": sha256_file(engine),
                "profiles": list(PROFILES),
                "uci_probes": probes,
                "same_binary_for_both_profiles": True,
            },
            "limits_ms": list(EXPECTED_TIME_MS),
            "tt": {"mode": "cold", "hash_mb": 16, "threads": 1},
            "rotation": {
                "enabled": True,
                "rule": "reverse profile order when (time_index + position_index) is odd",
                "base_order": list(PROFILES),
            },
            "rows": rows,
            "summary": summarize(rows),
            "decision": decision,
            "interpretation": [
                "This is a fixed-time performance and feature-isolation gate, not an Elo/SPRT result.",
                "Current remains unchanged; current-eval2 remains candidate-only.",
                "The retained S2.1 positions are diagnostic focus cases, not Stockfish oracle gates.",
                "Stockfish, games, and Elo/SPRT are not run by this tool.",
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"decision={decision['status']}")
        print(f"wrote {output}")
        return 0 if decision["status"] != "FAIL" else 3
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s2.2a_error {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
