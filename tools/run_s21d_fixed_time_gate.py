#!/usr/bin/env python3
"""Run the S2.1d king-danger evaluation fixed-time gate.

This gate compares only the approved Current profile with the bench-only
king-danger evaluation-only profile. It uses the nine saved S2.1 positions at
1, 3, and 10 seconds, with a cold 16 MB TT and one thread. Each search is a
fresh process, and the two profiles are reversed by position/time so neither
profile is systematically first. This tool records a decision for the fixed
time gate only; it does not run Stockfish, games, Elo, or SPRT.
"""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_s21_practical_gate import (
    EXPECTED_GROUPS,
    PROFILE_CURRENT,
    git_sha,
    load_manifest,
    parse_info_depth_lines,
    parse_key_values,
    parse_score,
    probe_uci,
    sha256_file,
)


PROFILE_EVAL_ONLY = "current-threat-aware-eval-only"
PROFILES = (PROFILE_CURRENT, PROFILE_EVAL_ONLY)
EXPECTED_POSITION_COUNT = 9
EXPECTED_TIME_MS = (1000, 3000, 10000)
EXPECTED_ROW_COUNT = EXPECTED_POSITION_COUNT * len(PROFILES) * len(EXPECTED_TIME_MS)

# Every counter that would expose accidental feature contamination is part of
# the hard gate, including zero-valued counters not used by this candidate.
FEATURE_COUNTERS = (
    "threat_ordered_moves",
    "root_reorders",
    "check_extensions",
    "single_evasion_extensions",
    "qsearch_check_moves",
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
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rotated_profile_order(time_index: int, position_index: int) -> tuple[str, ...]:
    if (time_index + position_index) % 2 == 0:
        return PROFILES
    return PROFILES[::-1]


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
        line
        for line in completed.stdout.splitlines()
        if line.startswith("bench_result ")
    ]
    if completed.returncode != 0 or len(result_lines) != 1 or completed.stderr:
        raise RuntimeError(
            "S2.1d search failed: "
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
        "completed_depth",
        "score",
        "bestmove",
        "nodes",
        "elapsed_ms",
        "nps",
        "pv",
        *FEATURE_COUNTERS,
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise RuntimeError(f"bench result missing fields {missing}: {result_lines[0]}")
    if fields["profile"] != profile:
        raise RuntimeError(
            f"bench profile mismatch: expected {profile}, got {fields['profile']}"
        )

    numeric = {
        key: int(fields[key])
        for key in ("completed_depth", "nodes", "elapsed_ms", "nps", *FEATURE_COUNTERS)
    }
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
        "source_teacher_move": position["teacher_move"],
        "source_teacher_agreement": fields["bestmove"] == position["teacher_move"],
        "completed_depth": numeric["completed_depth"],
        "nodes": numeric["nodes"],
        "elapsed_ms": numeric["elapsed_ms"],
        "nps": numeric["nps"],
        "pv": fields["pv"].split() if fields["pv"] else [],
        "score_by_depth": parse_info_depth_lines(completed.stdout),
        "counters": {key: numeric[key] for key in FEATURE_COUNTERS},
        "raw_result": result_lines[0],
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == EXPECTED_ROW_COUNT, f"S2.1d must contain {EXPECTED_ROW_COUNT} rows")
    identities = {
        (str(row["position_id"]), int(row["time_limit_ms"]), str(row["engine_profile"]))
        for row in rows
    }
    require(len(identities) == len(rows), "duplicate S2.1d position/time/profile row")
    require(
        {str(row["position_id"]) for row in rows}.__len__() == EXPECTED_POSITION_COUNT,
        "S2.1d must contain exactly 9 positions",
    )
    require(
        {str(row["engine_profile"]) for row in rows} == set(PROFILES),
        "S2.1d profile set is incomplete or unexpected",
    )
    require(
        {int(row["time_limit_ms"]) for row in rows} == set(EXPECTED_TIME_MS),
        "S2.1d time set must be exactly 1s, 3s, and 10s",
    )
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["time_index"]), int(row["position_index"]))
        by_key.setdefault(key, []).append(row)
    require(len(by_key) == EXPECTED_ROW_COUNT // len(PROFILES), "missing run groups")
    for (time_index, position_index), group in by_key.items():
        actual_order = tuple(
            row["engine_profile"] for row in sorted(group, key=lambda item: int(item["run_order"]))
        )
        require(
            actual_order == rotated_profile_order(time_index, position_index),
            f"time={time_index} position={position_index} did not use rotated order",
        )


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


def _median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_rows(rows)
    by_time: dict[str, Any] = {}
    for time_ms in EXPECTED_TIME_MS:
        time_rows = [row for row in rows if int(row["time_limit_ms"]) == time_ms]
        profiles: dict[str, Any] = {}
        for profile in PROFILES:
            selected = [row for row in time_rows if row["engine_profile"] == profile]
            profiles[profile] = {
                "positions": len(selected),
                "median_completed_depth": _median(
                    [int(row["completed_depth"]) for row in selected]
                ),
                "median_nodes": _median([int(row["nodes"]) for row in selected]),
                "median_elapsed_ms": _median(
                    [int(row["elapsed_ms"]) for row in selected]
                ),
                "median_nps": _median([int(row["nps"]) for row in selected]),
                "teacher_agreement_count": sum(
                    bool(row["source_teacher_agreement"]) for row in selected
                ),
                "counter_sums": {
                    name: sum(int(row["counters"][name]) for row in selected)
                    for name in FEATURE_COUNTERS
                },
                "bestmoves": {
                    row["position_id"]: row["bestmove"] for row in selected
                },
            }
        deltas = []
        pairs = []
        for position_id in sorted({row["position_id"] for row in time_rows}):
            pair = {
                row["engine_profile"]: row
                for row in time_rows
                if row["position_id"] == position_id
            }
            if set(pair) != set(PROFILES):
                continue
            delta = (
                int(pair[PROFILE_EVAL_ONLY]["completed_depth"])
                - int(pair[PROFILE_CURRENT]["completed_depth"])
            )
            deltas.append(delta)
            pairs.append(
                {
                    "position_id": position_id,
                    "current_depth": pair[PROFILE_CURRENT]["completed_depth"],
                    "eval_only_depth": pair[PROFILE_EVAL_ONLY]["completed_depth"],
                    "depth_delta": delta,
                    "current_bestmove": pair[PROFILE_CURRENT]["bestmove"],
                    "eval_only_bestmove": pair[PROFILE_EVAL_ONLY]["bestmove"],
                }
            )
        by_time[str(time_ms)] = {
            "pair_count": len(deltas),
            "candidate_deeper": sum(delta > 0 for delta in deltas),
            "equal": sum(delta == 0 for delta in deltas),
            "current_deeper": sum(delta < 0 for delta in deltas),
            "pairwise_depth_deltas": deltas,
            "profiles": profiles,
            "pairs": pairs,
        }

    cross_time: dict[str, list[int]] = {}
    for position_id in sorted({row["position_id"] for row in rows}):
        deltas = []
        for time_ms in EXPECTED_TIME_MS:
            pair = {
                row["engine_profile"]: row
                for row in rows
                if row["position_id"] == position_id
                and int(row["time_limit_ms"]) == time_ms
            }
            deltas.append(
                int(pair[PROFILE_EVAL_ONLY]["completed_depth"])
                - int(pair[PROFILE_CURRENT]["completed_depth"])
            )
        cross_time[position_id] = deltas

    all_deltas = [delta for deltas in cross_time.values() for delta in deltas]
    stable_regression_positions = [
        position_id for position_id, deltas in cross_time.items() if all(delta < 0 for delta in deltas)
    ]
    stable_two_plus_positions = [
        position_id
        for position_id, deltas in cross_time.items()
        if all(delta <= -2 for delta in deltas)
    ]
    return {
        "by_time_ms": by_time,
        "all_pairs": {
            "count": len(all_deltas),
            "candidate_deeper": sum(delta > 0 for delta in all_deltas),
            "equal": sum(delta == 0 for delta in all_deltas),
            "current_deeper": sum(delta < 0 for delta in all_deltas),
            "pairwise_depth_deltas": all_deltas,
        },
        "cross_time_by_position": cross_time,
        "stable_regression_positions": stable_regression_positions,
        "stable_two_plus_regression_positions": stable_two_plus_positions,
    }


def focus_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    focus_ids = (
        "s21-control-237-32",
        "s21-attack-616-31",
        "s21-defense-143-7",
    )
    result: dict[str, Any] = {}
    for position_id in focus_ids:
        position_rows = [row for row in rows if row["position_id"] == position_id]
        if not position_rows:
            continue
        result[position_id] = {
            str(time_ms): {
                profile: {
                    "bestmove": next(
                        row["bestmove"]
                        for row in position_rows
                        if int(row["time_limit_ms"]) == time_ms
                        and row["engine_profile"] == profile
                    ),
                    "completed_depth": next(
                        row["completed_depth"]
                        for row in position_rows
                        if int(row["time_limit_ms"]) == time_ms
                        and row["engine_profile"] == profile
                    ),
                }
                for profile in PROFILES
            }
            for time_ms in EXPECTED_TIME_MS
        }
    return result


def evaluate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(rows)
    reasons: list[str] = []
    median_depth_ok = {
        str(time_ms): summary["by_time_ms"][str(time_ms)]["profiles"][PROFILE_EVAL_ONLY][
            "median_completed_depth"
        ]
        >= summary["by_time_ms"][str(time_ms)]["profiles"][PROFILE_CURRENT][
            "median_completed_depth"
        ]
        for time_ms in EXPECTED_TIME_MS
    }
    if not all(median_depth_ok.values()):
        reasons.append("EvalOnly median depth is below Current in at least one time tier")
    all_pairs = summary["all_pairs"]
    pair_balance_ok = all_pairs["candidate_deeper"] >= all_pairs["current_deeper"]
    if not pair_balance_ok:
        reasons.append("EvalOnly deeper pairs are fewer than Current deeper pairs")
    stable_regression_ok = not summary["stable_regression_positions"]
    if not stable_regression_ok:
        reasons.append("a position is shallower for EvalOnly at all three time tiers")
    stable_two_plus_ok = not summary["stable_two_plus_regression_positions"]
    if not stable_two_plus_ok:
        reasons.append("a position has a stable two-or-more-depth regression")

    focus = focus_summary(rows)
    control_safe = True
    control_violations = []
    control_focus = focus.get("s21-control-237-32", {})
    for time_ms, values in control_focus.items():
        current_move = values[PROFILE_CURRENT]["bestmove"]
        eval_only_move = values[PROFILE_EVAL_ONLY]["bestmove"]
        if current_move == "g8f8" and eval_only_move == "g8h8":
            control_safe = False
            control_violations.append(
                {"time_ms": int(time_ms), "current": current_move, "eval_only": eval_only_move}
            )
    if not control_safe:
        reasons.append("EvalOnly reproduces control-237's harmful g8h8 while Current chooses g8f8")

    violations = counter_violations(rows)
    counters_zero = not violations
    if not counters_zero:
        reasons.append("feature-matrix counters are non-zero")

    quality_focus: dict[str, Any] = {}
    attack_focus = focus.get("s21-attack-616-31")
    if attack_focus is not None:
        quality_focus["attack_616_avoids_f5g4_at_3s_and_10s"] = all(
            attack_focus[str(time_ms)][PROFILE_EVAL_ONLY]["bestmove"] != "f5g4"
            for time_ms in (3000, 10000)
        )
    defense_focus = focus.get("s21-defense-143-7")
    if defense_focus is not None:
        quality_focus["defense_143_keeps_a7a6_at_3s_and_10s"] = all(
            defense_focus[str(time_ms)][PROFILE_EVAL_ONLY]["bestmove"] == "a7a6"
            for time_ms in (3000, 10000)
        )

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "criteria": {
            "median_depth_not_lower_by_time": median_depth_ok,
            "candidate_deeper_at_least_current_deeper": pair_balance_ok,
            "no_stable_one_plus_depth_regression": stable_regression_ok,
            "no_stable_two_plus_depth_regression": stable_two_plus_ok,
            "control_237_safe": control_safe,
            "feature_counters_all_zero": counters_zero,
        },
        "control_237_violations": control_violations,
        "counter_violations": violations,
        "quality_focus": quality_focus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("tests/data/s2.1-practical-positions.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2.1d/eval-only-fixed-time.json")
    )
    parser.add_argument(
        "--time-ms", type=int, nargs="+", default=list(EXPECTED_TIME_MS)
    )
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
        print("S2.1d time set is fixed at 1000 3000 10000 ms", file=sys.stderr)
        return 2

    try:
        positions = load_manifest(manifest)
        require(len(positions) == EXPECTED_POSITION_COUNT, "S2.1d manifest must contain 9 positions")
        identity = probe_uci(engine, PROFILE_CURRENT)
        rows: list[dict[str, Any]] = []
        for time_index, time_ms in enumerate(EXPECTED_TIME_MS):
            for position_index, position in enumerate(positions):
                order = rotated_profile_order(time_index, position_index)
                for profile in order:
                    print(
                        f"s2.1d time_ms={time_ms} position={position['id']} "
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
        artifact = {
            "schema_version": 1,
            "status": "GATE_RESULT",
            "analysis": "S2.1d king-danger evaluation fixed-time gate",
            "git_sha": git_sha(repo),
            "manifest": {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "position_count": len(positions),
                "groups": {group: 3 for group in EXPECTED_GROUPS},
            },
            "engine": {
                "path": str(engine),
                "sha256": sha256_file(engine),
                "uci_identity": identity,
                "profiles": list(PROFILES),
                "bench_only_candidate": PROFILE_EVAL_ONLY,
            },
            "limits_ms": list(EXPECTED_TIME_MS),
            "tt": {"mode": "cold", "hash_mb": 16, "threads": 1},
            "rotation": {
                "enabled": True,
                "rule": "reverse the two-profile order when (time_index + position_index) is odd",
                "base_order": list(PROFILES),
            },
            "rows": rows,
            "summary": summarize(rows),
            "focus": focus_summary(rows),
            "decision": evaluate_gate(rows),
            "interpretation": [
                "This is a fixed-time depth/quality gate, not an Elo/SPRT result.",
                "Current remains unchanged; EvalOnly remains bench-only until independently accepted.",
                "The teacher move fields are reused diagnostics from the saved S2.1 positions; no Stockfish search was run here.",
                "No forcing extensions, quiet checks, or other search features are part of this gate.",
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"decision={artifact['decision']['status']}")
        print(f"wrote {output}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s2.1d_error {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
