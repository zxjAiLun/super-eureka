#!/usr/bin/env python3
"""Run the S2.1b three-variant cost-attribution gate.

This is a short, bench-only diagnostic.  It keeps the approved ``Current``
and the full S2.1 candidate as references, then compares two isolated
variants on the same nine saved positions at one fixed time limit.  It does
not run UCI games, Stockfish, Elo, or any promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from run_s21_practical_gate import (
    EXPECTED_GROUPS,
    PROFILE_CANDIDATE as FULL_PROFILE,
    PROFILE_CURRENT,
    load_manifest,
    probe_uci,
    run_search,
    sha256_file,
    git_sha,
)


PROFILE_NO_QCHECKS = "current-threat-aware-no-qchecks"
PROFILE_EVAL_ORDER = "current-threat-aware-eval-order"
ATTRIBUTION_PROFILES = (
    PROFILE_CURRENT,
    FULL_PROFILE,
    PROFILE_NO_QCHECKS,
    PROFILE_EVAL_ORDER,
)
EXPECTED_POSITION_COUNT = 9
EXPECTED_TIME_MS = 3000
ATTRIBUTION_COUNTERS = (
    "check_extensions",
    "single_evasion_extensions",
    "qsearch_check_moves",
    "threat_ordered_moves",
    "root_reorders",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_int_field(raw_result: str, name: str) -> int:
    for token in raw_result.split():
        key, separator, value = token.partition("=")
        if key == name and separator:
            return int(value)
    raise ValueError(f"bench result missing {name}: {raw_result}")


def add_attribution_fields(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    raw = str(row["raw_result"])
    updated["qsearch_nodes"] = parse_int_field(raw, "qsearch_nodes")
    updated["total_nodes"] = parse_int_field(raw, "total_nodes")
    updated["completed_iterations"] = parse_int_field(raw, "completed_iterations")
    updated["component_counters"] = {
        name: int(row["counters"].get(name, parse_int_field(raw, name)))
        for name in ATTRIBUTION_COUNTERS
    }
    return updated


def validate_variants(rows: list[dict[str, Any]]) -> None:
    require(len(rows) == EXPECTED_POSITION_COUNT * len(ATTRIBUTION_PROFILES),
            "S2.1b must contain exactly 36 rows")
    identities = {
        (str(row["position_id"]), str(row["engine_profile"])) for row in rows
    }
    require(len(identities) == len(rows), "duplicate S2.1b position/profile row")
    position_ids = {str(row["position_id"]) for row in rows}
    require(
        len(position_ids) == EXPECTED_POSITION_COUNT,
        "S2.1b must contain exactly 9 underlying positions",
    )
    for position_id in position_ids:
        require(
            sum(row["position_id"] == position_id for row in rows)
            == len(ATTRIBUTION_PROFILES),
            f"position {position_id} must have one row for every S2.1b profile",
        )
    require(
        {row["engine_profile"] for row in rows} == set(ATTRIBUTION_PROFILES),
        "S2.1b profile set is incomplete or contains an unexpected profile",
    )
    require(
        {int(row["time_limit_ms"]) for row in rows} == {EXPECTED_TIME_MS},
        "S2.1b must use exactly one 3-second time limit",
    )


def _median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validate_variants(rows)
    profiles: dict[str, Any] = {}
    for profile in ATTRIBUTION_PROFILES:
        selected = [row for row in rows if row["engine_profile"] == profile]
        profiles[profile] = {
            "positions": len(selected),
            "median_completed_depth": _median([int(row["completed_depth"]) for row in selected]),
            "median_nodes": _median([int(row["nodes"]) for row in selected]),
            "median_qsearch_nodes": _median([int(row["qsearch_nodes"]) for row in selected]),
            "median_elapsed_ms": _median([int(row["elapsed_ms"]) for row in selected]),
            "median_nps": _median([int(row["nps"]) for row in selected]),
            "sum_counters": {
                name: sum(int(row["component_counters"][name]) for row in selected)
                for name in ATTRIBUTION_COUNTERS
            },
            "bestmoves": {row["position_id"]: row["bestmove"] for row in selected},
            "scores": {row["position_id"]: row["score"] for row in selected},
        }

    paired: dict[str, Any] = {}
    for position_id in sorted({row["position_id"] for row in rows}):
        position_rows = {row["engine_profile"]: row for row in rows if row["position_id"] == position_id}
        paired[position_id] = {
            profile: {
                "bestmove": position_rows[profile]["bestmove"],
                "score": position_rows[profile]["score"],
                "completed_depth": position_rows[profile]["completed_depth"],
                "nodes": position_rows[profile]["nodes"],
                "qsearch_nodes": position_rows[profile]["qsearch_nodes"],
                "counters": position_rows[profile]["component_counters"],
            }
            for profile in ATTRIBUTION_PROFILES
        }

    control_id = "s21-control-237-32"
    focus = {
        "control_regression_position": (
            {
                "position_id": control_id,
                "bestmoves": {
                    profile: paired[control_id][profile]["bestmove"]
                    for profile in ATTRIBUTION_PROFILES
                },
            }
            if control_id in paired
            else None
        ),
        "king_danger": {
            position_id: {
                profile: paired[position_id][profile]["bestmove"]
                for profile in ATTRIBUTION_PROFILES
            }
            for position_id in sorted(
                position_id
                for position_id in paired
                if position_id.startswith("s21-attack-")
            )
        },
        "defensive_resource": {
            position_id: {
                profile: paired[position_id][profile]["bestmove"]
                for profile in ATTRIBUTION_PROFILES
            }
            for position_id in sorted(
                position_id
                for position_id in paired
                if position_id.startswith("s21-defense-")
            )
        },
    }
    return {"profiles": profiles, "positions": paired, "focus": focus}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("tests/data/s2.1-practical-positions.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2.1b/cost-attribution-3s.json")
    )
    parser.add_argument("--time-ms", type=int, default=EXPECTED_TIME_MS)
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
    if args.time_ms != EXPECTED_TIME_MS:
        print("S2.1b attribution is fixed at --time-ms 3000", file=sys.stderr)
        return 2

    try:
        positions = load_manifest(manifest)
        require(len(positions) == EXPECTED_POSITION_COUNT, "S2.1b manifest must contain 9 positions")
        identity = {
            PROFILE_CURRENT: probe_uci(engine, PROFILE_CURRENT),
            FULL_PROFILE: probe_uci(engine, FULL_PROFILE),
        }
        rows: list[dict[str, Any]] = []
        for position in positions:
            for profile in ATTRIBUTION_PROFILES:
                print(
                    f"s2.1b position={position['id']} group={position['group']} profile={profile}",
                    flush=True,
                )
                result = run_search(engine, profile, position, args.time_ms, repo)
                require(
                    result["engine_profile"] == profile,
                    f"bench profile mismatch: expected {profile}, got {result['engine_profile']}",
                )
                rows.append(add_attribution_fields(result))
        validate_variants(rows)
        artifact = {
            "schema_version": 1,
            "status": "DIAGNOSTIC_ONLY_NO_DECISION",
            "analysis": "S2.1b threat-aware cost attribution",
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
                "interface": "bench profile; attribution variants are not UCI startup profiles",
            },
            "time_limit_ms": args.time_ms,
            "tt": {"mode": "cold", "hash_mb": 16, "threads": 1},
            "profiles": {
                PROFILE_CURRENT: "approved Current reference",
                FULL_PROFILE: "full S2.1 reference",
                PROFILE_NO_QCHECKS: "Full S2.1 without quiet-check qsearch",
                PROFILE_EVAL_ORDER: "threat-aware evaluation/order/root reorder only",
            },
            "rows": rows,
            "summary": summarize(rows),
            "interpretation": [
                "This is a 3-second cost-attribution diagnostic, not an Elo/SPRT result.",
                "Current remains unchanged and all threat-aware variants remain bench-only.",
                "No parameter retuning or additional pruning is part of this run.",
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s2.1b_error {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
