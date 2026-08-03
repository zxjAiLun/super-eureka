#!/usr/bin/env python3
"""Run the S2.1 practical-position diagnostic gate.

This is a fixed-position, fixed-time diagnostic, not an Elo runner.  Each
search is a fresh ``bench profile --mode cold`` process, so the TT starts empty
for every engine/profile/time/position combination.  The Rust bench output is
used because it exposes the candidate counters in addition to the normal
score, depth, nodes and PV fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROFILE_CURRENT = "current"
PROFILE_CANDIDATE = "current-threat-aware"
EXPECTED_GROUPS = ("king-danger", "defensive-resource", "control")
COUNTERS = (
    "check_extensions",
    "single_evasion_extensions",
    "qsearch_check_moves",
)


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


def parse_key_values(line: str) -> dict[str, str]:
    tokens = shlex.split(line, posix=True)
    if not tokens or tokens[0] != "bench_result":
        raise ValueError(f"not a bench_result line: {line!r}")
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if not separator:
            continue
        fields[key] = value
    return fields


def parse_score(value: str) -> dict[str, Any]:
    kind, separator, raw = value.partition(":")
    if not separator:
        return {"kind": "unknown", "value": value}
    if kind == "none":
        return {"kind": "none", "value": None}
    try:
        numeric: int | str = int(raw)
    except ValueError:
        numeric = raw
    return {"kind": kind, "value": numeric}


def parse_info_depth_lines(stdout: str) -> list[dict[str, Any]]:
    iterations: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        tokens = line.split()
        if len(tokens) < 6 or tokens[:2] != ["info", "depth"]:
            continue
        try:
            depth = int(tokens[2])
        except ValueError:
            continue
        if tokens[3] != "score" or len(tokens) < 6:
            continue
        score = parse_score(f"{tokens[4]}:{tokens[5]}")
        fields: dict[str, Any] = {"depth": depth, "score": score}
        index = 6
        while index < len(tokens):
            token = tokens[index]
            if token == "pv":
                fields["pv"] = tokens[index + 1 :]
                break
            if index + 1 < len(tokens) and token in {"nodes", "time", "nps"}:
                try:
                    fields[f"{token}_ms" if token == "time" else token] = int(
                        tokens[index + 1]
                    )
                except ValueError:
                    fields[f"{token}_ms" if token == "time" else token] = tokens[
                        index + 1
                    ]
                index += 2
            else:
                index += 1
        iterations.append(fields)
    return iterations


def probe_uci(engine: Path, expected_profile: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(engine), "--profile", expected_profile],
        input="uci\nisready\nquit\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    lines = completed.stdout.splitlines()
    reported = next(
        (
            line.split("search profile ", 1)[1]
            for line in lines
            if line.startswith("info string search profile ")
        ),
        None,
    )
    identity: dict[str, Any] = {
        "argv": [str(engine), "--profile", expected_profile],
        "returncode": completed.returncode,
        "reported_profile": reported,
        "id_name": next(
            (line[8:] for line in lines if line.startswith("id name ")), None
        ),
        "id_author": next(
            (line[9:] for line in lines if line.startswith("id author ")), None
        ),
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"UCI probe failed for {engine}: {identity}")
    if reported != expected_profile:
        raise RuntimeError(
            f"UCI profile mismatch for {engine}: expected {expected_profile}, "
            f"reported {reported!r}"
        )
    if completed.stderr:
        raise RuntimeError(f"UCI probe wrote stderr for {engine}: {completed.stderr!r}")
    return identity


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) != 9:
        raise ValueError("S2.1 manifest must contain exactly 9 positions")
    ids = [item.get("id") for item in positions]
    if len(set(ids)) != len(ids) or any(not isinstance(item, str) for item in ids):
        raise ValueError("S2.1 position IDs must be unique strings")
    groups = {item.get("group") for item in positions}
    if groups != set(EXPECTED_GROUPS):
        raise ValueError(f"S2.1 groups must be exactly {EXPECTED_GROUPS}, got {groups}")
    for group in EXPECTED_GROUPS:
        count = sum(item.get("group") == group for item in positions)
        if count != 3:
            raise ValueError(f"S2.1 group {group} must contain 3 positions, got {count}")
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
        timeout=max(10, time_ms / 1000 + 20),
        check=False,
    )
    wall_ms = round((time.monotonic() - started) * 1000, 3)
    result_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("bench_result ")
    ]
    if completed.returncode != 0 or len(result_lines) != 1 or completed.stderr:
        raise RuntimeError(
            "S2.1 search failed: "
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
        *COUNTERS,
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise RuntimeError(f"bench result missing fields {missing}: {result_lines[0]}")

    numeric_fields = {
        key: int(fields[key])
        for key in ("completed_depth", "nodes", "elapsed_ms", "nps", *COUNTERS)
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
        "wall_time_ms": wall_ms,
        "score": parse_score(fields["score"]),
        "bestmove": fields["bestmove"],
        "source_played_move": position["played_move"],
        "source_teacher_move": position["teacher_move"],
        "source_teacher_agreement": fields["bestmove"] == position["teacher_move"],
        "completed_depth": numeric_fields["completed_depth"],
        "nodes": numeric_fields["nodes"],
        "elapsed_ms": numeric_fields["elapsed_ms"],
        "nps": numeric_fields["nps"],
        "pv": fields["pv"].split() if fields["pv"] else [],
        "score_by_depth": parse_info_depth_lines(completed.stdout),
        "counters": {key: numeric_fields[key] for key in COUNTERS},
        "raw_result": result_lines[0],
    }


def median(values: list[float | int]) -> float | int | None:
    return statistics.median(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for time_ms in sorted({row["time_limit_ms"] for row in rows}):
        time_rows = [row for row in rows if row["time_limit_ms"] == time_ms]
        for group in EXPECTED_GROUPS + ("all",):
            group_rows = (
                time_rows
                if group == "all"
                else [row for row in time_rows if row["group"] == group]
            )
            profiles: dict[str, Any] = {}
            for profile in (PROFILE_CURRENT, PROFILE_CANDIDATE):
                selected = [
                    row for row in group_rows if row["engine_profile"] == profile
                ]
                profiles[profile] = {
                    "positions": len(selected),
                    "median_completed_depth": median(
                        [row["completed_depth"] for row in selected]
                    ),
                    "median_elapsed_ms": median(
                        [row["elapsed_ms"] for row in selected]
                    ),
                    "median_nodes": median([row["nodes"] for row in selected]),
                    "source_teacher_agreement_count": sum(
                        row["source_teacher_agreement"] for row in selected
                    ),
                    "sum_counters": {
                        key: sum(row["counters"][key] for row in selected)
                        for key in COUNTERS
                    },
                }
            pairs = []
            for position_id in sorted({row["position_id"] for row in group_rows}):
                pair = {
                    row["engine_profile"]: row
                    for row in group_rows
                    if row["position_id"] == position_id
                }
                if PROFILE_CURRENT in pair and PROFILE_CANDIDATE in pair:
                    pairs.append(
                        pair[PROFILE_CANDIDATE]["completed_depth"]
                        - pair[PROFILE_CURRENT]["completed_depth"]
                    )
            summary.append(
                {
                    "time_limit_ms": time_ms,
                    "group": group,
                    "pair_count": len(pairs),
                    "candidate_depth_uplift_cases": sum(delta > 0 for delta in pairs),
                    "baseline_depth_uplift_cases": sum(delta < 0 for delta in pairs),
                    "equal_depth_cases": sum(delta == 0 for delta in pairs),
                    "pairwise_depth_deltas": pairs,
                    "profiles": profiles,
                }
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-current", type=Path, required=True)
    parser.add_argument("--engine-candidate", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/data/s2.1-practical-positions.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2.1/practical-gate.json")
    )
    parser.add_argument(
        "--time-ms", type=int, nargs="+", default=[1000, 3000, 10000]
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    engine_current = args.engine_current.resolve()
    engine_candidate = args.engine_candidate.resolve()
    if not engine_current.is_file() or not engine_candidate.is_file():
        raise SystemExit("both engine paths must point to files")
    if any(time_ms <= 0 for time_ms in args.time_ms):
        raise SystemExit("all --time-ms values must be positive")

    positions = load_manifest(args.manifest)
    manifest_sha = sha256_file(args.manifest.resolve())
    engines = {
        PROFILE_CURRENT: {
            "path": str(engine_current),
            "sha256": sha256_file(engine_current),
            "expected_profile": PROFILE_CURRENT,
        },
        PROFILE_CANDIDATE: {
            "path": str(engine_candidate),
            "sha256": sha256_file(engine_candidate),
            "expected_profile": PROFILE_CANDIDATE,
        },
    }
    for profile, metadata in engines.items():
        metadata["uci_probe"] = probe_uci(Path(metadata["path"]), profile)

    rows: list[dict[str, Any]] = []
    for time_index, time_ms in enumerate(args.time_ms):
        for position_index, position in enumerate(positions):
            profiles = [PROFILE_CURRENT, PROFILE_CANDIDATE]
            if (time_index + position_index) % 2:
                profiles.reverse()
            for profile in profiles:
                print(
                    f"s2.1 position={position['id']} group={position['group']} "
                    f"time_ms={time_ms} profile={profile}",
                    flush=True,
                )
                rows.append(
                    run_search(
                        Path(engines[profile]["path"]),
                        profile,
                        position,
                        time_ms,
                        repo,
                    )
                )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY_NO_DECISION",
        "git_sha": git_sha(repo),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_sha,
            "position_count": len(positions),
            "groups": {group: 3 for group in EXPECTED_GROUPS},
        },
        "limits_ms": args.time_ms,
        "tt": {"mode": "cold", "hash_mb": 16, "threads": 1},
        "engines": engines,
        "rows": rows,
        "summary": summarize(rows),
        "interpretation": [
            "This is a practical diagnostic, not an Elo/SPRT result.",
            "The three groups are analyst labels on saved D1.14 positions, not objective oracles.",
            "Current remains the production profile; current-threat-aware remains candidate-only.",
        ],
    }
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s2.1_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
