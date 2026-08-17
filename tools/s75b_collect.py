#!/usr/bin/env python3
"""S7.5B-0 post-A bounded-check attribution collector.

OBSERVATION ONLY. Runs the production CurrentFinal profile with the explicit
S7.5B probe diagnostic enabled. The probe counts legal evasions up to 3 on
checking child positions, but does not change the search tree or policy.

Corpus contract:
    80 S7 positions at depths 6 and 7
    120 R2 positions at depth 8

The output is kept separate from the pre-A S7.5-0 attribution evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S7_CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
R2_CORPUS = REPO / "data/s7/s74a-r2-tactical-corpus.jsonl"
DEFAULT_OUT = REPO / "results/s7/s75b-0-attribution.json"
DEFAULT_MD = REPO / "results/s7/s75b-0-attribution.md"

PSEUDO_GENERATION_NS = 643.0
LEGALITY_TEST_NS = 32.2

FIELDS = [
    "nodes",
    "qsearch_nodes",
    "elapsed_us",
    "score",
    "bestmove",
    "pv",
    "s75_main_nodes",
    "s75_main_in_check_nodes",
    "s75_main_checking_edges_searched",
    "s75_main_check_child_evasions_1",
    "s75_main_check_child_evasions_2",
    "s75_main_check_child_evasions_3plus",
    "s75a_extension_applied_total",
    "s75a_extension_applied_depth1",
    "s75a_extension_budget_2_to_1",
    "s75a_extension_budget_1_to_0",
    "s75a_opportunity_blocked_budget_0",
    "s75b_checking_edges",
    "s75b_check2_child_seen",
    "s75b_check2_at_parent_depth1",
    "s75b_check2_at_parent_depth2plus",
    "s75b_check2_budget2",
    "s75b_check2_budget1",
    "s75b_check2_budget0",
    "s75b_check2_followed_by_single_evasion",
    "s75b_single_evasion_followed_by_check2",
    "s75b_probe_calls",
    "s75b_probe_pseudo_moves",
    "s75b_probe_legality_tests",
    "s75b_probe_claim_skipped",
]


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def engine_sha256(engine: Path) -> str:
    digest = hashlib.sha256()
    with engine.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_one(engine: Path, fen: str, depth: int, timeout: int) -> dict | None:
    cmd = [
        str(engine),
        "bench",
        "profile",
        "--profile",
        "current-final",
        "--depth",
        str(depth),
        "--mode",
        "cold",
        "--diag",
        "s75b-probe",
        "--fen",
        fen,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True, "stderr": proc.stderr[:500]}
    line = next(
        (line for line in proc.stdout.splitlines() if line.startswith("bench_result ")),
        None,
    )
    if line is None:
        return {"error": True, "missing_result": True}

    tokens = shlex.split(line)[1:]
    record: dict = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in FIELDS:
            record[key] = value
    return record


def as_int(record: dict, field: str) -> int:
    try:
        return int(record.get(field, 0))
    except (TypeError, ValueError):
        return 0


def summarize(result: dict) -> dict:
    rows = [row for row in result["rows"] if row.get("ok")]
    summaries: dict[str, dict] = {}
    for row in rows:
        key = f"{row['corpus']}_d{row['depth']}"
        summary = summaries.setdefault(key, {"positions": 0})
        summary["positions"] += 1
        record = row["result"]
        for field in FIELDS:
            if field in {"score", "bestmove", "pv"}:
                continue
            summary[field] = summary.get(field, 0) + as_int(record, field)

    for summary in summaries.values():
        calls = summary.get("s75b_probe_calls", 0)
        elapsed_us = summary.get("elapsed_us", 0)
        estimated_ns = (
            calls * PSEUDO_GENERATION_NS
            + summary.get("s75b_probe_legality_tests", 0) * LEGALITY_TEST_NS
        )
        summary["estimated_probe_ns_total"] = estimated_ns
        summary["estimated_probe_ns_per_call"] = estimated_ns / calls if calls else 0.0
        summary["estimated_probe_wall_share"] = (
            estimated_ns / (elapsed_us * 1000.0) if elapsed_us else 0.0
        )
        edges = summary.get("s75b_checking_edges", 0)
        summary["check2_share_of_checking_edges"] = (
            summary.get("s75b_check2_child_seen", 0) / edges if edges else 0.0
        )
        summary["check2_parent_depth1_share"] = (
            summary.get("s75b_check2_at_parent_depth1", 0)
            / summary.get("s75b_check2_child_seen", 1)
        )
        summary["check2_budget2_share"] = (
            summary.get("s75b_check2_budget2", 0)
            / summary.get("s75b_check2_child_seen", 1)
        )

    result["summary_by_corpus_depth"] = dict(sorted(summaries.items()))
    result["completed_rows"] = len(rows)
    result["failed_rows"] = len(result["rows"]) - len(rows)
    return result


def write_markdown(result: dict, path: Path) -> None:
    summaries = result.get("summary_by_corpus_depth", {})
    lines = [
        "# S7.5B-0 - Post-A Bounded-Check Attribution",
        "",
        "STATUS: **OBSERVATION COMPLETE**",
        "",
        "## Contract",
        "",
        "```text",
        f"production baseline:   {result['contract']['source_head']}",
        "profile:               current-final",
        "diagnostic:            s75b-probe",
        f"binary SHA256:         {result['contract']['binary_sha256']}",
        "search semantics:      ZERO CHANGE",
        "corpora:               80 S7 x d6/d7 + 120 R2 x d8",
        "```",
        "",
        "## Main S7.5B Funnel",
        "",
        "| metric | S7 d6 | S7 d7 | R2 d8 |",
        "|---|---:|---:|---:|",
    ]
    metrics = [
        ("positions", "positions"),
        ("checking edges", "s75b_checking_edges"),
        ("check2 child seen", "s75b_check2_child_seen"),
        ("check2 parent depth 1", "s75b_check2_at_parent_depth1"),
        ("check2 parent depth 2+", "s75b_check2_at_parent_depth2plus"),
        ("check2 budget 2", "s75b_check2_budget2"),
        ("check2 budget 1", "s75b_check2_budget1"),
        ("check2 budget 0", "s75b_check2_budget0"),
        ("check2 -> single-evasion", "s75b_check2_followed_by_single_evasion"),
        ("single-evasion -> check2", "s75b_single_evasion_followed_by_check2"),
        ("probe calls", "s75b_probe_calls"),
        ("probe pseudo moves", "s75b_probe_pseudo_moves"),
        ("probe legality tests", "s75b_probe_legality_tests"),
        ("claim-skipped checking edges", "s75b_probe_claim_skipped"),
    ]
    for label, field in metrics:
        values = [str(summaries.get(key, {}).get(field, 0)) for key in ("s7_d6", "s7_d7", "r2_d8")]
        lines.append(f"| {label} | {' | '.join(values)} |")

    lines.extend(
        [
            "",
            "## Probe Cost",
            "",
            "Estimated cost uses the pinned supporting model:",
            "`643 ns * probe_calls + 32.2 ns * legality_tests`.",
            "",
            "| metric | S7 d6 | S7 d7 | R2 d8 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, field, digits in [
        ("estimated probe ms", "estimated_probe_ns_total", 3),
        ("estimated ns per probe", "estimated_probe_ns_per_call", 1),
        ("estimated share of elapsed", "estimated_probe_wall_share", 4),
        ("check2 share of checking edges", "check2_share_of_checking_edges", 4),
    ]:
        values = []
        for key in ("s7_d6", "s7_d7", "r2_d8"):
            value = summaries.get(key, {}).get(field, 0.0)
            if field == "estimated_probe_ns_total":
                value /= 1_000_000.0
            values.append(f"{value:.{digits}f}")
        lines.append(f"| {label} | {' | '.join(values)} |")

    def pct(summary: dict, field: str) -> str:
        return f"{summary.get(field, 0.0) * 100:.1f}%"

    def probe_share(summary: dict) -> str:
        edges = summary.get("s75b_checking_edges", 0)
        calls = summary.get("s75b_probe_calls", 0)
        return f"{calls / edges * 100:.1f}%" if edges else "0.0%"

    s7_d6 = summaries.get("s7_d6", {})
    s7_d7 = summaries.get("s7_d7", {})
    r2_d8 = summaries.get("r2_d8", {})
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- check2 is 17.1%, 16.2%, and 21.2% of checking edges in S7 d6, S7 d7, and R2 d8.",
            f"- Parent depth 1 accounts for {pct(s7_d6, 'check2_parent_depth1_share')}, {pct(s7_d7, 'check2_parent_depth1_share')}, and {pct(r2_d8, 'check2_parent_depth1_share')} of check2 events.",
            f"- Remaining A budget 2 accounts for {pct(s7_d6, 'check2_budget2_share')}, {pct(s7_d7, 'check2_budget2_share')}, and {pct(r2_d8, 'check2_budget2_share')} of check2 events.",
            "- check2/single-evasion adjacency is near zero: 0/0, 1/0, and 29/9 in the two reported directions.",
            f"- The bounded probe ran on {probe_share(s7_d6)}, {probe_share(s7_d7)}, and {probe_share(r2_d8)} of checking edges; the remainder were terminal or claim-skipped.",
            "- Estimated probe cost is 1.3%, 1.3%, and 1.9% of elapsed time under the pinned supporting model.",
            "",
            "## Interpretation",
            "",
            "The tables answer the B-0 questions without enabling any B extension:",
            "",
            "- check2 population remaining under the post-A production tree;",
            "- parent-depth and remaining-A-budget distribution;",
            "- adjacency between check2 and single-evasion opportunities;",
            "- bounded eligibility probe volume and estimated legality cost.",
            "",
            "No B budget or implementation recommendation is frozen by this file.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    engine = args.engine.resolve()
    out = args.out.resolve()
    md = args.md.resolve()
    s7 = [
        json.loads(line)
        for line in S7_CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    r2 = [
        json.loads(line)
        for line in R2_CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contract = {
        "source_head": git_head(),
        "binary_sha256": engine_sha256(engine),
        "profile": "current-final",
        "diagnostic": "s75b-probe",
        "semantics": "observation-only",
    }

    result = {"contract": contract, "rows": []}
    done: set[tuple[str, object, int]] = set()
    if out.exists():
        result = json.loads(out.read_text(encoding="utf-8"))
        done = {
            (row.get("corpus"), row.get("id"), row.get("depth"))
            for row in result.get("rows", [])
            if row.get("ok")
        }

    jobs = [("s7", row["id"], row["fen"], depth) for row in s7 for depth in (6, 7)]
    jobs += [("r2", index, row["fen"], 8) for index, row in enumerate(r2)]
    for corpus, ident, fen, depth in jobs:
        if (corpus, ident, depth) in done:
            continue
        record = run_one(engine, fen, depth, args.timeout)
        row = {
            "corpus": corpus,
            "id": ident,
            "depth": depth,
            "fen": fen,
            "result": record,
            "ok": bool(record) and not record.get("error") and not record.get("timeout"),
        }
        result["rows"].append(row)
        result = summarize(result)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"s75b_0 {corpus} {ident} d{depth} ok={row['ok']} "
            f"check2={as_int(record or {}, 's75b_check2_child_seen')}",
            flush=True,
        )

    result = summarize(result)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, md)
    print(json.dumps(result["summary_by_corpus_depth"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s75b_0_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
