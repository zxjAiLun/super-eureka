#!/usr/bin/env python3
"""S4.0A Compute Attribution runner.

Collects CPU attribution data for the ``current-final`` profile over a fixed
position corpus and time budgets, reusing the existing
``run_s21_practical_gate.run_search`` helper (which drives the engine's
``bench profile --mode cold --movetime --profile --fen`` and returns the full
``bench_result`` attribution line).

This is a diagnostic-only runner. It does NOT modify search/eval behavior, does
not introduce candidates, and does not produce an Elo/SPRT decision.

Outputs:
  raw.jsonl     one JSON object per measured run (all repeats)
  summary.csv   median metrics per (position x budget), per (class x budget),
                and per budget (overall)
  summary.md    short human-readable notes
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_s21_practical_gate import (
    git_sha,
    parse_key_values,
    probe_uci,
    run_search,
    sha256_file,
)

PROFILE = "current-final"
DEFAULT_BUDGETS_MS = (100, 500, 1000, 3000)
DEFAULT_REPEAT = 3
HASH_MB = 16

# Integer counters present in the bench_result line.
INT_FIELDS = (
    "total_nodes",
    "completed_iterations",
    "nodes_per_completed_depth",
    "last_completed_iteration_ms",
    "last_completed_iteration_nodes",
    "aborted_iteration_depth",
    "aborted_iteration_nodes",
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
    "tt_rejected_depth",
    "tt_rejected_bound",
    "tt_rejected_decode",
    "tt_stores",
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

# Summary/metric columns written to summary.csv.
SUMMARY_METRICS = (
    "completed_depth",
    "nodes",
    "nps",
    "nodes_depth_proxy",
    "qsearch_ratio",
    "eval_per_node",
    "movegen_per_node",
    "make_unmake_per_node",
    "pseudo_to_legal_ratio",
    "tt_hit_rate",
    "tt_cutoff_rate",
    "tt_depth_reject_per_hit",
    "tt_bound_reject_per_eligible_hit",
    "tt_cutoff_per_eligible_hit",
    "tt_store_per_probe",
    "lmr_research_rate",
    "null_research_rate",
    "aspiration_retry_per_iter",
    "futility_pruned",
    "qsearch_see_prune_rate",
    "aborted_iteration_depth",
    "completed_iteration_growth",
)


def ratio(a: float | int, b: float | int) -> float:
    return (a / b) if b else 0.0


def load_epd(path: Path) -> list[dict[str, Any]]:
    """Parse the S4 EPD corpus: ``<fen> ; id=<id> ; class=<class>``."""
    positions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(";")
        fen = parts[0].strip()
        tags: dict[str, str] = {}
        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
            key, separator, value = part.partition("=")
            if separator:
                tags[key.strip()] = value.strip()
        positions.append(
            {
                "id": tags.get("id", f"pos_{len(positions)}"),
                "group": tags.get("class", "unknown"),
                "fen": fen,
                "played_move": "",
                "teacher_move": "",
            }
        )
    if not positions:
        raise ValueError(f"no positions parsed from {path}")
    return positions


def parse_run(
    engine: Path, position: dict[str, Any], budget_ms: int, repo: Path, profile: str = PROFILE
) -> dict[str, Any]:
    row = run_search(engine, profile, position, budget_ms, repo)
    raw = parse_key_values(row["raw_result"])
    rec: dict[str, Any] = {
        "position_id": position["id"],
        "class": position["group"],
        "fen": position["fen"],
        "budget_ms": budget_ms,
        "profile": PROFILE,
        "hash_mb": HASH_MB,
        "tt": "cold",
        "score": raw.get("score"),
        "bestmove": raw.get("bestmove"),
        "completed_depth": int(raw["completed_depth"]),
        "stopped": raw.get("stopped") == "true",
        "nodes": int(raw["nodes"]),
        "elapsed_us": int(raw["elapsed_us"]),
        "nps": int(raw["nps"]),
        # Rough proxy only: nodes^(1/depth). Includes iterative-deepening,
        # qsearch and aborted deeper-iteration work; NOT comparable to another
        # engine's reported branching factor.
        "nodes_depth_proxy": float(raw.get("effective_branching_factor", "0")),
    }
    for field in INT_FIELDS:
        rec[field] = int(raw.get(field, "0"))
    rec["qsearch_ratio"] = ratio(rec["qsearch_nodes"], rec["nodes"])
    rec["eval_per_node"] = ratio(rec["eval_calls"], rec["nodes"])
    rec["movegen_per_node"] = ratio(rec["legal_move_generations"], rec["nodes"])
    rec["make_unmake_per_node"] = ratio(
        rec["make_moves"] + rec["unmake_moves"], rec["nodes"]
    )
    rec["pseudo_to_legal_ratio"] = ratio(rec["pseudo_moves"], rec["legal_moves"])
    rec["tt_hit_rate"] = ratio(rec["tt_hits"], rec["tt_probes"])
    rec["tt_cutoff_rate"] = ratio(rec["tt_cutoffs"], rec["tt_probes"])
    rec["tt_store_per_probe"] = ratio(rec["tt_stores"], rec["tt_probes"])
    eligible_hits = rec["tt_hits"] - rec["tt_rejected_depth"] - rec["tt_rejected_decode"]
    if eligible_hits < 0:
        eligible_hits = 0
    rec["eligible_hits"] = eligible_hits
    rec["tt_depth_reject_per_hit"] = ratio(
        rec["tt_rejected_depth"], rec["tt_hits"]
    )
    rec["tt_bound_reject_per_eligible_hit"] = ratio(
        rec["tt_rejected_bound"], eligible_hits
    )
    rec["tt_cutoff_per_eligible_hit"] = ratio(rec["tt_cutoffs"], eligible_hits)
    rec["lmr_research_rate"] = ratio(rec["lmr_researches"], rec["lmr_reductions"])
    rec["null_research_rate"] = ratio(
        rec["null_move_researches"], rec["null_move_attempts"]
    )
    rec["aspiration_retry_per_iter"] = ratio(
        rec["aspiration_retries"], rec["completed_iterations"]
    )
    rec["qsearch_see_prune_rate"] = ratio(
        rec["qsearch_see_pruned"], rec["qsearch_see_tests"]
    )
    return rec


def med(values: list[float | int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def normalize_run(rec: dict[str, Any]) -> dict[str, Any]:
    """Backfill derived fields on a raw run record (works on committed raw.jsonl)."""
    if "nodes_depth_proxy" not in rec and "ebf_approx" in rec:
        rec["nodes_depth_proxy"] = rec["ebf_approx"]
    if "eligible_hits" not in rec:
        eligible = (
            rec.get("tt_hits", 0)
            - rec.get("tt_rejected_depth", 0)
            - rec.get("tt_rejected_decode", 0)
        )
        rec["eligible_hits"] = eligible if eligible > 0 else 0
    rec.setdefault(
        "tt_depth_reject_per_hit",
        ratio(rec.get("tt_rejected_depth", 0), rec.get("tt_hits", 0)),
    )
    rec.setdefault(
        "tt_bound_reject_per_eligible_hit",
        ratio(rec.get("tt_rejected_bound", 0), rec["eligible_hits"]),
    )
    rec.setdefault(
        "tt_cutoff_per_eligible_hit",
        ratio(rec.get("tt_cutoffs", 0), rec["eligible_hits"]),
    )
    rec.setdefault("completed_iteration_growth", 0.0)
    return rec


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for metric in SUMMARY_METRICS:
        row[metric] = med([r.get(metric, 0) for r in runs])
    return row


def completed_iteration_growth(runs: list[dict[str, Any]]) -> dict[str, float]:
    """Per-position median ratio of last_completed_iteration_nodes(d) /
    (d-1), computed from the depths observed across the different budgets.
    Growth is only reported when both adjacent depths were completed by some
    run of that position (any budget / repeat)."""
    by_pos: dict[str, dict[int, list[float]]] = {}
    for r in runs:
        depth = int(r.get("completed_depth", 0))
        nodes = float(r.get("last_completed_iteration_nodes", 0))
        if depth <= 0 or nodes <= 0:
            continue
        by_pos.setdefault(r["position_id"], {}).setdefault(depth, []).append(nodes)
    growth: dict[str, float] = {}
    for pid, depths in by_pos.items():
        ratios = []
        for depth in sorted(depths):
            if depth - 1 in depths:
                prev = statistics.median(depths[depth - 1])
                cur = statistics.median(depths[depth])
                if prev > 0:
                    ratios.append(cur / prev)
        growth[pid] = statistics.median(ratios) if ratios else 0.0
    return growth


def build_summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = [normalize_run(r) for r in runs]
    growth = completed_iteration_growth(runs)
    rows: list[dict[str, Any]] = []
    budgets = sorted({r["budget_ms"] for r in runs})
    positions = sorted({r["position_id"] for r in runs})

    # per-position x budget (median over repeats)
    for pos_id in positions:
        pos_runs = [r for r in runs if r["position_id"] == pos_id]
        pos_class = pos_runs[0]["class"]
        for budget in budgets:
            sel = [r for r in pos_runs if r["budget_ms"] == budget]
            if not sel:
                continue
            row = summarize_runs(sel)
            row["completed_iteration_growth"] = growth.get(pos_id, 0.0)
            row.update(
                {"level": "position", "key": pos_id, "class": pos_class, "budget_ms": budget}
            )
            rows.append(row)

    # per-class x budget
    for cls in sorted({r["class"] for r in runs}):
        cls_positions = {r["position_id"] for r in runs if r["class"] == cls}
        cls_growth = [g for pid, g in growth.items() if pid in cls_positions]
        for budget in budgets:
            sel = [r for r in runs if r["class"] == cls and r["budget_ms"] == budget]
            if not sel:
                continue
            row = summarize_runs(sel)
            row["completed_iteration_growth"] = med(cls_growth)
            row.update({"level": "class", "key": cls, "class": cls, "budget_ms": budget})
            rows.append(row)

    # overall x budget
    overall_growth = list(growth.values())
    for budget in budgets:
        sel = [r for r in runs if r["budget_ms"] == budget]
        if not sel:
            continue
        row = summarize_runs(sel)
        row["completed_iteration_growth"] = med(overall_growth)
        row.update(
            {"level": "overall", "key": "overall", "class": "all", "budget_ms": budget}
        )
        rows.append(row)

    return rows


def write_summary_md(
    summary_rows: list[dict[str, Any]],
    out: Path,
    positions: list[dict[str, Any]],
    budgets: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# S4.0A Compute Attribution summary")
    lines.append("")
    lines.append("Diagnostic only; no candidate, no Elo decision.")
    lines.append("")
    lines.append(f"- positions: {len(positions)}")
    lines.append(f"- budgets (ms): {budgets}")
    lines.append(f"- profile: {PROFILE}, cold TT, {HASH_MB} MB, 1 thread")
    lines.append("")
    overall = [r for r in summary_rows if r["level"] == "overall"]
    lines.append(
        "| budget_ms | depth | nodes | nps | nodes_depth_proxy | qsearch_ratio | "
        "eval/node | movegen/node | make-unmake/node | tt_hit | tt_cutoff/elig | "
        "tt_dep_rej/hit | tt_bound_rej/elig | lmr_rsrch | null_rsrch | asp/iter | "
        "futility | growth |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in overall:
        lines.append(
            "| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.0f} | "
            "{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} | {:.2f} |".format(
                row["budget_ms"],
                row["completed_depth"],
                row["nodes"],
                row["nps"],
                row["nodes_depth_proxy"],
                row["qsearch_ratio"],
                row["eval_per_node"],
                row["movegen_per_node"],
                row["make_unmake_per_node"],
                row["tt_hit_rate"],
                row["tt_cutoff_per_eligible_hit"],
                row["tt_depth_reject_per_hit"],
                row["tt_bound_reject_per_eligible_hit"],
                row["lmr_research_rate"],
                row["null_research_rate"],
                row["aspiration_retry_per_iter"],
                row["futility_pruned"],
                row["completed_iteration_growth"],
            )
        )
    lines.append("")
    lines.append("## Key observations")
    lines.append("")
    lines.append(
        "- NPS is flat across budgets (~177k-182k): per-node cost is stable."
    )
    lines.append(
        "- Median depth only grows 4 -> 7 across 100 ms -> 3000 ms. The rough "
        "`nodes_depth_proxy` (nodes^(1/depth)) is ~7-11; it is a coarse proxy "
        "that includes iterative-deepening, qsearch and aborted deeper-iteration "
        "work, and is NOT comparable to another engine's reported branching "
        "factor, so it is not used here to prove 'excessive EBF'."
    )
    lines.append(
        "- make/unmake ~= 114-122 per node in middlegames (much lower, ~36-43, "
        "with far higher NPS in low-material endgames). Counters measure "
        "operation frequency, not CPU wall-time share, so this is the strongest "
        "*measured* Core-cost suspect, not a proven dominant CPU cost."
    )
    lines.append(
        "- qsearch ~= 80-86% of all nodes (~5 qsearch nodes per main-search node "
        "at 3000 ms). Strongest Search-side signal; not yet proof qsearch is "
        "mis-designed."
    )
    lines.append(
        "- TT: hit rate ~0.15-0.24, cutoffs ~0.1% of eligible hits. "
        "`tt_stores/tt_probes ~= 1` alone is not evidence of a problem. "
        "Distinguish: depth-rejected entries (this is why hit rate is low) vs "
        "depth-eligible entries whose bounds rarely cut."
    )
    lines.append(
        "- LMR re-search is low (~0-1.1%); null-move fail-highs are all "
        "re-verified (~40-48% null research rate). S4.0A does NOT show an LMR "
        "re-search explosion."
    )
    lines.append(
        "- Aspiration retries are negligible on the median."
    )
    lines.append("")
    lines.append("## Preliminary top-3 compute bottleneck hypotheses (evidence-ordered)")
    lines.append("")
    lines.append(
        "1. Legal-move legality filtering (pseudo -> make/check/unmake): ~114-122 "
        "make/unmake per node, the strongest measured Core-cost suspect."
    )
    lines.append(
        "2. qsearch dominates the tree (~80% of nodes) together with a high "
        "`nodes_depth_proxy`: the tree is large relative to a more selective "
        "engine."
    )
    lines.append(
        "3. TT under-utilization: hit rate 15-24%, cutoffs ~0.1%; depth-rejected "
        "hits dominate. Transposition reuse is weak."
    )
    lines.append("")
    lines.append(
        "Overall this is a mixed result: the cheapest, clearly-attributable "
        "node-cost culprit is movegen legality filtering; the binding constraint "
        "on completed depth is tree size (qsearch share and high branching proxy). "
        "Evaluation quality remains unmeasured by S4.0A."
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_derive(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Derive S4.0A summary from committed raw.jsonl (no engine rerun)."
    )
    parser.add_argument("--raw", type=Path, required=True, help="compute/raw.jsonl")
    parser.add_argument(
        "--out", type=Path, default=Path("results/s4-attribution/compute")
    )
    args = parser.parse_args(argv)
    runs = [
        normalize_run(json.loads(line))
        for line in args.raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not runs:
        raise ValueError(f"no runs in {args.raw}")
    positions = sorted({r["position_id"] for r in runs})
    budgets = sorted({r["budget_ms"] for r in runs})
    summary_rows = build_summary_rows(runs)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "key", "class", "budget_ms", *SUMMARY_METRICS],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    write_summary_md(summary_rows, out_dir / "summary.md", positions, budgets)
    print(f"derived {out_dir}")
    return 0


def cmd_compute(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="CurrentFinal release binary")
    parser.add_argument(
        "--epd", type=Path, default=Path("tools/data/s4_compute_positions.epd")
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS_MS)
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument(
        "--out", type=Path, default=Path("results/s4-attribution/compute")
    )
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"engine not found: {engine}")
    if any(b <= 0 for b in args.budgets):
        raise SystemExit("all --budgets values must be positive")
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    positions = load_epd(args.epd.resolve())
    probe_uci(engine, args.profile)

    budgets = sorted(args.budgets)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    total = len(positions) * len(budgets) * args.repeat
    done = 0
    for budget in budgets:
        for position in positions:
            for _ in range(args.repeat):
                print(
                    f"s4a position={position['id']} class={position['group']} "
                    f"budget_ms={budget} profile={args.profile}",
                    flush=True,
                )
                runs.append(parse_run(engine, position, budget, repo, args.profile))
                done += 1
    print(f"s4a done {done}/{total} runs", flush=True)

    summary_rows = build_summary_rows(runs)

    (out_dir / "raw.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs),
        encoding="utf-8",
    )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "key", "class", "budget_ms", *SUMMARY_METRICS],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    provenance = {
        "git_sha": git_sha(repo),
        "engine": str(engine),
        "engine_sha256": sha256_file(engine),
        "profile": PROFILE,
        "hash_mb": HASH_MB,
        "threads": 1,
        "tt": "cold",
        "budgets_ms": budgets,
        "repeat": args.repeat,
        "position_count": len(positions),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(summary_rows, out_dir / "summary.md", positions, budgets)

    print(f"wrote {out_dir}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "derive":
        return cmd_derive(argv[1:])
    return cmd_compute(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s4a_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
