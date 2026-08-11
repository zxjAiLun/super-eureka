#!/usr/bin/env python3
"""S5.0A: Duplicate child full-legal generation attribution.

Bench-only. Measures how many full-legal generations are WASTED:

  - probe_child_generations: every child edge generates a full legal list
    only for terminal/claim detection, discarded on Continue;
  - negamax_body_generations: the Continue child's entered body regenerates
    the SAME position (the duplicate);
  - qsearch_edge_probe_generations: qsearch edges generate a full legal list
    that the qsearch body replaces with tactical/evasion lists.

Writes results/s4-attribution/core/s50a_report.md + s50a_data.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from s44a_attribution import run_position as run_position_s44a

PROFILE = "current-final"
DEPTH = 6

S5_FIELDS = (
    "probe_child_generations", "main_edge_probe_generations",
    "qsearch_edge_probe_generations", "root_edge_probe_generations",
    "negamax_body_generations", "root_generations",
    "final_evasion_generations",
)


def run_position(engine: Path, pos: dict, depth: int) -> dict | None:
    return run_position_s44a(engine, pos, depth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--epd", type=Path, default=Path("tools/data/s4_compute_positions.epd"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/s4-attribution/core"))
    args = parser.parse_args(sys.argv[1:])
    engine = args.engine.resolve()

    from s4_attribution import load_epd
    positions = load_epd(args.epd.resolve())
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for pos in positions:
        for rep in range(args.repeat):
            print(f"s50a rep={rep + 1} position={pos['id']}", flush=True)
            runs.append(run_position(engine, pos, DEPTH))

    def agg(sel: list[dict]) -> dict:
        base = {
            "nodes": sum(r["nodes"] for r in sel),
            "movegen_legal_calls": sum(r["movegen_legal_calls"] for r in sel),
            "movegen_legal_ns": sum(r["movegen_legal_ns"] for r in sel),
            "movegen_legal_samples": sum(r["movegen_legal_samples"] for r in sel),
        }
        for f in S5_FIELDS:
            base[f] = sum(r[f] for r in sel)
        base["duplicates"] = base["negamax_body_generations"]
        base["discarded_qsearch_probes"] = base["qsearch_edge_probe_generations"]
        base["wasted_calls"] = base["duplicates"] + base["discarded_qsearch_probes"]
        base["dup_share_of_calls"] = base["duplicates"] / base["movegen_legal_calls"]
        base["waste_share_of_calls"] = base["wasted_calls"] / base["movegen_legal_calls"]
        # Wall share: same per-call cost assumption for the full-legal bucket.
        base["movegen_legal_wall_ns"] = base["movegen_legal_ns"] * (
            base["movegen_legal_calls"] / base["movegen_legal_samples"]
        )
        base["waste_wall_ns"] = base["movegen_legal_wall_ns"] * base["waste_share_of_calls"]
        base["dup_wall_ns"] = base["movegen_legal_wall_ns"] * base["dup_share_of_calls"]
        base["total_elapsed_us"] = sum(r["elapsed_us"] for r in sel)
        return base

    n = len(positions)
    reps = [agg(runs[k::3]) for k in range(args.repeat)]
    all_agg = agg(runs)
    data = {
        "contract": {
            "profile": PROFILE, "depth": DEPTH, "hash_mb": 16, "threads": 1,
            "tt": "cold", "sampler": "sparse 256/1", "repeat": args.repeat,
            "positions": len(positions),
        },
        "aggregate": all_agg,
        "per_rep": reps,
    }
    (out_dir / "s50a_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(data, out_dir / "s50a_report.md")
    print("wrote", out_dir)
    return 0


def write_report(data: dict, out: Path) -> None:
    a = data["aggregate"]
    L: list[str] = []
    L.append("# S5.0A — Duplicate Child Full-Legal Generation Attribution")
    L.append("")
    L.append("ATTRIBUTION ONLY — no optimization candidate.")
    L.append("")
    L.append("## Contract")
    L.append("")
    L.append("```text")
    L.append(f"profile:  {data['contract']['profile']} (promoted, legality fast)")
    L.append(f"corpus:   {data['contract']['positions']}-position S4 corpus")
    L.append(f"limit:    fixed depth {data['contract']['depth']}")
    L.append(f"TT:       16 MB cold | threads 1 | sparse sampler 256/1")
    L.append(f"repeat:   {data['contract']['repeat']} full corpus repetitions")
    L.append("```")
    L.append("")
    L.append("## The duplicated path")
    L.append("")
    L.append("```text")
    L.append("parent makes a move")
    L.append("  probe_child_draw(child):  generate FULL legal list  (discarded on Continue)")
    L.append("    -> terminal / intended-claim detection needs ONLY the emptiness")
    L.append("  negamax entered body:     generate FULL legal list AGAIN  (the duplicate)")
    L.append("qsearch edges:              generate FULL legal list, body replaces it with")
    L.append("                            tactical/evasion lists (never used)")
    L.append("```")
    L.append("")
    L.append("## Full-legal generation accounting (aggregate, 3 reps)")
    L.append("")
    L.append("```text")
    L.append(f"total full-legal calls                 {a['movegen_legal_calls']:,}")
    L.append(f"  probe_child_generations (discarded)  {a['probe_child_generations']:,}")
    L.append(f"    main edges                         {a['main_edge_probe_generations']:,}")
    L.append(f"    qsearch edges                      {a['qsearch_edge_probe_generations']:,}")
    L.append(f"    root edges                         {a['root_edge_probe_generations']:,}")
    L.append(f"  negamax body generations (duplicate) {a['negamax_body_generations']:,}")
    L.append(f"  root generations (once per search)   {a['root_generations']:,}")
    L.append(f"  final-evasion generations (single)   {a['final_evasion_generations']:,}")
    L.append("")
    L.append(f"DUPLICATES (body re-gen of a just-probed position):")
    L.append(f"  {a['duplicates']:,} calls = {a['dup_share_of_calls']:.1%} of full-legal calls")
    L.append(f"DISCARDED qsearch probe lists (body uses tactical/evasion):")
    L.append(f"  {a['discarded_qsearch_probes']:,} calls = {a['discarded_qsearch_probes'] / a['movegen_legal_calls']:.1%} of full-legal calls")
    L.append(f"TOTAL waste: {a['wasted_calls']:,} calls = {a['waste_share_of_calls']:.1%} of full-legal calls")
    L.append("```")
    L.append("")
    L.append("## Wall impact (same per-call cost assumption)")
    L.append("")
    total_ns = a["total_elapsed_us"] * 1000
    L.append("```text")
    L.append(f"movegen_legal bucket wall   {a['movegen_legal_wall_ns'] / 1e9:7.2f} s  ({a['movegen_legal_wall_ns'] / total_ns:.1%} of elapsed)")
    L.append(f"duplicate-regeneration wall {a['dup_wall_ns'] / 1e9:7.2f} s  ({a['dup_wall_ns'] / total_ns:.1%} of elapsed)")
    L.append(f"qsearch discarded wall      {(a['waste_wall_ns'] - a['dup_wall_ns']) / 1e9:7.2f} s  ({(a['waste_wall_ns'] - a['dup_wall_ns']) / total_ns:.1%} of elapsed)")
    L.append(f"TOTAL waste wall            {a['waste_wall_ns'] / 1e9:7.2f} s  ({a['waste_wall_ns'] / total_ns:.1%} of elapsed)")
    L.append("```")
    L.append("")
    L.append("## Per repetition")
    L.append("")
    L.append("| rep | full-legal calls | duplicates | qsearch discarded | waste % of calls |")
    L.append("|---|---|---|---|---|")
    for i, r in enumerate(data["per_rep"], 1):
        L.append(f"| {i} | "
                 f"{r['movegen_legal_calls']:,} | {r['duplicates']:,} | {r['discarded_qsearch_probes']:,} | "
                 f"{r['waste_share_of_calls']:.1%} |")
    L.append("")
    L.append("## Reading / S5.0B implication")
    L.append("")
    L.append("- The child probe needs ONLY the emptiness boolean (terminal) and the")
    L.append("  claim check (list-independent): `has_any_legal_move` is exactly")
    L.append("  equivalent for both.")
    L.append("- Replacing the probe's full-legal generation with a has-any probe")
    L.append("  removes ALL probe generations (main + qsearch + root edges) while")
    L.append("  the entered body keeps its single full generation.")
    L.append("- This is a fixed-depth tree-IDENTICAL change by construction (empty")
    L.append("  list iff no legal move).")
    L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"s50a_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
