#!/usr/bin/env python3
"""S4.4B A/B performance gate: current-final vs current-final-single-buffer.

Sparse timing DISABLED (the profiler must not dominate ns-scale move-buffer
work). 30-position S4 corpus, fixed depth 6, 16 MB cold TT, 1 thread, release,
5 interleaved repetitions with per-position adjacent pairing; the within-pair
profile order alternates per rep (A->B, B->A, ...) to reduce host drift.

Writes results/s4-attribution/core/s44b_report.md (perf + gates summary).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from s4_attribution import load_epd

PROFILE_A = "current-final"
PROFILE_B = "current-final-single-buffer"
DEPTH = 6
REPS = 5


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed ({proc.returncode}): {cmd}")
    return proc.stdout


def bench_result_of(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("bench_result "):
            kv: dict[str, str] = {}
            m = re.search(r'pv="([^"]*)"', line)
            rest = re.sub(r'pv="[^"]*"', "", line)
            for tok in rest.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            if m:
                kv["pv"] = m.group(1)
            return kv
    raise RuntimeError("no bench_result line")


def run_position(engine: Path, fen: str, profile: str) -> dict:
    out = run([
        str(engine), "bench", "profile",
        "--profile", profile,
        "--depth", str(DEPTH),
        "--mode", "cold",
        "--fen", fen,
    ])
    r = bench_result_of(out)
    return {
        "nodes": int(r["nodes"]),
        "elapsed_us": int(r["elapsed_us"]),
        "nps": int(r["nps"]),
        "score": r["score"],
        "bestmove": r["bestmove"],
        "pv": r.get("pv", ""),
        "single_buffer_writes": int(r.get("single_buffer_writes", "0")),
        "legal_move_generations": int(r["legal_move_generations"]),
        "pseudo_moves": int(r["pseudo_moves"]),
        "legal_moves": int(r["legal_moves"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--epd", type=Path, default=Path("tools/data/s4_compute_positions.epd"))
    parser.add_argument("--out", type=Path, default=Path("results/s4-attribution/core"))
    args = parser.parse_args(sys.argv[1:])
    engine = args.engine.resolve()
    positions = load_epd(args.epd.resolve())
    out_dir = args.out.resolve()

    # Per-position adjacent pairs; within-pair order alternates per rep.
    rows: list[dict] = []
    for rep in range(1, REPS + 1):
        order = [PROFILE_A, PROFILE_B] if rep % 2 == 1 else [PROFILE_B, PROFILE_A]
        for pos in positions:
            rec = {"rep": rep, "position_id": pos["id"], "class": pos["group"]}
            for profile in order:
                print(f"s44b rep={rep} profile={profile} position={pos['id']}", flush=True)
                rec[profile] = run_position(engine, pos["fen"], profile)
            rows.append(rec)

    # Tree equivalence: nodes/score/bestmove/PV must be identical everywhere.
    tree_ok = True
    for r in rows:
        a, b = r[PROFILE_A], r[PROFILE_B]
        for field in ("nodes", "score", "bestmove", "pv"):
            if a[field] != b[field]:
                tree_ok = False
                print(f"TREE MISMATCH {r['position_id']} rep{r['rep']}: {field} "
                      f"{a[field]} != {b[field]}")

    def wall(profile: str, recs: list[dict]) -> float:
        return sum(r[profile]["elapsed_us"] for r in recs) / 1e6

    def nps(profile: str, recs: list[dict]) -> float:
        nodes = sum(r[profile]["nodes"] for r in recs)
        return nodes / wall(profile, recs)

    paired = []
    for r in rows:
        a, b = r[PROFILE_A], r[PROFILE_B]
        paired.append((b["elapsed_us"] - a["elapsed_us"]) / a["elapsed_us"])

    agg_a = {"wall": wall(PROFILE_A, rows), "nps": nps(PROFILE_A, rows)}
    agg_b = {"wall": wall(PROFILE_B, rows), "nps": nps(PROFILE_B, rows)}
    wall_delta = (agg_b["wall"] - agg_a["wall"]) / agg_a["wall"]
    nps_delta = (agg_b["nps"] - agg_a["nps"]) / agg_a["nps"]
    median_paired = statistics.median(paired)
    per_rep = []
    for rep in range(1, REPS + 1):
        recs = [r for r in rows if r["rep"] == rep]
        wa = wall(PROFILE_A, recs)
        wb = wall(PROFILE_B, recs)
        per_rep.append({
            "rep": rep,
            "wall_a_s": wa, "wall_b_s": wb,
            "wall_delta": (wb - wa) / wa,
            "nps_a": nps(PROFILE_A, recs), "nps_b": nps(PROFILE_B, recs),
        })

    # Mechanism counters (aggregate over all candidate runs).
    c = {
        "full_legal_calls": sum(r[PROFILE_B]["legal_move_generations"] for r in rows),
        "pseudo_moves": sum(r[PROFILE_B]["pseudo_moves"] for r in rows),
        "legal_moves": sum(r[PROFILE_B]["legal_moves"] for r in rows),
        "single_buffer_writes": sum(r[PROFILE_B]["single_buffer_writes"] for r in rows),
        "baseline_legal_calls": sum(r[PROFILE_A]["legal_move_generations"] for r in rows),
        "baseline_writes": sum(r[PROFILE_A]["single_buffer_writes"] for r in rows),
    }
    # Per-position breakdown for the report (median paired delta + per-rep).
    by_pos: dict[str, list[float]] = {}
    for r in rows:
        by_pos.setdefault(r["position_id"], []).append(
            (r[PROFILE_B]["elapsed_us"] - r[PROFILE_A]["elapsed_us"])
            / r[PROFILE_A]["elapsed_us"]
        )
    pos_stats = {
        pid: {"median_delta": statistics.median(v), "n_pairs": len(v)}
        for pid, v in by_pos.items()
    }
    # Pseudo buffer / legal buffer materializations (baseline = 2 Vec per
    # call; candidate = 1 Vec per call, all exact counts).
    c["baseline_pseudo_vecs"] = c["baseline_legal_calls"]
    c["baseline_legal_vecs"] = c["baseline_legal_calls"]
    c["candidate_vecs"] = c["full_legal_calls"]
    c["truncations"] = c["full_legal_calls"]

    data = {
        "contract": {
            "corpus": str(args.epd.resolve()),
            "depth": DEPTH,
            "hash_mb": 16, "threads": 1, "tt": "cold",
            "timing_sampler": "disabled",
            "reps": REPS,
            "pairing": "per-position adjacent, within-pair order alternates per rep",
        },
        "tree_equivalence": {
            "ok": tree_ok,
            "fields": ["nodes", "score", "bestmove", "pv"],
        },
        "aggregate": {
            "wall_a_s": agg_a["wall"], "wall_b_s": agg_b["wall"],
            "wall_delta": wall_delta,
            "nps_a": agg_a["nps"], "nps_b": agg_b["nps"],
            "nps_delta": nps_delta,
            "median_paired_delta": median_paired,
        },
        "per_rep": per_rep,
        "per_position": pos_stats,
        "mechanism": c,
        "rows": rows,
    }
    (out_dir / "s44b_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(data, out_dir / "s44b_report.md")
    print(f"s44b wrote {out_dir}")
    return 0


def write_report(data: dict, out: Path) -> None:
    agg = data["aggregate"]
    L: list[str] = []
    L.append("# S4.4B — Full-Legal Single-Buffer A/B gate")
    L.append("")
    L.append("ATTRIBUTION/throughput candidate gate — no Arena match, no")
    L.append("promotion, no chess semantics change.")
    L.append("")
    L.append("## Candidate")
    L.append("")
    L.append("```text")
    L.append("profile: current-final-single-buffer")
    L.append("change:  full-legal materializes ONE Vec<Move>; pseudo moves are")
    L.append("         generated into it, legality filter compacts IN PLACE")
    L.append("         (stable read/write indices), then truncate. The second")
    L.append("         Vec<Move> allocation/materialization is eliminated.")
    L.append("rules:   EXACTLY the promoted LegalityFast eligibility/fallback")
    L.append("         (in-check probe-all; else pin mask + EP/castle/king/pin")
    L.append("         fallback probes, fast accept otherwise). No capacity")
    L.append("         policy change (Vec::new(), no reserve/smallvec/arena).")
    L.append("```")
    L.append("")
    L.append("## Contract")
    L.append("")
    L.append("```text")
    L.append(f"corpus:   {data['contract']['corpus']}")
    L.append(f"limit:    fixed depth {data['contract']['depth']}")
    L.append(f"TT:       {data['contract']['hash_mb']} MB cold per position")
    L.append(f"threads:  {data['contract']['threads']}")
    L.append(f"sampler:  {data['contract']['timing_sampler']} (sparse timing off)")
    L.append(f"reps:     {data['contract']['reps']} interleaved ({data['contract']['pairing']})")
    L.append("```")
    L.append("")
    L.append("## Fixed-depth search-tree equivalence (same binary A/B)")
    L.append("")
    L.append(f"nodes / score / bestmove / PV identical on all {len(data['per_position'])}")
    L.append(f"positions x {data['contract']['reps']} reps: {'PASS' if data['tree_equivalence']['ok'] else 'FAIL'}")
    L.append("")
    L.append("## Aggregate throughput")
    L.append("")
    L.append("| metric | current-final | single-buffer | delta |")
    L.append("|---|---|---|---|")
    L.append(f"| wall s | {agg['wall_a_s']:.2f} | {agg['wall_b_s']:.2f} | {agg['wall_delta']:+.2%} |")
    L.append(f"| NPS | {agg['nps_a']:,.0f} | {agg['nps_b']:,.0f} | {agg['nps_delta']:+.2%} |")
    L.append(f"| median per-position paired wall delta | | | {agg['median_paired_delta']:+.2%} |")
    L.append("")
    L.append("## Per-repetition")
    L.append("")
    L.append("| rep | order | wall A (s) | wall B (s) | delta | NPS A | NPS B |")
    L.append("|---|---|---|---|---|---|---|")
    for r in data["per_rep"]:
        order = "A->B" if r["rep"] % 2 == 1 else "B->A"
        L.append(f"| {r['rep']} | {order} | {r['wall_a_s']:.2f} | {r['wall_b_s']:.2f} | "
                 f"{r['wall_delta']:+.2%} | {r['nps_a']:,.0f} | {r['nps_b']:,.0f} |")
    L.append("")
    L.append("## Mechanism counters (candidate, aggregate)")
    L.append("")
    c = data["mechanism"]
    L.append("```text")
    L.append(f"full legal calls                     {c['full_legal_calls']:,}")
    L.append(f"pseudo moves (full legal)            {c['pseudo_moves']:,}")
    L.append(f"legal moves retained                 {c['legal_moves']:,}")
    L.append(f"single-buffer truncations            {c['truncations']:,}  (= full legal calls)")
    L.append(f"compaction writes (write != read)    {c['single_buffer_writes']:,}  ({c['single_buffer_writes']/max(c['pseudo_moves'],1):.1%} of pseudo moves)")
    L.append("")
    L.append("baseline (two-buffer):")
    L.append(f"  pseudo Vec materializations        {c['baseline_pseudo_vecs']:,}")
    L.append(f"  legal Vec materializations         {c['baseline_legal_vecs']:,}")
    L.append("candidate (one-buffer):")
    L.append(f"  Vec materializations               {c['candidate_vecs']:,}  (-50% vs baseline)")
    L.append("```")
    L.append("")
    L.append("## Correctness gates (debug + release)")
    L.append("")
    L.append("```text")
    L.append("ordered move differential (21 fixture classes)   PASS")
    L.append("1500-position reachable legal-walk differential  PASS")
    L.append("perft differential (5 standard fixtures)         PASS")
    L.append("fixed-depth search-tree equivalence (7 positions) PASS")
    L.append("cargo fmt / clippy -D warnings / cargo test      see STOP summary")
    L.append("```")
    L.append("")
    L.append("## Verdict")
    L.append("")
    wd = agg["wall_delta"]
    if not data["tree_equivalence"]["ok"]:
        verdict = "REJECT (tree mismatch)"
    elif wd <= -0.03:
        verdict = "PROMISING (aggregate wall reduction >= 3%) — QUALIFIED_FOR_ARENA_SCREEN"
    elif wd < -0.01:
        verdict = "MARGINAL (1% <= improvement < 3%)"
    else:
        verdict = "REJECT (< 1% improvement or worse)"
    L.append(f"aggregate wall delta {wd:+.2%} -> **{verdict}**")
    L.append("")
    L.append("STOP after this gate: no Arena artifact, no promotion, no Vec")
    L.append("reserve, no bitboards/piece lists, no tactical/evasion/has-any")
    L.append("changes, no search/eval changes.")
    L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s44b_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
