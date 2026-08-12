#!/usr/bin/env python3
"""S5.0B A/B performance gate: current-final vs current-final-single-generation.

Sparse timing DISABLED for the perf comparison (30-position S4 corpus, fixed
depth 6, 16 MB cold TT, 1 thread, release, 5 interleaved per-position paired
repetitions with alternating within-pair order). A separate timing-enabled
single pass collects the S5.0A mechanism counters (probe generations, has-any
calls) proving the mechanism. Writes results/s4-attribution/core/s50b_report.md
+ s50b_data.json.
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
PROFILE_B = "current-final-single-generation"
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


def bench_timing_of(stdout: str) -> dict[str, str]:
    for line in stdout.splitlines():
        if line.startswith("bench_timing "):
            return dict(
                (kv[0], kv[1])
                for kv in (tok.split("=", 1) for tok in line.split())
                if len(kv) == 2
            )
    raise RuntimeError("no bench_timing line")


def run_position(engine: Path, fen: str, profile: str, timing: bool) -> dict:
    cmd = [
        str(engine), "bench", "profile",
        "--profile", profile,
        "--depth", str(DEPTH),
        "--mode", "cold",
        "--fen", fen,
    ]
    if timing:
        cmd += ["--timing-sample", "256"]
    out = run(cmd)
    r = bench_result_of(out)
    rec = {
        "nodes": int(r["nodes"]),
        "elapsed_us": int(r["elapsed_us"]),
        "nps": int(r["nps"]),
        "score": r["score"],
        "bestmove": r["bestmove"],
        "pv": r.get("pv", ""),
    }
    if timing:
        t = bench_timing_of(out)
        rec.update({
            "movegen_legal_calls": int(t["movegen_legal_calls"]),
            "movegen_has_any_calls": int(t["movegen_has_any_calls"]),
            "probe_child_generations": int(t["probe_child_generations"]),
            "negamax_body_generations": int(t["negamax_body_generations"]),
        })
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--epd", type=Path, default=Path("tools/data/s4_compute_positions.epd"))
    parser.add_argument("--out", type=Path, default=Path("results/s4-attribution/core"))
    args = parser.parse_args(sys.argv[1:])
    engine = args.engine.resolve()
    positions = load_epd(args.epd.resolve())
    out_dir = args.out.resolve()

    rows: list[dict] = []
    for rep in range(1, REPS + 1):
        order = [PROFILE_A, PROFILE_B] if rep % 2 == 1 else [PROFILE_B, PROFILE_A]
        for pos in positions:
            rec = {"rep": rep, "position_id": pos["id"], "class": pos["group"]}
            for profile in order:
                print(f"s50b rep={rep} profile={profile} position={pos['id']}", flush=True)
                rec[profile] = run_position(engine, pos["fen"], profile, timing=False)
            rows.append(rec)

    tree_ok = True
    for r in rows:
        a, b = r[PROFILE_A], r[PROFILE_B]
        for field in ("nodes", "score", "bestmove", "pv"):
            if a[field] != b[field]:
                tree_ok = False
                print(f"TREE MISMATCH {r['position_id']} rep{r['rep']}: {field}")

    def wall(profile: str, recs: list[dict]) -> float:
        return sum(r[profile]["elapsed_us"] for r in recs) / 1e6

    def nps(profile: str, recs: list[dict]) -> float:
        return sum(r[profile]["nodes"] for r in recs) / wall(profile, recs)

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
        wa, wb = wall(PROFILE_A, recs), wall(PROFILE_B, recs)
        per_rep.append({
            "rep": rep,
            "wall_a_s": wa, "wall_b_s": wb,
            "wall_delta": (wb - wa) / wa,
            "nps_a": nps(PROFILE_A, recs), "nps_b": nps(PROFILE_B, recs),
        })

    # Mechanism pass (timing enabled, one corpus pass per profile).
    mech: dict[str, dict] = {}
    for profile in (PROFILE_A, PROFILE_B):
        total = {k: 0 for k in ("movegen_legal_calls", "movegen_has_any_calls",
                                "probe_child_generations", "negamax_body_generations")}
        for pos in positions:
            print(f"s50b mech {profile} {pos['id']}", flush=True)
            rec = run_position(engine, pos["fen"], profile, timing=True)
            for k in total:
                total[k] += rec[k]
        mech[profile] = total

    data = {
        "contract": {
            "corpus": str(args.epd.resolve()), "depth": DEPTH, "hash_mb": 16,
            "threads": 1, "tt": "cold", "sampler": "disabled (perf)",
            "reps": REPS, "pairing": "per-position adjacent, alternating order",
        },
        "tree_equivalence": {"ok": tree_ok, "fields": ["nodes", "score", "bestmove", "pv"]},
        "aggregate": {
            "wall_a_s": agg_a["wall"], "wall_b_s": agg_b["wall"],
            "wall_delta": wall_delta,
            "nps_a": agg_a["nps"], "nps_b": agg_b["nps"],
            "nps_delta": nps_delta,
            "median_paired_delta": median_paired,
        },
        "per_rep": per_rep,
        "mechanism": mech,
    }
    (out_dir / "s50b_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(data, out_dir / "s50b_report.md")
    print("wrote", out_dir)
    return 0


def write_report(data: dict, out: Path) -> None:
    agg = data["aggregate"]
    L: list[str] = []
    L.append("# S5.0B — Single-Generation Child Probe A/B gate")
    L.append("")
    L.append("Same-tree throughput candidate — no chess semantics change.")
    L.append("")
    L.append("## Candidate")
    L.append("")
    L.append("```text")
    L.append("profile: current-final-single-generation")
    L.append("change:  probe_child_draw uses has_any_legal_move (emptiness")
    L.append("         boolean) instead of a full legal list discarded on")
    L.append("         Continue. The entered body keeps its single full")
    L.append("         generation. S5.0A: 64.8% of full-legal calls were")
    L.append("         discarded probe lists (35.2% duplicate body re-gen,")
    L.append("         29.6% qsearch probe lists).")
    L.append("equivalence: has_any false <=> no legal move -> identical")
    L.append("         terminal/claim decisions -> fixed-depth tree identical")
    L.append("         by construction.")
    L.append("```")
    L.append("")
    L.append("## Contract")
    L.append("")
    L.append("```text")
    L.append(f"corpus:   {data['contract']['corpus']}")
    L.append(f"limit:    fixed depth {data['contract']['depth']}")
    L.append(f"TT:       {data['contract']['hash_mb']} MB cold per position")
    L.append(f"threads:  {data['contract']['threads']}")
    L.append(f"sampler:  {data['contract']['sampler']} (sparse timing off)")
    L.append(f"reps:     {data['contract']['reps']} interleaved paired")
    L.append("```")
    L.append("")
    L.append("## Fixed-depth search-tree equivalence (same binary A/B)")
    L.append("")
    L.append(f"nodes / score / bestmove / PV identical on all positions x reps: "
            f"{'PASS' if data['tree_equivalence']['ok'] else 'FAIL'}")
    L.append("")
    L.append("## Aggregate throughput")
    L.append("")
    L.append("| metric | current-final | single-generation | delta |")
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
    L.append("## Mechanism counters (one timing-enabled corpus pass per profile)")
    L.append("")
    m = data["mechanism"]
    L.append("```text")
    for k in ("movegen_legal_calls", "probe_child_generations",
              "negamax_body_generations", "movegen_has_any_calls"):
        L.append(f"{k:28s} A={m[PROFILE_A][k]:>12,}   B={m[PROFILE_B][k]:>12,}")
    L.append("```")
    L.append("")
    L.append("## Verdict")
    L.append("")
    wd = agg["wall_delta"]
    if not data["tree_equivalence"]["ok"]:
        verdict = "REJECT (tree mismatch)"
    elif wd <= -0.03:
        verdict = "PROMISING (aggregate wall reduction >= 3%)"
    elif wd < -0.01:
        verdict = "MARGINAL (1% <= improvement < 3%)"
    else:
        verdict = "REJECT (< 1% improvement or worse)"
    L.append(f"aggregate wall delta {wd:+.2%} -> **{verdict}**")
    L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"s50b_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
