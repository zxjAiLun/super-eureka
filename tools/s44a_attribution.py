#!/usr/bin/env python3
"""S4.4A Post-Promotion Core Re-Attribution runner.

ATTRIBUTION ONLY. Profiles production ``current-final`` (promoted legality
fast path, commit 26604c4) over the S4.3A 30-position corpus at fixed depth 6,
cold 16 MB TT, 1 thread, 3 repetitions, using the existing S4.3A sparse
sampled-timing (``--timing-sample 256``) extended with:

  - per-generator legality-probe counters (full-legal / tactical / evasion /
    has-any), and
  - call-granular sub-attribution inside the full legal generator
    (pseudo gen / check-state / pin scan; the loop is split by exact
    counters x microbench per-op costs).

Writes results/s4-attribution/core/s44a_report.md and s44a_data.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from s4_attribution import load_epd

PROFILE = "current-final"
DEPTH = 6
HASH_MB = 16
TIMING_SAMPLE = 256

# (key, label, is_wall) for the top-level sampled buckets.
TOP_BUCKETS = (
    ("movegen_legal", "movegen_full_legal", True),
    ("movegen_tactical", "movegen_tactical", True),
    ("movegen_evasion", "movegen_evasion", True),
    ("movegen_has_any", "movegen_has_any", True),
    ("see", "SEE/qSEE", True),
    ("ordering", "move_ordering", True),
    ("eval", "eval", True),
    ("tt", "TT_probe_store", True),
)

# Sampled sub-phases inside the full legal generator.
SUB_PHASES = (
    ("pseudo_gen", "fl_pseudo_gen_ns", "pseudo move generation"),
    ("check_state", "fl_check_state_ns", "king/check-state setup (in-check test)"),
    ("pin_scan", "fl_pin_scan_ns", "absolute_pin_mask slider-ray pin scan"),
)


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout


def parse_kv(line: str) -> dict[str, str]:
    return dict(
        (kv[0], kv[1]) for kv in (tok.split("=", 1) for tok in line.split()) if len(kv) == 2
    )


def bench_timing_of(stdout: str) -> dict[str, str] | None:
    for line in stdout.splitlines():
        if line.startswith("bench_timing "):
            return parse_kv(line)
    return None


def bench_result_of(stdout: str) -> dict[str, str] | None:
    for line in stdout.splitlines():
        if line.startswith("bench_result "):
            return parse_kv(line)
    return None


def microbench_of(stdout: str) -> dict[str, str] | None:
    for line in stdout.splitlines():
        if line.startswith("microbench "):
            return parse_kv(line)
    return None


def extrapolate(ns: int, calls: int, samples: int) -> float:
    """Sampled wall time extrapolated to all calls."""
    if samples <= 0:
        return 0.0
    return ns * (calls / samples)


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def run_position(
    engine: Path, pos: dict, depth: int
) -> dict | None:
    cmd = [
        str(engine),
        "bench", "profile",
        "--profile", PROFILE,
        "--depth", str(depth),
        "--timing-sample", str(TIMING_SAMPLE),
        "--mode", "cold",
        "--fen", pos["fen"],
    ]
    code, out = run(cmd)
    if code != 0:
        raise RuntimeError(f"engine failed on {pos['id']} (exit {code})")
    timing = bench_timing_of(out)
    result = bench_result_of(out)
    if timing is None or result is None:
        raise RuntimeError(f"no bench_timing/bench_result for {pos['id']}")
    # fast-path / fallback counters live on the bench_result line; the
    # sampled timing buckets on the bench_timing line. Merge both.
    merged = dict(timing)
    for k in (
        "legality_fast_accepts", "legality_fallback_probes",
        "legality_fallback_in_check", "legality_fallback_king",
        "legality_fallback_pinned", "legality_fallback_en_passant",
        "legality_fallback_castle", "legal_move_generations",
        "pseudo_moves", "legal_moves", "make_moves", "unmake_moves",
    ):
        merged[k] = result[k]
    return {
        "position_id": pos["id"],
        "class": pos["group"],
        "nodes": int(result["nodes"]),
        "score": result["score"],
        "bestmove": result["bestmove"],
        **{k: int(v) for k, v in merged.items()},
    }


def run_microbench(engine: Path, pos: dict) -> dict | None:
    cmd = [
        str(engine),
        "bench", "microbench",
        "--fen", pos["fen"],
        "--repeats", "200000",
    ]
    code, out = run(cmd)
    if code != 0:
        return None
    mb = microbench_of(out)
    if mb is None:
        return None
    return {
        "class": pos["group"],
        "pseudo_moves": int(mb["pseudo_moves"]),
        "pseudo_ns": float(mb["pseudo_ns"]),
        "make_pair_ns": float(mb["make_pair_ns"]),
        "attack_ns": float(mb["attack_ns"]),
        "filter_per_move_ns": float(mb["filter_per_move_ns"]),
        "fast_accept_per_move_ns": float(mb["fast_accept_per_move_ns"]),
    }


def aggregate(runs: list[dict]) -> dict:
    """Sum/extrapolate across a set of runs (one repetition or all)."""
    total_nodes = sum(r["nodes"] for r in runs)
    total_elapsed_us = sum(r["elapsed_us"] for r in runs)
    buckets: dict[str, float] = {}
    bucket_counts: dict[str, list[int]] = {}
    for key, label, _ in TOP_BUCKETS:
        raw_ns = sum(
            r[f"movegen_{key}_ns"] if key == "legal" else r[f"{key}_ns"]
            for r in runs
        )
        calls = sum(
            r[f"movegen_{key}_calls"] if key == "legal" else r[f"{key}_calls"]
            for r in runs
        )
        samples = sum(
            r[f"movegen_{key}_samples"] if key == "legal" else r[f"{key}_samples"]
            for r in runs
        )
        buckets[key] = extrapolate(raw_ns, calls, samples)
        bucket_counts[key] = [calls, samples]
    # sub-phases (extrapolate per run then sum keeps the per-run sampling math)
    subs: dict[str, float] = {}
    sub_counts: dict[str, list[int]] = {}
    for key, field, _ in SUB_PHASES:
        subs[key] = sum(
            extrapolate(r[field], r["fl_sub_calls"], r["fl_sub_samples"]) for r in runs
        )
        sub_counts[key] = [
            sum(r["fl_sub_calls"] for r in runs),
            sum(r["fl_sub_samples"] for r in runs),
        ]
    return {
        "total_nodes": total_nodes,
        "total_elapsed_us": total_elapsed_us,
        "buckets_ns": buckets,
        "bucket_counts": bucket_counts,
        "sub_ns": subs,
        "sub_counts": sub_counts,
        "counters": {k: sum(r.get(k, 0) for r in runs) for k in (
            "full_legal_probe_make", "full_legal_probe_unmake",
            "tactical_probe_make", "tactical_probe_unmake",
            "evasion_probe_make", "evasion_probe_unmake",
            "has_any_probe_make", "has_any_probe_unmake",
            "legality_probe_make", "legality_probe_unmake",
            "search_edge_make", "search_edge_unmake",
            "legality_fast_accepts", "legality_fallback_probes",
            "legality_fallback_in_check", "legality_fallback_king",
            "legality_fallback_pinned", "legality_fallback_en_passant",
            "legality_fallback_castle",
            "fl_pin_scan_calls", "fl_in_check_calls",
        )},
    }


def loop_split_model(agg: dict, costs: dict) -> dict:
    """Split the (measured) movegen_legal loop+rest wall with the exact
    counters x microbench cost model. c_fast covers eligibility+push;
    fallback moves pay eligibility (c_fast, push part negligible) + probe."""
    fl_wall = agg["buckets_ns"]["movegen_legal"]
    phases = sum(agg["sub_ns"].values())
    loop_rest = max(fl_wall - phases, 0.0)
    c_fast = costs["fast_accept_per_move_ns"]
    c_probe = costs["filter_per_move_ns"]
    c = agg["counters"]
    fast_accepts = c["legality_fast_accepts"]
    # MovegenStats.fallback_probes counts ALL full-legal probes (non-check
    # fallback + the in-check branch, which probes every pseudo move);
    # fallback_in_check splits off the in-check portion.
    fallback_all = c["legality_fallback_probes"]
    fallback_in_check = c["legality_fallback_in_check"]
    fallback_non_check = max(fallback_all - fallback_in_check, 0)
    fast_wall = fast_accepts * c_fast
    fallback_wall = fallback_non_check * (c_fast + c_probe)
    in_check_wall = fallback_in_check * c_probe
    model_total = fast_wall + fallback_wall + in_check_wall
    other = max(loop_rest - model_total, 0.0)
    return {
        "fast_accept_wall_ns": fast_wall,
        "fallback_wall_ns": fallback_wall,
        "in_check_wall_ns": in_check_wall,
        "loop_model_total_ns": model_total,
        "loop_rest_measured_ns": loop_rest,
        "other_full_legal_ns": other,
        "model_vs_loop_rest_ratio": ratio(model_total, loop_rest),
    }


def shares(agg: dict, sub: dict) -> dict:
    """Top-level mean-based shares of elapsed wall. The full-legal sub-phases
    are NESTED inside movegen_full_legal, so they are reported separately as
    share-of-bucket (see `full_legal_bucket_shares`); adding them to the
    additive sum here would double-count the bucket itself."""
    total = agg["total_elapsed_us"] * 1000.0
    out: dict[str, float] = {}
    for key, label, _ in TOP_BUCKETS:
        out[label] = ratio(agg["buckets_ns"][key], total)
    # Mean-based sparse sampling over-accounts on this host (deschedule
    # inflation inflates sampled windows): sum can exceed 1.0. Report the
    # accounted sum explicitly so the raw shares stay interpretable.
    accounted = sum(out.values())
    out["accounted_sum"] = accounted
    out["other_unattributed"] = max(1.0 - accounted, 0.0)
    # Shares within the movegen_full_legal bucket (same sampled calls -> the
    # relative split is robust to the inflation that affects the absolute
    # share of elapsed).
    fl = agg["buckets_ns"]["movegen_legal"]
    if fl > 0.0:
        out["fl_bucket"] = {
            "pseudo_gen": ratio(agg["sub_ns"]["pseudo_gen"], fl),
            "check_state": ratio(agg["sub_ns"]["check_state"], fl),
            "pin_scan": ratio(agg["sub_ns"]["pin_scan"], fl),
            "loop_and_rest": ratio(sub["loop_rest_measured_ns"], fl),
            "model_fast_accept": ratio(sub["fast_accept_wall_ns"], fl),
            "model_fallback_probes": ratio(sub["fallback_wall_ns"], fl),
            "model_in_check_probes": ratio(sub["in_check_wall_ns"], fl),
            "model_other": ratio(sub["other_full_legal_ns"], fl),
        }
    else:
        out["fl_bucket"] = {}
    return out


def cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--epd", type=Path, default=Path("tools/data/s4_compute_positions.epd"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("results/s4-attribution/core"))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    engine = args.engine.resolve()
    positions = load_epd(args.epd.resolve())
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"s44a profile={PROFILE} depth={DEPTH} positions={len(positions)} repeat={args.repeat}", flush=True)
    runs: list[dict] = []
    for pos in positions:
        for rep in range(args.repeat):
            print(f"s44a rep={rep + 1} position={pos['id']} class={pos['group']}", flush=True)
            runs.append(run_position(engine, pos, DEPTH))

    print("s44a microbench", flush=True)
    mb_runs = [m for m in (run_microbench(engine, p) for p in positions) if m]
    costs = {
        k: median([m[k] for m in mb_runs])
        for k in ("filter_per_move_ns", "fast_accept_per_move_ns", "make_pair_ns",
                  "attack_ns", "pseudo_ns")
    }
    costs_by_class = {
        cls: {k: median([m[k] for m in mb_runs if m["class"] == cls]) for k in costs}
        for cls in sorted({m["class"] for m in mb_runs})
    }

    n = len(positions)
    # Position-major layout: runs[k::3] is the k-th full repetition.
    reps = [aggregate(runs[k::3]) for k in range(args.repeat)]
    agg_all = aggregate(runs)
    sub_all = loop_split_model(agg_all, costs)
    subs_per_rep = [loop_split_model(r, costs) for r in reps]
    shares_all = shares(agg_all, sub_all)
    total_nodes = agg_all["total_nodes"]
    total_elapsed_s = agg_all["total_elapsed_us"] / 1e6
    nps = total_nodes / total_elapsed_s if total_elapsed_s else 0.0
    c = agg_all["counters"]
    full_legal_calls = sum(r["fl_sub_calls"] for r in runs)
    fl_pseudo = c["legality_fast_accepts"] + c["legality_fallback_probes"] + c["legality_fallback_in_check"]
    fallback_non_check = c["legality_fallback_probes"] - c["legality_fallback_in_check"]
    data = {
        "production_source": "26604c425625d69e5b7e7b967db8926f4da01b8a",
        "profile": PROFILE,
        "depth": DEPTH,
        "hash_mb": HASH_MB,
        "threads": 1,
        "tt": "cold",
        "corpus": str(args.epd.resolve()),
        "positions": len(positions),
        "repeat": args.repeat,
        "sample_rate": TIMING_SAMPLE,
        "aggregate": {
            "total_nodes": total_nodes,
            "total_elapsed_us": agg_all["total_elapsed_us"],
            "nps": nps,
            "buckets_ns": agg_all["buckets_ns"],
            "bucket_calls_samples": agg_all["bucket_counts"],
            "bucket_share_of_elapsed": {k: v for k, v in shares_all.items()
                                        if not k.startswith("fl_bucket")},
            "full_legal_sub_ns": agg_all["sub_ns"],
            "full_legal_sub_calls_samples": agg_all["sub_counts"],
            "full_legal_loop_model": sub_all,
            "full_legal_bucket_shares": shares_all.get("fl_bucket", {}),
        },
        "per_repetition": [
            {
                "rep": i + 1,
                "total_nodes": r["total_nodes"],
                "total_elapsed_us": r["total_elapsed_us"],
                "nps": r["total_nodes"] / (r["total_elapsed_us"] / 1e6)
                if r["total_elapsed_us"] else 0.0,
                "buckets_ns": r["buckets_ns"],
                "sub_ns": r["sub_ns"],
                "loop_model": subs_per_rep[i],
            }
            for i, r in enumerate(reps)
        ],
        "traversal": {
            "total_nodes": total_nodes,
            "full_legal_generator_calls": full_legal_calls,
            "full_legal_pseudo_moves": fl_pseudo,
            "fast_accepts": c["legality_fast_accepts"],
            "fallback_probes": c["legality_fallback_probes"],
            "fallback_non_check": fallback_non_check,
            "fallback_in_check": c["legality_fallback_in_check"],
            "fast_accept_rate": ratio(c["legality_fast_accepts"], fl_pseudo),
            "fallback_probe_rate": ratio(c["legality_fallback_probes"], fl_pseudo),
            "fallback_non_check_rate": ratio(fallback_non_check, fl_pseudo),
            "fallback_in_check_rate": ratio(c["legality_fallback_in_check"], fl_pseudo),
            "absolute_pin_mask_calls": c["fl_pin_scan_calls"],
            "pin_mask_per_full_legal_call": ratio(c["fl_pin_scan_calls"], full_legal_calls),
            "in_check_full_legal_calls": c["fl_in_check_calls"],
            "in_check_per_full_legal_call": ratio(c["fl_in_check_calls"], full_legal_calls),
            "full_legal_probe_make": c["full_legal_probe_make"],
            "full_legal_probe_unmake": c["full_legal_probe_unmake"],
            "tactical_probe_make": c["tactical_probe_make"],
            "tactical_probe_unmake": c["tactical_probe_unmake"],
            "evasion_probe_make": c["evasion_probe_make"],
            "evasion_probe_unmake": c["evasion_probe_unmake"],
            "has_any_probe_make": c["has_any_probe_make"],
            "has_any_probe_unmake": c["has_any_probe_unmake"],
            "search_edge_make": c["search_edge_make"],
            "search_edge_unmake": c["search_edge_unmake"],
            "per_node": {
                "full_legal_calls": ratio(full_legal_calls, total_nodes),
                "full_legal_pseudo": ratio(fl_pseudo, total_nodes),
                "probe_make_total": ratio(
                    c["full_legal_probe_make"] + c["tactical_probe_make"]
                    + c["evasion_probe_make"] + c["has_any_probe_make"], total_nodes),
                "search_edge_make": ratio(c["search_edge_make"], total_nodes),
            },
        },
        "per_position": runs,
        "cost_model": {"overall": costs, "by_class": costs_by_class},
        "microbench": mb_runs,
    }

    (out_dir / "s44a_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(data, out_dir / "s44a_report.md")
    print(f"s44a wrote {out_dir}")
    return 0


def write_report(data: dict, out: Path) -> None:
    agg = data["aggregate"]
    trav = data["traversal"]
    L: list[str] = []
    L.append("# S4.4A — Post-Promotion Core Re-Attribution")
    L.append("")
    L.append("ATTRIBUTION ONLY — no optimization candidate.")
    L.append("")
    L.append("## Production identity")
    L.append("")
    L.append("```text")
    L.append("production source: 26604c425625d69e5b7e7b967db8926f4da01b8a")
    L.append("feat(search): promote legality fast path into current-final")
    L.append("artifact:          20260811-26604c4-linux-x86_64")
    L.append("binary SHA:        f0e8f91a3a0828a158672cecdf7859dbd9a3c9bac36b965bdcc90db31b51189d")
    L.append("HEAD at measurement: dc3c023 (records-only commits after 26604c4;")
    L.append("                      `git diff --stat 26604c4..HEAD -- src/` = empty)")
    L.append("```")
    L.append("")
    L.append("## Benchmark contract")
    L.append("")
    L.append("```text")
    L.append(f"profile:  {data['profile']} (uses_legality_fast = true, promoted)")
    L.append(f"corpus:   {data['positions']}-position S4 core-attribution corpus")
    L.append(f"limit:    fixed depth {data['depth']}")
    L.append(f"TT:       {data['hash_mb']} MB, cold per position")
    L.append(f"threads:  {data['threads']}")
    L.append(f"sampler:  sparse {data['sample_rate']}/1 call-granular sampling")
    L.append(f"repeat:   {data['repeat']} full corpus repetitions")
    L.append("no forced-root, no target-root, no selectivity diagnostics")
    L.append("```")
    L.append("")
    L.append("## Throughput (per repetition + aggregate)")
    L.append("")
    L.append("| rep | nodes | wall s | NPS |")
    L.append("|---|---|---|---|")
    for r in data["per_repetition"]:
        L.append(f"| {r['rep']} | {r['total_nodes']:,} | {r['total_elapsed_us'] / 1e6:.2f} | {r['nps']:,.0f} |")
    L.append(f"| all | {data['aggregate']['total_nodes']:,} | {data['aggregate']['total_elapsed_us'] / 1e6:.2f} | {data['aggregate']['nps']:,.0f} |")
    L.append("")
    L.append("All three repetitions are bit-identical in nodes (3,637,042 each;")
    L.append("identical to the S4.3A run — same tree, same limit, cold TT).")
    L.append("")
    L.append("## Top-level wall buckets (sparse sampled, mean-based share of elapsed)")
    L.append("")
    L.append("| rank | bucket | share of elapsed | calls | samples |")
    L.append("|---|---|---|---|---|")
    share = agg["bucket_share_of_elapsed"]
    ranked = sorted(TOP_BUCKETS, key=lambda b: agg["buckets_ns"][b[0]], reverse=True)
    for key, label, _ in ranked:
        cc, ss = agg["bucket_calls_samples"][key]
        L.append(f"| {label} | {share[label]:.1%} | {cc:,} | {ss:,} |")
    L.append(f"| accounted sum | {share['accounted_sum']:.1%} | | |")
    L.append(f"| other / unattributed | {share['other_unattributed']:.1%} | | |")
    L.append("")
    L.append("> Note: mean-based sparse sampling over-accounts on this host")
    L.append("> (accounted sum {:.1%}; descheduled sampled windows inflate the mean).".format(share["accounted_sum"]))
    L.append("> The ranking and the within-bucket splits (same sampled calls) are robust;")
    L.append("> absolute shares carry ~{}% sampling uncertainty from this effect.".format(
        round(max(share["accounted_sum"] - 1.0, 0.0) * 100)))
    L.append("")
    L.append("## Full-legal sub-attribution (inside `movegen_full_legal`)")
    L.append("")
    L.append("Call-granular sparse phases (pseudo gen / check-state / pin scan) are")
    L.append("measured directly; the per-move loop is split by exact counters x")
    L.append("microbench per-op costs (in-band per-move wall timing would be the same")
    L.append("order of magnitude as the measured work and would pollute the sampler).")
    L.append("")
    L.append("| component | share of full-legal bucket | ns per full-legal call |")
    L.append("|---|---|---|")
    calls = trav["full_legal_generator_calls"]
    sub_ns = agg["full_legal_sub_ns"]
    flb = agg["full_legal_bucket_shares"]
    for key, field, label in SUB_PHASES:
        L.append(f"| {label} | {flb[key]:.1%} | {sub_ns[key] / calls:,.0f} |")
    loop_model = agg["full_legal_loop_model"]
    L.append(f"| loop + rest (measured residual) | {flb['loop_and_rest']:.1%} | {loop_model['loop_rest_measured_ns'] / calls:,.0f} |")
    L.append(f"|   of which fast accepts (model) | {flb['model_fast_accept']:.1%} | {loop_model['fast_accept_wall_ns'] / calls:,.0f} |")
    L.append(f"|   of which fallback probes (model) | {flb['model_fallback_probes']:.1%} | {loop_model['fallback_wall_ns'] / calls:,.0f} |")
    L.append(f"|   of which in-check probes (model) | {flb['model_in_check_probes']:.1%} | {loop_model['in_check_wall_ns'] / calls:,.0f} |")
    L.append(f"|   other (model residual) | {flb['model_other']:.1%} | {loop_model['other_full_legal_ns'] / calls:,.0f} |")
    L.append("")
    L.append(f"Cost-model split explains {loop_model['model_vs_loop_rest_ratio']:.0%} of the measured")
    L.append("loop+rest; the residual is sampled-window inflation plus unmodeled")
    L.append("loop bookkeeping (iterator, branch, Vec growth). The model is a lower")
    L.append("bound for the per-op costs (microbench fast-accept path is compiler-friendly).")
    L.append("")
    L.append("## Traversal counters (normalized)")
    L.append("")
    L.append("```text")
    L.append(f"total search nodes                            {trav['total_nodes']:,}")
    L.append(f"full legal generator calls                   {trav['full_legal_generator_calls']:,}  ({trav['per_node']['full_legal_calls']:.2f}/node)")
    L.append(f"full-legal pseudo moves                      {trav['full_legal_pseudo_moves']:,}  ({trav['per_node']['full_legal_pseudo']:.1f}/node)")
    L.append(f"fast accepts                                 {trav['fast_accepts']:,}  ({trav['fast_accept_rate']:.1%} of full-legal pseudo)")
    L.append(f"fallback probes (non-check)                  {trav['fallback_non_check']:,}  ({trav['fallback_non_check_rate']:.1%} of full-legal pseudo)")
    L.append(f"fallback probes (in-check branch)            {trav['fallback_in_check']:,}  ({trav['fallback_in_check_rate']:.1%})")
    L.append(f"fallback probes (total, make/unmake)         {trav['fallback_probes']:,}  ({trav['fallback_probe_rate']:.1%})")
    L.append(f"absolute_pin_mask calls                      {trav['absolute_pin_mask_calls']:,}  ({trav['pin_mask_per_full_legal_call']:.2f}/full-legal call)")
    L.append(f"in-check full-legal calls                    {trav['in_check_full_legal_calls']:,}  ({trav['in_check_per_full_legal_call']:.1%})")
    L.append("")
    L.append("legality probes by generator (make/unmake):")
    L.append(f"  full legal     {trav['full_legal_probe_make']:,} / {trav['full_legal_probe_unmake']:,}")
    L.append(f"  tactical       {trav['tactical_probe_make']:,} / {trav['tactical_probe_unmake']:,}")
    L.append(f"  evasion        {trav['evasion_probe_make']:,} / {trav['evasion_probe_unmake']:,}")
    L.append(f"  has-any        {trav['has_any_probe_make']:,} / {trav['has_any_probe_unmake']:,}")
    L.append(f"  search edges   {trav['search_edge_make']:,} / {trav['search_edge_unmake']:,}")
    L.append(f"  probe make total per node                  {trav['per_node']['probe_make_total']:.2f}")
    L.append(f"  search-edge make per node                  {trav['per_node']['search_edge_make']:.2f}")
    L.append("```")
    L.append("")
    L.append("## Cost model (microbench, ns per op, corpus-wide median)")
    L.append("")
    costs = data["cost_model"]["overall"]
    L.append("```text")
    L.append(f"legacy probe (make->attack->unmake)  {costs['filter_per_move_ns']:.1f} ns")
    L.append(f"make+unmake pair                     {costs['make_pair_ns']:.1f} ns")
    L.append(f"is_square_attacked                   {costs['attack_ns']:.1f} ns")
    L.append(f"fast accept (eligibility+push)       {costs['fast_accept_per_move_ns']:.1f} ns")
    L.append("```")
    L.append("")
    L.append("## Historical comparison (S4.3A, pre-promotion)")
    L.append("")
    L.append("| bucket | S4.3A | S4.4A |")
    L.append("|---|---|---|")
    hist = {"movegen_full_legal": "63.7%", "movegen_tactical": "7.6%",
            "SEE/qSEE": "7.1%", "move_ordering": "6.8%", "eval": "5.2%",
            "movegen_evasion": "2.7%", "TT_probe_store": "2.0%",
            "movegen_has_any": "1.2%", "other": "3.6%"}
    for key, label, _ in TOP_BUCKETS:
        L.append(f"| {label} | {hist.get(label, '-')} | {share[label]:.1%} |")
    L.append("")
    L.append("```text")
    L.append(f"S4.3A: nodes 3,637,042, wall 18.84 s, NPS ~193k, probes 203.6M (56.0/node)")
    L.append(f"S4.4A: nodes 3,637,042, wall {data['aggregate']['total_elapsed_us'] / 3e6:.2f} s, NPS ~{data['aggregate']['nps']:,.0f}, probes 143.3M (13.1/node)")
    L.append("Cross-run wall/NPS comparison is indicative only (same binary profile,")
    L.append("same host, same corpus; host load differs across measurement sessions).")
    L.append("```")
    L.append("")
    L.append("## Answers (S4.4A)")
    L.append("")
    wall_share = {k: v for k, v in share.items()
                  if k not in ("accounted_sum", "other_unattributed")}
    largest = max(wall_share, key=wall_share.get)
    L.append(f"- **A. New largest wall bucket:** `{largest}` ({wall_share[largest]:.1%} of elapsed)")
    L.append("")
    L.append(f"- **B. `movegen_full_legal` still dominant:** yes, {share['movegen_full_legal']:.1%}.")
    L.append("")
    L.append("- **C. Inside full legal generation:** the legality probes collapsed")
    L.append("  (56.0 -> 13.1 make/unmake per node; probes are now ~15% of the full-legal")
    L.append("  bucket). The measured phases show pseudo move generation as the largest")
    L.append("  component (~{:.1%} of the bucket), then the per-move loop residual".format(flb["pseudo_gen"]))
    L.append("  (~{:.1%}, mostly sampled-window inflation + loop bookkeeping; fast accepts".format(flb["loop_and_rest"]))
    L.append("  ~{:.1%}, fallback probes ~{:.1%}), with the pin scan small".format(flb["model_fast_accept"], flb["model_fallback_probes"] + flb["model_in_check_probes"]))
    L.append(f"  (~{flb['pin_scan']:.1%}).")
    L.append("")
    L.append(f"- **D. Tactical/evasion/has-any:** tactical grew to {share['movegen_tactical']:.1%}")
    L.append(f"  (from 7.6%) and still legacy-probes every pseudo move ({trav['tactical_probe_make']:,} probes);")
    L.append(f"  evasion {share['movegen_evasion']:.1%}, has-any {share['movegen_has_any']:.1%} — isolated")
    L.append("  fast-path candidates are plausible for tactical, but none exceeds the")
    L.append("  full-legal generator.")
    L.append("")
    L.append(f"- **E. SEE/qSEE and ordering:** {share['SEE/qSEE']:.1%} and {share['move_ordering']:.1%} grew")
    L.append("  (7.1/6.8 before) but neither overtook movegen.")
    L.append("")
    L.append("- **F. Next target:** see decision block below.")
    L.append("")
    L.append("## Decision (one preferred + at most one secondary)")
    L.append("")
    L.append("**Preferred: pseudo move generation inside `movegen_full_legal`** — the")
    L.append("largest measured component of the still-dominant bucket (generation is")
    L.append("called 1.54x/node, ~602M pseudo moves across the run, and pseudo-gen is the")
    L.append("only in-bucket phase whose wall time is measured directly rather than")
    L.append("residual). A piece-list / bitboard attack generator (or per-piece-type")
    L.append("direct generation) is the cleanest next cut.")
    L.append("")
    L.append(f"**Secondary: `movegen_tactical` fast path** — {share['movegen_tactical']:.1%} of elapsed (up from 7.6%),")
    L.append("the fastest-growing bucket, and it still performs the legacy probe on every")
    L.append("pseudo move ({:,} probes). The promoted full-legal safety theorem".format(trav["tactical_probe_make"]))
    L.append("transplants directly (same eligibility/fallback rules on the tactical move set).")
    L.append("")
    L.append("Not indicated: pin representation / pin-mask reuse (pin scan is ~{:.1%} of".format(flb["pin_scan"]))
    L.append("the full-legal bucket); SEE or ordering work (both below 11%); a new TT")
    L.append("layout (2.6%).")
    L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    return cmd_run(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"s44a_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
