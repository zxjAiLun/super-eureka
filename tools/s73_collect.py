#!/usr/bin/env python3
"""S7.3 selectivity attribution collector (OBSERVATION ONLY).

Runs `bench profile --profile current-final` over the 80-position S7 corpus
(SHA 8786ffca...) at depth 6 and 7, parses the s73_* counters emitted after
the s72 block, and aggregates them into results/s7/s73-selectivity-attribution.json.

Aggregate = 80 unique S7 corpus only (the S4 30 positions are a subset of A1
and are NOT double-counted here; performance summaries stay in S7.0/S7.2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
OUT = REPO / "results/s7/s73-selectivity-attribution.json"


def parse_bench_line(line: str) -> dict:
    """Flat token=value parse; expands `key:sub=v,sub=v` groups to key_sub."""
    toks: dict[str, str] = {}
    for tok in line.split():
        if "=" not in tok:
            continue
        key, val = tok.split("=", 1)
        if ":" in key:
            # `base:sub0=v0,sub1=v1,...`: the first subkey rides inside the
            # token key, the rest ride inside the token value.
            base, first = key.split(":", 1)
            for sub in f"{first}={val}".split(","):
                if "=" in sub:
                    sk, sv = sub.split("=", 1)
                    toks[f"{base}_{sk}"] = sv
        else:
            toks[key] = val
    return toks


def repair_stats(stats: dict) -> dict:
    """In-place repair of stats dicts saved by the buggy parser (pre-reparse):
    entries like `base:sub0` -> `v0,sub1=v1,...` expand to `base_sub0=v0` ..."""
    out: dict = {}
    for k, v in stats.items():
        if ":" in k:
            base, first = k.split(":", 1)
            for sub in f"{first}={v}".split(","):
                if "=" in sub:
                    sk, sv = sub.split("=", 1)
                    out[f"{base}_{sk}"] = sv
        else:
            out[k] = v
    return out


def run(engine: Path, fen: str, depth: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", "current-final",
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = parse_bench_line(line)
            out = {k: v for k, v in toks.items() if k.startswith("s73_")}
            out["nodes"] = toks.get("nodes")
            out["null_move_attempts"] = toks.get("null_move_attempts")
            out["null_move_fail_highs"] = toks.get("null_move_fail_highs")
            out["futility_pruned"] = toks.get("futility_pruned")
            out["lmr_reductions"] = toks.get("lmr_reductions")
            out["lmr_researches"] = toks.get("lmr_researches")
            out["s72_fail_low_nodes"] = toks.get("s72_fail_low_nodes")
            out["s72_fail_low_searched_sum"] = toks.get(
                "s72_fail_low_searched_sum")
            return out
    return {"error": True}


def merge_totals(totals: dict, rec: dict | None) -> None:
    if not rec or rec.get("timeout") or rec.get("error"):
        return
    for k, v in rec.items():
        if k in ("timeout", "error"):
            continue
        try:
            totals[k] = totals.get(k, 0) + int(v)
        except (TypeError, ValueError):
            pass


def pct(a: float, b: float) -> float | None:
    return round(a * 100.0 / b, 3) if b else None


def summarize(t: dict) -> dict:
    s: dict = {}
    loop = t.get("s73_loop_nodes", 0)
    nocut = t.get("s73_nocut_pv", 0) + t.get("s73_nocut_nonpv", 0)
    s["move_loop_nodes"] = loop
    s["no_cutoff_loop_nodes"] = nocut
    s["no_cutoff_share_of_loops_pct"] = pct(nocut, loop)
    s["no_cutoff_pv"] = t.get("s73_nocut_pv", 0)
    s["no_cutoff_nonpv"] = t.get("s73_nocut_nonpv", 0)
    s["no_cutoff_incheck"] = t.get("s73_nocut_incheck", 0)
    s["no_cutoff_null_attempted"] = t.get("s73_nocut_null_attempted", 0)
    s["no_cutoff_mean_moves_searched"] = (
        round(t.get("s73_nocut_searched_sum", 0) / nocut, 3) if nocut else None)
    hist_keys = ["le0", "h1_15", "h16_63", "h64_255", "h256p", "unused"]
    s["no_cutoff_searched_quiet_by_history"] = {
        k: t.get(f"s73_nocut_searched_hist_{k}", 0) for k in hist_keys}
    s["null_eligible_nodes"] = t.get("s73_null_eligible", 0)
    s["fut_quiet_kept_at_eligible_nodes"] = t.get("s73_fut_quiet_kept", 0)
    q4 = t.get("s73_q4p_quiet_searched", 0)
    s["q4p_quiet_searched"] = q4
    idx_keys = ["i0", "i1", "i2_3", "i4_7", "i8p"]
    red_keys = ["r0", "r1", "r2"]
    s["q4p_quiet_by_idx"] = {
        k: t.get(f"s73_q4p_quiet_idx_{k}", 0) for k in idx_keys}
    s["q4p_quiet_by_reduction"] = {
        k: t.get(f"s73_q4p_quiet_red_{k}", 0) for k in red_keys}
    s["q4p_quiet_red_idx"] = {
        f"{r}_{i}": t.get(f"s73_q4p_quiet_red_idx_{r}_{i}", 0)
        for r in red_keys for i in idx_keys}
    s["q4p_quiet_cutoff_by_reduction"] = {
        k: t.get(f"s73_q4p_quiet_cutoff_red_{k}", 0) for k in red_keys}
    s["q4p_quiet_cutoff_by_idx"] = {
        k: t.get(f"s73_q4p_quiet_cutoff_idx_{k}", 0) for k in idx_keys}
    s["q4p_scout_faillow_by_reduction"] = {
        k: t.get(f"s73_q4p_scout_faillow_red_{k}", 0) for k in red_keys}
    s["q4p_quiet_researched"] = t.get("s73_q4p_quiet_researched", 0)
    s["q4p_full_depth_share_pct"] = pct(t.get("s73_q4p_quiet_red_r0", 0), q4)
    s["q4p_r1_share_pct"] = pct(t.get("s73_q4p_quiet_red_r1", 0), q4)
    s["q4p_r2_share_pct"] = pct(t.get("s73_q4p_quiet_red_r2", 0), q4)
    s["q4p_late_idx_8p_share_pct"] = pct(t.get("s73_q4p_quiet_idx_i8p", 0), q4)
    s["q4p_late_idx_8p_full_depth"] = t.get("s73_q4p_quiet_red_idx_r0_i8p", 0)
    s["q4p_late_idx_8p_full_depth_share_of_i8p_pct"] = pct(
        t.get("s73_q4p_quiet_red_idx_r0_i8p", 0),
        t.get("s73_q4p_quiet_idx_i8p", 0))
    s["q4p_research_rate_pct"] = pct(t.get("s73_q4p_quiet_researched", 0), q4)
    s["ctx_null_move_attempts"] = t.get("null_move_attempts", 0)
    s["ctx_null_move_fail_highs"] = t.get("null_move_fail_highs", 0)
    s["ctx_futility_pruned"] = t.get("futility_pruned", 0)
    s["ctx_lmr_reductions"] = t.get("lmr_reductions", 0)
    s["ctx_lmr_researches"] = t.get("lmr_researches", 0)
    s["ctx_s72_no_cutoff_nodes"] = t.get("s72_fail_low_nodes", 0)
    s["ctx_s72_no_cutoff_searched_sum"] = t.get(
        "s72_fail_low_searched_sum", 0)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--depths", default="6,7")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reparse", action="store_true",
                    help="repair sub-key groups already saved in OUT and "
                         "recompute totals/aggregate without running the "
                         "engine (the raw strings are lossless)")
    args = ap.parse_args(sys.argv[1:])

    if args.reparse:
        if not OUT.exists():
            print("s73_collect_error --reparse requires existing output",
                  file=sys.stderr)
            return 1
        results = json.loads(OUT.read_text(encoding="utf-8"))
        for rec in results["per_position"]:
            if rec.get("stats"):
                rec["stats"] = repair_stats(rec["stats"])
        for depth, block in results["by_depth"].items():
            totals: dict = {}
            for rec in results["per_position"]:
                if rec["depth"] == int(depth):
                    merge_totals(totals, rec.get("stats"))
            block["totals"] = totals
            block["summary"] = summarize(totals)
        combined: dict = {}
        for block in results["by_depth"].values():
            merge_totals(combined, block["totals"])
        results["aggregate"] = summarize(combined)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                       + "\n", encoding="utf-8")
        print("s73_collect reparse wrote " + str(OUT), flush=True)
        return 0

    if args.engine is None:
        ap.error("--engine is required unless --reparse is given")
    engine = args.engine.resolve()
    depths = [int(d) for d in args.depths.split(",")]
    rows = [json.loads(l) for l in
            CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    results: dict = {"per_position": [], "by_depth": {}}
    done = set()
    if args.resume and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r["id"], r["depth"]) for r in results["per_position"]}

    for depth in depths:
        totals: dict = {}
        for pos in rows:
            if (pos["id"], depth) in done:
                rec_prev = next(r for r in results["per_position"]
                                if r["id"] == pos["id"]
                                and r["depth"] == depth)
                merge_totals(totals, rec_prev.get("stats"))
                continue
            stats = run(engine, pos["fen"], depth)
            rec = {"id": pos["id"], "depth": depth, "stats": stats}
            results["per_position"].append(rec)
            merge_totals(totals, stats)
            ok = bool(stats and "error" not in stats and "timeout" not in stats)
            print(f"s73_collect {pos['id']} d{depth} ok={ok}", flush=True)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
        results["by_depth"][str(depth)] = {"totals": totals,
                                           "summary": summarize(totals)}
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                       + "\n", encoding="utf-8")
        print(f"s73_collect depth {depth} done", flush=True)

    combined: dict = {}
    for d in results["by_depth"].values():
        merge_totals(combined, d["totals"])
    results["aggregate"] = summarize(combined)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("s73_collect wrote " + str(OUT), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s73_collect_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
