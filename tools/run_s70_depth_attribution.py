#!/usr/bin/env python3
"""S7.0 Depth Attribution runner.

OBSERVATION ONLY. Measures WHERE the search tree's effort goes, so depth
bottlenecks can be ranked (ordering / branching / qsearch / TT / LMR /
eval-instability) before any S7.1 implementation.

For every position in tools/data/s7_depth_attribution_corpus.jsonl it runs
fixed-depth cold searches at depth 4..8 (and 9 when depth-8 wall is under the
predeclared cap), each via `bench profile --profile current-final --depth d
--mode cold --fen <fen>`, and records the full profiling counter set plus
per-iteration derived metrics (iteration nodes, growth factor, first-move
cutoff %, searched-moves/node, qsearch %, bestmove/score stability).

Writes results/s7/s70-depth-attribution.json + .md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
OUT_DIR = REPO / "results/s7"
OUT_JSON = OUT_DIR / "s70-depth-attribution.json"
OUT_MD = OUT_DIR / "s70-depth-attribution.md"

PROFILE = "current-final"
HASH_MB = 16
DEPTHS = [4, 5, 6, 7, 8]
DEPTH9_WALL_CAP_S = 20.0  # run depth 9 only if depth-8 wall is under this

# The |teacher cp| swing that marks a "major reversal" in the stability table.
REVERSAL_CP = 200


def parse_bench_result(line: str) -> dict[str, str]:
    """Split a bench_result line into key=value, preserving the quoted PV."""
    pv = ""
    m = re.search(r'pv="([^"]*)"', line)
    if m:
        pv = m.group(1)
        line = line.replace(m.group(0), 'pv=""')
    d: dict[str, str] = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    d["pv"] = pv
    return d


def int_of(d: dict[str, str], key: str) -> int:
    v = d.get(key)
    if v is None:
        return 0
    try:
        return int(v)
    except ValueError:
        return 0


def run_depth(engine: Path, fen: str, depth: int) -> dict | None:
    cmd = [
        str(engine),
        "bench", "profile",
        "--profile", PROFILE,
        "--depth", str(depth),
        "--mode", "cold",
        "--fen", fen,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            return parse_bench_result(line)
    return None


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def p90(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(0.9 * (len(s) - 1)) + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    corpus_rows = [
        json.loads(line)
        for line in args.corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus_text = args.corpus.read_text(encoding="utf-8")
    corpus_sha = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    positions: list[dict] = []
    for pos in corpus_rows:
        per_depth: dict[str, dict] = {}
        for d in DEPTHS:
            print(f"s70 {pos['id']} depth={d}", flush=True)
            r = run_depth(engine, pos["fen"], d)
            if r is None:
                per_depth[str(d)] = {"error": "engine_failed"}
                continue
            per_depth[str(d)] = r
        # Optional depth 9, gated on depth-8 wall.
        d8 = per_depth.get("8", {})
        if d8.get("elapsed_ms"):
            if int(d8["elapsed_ms"]) / 1000.0 < DEPTH9_WALL_CAP_S:
                print(f"s70 {pos['id']} depth=9 (eligible)", flush=True)
                r9 = run_depth(engine, pos["fen"], 9)
                per_depth["9"] = r9 if r9 is not None else {"error": "engine_failed"}
            else:
                per_depth["9"] = {"skipped": "depth8_over_wall_cap"}
        positions.append({
            "id": pos["id"],
            "stratum": pos.get("stratum"),
            "class": pos.get("class"),
            "fen": pos["fen"],
            "per_depth": per_depth,
        })

    # ---- derived per-depth metrics ----
    for pos in positions:
        prev_nodes = 0
        for d in sorted(int(k) for k in pos["per_depth"] if k.isdigit()):
            rec = pos["per_depth"][str(d)]
            if "error" in rec or "skipped" in rec:
                continue
            nodes = int_of(rec, "nodes")
            rec["iteration_nodes"] = nodes - prev_nodes
            prev_nodes = nodes
            rec["qsearch_pct"] = (
                int_of(rec, "qsearch_nodes") / nodes * 100.0 if nodes else 0.0
            )
            beta = int_of(rec, "beta_cutoffs")
            rec["first_move_cutoff_pct"] = (
                int_of(rec, "beta_cutoff_idx_0") / beta * 100.0 if beta else 0.0
            )
            rec["searched_moves_per_node"] = (
                int_of(rec, "moves_searched") / nodes if nodes else 0.0
            )
            rec["seldepth_minus_depth"] = int_of(rec, "seldepth") - d
            rec["tt_hit_pct"] = (
                int_of(rec, "tt_hits") / int_of(rec, "tt_probes") * 100.0
                if int_of(rec, "tt_probes") else 0.0
            )
            rec["tt_cutoff_pct"] = (
                int_of(rec, "tt_cutoffs") / int_of(rec, "tt_probes") * 100.0
                if int_of(rec, "tt_probes") else 0.0
            )
            lmr_red = int_of(rec, "lmr_reductions")
            rec["lmr_research_pct"] = (
                int_of(rec, "lmr_researches") / lmr_red * 100.0 if lmr_red else 0.0
            )
            # bestmove change / score delta vs previous depth
            if str(d - 1) in pos["per_depth"]:
                prev = pos["per_depth"][str(d - 1)]
                if "bestmove" in prev and "bestmove" in rec:
                    rec["bestmove_changed"] = prev["bestmove"] != rec["bestmove"]

    # ---- aggregate per depth ----
    agg: dict[str, dict] = {}
    for d in DEPTHS:
        keys = ["nodes", "iteration_nodes", "qsearch_pct", "first_move_cutoff_pct",
                "searched_moves_per_node", "seldepth", "seldepth_minus_depth",
                "tt_hit_pct", "tt_cutoff_pct", "lmr_research_pct", "elapsed_ms",
                "nps", "moves_searched", "beta_cutoffs", "qsearch_nodes",
                "main_seldepth", "qsearch_seldepth"]
        vals = {k: [] for k in keys}
        changed = 0
        for pos in positions:
            rec = pos["per_depth"].get(str(d), {})
            if "error" in rec or "skipped" in rec:
                continue
            for k in keys:
                v = rec.get(k)
                if v is not None:
                    vals[k].append(float(v) if isinstance(v, (int, float)) else 0.0)
            if rec.get("bestmove_changed"):
                changed += 1
        agg[str(d)] = {
            "completed": len(vals["nodes"]),
            "median_nodes": median(vals["nodes"]),
            "median_iteration_nodes": median(vals["iteration_nodes"]),
            "median_growth": (
                median(vals["iteration_nodes"])
                # growth vs previous depth handled below
            ),
            "median_wall_ms": median(vals["elapsed_ms"]),
            "median_seldepth": median(vals["seldepth"]),
            "median_seldepth_minus_depth": median(vals["seldepth_minus_depth"]),
            "median_qsearch_pct": median(vals["qsearch_pct"]),
            "median_first_move_cutoff_pct": median(vals["first_move_cutoff_pct"]),
            "median_searched_moves_per_node": median(vals["searched_moves_per_node"]),
            "median_tt_hit_pct": median(vals["tt_hit_pct"]),
            "median_tt_cutoff_pct": median(vals["tt_cutoff_pct"]),
            "median_lmr_research_pct": median(vals["lmr_research_pct"]),
            "bestmove_changes": changed,
        }
    # growth = median iteration_nodes[d] / median iteration_nodes[d-1]
    prev_it = None
    for d in DEPTHS:
        it = agg[str(d)]["median_iteration_nodes"]
        if prev_it and prev_it > 0:
            agg[str(d)]["median_growth"] = it / prev_it
        else:
            agg[str(d)]["median_growth"] = 0.0
        prev_it = it

    data = {
        "contract": {
            "observation_source": "d71c3e7",  # release/telemetry-only base
            "chess_baseline": "Eureka v0.1.0",
            "profile": PROFILE,
            "hash_mb": HASH_MB,
            "threads": 1,
            "tt": "cold per depth",
            "depths": DEPTHS,
            "depth9_wall_cap_s": DEPTH9_WALL_CAP_S,
            "corpus_sha256": corpus_sha,
            "positions": len(positions),
        },
        "aggregate": agg,
        "per_position": positions,
    }
    OUT_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(data, OUT_MD)
    print(f"s70 wrote {out_dir}")
    return 0


def write_report(data: dict, out: Path) -> None:
    L: list[str] = []
    L.append("# S7.0 — Search Depth Attribution")
    L.append("")
    L.append("ATTRIBUTION ONLY — no optimization candidate.")
    L.append("")
    L.append("## Contract")
    L.append("")
    L.append("```text")
    L.append(f"observation source:  {data['contract']['observation_source']} (telemetry/build only)")
    L.append(f"chess baseline:       {data['contract']['chess_baseline']} (S3 + LegalityFast + SingleBuffer + SingleGeneration)")
    L.append(f"profile:              {data['contract']['profile']}")
    L.append(f"TT:                   {data['contract']['hash_mb']} MB cold per depth | threads {data['contract']['threads']}")
    L.append(f"depths:               {data['contract']['depths']}")
    L.append(f"depth-9 wall cap:     {data['contract']['depth9_wall_cap_s']} s (per position)")
    L.append(f"corpus SHA-256:       {data['contract']['corpus_sha256']}")
    L.append(f"positions:            {data['contract']['positions']}")
    L.append("```")
    L.append("")
    L.append("## Aggregate per depth")
    L.append("")
    L.append("| depth | completed | median iteration nodes | growth | median wall ms | median seldepth | qsearch % | 1st-move cutoff % | searched moves/node | TT hit % | TT cutoff % | LMR research % | bestmove changes |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in data["contract"]["depths"]:
        a = data["aggregate"][str(d)]
        L.append(
            f"| {d} | {a['completed']} | {a['median_iteration_nodes']:,.0f} | "
            f"{a['median_growth']:.2f} | {a['median_wall_ms']:,.0f} | "
            f"{a['median_seldepth']:.0f} | {a['median_qsearch_pct']:.1f} | "
            f"{a['median_first_move_cutoff_pct']:.1f} | "
            f"{a['median_searched_moves_per_node']:.2f} | "
            f"{a['median_tt_hit_pct']:.1f} | {a['median_tt_cutoff_pct']:.1f} | "
            f"{a['median_lmr_research_pct']:.1f} | {a['bestmove_changes']} |"
        )
    L.append("")
    L.append("## Diagnosis (top-3 depth bottlenecks)")
    L.append("")
    L.append("_Filled by S7.1 decision after reading the table above._")
    L.append("")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s70_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
