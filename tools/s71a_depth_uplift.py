#!/usr/bin/env python3
"""S7.1A depth-uplift (descriptive): fixed wall 1000ms / 3000ms on the S7
corpus. Reports completed depth / seldepth / nodes for current-final vs
current-final-qsearch-lazy. Descriptive only - no pass/fail threshold.

Writes results/s7/s71a-depth-uplift.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
OUT = REPO / "results/s7/s71a-depth-uplift.json"
WALLS = [1000, 3000]


def run(engine: Path, profile: str, fen: str, movetime_ms: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--movetime", str(movetime_ms), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return {
                "completed_depth": int(toks["completed_depth"]),
                "nodes": int(toks["nodes"]),
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    args = ap.parse_args(sys.argv[1:])
    engine = args.engine.resolve()

    fens = [json.loads(l)["fen"] for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    result = {"per_wall": {}}
    for wall in WALLS:
        cf_depths, lz_depths = [], []
        cf_nodes, lz_nodes = [], []
        for fen in fens:
            a = run(engine, "current-final", fen, wall)
            b = run(engine, "current-final-qsearch-lazy", fen, wall)
            if a and b:
                cf_depths.append(a["completed_depth"])
                lz_depths.append(b["completed_depth"])
                cf_nodes.append(a["nodes"])
                lz_nodes.append(b["nodes"])
        result["per_wall"][str(wall)] = {
            "positions": len(cf_depths),
            "cf_median_depth": statistics.median(cf_depths) if cf_depths else 0,
            "lazy_median_depth": statistics.median(lz_depths) if lz_depths else 0,
            "cf_median_nodes": statistics.median(cf_nodes) if cf_nodes else 0,
            "lazy_median_nodes": statistics.median(lz_nodes) if lz_nodes else 0,
            "cf_sum_nodes": sum(cf_nodes),
            "lazy_sum_nodes": sum(lz_nodes),
        }
        print(f"s71a_depth_uplift wall={wall}ms cf_depth={result['per_wall'][str(wall)]['cf_median_depth']} "
              f"lazy_depth={result['per_wall'][str(wall)]['lazy_median_depth']}", flush=True)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s71a_depth_uplift_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
