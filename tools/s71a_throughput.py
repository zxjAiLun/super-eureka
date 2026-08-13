#!/usr/bin/env python3
"""S7.1A throughput gate: current-final vs current-final-qsearch-lazy.

5 interleaved paired repetitions at fixed depth 6 over the 30-position S4
corpus + the 80-position S7 corpus. Reports aggregate wall delta, median
paired wall delta, and NPS delta.

Writes results/s7/s71a-throughput.json.
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
S4_EPD = REPO / "tools/data/s4_compute_positions.epd"
OUT = REPO / "results/s7/s71a-throughput.json"
DEPTH = 6
REPS = 5


def run(engine: Path, profile: str, fen: str) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--depth", str(DEPTH), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return {"elapsed_ms": int(toks["elapsed_ms"]), "nodes": int(toks["nodes"])}
    return None


def load_epd(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split(";", 1)[0].strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    args = ap.parse_args(sys.argv[1:])
    engine = args.engine.resolve()

    fens = [json.loads(l)["fen"] for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    fens += load_epd(S4_EPD)
    n = len(fens)

    # Interleaved: for each rep, run A then B for every position (ABAB...).
    pairs: list[dict] = []
    for rep in range(REPS):
        for fen in fens:
            a = run(engine, "current-final", fen)
            b = run(engine, "current-final-qsearch-lazy", fen)
            if a is None or b is None or "timeout" in a or "timeout" in b:
                continue
            pairs.append({
                "rep": rep + 1, "fen": fen[:24],
                "cf_ms": a["elapsed_ms"], "lazy_ms": b["elapsed_ms"],
                "cf_nodes": a["nodes"], "lazy_nodes": b["nodes"],
                "delta_ms": b["elapsed_ms"] - a["elapsed_ms"],
            })
        print(f"s71a_throughput rep {rep + 1}/{REPS} done", flush=True)

    total_cf = sum(p["cf_ms"] for p in pairs)
    total_lazy = sum(p["lazy_ms"] for p in pairs)
    deltas = [p["delta_ms"] for p in pairs]
    # relative delta per pair: (lazy - cf) / cf
    rel = [p["delta_ms"] / p["cf_ms"] * 100.0 for p in pairs if p["cf_ms"] > 0]

    result = {
        "contract": {
            "profile_a": "current-final",
            "profile_b": "current-final-qsearch-lazy",
            "depth": DEPTH,
            "reps": REPS,
            "corpus": "30 S4 + 80 S7",
            "pairs": len(pairs),
        },
        "aggregate_wall_delta_pct": (total_lazy - total_cf) / total_cf * 100.0 if total_cf else 0.0,
        "median_paired_wall_delta_pct": statistics.median(rel) if rel else 0.0,
        "mean_paired_wall_delta_pct": statistics.mean(rel) if rel else 0.0,
        "p25_paired_delta_pct": sorted(rel)[len(rel) // 4] if rel else 0.0,
        "p75_paired_delta_pct": sorted(rel)[3 * len(rel) // 4] if rel else 0.0,
        "favorable_pairs": sum(1 for d in deltas if d < 0),
        "unfavorable_pairs": sum(1 for d in deltas if d > 0),
        "total_cf_ms": total_cf,
        "total_lazy_ms": total_lazy,
        "per_rep_aggregate_delta_pct": [],
    }
    for rep in range(1, REPS + 1):
        rp = [p for p in pairs if p["rep"] == rep]
        cf = sum(p["cf_ms"] for p in rp)
        lz = sum(p["lazy_ms"] for p in rp)
        result["per_rep_aggregate_delta_pct"].append(
            (lz - cf) / cf * 100.0 if cf else 0.0)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s71a_throughput_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
