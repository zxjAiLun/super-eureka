#!/usr/bin/env python3
"""S7.4A fixed-depth node gate: current-final vs current-final-lmr-null-window.

TREE-CHANGING gate (reduced searches on null-window nodes). Primary metric:
total-node reduction.

Corpus: 80 unique S7 positions (S4 30 are a subset and are NOT double-run).
Depths: 6, 7 (pass --depths). d8 subset uses --d8-subset (first 20 positions).

Predeclared mechanism gate:
  <= -15% nodes  STRONG_MECHANISM
  -8%..-15%      USEFUL_MECHANISM
  > -8%          TOO_SMALL -> REJECT/CLOSE

Writes results/s7/s74a-node-gate.json (incremental, --resume supported).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
OUT = REPO / "results/s7/s74a-node-gate.json"
A = "current-final"
B = "current-final-lmr-null-window"

FIELDS = ("nodes", "qsearch_nodes", "elapsed_us", "nps", "seldepth",
          "s74_lmr_proposed", "s74_lmr_applied_existing_pvs",
          "s74_lmr_suppressed_by_null_window", "s74_lmr_applied_null_window",
          "s74_lmr_nw_fail_low", "s74_lmr_nw_research",
          "s74_lmr_nw_verified_cutoff")


def run(engine: Path, profile: str, fen: str, depth: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            rec = {k: toks.get(k) for k in FIELDS}
            rec.update({"score": toks.get("score"),
                        "bestmove": toks.get("bestmove"),
                        "pv": toks.get("pv")})
            return rec
    return {"error": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--depths", default="6")
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N positions (subset runs)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    depths = [int(d) for d in args.depths.split(",")]

    results = {"rows": []}
    done = set()
    if args.resume and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r["id"], r["depth"]) for r in results["rows"]}

    rows7 = [json.loads(l) for l in
             CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows7 = rows7[:args.limit]

    for d in depths:
        for pos in rows7:
            key = (pos["id"], d)
            if key in done:
                continue
            a = run(engine, A, pos["fen"], d)
            b = run(engine, B, pos["fen"], d)
            rec = {"id": pos["id"], "depth": d, "fen": pos["fen"],
                   "baseline": a, "candidate": b}
            if a and b and a.get("nodes") and b.get("nodes"):
                try:
                    na, nb = int(a["nodes"]), int(b["nodes"])
                    rec["node_reduction_pct"] = round((na - nb) * 100.0 / na, 3)
                except (TypeError, ValueError):
                    pass
            results["rows"].append(rec)
            print(f"s74a_gate {pos['id']} d{d} "
                  f"nodes {a.get('nodes') if a else '?'} -> "
                  f"{b.get('nodes') if b else '?'} "
                  f"({rec.get('node_reduction_pct', '?')}%)", flush=True)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")

    ok = [r for r in results["rows"] if r.get("node_reduction_pct") is not None]
    if ok:
        by_depth: dict[int, list] = {}
        for r in ok:
            by_depth.setdefault(r["depth"], []).append(r)
        summaries = {}
        for d, rows_d in sorted(by_depth.items()):
            na = sum(int(r["baseline"]["nodes"]) for r in rows_d)
            nb = sum(int(r["candidate"]["nodes"]) for r in rows_d)
            red = (na - nb) * 100.0 / na
            prop = sum(int(r["candidate"].get("s74_lmr_proposed") or 0)
                       for r in rows_d)
            appl = sum(int(r["candidate"].get("s74_lmr_applied_null_window") or 0)
                       for r in rows_d)
            verdict = ("STRONG_MECHANISM" if red >= 15.0 else
                       "USEFUL_MECHANISM" if red >= 8.0 else
                       "TOO_SMALL -> REJECT/CLOSE")
            summaries[f"depth{d}"] = {
                "completed_rows": len(rows_d), "total_nodes_baseline": na,
                "total_nodes_candidate": nb,
                "total_node_reduction_pct": round(red, 3),
                "lmr_proposed": prop, "lmr_applied_null_window": appl,
                "predeclared_verdict": verdict,
            }
        results["summary_by_depth"] = summaries
        na = sum(int(r["baseline"]["nodes"]) for r in ok)
        nb = sum(int(r["candidate"]["nodes"]) for r in ok)
        red = (na - nb) * 100.0 / na
        results["summary"] = {
            "completed_rows": len(ok), "total_nodes_baseline": na,
            "total_nodes_candidate": nb,
            "total_node_reduction_pct": round(red, 3),
        }
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                       + "\n", encoding="utf-8")
        print(f"s74a_gate summary_by_depth: "
              f"{json.dumps(summaries)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s74a_gate_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
