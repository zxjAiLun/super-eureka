#!/usr/bin/env python3
"""S7.1B fixed-depth node gate: current-final vs current-final-qsearch-delta.

TREE-CHANGING gate: the tree is EXPECTED to differ. Primary mechanism metric
is total-node reduction; the tactical-safety gates live elsewhere.

Corpus: 30 S4 compute positions + 80 S7 depth-attribution positions.
Default depth 6 (pass --depths 6,7 to also run S7 depth 7).

Predeclared interpretation of total-node reduction:
  >= 20%  STRONG_MECHANISM
  10-20% USEFUL_MECHANISM
  <  10% TOO_SMALL -> REJECT S7.1B (do not Arena-test a tree-changing
                      candidate for a micro gain)

Writes results/s7/s71b-node-gate.json (incremental, --resume supported).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
S4_EPD = REPO / "tools/data/s4_compute_positions.epd"
OUT = REPO / "results/s7/s71b-node-gate.json"

FIELDS = ("nodes", "qsearch_nodes", "delta_tests", "delta_pruned",
          "delta_pruned_pawn", "delta_pruned_minor", "delta_pruned_rook",
          "delta_pruned_queen", "delta_qply_0_1", "delta_qply_2_3",
          "delta_qply_4p", "qsearch_see_tests", "qsearch_see_pruned",
          "qsearch_moves_searched", "elapsed_us", "nps", "seldepth")


def run(engine: Path, profile: str, fen: str, depth: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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


def load_s4() -> list[tuple[str, str]]:
    out = []
    for line in S4_EPD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen = line.split(";", 1)[0].strip()
        out.append((f"s4_{len(out):02d}", fen))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--depths", default="6")
    ap.add_argument("--skip-s4", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    depths = [int(d) for d in args.depths.split(",")]

    results = {"rows": []}
    done = set()
    if args.resume and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r["id"], r["depth"]) for r in results["rows"]}

    jobs: list[tuple[str, str, int]] = []
    rows7 = [json.loads(l) for l in
             CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for d in depths:
        for pos in rows7:
            jobs.append((pos["id"], pos["fen"], d))
    if not args.skip_s4 and 6 in depths:
        for pid, fen in load_s4():
            jobs.append((pid, fen, 6))

    for entry_id, fen, depth in jobs:
        key = (entry_id, depth)
        if key in done:
            continue
        a = run(engine, "current-final", fen, depth)
        b = run(engine, "current-final-qsearch-delta", fen, depth)
        rec = {"id": entry_id, "depth": depth, "fen": fen,
               "baseline": a, "candidate": b}
        if a and b and a.get("nodes") and b.get("nodes"):
            try:
                na, nb = int(a["nodes"]), int(b["nodes"])
                rec["node_reduction_pct"] = round((na - nb) * 100.0 / na, 3)
                qa, qb = int(a["qsearch_nodes"]), int(b["qsearch_nodes"])
                rec["qsearch_node_reduction_pct"] = (
                    round((qa - qb) * 100.0 / qa, 3) if qa else None)
            except (TypeError, ValueError):
                pass
        results["rows"].append(rec)
        print(f"s71b_node_gate {entry_id} d{depth} "
              f"nodes {a.get('nodes') if a else '?'} -> "
              f"{b.get('nodes') if b else '?'} "
              f"({rec.get('node_reduction_pct', '?')}%)", flush=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    # Aggregate summary over completed rows.
    ok = [r for r in results["rows"] if r.get("node_reduction_pct") is not None]
    if ok:
        na = sum(int(r["baseline"]["nodes"]) for r in ok)
        nb = sum(int(r["candidate"]["nodes"]) for r in ok)
        dtests = sum(int(r["candidate"].get("delta_tests") or 0) for r in ok)
        dpruned = sum(int(r["candidate"].get("delta_pruned") or 0) for r in ok)
        reduction = (na - nb) * 100.0 / na
        if reduction >= 20.0:
            verdict = "STRONG_MECHANISM"
        elif reduction >= 10.0:
            verdict = "USEFUL_MECHANISM"
        else:
            verdict = "TOO_SMALL -> REJECT S7.1B"
        results["summary"] = {
            "completed_rows": len(ok), "total_nodes_baseline": na,
            "total_nodes_candidate": nb,
            "total_node_reduction_pct": round(reduction, 3),
            "delta_pruned_over_delta_tests": (
                round(dpruned * 100.0 / dtests, 3) if dtests else None),
            "delta_tests": dtests, "delta_pruned": dpruned,
            "predeclared_verdict": verdict,
        }
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                       + "\n", encoding="utf-8")
        print(f"s71b_node_gate summary: {json.dumps(results['summary'])}",
              flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s71b_node_gate_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
