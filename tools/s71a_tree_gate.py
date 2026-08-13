#!/usr/bin/env python3
"""S7.1A exact-tree gate: current-final vs current-final-qsearch-lazy.

Requires IDENTICAL nodes / score / bestmove / PV for every completed
position/depth. Any mismatch = REJECT.

Writes results/s7/s71a-tree-gate.json.
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
OUT = REPO / "results/s7/s71a-tree-gate.json"
DEPTHS = [4, 5, 6, 7, 8]
S4_DEPTH = 6


def run(engine: Path, profile: str, fen: str, depth: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return {"nodes": toks["nodes"], "score": toks["score"],
                    "bestmove": toks["bestmove"], "pv": toks["pv"]}
    return {"error": True}


def load_epd(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen = line.split(";", 1)[0].strip()
        out.append((fen, fen))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    rows = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    s4 = load_epd(S4_EPD)

    results = {"s7": [], "s4": [], "mismatches": []}
    done = set()
    if args.resume and OUT.exists():
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        results = prev
        done = {(r["id"], str(r["depth"])) for r in results["s7"] + results["s4"]}

    def gate_entry(entry_id, fen, depth, basin):
        key = (entry_id, str(depth))
        if key in done:
            return
        a = run(engine, "current-final", fen, depth)
        b = run(engine, "current-final-qsearch-lazy", fen, depth)
        rec = {"id": entry_id, "depth": depth, "fen": fen[:24], "basin": basin,
               "cf": a, "lazy": b,
               "match": a is not None and b is not None
               and a.get("nodes") == b.get("nodes")
               and a.get("score") == b.get("score")
               and a.get("bestmove") == b.get("bestmove")
               and a.get("pv") == b.get("pv")}
        basin_list = results["s4"] if basin == "s4" else results["s7"]
        basin_list.append(rec)
        if not rec["match"]:
            results["mismatches"].append(rec)
        print(f"s71a_gate {entry_id} d{depth} {'MATCH' if rec['match'] else 'MISMATCH'}",
              flush=True)

    for pos in rows:
        for d in DEPTHS:
            gate_entry(pos["id"], pos["fen"], d, "s7")
    for i, (fen, _) in enumerate(s4):
        gate_entry(f"s4_{i:02d}", fen, S4_DEPTH, "s4")

    results["pass"] = len(results["mismatches"]) == 0
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"s71a_tree_gate done: pass={results['pass']} mismatches={len(results['mismatches'])}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s71a_tree_gate_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
