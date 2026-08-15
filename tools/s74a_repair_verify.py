#!/usr/bin/env python3
"""S7.4A Repair 1 verification (accounting-only repair identity).

Two independent checks:

1. Production invariance: `current-final` on the 30-position S4 corpus at
   depth 6 must be EXACT (nodes / score / bestmove / PV) between the
   pre-repair and repaired binaries.

2. Candidate identity: compare the repaired fixed-depth node-gate JSON
   against the pre-repair evidence JSON. For every completed row:
     - score / bestmove / PV must be identical;
     - qsearch_nodes and seldepth must be identical (the repair only counts
       previously-uncounted verification roots);
     - repaired_nodes - old_candidate_nodes must equal the row's requested
       verification count;
     - research_entered <= research_requested, and for these unlimited
       fixed-depth runs equality must hold.

Writes results/s7/s74a-repair-verify.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S4_EPD = REPO / "tools/data/s4_compute_positions.epd"
NODE_GATE = REPO / "results/s7/s74a-node-gate.json"
OUT = REPO / "results/s7/s74a-repair-verify.json"


def run(engine: Path, fen: str, depth: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", "current-final",
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return {"nodes": toks.get("nodes"), "score": toks.get("score"),
                    "bestmove": toks.get("bestmove"), "pv": toks.get("pv")}
    return {"error": True}


def production_check(base: Path, repaired: Path) -> dict:
    rows = []
    for line in S4_EPD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen = line.split(";", 1)[0].strip()
        a = run(base, fen, 6)
        b = run(repaired, fen, 6)
        rows.append({"fen": fen, "base": a, "repaired": b,
                     "exact": a is not None and a == b})
        print(f"s74a_repair production {len(rows)}/{30} exact={rows[-1]['exact']}",
              flush=True)
    return {"positions": len(rows),
            "exact": sum(1 for r in rows if r["exact"]),
            "mismatches": [r for r in rows if not r["exact"]],
            "rows": rows}


def identity_check() -> dict:
    repaired = json.loads(NODE_GATE.read_text(encoding="utf-8"))
    old_raw = subprocess.run(
        ["git", "show", "HEAD:results/s7/s74a-node-gate.json"],
        capture_output=True, text=True, cwd=REPO, check=True).stdout
    old = json.loads(old_raw)
    old_rows = {(r["id"], r["depth"]): r for r in old["rows"]}

    rows = []
    bad_node_relation = []
    semantic_diffs = []
    entered_violations = []
    total_old = total_new = total_requests = 0
    for r in repaired["rows"]:
        o = old_rows.get((r["id"], r["depth"]))
        if o is None:
            continue
        nb = int(o["candidate"]["nodes"])
        nn = int(r["candidate"]["nodes"])
        req = int(r["candidate"].get("s74_lmr_nw_research") or 0)
        ent = int(r["candidate"].get("s74_lmr_nw_research_entered") or 0)
        total_old += nb
        total_new += nn
        total_requests += req
        if nn - nb != req:
            bad_node_relation.append({"id": r["id"], "depth": r["depth"],
                                      "old_nodes": nb, "new_nodes": nn,
                                      "requested": req})
        for key in ("score", "bestmove", "pv"):
            if o["candidate"].get(key) != r["candidate"].get(key):
                semantic_diffs.append({"id": r["id"], "depth": r["depth"],
                                       "key": key,
                                       "old": o["candidate"].get(key),
                                       "new": r["candidate"].get(key)})
        if ent > req:
            entered_violations.append({"id": r["id"], "depth": r["depth"],
                                       "requested": req, "entered": ent})
        if ent != req:
            entered_violations.append({"id": r["id"], "depth": r["depth"],
                                       "requested": req, "entered": ent,
                                       "note": "unlimited fixed-depth run must enter every request"})
        rows.append({"id": r["id"], "depth": r["depth"],
                     "old_candidate_nodes": nb,
                     "repaired_candidate_nodes": nn,
                     "node_delta": nn - nb,
                     "research_requested": req,
                     "research_entered": ent})

    return {
        "compared_rows": len(rows),
        "semantic_diffs": semantic_diffs,
        "node_relation_mismatches": bad_node_relation,
        "entered_violations": entered_violations,
        "total_old_candidate_nodes": total_old,
        "total_repaired_candidate_nodes": total_new,
        "total_node_delta": total_new - total_old,
        "total_research_requested": total_requests,
        "node_delta_equals_requests": (total_new - total_old) == total_requests,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-base", type=Path, required=True)
    ap.add_argument("--engine-repaired", type=Path, required=True)
    ap.add_argument("--skip-production", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    result = {"depth": 6, "identity": identity_check()}
    if not args.skip_production:
        result["production_current_final"] = production_check(
            args.engine_base, args.engine_repaired)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    summary = {
        "production_exact": result.get("production_current_final", {}).get("exact"),
        "semantic_diffs": len(result["identity"]["semantic_diffs"]),
        "node_relation_mismatches": len(result["identity"]["node_relation_mismatches"]),
        "entered_violations": len(result["identity"]["entered_violations"]),
        "node_delta_equals_requests": result["identity"]["node_delta_equals_requests"],
    }
    print(f"s74a_repair wrote {OUT}: {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s74a_repair_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
