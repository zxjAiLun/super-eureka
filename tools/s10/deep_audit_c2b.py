#!/usr/bin/env python3
"""S10-C2B Repair 1: search-level deep accumulator audit.

Runs REAL recursive searches (main + qsearch + null move + LMR +
aspiration) with --nnue-audit on the incremental profile: every NNUE
static eval compares the search-stack accumulator against a fresh full
refresh (all 256 lanes + raw integer).

Gate per fixture:
    eval_calls > 0
    lane_mismatches == 0
    raw_mismatches == 0
    pushes == pops (and > 0)

Outputs results/s10/s10-c2b-deep-audit.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ENGINE = Path("target/release/eureka")
MODEL = Path("data/s10/b3/seed-20260818/nnue-v2-q01.bin")
MODEL_SHA = "b51a79b19999aeed974c2279eef60b01f890248c7d006cbe3d504cc7c0f28b9a"

# Representative subset covering all path families (verified: these
# collectively exercise qsearch, null move, LMR, aspiration retries).
CORPUS = [
    ("middlegame", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "depth", 5),
    ("nullmove-heavy", "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12", "depth", 5),
    ("tactical-2", "r1bq1rk1/ppp2bpp/2np1n2/4p3/2B1P3/1PN1B3/P1PPQPPP/R3K2R w KQ - 0 10", "depth", 5),
    ("startpos-depth", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth", 6),
    ("endgame-krp", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "depth", 6),
    ("ep-rich", "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3", "depth", 5),
    ("abort-1000", "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12", "nodes", 1000),
    ("abort-5000", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "nodes", 5000),
]


def run_audit(fen: str, kind: str, value: int) -> dict:
    proc = subprocess.run(
        [str(ENGINE), "bench", "profile",
         "--profile", "current-final-nnue-v2q",
         "--nnue-model", str(MODEL), "--nnue-audit",
         "--fen", fen, "--" + kind, str(value)],
        capture_output=True, text=True, timeout=1200,
        cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        return {"error": proc.stderr[-300:]}
    result = {}
    for line in proc.stdout.splitlines():
        if line.startswith("nnue_stack_result") or line.startswith(
                "nnue_audit_result"):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    result[k] = v
        elif line.startswith("bench_result"):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    result["search_" + k] = v
    return result


def main() -> int:
    engine_sha = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if model_sha != MODEL_SHA:
        print(f"FATAL: model SHA {model_sha} != frozen")
        return 4

    entries = []
    all_pass = True
    totals = {"eval_calls": 0, "lane_mismatches": 0, "raw_mismatches": 0,
              "pushes": 0, "pops": 0}
    coverage = {"qsearch": 0, "null_move": 0, "lmr": 0, "aspiration": 0}
    for name, fen, kind, value in CORPUS:
        r = run_audit(fen, kind, value)
        if "error" in r:
            entries.append({"fixture": name, "passed": False,
                            "error": r["error"]})
            all_pass = False
            continue
        evals = int(r.get("eval_calls", 0))
        lanes = int(r.get("lane_mismatches", 0))
        raws = int(r.get("raw_mismatches", 0))
        pushes = int(r.get("pushes", 0))
        pops = int(r.get("pops", 0))
        passed = (evals > 0 and lanes == 0 and raws == 0
                  and pushes == pops and pushes > 0)
        entry = {
            "fixture": name,
            "limit": f"{kind}:{value}",
            "eval_calls": evals,
            "lanes_checked": evals * 256,
            "lane_mismatches": lanes,
            "raw_mismatches": raws,
            "pushes": pushes,
            "pops": pops,
            "null_pushes": int(r.get("null_pushes", 0)),
            "delta_updates": int(r.get("delta_updates", 0)),
            "full_refreshes": int(r.get("full_refreshes", 0)),
            "max_depth": int(r.get("max_depth", 0)),
            "passed": passed,
        }
        entries.append(entry)
        if not passed:
            all_pass = False
        totals["eval_calls"] += evals
        totals["lane_mismatches"] += lanes
        totals["raw_mismatches"] += raws
        totals["pushes"] += pushes
        totals["pops"] += pops
        coverage["qsearch"] += int(r.get("search_qsearch_nodes", 0))
        coverage["null_move"] += int(r.get("search_null_move_attempts", 0))
        coverage["lmr"] += int(r.get("search_lmr_reductions", 0))
        coverage["aspiration"] += int(r.get("search_aspiration_retries", 0))

    result = {
        "stage": "s10_c2b_deep_audit",
        "engine_sha256": engine_sha,
        "model_sha256": model_sha,
        "fixtures": len(CORPUS),
        "totals": totals,
        "total_lanes_checked": totals["eval_calls"] * 256,
        "path_coverage": coverage,
        "gate": {
            "eval_calls_gt_0": totals["eval_calls"] > 0,
            "lane_mismatches_zero": totals["lane_mismatches"] == 0,
            "raw_mismatches_zero": totals["raw_mismatches"] == 0,
            "pushes_equal_pops": totals["pushes"] == totals["pops"],
            "coverage_qsearch": coverage["qsearch"] > 0,
            "coverage_null_move": coverage["null_move"] > 0,
            "coverage_lmr": coverage["lmr"] > 0,
            "coverage_aspiration": coverage["aspiration"] > 0,
        },
        "all_passed": all_pass,
        "per_fixture": entries,
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k != "per_fixture"}, indent=2))
    out = Path("results/s10/s10-c2b-deep-audit.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
