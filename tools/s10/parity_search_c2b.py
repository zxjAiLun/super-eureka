#!/usr/bin/env python3
"""S10-C2B search parity harness: FullRefresh vs Incremental NNUE profiles.

Runs both NNUE candidate profiles on a fixed FEN corpus with fixed
depth/node limits (never wall-clock) in the SAME release binary with the
SAME frozen quantized artifact, and demands EXACT equality of:
    bestmove, score, completed_depth, nodes, PV

Zero tolerance: any difference is an integration bug.

Outputs results/s10/s10-c2b-search-parity.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ENGINE = Path("target/release/eureka")
MODEL = Path("data/s10/b3/seed-20260818/nnue-v2-q01.bin")
ENGINE_SHA = None  # captured at runtime
MODEL_SHA = "b51a79b19999aeed974c2279eef60b01f890248c7d006cbe3d504cc7c0f28b9a"

PROFILES = {
    "full": "current-final-nnue-v2q-full",
    "incremental": "current-final-nnue-v2q",
}

# Mixed corpus: tactical / middlegame / endgame / startpos / castling /
# promotion / EP-rich / KQK mop-up / KRK mop-up / halfmove-claim context.
CORPUS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth", 6),
    ("middlegame", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "depth", 5),
    ("tactical", "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", "depth", 5),
    ("endgame-krp", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "depth", 6),
    ("kqk-mopup", "7k/8/8/8/8/8/8/KQ6 w - - 0 1", "depth", 6),
    ("krk-mopup", "7k/8/8/8/8/8/8/KR6 w - - 0 1", "depth", 6),
    ("promotion", "8/P7/8/8/8/8/k6K/8 w - - 0 1", "depth", 6),
    ("ep-rich", "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3", "depth", 5),
    ("castle-both", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "depth", 5),
    ("tactical-2", "r1bq1rk1/ppp2bpp/2np1n2/4p3/2B1P3/1PN1B3/P1PPQPPP/R3K2R w KQ - 0 10", "depth", 5),
    ("nullmove-heavy", "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12", "depth", 5),
    ("deep-endgame", "8/8/4k3/8/4p3/8/4K3/4R3 w - - 0 1", "depth", 7),
    # Node-limited runs (exercise abort/unwind + aspiration re-search paths).
    ("startpos-nodes", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "nodes", 5000),
    ("middlegame-nodes", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "nodes", 5000),
    ("abort-tiny-1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "nodes", 1),
    ("abort-tiny-7", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "nodes", 7),
    ("abort-100", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1", "nodes", 100),
    ("abort-1000", "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12", "nodes", 1000),
]

FIELDS = [
    "score", "bestmove", "completed_depth", "nodes", "qsearch_nodes", "pv",
    "null_move_attempts", "lmr_reductions", "aspiration_retries",
]


def run_fixture(profile: str, fen: str, kind: str, value: int,
                audit: bool = False) -> dict:
    limit = f"{kind}:{value}"
    cmd = [str(ENGINE), "bench", "profile", "--profile", profile,
           "--nnue-model", str(MODEL), "--fen", fen,
           "--" + kind, str(value)]
    if audit:
        cmd.append("--nnue-audit")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        return {"error": proc.stderr[-400:]}
    result = {}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result"):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    result[k] = v
        elif line.startswith("nnue_stack_result"):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    result["stack_" + k] = v
        elif line.startswith("nnue_audit_result"):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    result["audit_" + k] = v
    return result


def main() -> int:
    import hashlib
    engine_sha = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if model_sha != MODEL_SHA:
        print(f"FATAL: model SHA {model_sha} != frozen")
        return 4

    mismatches = []
    per_fixture = []
    for name, fen, kind, value in CORPUS:
        full = run_fixture(PROFILES["full"], fen, kind, value)
        inc = run_fixture(PROFILES["incremental"], fen, kind, value)
        entry = {"fixture": name.strip(), "limit": f"{kind}:{value}",
                 "full": {}, "incremental": {}, "match": True}
        if "error" in full or "error" in inc:
            entry["match"] = False
            entry["error"] = full.get("error", inc.get("error"))
            mismatches.append(name.strip())
            per_fixture.append(entry)
            continue
        for f in FIELDS:
            fv, iv = full.get(f), inc.get(f)
            entry["full"][f] = fv
            entry["incremental"][f] = iv
            if fv != iv:
                entry["match"] = False
        # S10-C2B Repair 1: stack lifecycle evidence — pushes must equal
        # pops (restore_root must NEVER have masked a leaked frame) and the
        # stack must end at depth 1.
        inc_pushes = int(inc.get("stack_pushes", 0))
        inc_pops = int(inc.get("stack_pops", 0))
        entry["lifecycle"] = {
            "pushes": inc_pushes, "pops": inc_pops,
            "null_pushes": int(inc.get("stack_null_pushes", 0)),
            "delta_updates": int(inc.get("stack_delta_updates", 0)),
            "full_refreshes": int(inc.get("stack_full_refreshes", 0)),
            "max_depth": int(inc.get("stack_max_depth", 0)),
            "balanced": inc_pushes == inc_pops and inc_pushes > 0,
        }
        if not entry["lifecycle"]["balanced"]:
            entry["match"] = False
        # FullRefresh arm must show ZERO stack maintenance.
        full_pushes = int(full.get("stack_pushes", 0))
        entry["full_stack_pushes"] = full_pushes
        if full_pushes != 0:
            entry["match"] = False
        if not entry["match"]:
            mismatches.append(name.strip())
        per_fixture.append(entry)

    # Path coverage: prove the corpus exercised the interesting search paths.
    coverage = {}
    for name, fen, kind, value in CORPUS:
        full = run_fixture(PROFILES["full"], fen, kind, value)
        for key in ("qsearch_nodes", "null_move_attempts", "lmr_reductions",
                    "aspiration_retries"):
            v = int(full.get(key, 0))
            coverage[key] = coverage.get(key, 0) + v

    result = {
        "stage": "s10_c2b_search_parity",
        "engine_sha256": engine_sha,
        "model_sha256": model_sha,
        "profile_full": PROFILES["full"],
        "profile_incremental": PROFILES["incremental"],
        "fixtures": len(CORPUS),
        "fields_compared": FIELDS,
        "mismatched_fixtures": mismatches,
        "mismatches": len(mismatches),
        "passed": len(mismatches) == 0,
        "path_coverage": coverage,
        "per_fixture": per_fixture,
    }
    print(json.dumps({k: v for k, v in result.items()
                      if k != "per_fixture"}, indent=2))
    out = Path("results/s10/s10-c2b-search-parity.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
