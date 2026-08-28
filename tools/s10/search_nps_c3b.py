#!/usr/bin/env python3
"""S10-C3-B: three-arm search NPS gate.

Arms:
    A = current-final            (production Eval2)
    B = current-final-nnue-v2q-full
    C = current-final-nnue-v2q   (incremental)

Protocol (frozen):
    24-FEN balanced performance corpus (in-file, hence in Git)
    fixed node budget 200,000 per fixture/arm/round
    fresh cold 64 MiB TT per run (--hash-mb 64)
    Threads = 1 (single process, sequential)
    diagnostics OFF (no --nnue-audit / --nnue-stack-telemetry)
    12 rounds per fixture in a 6-order Latin rotation x2:
        ABC ACB BAC BCA CAB CBA (repeated)
    paired per-fixture-round ratios B/A, C/A, C/B; median primary
    B/C tree identity (nodes/qnodes/score/bestmove/PV) fail-closed per run

Outputs results/s10/s10-c3b-search-nps.json
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ENGINE = Path("target/release/eureka")
MODEL = Path("data/s10/b3/seed-20260818/nnue-v2-q01.bin")
MODEL_SHA = "b51a79b19999aeed974c2279eef60b01f890248c7d006cbe3d504cc7c0f28b9a"

NODES = 200_000
ROUNDS = 12
HASH_MB = 64

# 24-FEN balanced performance corpus:
#   4 opening / early middlegame
#   8 middlegame
#   4 tactical / high branching
#   4 endgame
#   2 sparse king-heavy
#   2 special EP/castling/promotion
CORPUS = [
    # opening / early middlegame
    ("opening-1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("opening-2", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("opening-3", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    ("opening-4", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    # middlegame
    ("mid-1", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1"),
    ("mid-2", "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12"),
    ("mid-3", "r1bq1rk1/ppp2bpp/2np1n2/4p3/2B1P3/1PN1B3/P1PPQPPP/R3K2R w KQ - 0 10"),
    ("mid-4", "r4rk1/pp1nqppp/2nb1n2/3pp3/3PP3/2NBBN2/PPP2PPP/R2Q1RK1 w - - 0 11"),
    ("mid-5", "3r1rk1/ppqbbppp/1p3n2/4p3/3P4/1P2PN2/P1PNBPPP/2RQ1RK1 w - - 0 15"),
    ("mid-6", "2rr2k1/pb1nqpp1/1p2pb1p/8/2PN4/1P4P1/PB2QPBP/2RR2K1 w - - 0 20"),
    ("mid-7", "r1bq1r1k/pp3ppp/2n1p3/3pP3/3P4/2PB1N2/PP1N1PPP/R1BQ1RK1 w - - 0 14"),
    ("mid-8", "r2q1rk1/1b2bppp/p2p1n2/1p1pp3/3PP3/1BN1BN2/PPP2PPP/R2Q1RK1 w - - 0 12"),
    # tactical / high branching
    ("tactical-1", "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1"),
    ("tactical-2", "r1b2rk1/pp2qppp/2n5/3p4/3P1B2/2N5/PPP2PPP/R2Q1RK1 b - - 0 12"),
    ("tactical-3", "2r3k1/pR2p3/2pq1p2/5p2/5P2/2Q5/P4P1P/6K1 w - - 0 1"),
    ("tactical-4", "1rb4r/pkPp3p/1b1P3n/1Q6/N3Pp2/8/P1P3PP/7K w - - 0 1"),
    # endgame
    ("endgame-1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("endgame-2", "8/8/4k3/8/4p3/8/4K3/4R3 w - - 0 1"),
    ("endgame-3", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
    ("endgame-4", "8/8/1p4k1/5p2/5P2/6K1/8/4R3 w - - 0 1"),
    # sparse king-heavy
    ("king-heavy-1", "7k/8/8/8/8/8/8/KQ6 w - - 0 1"),
    ("king-heavy-2", "7k/8/8/8/8/8/8/KR6 w - - 0 1"),
    # special EP / castling / promotion
    ("special-ep", "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3"),
    ("special-castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
]

PROFILE = {"A": "current-final",
           "B": "current-final-nnue-v2q-full",
           "C": "current-final-nnue-v2q"}

ORDERS = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"] * (ROUNDS // 6)

IDENTITY_FIELDS = ["nodes", "qsearch_nodes", "score", "bestmove", "pv"]


def run_arm(arm: str, fen: str) -> dict:
    cmd = [str(ENGINE), "bench", "profile",
           "--profile", PROFILE[arm],
           "--nodes", str(NODES),
           "--hash-mb", str(HASH_MB),
           "--fen", fen]
    if arm in ("B", "C"):
        cmd += ["--nnue-model", str(MODEL)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=1200,
                          cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        return {"error": proc.stderr[-300:]}
    result = {}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result"):
            for kv in line.split()[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    result[k] = v
    return result


def main() -> int:
    engine_sha = hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if model_sha != MODEL_SHA:
        print(f"FATAL: model SHA {model_sha} != frozen")
        return 4

    corpus_sha = hashlib.sha256(
        "\n".join(f for _, f in CORPUS).encode()).hexdigest()

    ratios = {"B/A": [], "C/A": [], "C/B": []}
    arm_nps = {"A": [], "B": [], "C": []}
    identity_failures = []
    per_fixture = []
    identity_checked = 0

    for name, fen in CORPUS:
        entry = {"fixture": name, "rounds": []}
        for round_idx in range(ROUNDS):
            order = ORDERS[round_idx % len(ORDERS)]
            results = {}
            for arm in order:
                r = run_arm(arm, fen)
                if "error" in r:
                    print(f"FATAL: {name}/{arm} round {round_idx}: "
                          f"{r['error']}")
                    return 4
                results[arm] = r
            # B/C tree identity (fail-closed)
            b, c = results["B"], results["C"]
            identical = all(b.get(f) == c.get(f) for f in IDENTITY_FIELDS)
            identity_checked += 1
            if not identical:
                diff = {f: (b.get(f), c.get(f)) for f in IDENTITY_FIELDS
                        if b.get(f) != c.get(f)}
                identity_failures.append(
                    {"fixture": name, "round": round_idx, "diff": diff})
                continue
            # node budget must be consumed exactly (fixed-budget contract)
            for arm in "ABC":
                if results[arm].get("nodes") != str(NODES):
                    identity_failures.append({
                        "fixture": name, "round": round_idx,
                        "diff": {arm: results[arm].get("nodes")}})
                    continue
            nps = {}
            for arm in "ABC":
                elapsed_us = int(results[arm]["elapsed_us"])
                v = int(results[arm]["nodes"]) * 1_000_000 / max(elapsed_us, 1)
                nps[arm] = v
                arm_nps[arm].append(v)
            ratios["B/A"].append(nps["B"] / nps["A"])
            ratios["C/A"].append(nps["C"] / nps["A"])
            ratios["C/B"].append(nps["C"] / nps["B"])
            entry["rounds"].append({
                "order": order,
                "nps": {a: round(nps[a], 1) for a in "ABC"},
            })
        per_fixture.append(entry)

    def stats(v):
        def pct(q):
            s = sorted(v)
            idx = round((len(s) - 1) * q / 100.0)
            return s[idx]
        return {
            "median": round(statistics.median(v), 4),
            "p25": round(pct(25), 4),
            "p75": round(pct(75), 4),
            "min": round(min(v), 4),
            "max": round(max(v), 4),
            "n": len(v),
        }

    headline = {k: stats(v) for k, v in ratios.items()}
    raw = {a: stats(v) for a, v in arm_nps.items()}

    # Frozen decision gate (C3-B review):
    #   C/A >= 0.95        -> Arena directly
    #   0.85 <= C/A < 0.95 -> Arena unless a cheap win exists
    #   C/A < 0.85         -> bounded runtime optimization first
    #   C/B > 1 expected (incremental must beat full refresh)
    ca = headline["C/A"]["median"]
    cb = headline["C/B"]["median"]
    decision = (
        "arena_direct" if ca >= 0.95 else
        "arena_with_optional_optimization" if ca >= 0.85 else
        "bounded_optimization_first"
    )

    result = {
        "stage": "s10_c3b_search_nps",
        "engine_sha256": engine_sha,
        "model_sha256": model_sha,
        "corpus_sha256": corpus_sha,
        "fen_count": len(CORPUS),
        "node_budget": NODES,
        "rounds": ROUNDS,
        "order_schedule": ORDERS,
        "tt": {"hash_mb": HASH_MB, "fresh_per_run": True, "threads": 1},
        "diagnostics": False,
        "headline_paired_ratios": headline,
        "raw_nps": raw,
        "identity_checked": identity_checked,
        "identity_failures": identity_failures,
        "decision_gate": {
            "C_over_A_median": ca,
            "C_over_B_median": cb,
            "decision": decision,
            "rules": {
                "C/A >= 0.95": "arena_direct",
                "0.85 <= C/A < 0.95": "arena_with_optional_optimization",
                "C/A < 0.85": "bounded_optimization_first",
                "C/B <= 1.0": "anomaly: incremental slower than full refresh",
            },
        },
        "per_fixture": per_fixture,
    }
    passed = (len(identity_failures) == 0)
    result["passed"] = passed
    print(json.dumps({k: v for k, v in result.items()
                      if k != "per_fixture"}, indent=2))
    Path("results/s10/s10-c3b-search-nps.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
