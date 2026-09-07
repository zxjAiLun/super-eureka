#!/usr/bin/env python3
"""S11-B3: paired search NPS benchmark — R12 hybrid (A) vs E3 V2 material (B).

Same binary, same machine, back-to-back arms per fixture/round (paired,
immune to session drift). Protocol copied from the C3-C harness:
  - 24-FEN balanced performance corpus (in-file, frozen)
  - fixed node budget 200,000 per fixture/arm/round
  - fresh cold 64 MiB TT per run (--hash-mb 64)
  - Threads = 1
  - 8 rounds per fixture, alternating ABBA/BAAB order per round
  - tree identity NOT required (different models => different evals);
    this harness measures NPS ratio only

Outputs results/s10/s11-b2b3-nps.json
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ENGINE = Path("target/release/eureka")
MODEL_B = Path(
    "data/s10/e3/scale-1m-win/seed-20260820/nnue-v2-q01-material-v3twin.bin")
MODEL_A = Path(
    "data/s10/s11/r2/seed-20260819/nnue-v2-q01-material-r12-v4.bin")
PROFILE_B = "current-final-nnue-v2q-material"
PROFILE_A = "current-final-nnue-v2q-material-r12"

NODES = 200_000
ROUNDS = 8
HASH_MB = 64

# The C3-C frozen 24-FEN performance corpus (identical list).
CORPUS = [
    # The S10-C3-C frozen 24-FEN performance corpus (identical list).
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/1b2bppp/p2ppn2/1p6/3NPP2/1BN1B3/PPPQ2PP/2KR3R w - - 0 12",
    "r1bq1rk1/ppp2bpp/2np1n2/4p3/2B1P3/1PN1B3/P1PPQPPP/R3K2R w KQ - 0 10",
    "r4rk1/pp1nqppp/2nb1n2/3pp3/3PP3/2NBBN2/PPP2PPP/R2Q1RK1 w - - 0 11",
    "3r1rk1/ppqbbppp/1p3n2/4p3/3P4/1P2PN2/P1PNBPPP/2RQ1RK1 w - - 0 15",
    "2rr2k1/pb1nqpp1/1p2pb1p/8/2PN4/1P4P1/PB2QPBP/2RR2K1 w - - 0 20",
    "r1bq1r1k/pp3ppp/2n1p3/3pP3/3P4/2PB1N2/PP1N1PPP/R1BQ1RK1 w - - 0 14",
    "r2q1rk1/1b2bppp/p2p1n2/1p1pp3/3PP3/1BN1BN2/PPP2PPP/R2Q1RK1 w - - 0 12",
    "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
    "r1b2rk1/pp2qppp/2n5/3p4/3P1B2/2N5/PPP2PPP/R2Q1RK1 b - - 0 12",
    "2r3k1/pR2p3/2pq1p2/5p2/5P2/2Q5/P4P1P/6K1 w - - 0 1",
    "1rb4r/pkPp3p/1b1P3n/1Q6/N3Pp2/8/P1P3PP/7K w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/8/4k3/8/4p3/8/4K3/4R3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/8/1p4k1/5p2/5P2/6K1/8/4R3 w - - 0 1",
    "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
    "7k/8/8/8/8/8/8/KR6 w - - 0 1",
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
]


def run_arm(fen: str, profile: str, model: Path) -> dict:
    cmd = [
        str(ENGINE), "bench", "profile",
        "--nodes", str(NODES),
        "--profile", profile,
        "--nnue-model", str(model),
        "--hash-mb", str(HASH_MB),
        "--fen", fen,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"FATAL arm run: {proc.stderr[:400]}")
    # last bench_result line
    line = None
    for l in proc.stdout.splitlines():
        if l.startswith("bench_result"):
            line = l
    if line is None:
        raise SystemExit(f"FATAL: no bench_result for {fen[:30]} {profile}")
    fields = dict(
        re.findall(r"(\w+)=([^ ]+)", line))
    return {
        "nodes": int(fields["nodes"]),
        "elapsed_us": int(fields["elapsed_us"]),
        "nps": int(fields["nps"]),
        "eval_calls": int(fields["eval_calls"]),
        "qsearch_nodes": int(fields["qsearch_nodes"]),
        "score": fields.get("score", "?"),
        "bestmove": fields.get("bestmove", "?"),
    }


def main() -> int:
    per_fixture = []
    ratios = []
    for i, fen in enumerate(CORPUS):
        nps_a, nps_b = [], []
        for rnd in range(ROUNDS):
            # alternate order per round (AB then BA) to cancel drift
            order = [("A", PROFILE_A, MODEL_A), ("B", PROFILE_B, MODEL_B)]
            if rnd % 2:
                order = order[::-1]
            for arm, prof, model in order:
                r = run_arm(fen, prof, model)
                if arm == "A":
                    nps_a.append(r["nps"])
                else:
                    nps_b.append(r["nps"])
        med_a = statistics.median(nps_a)
        med_b = statistics.median(nps_b)
        ratio = med_a / med_b
        ratios.append(ratio)
        per_fixture.append({
            "fen": fen,
            "nps_a_median": med_a,
            "nps_b_median": med_b,
            "ratio": round(ratio, 4),
        })
        print(f"[{i+1:2}/24] A={med_a:9.0f} B={med_b:9.0f} "
              f"ratio={ratio:.3f}  {fen[:40]}", flush=True)

    overall_median = statistics.median(ratios)
    report = {
        "schema_version": 1,
        "stage": "s11_b3_paired_nps",
        "nodes": NODES,
        "rounds": ROUNDS,
        "hash_mb": HASH_MB,
        "engine": str(ENGINE),
        "arm_a": {
            "profile": PROFILE_A,
            "model": str(MODEL_A),
        },
        "arm_b": {
            "profile": PROFILE_B,
            "model": str(MODEL_B),
        },
        "per_fixture": per_fixture,
        "ratio_median": round(overall_median, 4),
        "ratio_min": round(min(ratios), 4),
        "ratio_max": round(max(ratios), 4),
        "ratio_mean": round(sum(ratios) / len(ratios), 4),
        "gate": {
            "nps_ge_090": overall_median >= 0.90,
            "nps_ge_080": overall_median >= 0.80,
        },
    }
    out = Path("results/s10/s11-b2b3-nps.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("ratio_median", "ratio_min", "ratio_max",
                       "ratio_mean", "gate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
