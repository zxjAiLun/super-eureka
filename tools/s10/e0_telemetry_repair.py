"""S10-E0b Repair 1: diagnostic-intervention telemetry for arms B/C.

Re-runs the SAME 400 validation positions / seed 2026083004 / depth 7 for
arms B (NNUE + full CurrentFinal search) and C (NNUE with null move +
futility disabled), now harvesting the bench harness's own selectivity
counters from the `bench_result` line.

Evidence gates (frozen in Repair 1):
  * B actually exercised null move and futility at scale;
  * disabling them materially changed the search workload;
  * B/C bestmove agreement stayed high anyway -> the search's
    eval-dependent selectivity is NOT what sank the candidate.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_tactical_audit import load_validation, run_search  # noqa: E402

TELEM_FIELDS = [
    "nodes",
    "null_move_attempts",
    "null_move_fail_highs",
    "futility_pruned",
    "futility_considered",
    "lmr_reductions",
    "qsearch_nodes",
]


def parse_telemetry(proc_stdout: str) -> dict:
    out = {}
    for line in proc_stdout.splitlines():
        if line.startswith("bench_result"):
            for pair in line.split()[1:]:
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k in TELEM_FIELDS:
                        try:
                            out[k] = int(v)
                        except ValueError:
                            pass
            break
    return out


def run_with_telemetry(
    engine: Path, fen: str, depth: int, profile: str,
    nnue_model: Path | None, diag: list[str],
) -> dict:
    import subprocess

    argv = [
        str(engine), "bench", "profile",
        "--fen", fen, "--profile", profile, "--depth", str(depth),
    ]
    if nnue_model is not None:
        argv += ["--nnue-model", str(nnue_model)]
    for d in diag:
        argv += ["--diag", d]
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(f"bench rc={proc.returncode}: {proc.stderr[:200]}")
    telem = parse_telemetry(proc.stdout)
    # final bestmove from bench_result
    bm = None
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result"):
            for pair in line.split()[1:]:
                if pair.startswith("bestmove="):
                    bm = pair.split("=", 1)[1]
            break
    telem["bestmove"] = bm
    return telem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nnue-model", type=Path, required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026083004)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pool = load_validation(args.dataset)
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))

    arms = {
        "B": dict(
            profile="current-final-nnue-v2q", nnue=args.nnue_model, diag=[]
        ),
        "C": dict(
            profile="current-final-nnue-v2q", nnue=args.nnue_model,
            diag=["no-null", "no-futility"],
        ),
    }

    def one(pos: dict) -> dict:
        out = {"fen": pos["fen"]}
        for arm, cfg in arms.items():
            try:
                out[arm] = run_with_telemetry(
                    args.engine, pos["fen"], args.depth,
                    cfg["profile"], cfg["nnue"], cfg["diag"],
                )
            except RuntimeError as exc:
                out[arm] = {"error": str(exc)}
        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(one, sample)):
            results.append(res)
            if (i + 1) % 100 == 0:
                print(f"  position {i + 1}/{len(sample)}", flush=True)

    def totals(arm):
        rows = [r[arm] for r in results if "error" not in r.get(arm, {})]
        agg = {
            f: sum(r.get(f, 0) for r in rows) for f in TELEM_FIELDS
        }
        agg["positions_with_null"] = sum(
            1 for r in rows if r.get("null_move_attempts", 0) > 0
        )
        agg["positions_with_futility"] = sum(
            1 for r in rows if r.get("futility_pruned", 0) > 0
        )
        agg["n_ok"] = len(rows)
        return agg

    b_tot = totals("B")
    c_tot = totals("C")

    # paired comparisons
    pairs = []
    for r in results:
        b, c = r.get("B", {}), r.get("C", {})
        if "error" in b or "error" in c:
            continue
        pairs.append((b, c))
    agree = sum(1 for b, c in pairs if b["bestmove"] == c["bestmove"])
    node_ratios = sorted(
        c.get("nodes", 1) / max(1, b.get("nodes", 1)) for b, c in pairs
    )
    node_changed = sum(
        1 for b, c in pairs if b.get("nodes", 0) != c.get("nodes", 0)
    )

    def q(v, p):
        if not v:
            return None
        idx = p * (len(v) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(v) - 1)
        frac = idx - lo
        return round(v[lo] + (v[hi] - v[lo]) * frac, 3)

    summary = {
        "n_positions": len(sample),
        "depth": args.depth,
        "seed": args.seed,
        "B_totals": b_tot,
        "C_totals": c_tot,
        "paired": {
            "n": len(pairs),
            "bestmove_agreement": agree,
            "bestmove_agreement_rate": round(
                agree / max(1, len(pairs)), 4
            ),
            "nodes_changed": node_changed,
            "nodes_ratio_C_over_B": {
                "median": q(node_ratios, 0.5),
                "p25": q(node_ratios, 0.25),
                "p75": q(node_ratios, 0.75),
            },
        },
        "gates": {
            "null_exercised_at_scale": (
                b_tot["null_move_attempts"] > 1000
                and b_tot["positions_with_null"] > 0.5 * b_tot["n_ok"]
            ),
            "futility_exercised_at_scale": (
                b_tot["futility_pruned"] > 1000
                and b_tot["positions_with_futility"]
                > 0.5 * b_tot["n_ok"]
            ),
            "workload_materially_changed": (
                node_changed > 0.5 * len(pairs)
                and (q(node_ratios, 0.5) or 0) > 1.05
            ),
        },
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
