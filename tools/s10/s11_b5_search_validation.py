#!/usr/bin/env python3
"""S11-B5: R12 vs E3 search-quality validation (256 roots x 100k fixed nodes).

Frozen protocol (S11-B5, approved):
  A = R12 incremental  (current-final-nnue-v2q-material-r12-inc + R12 v4 seed20260819)
  B = E3 canonical     (current-final-nnue-v2q-material       + E3 v3twin)
  same binary, Threads=1 (single process), Hash 32 MiB, 100,000 nodes per root,
  fresh process per (root, arm) — absolute TT isolation, no cross-root leakage.

Scoring: frozen H0-E teacher semantics (j0_lockbox._teacher):
  teacher(move) = -sf.cp of the CHILD (child's stm is the opponent), mate -> +-2000,
  clamped to +-2000. regret = teacher_best_sibling - teacher(selected_move).
  Selected move must be a legal sibling (fail-closed integrity count).

Gate (frozen): R12 mean regret <= 64.6 AND acc@20 >= 66.2% AND p90 <= 100.
Anti-drift: the E3 arm is RERUN on this same harness this round; the historical
E3 (69.6 / 67.2 / 93) is reported alongside, and a >3cp / >1pp drift is a harness
flag, not an R12 result.

Outputs results/s10/s11-b5-search-validation.json
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ENGINE = Path(r"target\release\eureka.exe")
MODEL_A = Path(
    r"data\s10\s11\r2\seed-20260819\nnue-v2-q01-material-r12-v4.bin")
MODEL_B = Path(
    r"data\s10\e3\scale-1m-win\seed-20260820\nnue-v2-q01-material-v3twin.bin")
PROFILE_A = "current-final-nnue-v2q-material-r12-inc"
PROFILE_B = "current-final-nnue-v2q-material"

NODES = 100_000
HASH_MB = 32
CLIP = 2000

H0C_CACHE = Path(
    r"C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache")
OUT = Path("results/s10/s11-b5-search-validation.json")

# Historical E3 search reference (H0-E, frozen in
# results/s10/s10-h0-e-sibling-ranking.json — DIFFERENT harness; drift-flagged).
E3_HISTORICAL = {
    "mean_regret": 69.6,
    "acc20": 67.2,
    "p90_regret": 93,
    "top1": 53.1,
    "acc50": 81.6,
}

GATE = {
    "mean_regret_max": 64.6,
    "acc20_min": 66.2,
    "p90_max": 100.0,
}
DRIFT_FLAG = {"mean_regret_cp": 3.0, "acc20_pp": 1.0}


def sha_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_corpus():
    parents = [json.loads(l) for l in
               (H0C_CACHE / "e_search.jsonl").read_text(
                   encoding="utf-8").splitlines() if l.strip()]
    return parents


def teacher(sib) -> float:
    s = sib["sf"]
    if s.get("cp") is not None:
        return -s["cp"]
    if s.get("mate") is not None:
        return -CLIP if s["mate"] > 0 else CLIP
    raise SystemExit(f"missing teacher for {sib}")


def clamp(v: float) -> float:
    return max(-CLIP, min(CLIP, v))


def search_root(fen: str, profile: str, model: Path) -> dict:
    """One fresh-process 100k-node search; returns parsed bench_result."""
    cmd = [str(ENGINE), "bench", "profile",
           "--nodes", str(NODES),
           "--hash-mb", str(HASH_MB),
           "--profile", profile,
           "--nnue-model", str(model),
           "--fen", fen]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[:300]}
    line = None
    for l in proc.stdout.splitlines():
        if l.startswith("bench_result"):
            line = l
    if line is None:
        return {"error": "no bench_result line"}
    fields = {}
    for token in line.split():
        if "=" in token:
            k, _, v = token.partition("=")
            fields[k] = v
    return fields


def evaluate_arm(parents, profile: str, model: Path, arm: str):
    per_parent = []
    failures = 0
    missing_teacher_moves = 0
    for i, p in enumerate(parents):
        fen = p["fen"]
        uci_by_move = {s["uci"]: s for s in p["siblings"]}
        res = search_root(fen, profile, model)
        if "error" in res:
            failures += 1
            per_parent.append({
                "root_id": p["root_id"], "phase": p["phase"],
                "error": res["error"]})
            print(f"  [{arm}] root {i+1}/256 FAILURE: {res['error'][:80]}",
                  flush=True)
            continue
        bm = res.get("bestmove", "")
        if bm not in uci_by_move:
            # engine picked a move not among labeled siblings (should be
            # impossible — siblings are ALL legal moves; integrity flag)
            missing_teacher_moves += 1
            per_parent.append({
                "root_id": p["root_id"], "phase": p["phase"],
                "error": f"bestmove {bm} not in siblings"})
            print(f"  [{arm}] root {i+1}/256 bestmove-not-in-siblings: {bm}",
                  flush=True)
            continue
        vals = {u: clamp(teacher(s)) for u, s in uci_by_move.items()}
        best = max(vals.values())
        regret = best - vals[bm]
        per_parent.append({
            "root_id": p["root_id"], "phase": p["phase"],
            "bestmove": bm, "regret": regret,
            "nodes": int(res.get("nodes", "0")),
            "score": res.get("score", "?"),
        })
        if (i + 1) % 32 == 0:
            done = [r["regret"] for r in per_parent if "regret" in r]
            print(f"  [{arm}] {i+1}/256 roots, running mean regret "
                  f"{statistics.mean(done):.1f}", flush=True)
    ok = [r for r in per_parent if "regret" in r]
    regrets = sorted(r["regret"] for r in ok)
    n = len(regrets)
    acc20 = 100.0 * sum(1 for r in regrets if r <= 20) / n
    acc50 = 100.0 * sum(1 for r in regrets if r <= 50) / n
    top1 = 100.0 * sum(
        1 for r in ok if r["regret"] == 0) / n

    def pct(p):
        return regrets[min(n - 1, int(round(p / 100.0 * (n - 1))))]

    # phase table
    phases = {}
    for ph in ("high", "mid", "low", "zero"):
        sel = [r["regret"] for r in ok if r["phase"] == ph]
        if sel:
            phases[ph] = {
                "n": len(sel),
                "mean_regret": round(statistics.mean(sel), 1),
                "acc20": round(
                    100.0 * sum(1 for r in sel if r <= 20) / len(sel), 1),
            }
    worst10 = sorted(ok, key=lambda r: -r["regret"])[:10]
    return {
        "arm": arm,
        "profile": profile,
        "model": str(model),
        "n_roots_ok": n,
        "failures": failures,
        "missing_teacher_moves": missing_teacher_moves,
        "mean_regret": round(statistics.mean(regrets), 1),
        "median_regret": regrets[n // 2],
        "p90_regret": pct(90),
        "acc20": round(acc20, 1),
        "acc50": round(acc50, 1),
        "top1": round(top1, 1),
        "regret_gt500": sum(1 for r in regrets if r > 500),
        "regret_gt1000": sum(1 for r in regrets if r > 1000),
        "phase_table": phases,
        "worst10": [{"root_id": r["root_id"], "phase": r["phase"],
                     "regret": r["regret"], "bestmove": r["bestmove"]}
                    for r in worst10],
    }


def main() -> int:
    parents = load_corpus()
    assert len(parents) == 256, f"expected 256 parents, got {len(parents)}"
    for p in parents:
        assert p["siblings"], f"parent {p['root_id']} has no siblings"

    identity = {
        "engine": str(ENGINE),
        "engine_sha256": sha_of(ENGINE),
        "model_a": str(MODEL_A),
        "model_a_sha256": sha_of(MODEL_A),
        "model_b": str(MODEL_B),
        "model_b_sha256": sha_of(MODEL_B),
        "profile_a": PROFILE_A,
        "profile_b": PROFILE_B,
        "nodes": NODES,
        "hash_mb": HASH_MB,
        "process_isolation": "fresh process per (root, arm)",
    }

    print("B arm (E3 canonical) — same-harness rerun ...", flush=True)
    e3 = evaluate_arm(parents, PROFILE_B, MODEL_B, "E3")
    print(json.dumps({k: e3[k] for k in (
        "n_roots_ok", "failures", "mean_regret", "p90_regret",
        "acc20", "acc50", "top1")}), flush=True)

    print("A arm (R12 incremental) ...", flush=True)
    r12 = evaluate_arm(parents, PROFILE_A, MODEL_A, "R12")
    print(json.dumps({k: r12[k] for k in (
        "n_roots_ok", "failures", "mean_regret", "p90_regret",
        "acc20", "acc50", "top1")}), flush=True)

    # anti-drift check
    drift = {
        "mean_regret_delta": round(
            e3["mean_regret"] - E3_HISTORICAL["mean_regret"], 1),
        "acc20_delta_pp": round(
            e3["acc20"] - E3_HISTORICAL["acc20"], 1),
        "within_flag": (
            abs(e3["mean_regret"] - E3_HISTORICAL["mean_regret"])
            <= DRIFT_FLAG["mean_regret_cp"]
            and abs(e3["acc20"] - E3_HISTORICAL["acc20"])
            <= DRIFT_FLAG["acc20_pp"]),
    }

    gate_pass = (r12["mean_regret"] <= GATE["mean_regret_max"]
                 and r12["acc20"] >= GATE["acc20_min"]
                 and r12["p90_regret"] <= GATE["p90_max"])
    strong = (r12["acc20"] >= 69.2
              or r12["mean_regret"] <= 59.6)

    report = {
        "schema_version": 1,
        "stage": "s11_b5_search_validation",
        "corpus": {
            "parents": 256,
            "corpus_source": "H0-E frozen e_search.jsonl (256 parents / "
                             "6142 siblings / SF18-32k-constrained labels)",
            "note": "0 new SF labels; frozen corpus, ordered",
        },
        "identity": identity,
        "results": {
            "E3_current_run": e3,
            "R12_incremental": r12,
            "E3_historical_reference": E3_HISTORICAL,
        },
        "delta": {
            "mean_regret": round(
                e3["mean_regret"] - r12["mean_regret"], 1),
            "median_regret": e3["median_regret"] - r12["median_regret"],
            "p90": e3["p90_regret"] - r12["p90_regret"],
            "acc20_pp": round(r12["acc20"] - e3["acc20"], 1),
            "acc50_pp": round(r12["acc50"] - e3["acc50"], 1),
            "top1_pp": round(r12["top1"] - e3["top1"], 1),
            "gt500": e3["regret_gt500"] - r12["regret_gt500"],
            "gt1000": e3["regret_gt1000"] - r12["regret_gt1000"],
        },
        "anti_drift": drift,
        "gate": {
            **GATE,
            "pass": gate_pass,
            "strong_pass": strong,
        },
        "integrity": {
            "roots": f"{r12['n_roots_ok']}+{e3['n_roots_ok']} / 256x2",
            "nodes_requested": NODES,
            "missing_teacher_moves": (
                r12["missing_teacher_moves"]
                + e3["missing_teacher_moves"]),
            "search_failures": r12["failures"] + e3["failures"],
        },
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate": report["gate"], "delta": report["delta"],
        "anti_drift": report["anti_drift"],
        "integrity": report["integrity"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
