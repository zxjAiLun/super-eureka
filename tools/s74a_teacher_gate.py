#!/usr/bin/env python3
"""S7.4A tactical safety gate: S6 teacher challenge (178 positions).

Runs current-final (A) and current-final-lmr-null-window (B) at identical
fixed depth on the frozen S6 teacher corpus and compares teacher-bestmove
agreement, score divergence, and mate transitions.

Writes results/s7/s74a-teacher-gate.json (incremental, --resume).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data/s6/s6-teacher-challenge-v1.jsonl"
OUT = REPO / "results/s7/s74a-teacher-gate.json"
A = "current-final"
B = "current-final-lmr-null-window"


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
            return {"bestmove": toks.get("bestmove"),
                    "score": toks.get("score")}
    return {"error": True}


def cp_value(score: str | None) -> int | None:
    if not score or score == "none" or score.startswith("mate:"):
        return None
    try:
        return int(score[3:])
    except ValueError:
        return None


def mate_distance(score: str | None) -> int | None:
    if not score or not score.startswith("mate:"):
        return None
    try:
        return int(score[5:])
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--depths", default="6")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    depths = [int(d) for d in args.depths.split(",")]
    rows = [json.loads(l) for l in
            CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    results = {"rows": []}
    done = set()
    if args.resume and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r["fen"], r["depth"]) for r in results["rows"]}

    for d in depths:
        for i, pos in enumerate(rows):
            if (pos["fen"], d) in done:
                continue
            a = run(engine, A, pos["fen"], d)
            b = run(engine, B, pos["fen"], d)
            rec = {"i": i, "depth": d, "fen": pos["fen"],
                   "teacher_bestmove": pos["teacher_bestmove"],
                   "teacher_mate": pos.get("teacher_mate"),
                   "a": a, "b": b}
            if pos["teacher_bestmove"] in (None, "(none)"):
                rec["classification"] = "TERMINAL/NOT_APPLICABLE"
            elif (a or {}).get("error") or (a or {}).get("timeout") \
                    or (b or {}).get("error") or (b or {}).get("timeout"):
                rec["classification"] = "ENGINE_FAILURE"
            else:
                rec["classification"] = "OK"
            if a and b:
                rec["a_match"] = a.get("bestmove") == pos["teacher_bestmove"]
                rec["b_match"] = b.get("bestmove") == pos["teacher_bestmove"]
                ca, cb = cp_value(a.get("score")), cp_value(b.get("score"))
                if ca is not None and cb is not None:
                    rec["cp_delta"] = abs(ca - cb)
                ma, mb = mate_distance(a.get("score")), mate_distance(b.get("score"))
                rec["mate_transition"] = (ma is None) != (mb is None)
                if ma is not None and mb is not None and ma != mb:
                    rec["mate_distance_change"] = [ma, mb]
            results["rows"].append(rec)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
            print(f"s74a_teacher d{d} {i} a={rec.get('a_match')} "
                  f"b={rec.get('b_match')}", flush=True)

    # Aggregate per depth. A row is evaluable only if BOTH A and B produced a
    # real bench result. Rows whose teacher_bestmove is "(none)" are terminal
    # / no-move fixtures: label them TERMINAL/NOT_APPLICABLE instead of
    # counting them as engine failures.
    summaries = {}
    for d in depths:
        rs = [r for r in results["rows"] if r["depth"] == d
              and r.get("a") and r.get("b") and "error" not in (r["a"] or {})
              and "timeout" not in (r["a"] or {})
              and "error" not in (r["b"] or {}) and "timeout" not in (r["b"] or {})]
        terminal = [r for r in results["rows"] if r["depth"] == d
                    and r.get("teacher_bestmove") in (None, "(none)")]
        genuine_fail = [r for r in results["rows"] if r["depth"] == d
                        and r not in rs and r not in terminal]
        a_m = sum(1 for r in rs if r.get("a_match"))
        b_m = sum(1 for r in rs if r.get("b_match"))
        deltas = [r.get("cp_delta") for r in rs if r.get("cp_delta") is not None]
        summaries[f"depth{d}"] = {
            "positions": len(rs),
            "terminal_not_applicable": len(terminal),
            "genuine_engine_failures": len(genuine_fail),
            "mate_labelled_positions": sum(
                1 for r in rs
                if str(r.get("teacher_mate", "none")).lstrip("-").isdigit()
                and int(r["teacher_mate"]) != 0),
            "a_matches": a_m, "b_matches": b_m,
            "a_only": [r["i"] for r in rs if r.get("a_match") and not r.get("b_match")],
            "b_only": [r["i"] for r in rs if r.get("b_match") and not r.get("a_match")],
            "cp_div_ge_100": sum(1 for v in deltas if v >= 100),
            "cp_div_ge_300": sum(1 for v in deltas if v >= 300),
            "cp_div_ge_500": sum(1 for v in deltas if v >= 500),
            "mate_transitions": [r["i"] for r in rs if r.get("mate_transition")],
            "mate_distance_changes": [r for r in rs
                                      if r.get("mate_distance_change")],
        }
    results["summary"] = summaries
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"s74a_teacher summary: {json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in summaries.items()})}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s74a_teacher_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
