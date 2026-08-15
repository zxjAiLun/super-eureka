#!/usr/bin/env python3
"""S7.4A Repair 1: R2 tactical/horizon safety gate (depth 8).

The d6 teacher gate cannot exercise the candidate's R2 regime (R2 requires
non-root remaining depth >= 7, i.e. root depth >= 8). This gate runs the
frozen tactical corpus at depth 8 where tractable; mate-labelled rows MUST
complete (long timeout, no skip).

Hard reject: any position where baseline finds the teacher-labelled mate
(correct side) and candidate loses it. The hard-reject scan is reported
explicitly; the final call stays with the human reviewer.

Writes results/s7/s74a-r2-gate.json (incremental, --resume).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data/s7/s74a-r2-tactical-corpus.jsonl"
OUT = REPO / "results/s7/s74a-r2-gate.json"
A = "current-final"
B = "current-final-lmr-null-window"


def run(engine: Path, profile: str, fen: str, depth: int,
        mate_required: bool) -> dict | None:
    timeout = 7200 if mate_required else 1800
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": True, "mate_required": mate_required}
    if proc.returncode != 0:
        return {"error": True, "mate_required": mate_required}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return {"bestmove": toks.get("bestmove"),
                    "score": toks.get("score")}
    return {"error": True, "mate_required": mate_required}


def cp_value(score):
    if not score or score == "none" or score.startswith("mate:"):
        return None
    try:
        return int(score[3:])
    except ValueError:
        return None


def mate_signed(score):
    """Return signed mate distance (positive = side to move mates)."""
    if not score or not score.startswith("mate:"):
        return None
    try:
        return int(score[5:])
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    rows = [json.loads(l) for l in
            CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    results = {"rows": [], "depth": args.depth}
    done = set()
    if args.resume and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r["fen"], r["depth"]) for r in results["rows"]}

    for i, pos in enumerate(rows):
        if (pos["fen"], args.depth) in done:
            continue
        mate_req = pos.get("teacher_mate") is not None
        a = run(engine, A, pos["fen"], args.depth, mate_req)
        b = run(engine, B, pos["fen"], args.depth, mate_req)
        rec = {"i": i, "depth": args.depth, "fen": pos["fen"],
               "source": pos.get("source"),
               "teacher_bestmove": pos.get("teacher_bestmove"),
               "teacher_mate": pos.get("teacher_mate"),
               "teacher_cp_stm": pos.get("teacher_cp_stm"),
               "a": a, "b": b}
        if a and b and "error" not in a and "timeout" not in a \
                and "error" not in b and "timeout" not in b:
            rec["a_match"] = a.get("bestmove") == pos.get("teacher_bestmove")
            rec["b_match"] = b.get("bestmove") == pos.get("teacher_bestmove")
            ca, cb = cp_value(a.get("score")), cp_value(b.get("score"))
            if ca is not None and cb is not None:
                rec["cp_delta"] = abs(ca - cb)
            ma, mb = mate_signed(a.get("score")), mate_signed(b.get("score"))
            rec["a_mate"], rec["b_mate"] = ma, mb
            rec["mate_transition"] = (ma is None) != (mb is None)
            if ma is not None and mb is not None and ma != mb:
                rec["mate_distance_change"] = [ma, mb]
        results["rows"].append(rec)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                       + "\n", encoding="utf-8")
        print(f"s74a_r2 {i}/{len(rows)} a_match={rec.get('a_match')} "
              f"b_match={rec.get('b_match')}", flush=True)
    summarize(results)
    return 0


def summarize(results: dict) -> None:
    ok = [r for r in results["rows"] if r.get("a_match") is not None]
    mate_rows = [r for r in ok if r.get("teacher_mate") is not None]
    deltas = [r.get("cp_delta") for r in ok if r.get("cp_delta") is not None]

    def side_ok(mate: int | None, teacher_mate: int) -> bool:
        return isinstance(mate, int) and (
            (teacher_mate > 0 and mate > 0) or (teacher_mate < 0 and mate < 0)
        )

    # Hard-reject scan: teacher says the side to move either mates
    # (teacher_mate > 0) or is being mated (teacher_mate < 0); baseline
    # finds that signed mate and candidate loses it / flips side.
    mate_losses = []
    for r in mate_rows:
        tm = r["teacher_mate"]
        if not isinstance(tm, int) or tm == 0:
            continue
        a_ok = side_ok(r.get("a_mate"), tm)
        b_ok = side_ok(r.get("b_mate"), tm)
        if a_ok and not b_ok:
            mate_losses.append({"i": r["i"], "fen": r["fen"],
                                "teacher_mate": tm,
                                "a_score": r["a"].get("score"),
                                "b_score": r["b"].get("score")})

    results["summary"] = {
        "corpus_positions": len(results["rows"]),
        "completed_rows": len(ok),
        "timeouts_or_errors": [
            {"i": r["i"], "a": r["a"], "b": r["b"]}
            for r in results["rows"] if r.get("a_match") is None],
        "mate_labelled_positions": len(mate_rows),
        "completed_mate_labelled_positions": sum(
            1 for r in mate_rows if r.get("a_match") is not None),
        "a_matches": sum(1 for r in ok if r["a_match"]),
        "b_matches": sum(1 for r in ok if r["b_match"]),
        "a_only": [r["i"] for r in ok if r["a_match"] and not r["b_match"]],
        "b_only": [r["i"] for r in ok if r["b_match"] and not r["a_match"]],
        "a_mate_detected": sum(1 for r in ok if r.get("a_mate") is not None),
        "b_mate_detected": sum(1 for r in ok if r.get("b_mate") is not None),
        "mate_side_mismatches": [
            r["i"] for r in ok
            if isinstance(r.get("a_mate"), int) and isinstance(r.get("b_mate"), int)
            and (r["a_mate"] > 0) != (r["b_mate"] > 0)],
        "cp_div_ge_100": sum(1 for v in deltas if v >= 100),
        "cp_div_ge_300": sum(1 for v in deltas if v >= 300),
        "cp_div_ge_500": sum(1 for v in deltas if v >= 500),
        "mate_transitions": [r["i"] for r in ok if r.get("mate_transition")],
        "mate_distance_changes": [r["i"] for r in ok
                                  if r.get("mate_distance_change")],
        "hard_reject_mate_losses": mate_losses,
    }
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"s74a_r2 done: {json.dumps(results['summary'])[:600]}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s74a_r2_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
