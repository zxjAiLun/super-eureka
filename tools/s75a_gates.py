#!/usr/bin/env python3
"""S7.5A single-evasion extension gate runner.

Runs the frozen gate chain against the same release binary:

    A = current-final                  (production baseline)
    B = current-final-single-evasion   (S7.5A candidate)

Gates:
  G3   fixed-depth explosion fuse: 80 S7 x d6/d7, nodes < 2x, wall < 2x
  G4   fixed-wall cost gate:        80 S7 x 1s/3s, frozen cost pass rule
  G5   fixed-depth teacher gate:    S6 teacher challenge, d6
  G5W  same-time tactical gate:     120 R2 x 1s/3s, teacher-directed
  G6   depth-stability diagnostic:  80 S7 d6 -> d7

Writes results/s7/s75a-gates.json incrementally (--resume skips done rows).
"""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S7_CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
R2_CORPUS = REPO / "data/s7/s74a-r2-tactical-corpus.jsonl"
TEACHER = REPO / "data/s6/s6-teacher-challenge-v1.jsonl"
OUT = REPO / "results/s7/s75a-gates.json"

A = "current-final"
B = "current-final-single-evasion"

FIELDS = [
    "nodes", "qsearch_nodes", "elapsed_us", "nps", "seldepth",
    "completed_depth", "score", "bestmove", "pv",
    "s75a_extension_applied_total", "s75a_extension_applied_depth1",
    "s75a_extension_budget_2_to_1", "s75a_extension_budget_1_to_0",
    "s75a_opportunity_blocked_budget_0",
]


def run_bench(engine: Path, profile: str, fen: str, limit: str,
              timeout: int = 7200) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", profile,
           "--mode", "cold", "--fen", fen] + shlex.split(limit)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True, "stderr": proc.stderr[:300]}
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("bench_result ")), None)
    if line is None:
        return {"error": True, "missing": True}
    rec: dict = {}
    for tok in shlex.split(line)[1:]:
        if "=" not in tok:
            continue
        key, value = tok.split("=", 1)
        if key in FIELDS:
            rec[key] = value
    return rec


def int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def cp_value(score: str | None) -> int | None:
    if not score or score == "none" or score.startswith("mate:"):
        return None
    try:
        return int(score[3:])
    except (TypeError, ValueError):
        return None


def mate_signed(score: str | None) -> int | None:
    if not score or not score.startswith("mate:"):
        return None
    try:
        return int(score[5:])
    except (TypeError, ValueError):
        return None


def row_ok(rec: dict | None) -> bool:
    return bool(rec) and not rec.get("error") and not rec.get("timeout")


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------
def gate_g3(engine: Path, result: dict, resume: bool) -> None:
    sec = result.setdefault("g3_rows", [])
    done = {(r["id"], r["depth"]) for r in sec} if resume else set()
    rows = [json.loads(l) for l in
            S7_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for depth in (6, 7):
        for pos in rows:
            if (pos["id"], depth) in done:
                continue
            a = run_bench(engine, A, pos["fen"], f"--depth {depth}")
            b = run_bench(engine, B, pos["fen"], f"--depth {depth}")
            sec.append({"id": pos["id"], "depth": depth, "fen": pos["fen"],
                        A: a, B: b})
            write(result)
            print(f"s75a_g3 {pos['id']} d{depth} "
                  f"nodes {a.get('nodes') if a else '?'} -> "
                  f"{b.get('nodes') if b else '?'}", flush=True)
    summaries = {}
    for depth in (6, 7):
        rs = [r for r in sec if r["depth"] == depth
              and row_ok(r[A]) and row_ok(r[B])]
        nodes_a = sum(int_or_none(r[A].get("nodes")) or 0 for r in rs)
        nodes_b = sum(int_or_none(r[B].get("nodes")) or 0 for r in rs)
        wall_a = sum(int_or_none(r[A].get("elapsed_us")) or 0 for r in rs)
        wall_b = sum(int_or_none(r[B].get("elapsed_us")) or 0 for r in rs)
        summaries[f"depth{depth}"] = {
            "positions": len(rs),
            "nodes_a": nodes_a, "nodes_b": nodes_b,
            "nodes_ratio": round(nodes_b / nodes_a, 6) if nodes_a else None,
            "wall_a_us": wall_a, "wall_b_us": wall_b,
            "wall_ratio": round(wall_b / wall_a, 6) if wall_a else None,
            "nodes_pass": nodes_b < 2 * nodes_a,
            "wall_pass": wall_b < 2 * wall_a,
        }
    result["g3_summary"] = summaries
    write(result)
    print(f"s75a_g3_summary {json.dumps(summaries, indent=2)}", flush=True)


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------
def gate_g4(engine: Path, result: dict, resume: bool) -> None:
    sec = result.setdefault("g4_rows", [])
    done = {(r["id"], r["movetime"]) for r in sec} if resume else set()
    rows = [json.loads(l) for l in
            S7_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for t in (1000, 3000):
        for idx, pos in enumerate(rows):
            if (pos["id"], t) in done:
                continue
            a = run_bench(engine, A, pos["fen"], f"--movetime {t}")
            b = run_bench(engine, B, pos["fen"], f"--movetime {t}")
            sec.append({"id": pos["id"], "movetime": t, "fen": pos["fen"],
                        A: a, B: b})
            write(result)
            da, db = (int_or_none(a.get("completed_depth")),
                      int_or_none(b.get("completed_depth")))
            print(f"s75a_g4 {pos['id']} {t}ms d {da} -> {db}", flush=True)
    summaries = {}
    for t in (1000, 3000):
        rs = [r for r in sec if r["movetime"] == t
              and row_ok(r[A]) and row_ok(r[B])
              and r[A].get("bestmove") and r[B].get("bestmove")]
        da = [int_or_none(r[A].get("completed_depth")) or 0 for r in rs]
        db = [int_or_none(r[B].get("completed_depth")) or 0 for r in rs]
        drops = [a - b for a, b in zip(da, db) if a - b >= 2]
        ext = sum(int_or_none(r[B].get("s75a_extension_applied_total")) or 0
                  for r in rs)
        ext1 = sum(int_or_none(r[B].get("s75a_extension_applied_depth1")) or 0
                   for r in rs)
        b21 = sum(int_or_none(r[B].get("s75a_extension_budget_2_to_1")) or 0
                  for r in rs)
        b10 = sum(int_or_none(r[B].get("s75a_extension_budget_1_to_0")) or 0
                  for r in rs)
        blocked = sum(
            int_or_none(r[B].get("s75a_opportunity_blocked_budget_0")) or 0
            for r in rs)
        summaries[f"movetime{t}"] = {
            "completed": len(rs),
            "median_depth_a": statistics.median(da),
            "median_depth_b": statistics.median(db),
            "mean_depth_a": round(statistics.mean(da), 3),
            "mean_depth_b": round(statistics.mean(db), 3),
            "median_seldepth_a": statistics.median(
                [int_or_none(r[A].get("seldepth")) or 0 for r in rs]),
            "median_seldepth_b": statistics.median(
                [int_or_none(r[B].get("seldepth")) or 0 for r in rs]),
            "total_nodes_a": sum(int_or_none(r[A].get("nodes")) or 0 for r in rs),
            "total_nodes_b": sum(int_or_none(r[B].get("nodes")) or 0 for r in rs),
            "extensions_applied": ext,
            "depth1_extensions": ext1,
            "budget_2_to_1": b21,
            "budget_1_to_0": b10,
            "budget0_blocked": blocked,
            "losing_ge_2_plies": len(drops),
            "pass_completed": len(rs) == len(rows),
            "pass_median": statistics.median(db) >= statistics.median(da) - 1,
            "pass_drops": len(drops) <= 8,
            "pass": (len(rs) == len(rows)
                     and statistics.median(db) >= statistics.median(da) - 1
                     and len(drops) <= 8),
        }
    result["g4_summary"] = summaries
    write(result)
    print(f"s75a_g4_summary {json.dumps(summaries, indent=2)}", flush=True)


# ---------------------------------------------------------------------------
# G5
# ---------------------------------------------------------------------------
def _mate_state_correct(m, teacher_mate):
    if not isinstance(m, int) or not isinstance(teacher_mate, int):
        return False
    return (m > 0 and teacher_mate > 0) or (m < 0 and teacher_mate < 0)


def gate_g5(engine: Path, result: dict, resume: bool) -> None:
    sec = result.setdefault("g5_rows", [])
    done = {r["i"] for r in sec} if resume else set()
    rows = [json.loads(l) for l in
            TEACHER.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, pos in enumerate(rows):
        if i in done:
            continue
        a = run_bench(engine, A, pos["fen"], "--depth 6", timeout=1800)
        b = run_bench(engine, B, pos["fen"], "--depth 6", timeout=1800)
        rec = {"i": i, "fen": pos["fen"], A: a, B: b,
               "teacher_bestmove": pos.get("teacher_bestmove"),
               "teacher_cp_stm": pos.get("teacher_cp_stm"),
               "teacher_mate": pos.get("teacher_mate")}
        if pos.get("teacher_bestmove") in (None, "(none)"):
            rec["classification"] = "TERMINAL/NOT_APPLICABLE"
        elif not row_ok(a) or not row_ok(b):
            rec["classification"] = "ENGINE_FAILURE"
        else:
            rec["classification"] = "OK"
            rec["a_match"] = a.get("bestmove") == pos.get("teacher_bestmove")
            rec["b_match"] = b.get("bestmove") == pos.get("teacher_bestmove")
            ca, cb = cp_value(a.get("score")), cp_value(b.get("score"))
            if ca is not None and cb is not None and pos.get("teacher_cp_stm") is not None:
                rec["a_error"] = abs(ca - pos["teacher_cp_stm"])
                rec["b_error"] = abs(cb - pos["teacher_cp_stm"])
            ma, mb = mate_signed(a.get("score")), mate_signed(b.get("score"))
            rec["a_mate"], rec["b_mate"] = ma, mb
        sec.append(rec)
        write(result)
        print(f"s75a_g5 {i} class={rec['classification']}", flush=True)

    rs = [r for r in sec if r.get("classification") == "OK"]
    hard = []
    wins = {"b_match": 0, "a_match": 0}
    cp_impr = {100: 0, 300: 0, 500: 0}
    cp_regr = {100: 0, 300: 0, 500: 0}
    mate_impr = 0
    mate_regr = 0
    mate_distance_impr = 0
    mate_distance_regr = 0
    for r in rs:
        tm = r.get("teacher_mate")
        ma, mb = r.get("a_mate"), r.get("b_mate")
        if isinstance(tm, int) and tm != 0:
            a_correct = _mate_state_correct(ma, tm)
            b_correct = _mate_state_correct(mb, tm)
            if a_correct and not b_correct:
                hard.append({"i": r["i"], "kind": "baseline_correct_mate_lost",
                             "fen": r["fen"]})
            if isinstance(mb, int) and not b_correct:
                hard.append({"i": r["i"], "kind": "candidate_wrong_mate_side",
                             "fen": r["fen"]})
            if not a_correct and b_correct:
                mate_impr += 1
            if a_correct and not b_correct:
                mate_regr += 1
            if a_correct and b_correct and isinstance(ma, int) and isinstance(mb, int):
                if abs(mb) < abs(ma):
                    mate_distance_impr += 1
                elif abs(mb) > abs(ma):
                    mate_distance_regr += 1
        if r.get("a_match") and not r.get("b_match"):
            wins["a_match"] += 1
        if r.get("b_match") and not r.get("a_match"):
            wins["b_match"] += 1
        if "a_error" in r and "b_error" in r:
            for thr in (100, 300, 500):
                if r["a_error"] - r["b_error"] >= thr:
                    cp_impr[thr] += 1
                if r["b_error"] - r["a_error"] >= thr:
                    cp_regr[thr] += 1
    summary = {
        "positions": len(rs),
        "terminal_not_applicable": sum(1 for r in sec if r.get("classification") == "TERMINAL/NOT_APPLICABLE"),
        "engine_failures": sum(1 for r in sec if r.get("classification") == "ENGINE_FAILURE"),
        "a_matches": sum(1 for r in rs if r.get("a_match")),
        "b_matches": sum(1 for r in rs if r.get("b_match")),
        "candidate_only_matches": wins["b_match"],
        "baseline_only_matches": wins["a_match"],
        "cp_improvements": cp_impr,
        "cp_regressions": cp_regr,
        "mate_improvements": mate_impr,
        "mate_regressions": mate_regr,
        "mate_distance_improvements": mate_distance_impr,
        "mate_distance_regressions": mate_distance_regr,
        "hard_rejects": hard,
    }
    summary["pass"] = (
        len(hard) == 0
        and wins["b_match"] >= wins["a_match"]
        and cp_impr[100] >= cp_regr[100]
        and cp_impr[300] >= cp_regr[300]
        and cp_regr[500] == 0
    )
    result["g5_summary"] = summary
    write(result)
    print(f"s75a_g5_summary {json.dumps(summary, indent=2)}", flush=True)


# ---------------------------------------------------------------------------
# G5W
# ---------------------------------------------------------------------------
def _g5w_classify(a, b, pos):
    """Returns (candidate_wins, candidate_losses, hard_reject_kind)."""
    ma, mb = mate_signed(a.get("score")), mate_signed(b.get("score"))
    tm = pos.get("teacher_mate")
    if isinstance(tm, int) and tm != 0:
        a_ok = _mate_state_correct(ma, tm)
        b_ok = _mate_state_correct(mb, tm)
        if a_ok and not b_ok:
            return False, True, "baseline_correct_mate_lost"
        if isinstance(mb, int) and not b_ok:
            return False, True, "candidate_wrong_mate_side"
        if not a_ok and b_ok:
            return True, False, None
        if a_ok and b_ok and isinstance(ma, int) and isinstance(mb, int):
            if abs(mb) < abs(ma):
                return True, False, None
            if abs(mb) > abs(ma):
                return False, True, None
        return False, False, None
    teacher_move = pos.get("teacher_bestmove")
    am = a.get("bestmove") == teacher_move
    bm = b.get("bestmove") == teacher_move
    if am != bm:
        return bm, am, None
    if not am and not bm:
        ca, cb = cp_value(a.get("score")), cp_value(b.get("score"))
        tc = pos.get("teacher_cp_stm")
        if ca is not None and cb is not None and isinstance(tc, int):
            ea, eb = abs(ca - tc), abs(cb - tc)
            if ea - eb >= 100:
                return True, False, None
            if eb - ea >= 100:
                return False, True, None
    return False, False, None


def gate_g5w(engine: Path, result: dict, resume: bool) -> None:
    sec = result.setdefault("g5w_rows", [])
    done = {(r["i"], r["movetime"]) for r in sec} if resume else set()
    rows = [json.loads(l) for l in
            R2_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for t in (1000, 3000):
        for i, pos in enumerate(rows):
            if (i, t) in done:
                continue
            a = run_bench(engine, A, pos["fen"], f"--movetime {t}")
            b = run_bench(engine, B, pos["fen"], f"--movetime {t}")
            rec = {"i": i, "movetime": t, "fen": pos["fen"], A: a, B: b,
                   "teacher_bestmove": pos.get("teacher_bestmove"),
                   "teacher_cp_stm": pos.get("teacher_cp_stm"),
                   "teacher_mate": pos.get("teacher_mate")}
            if row_ok(a) and row_ok(b) and a.get("bestmove") and b.get("bestmove"):
                win, loss, hard = _g5w_classify(a, b, pos)
                rec.update({"candidate_win": win, "candidate_loss": loss,
                            "hard_reject": hard, "completed": True})
            else:
                rec["completed"] = False
            sec.append(rec)
            write(result)
            print(f"s75a_g5w {i} {t}ms win={rec.get('candidate_win')} "
                  f"loss={rec.get('candidate_loss')} hard={rec.get('hard_reject')}",
                  flush=True)
    summaries = {}
    for t in (1000, 3000):
        rs = [r for r in sec if r["movetime"] == t]
        comp = [r for r in rs if r.get("completed")]
        wins = sum(1 for r in comp if r.get("candidate_win"))
        losses = sum(1 for r in comp if r.get("candidate_loss"))
        hard = [r for r in comp if r.get("hard_reject")]
        summaries[f"movetime{t}"] = {
            "completed": len(comp),
            "expected": len(rows),
            "candidate_wins": wins,
            "candidate_losses": losses,
            "hard_rejects": hard,
            "pass": len(comp) == len(rows) and len(hard) == 0
                    and (wins > losses if t == 3000 else wins >= losses),
        }
    result["g5w_summary"] = summaries
    write(result)
    print(f"s75a_g5w_summary {json.dumps(summaries, indent=2)}", flush=True)


# ---------------------------------------------------------------------------
# G6
# ---------------------------------------------------------------------------
def gate_g6(engine: Path, result: dict, resume: bool) -> None:
    sec = result.setdefault("g6_rows", [])
    done = {r["id"] for r in sec} if resume else set()
    rows = [json.loads(l) for l in
            S7_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for pos in rows:
        if pos["id"] in done:
            continue
        rec = {"id": pos["id"], "fen": pos["fen"]}
        for prof in (A, B):
            d6 = run_bench(engine, prof, pos["fen"], "--depth 6")
            d7 = run_bench(engine, prof, pos["fen"], "--depth 7")
            rec[prof] = {"d6": d6, "d7": d7}
            if row_ok(d6) and row_ok(d7):
                rec[f"{prof}_bestmove_change"] = d6.get("bestmove") != d7.get("bestmove")
                c6, c7 = cp_value(d6.get("score")), cp_value(d7.get("score"))
                if c6 is not None and c7 is not None and abs(c6 - c7) >= 200:
                    rec[f"{prof}_reversal_ge_200"] = True
                m6, m7 = mate_signed(d6.get("score")), mate_signed(d7.get("score"))
                rec[f"{prof}_mate_transition"] = (m6 is None) != (m7 is None)
        sec.append(rec)
        write(result)
        print(f"s75a_g6 {pos['id']}", flush=True)
    summary = {
        "positions": len(sec),
        "a_bestmove_changes": sum(1 for r in sec if r.get(f"{A}_bestmove_change")),
        "b_bestmove_changes": sum(1 for r in sec if r.get(f"{B}_bestmove_change")),
        "a_reversals_ge_200": sum(1 for r in sec if r.get(f"{A}_reversal_ge_200")),
        "b_reversals_ge_200": sum(1 for r in sec if r.get(f"{B}_reversal_ge_200")),
        "a_mate_transitions": sum(1 for r in sec if r.get(f"{A}_mate_transition")),
        "b_mate_transitions": sum(1 for r in sec if r.get(f"{B}_mate_transition")),
    }
    result["g6_summary"] = summary
    write(result)
    print(f"s75a_g6_summary {json.dumps(summary, indent=2)}", flush=True)


def write(result: dict) -> None:
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--gates", default="g3,g4,g5,g5w,g6")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    engine = args.engine.resolve()
    result = {}
    if args.resume and OUT.exists():
        result = json.loads(OUT.read_text(encoding="utf-8"))
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    for gate in gates:
        if gate == "g3":
            gate_g3(engine, result, args.resume)
        elif gate == "g4":
            gate_g4(engine, result, args.resume)
        elif gate == "g5":
            gate_g5(engine, result, args.resume)
        elif gate == "g5w":
            gate_g5w(engine, result, args.resume)
        elif gate == "g6":
            gate_g6(engine, result, args.resume)
        else:
            raise SystemExit(f"unknown gate {gate}")
    write(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s75a_gates_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
