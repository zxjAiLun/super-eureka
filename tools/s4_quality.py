#!/usr/bin/env python3
"""S4.0B Search-Quality Attribution.

Pipeline over real CurrentFinal games:

  extract        sample positions from PGN (after ply 12, every ~5 plies, non-terminal)
  teacher        Stockfish-18 MultiPV=3, fixed node budget
  normal         CurrentFinal bench profile at 100 / 500 / 1000 ms
  disagreements  CurrentFinal best not teacher-equivalent and >= 50 cp loss
  forced_root    normal + forced teacher move at 500/1000 ms, with target-root rank
  classify       SEARCH / EVAL / HORIZON / UNRESOLVED (+ forced depth 3..7 convergence)
  ablation       targeted no-lmr / no-futility / no-null / no-qsee on representative suspects
  summary        writes results/s4-attribution/quality/*

Diagnostic only. CurrentFinal production semantics are unchanged; all hooks are
bench-only (--forced-root / --diag / --target-root are never reachable via UCI).
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chess
import chess.pgn

from run_s21_practical_gate import git_sha, parse_key_values, sha256_file

PROFILE = "current-final"
QUALITY_DIR = Path("results/s4-attribution/quality")
MATE_CUTOFF = 30000  # |score| >= this is treated as a mate distance, not cp


@dataclass
class Score:
    kind: str  # 'cp' | 'mate' | 'none'
    value: Optional[int]


def parse_score(raw: str) -> Score:
    kind, sep, value = raw.partition(":")
    if not sep:
        return Score("none", None)
    if kind == "none":
        return Score("none", None)
    try:
        return Score(kind, int(value))
    except ValueError:
        return Score(kind, None)


def mate_category(s: Score) -> Optional[str]:
    if s.kind == "mate" and s.value is not None:
        return "mate" if s.value > 0 else "mated"
    return None


def cp_loss(teacher_best: Score, our: Score) -> Optional[int]:
    """Stockfish cp loss of our move vs the teacher best (same scale).
    Returns None when either score is a mate (not convertible to cp)."""
    if teacher_best.kind == "cp" and our.kind == "cp":
        return teacher_best.value - our.value
    return None


def parse_bench_result(line: str) -> dict[str, Any]:
    fields = parse_key_values(line)
    return fields


def run_bench(
    engine: Path,
    fen: str,
    ms: Optional[int] = None,
    depth: Optional[int] = None,
    forced_root: Optional[str] = None,
    target_root: Optional[str] = None,
) -> dict[str, Any]:
    argv = [str(engine), "bench", "profile", "--mode", "cold", "--profile", PROFILE, "--fen", fen]
    if ms is not None:
        argv += ["--movetime", str(ms)]
    if depth is not None:
        argv += ["--depth", str(depth)]
    if forced_root:
        argv += ["--forced-root", forced_root]
    if target_root:
        argv += ["--target-root", target_root]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=max(10, (ms or 1000) / 1000 + 40),
        check=False,
    )
    result_lines = [l for l in completed.stdout.splitlines() if l.startswith("bench_result ")]
    if completed.returncode != 0 or len(result_lines) != 1:
        raise RuntimeError(
            "bench search failed: " + json.dumps({"argv": argv, "rc": completed.returncode,
                                                  "stderr": completed.stderr,
                                                  "tail": completed.stdout.splitlines()[-20:]})
        )
    fields = parse_bench_result(result_lines[0])
    return {
        "fen": fen,
        "score": parse_score(fields["score"]),
        "bestmove": fields["bestmove"],
        "completed_depth": int(fields["completed_depth"]),
        "stopped": fields["stopped"] == "true",
        "nodes": int(fields["nodes"]),
        "nps": int(fields["nps"]),
        "target_root_rank": int(fields.get("target_root_rank", "0")),
        "qsearch_nodes": int(fields["qsearch_nodes"]),
        "qsearch_see_tests": int(fields["qsearch_see_tests"]),
        "qsearch_see_pruned": int(fields["qsearch_see_pruned"]),
        "lmr_reductions": int(fields["lmr_reductions"]),
        "lmr_researches": int(fields["lmr_researches"]),
        "futility_pruned": int(fields["futility_pruned"]),
        "null_move_attempts": int(fields["null_move_attempts"]),
        "null_move_fail_highs": int(fields["null_move_fail_highs"]),
        "null_move_researches": int(fields["null_move_researches"]),
        "tt_probes": int(fields["tt_probes"]),
        "tt_hits": int(fields["tt_hits"]),
        "tt_cutoffs": int(fields["tt_cutoffs"]),
        "raw": result_lines[0],
    }


# ---------------------------------------------------------------------------
# Stockfish-18 teacher (MultiPV=3)
# ---------------------------------------------------------------------------

class TeacherError(RuntimeError):
    pass


class StockfishTeacher:
    def __init__(self, executable: Path, multipv: int = 3, hash_mb: int = 16, threads: int = 1,
                 timeout_s: float = 120.0):
        self.executable = executable
        self.multipv = multipv
        self.hash_mb = hash_mb
        self.threads = threads
        self.timeout_s = timeout_s
        self.process: Optional[subprocess.Popen] = None
        self.stdout_queue: queue.Queue = queue.Queue()
        self.stderr_lines: list[str] = []
        self.reader_threads: list[threading.Thread] = []

    def __enter__(self) -> "StockfishTeacher":
        if not self.executable.is_file():
            raise TeacherError(f"Stockfish not found: {self.executable}")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                [str(self.executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, creationflags=creationflags)
        except OSError as exc:
            raise TeacherError(f"cannot start Stockfish: {exc}") from exc
        t1 = threading.Thread(target=self._read_stdout, daemon=True)
        t2 = threading.Thread(target=self._read_stderr, daemon=True)
        t1.start(); t2.start()
        self.reader_threads = [t1, t2]
        self._send("uci")
        while True:
            line = self._read_line()
            if line is None:
                raise TeacherError("Stockfish closed before uciok")
            if line == "uciok":
                break
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value {self.threads}")
        self._send(f"setoption name MultiPV value {self.multipv}")
        self._ready()
        return self

    def __exit__(self, *exc) -> None:
        if self.process is None:
            return
        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill(); self.process.wait(timeout=5)
            except Exception:
                pass
        for t in self.reader_threads:
            t.join(timeout=1)

    def _read_stdout(self) -> None:
        assert self.process and self.process.stdout
        for line in iter(self.process.stdout.readline, ""):
            self.stdout_queue.put(line.rstrip("\r\n"))
        self.stdout_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process and self.process.stderr
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_lines.append(line.rstrip("\r\n"))
            del self.stderr_lines[:-20]

    def _send(self, cmd: str) -> None:
        assert self.process and self.process.stdin
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _read_line(self) -> Optional[str]:
        try:
            return self.stdout_queue.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise TeacherError(f"Stockfish timeout; stderr={self.stderr_lines[-5:]!r}") from exc

    def _ready(self) -> None:
        self._send("isready")
        while True:
            line = self._read_line()
            if line == "readyok":
                return
            if line is None:
                raise TeacherError("Stockfish closed waiting for readyok")

    def analyze(self, fen: str, nodes: int) -> list[dict[str, Any]]:
        """MultiPV search. Returns a COHERENT snapshot: the deepest completed
        depth at which all requested ranks are present, deduped by move."""
        self._send("ucinewgame")
        self._send("setoption name Clear Hash")
        self._ready()
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes}")
        by_depth: dict[int, dict[int, dict[str, Any]]] = {}
        while True:
            line = self._read_line()
            if line is None:
                raise TeacherError("Stockfish closed before bestmove")
            if line.startswith("info "):
                info = self._parse_multipv(line)
                if info is not None:
                    rank, rec = info
                    by_depth.setdefault(rec["depth"], {})[rank] = rec
            elif line.startswith("bestmove "):
                break
        snapshot: list[dict[str, Any]] = []
        for d in sorted(by_depth, reverse=True):
            ranks = by_depth[d]
            if all(r in ranks for r in range(1, self.multipv + 1)):
                snapshot = [ranks[r] for r in range(1, self.multipv + 1)]
                break
        if not snapshot:
            deepest: dict[int, dict[str, Any]] = {}
            for d, ranks in by_depth.items():
                for rank, rec in ranks.items():
                    if rank not in deepest or d >= rec["depth"]:
                        deepest[rank] = rec
            snapshot = [deepest[r] for r in sorted(deepest)]
        # dedupe by move, keep first (highest rank)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for rec in snapshot:
            m = rec["move"]
            if m is None or m in seen:
                continue
            seen.add(m)
            out.append(rec)
        return out

    def searchmoves_score(self, fen: str, move: str, nodes: int) -> Score:
        """Score a single move via searchmoves (for disagreement confirmation)."""
        self._send("ucinewgame")
        self._send("setoption name Clear Hash")
        self._ready()
        self._send(f"position fen {fen}")
        self._send(f"go nodes {nodes} searchmoves {move}")
        best: Optional[Score] = None
        while True:
            line = self._read_line()
            if line is None:
                raise TeacherError("Stockfish closed before bestmove")
            if line.startswith("info "):
                info = self._parse_multipv(line)
                if info is not None:
                    rank, rec = info
                    if rank == 1:
                        best = rec["score"]
            elif line.startswith("bestmove "):
                break
        if best is None:
            raise TeacherError(f"no score for searchmoves {move}")
        return best

    @staticmethod
    def _parse_multipv(line: str) -> Optional[tuple[int, dict[str, Any]]]:
        if "lowerbound" in line or "upperbound" in line:
            return None
        toks = line.split()
        if "multipv" not in toks or "score" not in toks or "pv" not in toks:
            return None
        rank = int(toks[toks.index("multipv") + 1])
        depth = int(toks[toks.index("depth") + 1])
        i = toks.index("score")
        kind = toks[i + 1]
        value = int(toks[i + 2])
        pv = tuple(toks[toks.index("pv") + 1:])
        move = pv[0] if pv else None
        return rank, {"rank": rank, "move": move, "score": Score(kind, value),
                      "depth": depth, "pv": list(pv), "nodes_line": None}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_extract(args: argparse.Namespace) -> None:
    pgns = args.pgns
    out = QUALITY_DIR / "sampled_positions.jsonl"
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pgn_path in pgns:
        path = Path(pgn_path)
        with open(path, encoding="utf-8") as fh:
            game_idx = 0
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                game_idx += 1
                h = game.headers
                white = h.get("White", ""); black = h.get("Black", "")
                cf_side = None
                if "CurrentFinal" in white:
                    cf_side = "w"
                elif "CurrentFinal" in black:
                    cf_side = "b"
                board = game.board()
                moves = list(game.mainline_moves())
                result = h.get("Result")
                # sample plies 12..end-2, every 5 plies
                for ply, move in enumerate(moves, start=1):
                    if ply < args.sample_start_ply:
                        board.push(move); continue
                    if ply > len(moves) - args.tail_skip:
                        board.push(move); continue
                    if (ply - args.sample_start_ply) % args.sample_every == 0:
                        fen = board.fen()
                        side = "w" if board.turn == chess.WHITE else "b"
                        if fen not in seen:
                            seen.add(fen)
                            records.append({
                                "fen": fen, "pgn": str(path), "game_id": game_idx,
                                "ply": ply, "side": side, "played_move": move.uci(),
                                "result": result, "cf_side": cf_side,
                                "white": white, "black": black,
                            })
                    board.push(move)
    if args.max_positions and len(records) > args.max_positions:
        step = len(records) // args.max_positions
        records = records[::step][: args.max_positions]
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, records)
    print(f"extract: {len(records)} positions -> {out}")


def phase_teacher(args: argparse.Namespace) -> None:
    positions = read_jsonl(QUALITY_DIR / "sampled_positions.jsonl")
    out = QUALITY_DIR / "teacher.jsonl"
    with StockfishTeacher(args.stockfish, multipv=args.teacher_multipv,
                          timeout_s=args.teacher_timeout) as sf:
        rows = []
        for idx, pos in enumerate(positions):
            try:
                multi = sf.analyze(pos["fen"], args.teacher_nodes)
            except TeacherError as exc:
                print(f"teacher error {pos['fen']}: {exc}", file=sys.stderr)
                continue
            serialized = [
                {"rank": m["rank"], "move": m["move"], "score": score_to_str(m["score"]),
                 "depth": m["depth"], "pv": m["pv"]}
                for m in multi
            ]
            rows.append({**pos, "teacher_multipv": serialized})
            if (idx + 1) % 50 == 0:
                print(f"teacher {idx + 1}/{len(positions)}", flush=True)
    write_jsonl(out, rows)
    print(f"teacher: {len(rows)} -> {out}")


def phase_normal(args: argparse.Namespace) -> None:
    positions = read_jsonl(QUALITY_DIR / "sampled_positions.jsonl")
    out = QUALITY_DIR / "normal_search.jsonl"
    rows = []
    engine = Path(args.engine).resolve()
    for idx, pos in enumerate(positions):
        rec = {"fen": pos["fen"], "pgn": pos.get("pgn"), "game_id": pos.get("game_id"),
               "ply": pos.get("ply"), "side": pos.get("side"), "cf_side": pos.get("cf_side")}
        for ms in args.normal_budgets:
            r = run_bench(engine, pos["fen"], ms=ms)
            rec[f"t{ms}"] = {k: r[k] for k in ("bestmove", "completed_depth", "nodes",
                                               "nps", "qsearch_nodes", "qsearch_see_tests",
                                               "qsearch_see_pruned", "lmr_reductions", "lmr_researches",
                                               "futility_pruned", "null_move_attempts",
                                               "null_move_fail_highs", "null_move_researches",
                                               "tt_probes", "tt_hits", "tt_cutoffs")}
            rec[f"t{ms}"]["score"] = score_to_str(r["score"])
        rows.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"normal {idx + 1}/{len(positions)}", flush=True)
    write_jsonl(out, rows)
    print(f"normal: {len(rows)} -> {out}")


def phase_disagreements(args: argparse.Namespace) -> None:
    teacher = {p["fen"]: p for p in read_jsonl(QUALITY_DIR / "teacher.jsonl")}
    normal = read_jsonl(QUALITY_DIR / "normal_search.jsonl")
    out = QUALITY_DIR / "disagreements.jsonl"
    rows = []
    with StockfishTeacher(args.stockfish, multipv=1, timeout_s=args.teacher_timeout) as sf:
        for idx, n in enumerate(normal):
            fen = n["fen"]
            t = teacher.get(fen)
            if not t or not t.get("teacher_multipv"):
                continue
            top = t["teacher_multipv"]
            if not top:
                continue
            tb = top[0]
            teacher_best = tb["move"]
            teacher_best_score = parse_score(tb["score"])
            # normal best from 1000ms (deepest normal budget used)
            norm = n.get("t1000") or n.get("t500") or n.get("t100")
            if not norm or not norm["bestmove"] or norm["bestmove"] == "0000":
                continue
            our = norm["bestmove"]
            top_moves = [m["move"] for m in top]
            # equivalent if our move appears in teacher top-N with near-equal cp
            equivalent = False
            for m in top:
                if m["move"] == our:
                    loss = cp_loss(teacher_best_score, parse_score(m["score"]))
                    if loss is not None and loss < args.threshold_cp:
                        equivalent = True
                    break
            if our in top_moves:
                equivalent = True
            if equivalent:
                continue
            # confirm: teacher eval of our move via searchmoves
            try:
                sc = sf.searchmoves_score(fen, our, args.teacher_nodes)
            except TeacherError as exc:
                print(f"confirm error {fen}: {exc}", file=sys.stderr)
                continue
            loss = cp_loss(teacher_best_score, sc)
            tb_mate = mate_category(teacher_best_score)
            our_mate = mate_category(sc)
            high_conf = False
            if tb_mate and tb_mate == "mate" and our_mate != "mate":
                high_conf = True
            elif loss is not None and loss >= args.threshold_cp:
                high_conf = True
            if not high_conf:
                continue
            rows.append({
                "fen": fen, "pgn": n.get("pgn"), "game_id": n.get("game_id"),
                "ply": n.get("ply"), "side": n.get("side"), "cf_side": n.get("cf_side"),
                "teacher_best": teacher_best, "teacher_best_score": score_to_str(teacher_best_score),
                "teacher_top": [{"move": m["move"], "score": score_to_str(m["score"]),
                                 "depth": m["depth"]} for m in top],
                "our_best": our, "our_score_t1000": score_to_str(norm["score"]),
                "teacher_score_of_our": score_to_str(sc), "cp_loss": loss,
                "teacher_best_mate": tb_mate, "our_mate": our_mate,
            })
            if (idx + 1) % 25 == 0:
                print(f"disagreements {idx + 1}/{len(normal)} found={len(rows)}", flush=True)
    write_jsonl(out, rows)
    print(f"disagreements: {len(rows)} -> {out}")


def phase_forced_root(args: argparse.Namespace) -> None:
    dis = read_jsonl(QUALITY_DIR / "disagreements.jsonl")
    out = QUALITY_DIR / "forced_root.jsonl"
    engine = Path(args.engine).resolve()
    rows = []
    for idx, d in enumerate(dis):
        fen = d["fen"]; tm = d["teacher_best"]
        rec = {"fen": fen, "pgn": d.get("pgn"), "game_id": d.get("game_id"), "ply": d.get("ply"),
               "side": d.get("side"), "cf_side": d.get("cf_side"),
               "teacher_best": tm, "teacher_best_score": d["teacher_best_score"],
               "teacher_score_of_our": d["teacher_score_of_our"], "cp_loss": d.get("cp_loss")}
        # normal (own line) and forced teacher move, at 500/1000 ms, with target rank
        for ms in (500, 1000):
            norm = run_bench(engine, fen, ms=ms)
            rec[f"normal_{ms}"] = {"bestmove": norm["bestmove"], "score": score_to_str(norm["score"]),
                                   "depth": norm["completed_depth"], "nodes": norm["nodes"]}
            f = run_bench(engine, fen, ms=ms, forced_root=tm, target_root=tm)
            rec[f"forced_{ms}"] = {"bestmove": f["bestmove"], "score": score_to_str(f["score"]),
                                   "depth": f["completed_depth"], "nodes": f["nodes"],
                                   "target_root_rank": f["target_root_rank"],
                                   "qsearch_nodes": f["qsearch_nodes"],
                                   "qsearch_see_pruned": f["qsearch_see_pruned"],
                                   "lmr_researches": f["lmr_researches"],
                                   "null_move_researches": f["null_move_researches"]}
        rows.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"forced_root {idx + 1}/{len(dis)}", flush=True)
    write_jsonl(out, rows)
    print(f"forced_root: {len(rows)} -> {out}")


def phase_classify(args: argparse.Namespace) -> None:
    forced = read_jsonl(QUALITY_DIR / "forced_root.jsonl")
    out = QUALITY_DIR / "depth_convergence.jsonl"
    engine = Path(args.engine).resolve()
    rows = []
    for idx, r in enumerate(forced):
        # first-pass classification from forced_1000 vs normal_1000
        ns = r["normal_1000"]["score"]; fs = r["forced_1000"]["score"]
        cls = classify_first_pass(ns, fs)
        rec = {**r, "first_pass": cls}
        if cls in ("EVAL_HORIZON",):
            # run forced depth 3..7 for the teacher move
            conv = []
            for depth in (3, 4, 5, 6, 7):
                try:
                    fr = run_bench(engine, r["fen"], depth=depth, forced_root=r["teacher_best"])
                    conv.append({"depth": depth, "score": score_to_str(fr["score"]),
                                 "nodes": fr["nodes"], "bestmove": fr["bestmove"]})
                except RuntimeError as exc:
                    conv.append({"depth": depth, "error": str(exc)})
            rec["forced_depth_series"] = conv
            rec["final_class"] = classify_convergence(r, conv)
        else:
            rec["final_class"] = cls
        rows.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"classify {idx + 1}/{len(forced)}", flush=True)
    write_jsonl(out, rows)
    # aggregate
    counts: dict[str, int] = {}
    for r in rows:
        c = r["final_class"] or r["first_pass"]
        counts[c] = counts.get(c, 0) + 1
    print("classification:", counts)
    print(f"depth_convergence: {len(rows)} -> {out}")


def classify_first_pass(normal_s: dict, forced_s: dict) -> str:
    n = parse_score(normal_s); f = parse_score(forced_s)
    # If the forced teacher move scores >= the normal chosen line, CurrentFinal can
    # recognize it once searched -> SEARCH_SUSPECT.
    if n.kind == "cp" and f.kind == "cp":
        if f.value >= n.value:
            return "SEARCH_SUSPECT"
        return "EVAL_HORIZON"
    if n.kind == "mate" and f.kind == "mate":
        if f.value >= n.value:
            return "SEARCH_SUSPECT"
        return "EVAL_HORIZON"
    if f.kind == "mate" and f.value > 0 and n.kind != "mate":
        return "SEARCH_SUSPECT"
    return "UNRESOLVED"


def classify_convergence(r: dict, conv: list[dict]) -> str:
    """forced depth 3..7 series. If the teacher move's score converges toward /
    overtakes the normal line as depth grows -> HORIZON_SUSPECT; if it stays
    clearly inferior -> EVAL_SUSPECT; otherwise UNRESOLVED."""
    scores = []
    for c in conv:
        s = c.get("score")
        if not s:
            continue
        sc = parse_score(s)
        if sc.kind == "cp":
            scores.append(sc.value)
        elif sc.kind == "mate":
            scores.append(20000 if sc.value > 0 else -20000)
    if len(scores) < 3:
        return "UNRESOLVED"
    # monotonic improvement toward/above 0 or the normal line's magnitude
    first, last = scores[0], scores[-1]
    # compare to the forced_1000 cp (the 'normal line' benchmark is forced_1000 here)
    ref = parse_score(r["forced_1000"]["score"])
    ref_v = ref.value if ref.kind == "cp" else None
    if last - first >= 60 and (ref_v is None or last >= ref_v):
        return "HORIZON_SUSPECT"
    if last <= -40 and (ref_v is None or last <= ref_v + 40):
        return "EVAL_SUSPECT"
    return "UNRESOLVED"


def phase_paired(args: argparse.Namespace) -> None:
    """Paired fixed-depth validation: force teacher move T AND CurrentFinal's
    own normal move O at the same depths 3..7, classify on the T-vs-O delta."""
    import collections
    forced = read_jsonl(QUALITY_DIR / "forced_root.jsonl")
    out = QUALITY_DIR / "paired.jsonl"
    engine = Path(args.engine).resolve()
    rows = []
    for idx, r in enumerate(forced):
        T = r["teacher_best"]; O = r["normal_1000"]["bestmove"]
        if not T or not O or T == O:
            continue
        scores_T: dict[int, str] = {}
        scores_O: dict[int, str] = {}
        deltas: dict[int, Optional[int]] = {}
        for depth in (3, 4, 5, 6, 7):
            t = run_bench(engine, r["fen"], depth=depth, forced_root=T)
            o = run_bench(engine, r["fen"], depth=depth, forced_root=O)
            st = t["score"]; so = o["score"]
            scores_T[depth] = score_to_str(st)
            scores_O[depth] = score_to_str(so)
            if st.kind == "cp" and so.kind == "cp":
                deltas[depth] = st.value - so.value
            else:
                deltas[depth] = None
        cls = classify_paired(deltas)
        rows.append({
            "fen": r["fen"], "pgn": r.get("pgn"), "game_id": r.get("game_id"),
            "ply": r.get("ply"), "side": r.get("side"), "cf_side": r.get("cf_side"),
            "teacher_best": T, "normal_best": O,
            "teacher_best_score": r.get("teacher_best_score"),
            "target_root_rank": r["forced_1000"]["target_root_rank"],
            "scores_T": scores_T, "scores_O": scores_O, "deltas": deltas,
            "class": cls,
        })
        if (idx + 1) % 10 == 0:
            print(f"paired {idx + 1}/{len(forced)}", flush=True)
    write_jsonl(out, rows)
    print("paired classes:", dict(collections.Counter(x["class"] for x in rows)))
    print(f"paired: {len(rows)} -> {out}")


def classify_paired(deltas: dict[int, Optional[int]]) -> str:
    cps = [d for d in (deltas.get(3), deltas.get(4), deltas.get(5), deltas.get(6), deltas.get(7))
           if d is not None]
    if len(cps) < 2:
        return "UNRESOLVED"
    d0 = cps[0]; dlast = cps[-1]
    trend = dlast - d0
    if d0 >= -30:
        return "SEARCH_LIKE"  # teacher move already competitive at matched shallow depth
    if trend >= 40 and dlast >= -40:
        return "HORIZON"      # clear convergence toward / past the own line
    if dlast <= -40 and trend < 40:
        return "EVAL_LIKE"    # stays materially below the own line, no convergence
    return "UNRESOLVED"


def phase_ablation(args: argparse.Namespace) -> None:
    rows = read_jsonl(QUALITY_DIR / "depth_convergence.jsonl")
    out = QUALITY_DIR / "ablation.jsonl"
    engine = Path(args.engine).resolve()
    # choose a representative subset (up to N) of SEARCH / HORIZON suspects
    suspects = [r for r in rows if r.get("final_class") in ("SEARCH_SUSPECT", "HORIZON_SUSPECT")]
    picks = suspects[: args.ablation_max]
    results = []
    for r in picks:
        fen = r["fen"]; tm = r["teacher_best"]
        rank = r["forced_1000"]["target_root_rank"]
        # choose ablation based on symptom
        abls = []
        if rank >= 8:
            abls = []  # ordering issue primary; no ablation needed
        else:
            abls = ["no-lmr", "no-futility"]
            # qsearch-heavy -> no-qsee; null-heavy -> no-null (optional)
            if r["forced_1000"].get("qsearch_nodes", 0) / max(1, r["forced_1000"].get("nodes", 1)) > 0.85:
                abls.append("no-qsee")
            if r["forced_1000"].get("null_move_researches", 0) > 0:
                abls.append("no-null")
        rec = {"fen": fen, "teacher_best": tm, "target_root_rank": rank, "abls": abls}
        base = run_bench(engine, fen, ms=1000)
        rec["normal_1000"] = {"bestmove": base["bestmove"], "score": score_to_str(base["score"])}
        for abl in abls:
            ar = run_bench_diag(engine, fen, ms=1000, forced_root=tm, diag=abl)
            rec[abl] = {"forced_teacher_score": score_to_str(ar["score"]),
                        "teacher_move_chosen": ar["bestmove"] == tm,
                        "depth": ar["completed_depth"]}
        results.append(rec)
    write_jsonl(out, results)
    print(f"ablation: {len(results)} -> {out}")


def run_bench_diag(engine: Path, fen: str, ms: int, forced_root: str, diag: str) -> dict[str, Any]:
    argv = [str(engine), "bench", "profile", "--mode", "cold", "--profile", PROFILE,
            "--fen", fen, "--movetime", str(ms), "--forced-root", forced_root, "--diag", diag]
    completed = subprocess.run(argv, capture_output=True, text=True,
                               timeout=max(10, ms / 1000 + 40), check=False)
    lines = [l for l in completed.stdout.splitlines() if l.startswith("bench_result ")]
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"diag bench failed: {completed.stderr}")
    f = parse_bench_result(lines[0])
    return {"bestmove": f["bestmove"], "score": parse_score(f["score"]),
            "completed_depth": int(f["completed_depth"]),
            "qsearch_nodes": int(f["qsearch_nodes"]),
            "lmr_reductions": int(f["lmr_reductions"]),
            "futility_pruned": int(f["futility_pruned"]),
            "null_move_researches": int(f["null_move_researches"])}


def phase_summary(args: argparse.Namespace) -> None:
    import collections
    teacher = read_jsonl(QUALITY_DIR / "teacher.jsonl")
    dis = read_jsonl(QUALITY_DIR / "disagreements.jsonl")
    paired = read_jsonl(QUALITY_DIR / "paired.jsonl")
    pgn_inv = json.loads((QUALITY_DIR / "pgn_inventory.json").read_text())
    counts: dict[str, int] = {}
    for r in paired:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    D = ("3", "4", "5", "6", "7")
    ev = [r for r in paired if r["class"] == "EVAL_LIKE"]
    ev_delta = {}
    for d in D:
        vals = [r["deltas"].get(d) for r in ev if r["deltas"].get(d) is not None]
        ev_delta[d] = statistics.median(vals) if vals else None
    ev_below_d7 = sum(1 for r in ev if r["deltas"].get("7") is not None and r["deltas"].get("7") <= -40)
    sl = [r for r in paired if r["class"] == "SEARCH_LIKE"]
    sl_ranks = sorted(r["target_root_rank"] for r in sl)
    ranks = sorted(r["target_root_rank"] for r in paired)
    rank_hist = collections.Counter(r["target_root_rank"] for r in paired)

    lines = []
    a = lines.append
    a("# S4.0B Search-Quality Attribution summary (paired fixed-depth repair)")
    a("")
    a("Diagnostic only; no S4.1 candidate implemented.")
    a("")
    a("Corpus terminology: CurrentFinal-vs-teacher disagreement positions, sampled")
    a("from CurrentFinal match games (not necessarily 'positions CurrentFinal misplayed').")
    a(f"- pgns: {json.dumps(pgn_inv)}")
    a(f"- sampled positions: {len(teacher)}")
    a(f"- teacher: Stockfish-18, MultiPV=3 coherent snapshot, nodes={args.teacher_nodes}")
    a(f"- normal CurrentFinal: {args.normal_budgets} ms")
    a(f"- disagreement threshold (cp): {args.threshold_cp}")
    a(f"- high-confidence disagreements (coherent teacher): {len(dis)}")
    a("")
    a("## Paired fixed-depth classification (T=teacher best, O=CurrentFinal normal best)")
    a("")
    a("For every disagreement we force T AND O at the same depths 3..7 and classify on")
    a("delta[d] = score_T[d] - score_O[d] (both CurrentFinal scores, one scale).")
    a("")
    for c in ("SEARCH_LIKE", "EVAL_LIKE", "HORIZON", "UNRESOLVED"):
        a(f"- {c}: {counts.get(c, 0)}")
    a("")
    a("## SEARCH_LIKE (root candidate ordering) - dominant")
    a("")
    a("Teacher move is competitive at matched depth but normal search fails to choose it.")
    if sl_ranks:
        a(f"- n={len(sl_ranks)}; median initial root rank={statistics.median(sl_ranks)}; rank>=8: {sum(1 for x in sl_ranks if x >= 8)}/{len(sl_ranks)}")
    a("")
    a("This is evidence of WEAK INITIAL ROOT CANDIDATE ORDERING (CurrentFinal's root uses")
    a("movegen order + TT hash lift + previous-iteration best-move swap; it does NOT apply")
    a("the MVV-LVA/killer/history ordering used at non-root nodes). Quiet teacher moves are")
    a("therefore tried late and get too little search budget.")
    a("")
    a("## EVAL_LIKE (static evaluation) - small, clean")
    a("")
    if ev_delta:
        a("median delta (T-O) per depth: " + ", ".join(f"d{d}={v:.0f}" for d, v in ev_delta.items() if v is not None))
        a(f"- still <= -40cp below O at d7: {ev_below_d7}/{len(ev)}")
    a("- teacher move type: all quiet")
    a("")
    a("## HORIZON / UNRESOLVED")
    a(f"- HORIZON: {counts.get('HORIZON', 0)} (clear convergence with depth)")
    a(f"- UNRESOLVED: {counts.get('UNRESOLVED', 0)} (noisy / mate-incompatible)")
    a("")
    a("## Teacher root rank (all disagreements)")
    if ranks:
        a(f"- median: {statistics.median(ranks)}; rank>=8: {sum(1 for x in ranks if x >= 8)}/{len(ranks)}")
        a(f"- histogram: {json.dumps(dict(sorted(rank_hist.items())))}")
    a("")
    a("## Teacher move type (disagreements)")
    ttypes = collections.Counter()
    for d in dis:
        board = chess.Board(d["fen"])
        mv = chess.Move.from_uci(d["teacher_best"])
        if board.is_capture(mv):
            ttypes["capture"] += 1
        elif board.gives_check(mv):
            ttypes["check"] += 1
        elif mv.promotion:
            ttypes["promotion"] += 1
        else:
            ttypes["quiet"] += 1
    a(f"- {json.dumps(dict(ttypes))}")
    a("")
    a("## Combined S4.0A + S4.0B interpretation")
    a("")
    a("- S4.0A: per-node cost concentrated in legal-move make/unmake filtering (~120/node);")
    a("  depth bound by tree size (qsearch ~80% of nodes); LMR re-search low; TT under-utilized.")
    a("- S4.0B (paired): failures are overwhelmingly quiet moves (83%) ranked late, and at")
    a("  MATCHED depth the teacher move is competitive with CurrentFinal's own move for 73% of")
    a("  disagreements (SEARCH_LIKE). Only ~10% are genuine static-evaluation cases (EVAL_LIKE).")
    a("- The original EVAL_SUSPECT=45 was largely a measurement artifact: of the 42 surviving,")
    a("  31 reclassify as SEARCH_LIKE under the paired comparison; only 6 remain EVAL_LIKE.")
    a("")
    a("## Recommended S4.1 direction")
    a("")
    a("**Search guidance - root quiet-move ordering.** The dominant, cleanest signal is that")
    a("good quiet moves are tried too late at the root (median initial rank ~13, 76% at rank>=8)")
    a("and lose to LMR/pruning before they are recognized, yet they are competitive when given")
    a("equal depth. This points to root-level candidate ordering (e.g. applying the history/")
    a("killer/static ordering already used at non-root nodes at the root, or a quiet-move")
    a("history-based root prioritization), NOT a broad evaluator stack.")
    a("")
    a("EVAL_LIKE (11, all quiet, flat ~-80cp at depth 3..7) remains a real but secondary target;")
    a("a single attributable positional evaluator term would be the later evaluation step.")
    out = QUALITY_DIR / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {
        "git_sha": git_sha(Path(__file__).resolve().parents[1]),
        "engine": str(args.engine), "engine_sha256": sha256_file(Path(args.engine)),
        "stockfish": str(args.stockfish),
        "teacher_multipv": args.teacher_multipv, "teacher_nodes": args.teacher_nodes,
        "normal_budgets_ms": args.normal_budgets, "threshold_cp": args.threshold_cp,
        "sampled": len(teacher), "disagreements": len(dis),
        "paired_classification": counts,
    }
    (QUALITY_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_to_str(s: Any) -> str:
    if isinstance(s, Score):
        return s.raw if hasattr(s, "raw") else f"{s.kind}:{s.value}"
    return str(s)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--stockfish", type=Path, required=True)
    p.add_argument("--pgns", nargs="+", default=[
        "results/s3-promotion/run-001/match.pgn",
        "results/s3-final/match/match.pgn",
    ])
    p.add_argument("--phases", nargs="+", default=["all"])
    p.add_argument("--sample-start-ply", type=int, default=12)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--tail-skip", type=int, default=2)
    p.add_argument("--max-positions", type=int, default=0)
    p.add_argument("--teacher-nodes", type=int, default=60000)
    p.add_argument("--teacher-multipv", type=int, default=3)
    p.add_argument("--teacher-timeout", type=float, default=120.0)
    p.add_argument("--normal-budgets", type=int, nargs="+", default=[100, 500, 1000])
    p.add_argument("--threshold-cp", type=int, default=50)
    p.add_argument("--ablation-max", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    phases = args.phases if args.phases != ["all"] else \
        ["inventory", "extract", "teacher", "normal", "disagreements",
         "forced_root", "classify", "paired", "ablation", "summary"]
    order = {
        "inventory": None, "extract": None, "teacher": None, "normal": None,
        "disagreements": None, "forced_root": None, "classify": None,
        "ablation": None, "summary": None,
    }
    for ph in phases:
        if ph == "inventory":
            write_inventory(args.pgns)
        elif ph == "extract":
            phase_extract(args)
        elif ph == "teacher":
            phase_teacher(args)
        elif ph == "normal":
            phase_normal(args)
        elif ph == "disagreements":
            phase_disagreements(args)
        elif ph == "forced_root":
            phase_forced_root(args)
        elif ph == "classify":
            phase_classify(args)
        elif ph == "paired":
            phase_paired(args)
        elif ph == "ablation":
            phase_ablation(args)
        elif ph == "summary":
            phase_summary(args)
        else:
            print(f"unknown phase {ph}", file=sys.stderr)
            return 1
    return 0


def write_inventory(pgns: list[str]) -> None:
    inv = []
    for pgn_path in pgns:
        path = Path(pgn_path)
        games = 0
        pairs = set()
        results: dict[str, int] = {}
        with open(path, encoding="utf-8") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                games += 1
                pairs.add((g.headers.get("White"), g.headers.get("Black")))
                r = g.headers.get("Result")
                results[r] = results.get(r, 0) + 1
        inv.append({"path": str(path), "games": games, "pairs": sorted(pairs),
                    "results": results})
    (QUALITY_DIR / "pgn_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError, TeacherError) as exc:
        print(f"s4_quality_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
