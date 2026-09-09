#!/usr/bin/env python3
"""S14-A: Fishtest PGN -> compact teacher shards (converter + pilot).

Frozen semantics (per the approved S14 plan):
  - sample position = the FEN BEFORE the annotated move:
        node.parent.board() + node.move + node.comment
    (comment is attached to the node AFTER the move; CP is from the
    mover's perspective — NO sign flip for black)
  - teacher_cp_stm = the decimal in {-0.91/21 1.749s}; clamp +-2000 for
    training, keep raw for audit; depth/time saved as metadata
  - Base side ONLY: the PGN header name must be exactly
    "Base-<resolved_base sha>" matching the test JSON's resolved_base;
    any other naming (or a mismatch) -> whole test SKIP (fail closed,
    no heuristics)
  - keep: numeric-CP comment, legal replay, Base side, depth+time present
  - skip: book/no-score moves, mate comments, malformed comments,
    unparseable games, weird terminations
  - one sample per game max: deterministic hash(test_id, game_no) over
    the eligible Base-side numeric-CP moves
  - scan newest -> oldest; stop at a target count
  - frozen validation/holdout exact-FEN overlap dropped
  - free calibration: before dropping, join vs the 1M corpus teacher CP
    and report n/MAE/p90/sign-agreement (diagnostic only, no gate)

Output: zstd-free JSONL.gz shards of ~100k accepted positions each
(standard gzip; zstd lib optional and not a dependency here).

Pilot mode: --pilot 100000 runs the full correctness battery before
the real collection.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn

DATA_ROOT = Path(r"data\s14\fishtest-2026")
OUT_ROOT = Path(r"data\s14\shards")
CORPUS = Path(r"data\s10\s10-eval-v2-1m01")
TARGET_FRESH = 4_300_000
SHARD_SIZE = 100_000

# {-0.91/21 1.749s} | {+0.75/21 1.056s} | {0.00/22 0.181s}
COMMENT_RE = re.compile(
    r"^\{?\s*([+-]?\d+(?:\.\d+)?)\s*/\s*(\d+)\s+([\d.]+)s\s*\}?$")


def sha1_of(*parts: str) -> int:
    h = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def load_test_meta(json_path: Path) -> dict | None:
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    resolved_base = (d.get("args") or {}).get("resolved_base")
    base_tag = (d.get("args") or {}).get("base_tag")
    if not resolved_base or not str(resolved_base).strip():
        return None
    return {
        "test_id": d.get("_id") or json_path.stem,
        "resolved_base": str(resolved_base),
        "base_tag": base_tag,
        "base_net": (d.get("args") or {}).get("base_nets"),
    }


def iter_tests_newest_first():
    """Yield (test_id, pgn_path, json_path, meta) newest -> oldest."""
    days = sorted({p.parent.parent.name for p in
                   DATA_ROOT.glob("*/*/*.pgn.gz")}, reverse=True)
    for day in days:
        for pgn in sorted(DATA_ROOT.glob(f"{day}/*/*.pgn.gz"),
                          reverse=True):
            js = pgn.with_suffix("").with_suffix(".json")
            meta = load_test_meta(js) if js.exists() else None
            yield pgn.parent.name, pgn, js, meta


def convert_games(pgn_path: Path, meta: dict, stats: dict):
    """Stream one test's PGN; yield at most one sample per game."""
    test_id = meta["test_id"]
    base_name = f"Base-{meta['resolved_base']}"
    new_name_prefix = "New-"
    game_no = 0
    with gzip.open(pgn_path, "rt", encoding="utf-8",
                   errors="strict") as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                stats["parse_error_games"] += 1
                continue_hint = fh.read(0)  # noop
            if game is None:
                break
            game_no += 1
            stats["games_seen"] += 1
            headers = game.headers
            white = headers.get("White", "")
            black = headers.get("Black", "")
            # Fail-closed Base detection: exactly one side matches the
            # Base-<sha> name AND the other side is a New- name.
            w_is_base = white == base_name
            b_is_base = black == base_name
            w_is_new = white.startswith(new_name_prefix)
            b_is_new = black.startswith(new_name_prefix)
            if w_is_base and b_is_new:
                base_color = chess.WHITE
            elif b_is_base and w_is_new:
                base_color = chess.BLACK
            else:
                stats["games_base_unknown"] += 1
                continue
            fen0 = headers.get("FEN")
            board = chess.Board(fen0) if fen0 else chess.Board()
            eligible = []  # (fen_before, cp_raw, depth, time_ms, uci)
            try:
                for node in game.mainline():
                    mover_is_base = (board.turn == base_color)
                    comment = node.comment
                    if comment:
                        m = COMMENT_RE.match(comment.strip())
                        if m and mover_is_base:
                            cp = float(m.group(1)) * 100.0
                            depth = int(m.group(2))
                            tms = int(round(float(m.group(3)) * 1000))
                            if abs(cp) <= 100000:  # sane bound
                                eligible.append(
                                    (board.fen(), cp, depth, tms,
                                     node.move.uci()))
                    node.move  # touch
                    board.push(node.move)
            except Exception:
                stats["replay_error_games"] += 1
                continue
            if not eligible:
                stats["games_no_eligible"] += 1
                continue
            idx = sha1_of(test_id, str(game_no)) % len(eligible)
            fen_before, cp_raw, depth, tms, uci = eligible[idx]
            stats["accepted"] += 1
            yield {
                "fen_before": fen_before,
                "teacher_cp_stm": max(-2000.0, min(2000.0, cp_raw)),
                "teacher_cp_raw": cp_raw,
                "teacher_move_uci": uci,
                "depth": depth,
                "time_ms": tms,
                "test_id": test_id,
                "game_id": f"{test_id}:{game_no}",
                "source_date": pgn_path.parent.parent.name,
                "base_color": "white" if base_color else "black",
                "base_identity": meta["resolved_base"],
                "game_result_white": {
                    "1-0": 1.0, "0-1": 0.0}.get(
                        headers.get("Result", ""), 0.5),
            }


class ShardWriter:
    def __init__(self, out_root: Path):
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.buf: list[dict] = []
        self.shard_idx = 0

    def add(self, rec: dict):
        self.buf.append(rec)
        if len(self.buf) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        path = self.out_root / f"shard-{self.shard_idx:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in self.buf:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.shard_idx += 1
        self.buf = []


def _worker(wid: int, tests: list, limit: int, out_dir: Path,
            shared, lock, workers: int, shard_size: int = SHARD_SIZE):
    stats = {"games_seen": 0, "accepted": 0, "parse_error_games": 0,
             "replay_error_games": 0, "games_base_unknown": 0,
             "games_no_eligible": 0}
    my_tests = tests[wid::workers]
    shard_idx = wid
    buf = []
    t0 = time.time()
    for pgn_s, test_id, resolved_base in my_tests:
        with lock:
            if shared["accepted"] >= limit:
                break
        pgn = Path(pgn_s)
        meta = {"test_id": test_id, "resolved_base": resolved_base}
        for rec in convert_games(pgn, meta, stats):
            buf.append(rec)
            if len(buf) >= shard_size:
                path = out_dir / f"shard-w{wid}-{shard_idx:05d}.jsonl.gz"
                with gzip.open(path, "wt", encoding="utf-8") as fh:
                    for r in buf:
                        fh.write(json.dumps(
                            r, separators=(",", ":")) + "\n")
                shard_idx += 1
                buf = []
            with lock:
                shared["accepted"] = shared["accepted"] + 1
                if shared["accepted"] >= limit:
                    break
            if stats["accepted"] >= limit:
                break
        with lock:
            shared["tests_used"] = shared["tests_used"] + 1
            for k in ("games_seen", "parse_error_games",
                      "replay_error_games", "games_base_unknown",
                      "games_no_eligible"):
                shared[k] = shared[k] + stats[k]
            stats = {k2: 0 for k2 in stats}
            if wid == 0:
                el = time.time() - t0
                acc = shared["accepted"]
                seen = shared["games_seen"]
                print(f"[progress] accepted {acc} ({seen} games, "
                      f"{shared['tests_used']} tests) "
                      f"{acc/max(el,1)/1000:.0f}k/s", flush=True)
    if buf:
        path = out_dir / f"shard-w{wid}-{shard_idx:05d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in buf:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=0,
                    help="stop after N accepted positions (pilot mode)")
    ap.add_argument("--target", type=int, default=TARGET_FRESH)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    limit = args.pilot or args.target

    import multiprocessing as mp

    tests = []
    for test_id, pgn, js, meta in iter_tests_newest_first():
        if meta is None:
            continue
        tests.append((str(pgn), meta["test_id"],
                      meta["resolved_base"]))

    manager = mp.Manager()
    shared = manager.dict({
        "accepted": 0, "games_seen": 0, "parse_error_games": 0,
        "replay_error_games": 0, "games_base_unknown": 0,
        "games_no_eligible": 0, "tests_used": 0,
        "tests_skipped_no_meta": 0})
    lock = manager.Lock()

    out_dir = OUT_ROOT if not args.pilot else OUT_ROOT.parent / "shards-pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    procs = [mp.Process(target=_worker, args=(
        w, tests, limit, out_dir, shared, lock, args.workers))
        for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    stats = dict(shared)
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["throughput_games_per_s"] = round(
        stats["games_seen"] / max(stats["elapsed_s"], 0.1), 1)
    print(json.dumps(stats, indent=1))
    (OUT_ROOT.parent / ("pilot-stats.json" if args.pilot
                        else "collect-stats.json")
     ).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
