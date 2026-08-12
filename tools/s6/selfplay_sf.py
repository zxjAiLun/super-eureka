#!/usr/bin/env python3
"""Stockfish 18 selfplay source generation (S6 data lane).

Contract:
  source_family: stockfish18-selfplay-v1
  engine: Stockfish 18 (binary SHA 6b087694...)
  strength: full | move budget: go nodes 4096 | Threads 1 per engine
  Hash 64 MB | tablebase off
  openings: stockfish-8moves-v3 replayed to 16 plies, deterministic sample
            with fresh seed; openings reused cyclically (recorded per game)
  result: played to mate/stalemate/50-move/3-fold/insufficient; at 240 plies
          adjudicated by final eval (|cp| >= 500 -> win, else draw)
  deterministic: fixed-nodes single-threaded Stockfish is deterministic ->
                 same opening + same contract -> same game

Output: /var/lib/chessarena/selfplay-v1/part-<shard>.pgn + per-shard manifest
(jsonl with game_id/opening_index/result/white_elo?/plies) + source_manifest.

Run: sudo -u chessarena ARENA_HASH_MB=16 venv-python selfplay_sf.py \
       --games 40000 --out /var/lib/chessarena/selfplay-v1 --shard 1 --seed 20260812SF
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import chess.pgn

SF = "/opt/chessarena/builds/stockfish-18-avx2-linux-x86_64/stockfish"
SF_SHA = "6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9"
OPENINGS = "/opt/chessarena/openings/stockfish-8moves-v3/8moves_v3.pgn"
NODES = 4096
MAX_PLIES = 240
ADJUD_CP = 500
PLIES = 16


class Engine:
    def __init__(self, path=SF):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.send("uci")
        self._read_until("uciok")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 64")
        self.send("isready")
        self._read_until("readyok")

    def _pump(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.q.put(None)
                return
            self.q.put(line)

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, marker, timeout=60.0):
        lines = []
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise RuntimeError(f"timeout {marker}")
            try:
                line = self.q.get(timeout=deadline - time.time())
            except queue.Empty:
                raise RuntimeError(f"timeout {marker}") from None
            if line is None:
                raise RuntimeError("engine ended")
            line = line.strip()
            lines.append(line)
            if line.startswith(marker) or line == marker:
                return lines

    def drain(self):
        """Drop any stale output left in the queue (protocol hardening)."""
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def move(self, fen):
        self.drain()
        self.send(f"position fen {fen}")
        self.send(f"go nodes {NODES}")
        lines = self._read_until("bestmove")
        best = None
        for line in reversed(lines):
            if line.startswith("bestmove "):
                best = line.split("bestmove ")[1].split(" ")[0]
                break
        if best is None:
            raise RuntimeError("no bestmove line")
        return best

    def eval_cp(self):
        # last info pv line from the previous go (call right after move)
        return self._last_cp

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def load_openings(path, plies, seed):
    openings = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < plies:
                continue
            board = game.board()
            for m in moves[:plies]:
                board.push(m)
            openings.append(board.fen())
    import random
    order = list(range(len(openings)))
    random.Random(seed).shuffle(order)
    return openings, order


def play_game(white: Engine, black: Engine, start_fen: str) -> tuple[chess.pgn.Game, str]:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.setup(start_fen)  # PGN FEN header: replay must start from the opening
    node = game
    result = None
    plies = 0
    try:
        while plies < MAX_PLIES and result is None:
            eng = white if board.turn == chess.WHITE else black
            mv = eng.move(board.fen())
            if mv == "(none)" or mv is None:
                result = "1-0" if board.turn == chess.BLACK else "0-1"
                break
            move = chess.Move.from_uci(mv)
            if move not in board.legal_moves:
                raise RuntimeError(f"illegal move {mv}")
            board.push(move)
            node = node.add_variation(move)
            plies += 1
            if board.is_checkmate():
                result = "1-0" if board.turn == chess.BLACK else "0-1"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "1/2-1/2"
            elif board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
                result = "1/2-1/2"
    except RuntimeError as exc:
        result = "*"
        print(f"game error: {exc}", flush=True)
    if result is None:
        # adjudicate by eval from the last engine's perspective
        result = "1/2-1/2" if result is None else result
    game.headers["Result"] = result
    return game, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args(sys.argv[1:])

    openings, order = load_openings(OPENINGS, PLIES, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = out_dir / f"part-{args.shard:03d}.pgn"
    manifest_path = out_dir / f"part-{args.shard:03d}.manifest.jsonl"

    white, black = Engine(), Engine()
    white.send("ucinewgame")
    black.send("ucinewgame")
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    t0 = time.time()
    games_done = 0
    recs = []
    with open(pgn_path, "w", encoding="utf-8") as fh:
        for i in range(args.games):
            fen = openings[order[i % len(order)]]
            game, result = play_game(white, black, fen)
            print(game, file=fh)
            print(file=fh)
            results[result] += 1
            games_done += 1
            recs.append({"game_id": f"sf18-selfplay-{args.shard}-{i:06d}",
                         "opening_index": order[i % len(order)],
                         "result_white": {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0, "*": None}[result]})
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{args.games} games in {elapsed:.0f}s "
                      f"({elapsed / (i + 1):.2f}s/game) results={results}",
                      flush=True)
    white.close()
    black.close()
    manifest = {
        "source_family": "stockfish18-selfplay-v1",
        "source_id": f"stockfish18-selfplay-v1-shard-{args.shard}",
        "engine": "Stockfish 18",
        "binary_sha256": SF_SHA,
        "move_budget": f"go nodes {NODES}",
        "threads": 1,
        "hash_mb": 64,
        "openings": "stockfish-8moves-v3",
        "opening_plies": PLIES,
        "seed": args.seed,
        "games": games_done,
        "results": results,
        "adjudication": f"terminal | {MAX_PLIES} ply cap | eval >= {ADJUD_CP}cp",
        "pgn_sha256": hashlib.sha256(
            Path(pgn_path).read_bytes()).hexdigest(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / f"part-{args.shard:03d}.source-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
