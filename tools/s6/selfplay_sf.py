#!/usr/bin/env python3
"""Stockfish 18 selfplay source generation - stockfish18-selfplay-v1.

FROZEN contract (v1):
- pre-run FAIL CLOSED: actual SF executable SHA == 6b087694... AND actual
  opening PGN SHA == declared SHA;
- every game: `ucinewgame` + `setoption name Clear Hash` + `isready` on
  BOTH engines (no TT pollution across games);
- every search: `position fen <opening_fen> moves <all continuation moves>`
  then `go nodes 4096` (full history, not just the current FEN);
- play to mate/stalemate/insufficient/50-move/3-fold; at the 240-ply cap
  adjudicate with a REAL search: `go nodes 16384` on the final position,
  score converted to WHITE perspective: mate -> corresponding win,
  cp >= +500 -> 1-0, cp <= -500 -> 0-1, else draw;
- any engine timeout/crash/illegal move -> STOP GENERATION (no "*" games);
- --games <= number of unique openings (no cyclic reuse), FAIL CLOSED;
- deterministic: fixed nodes, single-threaded, per-game cleared state ->
  same seed/contracts -> byte-identical PGN.

Manifest: generator git SHA, engine SHA, opening SHA, seed, unique opening
indices SHA, games, WDL, capped count, adjudicated W/L/D, terminal W/L/D,
error count = 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
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
OPENINGS_SHA = "5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e"
PLAY_NODES = 4096
ADJUD_NODES = 16384
ADJUD_CP = 500
MAX_PLIES = 240
PLIES = 16


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        if self.proc.poll() is not None:
            raise RuntimeError("engine not running")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, marker, timeout=120.0):
        lines = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(f"timeout waiting for {marker}")
            try:
                line = self.q.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError(f"timeout waiting for {marker}") from None
            if line is None:
                raise RuntimeError("engine ended")
            line = line.strip()
            lines.append(line)
            if line.startswith(marker) or line == marker:
                return lines

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def new_game(self):
        self.drain()
        self.send("ucinewgame")
        self.send("setoption name Clear Hash")
        self.send("isready")
        self._read_until("readyok")

    def close(self):
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def search(self, position_cmd: str, nodes: int) -> dict:
        self.send(position_cmd)
        self.send(f"go nodes {nodes}")
        lines = self._read_until("bestmove")
        best = None
        cp = None
        mate = None
        for line in reversed(lines):
            if line.startswith("bestmove ") and best is None:
                best = line.split("bestmove ")[1].split(" ")[0]
        for line in lines:
            if not line.startswith("info") or " pv " not in line:
                continue
            m = re.search(r"score cp (-?\d+)|score mate (-?\d+)", line)
            if m:
                if m.group(1) is not None:
                    cp = int(m.group(1))
                    mate = None
                else:
                    mate = int(m.group(2))
                    cp = None
        return {"bestmove": best, "cp": cp, "mate": mate}


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
            openings.append((board.fen(), [m.uci() for m in moves[:plies]]))
    order = list(range(len(openings)))
    random.Random(seed).shuffle(order)
    return openings, order


def play_game(white: Engine, black: Engine, start_fen, history) -> dict:
    white.new_game()
    black.new_game()
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.setup(start_fen)
    node = game
    played: list[str] = []
    result = None
    capped = False
    for ply in range(MAX_PLIES):
        if result is not None:
            break
        eng = white if board.turn == chess.WHITE else black
        position_cmd = f"position fen {start_fen} moves {' '.join(played)}"
        resp = eng.search(position_cmd, PLAY_NODES)
        mv = resp["bestmove"]
        if mv is None or mv == "(none)":
            result = "1-0" if board.turn == chess.BLACK else "0-1"
            break
        move = chess.Move.from_uci(mv)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal move {mv} in {board.fen()}")
        board.push(move)
        played.append(mv)
        node = node.add_variation(move)
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material():
            result = "1/2-1/2"
        elif board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            result = "1/2-1/2"
    if result is None:
        capped = True
        # REAL adjudication search on the final position, White perspective
        eng = Engine()  # fresh adjudicator instance
        try:
            eng.new_game()
            resp = eng.search(
                f"position fen {board.fen()}", ADJUD_NODES)
            mate = resp["mate"]
            cp = resp["cp"]
            stm_white = board.turn == chess.WHITE
            if mate is not None:
                score = float("inf") if mate > 0 else float("-inf")
                if not stm_white:
                    score = -score
            else:
                score = float(cp or 0)
                if not stm_white:
                    score = -score
            if score == float("inf") or score >= ADJUD_CP:
                result = "1-0"
            elif score == float("-inf") or score <= -ADJUD_CP:
                result = "0-1"
            else:
                result = "1/2-1/2"
        finally:
            eng.close()
    game.headers["Result"] = result
    return {"game": game, "result": result, "capped": capped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args(sys.argv[1:])

    # pre-run FAIL CLOSED identity verification
    actual_sf = sha256_file(SF)
    if actual_sf != SF_SHA:
        print(f"FAIL CLOSED: SF executable SHA {actual_sf} != {SF_SHA}")
        return 3
    actual_op = sha256_file(OPENINGS)
    if actual_op != OPENINGS_SHA:
        print(f"FAIL CLOSED: opening SHA {actual_op} != {OPENINGS_SHA}")
        return 3

    openings, order = load_openings(OPENINGS, PLIES, args.seed)
    if args.games > len(openings):
        print(f"FAIL CLOSED: --games {args.games} > unique openings "
              f"{len(openings)} (no cyclic reuse)")
        return 4
    indices_sha = hashlib.sha256(
        json.dumps(order[: args.games]).encode("utf-8")).hexdigest()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = out_dir / "selfplay-v1.pgn"

    white, black = Engine(), Engine()
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    terminal = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    adjud = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    capped = 0
    errors = 0
    t0 = time.time()
    try:
        with open(pgn_path, "w", encoding="utf-8") as fh:
            for i in range(args.games):
                start_fen, history = openings[order[i]]
                try:
                    rec = play_game(white, black, start_fen, history)
                except RuntimeError as exc:
                    print(f"STOP GENERATION: game {i} error: {exc}")
                    errors += 1
                    return 5
                print(rec["game"], file=fh)
                print(file=fh)
                results[rec["result"]] += 1
                if rec["capped"]:
                    capped += 1
                    adjud[rec["result"]] += 1
                else:
                    terminal[rec["result"]] += 1
                if (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    print(f"  {i + 1}/{args.games} in {elapsed:.0f}s "
                          f"({elapsed / (i + 1):.2f}s/game) {results} capped={capped}",
                          flush=True)
    finally:
        white.close()
        black.close()

    manifest = {
        "source_family": "stockfish18-selfplay-v1",
        "source_id": "stockfish18-selfplay-v1",
        "generator_contract": "selfplay_sf.py v1 (frozen)",
        "engine": "Stockfish 18",
        "binary_sha256": SF_SHA,
        "opening_file": OPENINGS,
        "opening_sha256": OPENINGS_SHA,
        "opening_plies": PLIES,
        "seed": args.seed,
        "unique_opening_indices_sha256": indices_sha,
        "games": args.games,
        "results": results,
        "capped_games": capped,
        "adjudicated": adjud,
        "terminal": terminal,
        "error_count": errors,
        "contract": {
            "play_nodes": PLAY_NODES,
            "adjudication_nodes": ADJUD_NODES,
            "adjudication_cp": ADJUD_CP,
            "max_plies": MAX_PLIES,
            "per_game": "ucinewgame + Clear Hash + isready (both engines)",
            "position": "position fen <opening> moves <history>",
        },
        "pgn_sha256": sha256_file(str(pgn_path)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
