#!/usr/bin/env python3
"""Build the S7.0 depth-attribution corpus (80 positions, 3 strata).

Stratum A1 (30): the S4.0A compute-attribution corpus
    tools/data/s4_compute_positions.epd (7 classes, already validated).

Stratum A2 (30): real middlegames extracted deterministically from Arena
    self-play PGNs (results/s3-promotion/run-001/match.pgn and
    results/s3-final/match/match.pgn). Filters: ply 20-80, nonterminal,
    not in check, queen-on-board majority, castled/uncastled mix, dedup on
    canonical FEN4.

Stratum A3 (20): tactical / king-danger positions from the teacher
    disagreement corpus (data/s6/s6-teacher-challenge-v1.jsonl), selected
    deterministically across the |teacher cp| range (no hand design).

Deterministic: fixed source order + fixed tie-break keys; no RNG.

Writes tools/data/s7_depth_attribution_corpus.jsonl and prints its SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chess
import chess.pgn

REPO = Path(__file__).resolve().parents[1]
EPD = REPO / "tools/data/s4_compute_positions.epd"
PGNS = [
    REPO / "results/s3-promotion/run-001/match.pgn",
    REPO / "results/s3-final/match/match.pgn",
]
TEACHER = REPO / "data/s6/s6-teacher-challenge-v1.jsonl"
OUT = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"

N_A1 = 30
N_A2 = 30
N_A3 = 20
PLY_MIN = 20
PLY_MAX = 80


def fen4(fen: str) -> str:
    return " ".join(fen.split()[:4])


def material(board: chess.Board) -> dict:
    pieces = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0, "K": 0}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            pieces[p.symbol().upper()] += 1
    return pieces


def king_castled(board: chess.Board, color: chess.Color) -> bool:
    k = board.king(color)
    if k is None:
        return False
    return chess.square_file(k) in (0, 1, 2, 6, 7)  # moved off e-file proxy


def classify(board: chess.Board) -> dict:
    mat = material(board)
    white_castled = king_castled(board, chess.WHITE)
    black_castled = king_castled(board, chess.BLACK)
    if white_castled and black_castled:
        castled = "both"
    elif white_castled or black_castled:
        castled = "one"
    else:
        castled = "none"
    total_pawns = mat["P"]
    queens = mat["Q"]
    return {
        "queens": queens,
        "queen_on_board": queens >= 1,
        "castled": castled,
        "pawns": total_pawns,
        "open": total_pawns <= 14,
    }


def load_a1() -> list[dict]:
    out = []
    for line in EPD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen, rest = line.split(";", 1)
        fen = fen.strip()
        meta = dict(
            (kv.split("=", 1)[0].strip(), kv.split("=", 1)[1].strip())
            for kv in rest.split(";")
            if "=" in kv
        )
        board = chess.Board(fen)
        if board.is_game_over():
            raise SystemExit(f"A1 terminal FEN: {fen}")
        out.append({
            "id": f"s7_s4_{meta['id']}",
            "stratum": "s4",
            "class": meta["class"],
            "fen": fen,
            "source": "tools/data/s4_compute_positions.epd",
        })
    assert len(out) == N_A1, f"A1 expected {N_A1}, got {len(out)}"
    return out


def load_a2_candidates() -> list[dict]:
    seen: set[str] = set()
    candidates: list[dict] = []
    for pgn_path in PGNS:
        with open(pgn_path, encoding="utf-8") as fh:
            game = chess.pgn.read_game(fh)
            while game:
                board = game.board()
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    board.push(move)
                    if ply < PLY_MIN or ply > PLY_MAX:
                        continue
                    if board.is_game_over():
                        continue
                    if board.is_check():
                        continue
                    if board.is_fifty_moves():
                        continue
                    fen = board.fen()
                    k = fen4(fen)
                    if k in seen:
                        continue
                    seen.add(k)
                    feat = classify(board)
                    candidates.append({
                        "fen": fen,
                        "ply": ply,
                        "queens": feat["queens"],
                        "queen_on_board": feat["queen_on_board"],
                        "castled": feat["castled"],
                        "pawns": feat["pawns"],
                        "open": feat["open"],
                        "source": str(pgn_path.relative_to(REPO)),
                    })
                game = chess.pgn.read_game(fh)
    return candidates


def select_a2(candidates: list[dict], n: int) -> list[dict]:
    # queen-on-board majority: ~2/3 with queens, ~1/3 queenless.
    with_q = [c for c in candidates if c["queen_on_board"]]
    no_q = [c for c in candidates if not c["queen_on_board"]]
    with_q.sort(key=lambda c: (c["ply"], c["fen"]))
    no_q.sort(key=lambda c: (c["ply"], c["fen"]))

    n_queenless = max(1, round(n * 0.33))
    n_with = n - n_queenless

    # castled/uncastled diversity within the queen-on-board group.
    def pick_spread(pool: list[dict], k: int) -> list[dict]:
        if not pool:
            return []
        if k >= len(pool):
            return pool
        idx = [round(i * (len(pool) - 1) / (k - 1)) for i in range(k)]
        return [pool[i] for i in idx]

    chosen = pick_spread(with_q, n_with) + pick_spread(no_q, n_queenless)
    # also ensure a castled/uncastled mix: reorder so both appear when possible
    chosen.sort(key=lambda c: (c["ply"], c["fen"]))
    assert len(chosen) == n, f"A2 expected {n}, got {len(chosen)}"
    return chosen


def load_a3() -> list[dict]:
    rows = [
        json.loads(line)
        for line in TEACHER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid = []
    for r in rows:
        board = chess.Board(r["fen"])
        if board.is_game_over():
            continue
        if board.is_check():
            continue
        if not r.get("teacher_bestmove"):
            continue
        valid.append(r)
    # deterministic spread across tactical sharpness |teacher_cp_stm|.
    valid.sort(key=lambda r: (abs(r.get("teacher_cp_stm") or 0), r["fen"]))
    out = []
    if len(valid) <= N_A3:
        sel = valid
    else:
        idx = [round(i * (len(valid) - 1) / (N_A3 - 1)) for i in range(N_A3)]
        sel = [valid[i] for i in idx]
    for i, r in enumerate(sel, 1):
        out.append({
            "id": f"s7_tac_{i:02d}",
            "stratum": "tactical",
            "class": "tactical",
            "fen": r["fen"],
            "source": "data/s6/s6-teacher-challenge-v1.jsonl",
            "teacher_bestmove": r.get("teacher_bestmove"),
            "teacher_cp_stm": r.get("teacher_cp_stm"),
        })
    return out


def main() -> int:
    a1 = load_a1()
    a2_cand = load_a2_candidates()
    a2 = select_a2(a2_cand, N_A2)
    a3 = load_a3()

    corpus = a1 + [
        {
            "id": f"s7_mid_{i:02d}",
            "stratum": "middlegame",
            "class": "open" if c["open"] else "closed",
            "fen": c["fen"],
            "source": c["source"],
            "ply": c["ply"],
            "queens": c["queens"],
            "castled": c["castled"],
        }
        for i, c in enumerate(a2, 1)
    ] + a3

    assert len(corpus) == 80, f"corpus size {len(corpus)} != 80"
    fens = [c["fen"] for c in corpus]
    assert len({fen4(f) for f in fens}) == 80, "duplicate canonical FEN4 in corpus"

    queens_on_board = sum(
        1 for c in corpus if "queen_on_board" not in c
    ) + sum(1 for c in a2 if c["queen_on_board"])
    mid_queens = sum(1 for c in a2 if c["queen_on_board"])

    lines = [json.dumps(c, ensure_ascii=False) for c in corpus]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    print(f"corpus: {len(corpus)} positions -> {OUT.relative_to(REPO)}")
    print(f"  A1 s4:        {len(a1)}")
    print(f"  A2 middlegame:{len(a2)} (queens-on-board {mid_queens}/{len(a2)})")
    print(f"  A3 tactical:  {len(a3)}")
    print(f"  SHA-256: {sha}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"build_s7_corpus_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
