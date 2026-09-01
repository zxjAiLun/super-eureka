"""S10-E2-W0: independent cross-platform qualification corpus.

Deterministic 2048-position corpus (seed 2026090101) generated from
startpos random legal playouts — completely independent of any training
dataset. Phase quotas mirror the B1 stratification and the material
balance is spread across buckets so the corpus exercises both quiet and
imbalanced positions.

Eligibility (same as the dataset builder):
    board.is_valid(), non-terminal, STM not in check, has legal move.

Output: results/s10/s10-e2-w0-corpus.jsonl (one FEN per line, with
phase + material metadata) + corpus SHA in the report.

Usage:
    python tools/s10/e2_w0_corpus.py --out results/s10/s10-e2-w0-corpus.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.train_nnue import CANONICAL_PIECE_CP  # noqa: E402

CORPUS_ID = "s10-e2-w0-crossplatform-2048"
SEED = 2026090101
N = 2048
PHASE_QUOTA = {"high": 512, "mid": 768, "low": 512, "zero": 256}
# target material-bucket spread per phase (soft targets; final spread is
# reported honestly, not forced)
MAT_BUCKETS = ["<=50", "51-250", "251-500", ">500"]

PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2,
                 chess.QUEEN: 4}


def phase_of(board: chess.Board) -> int:
    return sum(PHASE_WEIGHTS[pt] for pt in (chess.KNIGHT, chess.BISHOP,
                                            chess.ROOK, chess.QUEEN)
               for _ in board.pieces(pt, chess.WHITE)) + \
           sum(PHASE_WEIGHTS[pt] for pt in (chess.KNIGHT, chess.BISHOP,
                                            chess.ROOK, chess.QUEEN)
               for _ in board.pieces(pt, chess.BLACK))


def bucket_of(phase: int) -> str:
    if phase >= 18:
        return "high"
    if phase >= 8:
        return "mid"
    if phase >= 1:
        return "low"
    return "zero"


def material_abs(board: chess.Board) -> int:
    w = b = 0
    for sq, pc in board.piece_map().items():
        v = CANONICAL_PIECE_CP.get(pc.symbol().lower(), 0)
        if pc.color == chess.WHITE:
            w += v
        else:
            b += v
    return abs(w - b)


def mat_bucket(m: int) -> str:
    if m <= 50:
        return "<=50"
    if m <= 250:
        return "51-250"
    if m <= 500:
        return "251-500"
    return ">500"


def eligible(board: chess.Board) -> bool:
    if not board.is_valid():
        return False
    if board.is_game_over(claim_draw=False):
        return False
    if board.is_check():
        return False
    if not any(board.legal_moves):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(SEED)
    # playout PRNG: seeded separately so corpus generation is stable even
    # if the acceptance loop changes
    prng = random.Random(SEED ^ 0x5EED)

    filled = {k: [] for k in PHASE_QUOTA}
    playouts = 0
    while sum(len(v) for v in filled.values()) < N:
        playouts += 1
        board = chess.Board()
        max_plies = 220
        for _ply in range(max_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(prng.choice(moves))
            if _ply >= 10 and eligible(board):
                ph = bucket_of(phase_of(board))
                if len(filled[ph]) < PHASE_QUOTA[ph]:
                    filled[ph].append(board.fen())
                    break
        if playouts > 2_000_000:
            raise SystemExit("FAIL: playout budget exhausted")

    records = []
    for ph in ("high", "mid", "low", "zero"):
        if len(filled[ph]) != PHASE_QUOTA[ph]:
            raise SystemExit(
                f"FAIL: phase {ph} got {len(filled[ph])} != {PHASE_QUOTA[ph]}")
        records.extend(filled[ph])

    # deterministically interleave phases so the corpus is not phase-blocked
    rng.shuffle(records)

    lines = []
    phase_counts = {k: 0 for k in PHASE_QUOTA}
    mat_counts = {k: 0 for k in MAT_BUCKETS}
    for fen in records:
        b = chess.Board(fen)
        ph = bucket_of(phase_of(b))
        mb = mat_bucket(material_abs(b))
        phase_counts[ph] += 1
        mat_counts[mb] += 1
        lines.append(json.dumps({
            "fen": fen,
            "phase_bucket": ph,
            "phase": phase_of(b),
            "material_bucket": mb,
        }, sort_keys=True))

    text = "\n".join(lines) + "\n"
    corpus_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    report = {
        "corpus_id": CORPUS_ID,
        "seed": SEED,
        "n": N,
        "playouts": playouts,
        "phase_counts": phase_counts,
        "material_counts": mat_counts,
        "corpus_sha256": corpus_sha,
        "out": str(args.out),
    }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
