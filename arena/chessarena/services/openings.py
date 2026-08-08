"""Opening resolution for tournaments (P4.F1 Phase C).

Supports legacy single-line EPD sets and PGN books (official Stockfish
suites): a PGN book entry is replayed to a chosen number of plies with
python-chess to produce the starting FEN.  Selection is deterministic for a
frozen ``opening_seed`` — random sampling without replacement from the
eligible pool (entries with at least the requested plies).
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.pgn

from .cutechess import CutechessLaunchError


def _format(opening_set) -> str:
    return (opening_set.manifest or {}).get("format") or opening_set.format


def _epd_fen_for_index(opening_set, opening_index: int) -> str:
    lines = [
        ln.strip()
        for ln in Path(opening_set.file_path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if opening_index >= len(lines):
        raise CutechessLaunchError(
            f"opening_index {opening_index} out of range ({len(lines)} lines)"
        )
    return lines[opening_index].split(";")[0].strip()


def _iter_games(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                return
            yield game


def _game_fen(game, plies: int) -> str:
    moves = list(game.mainline_moves())
    if len(moves) < plies:
        raise CutechessLaunchError(
            f"opening shorter than requested {plies} plies ({len(moves)} available)"
        )
    board = game.board()
    for move in moves[:plies]:
        board.push(move)
    return board.fen()


def _pgn_fen_for_index(opening_set, opening_index: int, plies: int) -> str:
    for i, game in enumerate(_iter_games(Path(opening_set.file_path))):
        if i == opening_index:
            return _game_fen(game, plies)
    raise CutechessLaunchError(
        f"opening_index {opening_index} out of range"
    )


def opening_fen_for_index(
    opening_set, opening_index: int, plies: int | None = None
) -> str:
    """The starting FEN for one opening entry (EPD line or replayed PGN)."""
    if _format(opening_set) == "pgn":
        if plies is None:
            raise CutechessLaunchError("plies required for PGN opening sets")
        return _pgn_fen_for_index(opening_set, opening_index, plies)
    return _epd_fen_for_index(opening_set, opening_index)


def eligible_openings(opening_set, plies: int | None = None) -> list[int]:
    """Indices of openings with at least ``plies`` moves (all entries for
    EPD sets, which have no depth notion)."""
    if _format(opening_set) == "pgn":
        eligible: list[int] = []
        for i, game in enumerate(_iter_games(Path(opening_set.file_path))):
            if len(list(game.mainline_moves())) >= (plies or 0):
                eligible.append(i)
        return eligible
    return list(range(opening_set.position_count))


def select_opening_indices(
    opening_set, count: int, plies: int | None, seed: int
) -> list[int]:
    """Deterministic sample without replacement from the eligible pool.

    The same (opening_set, plies, seed, count) always yields the same
    indices; indices are stable across runs for a frozen snapshot.
    """
    pool = eligible_openings(opening_set, plies)
    if len(pool) < count:
        raise CutechessLaunchError(
            f"opening book has only {len(pool)} eligible openings "
            f"(requested {plies} plies); need {count}"
        )
    return random.Random(seed).sample(pool, count)
