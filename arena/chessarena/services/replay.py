"""Public replay helpers: extract a single verified game's PGN.

A pair's ``match.pgn`` holds two games (strict color swap).  Each ``Game``
row records which engines were White/Black via ENGINE_A_NAME/ENGINE_B_NAME,
which matches the PGN White/Black headers, so we can pick the exact game.

Note: python-chess ``read_game`` does not reliably preserve the headers of
the second game in a multi-game file (it falls back to defaults), so the
PGN is split into per-game blocks by their ``[Event`` header and matched by
regex instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Game


class ReplayError(Exception):
    """Raised when a game's PGN cannot be produced for public replay."""


def _split_pgn_blocks(text: str) -> list[str]:
    parts = re.split(r"(?=\[Event )", text)
    return [part.strip() for part in parts if part.strip()]


def _pgn_header(block: str, tag: str) -> str:
    m = re.search(rf"^\[{tag} \"([^\"]*)\"\]", block, re.MULTILINE)
    return m.group(1) if m else ""


def read_single_game_pgn(game: Game) -> str:
    """Return the full PGN text (headers + movetext) of a single verified game.

    Raises ReplayError for unverified games, missing files, or when the
    game's engines cannot be matched inside the pair PGN.
    """
    if not game.verified:
        raise ReplayError("game is not verified")

    path = Path(game.pgn_path)
    if not path.is_file():
        raise ReplayError(f"PGN file missing: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    for block in _split_pgn_blocks(text):
        if (
            _pgn_header(block, "White") == game.white_engine
            and _pgn_header(block, "Black") == game.black_engine
        ):
            return block

    raise ReplayError(
        f"no game with {game.white_engine} vs {game.black_engine} in {path}"
    )
