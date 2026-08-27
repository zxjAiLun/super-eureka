"""Unified game fingerprinting and canonical identity for source datasets.

Follows the frozen S6 lichess_select.py canonical JSON serialization contract:
- initial_fen (standard startpos when FEN header missing)
- result (1-0, 0-1, 1/2-1/2, or empty string)
- moves (mainline moves in UCI)
- json.dumps(..., sort_keys=True, ensure_ascii=False, separators=(",", ":"))
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chess
import chess.pgn

FINGERPRINT_FIELDS = ("initial_fen", "moves", "result")


def game_fingerprint(game: chess.pgn.Game) -> str:
    """Deterministic canonical identity of a game: sha256(canonical JSON)."""
    headers = game.headers
    payload = {
        "initial_fen": headers.get("FEN") or chess.STARTING_FEN,
        "result": headers.get("Result", ""),
        "moves": [move.uci() for move in game.mainline_moves()],
    }
    if set(payload) != set(FINGERPRINT_FIELDS):
        raise SystemExit("FAIL CLOSED: fingerprint field set drifted")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fingerprint_set_sha256(fingerprints: set[str] | list[str]) -> str:
    """Order-independent SHA-256 of a fingerprint set (sorted, newline joined)."""
    return hashlib.sha256("\n".join(sorted(set(fingerprints))).encode("utf-8")).hexdigest()


def load_pgn_fingerprints(path: Path) -> set[str]:
    """Extract canonical fingerprints for all games in a PGN file."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"FAIL CLOSED: exclusion PGN not found: {path}")
    fingerprints = set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            fingerprints.add(game_fingerprint(game))
    return fingerprints
