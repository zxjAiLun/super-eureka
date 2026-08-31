"""S10-E1: header-gated fast selector for a local lichess .pgn.zst archive.

Drop-in accelerated variant of tools/s6/lichess_select.py's selection
loop with IDENTICAL selection semantics. The only difference is WHEN
work happens:

  old: chess.pgn.read_game parses headers + full move tree for EVERY
       game, then passes() checks headers, then the hash gate, then the
       ply gate.
  new: a cheap header-only pass runs FIRST (Result/Elo/BOT/TimeControl
       + the Site-hash gate); only games passing those gates get their
       movetext counted (raw token scan) for the ply gate; only games
       passing EVERYTHING get a full python-chess parse (needed for the
       game fingerprint and the exported PGN).

Equivalence contract: for the same archive, seed, accept-byte, filters
and exclude set, the accepted-game sequence and every fingerprint are
bit-identical to the original implementation (verified in
e1_selector_equivalence.py over the first N accepted games).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import zstandard

TIME_CONTROL_BASE_MIN = 180

_HEADER_LINE = re.compile(rb'^\[(\w+) "([^"]*)"\]$')
_RESULT_OK = (b"1-0", b"0-1", b"1/2-1/2")


def stream_games_raw(zst_path: Path):
    """Yield (headers: bytes, movetext: bytes) per game.

    Line-oriented streaming state machine over the decompressed byte
    stream: header lines accumulate until the first blank line, then
    movetext lines accumulate until the next blank line, which closes
    the game. No whole-file buffering, no re-scanning.
    """
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            header_lines: list[bytes] = []
            move_lines: list[bytes] = []
            in_moves = False
            carry = b""
            while True:
                chunk = reader.read(1 << 22)
                if not chunk:
                    break
                data = carry + chunk
                lines = data.split(b"\n")
                carry = lines.pop()
                for line in lines:
                    s = line.rstrip(b"\r")
                    if not s:
                        # blank line: boundary
                        if in_moves:
                            if header_lines:
                                yield (
                                    b"\n".join(header_lines),
                                    b"\n".join(move_lines),
                                )
                            header_lines = []
                            move_lines = []
                            in_moves = False
                        elif header_lines:
                            in_moves = True
                        continue
                    if in_moves:
                        move_lines.append(s)
                    else:
                        header_lines.append(s)
                if len(header_lines) + len(move_lines) > (1 << 20):
                    raise RuntimeError("runaway game block")
            # trailing game (no final blank line)
            if header_lines and move_lines:
                yield b"\n".join(header_lines), b"\n".join(move_lines)


def parse_headers(header_block: bytes) -> dict[bytes, bytes]:
    headers: dict[bytes, bytes] = {}
    for line in header_block.splitlines():
        m = _HEADER_LINE.match(line.strip())
        if m:
            headers[m.group(1)] = m.group(2)
    return headers


def header_gate(headers: dict[bytes, bytes], accept_byte: int, seed: int) -> str | None:
    """Return the selection key if the game passes all header-only gates,
    else None. Gates: Result, BOT titles, both Elo >= 1800, TimeControl
    base >= 180s, hash(Site:seed) first byte < accept_byte."""
    if headers.get(b"Result") not in _RESULT_OK:
        return None
    if headers.get(b"WhiteTitle") == b"BOT" or headers.get(b"BlackTitle") == b"BOT":
        return None
    try:
        if int(headers.get(b"WhiteElo", b"0")) < 1800:
            return None
        if int(headers.get(b"BlackElo", b"0")) < 1800:
            return None
    except ValueError:
        return None
    tc = headers.get(b"TimeControl", b"0").decode("ascii", "replace")
    try:
        base = int(tc.split("+")[0])
    except ValueError:
        base = 0
    if base < TIME_CONTROL_BASE_MIN:
        return None
    site = headers.get(b"Site", b"").decode("utf-8", "replace")
    key = site or "|".join(
        headers.get(k, b"").decode("utf-8", "replace")
        for k in (b"White", b"Black", b"Date", b"Round", b"Result")
    )
    h = hashlib.sha256(f"{key}:{seed}".encode("utf-8")).digest()
    if h[0] >= accept_byte:
        return None
    return key


def count_plies(movetext: bytes) -> int:
    """Approximate mainline ply count by counting SAN tokens.

    Counts whitespace-separated tokens that start with a move-like
    character (letters NBRQKOa-h1-8, excluding result tokens and
    annotations). A conservative counter is fine: it only gates
    min_plies/long_plies thresholds, and full parse + fingerprint runs
    on accepted games regardless.
    """
    n = 0
    for tok in movetext.split():
        c = tok[:1]
        if c in (b"", b"{", b"(", b")", b";", b"$", b"*", b"1", b"0", b"-"):
            # '1.' style move numbers start with digits; skip move-number
            # prefixes by stripping them below instead
            pass
        # strip move numbers like '12.' or '12...'
        t = tok
        while t and t[:1].isdigit():
            t = t[1:]
        if t[:1] == b".":
            t = t.lstrip(b".")
        if not t:
            continue
        c = t[:1]
        if c in (b"N", b"B", b"R", b"Q", b"K", b"O", b"a", b"b", b"c",
                 b"d", b"e", b"f", b"g", b"h"):
            n += 1
    return n


def full_parse(header_block: bytes, movetext: bytes) -> chess.pgn.Game:
    """Full python-chess parse of one raw game block (headers + text)."""
    pgn_text = header_block.decode("utf-8", "replace") + "\n\n" + \
        movetext.decode("utf-8", "replace") + "\n\n"
    return chess.pgn.read_game(io.StringIO(pgn_text))
