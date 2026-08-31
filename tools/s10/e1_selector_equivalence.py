"""Equivalence check: header-gated fast selector vs the original
lichess_select.py selection loop, over the first N accepted games of the
local 2026-07 archive.

Compares:
  * accepted-game count and order,
  * per-game Site keys,
  * per-game fingerprints (game_fingerprint semantics identical).
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path

import chess.pgn
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.s6.lichess_select import (  # noqa: E402
    game_fingerprint,
    passes,
    select_key,
    time_control_base,
)
from tools.s10.e1_fast_select_lib import (  # noqa: E402
    count_plies,
    full_parse,
    header_gate,
    parse_headers,
    stream_games_raw,
)

ZST = Path(r"E:\ubuntudownload\lichess_db_standard_rated_2026-07.pgn.zst")
SEED = 20260830
ACCEPT_BYTE = 0x05
MIN_PLIES = 40
LONG_PLIES = 100
N = 300


def original_stream(zst: Path):
    dctx = zstandard.ZstdDecompressor()
    with open(zst, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            while True:
                game = chess.pgn.read_game(text)
                if game is None:
                    return
                yield game


def main() -> int:
    # ---- original path (slower; runs to N accepted) ----
    t0 = time.time()
    orig_accepted = []
    orig_seen = 0
    for game in original_stream(ZST):
        orig_seen += 1
        if orig_seen % 100000 == 0:
            print(f"  orig {orig_seen} games scanned, "
                  f"{len(orig_accepted)} accepted, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if not passes(game, min_plies=MIN_PLIES):
            continue
        key = select_key(game)
        h = hashlib.sha256(f"{key}:{SEED}".encode()).digest()
        if h[0] >= ACCEPT_BYTE:
            continue
        orig_accepted.append(
            (key, game_fingerprint(game),
             len(list(game.mainline_moves())))
        )
        if len(orig_accepted) >= N:
            break
    dt_orig = time.time() - t0
    print(f"original: {len(orig_accepted)} accepted in {dt_orig:.1f}s "
          f"({orig_seen} games scanned)")

    # ---- fast path ----
    t0 = time.time()
    games_seen = 0
    fast_accepted = []
    for header_block, movetext in stream_games_raw(ZST):
        games_seen += 1
        if games_seen % 100000 == 0:
            print(f"  fast {games_seen} games scanned, "
                  f"{len(fast_accepted)} accepted, "
                  f"{time.time() - t0:.0f}s", flush=True)
        headers = parse_headers(header_block)
        key = header_gate(headers, ACCEPT_BYTE, SEED)
        if key is None:
            continue
        plies = count_plies(movetext)
        if plies < MIN_PLIES:
            continue
        game = full_parse(header_block, movetext)
        real_plies = len(list(game.mainline_moves()))
        if real_plies < MIN_PLIES:
            continue  # counter overcounted; original would reject too
        fast_accepted.append(
            (key, game_fingerprint(game), real_plies)
        )
        if len(fast_accepted) >= N:
            break
    dt_fast = time.time() - t0
    print(f"fast:     {len(fast_accepted)} accepted in {dt_fast:.1f}s "
          f"({games_seen} raw games seen)")

    if len(orig_accepted) != len(fast_accepted):
        print(f"MISMATCH count {len(orig_accepted)} != {len(fast_accepted)}")
        return 1
    for i, (o, f) in enumerate(zip(orig_accepted, fast_accepted)):
        if o != f:
            print(f"MISMATCH at {i}:")
            print("  orig:", o[0], o[1][:16], o[2])
            print("  fast:", f[0], f[1][:16], f[2])
            return 1
    print(f"EQUIVALENT: {N}/{N} accepted games identical "
          f"(key, fingerprint, plies)")
    print(f"speedup: {dt_orig / max(dt_fast, 0.001):.1f}x on accepted "
          f"prefix scan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
