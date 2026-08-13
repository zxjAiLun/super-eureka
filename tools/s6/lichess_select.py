#!/usr/bin/env python3
"""lichess-standard-rated-v1 selection tool (S6 third source family).

Streams official Lichess monthly standard-rated .pgn.zst files (CC0) with
python-zstandard - never extracts the full archive to disk. Deterministic
selection: hash(GameURL + selection_seed) first byte < threshold, capped per
month and per long-game stratum.

filters:
  Event == "Rated Standard game"
  Result in {1-0, 0-1, 1/2-1/2}
  WhiteElo >= 1800 and BlackElo >= 1800
  WhiteTitle != BOT and BlackTitle != BOT
  TimeControl base >= 180 sec
  mainline >= 40 plies

long-game stratum: ~1/3 of selected games must have >= 80 plies (deterministic).

Outputs:
  out/lichess-standard-rated-v1.pgn   selected games (raw, with headers)
  out/source-manifest.json           provenance (CC0, upstream URLs, official
                                     SHA256s, filters, script SHA, selection
                                     seed, counts, output SHA256)

Usage:
  python tools/s6/lichess_select.py \
      --months 2026-05,2026-06,2026-07 \
      --games-per-month 12000 \
      --seed 20260812 \
      --out data/s6/sources/lichess-standard-rated-v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import zstandard

BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_"
TIME_CONTROL_BASE_MIN = 180
LONG_GAME_FRACTION = 1.0 / 3.0
ACCEPT_BYTE = 0x05  # ~2% of passing games accepted


def time_control_base(tc: str) -> int:
    base = tc.split("+")[0]
    try:
        return int(base)
    except ValueError:
        return 0


def passes(game: chess.pgn.Game) -> bool:
    h = game.headers
    # the lichess "standard" database covers standard chess across all time
    # controls (Event values like "Rated Rapid tournament ..."); the speed
    # gate is the TimeControl base, not the Event tag.
    if h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return False
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
        return False
    try:
        if int(h.get("WhiteElo", "0")) < 1800 or int(h.get("BlackElo", "0")) < 1800:
            return False
    except ValueError:
        return False
    if time_control_base(h.get("TimeControl", "0")) < TIME_CONTROL_BASE_MIN:
        return False
    if len(list(game.mainline_moves())) < 40:
        return False
    return True


def select_key(game: chess.pgn.Game) -> str:
    h = game.headers
    url = h.get("Site", "")
    if url:
        return url
    return "|".join([h.get("White", ""), h.get("Black", ""),
                     h.get("Date", ""), h.get("Round", ""),
                     h.get("Result", "")])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default="2026-05,2026-06,2026-07")
    parser.add_argument("--games-per-month", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", required=True)
    parser.add_argument("--official-sha256", default="",
                        help="'month=sha,...' upstream checksums for provenance")
    args = parser.parse_args(sys.argv[1:])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = out_dir / "lichess-standard-rated-v1.pgn"
    months = [m.strip() for m in args.months.split(",")]
    long_target = int(args.games_per_month * LONG_GAME_FRACTION)
    total = 0
    t0 = time.time()
    dctx = zstandard.ZstdDecompressor()

    with open(pgn_path, "w", encoding="utf-8") as fh:
        for month in months:
            url = f"{BASE}{month}.pgn.zst"
            print(f"streaming {url}", flush=True)
            month_selected = 0
            month_long = 0
            seen = 0
            with urllib.request.urlopen(url, timeout=120) as resp:
                reader = dctx.stream_reader(resp)
                text = io.TextIOWrapper(reader, encoding="utf-8",
                                        errors="replace")
                while month_selected < args.games_per_month:
                    game = chess.pgn.read_game(text)
                    if game is None:
                        break
                    seen += 1
                    if not passes(game):
                        continue
                    key = select_key(game)
                    h = hashlib.sha256(
                        f"{key}:{args.seed}".encode("utf-8")).digest()
                    if h[0] >= ACCEPT_BYTE:
                        continue
                    is_long = len(list(game.mainline_moves())) >= 80
                    if is_long:
                        if month_long >= long_target:
                            continue
                        month_long += 1
                    elif month_selected - month_long >= \
                            args.games_per_month - long_target:
                        continue
                    month_selected += 1
                    print(game, file=fh)
                    print(file=fh)
            total += month_selected
            elapsed = time.time() - t0
            print(f"{month}: {month_selected} selected ({month_long} long) "
                  f"of {seen} passing-eligible seen, {elapsed:.0f}s",
                  flush=True)

    manifest = {
        "source_family": "lichess-standard-rated-v1",
        "source_id": "lichess-standard-rated-v1",
        "license": "CC0",
        "upstream": [f"{BASE}{m}.pgn.zst" for m in months],
        "official_sha256": args.official_sha256 or None,
        "selection_seed": args.seed,
        "filters": {
            "database": "standard rated (all standard-chess time controls)",
            "elo_min": 1800,
            "no_bot_titles": True,
            "time_control_base_min_sec": TIME_CONTROL_BASE_MIN,
            "mainline_plies_min": 40,
            "long_stratum_plies_min": 80,
            "long_fraction": LONG_GAME_FRACTION,
        },
        "games_selected": total,
        "selection": "hash(GameURL, seed) first byte < 0x05, capped per "
                     "month and per long stratum",
        "pgn_sha256": hashlib.sha256(pgn_path.read_bytes()).hexdigest(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
