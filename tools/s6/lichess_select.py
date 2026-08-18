#!/usr/bin/env python3
"""lichess-standard-rated-v1 selection tool (S6 third source family).

Streams official Lichess monthly standard-rated .pgn.zst files (CC0) with
python-zstandard - never extracts the full archive to disk. The COMPRESSED
stream is SHA-256-verified in flight against the official per-month checksum
from the upstream `sha256sums.txt`; a missing or mismatching checksum fails
closed. Deterministic selection: hash(GameURL + selection_seed) first byte
< threshold, capped per month and per long-game stratum.

filters (matches passes() exactly):
  Result in {1-0, 0-1, 1/2-1/2}
  WhiteElo >= 1800 and BlackElo >= 1800
  WhiteTitle != BOT and BlackTitle != BOT
  TimeControl base >= 180 sec   (the speed gate is TimeControl, NOT Event)
  mainline >= 40 plies

The database covers standard chess across all time controls (Event values
like "Rated Rapid tournament ..."), so the Event tag is deliberately not a
filter. long-game stratum: ~1/3 of selected games must have >= 80 plies
(deterministic).

Outputs (staged, then atomically published; nothing is published if any month
selects fewer than --games-per-month games):
  out/lichess-standard-rated-v1.pgn   selected games (raw, with headers)
  out/source-manifest.json           provenance (CC0, upstream URLs, official
                                     SHA256s, script SHA, selection seed,
                                     counts, output SHA256)

Usage:
  python tools/s6/lichess_select.py \
      --months 2026-07 \
      --games-per-month 2000 \
      --seed 20260812 \
      --out data/s6/sources/lichess-standard-rated-v1
  (add --local <file.zst> to read a pre-downloaded, SHA-verified archive)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import zstandard

BASE = "https://database.lichess.org/standard/"
CHECKSUMS_URL = BASE + "sha256sums.txt"
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


class HashingReader:
    """File-like wrapper that feeds every read into a live SHA-256."""

    def __init__(self, resp, hasher):
        self._resp = resp
        self._hasher = hasher

    def read(self, size: int = -1) -> bytes:
        data = self._resp.read(size if size and size > 0 else 1 << 20)
        if data:
            self._hasher.update(data)
        return data


def fetch_official_sha256(month: str) -> str:
    """Official checksum for one monthly archive from sha256sums.txt."""
    try:
        with urllib.request.urlopen(CHECKSUMS_URL, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"FAIL CLOSED: cannot fetch official sha256sums.txt: {exc}")
    filename = f"lichess_db_standard_rated_{month}.pgn.zst"
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].endswith(filename):
            return parts[0].strip().lower()
    raise SystemExit(
        f"FAIL CLOSED: no official SHA256 for {filename} in sha256sums.txt")


def open_month_stream(month: str, local: Path | None):
    """Return (decompressed_reader, hasher, raw_source, official_sha, url).

    Every compressed byte is fed into `hasher` exactly once as it is read
    (by the decompressor or by the caller's final drain); the caller must
    drain `raw_source` to EOF after selection and compare the digest.
    """
    url = f"{BASE}lichess_db_standard_rated_{month}.pgn.zst"
    official = fetch_official_sha256(month)
    if local is not None:
        raw = open(local, "rb")
        display = str(local)
    else:
        try:
            raw = urllib.request.urlopen(url, timeout=120)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"FAIL CLOSED: cannot open {url}: {exc}") from exc
        display = url
    hasher = hashlib.sha256()
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(HashingReader(raw, hasher))
    return reader, hasher, raw, official, display


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", default="2026-05,2026-06,2026-07")
    parser.add_argument("--games-per-month", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", required=True)
    parser.add_argument("--local", type=Path, default=None,
                        help="pre-downloaded monthly archive (single month); "
                             "SHA-verified against the official checksum")
    parser.add_argument("--official-sha256", default="",
                        help="legacy 'month=sha,...' override (unused when "
                             "sha256sums.txt is reachable)")
    args = parser.parse_args(sys.argv[1:])

    out_dir = Path(args.out)
    if out_dir.exists():
        print(f"FAIL CLOSED: {out_dir} already exists; refusing to overwrite")
        return 3
    months = [m.strip() for m in args.months.split(",")]
    if args.local is not None and len(months) != 1:
        raise SystemExit("FAIL CLOSED: --local requires exactly one --months")
    long_target = int(args.games_per_month * LONG_GAME_FRACTION)
    total = 0
    t0 = time.time()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    staging = Path(str(out_dir) + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    pgn_path = staging / "lichess-standard-rated-v1.pgn"
    official_sha256: dict[str, str] = {}

    with open(pgn_path, "w", encoding="utf-8") as fh:
        for month in months:
            reader, hasher, raw, official, display = open_month_stream(month, args.local)
            official_sha256[month] = official
            print(f"streaming {display} (official sha {official[:16]}...)",
                  flush=True)
            month_selected = 0
            month_long = 0
            seen = 0
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
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
            # Consume the ENTIRE compressed stream so the in-flight SHA-256
            # covers every byte, then fail closed on any mismatch.
            while raw.read(1 << 20):
                pass
            actual = hasher.hexdigest()
            if actual != official:
                print(
                    f"FAIL CLOSED: stream SHA mismatch for {month}: "
                    f"{actual[:12]} != {official[:12]}", flush=True)
                return 5
            if month_selected < args.games_per_month:
                print(
                    f"FAIL CLOSED: {month} selected only {month_selected} "
                    f"games (< {args.games_per_month}); staging NOT published",
                    flush=True)
                return 4
            total += month_selected
            elapsed = time.time() - t0
            print(f"{month}: {month_selected} selected ({month_long} long) "
                  f"of {seen} passing-eligible seen, {elapsed:.0f}s",
                  flush=True)

    manifest = {
        "source_family": "lichess-standard-rated-v1",
        "source_id": "lichess-standard-rated-v1",
        "license": "CC0",
        "upstream": [f"{BASE}lichess_db_standard_rated_{m}.pgn.zst"
                     for m in months],
        "official_sha256": official_sha256,
        "script_sha256": script_sha,
        "selection_seed": args.seed,
        "filters": {
            "database": "standard rated (all standard-chess time controls)",
            "event_filter": "none (TimeControl is the speed gate)",
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
    (staging / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2))
    shutil.move(str(staging), str(out_dir))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
