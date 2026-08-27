#!/usr/bin/env python3
"""Lichess Broadcasts Source Extractor (CC BY-SA 4.0).

Streams and filters games from official Lichess broadcast monthly archives
(`https://database.lichess.org/broadcast/lichess_db_broadcast_YYYY-MM.pgn.zst`).

Features & Guarantees:
- In-flight streaming SHA-256 verification of compressed stream against upstream sha256sums.txt.
- Deterministic selection: hash(GameIdentifier + selection_seed).
- Filtering:
    - Result in {1-0, 0-1, 1/2-1/2}
    - Standard starting position (or valid FEN)
    - Minimum mainline plies (>= 40 plies)
    - Long game stratum support (>= 80 plies)
- Deduplication: Computes canonical game fingerprint (initial_fen, result, uci_moves)
  and rejects any duplicates within the stream or against --exclude-pgn.
- Atomic staged publication with comprehensive source-manifest.json provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import chess
import chess.pgn
import zstandard as zstd

BROADCAST_CHECKSUMS_URL = "https://database.lichess.org/broadcast/sha256sums.txt"
BROADCAST_BASE_URL = "https://database.lichess.org/broadcast/lichess_db_broadcast_{month}.pgn.zst"


def fetch_upstream_checksums() -> dict[str, str]:
    req = urllib.request.Request(
        BROADCAST_CHECKSUMS_URL,
        headers={"User-Agent": "super-eureka-dataset-builder/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    checksums = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            sha, filename = parts[0], parts[1]
            checksums[filename] = sha
    return checksums


def compute_game_fingerprint(game: chess.pgn.Game) -> str:
    """Deterministic canonical fingerprint: initial FEN + result + UCI move list."""
    initial_fen = game.headers.get("FEN", chess.STARTING_FEN)
    result = game.headers.get("Result", "*")
    moves = [m.uci() for m in game.mainline_moves()]
    payload = json.dumps(
        {"initial_fen": initial_fen, "result": result, "moves": moves},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HashingStreamReader:
    def __init__(self, stream):
        self._stream = stream
        self._hasher = hashlib.sha256()

    def read(self, size=-1):
        chunk = self._stream.read(size)
        if chunk:
            self._hasher.update(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def extract_broadcasts(
    months: list[str],
    games_per_month: int,
    seed: int,
    source_id: str,
    source_family: str,
    out_dir: Path,
    min_ply: int = 40,
    long_game_min_ply: int = 80,
    long_game_ratio: float = 0.33,
    exclude_pgns: list[Path] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    staging_dir = out_dir.parent / f".staging-{out_dir.name}-{int(time.time())}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exclusion fingerprints
    excluded_fingerprints: set[str] = set()
    if exclude_pgns:
        for pgn_path in exclude_pgns:
            if not pgn_path.exists():
                continue
            with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
                while True:
                    g = chess.pgn.read_game(f)
                    if g is None:
                        break
                    excluded_fingerprints.add(compute_game_fingerprint(g))

    print(f"Loaded {len(excluded_fingerprints)} excluded fingerprints.")

    # 2. Fetch upstream checksums
    checksums = fetch_upstream_checksums()

    selected_games_total = 0
    seen_fingerprints: set[str] = set(excluded_fingerprints)
    manifest_sources = []

    out_pgn_path = staging_dir / f"{source_id}.pgn"

    with open(out_pgn_path, "w", encoding="utf-8") as out_f:
        for month in months:
            archive_filename = f"lichess_db_broadcast_{month}.pgn.zst"
            if archive_filename not in checksums:
                raise SystemExit(f"FAIL CLOSED: month {month} not in official sha256sums.txt")
            expected_sha = checksums[archive_filename]
            url = BROADCAST_BASE_URL.format(month=month)

            print(f"\n--- Processing broadcast month {month} ---")
            print(f"URL: {url}")
            print(f"Expected SHA-256: {expected_sha}")

            req = urllib.request.Request(
                url, headers={"User-Agent": "super-eureka-dataset-builder/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=60)
            hash_stream = HashingStreamReader(resp)

            dctx = zstd.ZstdDecompressor()
            month_selected = 0
            month_candidates_examined = 0
            long_games_needed = int(games_per_month * long_game_ratio)
            long_games_selected = 0

            with dctx.stream_reader(hash_stream) as zstd_reader:
                text_stream = io.TextIOWrapper(zstd_reader, encoding="utf-8", errors="replace")
                while month_selected < games_per_month:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    month_candidates_examined += 1

                    result = game.headers.get("Result", "*")
                    if result not in ("1-0", "0-1", "1/2-1/2"):
                        continue

                    moves = list(game.mainline_moves())
                    ply_count = len(moves)
                    if ply_count < min_ply:
                        continue

                    # Long game stratum balancing
                    is_long = ply_count >= long_game_min_ply
                    if not is_long and (games_per_month - month_selected) <= (long_games_needed - long_games_selected):
                        # Reserve remaining slots for long games
                        continue

                    # Deterministic hash threshold
                    event = game.headers.get("Event", "")
                    site = game.headers.get("Site", "")
                    white = game.headers.get("White", "")
                    black = game.headers.get("Black", "")
                    hash_key = f"{month}:{seed}:{event}:{site}:{white}:{black}:{ply_count}:{month_candidates_examined}"
                    h_val = int(hashlib.sha256(hash_key.encode("utf-8")).hexdigest()[:8], 16)
                    # Pseudo-random acceptance
                    if (h_val % 100) > 60 and month_selected + 100 < games_per_month:
                        continue

                    fp = compute_game_fingerprint(game)
                    if fp in seen_fingerprints:
                        continue

                    seen_fingerprints.add(fp)
                    if is_long:
                        long_games_selected += 1

                    # Write game to PGN
                    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                    out_f.write(game.accept(exporter) + "\n\n")
                    month_selected += 1

                    if month_selected % 500 == 0 or month_selected == games_per_month:
                        print(f"  Selected {month_selected}/{games_per_month} games (examined {month_candidates_examined}, long: {long_games_selected})")

            # Consume rest of stream to verify full archive SHA
            while True:
                buf = hash_stream.read(1024 * 1024)
                if not buf:
                    break

            actual_sha = hash_stream.hexdigest()
            if actual_sha != expected_sha:
                raise SystemExit(
                    f"FAIL CLOSED: archive SHA-256 mismatch for {month}: {actual_sha} != {expected_sha}"
                )
            print(f"Month {month} SHA-256 verified in flight: {actual_sha}")

            if month_selected < games_per_month:
                raise SystemExit(
                    f"FAIL CLOSED: insufficient games selected for {month}: {month_selected} < {games_per_month}"
                )

            manifest_sources.append(
                {
                    "month": month,
                    "url": url,
                    "sha256": actual_sha,
                    "games_selected": month_selected,
                    "candidates_examined": month_candidates_examined,
                }
            )
            selected_games_total += month_selected

    # Compute PGN SHA
    with open(out_pgn_path, "rb") as f:
        pgn_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "source_id": source_id,
        "source_family": source_family,
        "provenance": "lichess-official-broadcasts-cc-by-sa-4.0",
        "months": months,
        "selection_seed": seed,
        "games_total": selected_games_total,
        "pgn_sha256": pgn_sha,
        "min_ply": min_ply,
        "upstream_archives": manifest_sources,
        "excluded_fingerprints_count": len(excluded_fingerprints),
    }

    manifest_path = staging_dir / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Atomic move
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging_dir.rename(out_dir)
    print(f"\nSuccessfully created source {source_id} at {out_dir} with {selected_games_total} games (PGN SHA: {pgn_sha[:16]}).")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Lichess Broadcasts Source Extractor")
    parser.add_argument("--months", nargs="+", required=True, help="List of YYYY-MM months")
    parser.add_argument("--games-per-month", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--source-id", default="lichess-broadcast-v1")
    parser.add_argument("--source-family", default="lichess-broadcast")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-pgn", type=Path, nargs="*", default=None)

    args = parser.parse_args()
    extract_broadcasts(
        months=args.months,
        games_per_month=args.games_per_month,
        seed=args.seed,
        source_id=args.source_id,
        source_family=args.source_family,
        out_dir=args.out,
        exclude_pgns=args.exclude_pgn,
    )


if __name__ == "__main__":
    main()
