#!/usr/bin/env python3
"""Lichess Broadcasts Source Extractor (CC BY-SA 4.0).

Streams and filters games from official Lichess broadcast monthly archives
(`https://database.lichess.org/broadcast/lichess_db_broadcast_YYYY-MM.pgn.zst`).

Features & Guarantees:
- In-flight streaming SHA-256 verification of compressed stream against upstream sha256sums.txt.
  Compressed SHA is fully verified BEFORE decompression stream closing.
- Global whole-month deterministic top-K selection: hash(seed || month || canonical_fingerprint).
- Explicit long/short stratum balancing: ~1/3 long games (>= 80 plies) and 2/3 short games (>= 40 plies).
- Canonical game fingerprinting matching S6 lichess_select.py.
- Explicit --exclude-pgn missing fails closed.
- Existing out_dir fails closed (never overwrites an already-published source).
- Comprehensive source-manifest.json provenance.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chess
import chess.pgn
import zstandard as zstd

from tools.s10.source_identity import (
    game_fingerprint,
    fingerprint_set_sha256,
    load_pgn_fingerprints,
)

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


def compute_selection_rank(seed: int, month: str, fingerprint: str) -> str:
    """Deterministic selection rank independent of archive encounter order."""
    key = f"{seed}:{month}:{fingerprint}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


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
    local_archives: dict[str, Path] | None = None,
    mock_checksums: dict[str, str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"FAIL CLOSED: destination source dir already exists: {out_dir}")

    staging_dir = out_dir.parent / f".staging-{out_dir.name}-{int(time.time())}-{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exclusion fingerprints (fails closed if file missing)
    excluded_fingerprints: set[str] = set()
    if exclude_pgns:
        for pgn_path in exclude_pgns:
            excluded_fingerprints.update(load_pgn_fingerprints(pgn_path))

    print(f"Loaded {len(excluded_fingerprints)} excluded fingerprints.")

    # 2. Upstream checksums
    checksums = mock_checksums if mock_checksums is not None else fetch_upstream_checksums()

    # Calculate stratum targets ensuring long_quota + short_quota == games_per_month exactly
    long_quota = int(games_per_month * long_game_ratio)
    if games_per_month > 0 and long_quota == 0 and long_game_ratio > 0.0:
        long_quota = 1
    short_quota = games_per_month - long_quota

    selected_fingerprints_all: set[str] = set()
    manifest_sources = []

    out_pgn_path = staging_dir / f"{source_id}.pgn"

    try:
        with open(out_pgn_path, "w", encoding="utf-8") as out_f:
            for month in months:
                archive_filename = f"lichess_db_broadcast_{month}.pgn.zst"
                if archive_filename not in checksums:
                    raise SystemExit(f"FAIL CLOSED: month {month} not in upstream checksums")
                expected_sha = checksums[archive_filename]
                url = BROADCAST_BASE_URL.format(month=month)

                print(f"\n--- Processing broadcast month {month} ---")
                print(f"Expected SHA-256: {expected_sha}")

                if local_archives and month in local_archives:
                    raw_stream = open(local_archives[month], "rb")
                else:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "super-eureka-dataset-builder/1.0"}
                    )
                    raw_stream = urllib.request.urlopen(req, timeout=60)

                hashing_reader = HashingStreamReader(raw_stream)
                dctx = zstd.ZstdDecompressor()

                # Stream and collect all eligible candidates for this month
                long_candidates: list[tuple[str, str, str]] = []
                short_candidates: list[tuple[str, str, str]] = []

                candidates_examined = 0
                eligible_candidates = 0

                zstd_stream = dctx.stream_reader(hashing_reader)
                text_stream = io.TextIOWrapper(zstd_stream, encoding="utf-8", errors="replace")

                try:
                    while True:
                        game = chess.pgn.read_game(text_stream)
                        if game is None:
                            break
                        candidates_examined += 1

                        result = game.headers.get("Result", "*")
                        if result not in ("1-0", "0-1", "1/2-1/2"):
                            continue

                        moves = list(game.mainline_moves())
                        ply_count = len(moves)
                        if ply_count < min_ply:
                            continue

                        fp = game_fingerprint(game)
                        if fp in excluded_fingerprints or fp in selected_fingerprints_all:
                            continue

                        eligible_candidates += 1
                        rank = compute_selection_rank(seed, month, fp)

                        # Export PGN string
                        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                        pgn_text = game.accept(exporter)

                        if ply_count >= long_game_min_ply:
                            long_candidates.append((rank, pgn_text, fp))
                        else:
                            short_candidates.append((rank, pgn_text, fp))

                    # Drain the remainder of hashing_reader BEFORE closing decompressor
                    while True:
                        buf = hashing_reader.read(1024 * 1024)
                        if not buf:
                            break

                    actual_sha = hashing_reader.hexdigest()
                    if actual_sha != expected_sha:
                        raise SystemExit(
                            f"FAIL CLOSED: archive SHA-256 mismatch for {month}: {actual_sha} != {expected_sha}"
                        )
                    print(f"Month {month} SHA-256 verified in flight: {actual_sha}")

                finally:
                    text_stream.close()
                    zstd_stream.close()
                    raw_stream.close()

                if len(long_candidates) < long_quota or len(short_candidates) < short_quota:
                    raise SystemExit(
                        f"FAIL CLOSED: insufficient candidates for month {month}: "
                        f"got {len(long_candidates)}/{long_quota} long, {len(short_candidates)}/{short_quota} short "
                        f"(examined: {candidates_examined}, eligible: {eligible_candidates})"
                    )

                # Deterministic global top-K selection by rank
                long_candidates.sort(key=lambda item: item[0])
                short_candidates.sort(key=lambda item: item[0])

                selected_long = long_candidates[:long_quota]
                selected_short = short_candidates[:short_quota]

                selected_month_games = sorted(selected_long + selected_short, key=lambda item: item[0])
                for rank, pgn_text, fp in selected_month_games:
                    out_f.write(pgn_text + "\n\n")
                    selected_fingerprints_all.add(fp)

                manifest_sources.append(
                    {
                        "month": month,
                        "url": url,
                        "sha256": actual_sha,
                        "games_selected": len(selected_month_games),
                        "long_games_selected": len(selected_long),
                        "short_games_selected": len(selected_short),
                        "candidates_examined": candidates_examined,
                        "eligible_candidates": eligible_candidates,
                    }
                )

        # 3. Compute PGN SHA
        with open(out_pgn_path, "rb") as f:
            pgn_sha = hashlib.sha256(f.read()).hexdigest()

        # Compute script SHA
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

        manifest = {
            "source_id": source_id,
            "source_family": source_family,
            "provenance": "lichess-official-broadcasts",
            "license": "CC-BY-SA-4.0",
            "selection_algorithm_version": 2,
            "selection_strategy": "global_whole_month_deterministic_top_k",
            "script_sha256": script_sha,
            "python_version": sys.version,
            "python_chess_version": chess.__version__,
            "zstandard_version": zstd.__version__,
            "months": months,
            "selection_seed": seed,
            "games_total": len(selected_fingerprints_all),
            "pgn_sha256": pgn_sha,
            "min_ply": min_ply,
            "long_game_min_ply": long_game_min_ply,
            "long_game_ratio": long_game_ratio,
            "selected_fingerprints_count": len(selected_fingerprints_all),
            "selected_fingerprints_sha256": fingerprint_set_sha256(selected_fingerprints_all),
            "excluded_fingerprints_count": len(excluded_fingerprints),
            "excluded_fingerprints_sha256": fingerprint_set_sha256(excluded_fingerprints),
            "fingerprint_intersection_count": len(selected_fingerprints_all & excluded_fingerprints),
            "upstream_archives": manifest_sources,
        }

        if manifest["fingerprint_intersection_count"] != 0:
            raise SystemExit("FAIL CLOSED: fingerprint intersection with excluded PGN is non-zero")

        manifest_path = staging_dir / "source-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Atomic publish
        staging_dir.rename(out_dir)
        print(f"\nSuccessfully published source {source_id} to {out_dir} ({len(selected_fingerprints_all)} games).")
        return manifest

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


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
