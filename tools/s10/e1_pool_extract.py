"""S10-E1: per-source parallel candidate extraction with resumable cache.

Scans each B1 source PGN independently (multiprocessing) and writes a
compact, immutable per-source cache:

    data/s10/e1-pool-cache/<source_id>.jsonl          one record per kept
                                                       candidate (top-8 per
                                                       game, frozen
                                                       sampling_version 2)
    data/s10/e1-pool-cache/<source_id>.manifest.json   provenance binding

The cache manifest pins: source_id, PGN SHA-256, sampling_version,
DATASET_SEED, MIN/MAX_PLY, MAX_PER_GAME, extractor contract constants
and the candidate list's own SHA-256. A completed cache is immutable; a
dropped session only re-extracts missing sources, never the whole pool.

Cross-source position dedup happens LATER, in the global merge
(e1_pool_merge.py) — cell capacities may only be announced after global
dedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s6.build_dataset import (  # noqa: E402
    DATASET_SEED,
    MAX_PER_GAME,
    MAX_PLY,
    MIN_PLY,
    SAMPLING_VERSION,
    SCHEMA_VERSION,
    canonical_fen4,
    eligible,
    find_pgn,
    game_result_white,
    game_split,
    load_games,
    load_source_catalog,
    phase_of,
    ply_priority,
    sha256_text,
)

CACHE_DIR = Path("data/s10/e1-pool-cache")

CACHE_CONTRACT = {
    "sampling_version": SAMPLING_VERSION,
    "dataset_seed": DATASET_SEED,
    "min_ply": MIN_PLY,
    "max_ply": MAX_PLY,
    "max_per_game": MAX_PER_GAME,
    "schema_version": SCHEMA_VERSION,
}


def _source_dirs(root: Path) -> list[Path]:
    """Root sources dir + every immediate subdir with a local manifest."""
    dirs = [root]
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (
            (sub / "source-manifest.json").is_file()
            or (sub / "source_manifest.json").is_file()
        ):
            dirs.append(sub)
    return dirs


def extract_one(job: dict) -> dict:
    """Extract candidates for one source; uses the resumable cache."""
    source_name = job["source_name"]
    entry = job["entry"]
    source_dirs = [Path(p) for p in job["source_dirs"]]
    cache_dir = Path(job["cache_dir"])

    pgn_path = source_dirs[0] / f"{source_name}.pgn"
    if not pgn_path.is_file():
        pgn_path = find_pgn(source_dirs, source_name, entry)

    pgn_sha = hashlib.sha256(pgn_path.read_bytes()).hexdigest()
    manifest_path = cache_dir / f"{source_name}.manifest.json"
    candidates_path = cache_dir / f"{source_name}.jsonl"

    # Resumable: a cache whose provenance still binds is immutable.
    if manifest_path.is_file() and candidates_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("pgn_sha256") == pgn_sha
            and manifest.get("contract") == CACHE_CONTRACT
            and manifest.get("candidates_sha256")
            == hashlib.sha256(
                candidates_path.read_bytes()
            ).hexdigest()
        ):
            return {
                "source": source_name,
                "cached": True,
                "candidates": manifest["candidate_count"],
            }

    import chess  # noqa: F401  (imported by build_dataset helpers)

    records = []
    game_no = 0
    for game in load_games(pgn_path):
        game_no += 1
        game_id = f"{entry['source_id']}:{game_no}"
        result = game_result_white(game)
        if result is None:
            continue
        moves = list(game.mainline_moves())
        if len(moves) < MIN_PLY:
            continue
        board = game.board()
        split = game_split(game_id)

        candidates = []
        for ply in range(1, len(moves) + 1):
            board.push(moves[ply - 1])
            if ply < MIN_PLY or ply > MAX_PLY:
                continue
            ok, _reason = eligible(board)
            if not ok:
                continue
            fen4 = canonical_fen4(board)
            candidates.append(
                (
                    ply_priority(game_id, ply, DATASET_SEED),
                    ply,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "position_id": sha256_text(fen4),
                        "fen": board.fen(),
                        "canonical_fen4": fen4,
                        "source_id": entry["source_id"],
                        "source_game_id": game_id,
                        "ply": ply,
                        "game_result_white": result,
                        "phase": phase_of(board),
                        "split": split,
                    },
                )
            )
        candidates.sort(key=lambda c: (c[0], c[1]))
        for _, _, rec in candidates[:MAX_PER_GAME]:
            records.append(rec)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = candidates_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    cand_sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(candidates_path)

    manifest = {
        "source_id": entry["source_id"],
        "source_name": source_name,
        "pgn_path": str(pgn_path),
        "pgn_sha256": pgn_sha,
        "games": game_no,
        "candidate_count": len(records),
        "candidates_sha256": cand_sha,
        "contract": CACHE_CONTRACT,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "source": source_name,
        "cached": False,
        "candidates": len(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources-root", type=Path, default=Path("data/s6/sources")
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    dirs = _source_dirs(args.sources_root)
    catalog = load_source_catalog(dirs)
    print(f"catalog: {len(catalog)} sources from {len(dirs)} dirs")

    jobs = [
        {
            "source_name": name,
            "entry": entry,
            "source_dirs": [str(d) for d in dirs],
            "cache_dir": str(args.cache_dir),
        }
        for name, entry in sorted(catalog.items())
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(extract_one, jobs):
            tag = "cache" if res["cached"] else "scan"
            print(f"  [{tag}] {res['source']}: {res['candidates']} candidates")
            results.append(res)

    total = sum(r["candidates"] for r in results)
    cached = sum(1 for r in results if r["cached"])
    print(f"\n{len(results)} sources, {total} candidates "
          f"({cached} from cache)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
