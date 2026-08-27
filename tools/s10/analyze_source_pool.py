#!/usr/bin/env python3
"""Source Pool Feasibility & Binding Constraint Profiler for S10 Datasets.

Directly imports and reuses the frozen builder functions and constants from
`tools/s6/build_dataset.py` (sampling-v2 top-8, eligible filter, canonical_fen4,
game_split, phase_of, bucket_of, family_of, DATASET_SEED, etc.).

Audits:
- 12-cell matrix: (split x phase) against exact target quotas
- Family distribution: (family x split x phase) post-deduplication
- Global family share against 70% ceiling
- Exact shortfalls and binding constraints
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.s6.build_dataset import (
    DATASET_SEED,
    MAX_PER_GAME,
    MIN_PLY,
    MAX_PLY,
    PHASE_BUCKETS,
    PHASE_QUOTAS,
    FINAL_SPLIT_TARGETS,
    FINAL_PHASE_TARGETS,
    SCHEMA_VERSION,
    bucket_of,
    canonical_fen4,
    eligible,
    family_of,
    find_pgn,
    game_result_white,
    game_split,
    load_games,
    load_source_catalog,
    phase_of,
    ply_priority,
    sha256_bytes,
    sha256_text,
)

TARGET_CELLS_300K = {
    "train": {"high": 60000, "mid": 108000, "low": 48000, "zero": 24000},
    "validation": {"high": 7500, "mid": 13500, "low": 6000, "zero": 3000},
    "holdout": {"high": 7500, "mid": 13500, "low": 6000, "zero": 3000},
}


def analyze_pool(
    source_dirs: list[Path],
    target_n: int = 300000,
    final_mode: bool = True,
) -> dict:
    catalog = load_source_catalog(source_dirs)
    print(f"Loaded {len(catalog)} source(s) into catalog.")

    records: list[dict] = []
    reject_stats = defaultdict(int)
    per_source_stats = {}

    for source_name in sorted(catalog):
        src = catalog[source_name]
        pgn_path = source_dirs[0] / f"{source_name}.pgn"
        if not pgn_path.is_file():
            pgn_path = find_pgn(source_dirs, source_name, src)

        game_no = 0
        source_candidates = 0
        for game in load_games(pgn_path):
            game_no += 1
            game_id = f"{src['source_id']}:{game_no}"
            result = game_result_white(game)
            if result is None:
                continue
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY:
                continue
            board = game.board()
            split = game_split(game_id)

            candidates: list[tuple[int, int, dict]] = []
            for ply in range(1, len(moves) + 1):
                board.push(moves[ply - 1])
                if ply < MIN_PLY or ply > MAX_PLY:
                    continue
                ok, reason = eligible(board)
                if not ok:
                    reject_stats[reason] += 1
                    continue
                fen4 = canonical_fen4(board)
                pos_id = sha256_text(fen4)
                candidates.append(
                    (
                        ply_priority(game_id, ply, DATASET_SEED),
                        ply,
                        {
                            "position_id": pos_id,
                            "source_id": src["source_id"],
                            "source_game_id": game_id,
                            "phase_bucket": bucket_of(phase_of(board)),
                            "split": split,
                        },
                    )
                )

            candidates.sort(key=lambda c: (c[0], c[1]))
            for _, _, rec in candidates[:MAX_PER_GAME]:
                records.append(rec)
                source_candidates += 1

        per_source_stats[source_name] = {
            "games": game_no,
            "candidates_top8": source_candidates,
        }

    # 1. Global deduplication (first seen by position_id)
    before_dedup = len(records)
    records.sort(key=lambda r: (r["position_id"], r["source_game_id"]))
    unique_records: list[dict] = []
    seen_pos: set[str] = set()
    for r in records:
        if r["position_id"] in seen_pos:
            continue
        seen_pos.add(r["position_id"])
        unique_records.append(r)

    dedup_removed = before_dedup - len(unique_records)
    print(f"\nTop-8 pool: {before_dedup} -> {len(unique_records)} post-dedup ({dedup_removed} duplicates removed)")

    # 2. 12-cell matrix calculation (split x phase)
    cell_counts: dict[str, dict[str, int]] = {
        split: {phase: 0 for phase in ("high", "mid", "low", "zero")}
        for split in ("train", "validation", "holdout")
    }

    # 3. Family x split x phase matrix
    family_cell_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {
            split: {phase: 0 for phase in ("high", "mid", "low", "zero")}
            for split in ("train", "validation", "holdout")
        }
    )

    family_totals: dict[str, int] = defaultdict(int)

    for r in unique_records:
        fam = family_of(r["source_id"], catalog, final_mode=final_mode)
        split = r["split"]
        phase = r["phase_bucket"]

        cell_counts[split][phase] += 1
        family_cell_counts[fam][split][phase] += 1
        family_totals[fam] += 1

    # 4. Target cells computation for arbitrary target_n
    if target_n == 300000:
        targets = TARGET_CELLS_300K
    else:
        targets = {}
        for split, split_want in FINAL_SPLIT_TARGETS.items():
            split_ratio = split_want / 300000.0
            targets[split] = {}
            for phase, phase_share in PHASE_QUOTAS.items():
                targets[split][phase] = round(target_n * split_ratio * phase_share)

    # 5. Shortfall calculation
    cell_eval: dict[str, dict[str, dict[str, int]]] = {}
    shortfalls_found = 0
    total_shortfall = 0

    print("\n" + "=" * 80)
    print(f"12-CELL AUDIT MATRIX (Target N = {target_n})")
    print("=" * 80)
    print(f"{'Split':<12} {'Phase':<8} {'Available':<12} {'Target':<10} {'Margin':<10} {'Status'}")
    print("-" * 80)

    for split in ("train", "validation", "holdout"):
        cell_eval[split] = {}
        for phase in ("high", "mid", "low", "zero"):
            avail = cell_counts[split][phase]
            tgt = targets[split][phase]
            margin = avail - tgt
            status = "PASS" if margin >= 0 else "DEFICIT"
            if margin < 0:
                shortfalls_found += 1
                total_shortfall += abs(margin)
            cell_eval[split][phase] = {
                "available": avail,
                "target": tgt,
                "margin": margin,
                "status": status,
            }
            print(f"{split:<12} {phase:<8} {avail:<12} {tgt:<10} {margin:<+10} {status}")

    # 6. Family breakdown
    print("\n" + "=" * 80)
    print("FAMILY BREAKDOWN POST-DEDUP")
    print("=" * 80)
    total_post_dedup = len(unique_records)
    largest_family_share = 0.0
    for fam, count in sorted(family_totals.items(), key=lambda x: -x[1]):
        share = count / max(1, total_post_dedup)
        if share > largest_family_share:
            largest_family_share = share
        print(f"Family '{fam}': {count} positions ({share * 100:.2f}%)")

    print("\nDetailed Family x Split x Phase:")
    for fam in sorted(family_cell_counts):
        print(f"\n  Family [{fam}]:")
        for split in ("train", "validation", "holdout"):
            phase_str = ", ".join(
                f"{p}: {family_cell_counts[fam][split][p]}"
                for p in ("high", "mid", "low", "zero")
            )
            print(f"    {split:<12}: {phase_str}")

    report = {
        "target_n": target_n,
        "total_games": sum(s["games"] for s in per_source_stats.values()),
        "candidates_top8": before_dedup,
        "unique_post_dedup": len(unique_records),
        "dedup_removed": dedup_removed,
        "reject_stats": dict(reject_stats),
        "per_source_stats": per_source_stats,
        "cell_matrix": cell_eval,
        "family_totals": dict(family_totals),
        "family_cell_counts": {k: dict(v) for k, v in family_cell_counts.items()},
        "largest_family_share": largest_family_share,
        "family_share_pass": largest_family_share <= 0.70,
        "shortfalls_count": shortfalls_found,
        "total_shortfall": total_shortfall,
        "is_feasible": shortfalls_found == 0 and largest_family_share <= 0.70,
    }

    print("\n" + "=" * 80)
    if report["is_feasible"]:
        print("RESULT: ALL 12 CELLS PASS & FAMILY SHARE <= 70% (FEASIBLE)")
    else:
        print(f"RESULT: FAIL CLOSED ({shortfalls_found} cell shortfalls, total deficit: {total_shortfall})")
    print("=" * 80)

    return report


def main():
    parser = argparse.ArgumentParser(description="Source Pool Feasibility Profiler")
    parser.add_argument("sources", type=Path, nargs="+", help="Source directories")
    parser.add_argument("--target", type=int, default=300000, help="Target record count (default 300,000)")
    parser.add_argument("--legacy", action="store_true", help="Allow legacy family fallback")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path")

    args = parser.parse_args()
    report = analyze_pool(
        source_dirs=args.sources,
        target_n=args.target,
        final_mode=not args.legacy,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.json}")

    if not report["is_feasible"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
