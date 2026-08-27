#!/usr/bin/env python3
"""Source Pool Feasibility & Binding Constraint Profiler for S10 Datasets.

Directly imports and reuses the frozen builder functions and constants from
`tools/s6/build_dataset.py` (sampling-v2 top-8, eligible filter, canonical_fen4,
game_split, phase_of, bucket_of, family_of, DATASET_SEED, etc.).

Implements the exact 3-tier builder pipeline:
1. Tier 1: Raw Post-Deduplication (top-8 per game, global position_id dedup)
2. Tier 2: Global Phase Stratification (replicates build_dataset.py lines 375-387)
   - Evaluates the 12 core cells (split x phase) on this stratified pool
   - Evaluates Pre-FINAL family share gate (>= 2 families, <= 70% largest share)
3. Tier 3: Exact FINAL Selection (exact per-cell targeting, sorted by position_id)
   - Evaluates Post-FINAL family share gate (>= 2 families, <= 70% largest share)
   - Checks final_selected_count == target_n
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
    TARGET as DEFAULT_TARGET,
    MIN_FAMILIES,
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


def profile_pool(
    source_dirs: list[Path],
    target_n: int = 300000,
    final_mode: bool = True,
    enforce_family_mix: bool = False,
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
                            "schema_version": SCHEMA_VERSION,
                            "position_id": pos_id,
                            "fen": board.fen(),
                            "canonical_fen4": fen4,
                            "source_id": src["source_id"],
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
                source_candidates += 1

        per_source_stats[source_name] = {
            "games": game_no,
            "candidates_top8": source_candidates,
        }

    # =========================================================================
    # TIER 1: Raw Post-Deduplication
    # =========================================================================
    raw_candidates_seen = len(records)
    records.sort(key=lambda r: (r["position_id"], r["source_game_id"]))
    unique_records: list[dict] = []
    seen_pos: set[str] = set()
    for r in records:
        if r["position_id"] in seen_pos:
            continue
        seen_pos.add(r["position_id"])
        unique_records.append(r)

    dedup_removed = raw_candidates_seen - len(unique_records)
    raw_post_dedup_count = len(unique_records)

    raw_family_totals: dict[str, int] = defaultdict(int)
    for r in unique_records:
        fam = family_of(r["source_id"], catalog, final_mode=final_mode)
        raw_family_totals[fam] += 1

    tier1_stats = {
        "candidates_top8": raw_candidates_seen,
        "unique_post_dedup": raw_post_dedup_count,
        "duplicates_removed": dedup_removed,
        "family_totals": dict(raw_family_totals),
    }

    # =========================================================================
    # TIER 2: Global Phase Stratification (Matching build_dataset.py lines 375-387)
    # =========================================================================
    n_pre = len(unique_records)
    strat_targets = {
        name: round(n_pre * share) for name, share in PHASE_QUOTAS.items()
    }
    quota_pool = {name: [] for name in PHASE_BUCKETS}
    for r in unique_records:
        quota_pool[bucket_of(r["phase"])].append(r)

    stratified: list[dict] = []
    for name in PHASE_BUCKETS:
        pool = sorted(quota_pool[name], key=lambda r: r["position_id"])
        stratified.extend(pool[: strat_targets[name]])

    stratified_count = len(stratified)

    # Pre-FINAL Family Gate Evaluation (only enforced if enforce_family_mix=True)
    pre_final_families: dict[str, int] = defaultdict(int)
    for r in stratified:
        fam = family_of(r["source_id"], catalog, final_mode=final_mode)
        pre_final_families[fam] += 1

    pre_final_largest_fam = max(pre_final_families.values()) if pre_final_families else 0
    pre_final_largest_share = pre_final_largest_fam / stratified_count if stratified_count else 0.0
    pre_final_family_pass = (
        (len(pre_final_families) >= MIN_FAMILIES and pre_final_largest_share <= 0.70)
        if enforce_family_mix
        else True
    )

    # 12 Core Cells Evaluation on Stratified Pool
    pools: dict[tuple[str, str], list[dict]] = {}
    family_cell_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {
            split: {phase: 0 for phase in ("high", "mid", "low", "zero")}
            for split in ("train", "validation", "holdout")
        }
    )

    for r in stratified:
        fam = family_of(r["source_id"], catalog, final_mode=final_mode)
        s = r["split"]
        b = bucket_of(r["phase"])
        pools.setdefault((s, b), []).append(r)
        family_cell_counts[fam][s][b] += 1

    cell_eval: dict[str, dict[str, dict[str, int]]] = {}
    shortfalls_count = 0
    total_shortfall = 0

    print("\n" + "=" * 80)
    print(f"TIER 2: 12-CELL AUDIT MATRIX (On Stratified Pre-FINAL Pool, Target N = {target_n})")
    print("=" * 80)
    print(f"{'Split':<12} {'Phase':<8} {'Available':<12} {'Target':<10} {'Margin':<10} {'Status'}")
    print("-" * 80)

    for split, split_want in FINAL_SPLIT_TARGETS.items():
        share = split_want / DEFAULT_TARGET
        cell_eval[split] = {}
        for bucket, bucket_want in FINAL_PHASE_TARGETS.items():
            if target_n == DEFAULT_TARGET:
                want = round(bucket_want * share)
            else:
                want = round(target_n * (split_want / DEFAULT_TARGET) * PHASE_QUOTAS[bucket])
            avail = len(pools.get((split, bucket), []))
            margin = avail - want
            status = "PASS" if margin >= 0 else "DEFICIT"
            if margin < 0:
                shortfalls_count += 1
                total_shortfall += abs(margin)
            cell_eval[split][bucket] = {
                "available": avail,
                "target": want,
                "margin": margin,
                "status": status,
            }
            print(f"{split:<12} {bucket:<8} {avail:<12} {want:<10} {margin:<+10} {status}")

    tier2_stats = {
        "stratified_total": stratified_count,
        "pre_final_families": dict(pre_final_families),
        "pre_final_family_count": len(pre_final_families),
        "pre_final_largest_share": pre_final_largest_share,
        "pre_final_family_pass": pre_final_family_pass,
        "cell_matrix": cell_eval,
        "shortfalls_count": shortfalls_count,
        "total_shortfall": total_shortfall,
        "family_cell_counts": {k: dict(v) for k, v in family_cell_counts.items()},
    }

    # =========================================================================
    # TIER 3: Exact FINAL Selection (Matching build_dataset.py lines 533-556)
    # =========================================================================
    selected: list[dict] = []
    for split, split_want in FINAL_SPLIT_TARGETS.items():
        share = split_want / DEFAULT_TARGET
        for bucket, bucket_want in FINAL_PHASE_TARGETS.items():
            if target_n == DEFAULT_TARGET:
                want = round(bucket_want * share)
            else:
                want = round(target_n * (split_want / DEFAULT_TARGET) * PHASE_QUOTAS[bucket])
            pool = sorted(pools.get((split, bucket), []), key=lambda r: r["position_id"])
            selected.extend(pool[:want])

    final_selected_count = len(selected)

    # Post-FINAL Family Gate Evaluation (only enforced if enforce_family_mix=True)
    post_final_families: dict[str, int] = defaultdict(int)
    for r in selected:
        fam = family_of(r["source_id"], catalog, final_mode=final_mode)
        post_final_families[fam] += 1

    post_final_largest_fam = max(post_final_families.values()) if post_final_families else 0
    post_final_largest_share = (
        post_final_largest_fam / final_selected_count if final_selected_count else 0.0
    )
    post_final_family_pass = (
        (len(post_final_families) >= MIN_FAMILIES and post_final_largest_share <= 0.70)
        if enforce_family_mix
        else True
    )

    tier3_stats = {
        "final_selected_count": final_selected_count,
        "target_n": target_n,
        "selection_matches_target": final_selected_count == target_n,
        "post_final_families": dict(post_final_families),
        "post_final_family_count": len(post_final_families),
        "post_final_largest_share": post_final_largest_share,
        "post_final_family_pass": post_final_family_pass,
    }

    # =========================================================================
    # OVERALL FEASIBILITY
    # =========================================================================
    is_feasible = (
        shortfalls_count == 0
        and pre_final_family_pass
        and final_selected_count == target_n
        and post_final_family_pass
    )

    print("\n" + "=" * 80)
    print("3-TIER CAPACITY SUMMARY")
    print("=" * 80)
    print(f"Tier 1 (Raw Post-Dedup):        {raw_post_dedup_count} positions")
    print(f"Tier 2 (Global Stratified):     {stratified_count} positions (Pre-FINAL Largest Family Share: {pre_final_largest_share * 100:.2f}%)")
    print(f"Tier 3 (Exact FINAL Selected):  {final_selected_count}/{target_n} positions (Post-FINAL Largest Family Share: {post_final_largest_share * 100:.2f}%)")
    print("-" * 80)
    if is_feasible:
        print("OVERALL FEASIBILITY: PASS (Ready for FINAL build)")
    else:
        print(f"OVERALL FEASIBILITY: FAIL CLOSED ({shortfalls_count} cell shortfalls, Pre-Gate: {pre_final_family_pass}, Post-Gate: {post_final_family_pass})")
    print("=" * 80)

    report = {
        "target_n": target_n,
        "total_games": sum(s["games"] for s in per_source_stats.values()),
        "reject_stats": dict(reject_stats),
        "per_source_stats": per_source_stats,
        "tier1_raw_post_dedup": tier1_stats,
        "tier2_stratified_pre_final": tier2_stats,
        "tier3_final_selected": tier3_stats,
        "is_feasible": is_feasible,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Source Pool Feasibility Profiler")
    parser.add_argument("sources", type=Path, nargs="+", help="Source directories")
    parser.add_argument("--target", type=int, default=300000, help="Target record count (default 300,000)")
    parser.add_argument("--legacy", action="store_true", help="Allow legacy family fallback")
    parser.add_argument("--enforce-family-mix", action="store_true", help="Enforce >=2 families and <=70% share")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path")

    args = parser.parse_args()
    report = profile_pool(
        source_dirs=args.sources,
        target_n=args.target,
        final_mode=not args.legacy,
        enforce_family_mix=args.enforce_family_mix,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.json}")

    if not report["is_feasible"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
