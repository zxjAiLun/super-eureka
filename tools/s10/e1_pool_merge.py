"""S10-E1: global merge of per-source caches + position dedup + 12-cell
capacity profile.

Cell capacities may ONLY be announced after the GLOBAL position_id
dedup — summing per-source cell counts would ignore cross-source
duplicates.

Dedup rule (frozen from build_dataset): for duplicate position_ids,
keep the copy from the earliest (source_game_id) in deterministic
order — matching the B1 builder's cross-split duplicate policy of
keeping the copy in the earliest game's split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SPLIT_TARGETS_1M = {
    "train": 800_000,
    "validation": 100_000,
    "holdout": 100_000,
}
PHASE_TARGETS_1M = {
    "high": 250_000,
    "mid": 450_000,
    "low": 200_000,
    "zero": 100_000,
}
CELL_TARGETS = {
    split: {
        phase: int(
            PHASE_TARGETS_1M[phase]
            * split_n
            / SPLIT_TARGETS_1M[split]
            * SPLIT_TARGETS_1M[split]
            // SPLIT_TARGETS_1M[split]
        )
        for phase in PHASE_TARGETS_1M
    }
    for split, split_n in SPLIT_TARGETS_1M.items()
}
# Simpler: per-cell target = phase_total * split_share (80/10/10).
CELL_TARGETS = {
    split: {
        phase: int(PHASE_TARGETS_1M[phase] * share)
        for phase in PHASE_TARGETS_1M
    }
    for split, share in (
        ("train", 0.8), ("validation", 0.1), ("holdout", 0.1)
    )
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/s10/e1-pool-cache")
    )
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    manifests = sorted(args.cache_dir.glob("*.manifest.json"))
    if not manifests:
        print("no source caches found")
        return 1

    total_seen = 0
    dup_removed = 0
    cells: Counter = Counter()
    kept_by_split: Counter = Counter()
    per_source_kept: Counter = Counter()

    # Global dedup: keep first occurrence in deterministic
    # (position_id, source_game_id) order — load ALL, sort, dedup.
    all_records = []
    for mf in manifests:
        m = json.loads(mf.read_text(encoding="utf-8"))
        cands = args.cache_dir / f"{m['source_name']}.jsonl"
        with open(cands, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    all_records.append(json.loads(line))
        total_seen += m["candidate_count"]

    # Order deterministically: by (position_id, source_game_id) so the
    # 'earliest game' copy wins the dedup, mirroring the B1 policy.
    all_records.sort(key=lambda r: (r["position_id"], r["source_game_id"]))

    # bucket_of: phase int -> high/mid/low/zero (frozen contract).
    PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7),
                     "zero": (0, 0)}

    def bucket_of(phase: int) -> str:
        for name, (lo, hi) in PHASE_BUCKETS.items():
            if lo <= phase <= hi:
                return name
        return "mid"

    seen: set[str] = set()
    for r in all_records:
        pid = r["position_id"]
        if pid in seen:
            dup_removed += 1
            continue
        seen.add(pid)
        cells[(r["split"], bucket_of(r["phase"]))] += 1
        kept_by_split[r["split"]] += 1
        per_source_kept[r["source_id"]] += 1

    # 12-cell matrix
    matrix = {}
    deficits = {}
    for split in ("train", "validation", "holdout"):
        for phase in ("high", "mid", "low", "zero"):
            avail = cells[(split, phase)]
            target = CELL_TARGETS[split][phase]
            matrix[f"{split}/{phase}"] = {
                "available": avail,
                "target": target,
                "margin": avail - target,
                "status": "OK" if avail >= target else "DEFICIT",
            }
            if avail < target:
                deficits[f"{split}/{phase}"] = target - avail

    summary = {
        "sources": len(manifests),
        "candidates_seen": total_seen,
        "duplicates_removed": dup_removed,
        "unique_positions": len(seen),
        "kept_by_split": dict(kept_by_split),
        "cell_matrix": matrix,
        "deficits": deficits,
        "feasible": not deficits,
    }
    print(json.dumps(summary, indent=2))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {**summary, "per_source_kept": dict(per_source_kept)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"written {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
