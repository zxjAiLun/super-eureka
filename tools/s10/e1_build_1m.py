"""S10-E1: nested 1M dataset builder.

Frozen contract (S10-E1):
  dataset_id     = s10-eval-v2-1m01
  records_total  = 1,000,000
  train/validation/holdout = 800,000 / 100,000 / 100,000
  phase quotas   = high 250k, mid 450k, low 200k, zero 100k

NESTED: every record of the frozen 300k dataset
(s10-eval-v1-300k01) is kept VERBATIM — same position_id, same split,
same record bytes. The additional 700k records are drawn so that:

  * high/mid/low cells are filled ONLY from the old B1 pool sources
    (v1/confirm/v2/v3/v4 + arena) — the v5 long-game source does NOT
    contribute to these cells (it would shift source composition);
  * zero cells draw from the old pool FIRST, then from v5 only for the
    remaining shortfall;
  * a position is eligible only if its position_id is NOT already in
    the 300k dataset and not already selected for the 700k;
  * games never cross splits (the split is a function of game_id, so
    this holds by construction);
  * selection within a cell is deterministic:
    sort by (ply_priority, ply) with DATASET_SEED-derived priority —
    the same deterministic top-K rule the B1 builder used — then take
    the needed count.

The result is written in the same on-disk schema as the 300k dataset
(part-*.jsonl shards + dataset_manifest.json); labels.jsonl and
teacher_manifest.json are NOT written (that is E2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s6.build_dataset import (  # noqa: E402
    DATASET_SEED,
    MAX_PER_GAME,
    MAX_PLY,
    MIN_PLY,
    SAMPLING_VERSION,
    SCHEMA_VERSION,
    ply_priority,
    sha256_text,
)

PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7),
                 "zero": (0, 0)}
BUCKET_ORDER = ["high", "mid", "low", "zero"]
SPLIT_SHARE = {"train": 0.8, "validation": 0.1, "holdout": 0.1}
PHASE_TOTALS_1M = {"high": 250_000, "mid": 450_000, "low": 200_000,
                   "zero": 100_000}
RECORDS_TOTAL = 1_000_000

DATASET_ID = "s10-eval-v2-1m01"
SEED_1M = 2026083002  # frozen for the 700k expansion draw

OLD_DATASET_DIR = Path("data/s10/s10-eval-v1-300k01")
CACHE_DIR = Path("data/s10/e1-pool-cache")
V5_SOURCE_ID = "lichess-standard-rated-v5"


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


def compute_dataset_sha(records: list[dict]) -> str:
    """Same canonical dataset SHA as verify_dataset.py: sha256 over
    json.dumps(r, ensure_ascii=False, sort_keys=True) + newline per
    record, in file order."""
    h = hashlib.sha256()
    for r in records:
        h.update(
            (json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            .encode("utf-8")
        )
    return h.hexdigest()


def load_pool(cache_dir: Path) -> list[dict]:
    records = []
    for mf in sorted(cache_dir.glob("*.manifest.json")):
        name = mf.stem.replace(".manifest", "")
        cands = cache_dir / f"{name}.jsonl"
        with open(cands, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-records", type=int, default=250_000)
    args = parser.parse_args()

    out = args.out
    if out.exists():
        raise SystemExit(f"FAIL CLOSED: {out} already exists")

    # 1. Load the frozen 300k dataset and index its records.
    print("loading frozen 300k dataset...", flush=True)
    old_records = []
    old_ids: set[str] = set()
    for shard in sorted(OLD_DATASET_DIR.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                old_records.append(rec)
                old_ids.add(rec["position_id"])
    old_manifest = json.loads(
        (OLD_DATASET_DIR / "dataset_manifest.json").read_text("utf-8")
    )
    assert old_manifest["records_total"] == len(old_records) == 300_000

    # Frozen cells occupied by the 300k.
    old_cells: dict[tuple[str, str], int] = defaultdict(int)
    for r in old_records:
        old_cells[(r["split"], bucket_of(r["phase"]))] += 1
    print("frozen 300k cells:", dict(old_cells))

    # 2. Targets per cell.
    cell_targets = {
        (split, phase): int(PHASE_TOTALS_1M[phase] * share)
        for split, share in SPLIT_SHARE.items()
        for phase in BUCKET_ORDER
    }
    # correct rounding so each split sums to its share
    split_totals = {"train": 800_000, "validation": 100_000,
                    "holdout": 100_000}
    for split in split_totals:
        # rescale to exact totals
        raw = {p: cell_targets[(split, p)] for p in BUCKET_ORDER}
        s = sum(raw.values())
        adj = split_totals[split] - s
        raw["mid"] += adj  # mid is the largest cell; absorb rounding
        for p in BUCKET_ORDER:
            cell_targets[(split, p)] = raw[p]

    # 3. Load pool, exclude frozen ids.
    print("loading pool cache...", flush=True)
    pool = load_pool(CACHE_DIR)

    # Global dedup on the pool (same rule as merge): keep the earliest
    # source_game_id copy.
    pool.sort(key=lambda r: (r["position_id"], r["source_game_id"]))
    deduped: list[dict] = []
    seen: set[str] = set()
    for r in pool:
        if r["position_id"] in seen:
            continue
        seen.add(r["position_id"])
        if r["position_id"] in old_ids:
            continue
        deduped.append(r)
    print(f"pool unique non-old positions: {len(deduped)}")

    # 4. Cell buckets with source policy:
    #    high/mid/low: old-pool sources ONLY (exclude v5)
    #    zero: old-pool first, v5 for the remainder.
    cell_pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in deduped:
        split = r["split"]
        phase = bucket_of(r["phase"])
        is_v5 = r["source_id"] == V5_SOURCE_ID
        if phase != "zero" and is_v5:
            continue
        cell_pools[(split, phase)].append(r)

    # 5. Deterministic selection: top-K by (ply_priority, ply) with the
    #    1M seed — the same rule shape as the B1 builder, but seeded for
    #    the expansion draw.
    new_records: list[dict] = []
    shortfall_report = {}
    for split in ("train", "validation", "holdout"):
        for phase in BUCKET_ORDER:
            key = (split, phase)
            need = cell_targets[key] - old_cells.get(key, 0)
            if need <= 0:
                shortfall_report[f"{split}/{phase}"] = {
                    "need": 0, "available": len(cell_pools.get(key, [])),
                    "taken_old_pool": 0, "taken_v5": 0,
                }
                continue
            candidates = cell_pools.get(key, [])
            # split zero: old-pool first
            if phase == "zero":
                old_pool = [r for r in candidates
                            if r["source_id"] != V5_SOURCE_ID]
                v5_pool = [r for r in candidates
                           if r["source_id"] == V5_SOURCE_ID]
            else:
                old_pool = candidates
                v5_pool = []
            for lst in (old_pool, v5_pool):
                lst.sort(
                    key=lambda r: (
                        ply_priority(r["source_game_id"], r["ply"], SEED_1M),
                        r["ply"],
                    )
                )
            taken_old = old_pool[:need]
            remaining = need - len(taken_old)
            taken_v5 = v5_pool[:remaining] if remaining > 0 else []
            new_records.extend(taken_old)
            new_records.extend(taken_v5)
            shortfall_report[f"{split}/{phase}"] = {
                "need": need,
                "available": len(candidates),
                "taken_old_pool": len(taken_old),
                "taken_v5": len(taken_v5),
                "unmet": need - len(taken_old) - len(taken_v5),
            }

    total = len(old_records) + len(new_records)
    print(json.dumps(shortfall_report, indent=2))
    if total != RECORDS_TOTAL:
        print(f"FAIL CLOSED: total {total} != {RECORDS_TOTAL}")
        return 4
    unmet = sum(v.get("unmet", 0) for v in shortfall_report.values())
    if unmet:
        print(f"FAIL CLOSED: {unmet} unmet cell requirements")
        return 4

    # 6. Serialize: old records VERBATIM (same JSON encoding as the
    #    source shards), then new records sorted deterministically.
    old_lines = []
    for shard in sorted(OLD_DATASET_DIR.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old_lines.append(line)
    new_records.sort(
        key=lambda r: (r["position_id"], r["source_game_id"])
    )
    new_lines = [
        json.dumps(r, sort_keys=True, separators=(",", ":"))
        for r in new_records
    ]
    all_lines = old_lines + new_lines
    assert len(all_lines) == RECORDS_TOTAL

    out.mkdir(parents=True)
    shard_idx = 0
    for i in range(0, len(all_lines), args.shard_records):
        shard_lines = all_lines[i:i + args.shard_records]
        (out / f"part-{shard_idx:04d}.jsonl").write_text(
            "\n".join(shard_lines) + "\n", encoding="utf-8"
        )
        shard_idx += 1

    dataset_sha = compute_dataset_sha(
        [json.loads(l) for l in all_lines]
    )

    split_counts = defaultdict(int)
    phase_counts = defaultdict(int)
    for line in all_lines:
        r = json.loads(line)
        split_counts[r["split"]] += 1
        phase_counts[bucket_of(r["phase"])] += 1

    manifest = {
        "dataset_id": DATASET_ID,
        "final": True,
        "final_target": RECORDS_TOTAL,
        "schema_version": SCHEMA_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "sampling_method": "deterministic_hash_top_k",
        "dataset_seed": DATASET_SEED,
        "expansion_seed": SEED_1M,
        "nested_base": {
            "dataset_id": old_manifest["dataset_id"],
            "dataset_sha256": old_manifest["dataset_sha256"],
            "records_total": old_manifest["records_total"],
            "preserved_verbatim": True,
        },
        "min_ply": MIN_PLY,
        "max_ply": MAX_PLY,
        "max_per_game": MAX_PER_GAME,
        "records_total": RECORDS_TOTAL,
        "games_parsed": None,
        "splits": dict(split_counts),
        "phase_buckets": dict(phase_counts),
        "phase_quotas": {"high": 0.25, "mid": 0.45, "low": 0.20,
                         "zero": 0.10},
        "phase_targets": dict(PHASE_TOTALS_1M),
        "cell_fill": shortfall_report,
        "source_policy": (
            "high/mid/low filled only from B1 pool sources; zero cells "
            f"old-pool first then {V5_SOURCE_ID}"
        ),
        "v5_source": {
            "source_id": V5_SOURCE_ID,
            "all_long_games": True,
            "long_min_plies": 100,
            "accept_byte": "0x1F",
            "selection_seed": 20260830,
            "archive_month": "2026-07",
            "archive_sha256": (
                "68738b1c448f051dc8d42db645d5b01749988a3b"
                "c1c24981adfe44ea92060dc7"
            ),
        },
        "created_utc": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()
        ),
        "dataset_sha256": dataset_sha,
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "dataset_id": DATASET_ID,
        "records_total": RECORDS_TOTAL,
        "dataset_sha256": dataset_sha,
        "split_counts": dict(split_counts),
        "phase_counts": dict(phase_counts),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
