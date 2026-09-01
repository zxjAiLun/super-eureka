"""S10-E1 (Repair 1): nested 1M dataset builder — B1 position_id rule.

Repair 1 restores the frozen B1 selection rule for BOTH the parent
reconstruction and the extension, removing the exploratory
`SEED_1M / ply_priority` drift:

  * the parent 300k is RECONSTRUCTED from the old 24-source global-dedup
    pool under the original B1 rule — per split x phase cell, sort by
    position_id, take the original cell quota — and must match the
    frozen dataset 300,000/300,000 (position_id, split and core record
    fields) with 0 mismatches;
  * the +700k extension continues the SAME rule on the remaining old
    pool: per cell, the next positions in position_id order;
  * zero cells take the old pool first and only the shortfall from v5,
    also in position_id order;
  * the 300k frozen records are then written VERBATIM (the
    reconstruction is an equivalence CHECK, not the source of the
    parent bytes).

Fail-close cache validation: every pool cache manifest is verified
(frozen contract, candidate_count, candidates_sha256, source_id)
before its JSONL is trusted.

The manifest records the 12-cell matrix, the parent-reconstruction
result, extension source counts, shard names + SHAs.
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
)
from tools.s10.e1_pool_extract import CACHE_CONTRACT  # noqa: E402

PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7),
                 "zero": (0, 0)}
BUCKET_ORDER = ["high", "mid", "low", "zero"]
SPLIT_TOTALS = {"train": 800_000, "validation": 100_000,
                "holdout": 100_000}
PHASE_TOTALS_1M = {"high": 250_000, "mid": 450_000, "low": 200_000,
                   "zero": 100_000}
RECORDS_TOTAL = 1_000_000

DATASET_ID = "s10-eval-v2-1m01"
OLD_DATASET_DIR = Path("data/s10/s10-eval-v1-300k01")
CACHE_DIR = Path("data/s10/e1-pool-cache")
V5_SOURCE_ID = "lichess-standard-rated-v5"

# B1 frozen parent cell quotas (split x phase).
PARENT_CELL_QUOTA = {
    ("train", "high"): 60_000, ("train", "mid"): 108_000,
    ("train", "low"): 48_000, ("train", "zero"): 24_000,
    ("validation", "high"): 7_500, ("validation", "mid"): 13_500,
    ("validation", "low"): 6_000, ("validation", "zero"): 3_000,
    ("holdout", "high"): 7_500, ("holdout", "mid"): 13_500,
    ("holdout", "low"): 6_000, ("holdout", "zero"): 3_000,
}


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


def compute_dataset_sha(records: list[dict]) -> str:
    """Canonical dataset SHA (same as verify_dataset.py)."""
    h = hashlib.sha256()
    for r in records:
        h.update(
            (json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            .encode("utf-8")
        )
    return h.hexdigest()


def load_pool_verified(cache_dir: Path) -> tuple[list[dict], dict]:
    """Load every cache with full fail-close validation.

    Checks per source: manifest contract == frozen CACHE_CONTRACT,
    candidate_count matches the JSONL line count, candidates_sha256
    matches the actual JSONL bytes, source_id in every record equals
    the manifest's source_id. Returns (records, per-source report).
    """
    records: list[dict] = []
    report: dict = {}
    for mf in sorted(cache_dir.glob("*.manifest.json")):
        name = mf.stem.replace(".manifest", "")
        m = json.loads(mf.read_text(encoding="utf-8"))
        cands = cache_dir / f"{name}.jsonl"
        if not cands.is_file():
            raise SystemExit(f"FAIL CLOSED: cache JSONL missing for {name}")
        if m.get("contract") != CACHE_CONTRACT:
            raise SystemExit(
                f"FAIL CLOSED: cache contract drift for {name}: "
                f"{m.get('contract')}"
            )
        data = cands.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if m.get("candidates_sha256") != actual_sha:
            raise SystemExit(
                f"FAIL CLOSED: cache SHA mismatch for {name}: "
                f"{m.get('candidates_sha256', '?')[:12]} != "
                f"{actual_sha[:12]}"
            )
        lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
        if m.get("candidate_count") != len(lines):
            raise SystemExit(
                f"FAIL CLOSED: cache count mismatch for {name}: "
                f"{m.get('candidate_count')} != {len(lines)}"
            )
        src_records = [json.loads(ln) for ln in lines]
        for r in src_records:
            if r["source_id"] != m["source_id"]:
                raise SystemExit(
                    f"FAIL CLOSED: record source_id drift in {name}: "
                    f"{r['source_id']}"
                )
        records.extend(src_records)
        report[name] = {
            "source_id": m["source_id"],
            "pgn_sha256": m.get("pgn_sha256"),
            "candidate_count": len(lines),
        }
    return records, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-records", type=int, default=250_000)
    args = parser.parse_args()

    out = args.out
    if out.exists():
        raise SystemExit(f"FAIL CLOSED: {out} already exists")

    # 1. Frozen 300k dataset.
    print("loading frozen 300k dataset...", flush=True)
    old_records = []
    old_by_pid: dict[str, dict] = {}
    for shard in sorted(OLD_DATASET_DIR.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                old_records.append(rec)
                old_by_pid[rec["position_id"]] = rec
    old_manifest = json.loads(
        (OLD_DATASET_DIR / "dataset_manifest.json").read_text("utf-8")
    )
    assert old_manifest["records_total"] == len(old_records) == 300_000

    # 2. Load + verify pool caches; global dedup (earliest game wins).
    print("loading (verified) pool cache...", flush=True)
    pool, cache_report = load_pool_verified(CACHE_DIR)
    pool.sort(key=lambda r: (r["position_id"], r["source_game_id"]))
    unique: list[dict] = []
    seen: set[str] = set()
    for r in pool:
        if r["position_id"] in seen:
            continue
        seen.add(r["position_id"])
        unique.append(r)
    print(f"pool: {len(pool)} candidates -> {len(unique)} unique")

    # 2b. The frozen B1 chain applies a GLOBAL phase-quota stratification
    # BEFORE the final split x phase downsample (build_dataset.py lines
    # 375-387: per-bucket targets = round(n_pre * PHASE_QUOTAS), sort by
    # position_id, take the bucket quota). The frozen 300k is a subset of
    # that stratified pool, NOT of the raw deduped pool — reconstructing
    # without this step mismatches ~50% of every cell. Reproduce it
    # exactly, over the OLD-B1-sources pool only (v5 joins later, at the
    # zero-cell shortfall fill).
    from tools.s6.build_dataset import PHASE_QUOTAS as B1_PHASE_QUOTAS
    from tools.s6.build_dataset import PHASE_BUCKETS as B1_PHASE_BUCKETS
    old_pool = [r for r in unique if r["source_id"] != V5_SOURCE_ID]
    n_pre = len(old_pool)
    global_targets = {
        name: round(n_pre * share)
        for name, share in B1_PHASE_QUOTAS.items()
    }
    quota_pool: dict[str, list[dict]] = {name: [] for name in B1_PHASE_BUCKETS}
    for r in old_pool:
        quota_pool[bucket_of(r["phase"])].append(r)
    stratified_old: list[dict] = []
    for name in BUCKET_ORDER:
        p = sorted(quota_pool[name], key=lambda r: r["position_id"])
        stratified_old.extend(p[: global_targets[name]])
    print(
        f"B1 global stratification: old pool {n_pre} -> "
        f"{len(stratified_old)} (targets {global_targets})"
    )

    # 3. Parent reconstruction under the B1 rule: per cell, sort by
    #    position_id, take the original quota.
    cell_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in stratified_old:
        cell_records[(r["split"], bucket_of(r["phase"]))].append(r)
    # zero cells may still need v5 for the 1M extension later; index the
    # v5 candidates separately, also cell-partitioned.
    v5_by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in unique:
        if r["source_id"] == V5_SOURCE_ID:
            v5_by_cell[(r["split"], bucket_of(r["phase"]))].append(r)
    for lst in cell_records.values():
        lst.sort(key=lambda r: r["position_id"])
    for lst in v5_by_cell.values():
        lst.sort(key=lambda r: r["position_id"])

    reconstructed: list[dict] = []
    for split in ("train", "validation", "holdout"):
        for phase in BUCKET_ORDER:
            quota = PARENT_CELL_QUOTA[(split, phase)]
            lst = cell_records[(split, phase)]
            if len(lst) < quota:
                raise SystemExit(
                    f"FAIL CLOSED: parent cell {split}/{phase} has only "
                    f"{len(lst)} < {quota} — pool drifted"
                )
            reconstructed.extend(lst[:quota])
    if len(reconstructed) != 300_000:
        raise SystemExit(
            f"FAIL CLOSED: reconstructed parent {len(reconstructed)} != 300000"
        )

    # Equivalence vs the frozen dataset.
    recon_by_pid = {r["position_id"]: r for r in reconstructed}
    if set(recon_by_pid) != set(old_by_pid):
        missing = set(old_by_pid) - set(recon_by_pid)
        extra = set(recon_by_pid) - set(old_by_pid)
        raise SystemExit(
            f"FAIL CLOSED: parent reconstruction id mismatch "
            f"({len(missing)} missing, {len(extra)} extra)"
        )
    CORE_FIELDS = (
        "position_id", "canonical_fen4", "fen", "source_id",
        "source_game_id", "ply", "game_result_white", "phase", "split",
    )
    mismatches = 0
    for pid, old in old_by_pid.items():
        rec = recon_by_pid[pid]
        for f in CORE_FIELDS:
            if rec[f] != old[f]:
                mismatches += 1
                break
    if mismatches:
        raise SystemExit(
            f"FAIL CLOSED: parent reconstruction {mismatches} record "
            "mismatches"
        )
    parent_split_ok = all(
        recon_by_pid[pid]["split"] == old_by_pid[pid]["split"]
        for pid in old_by_pid
    )
    if not parent_split_ok:
        raise SystemExit("FAIL CLOSED: parent split mismatch")
    print(f"parent reconstruction: 300000/300000 exact, 0 mismatches")

    # 4. Extension: continue the SAME position_id rule beyond the
    #    parent quota. high/mid/low: old-pool sources only; zero:
    #    old-pool first, v5 for the shortfall.
    parent_ids = set(old_by_pid)
    new_records: list[dict] = []
    cell_fill = {}
    for split in SPLIT_TOTALS:
        for phase in BUCKET_ORDER:
            key = (split, phase)
            target_1m = int(
                PHASE_TOTALS_1M[phase] * SPLIT_TOTALS[split] / 1_000_000
            )
            # exact-split rescale (mid absorbs rounding)
            raw = {
                p: int(PHASE_TOTALS_1M[p] * SPLIT_TOTALS[split] / 1_000_000)
                for p in BUCKET_ORDER
            }
            adj = SPLIT_TOTALS[split] - sum(raw.values())
            raw["mid"] += adj
            target_1m = raw[phase]

            used_parent = PARENT_CELL_QUOTA[key]
            need = target_1m - used_parent
            taken_old = taken_v5 = 0
            if need > 0:
                # remaining stratified-old-pool candidates in this cell,
                # in position_id order, excluding parent ids
                lst = cell_records[key]
                old_rest = [
                    r for r in lst
                    if r["position_id"] not in parent_ids
                ]
                if phase == "zero":
                    # zero cells take the old pool first, then v5 for the
                    # shortfall (v5 is a long-game source whose value is
                    # precisely its endgame density)
                    v5_rest = v5_by_cell.get(key, [])
                else:
                    # non-zero: v5 must NOT contribute
                    v5_rest = []
                # both lists are already position_id-sorted
                take_old = old_rest[:need]
                taken_old = len(take_old)
                remaining = need - taken_old
                take_v5 = v5_rest[:remaining] if remaining > 0 else []
                taken_v5 = len(take_v5)
                new_records.extend(take_old)
                new_records.extend(take_v5)
            cell_fill[f"{split}/{phase}"] = {
                "parent": used_parent,
                "target_1m": target_1m,
                "extension_old_pool": taken_old,
                "extension_v5": taken_v5,
                "unmet": max(0, need - taken_old - taken_v5),
            }

    unmet = sum(v["unmet"] for v in cell_fill.values())
    total = 300_000 + len(new_records)
    if unmet or total != RECORDS_TOTAL:
        raise SystemExit(
            f"FAIL CLOSED: unmet={unmet} total={total} != {RECORDS_TOTAL}"
        )

    # 5. Serialize: parent verbatim + extension (position_id order).
    old_lines = []
    for shard in sorted(OLD_DATASET_DIR.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old_lines.append(line)
    new_records.sort(key=lambda r: r["position_id"])
    new_lines = [
        json.dumps(r, ensure_ascii=False, sort_keys=True)
        for r in new_records
    ]
    all_lines = old_lines + new_lines
    assert len(all_lines) == RECORDS_TOTAL

    out.mkdir(parents=True)
    shard_names = []
    shard_hashes = {}
    shard_idx = 0
    for i in range(0, len(all_lines), args.shard_records):
        shard_lines = all_lines[i:i + args.shard_records]
        name = f"part-{shard_idx:04d}.jsonl"
        data = ("\n".join(shard_lines) + "\n").encode("utf-8")
        (out / name).write_bytes(data)
        shard_names.append(name)
        shard_hashes[name] = hashlib.sha256(data).hexdigest()
        shard_idx += 1

    dataset_sha = compute_dataset_sha(
        [json.loads(l) for l in all_lines]
    )

    split_counts = defaultdict(int)
    phase_split_counts = {}
    for split in SPLIT_TOTALS:
        phase_split_counts[split] = {}
        for phase in BUCKET_ORDER:
            phase_split_counts[split][phase] = PARENT_CELL_QUOTA[
                (split, phase)
            ]
    for r in new_records:
        split_counts[r["split"]] += 1
        phase_split_counts[r["split"]][bucket_of(r["phase"])] += 1
    for r in old_records:
        split_counts[r["split"]] += 1

    v5_count = sum(1 for r in new_records
                   if r["source_id"] == V5_SOURCE_ID)
    manifest = {
        "dataset_id": DATASET_ID,
        "final": True,
        "final_target": RECORDS_TOTAL,
        "schema_version": SCHEMA_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "sampling_method": "deterministic_hash_top_k",
        "dataset_seed": DATASET_SEED,
        "selection_rule": (
            "B1 rule continuation: per split x phase cell, sort by "
            "position_id; parent = first 300k quota (reconstructed and "
            "verified 300000/300000 exact), extension = the next "
            "positions in the same order; zero cells old-pool first, "
            "v5 only for the shortfall"
        ),
        "supersedes": {
            "dataset_sha256": (
                "4eda8fe1d241d418071888dec661b00fa8bd6000c07abce242"
                "1cec668ec53de0"
            ),
            "reason": "extension-selection rule drift (exploratory "
                      "ply_priority seed replaced by the frozen B1 "
                      "position_id continuation)",
        },
        "nested_base": {
            "dataset_id": old_manifest["dataset_id"],
            "dataset_sha256": old_manifest["dataset_sha256"],
            "records_total": old_manifest["records_total"],
            "preserved_verbatim": True,
        },
        "nested_parent_reconstruction": {
            "reconstructed": len(reconstructed),
            "matched": 300_000,
            "mismatches": 0,
            "split_preserved": True,
        },
        "min_ply": MIN_PLY,
        "max_ply": MAX_PLY,
        "max_per_game": MAX_PER_GAME,
        "records_total": RECORDS_TOTAL,
        "games_parsed": None,
        "splits": dict(split_counts),
        "phase_buckets": {
            p: sum(phase_split_counts[s][p] for s in phase_split_counts)
            for p in BUCKET_ORDER
        },
        "phase_split_counts": phase_split_counts,
        "phase_quotas": {"high": 0.25, "mid": 0.45, "low": 0.20,
                         "zero": 0.10},
        "phase_targets": dict(PHASE_TOTALS_1M),
        "cell_fill": cell_fill,
        "extension_source_counts": {
            "old_pool": len(new_records) - v5_count,
            "v5": v5_count,
        },
        "source_policy": (
            "high/mid/low extension only from B1 pool sources; zero "
            f"cells old-pool first then {V5_SOURCE_ID} for the shortfall"
        ),
        "v5_source": {
            "source_id": V5_SOURCE_ID,
            "all_long_games": True,
            "long_min_plies": 100,
            "accept_byte": "0x1F",
            "selection_seed": 2026083002,
            "archive_month": "2026-07",
            "archive_sha256": (
                "68738b1c448f051dc8d42db645d5b01749988a3b"
                "c1c24981adfe44ea92060dc7"
            ),
        },
        "pool_cache": {
            "sources": len(cache_report),
            "validated": "contract + count + sha256 + source_id per source",
        },
        "shards": shard_names,
        "shard_hashes": shard_hashes,
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
        "splits": dict(split_counts),
        "phase_split_counts": phase_split_counts,
        "extension_source_counts": manifest["extension_source_counts"],
        "supersedes": manifest["supersedes"]["dataset_sha256"][:16],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
