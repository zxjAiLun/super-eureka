#!/usr/bin/env python3
"""S6.0 Dataset verification: independent integrity audit.

Checks: manifest hash, record count, uniqueness, eligibility re-validation,
split isolation (position_id AND game_id), phase buckets vs quotas, labels
completeness, teacher determinism record. Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import chess

PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7), "zero": (0, 0)}
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def phase_of(board: chess.Board) -> int:
    total = 0
    for _, piece in board.piece_map().items():
        total += PHASE_WEIGHTS.get(piece.piece_type, 0)
    return min(24, total)


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


def load_records(dataset_dir: Path) -> list[dict]:
    records = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def verify(args) -> int:
    dataset_dir = Path(args.dataset)
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    records = load_records(dataset_dir)
    failures: list[str] = []

    # 1. canonical hash
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    actual_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual_sha != manifest["dataset_sha256"]:
        failures.append(f"dataset_sha256 mismatch: {actual_sha[:16]}")

    # 2. counts
    if len(records) != manifest["records_total"]:
        failures.append(f"records_total {len(records)} != manifest {manifest['records_total']}")

    # 3. uniqueness
    ids = [r["position_id"] for r in records]
    if len(set(ids)) != len(ids):
        failures.append("duplicate position_id in final records")

    # 4. eligibility re-validation (full pass; 5.9k is fast)
    for i, r in enumerate(records):
        try:
            board = chess.Board(r["fen"])
        except ValueError:
            failures.append(f"unparseable fen {i}")
            continue
        if board.is_game_over(claim_draw=False) or board.is_check():
            failures.append(f"ineligible (terminal/in-check) {r['position_id']}")
        if not any(board.legal_moves):
            failures.append(f"no legal move {r['position_id']}")
        if phase_of(board) != r["phase"]:
            failures.append(f"phase mismatch {r['position_id']}")
        if r["canonical_fen4"] != " ".join(board.fen().split(" ")[:4]):
            failures.append(f"canonical_fen4 mismatch {r['position_id']}")
        if r["position_id"] != hashlib.sha256(
                r["canonical_fen4"].encode("utf-8")).hexdigest():
            failures.append(f"position_id mismatch {r['position_id']}")
    if len(failures) > 10:
        failures = failures[:10] + [f"... ({len(failures)} total)"]

    # 5. split isolation
    splits: dict[str, set] = {s: set() for s in ("train", "validation", "holdout")}
    games: dict[str, set] = {s: set() for s in splits}
    for r in records:
        splits[r["split"]].add(r["position_id"])
        games[r["split"]].add(r["source_game_id"])
    for a in splits:
        for b in splits:
            if a < b:
                if splits[a] & splits[b]:
                    failures.append(f"cross-split position overlap {a}/{b}")
                if games[a] & games[b]:
                    failures.append(f"cross-split game overlap {a}/{b}")

    # 6. phase quotas
    buckets = {name: 0 for name in PHASE_BUCKETS}
    for r in records:
        buckets[bucket_of(r["phase"])] += 1
    manifest_buckets = manifest["phase_buckets"]
    if buckets != manifest_buckets:
        failures.append(f"phase buckets {buckets} != manifest {manifest_buckets}")

    # 7. labels
    labels_path = dataset_dir / "labels.jsonl"
    if labels_path.is_file():
        labels = {}
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[rec["position_id"]] = rec
        missing = set(ids) - set(labels)
        extra = set(labels) - set(ids)
        if missing:
            failures.append(f"{len(missing)} records without labels")
        if extra:
            failures.append(f"{len(extra)} labels without records")
        tm = json.loads(
            (dataset_dir / "teacher_manifest.json").read_text(encoding="utf-8"))
        if not tm.get("audit", {}).get("ok"):
            failures.append("teacher determinism audit not ok")
    else:
        failures.append("labels.jsonl missing")

    print(f"records: {len(records)}  splits: "
          f"{ {s: len(v) for s, v in splits.items()} }")
    print(f"phase buckets: {buckets}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("VERIFY_PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    return verify(parser.parse_args(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
