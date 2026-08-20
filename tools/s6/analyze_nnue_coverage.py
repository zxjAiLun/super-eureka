#!/usr/bin/env python3
"""S6-N3A NNUE feature-coverage analysis + pilot hard-gate evaluator.

Feature encoding stays in Rust: this script only calls `nnue-features-batch`
via the engine and never re-implements orientation / king buckets / relative
channels / feature_index.

Outputs (per split, all usable rows):
  - train union unique feature count and fraction of 40960
  - validation / holdout unseen-activation counts, rates, positions with
    unseen features
  - records by source family / phase bucket / split (and family x split)

With --gate, evaluates the frozen pilot hard gates and writes
results/s6/s6-n3a-data-pilot.{json,md} with status DATA_PILOT_PASS or
DATA_PILOT_FAIL (no auto-tuning, no retry).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as bd  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

PILOT_MIN_RECORDS = 10_000
PILOT_FAMILIES = {"arena", "lichess-standard-rated-v1"}
PILOT_FAMILY_MIN_SHARE = 0.30
PILOT_FAMILY_MAX_SHARE = 0.70
PILOT_LOW_ZERO_MIN = 0.10
PILOT_TRAIN_UNION_MIN = 6500


def load_dataset_records(dataset_dir: Path) -> dict:
    """Labels-free loader: shards + canonical dataset SHA (no labels needed)."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    actual = probe.compute_dataset_sha(records)
    if actual != manifest["dataset_sha256"]:
        raise SystemExit(
            f"PIPELINE_FAILURE: dataset_sha256 mismatch {actual[:16]} "
            f"!= manifest {manifest['dataset_sha256'][:16]}")
    return {"records": records, "manifest": manifest, "dataset_sha": actual}


def analyze(engine: Path, dataset_dir: Path, sources: list[str]) -> dict:
    data = load_dataset_records(dataset_dir)
    records = data["records"]
    catalog = bd.load_source_catalog([Path(s) for s in sources])

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "holdout": []}
    for r in records:
        splits[r["split"]].append(r)

    exported = probe.export_all_features(engine, records)

    train_union: set[int] = set()
    for r in splits["train"]:
        rec = exported[r["position_id"]]
        train_union |= {int(i) for i in rec["white"]}
        train_union |= {int(i) for i in rec["black"]}
    coverage = {
        "train": probe.coverage_for_split(exported, splits["train"], None),
        "validation": probe.coverage_for_split(
            exported, splits["validation"], train_union),
        "holdout": probe.coverage_for_split(
            exported, splits["holdout"], train_union),
    }
    coverage["train"]["union_fraction"] = round(
        len(train_union) / probe.NNUE_INPUTS, 6)
    coverage["train"]["union_unique"] = len(train_union)
    coverage["train"]["activation_frequency"] = probe.train_activation_frequency(
        exported, splits["train"])

    per_family: dict[str, int] = {}
    per_phase: dict[str, int] = {}
    per_split: dict[str, int] = {}
    family_split: dict[str, dict[str, int]] = {}
    for r in records:
        fam = bd.family_of(r["source_id"], catalog)
        phase = bd.bucket_of(r["phase"])
        per_family[fam] = per_family.get(fam, 0) + 1
        per_phase[phase] = per_phase.get(phase, 0) + 1
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
        fam_bucket = family_split.setdefault(fam, {})
        fam_bucket[r["split"]] = fam_bucket.get(r["split"], 0) + 1

    return {
        "dataset_sha256": data["dataset_sha"],
        "dataset_id": data["manifest"]["dataset_id"],
        "records_total": len(records),
        "per_family": per_family,
        "per_phase": per_phase,
        "per_split": per_split,
        "family_split": family_split,
        "coverage": coverage,
    }


def evaluate_gates(analysis: dict, rebuilt_sha: str | None,
                   verify_rc: int | None) -> dict:
    total = analysis["records_total"]
    families = analysis["per_family"]
    phases = analysis["per_phase"]
    n_fam = total or 1
    shares = {f: c / n_fam for f, c in families.items()}
    # low+zero comes from the PHASE distribution (per_phase), never from the
    # family counts.
    low_zero = (phases.get("low", 0) + phases.get("zero", 0)) / n_fam
    train_union = analysis["coverage"]["train"]["union_unique"]
    checks = {
        "records_total >= 10000": total >= PILOT_MIN_RECORDS,
        "families include arena + lichess-standard-rated-v1":
            PILOT_FAMILIES <= set(families),
        "each family share in [0.30, 0.70]":
            all(PILOT_FAMILY_MIN_SHARE <= shares.get(f, 0.0)
                <= PILOT_FAMILY_MAX_SHARE for f in PILOT_FAMILIES),
        "low+zero >= 10%": low_zero >= PILOT_LOW_ZERO_MIN,
        "train feature union >= 6500":
            train_union >= PILOT_TRAIN_UNION_MIN,
        "verify_dataset --allow-unlabeled pass":
            verify_rc == 0,
        "second rebuild dataset_sha256 identical":
            rebuilt_sha is not None and rebuilt_sha == analysis["dataset_sha256"],
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "facts": {
            "records_total": total,
            "family_shares": {f: round(s, 4) for f, s in shares.items()},
            "low_plus_zero_share": round(low_zero, 4),
            "train_union": train_union,
            "verify_rc": verify_rc,
            "rebuilt_sha256": rebuilt_sha,
            "manifest_sha256": analysis["dataset_sha256"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("results/s6/s6-n3a-coverage.json"))
    ap.add_argument("--gate", action="store_true",
                    help="evaluate the frozen pilot hard gates")
    ap.add_argument("--rebuilt-sha", default=None,
                    help="dataset_sha256 of an independent rebuild")
    ap.add_argument("--verify-rc", type=int, default=None,
                    help="exit code of verify_dataset --allow-unlabeled")
    args = ap.parse_args()

    analysis = analyze(args.engine, args.dataset, args.sources)
    result = {"analysis": analysis}
    if args.gate:
        gates = evaluate_gates(analysis, args.rebuilt_sha, args.verify_rc)
        result["gate"] = gates
        result["status"] = "DATA_PILOT_PASS" if gates["pass"] \
            else "DATA_PILOT_FAIL"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if not gates["pass"]:
            # Fail-closed: DATA_PILOT_FAIL must write the record AND exit
            # nonzero; no auto-tuning or retry.
            return 2
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n",
                            encoding="utf-8")
        print(json.dumps(analysis, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
