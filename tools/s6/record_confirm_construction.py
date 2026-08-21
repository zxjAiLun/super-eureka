#!/usr/bin/env python3
"""S6-N3D confirmation-set construction recorder (pre-metric, fail-closed).

Records what a confirmation-set CONSTRUCTION attempt produced, before and
independently of any model metric. This exists so a construction shortfall is
never silently deleted and never mislabelled as a scientific
`CONFIRMATION_FAIL`: the two outcomes mean different things, and only the
confirmation runner may emit the latter.

Reports, per attempt:
  selected_games, raw / usable / null-CP / position-overlap-excluded /
  eligible_usable counts, retained_fraction, the ordered game-fingerprint
  sequence and its SHA, and the PGN / source-manifest / dataset-manifest /
  labels / teacher-manifest SHAs.

status is CONSTRUCTION_SUFFICIENT only when BOTH pre-registered construction
preconditions hold (eligible usable >= 5000 and retained_fraction >= 0.90);
otherwise CONSTRUCTION_INSUFFICIENT with an explicit reason.

`part5_evaluation_run` and `confirmation_metrics_observed` are always false
here: this tool loads no checkpoint and computes no residual or classical
prediction, so recording an attempt can never peek at a confirmation metric.

Usage:
  python tools/s6/record_confirm_construction.py \
      --attempt-label g1000 \
      --source data/s6/sources/lichess-standard-rated-confirm-v1 \
      --dataset data/s6/s6-eval-v1-residual-confirm01 \
      --n3b-dataset data/s6/s6-eval-v1-multisource-pilot01 \
      --out results/s6/s6-n3d-construction-attempt-g1000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lichess_select as ls  # noqa: E402
import residual_probe as residual  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Pre-registered construction preconditions (frozen; never relaxed here).
MIN_ELIGIBLE_USABLE_POSITIONS = 5000
MIN_RETAINED_FRACTION = 0.90

SUFFICIENT = "CONSTRUCTION_SUFFICIENT"
INSUFFICIENT = "CONSTRUCTION_INSUFFICIENT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-label", required=True,
                        help="short artifact label, e.g. g1000 / g1400")
    parser.add_argument("--source", type=Path, required=True,
                        help="confirmation source dir (PGN + manifest)")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="built and labeled confirmation dataset dir")
    parser.add_argument("--n3b-dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    provenance = residual.script_provenance(REPO, Path(__file__))

    source_manifest_path = args.source / "source-manifest.json"
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8"))
    source_id = source_manifest["source_id"]
    pgn_path = args.source / f"{source_id}.pgn"

    fingerprints = ls.ordered_pgn_fingerprints(pgn_path)
    if len(set(fingerprints)) != len(fingerprints):
        raise SystemExit(
            "PIPELINE_FAILURE: duplicate game fingerprint in confirmation PGN")
    if len(fingerprints) != source_manifest["games_selected"]:
        raise SystemExit(
            f"PIPELINE_FAILURE: PGN has {len(fingerprints)} games != manifest "
            f"{source_manifest['games_selected']}")

    dataset = probe.load_dataset(args.dataset)
    labels = dataset["labels"]
    records = dataset["records"]
    usable = [row for row in records
              if labels[row["position_id"]].get("teacher_cp_stm") is not None]
    n3b = probe.load_dataset(args.n3b_dataset)
    n3b_pids = {row["position_id"] for row in n3b["records"]}
    excluded = sorted({row["position_id"] for row in usable} & n3b_pids)
    eligible = len(usable) - len(excluded)
    retained = eligible / len(usable) if usable else 0.0

    reasons = []
    if eligible < MIN_ELIGIBLE_USABLE_POSITIONS:
        reasons.append(
            f"eligible usable positions {eligible} < preregistered minimum "
            f"{MIN_ELIGIBLE_USABLE_POSITIONS}")
    if retained < MIN_RETAINED_FRACTION:
        reasons.append(
            f"retained_fraction {retained:.6f} < preregistered minimum "
            f"{MIN_RETAINED_FRACTION}")
    status = SUFFICIENT if not reasons else INSUFFICIENT

    teacher_manifest_path = args.dataset / "teacher_manifest.json"
    dataset_manifest_path = args.dataset / "dataset_manifest.json"
    labels_path = args.dataset / "labels.jsonl"

    record = {
        "attempt_label": args.attempt_label,
        "status": status,
        "reason": "; ".join(reasons) if reasons
        else "both preregistered construction preconditions satisfied",
        "preregistered_minimums": {
            "eligible_usable_positions": MIN_ELIGIBLE_USABLE_POSITIONS,
            "retained_fraction": MIN_RETAINED_FRACTION,
        },
        "part5_evaluation_run": False,
        "confirmation_metrics_observed": False,
        "provenance": provenance,
        "selection": {
            "source_id": source_id,
            "source_family": source_manifest["source_family"],
            "selected_games": source_manifest["games_selected"],
            "selection_seed": source_manifest["selection_seed"],
            "archive_official_sha256": source_manifest["official_sha256"],
            "selector_script_sha256": source_manifest["script_sha256"],
            "exclude_sources": source_manifest.get("exclude_sources"),
            "exclude_fingerprint_count": source_manifest.get(
                "exclude_fingerprint_count"),
            "exclude_fingerprints_sha256": source_manifest.get(
                "exclude_fingerprints_sha256"),
            "selected_fingerprints_sha256": source_manifest.get(
                "selected_fingerprints_sha256"),
            "fingerprint_intersection": source_manifest.get(
                "fingerprint_intersection"),
            "excluded_candidates_skipped": source_manifest.get(
                "excluded_candidates_skipped"),
            "duplicate_candidates_rejected": source_manifest.get(
                "duplicate_candidates_rejected"),
            "fingerprint_definition": source_manifest.get("fingerprint"),
        },
        "positions": {
            "raw_records": len(records),
            "usable_records": len(usable),
            "null_cp_records": len(records) - len(usable),
            "null_cp_rate": round(
                (len(records) - len(usable)) / len(records), 6)
            if records else 0.0,
            "position_overlap_excluded": len(excluded),
            "excluded_ids_sha256": ls.fingerprint_set_sha256(set(excluded)),
            "eligible_usable": eligible,
            "retained_fraction": round(retained, 6),
            "n3b_position_ids": len(n3b_pids),
        },
        "hashes": {
            "pgn_sha256": residual.sha256_file(pgn_path),
            "source_manifest_sha256": residual.sha256_file(
                source_manifest_path),
            "dataset_id": dataset["manifest"]["dataset_id"],
            "dataset_manifest_sha256": residual.sha256_file(
                dataset_manifest_path),
            "dataset_sha256": dataset["dataset_sha"],
            "labels_sha256": dataset["labels_sha"],
            "labels_file_sha256": residual.sha256_file(labels_path),
            "teacher_manifest_sha256": residual.sha256_file(
                teacher_manifest_path),
            "teacher_binary_sha256": dataset["teacher_manifest"].get(
                "verified_binary_sha256"),
            "n3b_dataset_sha256": n3b["dataset_sha"],
        },
        "teacher_audit": dataset["teacher_manifest"].get("audit"),
        "ordered_game_fingerprints": fingerprints,
        "ordered_game_fingerprint_count": len(fingerprints),
        "game_fingerprints_sha256": ls.fingerprint_set_sha256(
            set(fingerprints)),
    }
    out_sha = residual.write_json(args.out, record)
    print(f"attempt {args.attempt_label}: {status}", flush=True)
    print(f"  raw={len(records)} usable={len(usable)} "
          f"excluded={len(excluded)} eligible={eligible} "
          f"retained={retained:.6f}", flush=True)
    print(f"record written to {args.out} sha256={out_sha}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
