#!/usr/bin/env python3
"""S10-B3 Stage 2: evaluation-only blind holdout for a frozen checkpoint.

Loads the frozen dataset (fail-closed dataset/labels/teacher verification),
loads the selected .pt checkpoint, verifies the checkpoint's embedded summary
(feature_set, dataset SHA, labels SHA), exports ONLY holdout V2 features from
the Rust engine, and computes holdout SmoothL1 / MAE.

No optimizer is created. No training happens. Weights are never updated.
"""

from __future__ import annotations

import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from tools.s10.train_nnue import (
    NNUE_INPUTS_V2,
    TARGET_SCALE,
    LOSS_BETA,
    EncodedSplit,
    NnueModel,
    export_features_from_engine,
    load_dataset,
)

torch.use_deterministic_algorithms(True, warn_only=False)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_selection_binding(
    selection: dict,
    ckpt_sha: str,
    ckpt_summary: dict,
    dataset_sha: str,
    labels_sha: str,
    exporter_sha: str,
) -> None:
    """Fail closed unless the checkpoint and runtime identities match the
    frozen Stage-1 selection artifact exactly (checkpoint SHA, seed, dataset
    SHA, labels SHA, feature set, engine exporter SHA)."""
    ss = selection["selection_summary"]
    seed = str(ckpt_summary["seed"])

    if seed not in selection["seeds"]:
        raise SystemExit(
            f"FAIL CLOSED: checkpoint seed {seed} not in frozen selection "
            f"seeds {sorted(selection['seeds'])}"
        )
    sel_seed = selection["seeds"][seed]

    checks = [
        ("checkpoint_sha256", ckpt_sha, sel_seed["checkpoint_sha256"]),
        ("checkpoint_sha256 (summary)", ckpt_sha,
         ss["selected_checkpoint_sha256"]),
        ("dataset_sha256", dataset_sha, selection["dataset_sha256"]),
        ("labels_sha256", labels_sha, selection["labels_sha256"]),
        ("feature_set", ckpt_summary["feature_set"],
         selection["feature_set"]),
        ("engine_exporter_sha256", exporter_sha,
         selection["environment"]["engine_exporter_sha256"]),
    ]
    if ss["selected_seed"] != ckpt_summary["seed"]:
        raise SystemExit(
            f"FAIL CLOSED: checkpoint seed {seed} != frozen selected seed "
            f"{ss['selected_seed']}"
        )
    for field, actual, expected in checks:
        if actual != expected:
            raise SystemExit(
                f"FAIL CLOSED: {field} {actual!r} != frozen selection "
                f"{expected!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="S10-B3 evaluation-only holdout for a frozen checkpoint")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True,
                        help="frozen Stage-1 selection artifact "
                             "(results/s10/s10-b3-selection.json); the "
                             "checkpoint/runtime identities must match it "
                             "fail-closed")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # 1. Load + verify frozen dataset (dataset SHA, labels SHA, PID
    #    completeness, frozen teacher contract — all inside load_dataset).
    ds = load_dataset(args.dataset)
    records = ds["records"]
    labels = ds["labels"]

    # 2. Load checkpoint and verify its embedded provenance.
    ckpt_path = Path(args.checkpoint).resolve()
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    summary = ckpt["summary"]

    if summary["feature_set"] != "v2":
        print(f"FATAL: checkpoint feature_set {summary['feature_set']!r} != 'v2'")
        return 4
    if summary["dataset_sha256"] != ds["dataset_sha"]:
        print("FATAL: checkpoint dataset_sha256 mismatch")
        return 4
    if summary["labels_sha256"] != ds["labels_sha"]:
        print("FATAL: checkpoint labels_sha256 mismatch")
        return 4
    if summary.get("holdout_observed"):
        print("FATAL: checkpoint was trained with holdout observed")
        return 4

    # 2b. Fail-closed binding to the frozen Stage-1 selection artifact.
    selection = json.loads(
        Path(args.selection).read_text(encoding="utf-8"))
    engine_bin = Path(args.engine).resolve()
    exporter_sha = hashlib.sha256(engine_bin.read_bytes()).hexdigest()
    verify_selection_binding(
        selection,
        ckpt_sha=ckpt_sha,
        ckpt_summary=summary,
        dataset_sha=ds["dataset_sha"],
        labels_sha=ds["labels_sha"],
        exporter_sha=exporter_sha,
    )

    # 3. Collect usable holdout records (teacher_cp_stm not None).
    holdout = [
        (r, labels[r["position_id"]])
        for r in records
        if r["split"] == "holdout"
        and labels[r["position_id"]].get("teacher_cp_stm") is not None
    ]
    if not holdout:
        print("FATAL: 0 usable holdout records")
        return 4

    # 4. Export ONLY holdout features from the engine.
    exported = export_features_from_engine(
        args.engine, [r for r, _ in holdout], "v2")

    # 5. Build model, load frozen weights.
    model = NnueModel(num_inputs=NNUE_INPUTS_V2).to(device)
    model.load_state_dict(
        {k: v.to(device) for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    criterion = nn.SmoothL1Loss(beta=LOSS_BETA)

    items = []
    for r, lbl in holdout:
        pid = r["position_id"]
        exp = exported[pid]
        target_cp = max(-2000.0, min(2000.0, float(lbl["teacher_cp_stm"])))
        stm_is_white = r["fen"].split()[1] == "w"
        items.append({
            "position_id": pid,
            "white": exp["white"],
            "black": exp["black"],
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
            "target_scaled": target_cp / TARGET_SCALE,
            "target_cp": target_cp,
        })

    enc = EncodedSplit(items)
    stm_ind = enc.stm_indices.to(device)
    stm_off = enc.stm_offsets.to(device)
    nstm_ind = enc.nstm_indices.to(device)
    nstm_off = enc.nstm_offsets.to(device)
    targets = enc.targets.to(device)
    raw_cps = enc.raw_cps.to(device)

    with torch.no_grad():
        preds = model(stm_ind, stm_off, nstm_ind, nstm_off)
        h_loss = criterion(preds, targets).item()
        pred_cp = preds * TARGET_SCALE
        h_mae = torch.mean(torch.abs(pred_cp - raw_cps)).item()
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("FATAL: NaN/Inf in holdout predictions")
            return 4

    engine_bin = Path(args.engine).resolve()
    result = {
        "stage": "s10_b3_stage_2_blind_holdout",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_seed": summary["seed"],
        "checkpoint_best_epoch": summary["training"]["best_epoch"],
        "checkpoint_best_val_mae_cp": summary["training"]["best_val_mae"],
        "checkpoint_best_val_loss": summary["training"]["best_val_loss"],
        "dataset_sha256": ds["dataset_sha"],
        "labels_sha256": ds["labels_sha"],
        "feature_set": "v2",
        "selection_artifact": str(Path(args.selection).resolve()),
        "selection_binding_verified": True,
        "engine_exporter_sha256": exporter_sha,
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "holdout_positions_evaluated": len(items),
        "holdout_loss": h_loss,
        "holdout_mae_cp": h_mae,
        "optimizer_steps_performed": 0,
        "training_performed": False,
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
