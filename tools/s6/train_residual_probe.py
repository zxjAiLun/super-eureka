#!/usr/bin/env python3
"""S6-N3D canonical residual checkpoint trainer (formal, fail-closed).

Trains the ONE authorized residual candidate exactly once and saves it as the
canonical artifact the S6-N3D confirmation will load from disk:

    width 16, seed 20260818, target
    clamp(teacher_cp_stm - base_eval_stm, +-2000) / 1000,
    inference base_eval_stm + residual * 1000

Everything reused, nothing re-tuned: the N3B clean-source loader
(`train_nnue_probe.load_dataset` + `export_all_features`, engine is the only
feature-encoding source of truth), the exact two-family gate
(`attach_source_families`), best-state early stopping on validation only
(`train_probe`), the disk-roundtrip check, and the provenance gates (clean
tracked worktree, disk trainer == HEAD blob, disk THIS SCRIPT == HEAD blob).
The N3C classical cache is reused after strict validation and is never
rebuilt here.

The report this writes over the EXISTING N3B validation/holdout splits is
explicitly diagnostic: those rows come from the same game pool the residual
target was fitted against, so they can never satisfy the S6-N3D confirmation
gates. See docs/s6/s6-n3c-closure-and-n3d-authorization.md section 5.

Usage:
  python tools/s6/train_residual_probe.py \
      --engine target/release/eureka \
      --dataset data/s6/s6-eval-v1-multisource-pilot01 \
      --legacy-dataset data/s6/s6-eval-v1-core-shard01 \
      --sources data/s6/sources data/s6/sources/lichess-standard-rated-v1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import residual_probe as residual  # noqa: E402
import run_n3c_probe_diagnostics as diag  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Frozen, already-validated N3C identities. Any drift fails closed rather
# than silently producing a checkpoint bound to different data.
EXPECTED_ENGINE_BINARY_SHA256 = (
    "05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66")
EXPECTED_DATASET_SHA256 = (
    "5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af")
EXPECTED_LABELS_SHA256 = (
    "e6f036f426db8a5fffc6c28baa6ae5333b0fe441bd9eec13f56d4dda989896d9")
EXPECTED_CLASSICAL_CACHE_SHA256 = (
    "c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727")
EXPECTED_TEACHER_BINARY_SHA256 = (
    "6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9")

# N3C control-E width-16 seed-20260818 reference; this run must reproduce it.
N3C_CONTROL_E_REFERENCE = {
    "validation_clipped_mae_cp": 159.685,
    "holdout_clipped_mae_cp": 159.071,
}
REPRODUCTION_TOLERANCE_CP = 0.001


def expect_sha(actual: str, expected: str, label: str) -> str:
    if actual != expected:
        raise SystemExit(
            f"PIPELINE_FAILURE: {label} sha256 {actual[:16]} != "
            f"expected {expected[:16]}")
    return actual


def verify_labeled_dataset(dataset_dir: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_dataset.py"),
         "--dataset", str(dataset_dir)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"PIPELINE_FAILURE: verify_dataset rc={proc.returncode} for "
            f"{dataset_dir}: {proc.stdout[-400:]}")


def exploratory_report(split: dict, model: probe.NnueProbe,
                       classical: dict[str, float]) -> dict:
    """Classical vs classical+residual on an EXISTING N3B split.

    Diagnostic only: same game pool as the residual target, therefore not a
    confirmation surface.
    """
    classical_cp = residual.classical_predictions_cp(split, classical)
    residual_cp, loss = residual.residual_predictions_cp(
        model, split, classical)
    deltas = residual.abs_error_deltas(
        residual_cp, classical_cp, split["raw_target_cp"])
    return {
        "is_confirmation_gate": False,
        "reason_not_a_gate": (
            "same game pool as the residual target; fingerprint-disjoint "
            "confirmation games are required for any promotion decision"),
        "residual_loss_on_residual_target": round(loss, 6),
        "overall": residual.comparison(
            residual_cp, classical_cp, split["raw_target_cp"]),
        "paired_bootstrap": residual.paired_bootstrap(deltas),
        "residual_prediction_stats": probe.pred_stats(residual_cp),
        "classical_prediction_stats": probe.pred_stats(classical_cp),
        "by_group": residual.grouped_comparison(
            split, residual_cp, classical_cp),
    }


def render_markdown(result: dict) -> str:
    checkpoint = result["checkpoint"]
    lines = [
        "# S6-N3D - Canonical Residual Checkpoint (width 16, seed 20260818)",
        "",
        f"STATUS: **{result['status']}**",
        "",
        f"ROLE: **{result['role']}**",
        "",
        "## Provenance", "", "```text",
        f"run git:            {result['provenance']['run_git_sha']}",
        f"trainer blob:       {result['provenance']['committed_trainer_blob_sha256']}",
        f"this script blob:   {result['provenance']['committed_script_blob_sha256']}",
        f"engine binary:      {result['bindings']['engine_binary_sha256']}",
        f"dataset:            {result['bindings']['dataset_sha256']}",
        f"labels:             {result['bindings']['labels_sha256']}",
        f"classical cache:    {result['bindings']['classical_cache_sha256']}",
        f"teacher binary:     {result['bindings']['teacher_binary_sha256']}",
        "```", "",
        "## Canonical checkpoint", "", "```text",
        f"path:      {checkpoint['path']}",
        f"sha256:    {checkpoint['sha256']}",
        f"target:    {checkpoint['metadata']['target_mode']} "
        f"({checkpoint['metadata']['target_formula']})",
        f"inference: {checkpoint['metadata']['inference_formula']}",
        f"width:     {checkpoint['metadata']['width']}",
        f"seed:      {checkpoint['metadata']['seed']}",
        f"best epoch/loss: {checkpoint['metadata']['best_epoch']} / "
        f"{checkpoint['metadata']['best_val_loss']}",
        f"roundtrip validation loss: {checkpoint['roundtrip_validation_loss']}",
        "```", "",
        "## N3C control-E reproduction", "",
        "| split | reference MAE | this run MAE | delta |",
        "|---|---:|---:|---:|",
    ]
    parity = result["n3c_control_e_reproduction"]
    for split in ("validation", "holdout"):
        entry = parity["splits"][split]
        lines.append(
            f"| {split} | {entry['reference_clipped_mae_cp']} | "
            f"{entry['run_clipped_mae_cp']} | {entry['delta_cp']} |")
    lines += [
        "", f"reproduction within {REPRODUCTION_TOLERANCE_CP} cp: "
        f"**{parity['within_tolerance']}**", "",
        "## Exploratory comparison on EXISTING N3B splits", "",
        "Diagnostic only - NOT a confirmation gate (same game pool as the "
        "residual target).", "",
        "| split | n | classical MAE | residual MAE | improvement | "
        "classical RMSE | residual RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("validation", "holdout"):
        overall = result["exploratory"][name]["overall"]
        lines.append(
            f"| {name} | {overall['n']} | "
            f"{overall['classical']['clipped_mae_cp']} | "
            f"{overall['residual']['clipped_mae_cp']} | "
            f"{overall['mae_improvement_fraction']:.4%} | "
            f"{overall['classical']['clipped_rmse_cp']} | "
            f"{overall['residual']['clipped_rmse_cp']} |")
    lines += [
        "", "### Paired per-position absolute-error delta", "",
        "`abs(residual - teacher) - abs(classical - teacher)`; negative is "
        "better.", "",
        "| split | n | mean | median | improved | worsened | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("validation", "holdout"):
        boot = result["exploratory"][name]["paired_bootstrap"]
        lines.append(
            f"| {name} | {boot['n']} | {boot['mean_delta_cp']} | "
            f"{boot['median_delta_cp']} | {boot['positions_improved']} | "
            f"{boot['positions_worsened']} | "
            f"[{boot['ci_lower_cp']}, {boot['ci_upper_cp']}] |")
    for kind, title in (("family", "source family"), ("phase", "phase bucket"),
                        ("abs_cp", "|teacher CP| bucket")):
        lines += ["", f"### By {title}", "",
                  "| split | group | n | classical MAE | residual MAE | "
                  "delta | mean paired delta |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name in ("validation", "holdout"):
            for group, data in result["exploratory"][name]["by_group"][kind] \
                    .items():
                if data.get("n", 0) == 0:
                    lines.append(f"| {name} | {group} | 0 | - | - | - | - |")
                    continue
                lines.append(
                    f"| {name} | {group} | {data['n']} | "
                    f"{data['classical']['clipped_mae_cp']} | "
                    f"{data['residual']['clipped_mae_cp']} | "
                    f"{data['mae_delta_cp']} | "
                    f"{data['paired_delta']['mean_delta_cp']} |")
    lines += [
        "",
        "Caveat carried from the N3C closure: early stopping selects a very "
        "low best epoch, so this candidate is close to a learned shrinkage of "
        "the classical score rather than a deep learned correction.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="frozen N3B dataset (train/validation reused as-is)")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--classical-cache", type=Path,
                        default=REPO / "results/s6/s6-n3c-classical-cache.json",
                        help="validated N3C cache; never rebuilt here")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "data/s6/models"
                        / "s6-n3d-residual-w16-s20260818.pt")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3d-canonical-residual-checkpoint.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    provenance = residual.script_provenance(REPO, Path(__file__))
    engine_sha = expect_sha(residual.sha256_file(args.engine),
                            EXPECTED_ENGINE_BINARY_SHA256, "engine binary")

    verify_labeled_dataset(args.dataset)
    family_map, source_manifest_shas = probe.build_family_map(
        [str(path) for path in args.sources])
    current = diag.load_prepared(args.dataset, args.engine, family_map)
    probe.attach_source_families(current["prepared"], family_map)

    dataset_sha = expect_sha(current["data"]["dataset_sha"],
                             EXPECTED_DATASET_SHA256, "N3B dataset")
    labels_sha = expect_sha(current["data"]["labels_sha"],
                            EXPECTED_LABELS_SHA256, "N3B labels")
    teacher_manifest = current["data"]["teacher_manifest"]
    teacher_sha = expect_sha(
        teacher_manifest.get("verified_binary_sha256", ""),
        EXPECTED_TEACHER_BINARY_SHA256, "teacher binary")

    if not args.classical_cache.is_file():
        raise SystemExit(
            f"PIPELINE_FAILURE: classical cache missing {args.classical_cache}")
    cache = diag.validate_classical_cache(
        args.classical_cache, current, engine_sha)
    cache_sha = expect_sha(cache["sha256"], EXPECTED_CLASSICAL_CACHE_SHA256,
                           "classical cache")
    classical = cache["values"]
    print(f"cache validated: {len(classical)} positions", flush=True)

    residual_splits = {
        name: residual.residual_targets(current["prepared"][name], classical)
        for name in ("train", "validation", "holdout")
    }

    model = residual.build_residual_model()
    print(f"training residual probe width={residual.RESIDUAL_WIDTH} "
          f"seed={residual.RESIDUAL_SEED}...", flush=True)
    training = probe.train_probe(model, residual_splits["train"],
                                residual_splits["validation"],
                                seed=residual.RESIDUAL_SEED)
    print(f"best_epoch={training['best_epoch']} "
          f"best_val_loss={training['best_val_loss']}", flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(residual.canonical_checkpoint_payload(
        model, width=residual.RESIDUAL_WIDTH, seed=residual.RESIDUAL_SEED,
        dataset_sha256=dataset_sha, labels_sha256=labels_sha,
        classical_cache_sha256=cache_sha, engine_binary_sha256=engine_sha,
        best_epoch=training["best_epoch"],
        best_val_loss=training["best_val_loss"],
        trainer_git_sha=provenance["trainer_git_sha"],
        trainer_blob_sha256=provenance["committed_trainer_blob_sha256"],
        dataset_id=current["data"]["manifest"]["dataset_id"]),
        args.checkpoint)
    checkpoint_sha = residual.sha256_file(args.checkpoint)
    print(f"checkpoint saved {args.checkpoint} sha256={checkpoint_sha}",
          flush=True)

    # Every reported metric comes from the DISK-RELOADED checkpoint.
    loaded, metadata = residual.load_canonical_checkpoint(args.checkpoint)
    _, roundtrip_loss = probe.evaluate_split(
        loaded, residual_splits["validation"])
    if abs(roundtrip_loss - training["best_val_loss"]) > 1e-6:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint roundtrip validation loss "
            f"{roundtrip_loss} != best {training['best_val_loss']}")

    exploratory = {
        name: exploratory_report(residual_splits[name], loaded, classical)
        for name in ("validation", "holdout")
    }

    parity_splits = {}
    within_tolerance = True
    for name in ("validation", "holdout"):
        run_mae = exploratory[name]["overall"]["residual"]["clipped_mae_cp"]
        reference = N3C_CONTROL_E_REFERENCE[f"{name}_clipped_mae_cp"]
        delta = round(run_mae - reference, 6)
        ok = abs(delta) <= REPRODUCTION_TOLERANCE_CP
        within_tolerance = within_tolerance and ok
        parity_splits[name] = {
            "reference_clipped_mae_cp": reference,
            "run_clipped_mae_cp": run_mae,
            "delta_cp": delta,
            "within_tolerance": ok,
        }

    result = {
        "status": "CANONICAL_CHECKPOINT_COMPLETE" if within_tolerance
        else "REPRODUCTION_MISMATCH",
        "role": "canonical residual checkpoint; existing-split report is "
                "diagnostic only and is NOT a confirmation gate",
        "authorization": "docs/s6/s6-n3c-closure-and-n3d-authorization.md",
        "provenance": {
            **provenance,
            "source_id_to_family": family_map,
            "source_manifests": source_manifest_shas,
        },
        "bindings": {
            "engine_binary_sha256": engine_sha,
            "dataset_id": current["data"]["manifest"]["dataset_id"],
            "dataset_sha256": dataset_sha,
            "labels_sha256": labels_sha,
            "classical_cache_sha256": cache_sha,
            "classical_cache_path": str(args.classical_cache),
            "classical_cache_header": cache["header"],
            "teacher_binary_sha256": teacher_sha,
            "teacher_nodes": teacher_manifest.get("nodes"),
            "teacher_options": teacher_manifest.get("options"),
        },
        "config": {
            "target_mode": residual.TARGET_MODE,
            "target_formula": residual.RESIDUAL_TARGET_FORMULA,
            "inference_formula": residual.RESIDUAL_INFERENCE_FORMULA,
            "inputs": probe.NNUE_INPUTS,
            "width": residual.RESIDUAL_WIDTH,
            "seed": residual.RESIDUAL_SEED,
            "optimizer": "AdamW", "lr": probe.LR,
            "weight_decay": probe.WEIGHT_DECAY,
            "loss": "SmoothL1", "loss_beta": probe.LOSS_BETA,
            "batch_size": probe.BATCH_SIZE, "max_epochs": probe.MAX_EPOCHS,
            "patience": probe.PATIENCE,
            "clip_cp": probe.CLIP_CP, "target_scale": probe.TARGET_SCALE,
            "cpu_only": True, "cuda_used": False,
            "torch_num_threads": torch.get_num_threads(),
            "bootstrap_seed": residual.BOOTSTRAP_SEED,
            "bootstrap_resamples": residual.BOOTSTRAP_RESAMPLES,
        },
        "dataset": {
            "splits": {
                name: {
                    "positions": len(current["splits"][name]),
                    "cp_rows": int(len(current["prepared"][name]["target"])),
                }
                for name in ("train", "validation", "holdout")
            },
            "training_splits_used": ["train", "validation"],
            "holdout_used_for_selection": False,
        },
        "training": {key: value for key, value in training.items()
                     if key != "train_losses"},
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha,
            "metadata": metadata,
            "roundtrip_validation_loss": round(roundtrip_loss, 6),
        },
        "n3c_control_e_reproduction": {
            "tolerance_cp": REPRODUCTION_TOLERANCE_CP,
            "within_tolerance": within_tolerance,
            "splits": parity_splits,
        },
        "exploratory": exploratory,
        "outcome": {
            "all_configurations_reported": True,
            "widths_trained": [residual.RESIDUAL_WIDTH],
            "seeds_trained": [residual.RESIDUAL_SEED],
            "width_or_seed_reselected": False,
            "n3b_dataset_relabeled": False,
            "n3b_checkpoint_modified": False,
            "confirmation_gate_evaluated_here": False,
        },
    }
    out_sha = residual.write_json(args.out, result)
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(f"results written to {args.out} sha256={out_sha}", flush=True)
    print(f"markdown written to {md_path}", flush=True)
    print(f"status: {result['status']}", flush=True)
    return 0 if within_tolerance else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
