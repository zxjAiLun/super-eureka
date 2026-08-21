#!/usr/bin/env python3
"""S6-N3E residual specificity audit (one shot, fail-closed).

N3D proved `classical + NNUE residual` beats RAW classical on unseen games. It
did not prove the gain is position-specific: the published N3D statistics are
equally consistent with the NNUE having learned a near-constant +41.38 cp
global correction, which a single parameter reproduces at zero runtime cost.

This runner makes the NNUE compete against that cheap explanation. It fits a
parameter-poor calibration ladder on the N3B TRAIN split, selects one candidate
on the N3B VALIDATION split, and then compares the frozen NNUE hybrid against
that calibrator on N3B validation, N3B holdout, and the N3D eligible
confirmation set.

It never trains or fine-tunes the NNUE, never re-selects width or seed, never
re-invokes the engine (the two committed classical caches are the only source
of base evaluations), and never touches any existing N3D artifact.

Verdict, frozen in this module and in
docs/s6/s6-n3e-residual-specificity-contract.md section 5:
  all gates pass -> POSITION_SPECIFIC_GAIN_SUPPORTED /
                    BENCH_ONLY_RUNTIME_AUTHORIZED
  any gate fails -> CHEAP_CALIBRATION_SUFFICIENT /
                    NNUE_RUNTIME_NOT_AUTHORIZED, return code 2, and the NNUE
                    residual runtime sub-branch closes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import residual_calibration as calib  # noqa: E402
import residual_probe as residual  # noqa: E402
import run_n3c_probe_diagnostics as diag  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# FROZEN identities. Nothing here is refitted, reselected or regenerated.
# ---------------------------------------------------------------------------
EXPECTED_ENGINE_BINARY_SHA256 = (
    "05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66")
EXPECTED_N3B_DATASET_SHA256 = (
    "5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af")
EXPECTED_N3B_LABELS_SHA256 = (
    "e6f036f426db8a5fffc6c28baa6ae5333b0fe441bd9eec13f56d4dda989896d9")
EXPECTED_N3C_CACHE_SHA256 = (
    "c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727")
EXPECTED_CONFIRM_DATASET_SHA256 = (
    "3deff6a4a5cbafcdceb02b2b2c3d06ea0cd061e127cb66f24be4d2bc81d2c43d")
EXPECTED_CONFIRM_LABELS_SHA256 = (
    "e1c25844fd785d46625f6a2a24edaa1a2e8fbd2863f57edfc3f3769723e8edfb")
EXPECTED_CONFIRM_CACHE_SHA256 = (
    "126f7c82a5dfb29dbb4750b6a979da16652dca2bd850d1b8551c9361b3e5b169")
EXPECTED_CANONICAL_CHECKPOINT_SHA256 = (
    "5033d47cb101d96057e13aae9d3819d48fa8079e90bda8eae8cd935ac1006c55")
CONFIRM_DATASET_ID = "s6-eval-v1-residual-confirm01-g1400"
CONFIRM_SOURCE_ID = "lichess-standard-rated-confirm-v1-g1400"

# The N3D-published global shift this audit exists to rule out.
CITED_GLOBAL_SHIFT_CP = 41.3827

# Engine invocation semantics, split explicitly.
#
# The original N3E result recorded a single `bindings.engine_invoked=false`,
# which was too broad: `diag.load_prepared()` DOES invoke the engine's
# `bench nnue-features-batch` to export NNUE feature indices. What is actually
# true - and what the frozen contract requires - is that the engine is never
# invoked to RECOMPUTE base evaluations; those come only from the two committed,
# SHA-validated classical caches. The recorded engine binary SHA binds both
# uses, so the measurement was never affected, but the field semantics are now
# stated separately instead of collapsed into one flag.
ENGINE_INVOCATION = {
    "nnue_feature_export": True,
    "base_eval_recomputation": False,
}

# ---------------------------------------------------------------------------
# FROZEN gates (contract section 5). Do not retune.
# ---------------------------------------------------------------------------
MIN_CONFIRMATION_MAE_IMPROVEMENT_FRACTION = 0.02
CI_UPPER_MUST_BE_BELOW_CP = 0.0
MAX_GROUP_REGRESSION_FRACTION = 0.02
MIN_GROUP_N = 200
# N3B validation/holdout only require "not worse"; a tiny float wobble must not
# masquerade as a regression, so equality is allowed within this epsilon.
NOT_WORSE_EPSILON_CP = 1e-9

PASS_STATUS = "POSITION_SPECIFIC_GAIN_SUPPORTED"
PASS_AUTHORIZATION = "BENCH_ONLY_RUNTIME_AUTHORIZED"
FAIL_STATUS = "CHEAP_CALIBRATION_SUFFICIENT"
FAIL_AUTHORIZATION = "NNUE_RUNTIME_NOT_AUTHORIZED"

PRODUCTION_BASELINE = "bde9085"


def fail(message: str) -> None:
    raise SystemExit(f"PIPELINE_FAILURE: {message}")


def expect(actual, expected, label: str):
    if actual != expected:
        fail(f"{label}: {actual!r} != expected {expected!r}")
    return actual


def prediction_stats(values: list[float]) -> dict:
    """min/max/mean/std plus the percentiles the contract requires."""
    if not values:
        fail("prediction_stats on empty list")
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        fail("non-finite value in prediction_stats")
    p10, p50, p90 = (float(v) for v in np.percentile(array, [10, 50, 90]))
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "p10": round(p10, 4),
        "p50": round(p50, 4),
        "p90": round(p90, 4),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    """Pearson correlation; None when either side is constant."""
    if len(left) != len(right):
        fail("pearson length mismatch")
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(value):
        fail("non-finite pearson correlation")
    return round(value, 6)


def load_validated_cache(path: Path, dataset: dict, engine_sha: str,
                         expected_cache_sha: str, label: str) -> dict:
    """Reuse a COMMITTED classical cache; never recompute base evaluations."""
    if not path.is_file():
        fail(f"{label} classical cache missing {path}")
    cache = diag.validate_classical_cache(path, dataset, engine_sha)
    expect(cache["sha256"], expected_cache_sha, f"{label} cache sha256")
    return cache


def metrics_for(predictions: list[float], targets: list[float]) -> dict:
    metrics = probe.clipped_metrics(predictions, targets)
    for key in ("clipped_mae_cp", "clipped_rmse_cp", "raw_mae_cp"):
        if not math.isfinite(metrics[key]):
            fail(f"non-finite {key}")
    return metrics


def evaluate_split_set(name: str, split: dict, classical_cp: list[float],
                       hybrid_cp: list[float], fitted: dict[str, dict],
                       selected_name: str,
                       classical_map: dict[str, float]) -> dict:
    """Every predictor on one split: raw classical, each candidate, hybrid."""
    targets = split["raw_target_cp"]
    candidates = {}
    for candidate_name in calib.CANDIDATE_ORDER:
        predicted = calib.predict_calibrator(fitted[candidate_name], split,
                                             classical_map)
        candidates[candidate_name] = {
            "parameter_count": fitted[candidate_name]["parameter_count"],
            "metrics": metrics_for(predicted, targets),
        }
    selected_predicted = calib.predict_calibrator(fitted[selected_name], split,
                                                  classical_map)
    correction = [h - c for h, c in zip(hybrid_cp, classical_cp)]
    classical_metrics = metrics_for(classical_cp, targets)
    hybrid_metrics = metrics_for(hybrid_cp, targets)
    selected_metrics = metrics_for(selected_predicted, targets)

    classical_mae = classical_metrics["clipped_mae_cp"]
    hybrid_gain = classical_mae - hybrid_metrics["clipped_mae_cp"]
    calibrator_gain = classical_mae - selected_metrics["clipped_mae_cp"]
    captured = (round(calibrator_gain / hybrid_gain, 6)
                if abs(hybrid_gain) > 1e-12 else None)
    return {
        "n": len(targets),
        "raw_classical": classical_metrics,
        "calibrator_candidates": candidates,
        "selected_calibrator": {
            "name": selected_name,
            "parameter_count": fitted[selected_name]["parameter_count"],
            "metrics": selected_metrics,
        },
        "nnue_hybrid": hybrid_metrics,
        "hybrid_vs_selected_calibrator": {
            "mae_delta_cp": round(hybrid_metrics["clipped_mae_cp"]
                                  - selected_metrics["clipped_mae_cp"], 6),
            "mae_improvement_fraction": round(
                (selected_metrics["clipped_mae_cp"]
                 - hybrid_metrics["clipped_mae_cp"])
                / selected_metrics["clipped_mae_cp"], 6)
            if selected_metrics["clipped_mae_cp"] else None,
            "rmse_delta_cp": round(hybrid_metrics["clipped_rmse_cp"]
                                   - selected_metrics["clipped_rmse_cp"], 6),
        },
        "gain_decomposition": {
            "hybrid_gain_over_classical_cp": round(hybrid_gain, 6),
            "selected_calibrator_gain_over_classical_cp": round(
                calibrator_gain, 6),
            "fraction_of_nnue_gain_captured_by_calibrator": captured,
        },
        "classical_prediction_stats": prediction_stats(classical_cp),
        "correction_prediction_stats": prediction_stats(correction),
        "hybrid_prediction_stats": prediction_stats(hybrid_cp),
        "correction_vs_classical_pearson": pearson(correction, classical_cp),
        "cited_global_shift_cp": CITED_GLOBAL_SHIFT_CP,
    }


def grouped_hybrid_vs_calibrator(split: dict, hybrid_cp: list[float],
                                 calibrator_cp: list[float]) -> dict:
    """Per-group hybrid-vs-calibrator comparison (reuses the N3D grouping)."""
    return residual.grouped_comparison(split, hybrid_cp, calibrator_cp,
                                       kinds=("phase", "abs_cp"))


def group_gate(by_group: dict, kind: str) -> dict:
    checks = {}
    for name, data in by_group.get(kind, {}).items():
        n = data.get("n", 0)
        if n < MIN_GROUP_N:
            checks[name] = {"n": n, "evaluated": False,
                            "reason": f"n < {MIN_GROUP_N}"}
            continue
        # In grouped_comparison the "residual" slot holds the hybrid and the
        # "classical" slot holds the calibrator baseline passed in above.
        calibrator_mae = data["classical"]["clipped_mae_cp"]
        hybrid_mae = data["residual"]["clipped_mae_cp"]
        regression = ((hybrid_mae - calibrator_mae) / calibrator_mae
                      if calibrator_mae else None)
        checks[name] = {
            "n": n,
            "evaluated": True,
            "calibrator_clipped_mae_cp": calibrator_mae,
            "hybrid_clipped_mae_cp": hybrid_mae,
            "regression_fraction": round(regression, 6)
            if regression is not None else None,
            "passed": regression is not None
            and regression <= MAX_GROUP_REGRESSION_FRACTION,
        }
    evaluated = [item for item in checks.values() if item["evaluated"]]
    return {
        "min_group_n": MIN_GROUP_N,
        "max_regression_fraction": MAX_GROUP_REGRESSION_FRACTION,
        "groups": checks,
        "evaluated_groups": len(evaluated),
        "passed": all(item["passed"] for item in evaluated),
    }


def all_finite(obj, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            bad.extend(all_finite(value, f"{prefix}/{key}"))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            bad.extend(all_finite(value, f"{prefix}[{index}]"))
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(prefix)
    return bad


def evaluate_gates(validation: dict, holdout: dict, confirmation: dict,
                   bootstrap: dict, by_group: dict, integrity: dict,
                   finite_violations: list[str]) -> dict:
    def not_worse(report: dict, label: str) -> dict:
        delta = report["hybrid_vs_selected_calibrator"]["mae_delta_cp"]
        return {
            "split": label,
            "calibrator_clipped_mae_cp":
                report["selected_calibrator"]["metrics"]["clipped_mae_cp"],
            "hybrid_clipped_mae_cp": report["nnue_hybrid"]["clipped_mae_cp"],
            "mae_delta_cp": delta,
            "passed": delta <= NOT_WORSE_EPSILON_CP,
        }

    validation_check = not_worse(validation, "n3b_validation")
    holdout_check = not_worse(holdout, "n3b_holdout")
    gate1 = {
        "id": 1,
        "name": "hybrid_not_worse_than_calibrator_on_n3b_validation_and_holdout",
        "threshold": f"mae_delta <= {NOT_WORSE_EPSILON_CP} cp on both splits",
        "checks": [validation_check, holdout_check],
        "passed": validation_check["passed"] and holdout_check["passed"],
    }
    improvement = confirmation["hybrid_vs_selected_calibrator"][
        "mae_improvement_fraction"]
    gate2 = {
        "id": 2,
        "name": "confirmation_mae_improvement_over_calibrator",
        "threshold": f">= {MIN_CONFIRMATION_MAE_IMPROVEMENT_FRACTION}",
        "calibrator_clipped_mae_cp":
            confirmation["selected_calibrator"]["metrics"]["clipped_mae_cp"],
        "hybrid_clipped_mae_cp": confirmation["nnue_hybrid"]["clipped_mae_cp"],
        "observed_improvement_fraction": improvement,
        "passed": improvement is not None
        and improvement >= MIN_CONFIRMATION_MAE_IMPROVEMENT_FRACTION,
    }
    gate3 = {
        "id": 3,
        "name": "paired_abs_error_delta_bootstrap_ci_upper_below_zero",
        "definition": "abs(hybrid - teacher) - abs(calibrator - teacher)",
        "threshold": f"< {CI_UPPER_MUST_BE_BELOW_CP} cp",
        "bootstrap_seed": bootstrap["bootstrap_seed"],
        "bootstrap_resamples": bootstrap["bootstrap_resamples"],
        "mean_delta_cp": bootstrap["mean_delta_cp"],
        "median_delta_cp": bootstrap["median_delta_cp"],
        "ci_lower_cp": bootstrap["ci_lower_cp"],
        "ci_upper_cp": bootstrap["ci_upper_cp"],
        "passed": bootstrap["ci_upper_cp"] < CI_UPPER_MUST_BE_BELOW_CP,
    }
    calibrator_rmse = confirmation["selected_calibrator"]["metrics"][
        "clipped_rmse_cp"]
    hybrid_rmse = confirmation["nnue_hybrid"]["clipped_rmse_cp"]
    gate4 = {
        "id": 4,
        "name": "hybrid_rmse_not_above_calibrator",
        "threshold": "hybrid <= calibrator",
        "calibrator_clipped_rmse_cp": calibrator_rmse,
        "hybrid_clipped_rmse_cp": hybrid_rmse,
        "passed": hybrid_rmse <= calibrator_rmse + NOT_WORSE_EPSILON_CP,
    }
    phase = group_gate(by_group, "phase")
    gate5 = {"id": 5, "name": "phase_bucket_no_regression_vs_calibrator",
             **phase}
    abs_cp = group_gate(by_group, "abs_cp")
    gate6 = {"id": 6,
             "name": "abs_teacher_cp_bucket_no_regression_vs_calibrator",
             **abs_cp}
    gate7 = {
        "id": 7,
        "name": "provenance_cache_checkpoint_finite_integrity",
        "all_metrics_finite": not finite_violations,
        "non_finite_paths": finite_violations[:10],
        **integrity,
        "passed": bool(not finite_violations
                       and all(integrity[key] for key in integrity
                               if key.endswith("_ok"))),
    }
    gates = [gate1, gate2, gate3, gate4, gate5, gate6, gate7]
    return {
        "all_passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "frozen_thresholds": {
            "min_confirmation_mae_improvement_fraction":
                MIN_CONFIRMATION_MAE_IMPROVEMENT_FRACTION,
            "ci_upper_must_be_below_cp": CI_UPPER_MUST_BE_BELOW_CP,
            "max_group_regression_fraction": MAX_GROUP_REGRESSION_FRACTION,
            "min_group_n": MIN_GROUP_N,
            "not_worse_epsilon_cp": NOT_WORSE_EPSILON_CP,
            "bootstrap_seed": residual.BOOTSTRAP_SEED,
            "bootstrap_resamples": residual.BOOTSTRAP_RESAMPLES,
        },
    }


def render_markdown(result: dict) -> str:
    confirmation = result["splits"]["n3d_confirmation"]
    selection = result["calibration"]["selection"]
    boot = result["paired_bootstrap"]
    lines = [
        "# S6-N3E - Residual Specificity Audit", "",
        f"STATUS: **{result['status']}**", "",
        f"AUTHORIZATION: **{result['authorization']}**", "",
        "This is a post-confirmation mechanism audit opened AFTER N3D was "
        "accepted. It does not rewrite or supersede any N3D artifact.", "",
        "## Question", "",
        f"N3D's gain is consistent with a near-constant "
        f"**+{CITED_GLOBAL_SHIFT_CP} cp** global correction. Can a "
        f"parameter-poor calibrator reproduce it?", "",
        "## Provenance and bindings", "", "```text",
        f"run git:              {result['provenance']['run_git_sha']}",
        f"runner blob:          {result['provenance']['committed_script_blob_sha256']}",
        f"canonical checkpoint: {result['bindings']['canonical_checkpoint_sha256']}",
        f"engine binary:        {result['bindings']['engine_binary_sha256']}",
        f"N3B dataset:          {result['bindings']['n3b_dataset_sha256']}",
        f"N3C cache:            {result['bindings']['n3c_classical_cache_sha256']}",
        f"N3D dataset:          {result['bindings']['confirmation_dataset_sha256']}",
        f"N3D cache:            {result['bindings']['confirmation_cache_sha256']}",
        "```", "",
        "## Calibration ladder", "",
        f"Fitted on **{selection['fitted_on']}**, selected on "
        f"**{selection['selected_on']}**. Selected: "
        f"**{selection['selected']}** "
        f"({selection['selected_parameter_count']} parameters).", "",
        "| candidate | parameters | N3B train SmoothL1 | N3B validation SmoothL1 |",
        "|---|---:|---:|---:|",
    ]
    for name in calib.CANDIDATE_ORDER:
        fitted = result["calibration"]["fitted"][name]
        mark = " **<- selected**" if name == selection["selected"] else ""
        lines.append(
            f"| {name}{mark} | {fitted['parameter_count']} | "
            f"{fitted['train_smooth_l1']} | "
            f"{selection['validation_smooth_l1'][name]} |")
    lines += ["", "Selected calibrator parameters (cp where applicable):", "",
              "```json",
              json.dumps(result["calibration"]["selected_parameters"],
                         indent=2, sort_keys=True),
              "```", "",
              "## Predictor comparison (clipped MAE / RMSE, cp)", "",
              "| split | n | raw classical | selected calibrator | NNUE hybrid |",
              "|---|---:|---:|---:|---:|"]
    for key, label in (("n3b_validation", "N3B validation"),
                       ("n3b_holdout", "N3B holdout"),
                       ("n3d_confirmation", "N3D confirmation")):
        data = result["splits"][key]
        lines.append(
            f"| {label} | {data['n']} | "
            f"{data['raw_classical']['clipped_mae_cp']} / "
            f"{data['raw_classical']['clipped_rmse_cp']} | "
            f"{data['selected_calibrator']['metrics']['clipped_mae_cp']} / "
            f"{data['selected_calibrator']['metrics']['clipped_rmse_cp']} | "
            f"{data['nnue_hybrid']['clipped_mae_cp']} / "
            f"{data['nnue_hybrid']['clipped_rmse_cp']} |")
    lines += ["", "### Every calibrator candidate (clipped MAE, cp)", "",
              "| split | " + " | ".join(calib.CANDIDATE_ORDER) + " | hybrid |",
              "|---|" + "---:|" * (len(calib.CANDIDATE_ORDER) + 1)]
    for key, label in (("n3b_validation", "N3B validation"),
                       ("n3b_holdout", "N3B holdout"),
                       ("n3d_confirmation", "N3D confirmation")):
        data = result["splits"][key]
        cells = [str(data["calibrator_candidates"][name]["metrics"]
                     ["clipped_mae_cp"]) for name in calib.CANDIDATE_ORDER]
        cells.append(str(data["nnue_hybrid"]["clipped_mae_cp"]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "### Mechanism: what the correction actually looks like", "",
              "| split | correction mean | std | p10 | p50 | p90 | "
              "corr vs classical | calibrator captures |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("n3b_validation", "N3B validation"),
                       ("n3b_holdout", "N3B holdout"),
                       ("n3d_confirmation", "N3D confirmation")):
        data = result["splits"][key]
        stats = data["correction_prediction_stats"]
        captured = data["gain_decomposition"][
            "fraction_of_nnue_gain_captured_by_calibrator"]
        lines.append(
            f"| {label} | {stats['mean']} | {stats['std']} | {stats['p10']} | "
            f"{stats['p50']} | {stats['p90']} | "
            f"{data['correction_vs_classical_pearson']} | "
            f"{'-' if captured is None else format(captured, '.2%')} |")
    lines += ["", "### Paired per-position delta vs the selected calibrator",
              "",
              "`abs(hybrid - teacher) - abs(calibrator - teacher)` on the N3D "
              "confirmation set; negative favours the NNUE.", "", "```text",
              f"n                {boot['n']}",
              f"mean             {boot['mean_delta_cp']} cp",
              f"median           {boot['median_delta_cp']} cp",
              f"improved         {boot['positions_improved']}",
              f"worsened         {boot['positions_worsened']}",
              f"unchanged        {boot['positions_unchanged']}",
              f"bootstrap        numpy default_rng(seed="
              f"{boot['bootstrap_seed']}), {boot['bootstrap_resamples']} "
              f"resamples",
              f"95% CI           [{boot['ci_lower_cp']}, "
              f"{boot['ci_upper_cp']}] cp",
              "```", "", "## Gates", "", "| # | gate | pass |", "|---:|---|---|"]
    for gate in result["gate_evaluation"]["gates"]:
        lines.append(f"| {gate['id']} | {gate['name']} | "
                     f"{'PASS' if gate['passed'] else 'FAIL'} |")
    for gate_id, kind, title in ((5, "phase", "phase bucket"),
                                 (6, "abs_cp", "|teacher CP| bucket")):
        gate = next(g for g in result["gate_evaluation"]["gates"]
                    if g["id"] == gate_id)
        lines += ["", f"### By {title} (hybrid vs selected calibrator)", "",
                  "| group | n | calibrator MAE | hybrid MAE | regression | "
                  "evaluated |", "|---|---:|---:|---:|---:|---|"]
        for name, check in gate["groups"].items():
            if not check["evaluated"]:
                lines.append(f"| {name} | {check['n']} | - | - | - | no |")
                continue
            lines.append(
                f"| {name} | {check['n']} | "
                f"{check['calibrator_clipped_mae_cp']} | "
                f"{check['hybrid_clipped_mae_cp']} | "
                f"{check['regression_fraction']} | yes |")
    lines += ["", "## Outcome", "", result["outcome"]["summary"], "",
              result["outcome"]["next_step"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path,
                        default=REPO / "target/release/eureka",
                        help="hashed for binding only; never invoked")
    parser.add_argument("--n3b-dataset", type=Path, required=True)
    parser.add_argument("--n3b-sources", nargs="+", required=True)
    parser.add_argument("--confirm-dataset", type=Path, required=True)
    parser.add_argument("--confirm-source", type=Path, required=True)
    parser.add_argument("--n3c-cache", type=Path,
                        default=REPO / "results/s6/s6-n3c-classical-cache.json")
    parser.add_argument("--confirm-cache", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3d-classical-cache-confirm01.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "data/s6/models"
                        / "s6-n3d-residual-w16-s20260818.pt")
    parser.add_argument("--n3d-result", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3d-residual-confirmation.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3e-residual-specificity.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    provenance = residual.script_provenance(REPO, Path(__file__))
    engine_sha = expect(residual.sha256_file(args.engine),
                        EXPECTED_ENGINE_BINARY_SHA256, "engine binary sha256")

    # --- N3B side -----------------------------------------------------------
    n3b_family_map, _ = probe.build_family_map(
        [str(path) for path in args.n3b_sources])
    n3b = diag.load_prepared(args.n3b_dataset, args.engine, n3b_family_map)
    probe.attach_source_families(n3b["prepared"], n3b_family_map)
    expect(n3b["data"]["dataset_sha"], EXPECTED_N3B_DATASET_SHA256,
           "N3B dataset sha")
    expect(n3b["data"]["labels_sha"], EXPECTED_N3B_LABELS_SHA256,
           "N3B labels sha")
    n3c_cache = load_validated_cache(args.n3c_cache, n3b, engine_sha,
                                     EXPECTED_N3C_CACHE_SHA256, "N3C")
    n3b_classical = n3c_cache["values"]

    # --- N3D confirmation side ---------------------------------------------
    confirm_family_map, _ = probe.build_family_map([str(args.confirm_source)])
    confirm = diag.load_prepared(args.confirm_dataset, args.engine,
                                 confirm_family_map)
    expect(confirm["data"]["manifest"]["dataset_id"], CONFIRM_DATASET_ID,
           "confirmation dataset_id")
    expect(confirm["data"]["dataset_sha"], EXPECTED_CONFIRM_DATASET_SHA256,
           "confirmation dataset sha")
    expect(confirm["data"]["labels_sha"], EXPECTED_CONFIRM_LABELS_SHA256,
           "confirmation labels sha")
    confirm_cache = load_validated_cache(
        args.confirm_cache, confirm, engine_sha,
        EXPECTED_CONFIRM_CACHE_SHA256, "N3D confirmation")
    confirm_classical = confirm_cache["values"]

    # The eligible set is taken verbatim from the accepted N3D record so N3E
    # scores EXACTLY the positions N3D scored - it is never recomputed here.
    n3d_result = json.loads(args.n3d_result.read_text(encoding="utf-8"))
    expect(n3d_result["status"], "RESIDUAL_CONFIRMATION_PASS", "N3D status")
    expect(n3d_result["bindings"]["confirmation_dataset_sha256"],
           EXPECTED_CONFIRM_DATASET_SHA256, "N3D record dataset sha")
    expect(n3d_result["checkpoint"]["sha256"],
           EXPECTED_CANONICAL_CHECKPOINT_SHA256, "N3D record checkpoint sha")
    n3d_eligible = n3d_result["identity_audit"]["positions"][
        "eligible_positions"]
    n3b_pids = {row["position_id"] for row in
                probe.load_dataset(args.n3b_dataset)["records"]}
    labels = confirm["data"]["labels"]
    eligible_records = [
        row for row in confirm["records"]
        if labels[row["position_id"]].get("teacher_cp_stm") is not None
        and row["position_id"] not in n3b_pids]
    expect(len(eligible_records), n3d_eligible,
           "eligible confirmation positions vs the accepted N3D record")

    confirm_split = probe.prepare_split(confirm["exported"], eligible_records,
                                        labels)
    diag.add_family_metadata(confirm_split, confirm_family_map)

    # --- Fit and select the cheap ladder (train fits, validation selects) ---
    train_data = calib.calibration_inputs(n3b["prepared"]["train"],
                                          n3b_classical)
    validation_data = calib.calibration_inputs(n3b["prepared"]["validation"],
                                               n3b_classical)
    fitted = calib.fit_calibrators(train_data)
    selection = calib.validation_select_calibrator(fitted, validation_data)
    selected_name = selection["selected"]
    print(f"selected calibrator: {selected_name} "
          f"({fitted[selected_name]['parameter_count']} parameters)",
          flush=True)

    # --- Frozen NNUE hybrid (loaded from disk, SHA-bound, never trained) ----
    model, metadata = residual.load_canonical_checkpoint(
        args.checkpoint, expected_sha256=EXPECTED_CANONICAL_CHECKPOINT_SHA256)
    expect(metadata["dataset_sha256"], EXPECTED_N3B_DATASET_SHA256,
           "checkpoint training dataset sha")
    expect(metadata["classical_cache_sha256"], EXPECTED_N3C_CACHE_SHA256,
           "checkpoint cache sha")

    splits: dict[str, dict] = {}
    for key, split, classical_map in (
            ("n3b_validation", n3b["prepared"]["validation"], n3b_classical),
            ("n3b_holdout", n3b["prepared"]["holdout"], n3b_classical),
            ("n3d_confirmation", confirm_split, confirm_classical)):
        residual_split = residual.residual_targets(split, classical_map)
        classical_cp = residual.classical_predictions_cp(split, classical_map)
        hybrid_cp, _ = residual.residual_predictions_cp(
            model, residual_split, classical_map)
        splits[key] = evaluate_split_set(key, split, classical_cp, hybrid_cp,
                                         fitted, selected_name, classical_map)
        splits[key]["_hybrid_cp"] = hybrid_cp
        splits[key]["_split"] = split
        splits[key]["_classical_map"] = classical_map

    # --- Confirmation-set head-to-head vs the selected calibrator -----------
    confirmation_view = splits["n3d_confirmation"]
    calibrator_cp = calib.predict_calibrator(fitted[selected_name],
                                             confirm_split, confirm_classical)
    hybrid_cp = confirmation_view["_hybrid_cp"]
    deltas = residual.abs_error_deltas(hybrid_cp, calibrator_cp,
                                       confirm_split["raw_target_cp"])
    bootstrap = residual.paired_bootstrap(deltas)
    by_group = grouped_hybrid_vs_calibrator(confirm_split, hybrid_cp,
                                            calibrator_cp)

    for view in splits.values():
        for internal in ("_hybrid_cp", "_split", "_classical_map"):
            view.pop(internal, None)

    integrity = {
        "provenance_ok": bool(provenance["run_started_clean"]),
        "engine_binding_ok": engine_sha == EXPECTED_ENGINE_BINARY_SHA256,
        "n3c_cache_ok": n3c_cache["sha256"] == EXPECTED_N3C_CACHE_SHA256,
        "confirmation_cache_ok":
            confirm_cache["sha256"] == EXPECTED_CONFIRM_CACHE_SHA256,
        "checkpoint_sha_ok":
            metadata["checkpoint_sha256"] == EXPECTED_CANONICAL_CHECKPOINT_SHA256,
        "eligible_set_matches_n3d_ok": len(eligible_records) == n3d_eligible,
        "nnue_retrained": False,
        "engine_invoked_for_base_eval": False,
    }
    finite_violations = all_finite({"splits": splits,
                                    "paired_bootstrap": bootstrap,
                                    "by_group": by_group})
    gate_evaluation = evaluate_gates(
        splits["n3b_validation"], splits["n3b_holdout"],
        splits["n3d_confirmation"], bootstrap, by_group, integrity,
        finite_violations)
    passed = gate_evaluation["all_passed"]

    captured = splits["n3d_confirmation"]["gain_decomposition"][
        "fraction_of_nnue_gain_captured_by_calibrator"]
    if passed:
        summary = (
            f"The NNUE hybrid beats the best cheap calibrator "
            f"(`{selected_name}`, "
            f"{fitted[selected_name]['parameter_count']} parameters) on the "
            f"unseen confirmation set by "
            f"{splits['n3d_confirmation']['hybrid_vs_selected_calibrator']['mae_improvement_fraction']:.2%} "
            f"clipped MAE, so the N3D gain is NOT explained by a cheap global "
            f"recalibration.")
        next_step = (
            "N3F bench-only artifact parity and cost work is authorized in "
            "this round. No search, eval, UCI or Arena wiring; no production "
            "promotion verdict.")
    else:
        summary = (
            f"The cheap calibrator `{selected_name}` "
            f"({fitted[selected_name]['parameter_count']} parameters) already "
            f"captures "
            f"{'an undefined share' if captured is None else format(captured, '.2%')}"
            f" of the NNUE's gain over raw classical, and the NNUE does not "
            f"clear the specificity gates. The measured N3D improvement is "
            f"adequately explained by a cheap recalibration of the classical "
            f"evaluation.")
        next_step = (
            f"NNUE residual runtime sub-branch CLOSED. No exporter and no Rust "
            f"runtime file is created. Production baseline "
            f"{PRODUCTION_BASELINE} stays unchanged. The selected calibrator's "
            f"parameters are recorded above for whatever the cloud decides to "
            f"do with them.")

    result = {
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "authorization": PASS_AUTHORIZATION if passed else FAIL_AUTHORIZATION,
        "audit_type": "post-confirmation mechanism audit",
        "opened_after_seeing": "S6-N3D confirmation result",
        "supersedes_n3d": False,
        "n3d_artifacts_modified": False,
        "contract": "docs/s6/s6-n3e-residual-specificity-contract.md",
        "provenance": provenance,
        "bindings": {
            "engine_binary_sha256": engine_sha,
            "engine_invocation": dict(ENGINE_INVOCATION),
            "n3b_dataset_sha256": n3b["data"]["dataset_sha"],
            "n3b_labels_sha256": n3b["data"]["labels_sha"],
            "n3c_classical_cache_sha256": n3c_cache["sha256"],
            "n3c_classical_cache_header": n3c_cache["header"],
            "confirmation_dataset_id": CONFIRM_DATASET_ID,
            "confirmation_dataset_sha256": confirm["data"]["dataset_sha"],
            "confirmation_labels_sha256": confirm["data"]["labels_sha"],
            "confirmation_cache_sha256": confirm_cache["sha256"],
            "confirmation_cache_header": confirm_cache["header"],
            "canonical_checkpoint_sha256": metadata["checkpoint_sha256"],
            "canonical_checkpoint_metadata": metadata,
            "n3d_result_sha256": residual.sha256_file(args.n3d_result),
            "n3d_status": n3d_result["status"],
            "n3d_verdict": n3d_result["verdict"],
        },
        "config": {
            "cpu_only": True, "cuda_used": False,
            "torch_num_threads": torch.get_num_threads(),
            "width": residual.RESIDUAL_WIDTH,
            "seed": residual.RESIDUAL_SEED,
            "nnue_retrained": False,
            "width_or_seed_reselected": False,
            "calibration_dtype": "float64",
            "calibration_optimizer": "LBFGS",
            "calibration_max_iter": calib.LBFGS_MAX_ITER,
            "calibration_line_search": calib.LBFGS_LINE_SEARCH,
            "calibration_loss": "SmoothL1",
            "calibration_loss_beta": calib.LOSS_BETA,
            "calibration_init": "zeros",
            "calibration_rng_used": False,
        },
        "calibration": {
            "candidate_order": list(calib.CANDIDATE_ORDER),
            "parameter_counts": dict(calib.CANDIDATE_PARAMETER_COUNT),
            "fitted": fitted,
            "selection": selection,
            "selected_parameters": fitted[selected_name],
            "train_positions": train_data["n"],
            "validation_positions": validation_data["n"],
        },
        "splits": splits,
        "paired_bootstrap": bootstrap,
        "by_group_hybrid_vs_calibrator": by_group,
        "gate_evaluation": gate_evaluation,
        "outcome": {
            "single_evaluation_run": True,
            "nnue_retrained": False,
            "thresholds_changed": False,
            "n3d_artifacts_modified": False,
            "production_baseline": PRODUCTION_BASELINE,
            "n3f_bench_authorized": passed,
            "summary": summary,
            "next_step": next_step,
        },
    }
    out_sha = residual.write_json(args.out, result)
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(f"results written to {args.out} sha256={out_sha}", flush=True)
    print(f"markdown written to {md_path}", flush=True)
    print(f"status: {result['status']}", flush=True)
    print(f"authorization: {result['authorization']}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
