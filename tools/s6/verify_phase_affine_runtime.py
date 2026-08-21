#!/usr/bin/env python3
"""S6-C1 phase-affine runtime parity verifier (fail-closed).

Proves that the Rust `CurrentFinalPhaseAffine` evaluator implements EXACTLY the
frozen S6-N3E `phase_affine` calibrator, over every position the audit used.

Checks, all fail-closed:
  1. bind the N3E result JSON by SHA and confirm the SELECTED calibrator really
     is `phase_affine`;
  2. convert its eight float parameters and require they equal the fixed-point
     constants the Rust runtime compiles in;
  3. the rebuilt engine's `base_cp` must equal the committed classical cache
     value for every position - i.e. adding the candidate did not perturb the
     classical evaluation it is layered on;
  4. Rust integer output == the Python fixed-point formula EXACTLY;
  5. Rust integer output vs the continuous float formula within the quantization
     budget (per-position <= 0.51 cp, mean <= 0.26 cp);
  6. recomputed confirmation clipped MAE/RMSE within 0.05 cp of the N3E record;
  7. microbench candidate/base median ratio <= 1.10.

NaN/Inf, missing rows, duplicate position ids and count mismatches all fail.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import residual_probe as residual  # noqa: E402
import run_n3c_probe_diagnostics as diag  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# Frozen fixed-point constants the Rust runtime compiles in.
PHASE_ORDER = ("high", "mid", "low", "zero")
SCALE = 1_000_000
FACTOR = (738_618, 893_806, 988_214, 2_510_204)
BIAS_SCALED_CP = (36_717_418, 50_374_720, 21_702_803, -464_891)

EXPECTED_N3E_RESULT_SHA256 = (
    "4a1fb32d76cbd69d0cee5be02ed6f4cfd5177020b172ba8a08ecce84e982f627")
EXPECTED_N3C_CACHE_SHA256 = (
    "c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727")
EXPECTED_CONFIRM_CACHE_SHA256 = (
    "126f7c82a5dfb29dbb4750b6a979da16652dca2bd850d1b8551c9361b3e5b169")
EXPECTED_N3B_DATASET_SHA256 = (
    "5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af")
EXPECTED_CONFIRM_DATASET_SHA256 = (
    "3deff6a4a5cbafcdceb02b2b2c3d06ea0cd061e127cb66f24be4d2bc81d2c43d")

MAX_ABS_QUANTIZATION_CP = 0.51
MAX_MEAN_QUANTIZATION_CP = 0.26
MAX_METRIC_DRIFT_CP = 0.05
MAX_MICROBENCH_RATIO = 1.10
CANDIDATE_PROFILE = "current-final-phase-affine"
BASELINE_PROFILE = "current-final"

PASS_STATUS = "PHASE_AFFINE_RUNTIME_PARITY_PASS"
FAIL_STATUS = "PHASE_AFFINE_RUNTIME_PARITY_FAIL"


def fail(message: str) -> None:
    raise SystemExit(f"PIPELINE_FAILURE: {message}")


def expect(actual, expected, label: str):
    if actual != expected:
        fail(f"{label}: {actual!r} != expected {expected!r}")
    return actual


def bucket_of(phase: int) -> int:
    phase = max(0, min(24, int(phase)))
    if phase >= 18:
        return 0
    if phase >= 8:
        return 1
    if phase >= 1:
        return 2
    return 3


def round_symmetric(numerator: int, denominator: int) -> int:
    half = denominator // 2
    if numerator >= 0:
        return (numerator + half) // denominator
    return -((-numerator + half) // denominator)


def fixed_point_cp(base_cp: int, phase: int) -> int:
    index = bucket_of(phase)
    return round_symmetric(base_cp * FACTOR[index] + BIAS_SCALED_CP[index], SCALE)


def float_cp(base_cp: float, phase: int, slopes, biases) -> float:
    index = bucket_of(phase)
    return base_cp * (1.0 + slopes[index]) + biases[index] * 1000.0


def verify_constants(n3e: dict) -> dict:
    """The eight fitted floats must convert to the compiled constants."""
    selected = n3e["calibration"]["selection"]["selected"]
    expect(selected, "phase_affine", "N3E selected calibrator")
    params = n3e["calibration"]["selected_parameters"]
    expect(params["name"], "phase_affine", "selected parameter block name")
    expect(tuple(params["phase_order"]), PHASE_ORDER, "phase order")
    slopes = params["parameters"]["u"]
    biases = params["parameters"]["b"]
    if len(slopes) != 4 or len(biases) != 4:
        fail(f"expected 4 slopes and 4 biases, got {len(slopes)}/{len(biases)}")
    derived_factor, derived_bias = [], []
    for slope, bias in zip(slopes, biases):
        for value in (slope, bias):
            if not math.isfinite(value):
                fail("non-finite fitted parameter")
        derived_factor.append(round((1.0 + slope) * SCALE))
        derived_bias.append(round(bias * 1000.0 * SCALE))
    expect(tuple(derived_factor), FACTOR, "derived factor constants")
    expect(tuple(derived_bias), BIAS_SCALED_CP, "derived bias constants")
    return {
        "selected_calibrator": selected,
        "phase_order": list(PHASE_ORDER),
        "fitted_slopes_u": slopes,
        "fitted_biases_b": biases,
        "scale": SCALE,
        "factor": list(FACTOR),
        "bias_scaled_cp": list(BIAS_SCALED_CP),
        "derived_from_result": True,
    }


def bind_to_n3e_labels(n3e: dict, n3b: dict, confirm: dict,
                       confirm_eligible: list[dict]) -> dict:
    """Align the loaded label sets with the N3E result BEFORE any inference.

    `teacher_cp_stm != null` is what decides the usable and eligible sets, so a
    different labels file would silently change which positions are compared and
    what MAE/RMSE come out - while every dataset and cache SHA still matched.
    The eligible count is checked against the N3E record too, so a divergence in
    the null-CP pattern cannot slip through even if the file hashes were faked.
    """
    expect(n3b["data"]["labels_sha"], n3e["bindings"]["n3b_labels_sha256"],
           "N3B labels sha")
    expect(confirm["data"]["labels_sha"],
           n3e["bindings"]["confirmation_labels_sha256"],
           "confirmation labels sha")
    expect(len(confirm_eligible), n3e["splits"]["n3d_confirmation"]["n"],
           "confirmation eligible positions")
    return {
        "n3b_labels_sha256": n3b["data"]["labels_sha"],
        "confirmation_labels_sha256": confirm["data"]["labels_sha"],
        "confirmation_eligible_positions": len(confirm_eligible),
        "aligned_with_n3e_result": True,
    }


def rust_batch(engine: Path, rows: list[dict]) -> dict[str, dict]:
    """One `bench phase-affine-batch` call for every position."""
    with tempfile.TemporaryDirectory(prefix="s6-c1-batch-") as tmp:
        batch = Path(tmp) / "batch.txt"
        batch.write_text(
            "".join(f"{row['position_id']}|{row['fen']}\n" for row in rows),
            encoding="utf-8")
        proc = subprocess.run(
            [str(engine), "bench", "phase-affine-batch", "--batch", str(batch)],
            capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        fail(f"phase-affine-batch rc={proc.returncode}: {proc.stderr[:400]}")
    out: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        pid = record["position_id"]
        if pid in out:
            fail(f"duplicate position_id in Rust output {pid}")
        for key in ("phase", "bucket", "base_cp", "calibrated_cp"):
            value = record[key]
            if not isinstance(value, int):
                fail(f"non-integer {key} for {pid}")
        out[pid] = record
    if len(out) != len(rows):
        fail(f"Rust returned {len(out)} rows != {len(rows)} positions")
    missing = {row["position_id"] for row in rows} - set(out)
    if missing:
        fail(f"{len(missing)} positions missing from Rust output")
    return out


def compare_split(label: str, rows: list[dict], classical: dict[str, float],
                  labels: dict[str, dict], engine: Path,
                  slopes, biases) -> dict:
    ids = [row["position_id"] for row in rows]
    if len(set(ids)) != len(ids):
        fail(f"{label}: duplicate position_id in input rows")
    rust = rust_batch(engine, rows)
    exact_mismatches, base_mismatches = [], []
    abs_errors, predictions, targets = [], [], []
    for row in rows:
        pid = row["position_id"]
        record = rust[pid]
        expect(record["fen"], row["fen"], f"{label}/{pid} fen")
        cached_base = classical[pid]
        if float(record["base_cp"]) != float(cached_base):
            base_mismatches.append(
                {"position_id": pid, "rust": record["base_cp"],
                 "cache": cached_base})
        if record["bucket"] != bucket_of(record["phase"]):
            fail(f"{label}/{pid}: Rust bucket {record['bucket']} != "
                 f"{bucket_of(record['phase'])}")
        exact = fixed_point_cp(int(record["base_cp"]), int(record["phase"]))
        if exact != record["calibrated_cp"]:
            exact_mismatches.append(
                {"position_id": pid, "rust": record["calibrated_cp"],
                 "python_fixed_point": exact})
        continuous = float_cp(float(record["base_cp"]), int(record["phase"]),
                              slopes, biases)
        if not math.isfinite(continuous):
            fail(f"{label}/{pid}: non-finite float reference")
        error = abs(float(record["calibrated_cp"]) - continuous)
        abs_errors.append(error)
        predictions.append(float(record["calibrated_cp"]))
        teacher = labels[pid].get("teacher_cp_stm")
        if teacher is None:
            fail(f"{label}/{pid}: usable row has null teacher_cp_stm")
        targets.append(float(teacher))
    if base_mismatches:
        fail(f"{label}: rebuilt engine base_cp differs from the committed "
             f"classical cache for {len(base_mismatches)} positions "
             f"(first: {base_mismatches[0]})")
    if exact_mismatches:
        fail(f"{label}: Rust output differs from the Python fixed-point formula "
             f"for {len(exact_mismatches)} positions "
             f"(first: {exact_mismatches[0]})")
    max_abs = max(abs_errors)
    mean_abs = sum(abs_errors) / len(abs_errors)
    metrics = probe.clipped_metrics(predictions, targets)
    return {
        "positions": len(rows),
        "exact_fixed_point_match": True,
        "base_cp_matches_committed_cache": True,
        "quantization": {
            "max_abs_cp": round(max_abs, 6),
            "mean_abs_cp": round(mean_abs, 6),
            "max_abs_budget_cp": MAX_ABS_QUANTIZATION_CP,
            "mean_abs_budget_cp": MAX_MEAN_QUANTIZATION_CP,
            "max_abs_ok": max_abs <= MAX_ABS_QUANTIZATION_CP,
            "mean_abs_ok": mean_abs <= MAX_MEAN_QUANTIZATION_CP,
        },
        "clipped_mae_cp": metrics["clipped_mae_cp"],
        "clipped_rmse_cp": metrics["clipped_rmse_cp"],
    }


def microbench(engine: Path, rows: list[dict], iterations: int,
               rounds: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="s6-c1-bench-") as tmp:
        batch = Path(tmp) / "batch.txt"
        batch.write_text("".join(f"x|{row['fen']}\n" for row in rows),
                         encoding="utf-8")
        proc = subprocess.run(
            [str(engine), "bench", "phase-affine-microbench",
             "--batch", str(batch), "--iterations", str(iterations),
             "--rounds", str(rounds)],
            capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        fail(f"phase-affine-microbench rc={proc.returncode}: "
             f"{proc.stderr[:400]}")
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    ratio = float(report["ratio"])
    if not math.isfinite(ratio):
        fail("non-finite microbench ratio")
    report["max_ratio"] = MAX_MICROBENCH_RATIO
    report["feasible"] = ratio <= MAX_MICROBENCH_RATIO
    return report


def render_markdown(result: dict) -> str:
    lines = [
        "# S6-C1 - Phase-Affine Runtime Parity", "",
        f"STATUS: **{result['status']}**", "",
        "## Bindings", "", "```text",
        f"run git:        {result['provenance']['run_git_sha']}",
        f"engine binary:  {result['bindings']['engine_binary_sha256']}",
        f"N3E result:     {result['bindings']['n3e_result_sha256']}",
        f"N3B dataset:    {result['bindings']['n3b_dataset_sha256']}",
        f"N3D dataset:    {result['bindings']['confirmation_dataset_sha256']}",
        f"N3C cache:      {result['bindings']['n3c_cache_sha256']}",
        f"N3D cache:      {result['bindings']['confirmation_cache_sha256']}",
        "```", "",
        "## Frozen constants (derived from the N3E result)", "", "```text",
        f"phase order    {result['constants']['phase_order']}",
        f"scale          {result['constants']['scale']}",
        f"factor         {result['constants']['factor']}",
        f"bias_scaled_cp {result['constants']['bias_scaled_cp']}",
        "```", "",
        "## Parity", "",
        "| split | positions | exact fixed-point | base_cp vs cache | "
        "max quant | mean quant |", "|---|---:|---|---|---:|---:|",
    ]
    for key, label in (("n3b_usable", "N3B usable"),
                       ("n3d_eligible", "N3D eligible")):
        data = result["parity"][key]
        q = data["quantization"]
        lines.append(
            f"| {label} | {data['positions']} | "
            f"{'exact' if data['exact_fixed_point_match'] else 'MISMATCH'} | "
            f"{'equal' if data['base_cp_matches_committed_cache'] else 'DIFFERS'} | "
            f"{q['max_abs_cp']} | {q['mean_abs_cp']} |")
    drift = result["metric_reproduction"]
    lines += [
        "", "## Static metric reproduction (N3D confirmation set)", "",
        "| metric | N3E record | Rust runtime | drift | budget |",
        "|---|---:|---:|---:|---:|",
        f"| clipped MAE | {drift['n3e_clipped_mae_cp']} | "
        f"{drift['runtime_clipped_mae_cp']} | {drift['mae_drift_cp']} | "
        f"{MAX_METRIC_DRIFT_CP} |",
        f"| clipped RMSE | {drift['n3e_clipped_rmse_cp']} | "
        f"{drift['runtime_clipped_rmse_cp']} | {drift['rmse_drift_cp']} | "
        f"{MAX_METRIC_DRIFT_CP} |",
        "", "## Microbench (median of 5)", "", "```text",
        f"positions        {result['microbench']['positions']}",
        f"iterations       {result['microbench']['iterations']}",
        f"base ns/eval     {result['microbench']['base_ns_per_eval']}",
        f"candidate ns/eval {result['microbench']['candidate_ns_per_eval']}",
        f"ratio            {result['microbench']['ratio']} "
        f"(gate <= {MAX_MICROBENCH_RATIO})",
        "```", "", "## Gates", "", "| gate | pass |", "|---|---|",
    ]
    for name, value in result["gates"].items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines += ["", f"Arena baseline profile: `{BASELINE_PROFILE}`; candidate "
                  f"profile: `{CANDIDATE_PROFILE}`. Same binary, "
                  f"evaluator dispatch is the only difference.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path,
                        default=REPO / "target/release/eureka")
    parser.add_argument("--n3b-dataset", type=Path, required=True)
    parser.add_argument("--n3b-sources", nargs="+", required=True)
    parser.add_argument("--confirm-dataset", type=Path, required=True)
    parser.add_argument("--confirm-source", type=Path, required=True)
    parser.add_argument("--n3c-cache", type=Path,
                        default=REPO / "results/s6/s6-n3c-classical-cache.json")
    parser.add_argument("--confirm-cache", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3d-classical-cache-confirm01.json")
    parser.add_argument("--n3e-result", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3e-residual-specificity.json")
    parser.add_argument("--microbench-iterations", type=int, default=200)
    parser.add_argument("--microbench-rounds", type=int, default=5)
    parser.add_argument("--out", type=Path,
                        default=REPO / "results/s6"
                        / "s6-c1-phase-affine-runtime.json")
    args = parser.parse_args()

    provenance = residual.script_provenance(REPO, Path(__file__))
    engine_sha = residual.sha256_file(args.engine)
    n3e_sha = expect(residual.sha256_file(args.n3e_result),
                     EXPECTED_N3E_RESULT_SHA256, "N3E result sha256")
    n3e = json.loads(args.n3e_result.read_text(encoding="utf-8"))
    constants = verify_constants(n3e)
    slopes = constants["fitted_slopes_u"]
    biases = constants["fitted_biases_b"]

    n3b_family_map, _ = probe.build_family_map(
        [str(path) for path in args.n3b_sources])
    n3b = diag.load_prepared(args.n3b_dataset, args.engine, n3b_family_map)
    expect(n3b["data"]["dataset_sha"], EXPECTED_N3B_DATASET_SHA256,
           "N3B dataset sha")
    n3c_cache = load_cache_ignoring_engine(args.n3c_cache, n3b,
                                           EXPECTED_N3C_CACHE_SHA256)
    n3b_classical = n3c_cache["values"]
    n3b_labels = n3b["data"]["labels"]
    n3b_usable = [row for row in n3b["records"]
                  if n3b_labels[row["position_id"]].get("teacher_cp_stm")
                  is not None]

    confirm_family_map, _ = probe.build_family_map([str(args.confirm_source)])
    confirm = diag.load_prepared(args.confirm_dataset, args.engine,
                                 confirm_family_map)
    expect(confirm["data"]["dataset_sha"], EXPECTED_CONFIRM_DATASET_SHA256,
           "confirmation dataset sha")
    confirm_cache = load_cache_ignoring_engine(
        args.confirm_cache, confirm, EXPECTED_CONFIRM_CACHE_SHA256)
    confirm_classical = confirm_cache["values"]
    confirm_labels = confirm["data"]["labels"]
    n3b_pids = {row["position_id"] for row in n3b["records"]}
    confirm_eligible = [
        row for row in confirm["records"]
        if confirm_labels[row["position_id"]].get("teacher_cp_stm") is not None
        and row["position_id"] not in n3b_pids]

    # Fail closed on any label/count divergence BEFORE the engine is invoked.
    label_binding = bind_to_n3e_labels(n3e, n3b, confirm, confirm_eligible)

    print(f"comparing {len(n3b_usable)} N3B usable and "
          f"{len(confirm_eligible)} N3D eligible positions", flush=True)
    parity = {
        "n3b_usable": compare_split(
            "n3b_usable", n3b_usable, n3b_classical, n3b_labels, args.engine,
            slopes, biases),
        "n3d_eligible": compare_split(
            "n3d_eligible", confirm_eligible, confirm_classical,
            confirm_labels, args.engine, slopes, biases),
    }

    n3e_confirm = n3e["splits"]["n3d_confirmation"]["calibrator_candidates"][
        "phase_affine"]["metrics"]
    runtime = parity["n3d_eligible"]
    mae_drift = abs(runtime["clipped_mae_cp"] - n3e_confirm["clipped_mae_cp"])
    rmse_drift = abs(runtime["clipped_rmse_cp"]
                     - n3e_confirm["clipped_rmse_cp"])
    metric_reproduction = {
        "n3e_clipped_mae_cp": n3e_confirm["clipped_mae_cp"],
        "runtime_clipped_mae_cp": runtime["clipped_mae_cp"],
        "mae_drift_cp": round(mae_drift, 6),
        "n3e_clipped_rmse_cp": n3e_confirm["clipped_rmse_cp"],
        "runtime_clipped_rmse_cp": runtime["clipped_rmse_cp"],
        "rmse_drift_cp": round(rmse_drift, 6),
        "budget_cp": MAX_METRIC_DRIFT_CP,
        "mae_ok": mae_drift <= MAX_METRIC_DRIFT_CP,
        "rmse_ok": rmse_drift <= MAX_METRIC_DRIFT_CP,
    }

    bench = microbench(args.engine, confirm_eligible[:512],
                       args.microbench_iterations, args.microbench_rounds)

    gates = {
        "constants_derived_from_n3e_result": True,
        "labels_aligned_with_n3e_result": label_binding["aligned_with_n3e_result"],
        "n3b_exact_fixed_point_parity":
            parity["n3b_usable"]["exact_fixed_point_match"],
        "n3d_exact_fixed_point_parity":
            parity["n3d_eligible"]["exact_fixed_point_match"],
        "base_cp_unchanged_vs_committed_caches": True,
        "n3b_quantization_within_budget":
            parity["n3b_usable"]["quantization"]["max_abs_ok"]
            and parity["n3b_usable"]["quantization"]["mean_abs_ok"],
        "n3d_quantization_within_budget":
            parity["n3d_eligible"]["quantization"]["max_abs_ok"]
            and parity["n3d_eligible"]["quantization"]["mean_abs_ok"],
        "static_metrics_reproduced": metric_reproduction["mae_ok"]
        and metric_reproduction["rmse_ok"],
        "microbench_feasible": bench["feasible"],
    }
    passed = all(gates.values())

    result = {
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "contract": "docs/s6/s6-n3e-closure-and-c1-authorization.md",
        "provenance": provenance,
        "bindings": {
            "engine_binary_sha256": engine_sha,
            "n3e_result_sha256": n3e_sha,
            "n3b_dataset_sha256": n3b["data"]["dataset_sha"],
            "n3b_labels_sha256": n3b["data"]["labels_sha"],
            "confirmation_dataset_sha256": confirm["data"]["dataset_sha"],
            "confirmation_labels_sha256": confirm["data"]["labels_sha"],
            "n3c_cache_sha256": n3c_cache["sha256"],
            "label_binding": label_binding,
            "confirmation_cache_sha256": confirm_cache["sha256"],
            "baseline_profile": BASELINE_PROFILE,
            "candidate_profile": CANDIDATE_PROFILE,
        },
        "constants": constants,
        "commands": {
            "batch": f"{args.engine} bench phase-affine-batch --batch <file>",
            "microbench": (f"{args.engine} bench phase-affine-microbench "
                           f"--batch <file> --iterations "
                           f"{args.microbench_iterations} --rounds "
                           f"{args.microbench_rounds}"),
        },
        "parity": parity,
        "metric_reproduction": metric_reproduction,
        "microbench": bench,
        "gates": gates,
    }
    out_sha = residual.write_json(args.out, result)
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(f"results written to {args.out} sha256={out_sha}", flush=True)
    print(f"status: {result['status']}", flush=True)
    return 0 if passed else 2


def load_cache_ignoring_engine(path: Path, dataset: dict,
                               expected_sha: str) -> dict:
    """Validate a committed cache by its OWN SHA and its dataset binding.

    The cache header binds the engine binary that PRODUCED it. S6-C1 rebuilds
    the binary (it adds the candidate evaluator), so the header's engine SHA no
    longer equals the running binary and `validate_classical_cache` would reject
    it. Base-eval equivalence is instead proven directly and much more strongly:
    every `base_cp` the rebuilt engine reports is compared against these cached
    values position by position.
    """
    actual = residual.sha256_file(path)
    expect(actual, expected_sha, f"{path.name} sha256")
    payload = json.loads(path.read_text(encoding="utf-8"))
    header = payload["header"]
    expect(header["dataset_sha256"], dataset["data"]["dataset_sha"],
           f"{path.name} dataset binding")
    usable = {row["position_id"] for row in dataset["records"]
              if dataset["data"]["labels"][row["position_id"]]
              .get("teacher_cp_stm") is not None}
    values = payload["values"]
    if set(values) != usable:
        fail(f"{path.name} position-id set mismatch")
    for pid, value in values.items():
        if not math.isfinite(float(value)):
            fail(f"{path.name}: non-finite cached value for {pid}")
    return {"path": str(path), "sha256": actual, "header": header,
            "values": {pid: float(v) for pid, v in values.items()}}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
