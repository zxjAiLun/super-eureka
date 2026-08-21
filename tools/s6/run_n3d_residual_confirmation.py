#!/usr/bin/env python3
"""S6-N3D independent residual confirmation (one shot, fail-closed).

Loads the canonical width-16 seed-20260818 residual checkpoint FROM DISK and
evaluates it exactly once against confirmation games that are
fingerprint-disjoint from every game the N3B models were fitted on. It never
trains, never re-selects width or seed, never retunes a threshold, and never
retries.

All six gates are frozen in this module and in
docs/s6/s6-n3c-closure-and-n3d-authorization.md section 3, both committed
before any confirmation label existed:

  1 overall clipped MAE improvement of residual over classical >= 2%
  2 paired per-position absolute-error delta bootstrap 95% CI upper < 0 cp
  3 residual clipped RMSE <= classical clipped RMSE
  4 every phase bucket with n >= 200 not worse than 2%
  5 every |teacher CP| bucket with n >= 200 not worse than 2%
  6 metrics finite, checkpoint disk roundtrip, identity and cache gates pass

Any failure writes CONFIRMATION_FAIL and returns 2.

Usage:
  python tools/s6/run_n3d_residual_confirmation.py \
      --engine target/release/eureka \
      --dataset data/s6/s6-eval-v1-residual-confirm01 \
      --n3b-dataset data/s6/s6-eval-v1-multisource-pilot01 \
      --confirm-source data/s6/sources/lichess-standard-rated-confirm-v1 \
      --n3b-sources data/s6/sources \
                    data/s6/sources/lichess-standard-rated-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import chess.pgn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as bd  # noqa: E402
import lichess_select as ls  # noqa: E402
import residual_probe as residual  # noqa: E402
import run_n3c_probe_diagnostics as diag  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# FROZEN identities and gates. Do not retune; a failure ends the sub-branch.
# ---------------------------------------------------------------------------
CONFIRM_SEED = 20260821
CONFIRM_SOURCE_ID = "lichess-standard-rated-confirm-v1"
CONFIRM_SOURCE_FAMILY = "lichess-standard-rated-v1"
CONFIRM_DATASET_ID = "s6-eval-v1-residual-confirm01"
EXPECTED_CONFIRM_FAMILIES = {CONFIRM_SOURCE_FAMILY}

EXPECTED_ARCHIVE_SHA256 = (
    "68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7")
EXPECTED_ENGINE_BINARY_SHA256 = (
    "05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66")
EXPECTED_TEACHER_BINARY_SHA256 = (
    "6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9")
EXPECTED_TEACHER_NODES = 16384
EXPECTED_TEACHER_OPTIONS = {"Threads": "1", "Hash": "64", "MultiPV": "1",
                            "UCI_ShowWDL": "true"}
EXPECTED_N3B_DATASET_SHA256 = (
    "5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af")
EXPECTED_N3C_CLASSICAL_CACHE_SHA256 = (
    "c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727")

MIN_TEACHER_AUDIT_CHECKED = 1000
MIN_ELIGIBLE_USABLE_POSITIONS = 5000
MIN_RETAINED_FRACTION = 0.90

MIN_MAE_IMPROVEMENT_FRACTION = 0.02
CI_UPPER_MUST_BE_BELOW_CP = 0.0
MAX_GROUP_REGRESSION_FRACTION = 0.02
MIN_GROUP_N = 200

PASS_STATUS = "RESIDUAL_CONFIRMATION_PASS"
FAIL_STATUS = "CONFIRMATION_FAIL"
PASS_VERDICT = "RESIDUAL_CONFIRMED_AWAITING_RUNTIME_REVIEW"
FAIL_VERDICT = "RESIDUAL_NOT_CONFIRMED"


def fail(message: str) -> None:
    raise SystemExit(f"PIPELINE_FAILURE: {message}")


def expect(actual, expected, label: str):
    if actual != expected:
        fail(f"{label}: {actual!r} != expected {expected!r}")
    return actual


def pgn_fingerprints(path: Path, *, require_games: bool = True) -> dict[str, int]:
    """Fingerprint -> game count for every game in a PGN (streamed).

    `require_games=False` tolerates the empty placeholder PGNs the aggregate
    arena catalog carries (smoke-sprt-terminal, smoke-p412-rated); they
    contribute no games and therefore no fingerprints.
    """
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            key = ls.game_fingerprint(game)
            counts[key] = counts.get(key, 0) + 1
    if require_games and not counts:
        fail(f"no games in {path}")
    return counts


def n3b_source_fingerprints(source_dirs: list[Path]) -> tuple[set[str], dict]:
    """Fingerprints of EVERY game in EVERY N3B source PGN."""
    catalog = bd.load_source_catalog(source_dirs)
    keys: set[str] = set()
    per_source: dict[str, int] = {}
    for name in sorted(catalog):
        path = bd.find_pgn(source_dirs, name, catalog[name])
        counts = pgn_fingerprints(path, require_games=False)
        per_source[name] = sum(counts.values())
        keys |= set(counts)
    if not keys:
        fail("N3B sources produced no game fingerprints")
    return keys, {"per_source_games": per_source,
                  "unique_fingerprints": len(keys),
                  "total_games": sum(per_source.values())}


def verify_confirm_source(source_dir: Path) -> dict:
    """Gate the confirmation source manifest against the frozen contract."""
    manifest_path = source_dir / "source-manifest.json"
    if not manifest_path.is_file():
        fail(f"confirmation source manifest missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expect(manifest.get("source_id"), CONFIRM_SOURCE_ID, "confirm source_id")
    expect(manifest.get("source_family"), CONFIRM_SOURCE_FAMILY,
           "confirm source_family")
    expect(manifest.get("selection_seed"), CONFIRM_SEED, "confirm seed")
    expect(manifest.get("fingerprint_intersection"), 0,
           "confirm fingerprint intersection")
    official = manifest.get("official_sha256", {})
    if EXPECTED_ARCHIVE_SHA256 not in official.values():
        fail(f"confirmation archive SHA {official} != {EXPECTED_ARCHIVE_SHA256}")
    if manifest.get("exclude_fingerprint_count", 0) <= 0:
        fail("confirmation source recorded no exclude fingerprints")
    pgn_path = source_dir / f"{CONFIRM_SOURCE_ID}.pgn"
    actual_pgn_sha = residual.sha256_file(pgn_path)
    expect(actual_pgn_sha, manifest.get("pgn_sha256"), "confirm PGN sha256")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": residual.sha256_file(manifest_path),
        "pgn_path": str(pgn_path),
        "pgn_sha256": actual_pgn_sha,
        "archive_official_sha256": official,
        "selection_seed": manifest["selection_seed"],
        "games_selected": manifest.get("games_selected"),
        "selector_script_sha256": manifest.get("script_sha256"),
        "exclude_sources": manifest.get("exclude_sources"),
        "exclude_fingerprint_count": manifest.get("exclude_fingerprint_count"),
        "exclude_fingerprints_sha256": manifest.get(
            "exclude_fingerprints_sha256"),
        "selected_fingerprint_count": manifest.get("selected_fingerprint_count"),
        "selected_fingerprints_sha256": manifest.get(
            "selected_fingerprints_sha256"),
        "fingerprint_intersection": manifest["fingerprint_intersection"],
        "duplicate_candidates_rejected": manifest.get(
            "duplicate_candidates_rejected"),
        "excluded_candidates_skipped": manifest.get(
            "excluded_candidates_skipped"),
        "fingerprint_definition": manifest.get("fingerprint"),
    }


def verify_teacher(teacher_manifest: dict) -> dict:
    expect(teacher_manifest.get("verified_binary_sha256"),
           EXPECTED_TEACHER_BINARY_SHA256, "teacher binary sha256")
    expect(teacher_manifest.get("nodes"), EXPECTED_TEACHER_NODES,
           "teacher nodes")
    expect(teacher_manifest.get("options"), EXPECTED_TEACHER_OPTIONS,
           "teacher options")
    audit = teacher_manifest.get("audit", {})
    expect(audit.get("mode"), "fresh-second-pass", "teacher audit mode")
    if not audit.get("ok"):
        fail("teacher audit not ok")
    if audit.get("checked", 0) < MIN_TEACHER_AUDIT_CHECKED:
        fail(f"teacher audit checked {audit.get('checked')} < "
             f"{MIN_TEACHER_AUDIT_CHECKED}")
    if audit.get("mismatches"):
        fail(f"teacher audit mismatches {len(audit['mismatches'])} != 0")
    return {
        "uci_id_name": teacher_manifest.get("uci_id_name"),
        "binary_sha256": teacher_manifest.get("verified_binary_sha256"),
        "nodes": teacher_manifest.get("nodes"),
        "options": teacher_manifest.get("options"),
        "audit_mode": audit.get("mode"),
        "audit_checked": audit.get("checked"),
        "audit_mismatches": len(audit.get("mismatches", [])),
        "audit_sample_position_id_sha256": audit.get(
            "sample_position_id_sha256"),
        "labels_sha256": teacher_manifest.get("labels_sha256"),
        "labeled_positions": teacher_manifest.get("labeled_positions"),
    }


def verify_labeled_dataset(dataset_dir: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "verify_dataset.py"),
         "--dataset", str(dataset_dir)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        fail(f"verify_dataset rc={proc.returncode}: {proc.stdout[-400:]}")
    return {"returncode": proc.returncode,
            "verdict": "VERIFY_PASS" if "VERIFY_PASS" in proc.stdout
            else "UNKNOWN"}


def identity_audit(confirm: dict, n3b_records: list[dict],
                   confirm_pgn: Path,
                   n3b_source_dirs: list[Path]) -> tuple[dict, list[dict]]:
    """Fingerprint disjointness + exact position-id exclusion vs FULL N3B."""
    confirm_keys = pgn_fingerprints(confirm_pgn)
    n3b_keys, n3b_key_stats = n3b_source_fingerprints(n3b_source_dirs)
    key_intersection = sorted(set(confirm_keys) & n3b_keys)

    labels = confirm["data"]["labels"]
    records = confirm["records"]
    usable = [row for row in records
              if labels[row["position_id"]].get("teacher_cp_stm") is not None]
    n3b_pids = {row["position_id"] for row in n3b_records}
    usable_pids = {row["position_id"] for row in usable}
    excluded = sorted(usable_pids & n3b_pids)
    eligible = [row for row in usable if row["position_id"] not in set(excluded)]
    retained = len(eligible) / len(usable) if usable else 0.0

    audit = {
        "policy": "exclude-exact-position-id-overlap-before-inference",
        "compared_against": "FULL N3B dataset (all splits) and all N3B source "
                            "game fingerprints",
        "selection_uses_labels_or_predictions": False,
        "confirmation_used_for_training_or_early_stopping": False,
        "game_fingerprints": {
            "confirmation_games": sum(confirm_keys.values()),
            "confirmation_unique_fingerprints": len(confirm_keys),
            "confirmation_fingerprints_sha256": ls.fingerprint_set_sha256(
                set(confirm_keys)),
            "n3b_fingerprints_sha256": ls.fingerprint_set_sha256(n3b_keys),
            **n3b_key_stats,
            "intersection": len(key_intersection),
            "intersection_examples": key_intersection[:5],
        },
        "positions": {
            "raw_records": len(records),
            "usable_records": len(usable),
            "null_cp_records": len(records) - len(usable),
            "n3b_position_ids": len(n3b_pids),
            "excluded_positions": len(excluded),
            "excluded_ids_sha256": diag.sha256_ids(excluded),
            "eligible_positions": len(eligible),
            "retained_fraction": round(retained, 6),
        },
        "gates": {
            "game_fingerprint_intersection_zero": len(key_intersection) == 0,
            "eligible_usable_at_least_5000":
                len(eligible) >= MIN_ELIGIBLE_USABLE_POSITIONS,
            "retained_fraction_at_least_0.90":
                retained >= MIN_RETAINED_FRACTION,
        },
    }
    audit["passed"] = all(audit["gates"].values())
    return audit, eligible


def checkpoint_roundtrip(model: probe.NnueProbe, split: dict,
                         checkpoint_dir: Path) -> dict:
    """Disk roundtrip on THIS host: re-save, reload, require identical output."""
    before, loss_before = probe.evaluate_split(model, split)
    path = checkpoint_dir / "n3d-roundtrip.pt"
    torch.save({"state_dict": model.state_dict(),
                "architecture": {"inputs": probe.NNUE_INPUTS,
                                 "width": residual.RESIDUAL_WIDTH}},
               path)
    reloaded = residual.build_residual_model()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["state_dict"])
    after, loss_after = probe.evaluate_split(reloaded, split)
    state = diag.state_comparison(model.state_dict(), reloaded.state_dict())
    prediction_delta = float((before - after).abs().max().item()) \
        if len(before) else 0.0
    result = {
        "state_exact_equal": state["exact_equal"],
        "state_max_abs_delta": state["max_abs_delta"],
        "loss_before": round(loss_before, 8),
        "loss_after": round(loss_after, 8),
        "loss_abs_delta": round(abs(loss_before - loss_after), 12),
        "prediction_max_abs_delta_scaled": prediction_delta,
    }
    result["passed"] = bool(
        state["exact_equal"] and prediction_delta == 0.0
        and abs(loss_before - loss_after) <= 1e-6)
    return result


def all_finite(obj, prefix: str = "") -> list[str]:
    """Every float in a nested result payload must be finite."""
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


def group_gate(by_group: dict, kind: str) -> dict:
    """No bucket with n >= MIN_GROUP_N may regress by more than 2%."""
    checks = {}
    for name, data in by_group.get(kind, {}).items():
        n = data.get("n", 0)
        if n < MIN_GROUP_N:
            checks[name] = {"n": n, "evaluated": False,
                            "reason": f"n < {MIN_GROUP_N}"}
            continue
        classical_mae = data["classical"]["clipped_mae_cp"]
        residual_mae = data["residual"]["clipped_mae_cp"]
        regression = ((residual_mae - classical_mae) / classical_mae
                      if classical_mae else None)
        checks[name] = {
            "n": n,
            "evaluated": True,
            "classical_clipped_mae_cp": classical_mae,
            "residual_clipped_mae_cp": residual_mae,
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


def evaluate_gates(overall: dict, bootstrap: dict, by_group: dict,
                   identity: dict, cache_ok: bool, roundtrip: dict,
                   finite_violations: list[str]) -> dict:
    improvement = overall["mae_improvement_fraction"]
    gate1 = {
        "id": 1,
        "name": "overall_clipped_mae_improvement",
        "threshold": f">= {MIN_MAE_IMPROVEMENT_FRACTION}",
        "classical_clipped_mae_cp": overall["classical"]["clipped_mae_cp"],
        "residual_clipped_mae_cp": overall["residual"]["clipped_mae_cp"],
        "observed_improvement_fraction": improvement,
        "passed": improvement is not None
        and improvement >= MIN_MAE_IMPROVEMENT_FRACTION,
    }
    gate2 = {
        "id": 2,
        "name": "paired_abs_error_delta_bootstrap_ci_upper_below_zero",
        "threshold": f"< {CI_UPPER_MUST_BE_BELOW_CP} cp",
        "bootstrap_seed": bootstrap["bootstrap_seed"],
        "bootstrap_resamples": bootstrap["bootstrap_resamples"],
        "mean_delta_cp": bootstrap["mean_delta_cp"],
        "median_delta_cp": bootstrap["median_delta_cp"],
        "ci_lower_cp": bootstrap["ci_lower_cp"],
        "ci_upper_cp": bootstrap["ci_upper_cp"],
        "passed": bootstrap["ci_upper_cp"] < CI_UPPER_MUST_BE_BELOW_CP,
    }
    gate3 = {
        "id": 3,
        "name": "residual_clipped_rmse_not_worse",
        "threshold": "residual <= classical",
        "classical_clipped_rmse_cp": overall["classical"]["clipped_rmse_cp"],
        "residual_clipped_rmse_cp": overall["residual"]["clipped_rmse_cp"],
        "passed": overall["residual"]["clipped_rmse_cp"]
        <= overall["classical"]["clipped_rmse_cp"],
    }
    phase = group_gate(by_group, "phase")
    gate4 = {"id": 4, "name": "phase_bucket_no_regression", **phase}
    abs_cp = group_gate(by_group, "abs_cp")
    gate5 = {"id": 5, "name": "abs_teacher_cp_bucket_no_regression", **abs_cp}
    gate6 = {
        "id": 6,
        "name": "integrity",
        "all_metrics_finite": not finite_violations,
        "non_finite_paths": finite_violations[:10],
        "checkpoint_roundtrip": roundtrip,
        "identity_gates": identity["gates"],
        "classical_cache_validated": cache_ok,
        "passed": bool(not finite_violations and roundtrip["passed"]
                       and identity["passed"] and cache_ok),
    }
    gates = [gate1, gate2, gate3, gate4, gate5, gate6]
    return {
        "all_passed": all(gate["passed"] for gate in gates),
        "gates": gates,
        "frozen_thresholds": {
            "min_mae_improvement_fraction": MIN_MAE_IMPROVEMENT_FRACTION,
            "ci_upper_must_be_below_cp": CI_UPPER_MUST_BE_BELOW_CP,
            "max_group_regression_fraction": MAX_GROUP_REGRESSION_FRACTION,
            "min_group_n": MIN_GROUP_N,
            "min_eligible_usable_positions": MIN_ELIGIBLE_USABLE_POSITIONS,
            "min_retained_fraction": MIN_RETAINED_FRACTION,
            "bootstrap_seed": residual.BOOTSTRAP_SEED,
            "bootstrap_resamples": residual.BOOTSTRAP_RESAMPLES,
        },
    }


def render_markdown(result: dict) -> str:
    overall = result["confirmation"]["overall"]
    boot = result["confirmation"]["paired_bootstrap"]
    identity = result["identity_audit"]
    lines = [
        "# S6-N3D - Independent Residual Confirmation", "",
        f"STATUS: **{result['status']}**", "",
        f"VERDICT: **{result['verdict']}**", "",
        "## Provenance and bindings", "", "```text",
        f"run git:              {result['provenance']['run_git_sha']}",
        f"runner blob:          {result['provenance']['committed_script_blob_sha256']}",
        f"trainer blob:         {result['provenance']['committed_trainer_blob_sha256']}",
        f"canonical checkpoint: {result['checkpoint']['sha256']}",
        f"engine binary:        {result['bindings']['engine_binary_sha256']}",
        f"archive (official):   {result['bindings']['archive_sha256']}",
        f"confirm PGN:          {result['bindings']['confirmation_pgn_sha256']}",
        f"confirm dataset:      {result['bindings']['confirmation_dataset_sha256']}",
        f"confirm labels:       {result['bindings']['confirmation_labels_sha256']}",
        f"teacher binary:       {result['bindings']['teacher_binary_sha256']}",
        f"confirm cache:        {result['bindings']['confirmation_cache_sha256']}",
        f"selected keys sha:    {result['bindings']['selected_fingerprints_sha256']}",
        f"exclude keys sha:     {result['bindings']['exclude_fingerprints_sha256']}",
        "```", "",
        "## Identity audit vs the FULL N3B dataset", "",
        "| check | value |", "|---|---|",
        f"| confirmation games | {identity['game_fingerprints']['confirmation_games']} |",
        f"| N3B source games | {identity['game_fingerprints']['total_games']} |",
        f"| game fingerprint intersection | {identity['game_fingerprints']['intersection']} |",
        f"| raw records | {identity['positions']['raw_records']} |",
        f"| usable records | {identity['positions']['usable_records']} |",
        f"| excluded (position_id overlap) | {identity['positions']['excluded_positions']} |",
        f"| eligible positions | {identity['positions']['eligible_positions']} |",
        f"| retained fraction | {identity['positions']['retained_fraction']} |",
        "",
        "## Confirmation result (single run, disk-loaded checkpoint)", "",
        "| predictor | n | clipped MAE | clipped RMSE |",
        "|---|---:|---:|---:|",
        f"| classical | {overall['n']} | "
        f"{overall['classical']['clipped_mae_cp']} | "
        f"{overall['classical']['clipped_rmse_cp']} |",
        f"| classical + residual | {overall['n']} | "
        f"{overall['residual']['clipped_mae_cp']} | "
        f"{overall['residual']['clipped_rmse_cp']} |",
        "",
        f"MAE improvement: **{overall['mae_improvement_fraction']:.4%}** "
        f"({overall['mae_delta_cp']} cp)",
        "",
        "### Paired per-position absolute-error delta", "",
        "`abs(residual - teacher) - abs(classical - teacher)`, negative is "
        "better.", "", "```text",
        f"n                {boot['n']}",
        f"mean             {boot['mean_delta_cp']} cp",
        f"median           {boot['median_delta_cp']} cp",
        f"improved         {boot['positions_improved']}",
        f"worsened         {boot['positions_worsened']}",
        f"unchanged        {boot['positions_unchanged']}",
        f"bootstrap        numpy default_rng(seed={boot['bootstrap_seed']}), "
        f"{boot['bootstrap_resamples']} resamples",
        f"95% CI           [{boot['ci_lower_cp']}, {boot['ci_upper_cp']}] cp",
        "```", "",
        "## Gates", "", "| # | gate | pass |", "|---:|---|---|",
    ]
    for gate in result["gate_evaluation"]["gates"]:
        lines.append(f"| {gate['id']} | {gate['name']} | "
                     f"{'PASS' if gate['passed'] else 'FAIL'} |")
    for kind, title in (("phase", "phase bucket"),
                        ("abs_cp", "|teacher CP| bucket")):
        lines += ["", f"### By {title}", "",
                  "| group | n | classical MAE | residual MAE | regression | "
                  "evaluated |", "|---|---:|---:|---:|---:|---|"]
        for name, data in result["confirmation"]["by_group"][kind].items():
            gate_entry = next(
                (gate for gate in result["gate_evaluation"]["gates"]
                 if gate["id"] == (4 if kind == "phase" else 5)), {})
            check = gate_entry.get("groups", {}).get(name, {})
            if data.get("n", 0) == 0:
                lines.append(f"| {name} | 0 | - | - | - | no |")
                continue
            lines.append(
                f"| {name} | {data['n']} | "
                f"{data['classical']['clipped_mae_cp']} | "
                f"{data['residual']['clipped_mae_cp']} | "
                f"{check.get('regression_fraction', '-')} | "
                f"{'yes' if check.get('evaluated') else 'no'} |")
    lines += ["", "### By source family", "",
              "| group | n | classical MAE | residual MAE |",
              "|---|---:|---:|---:|"]
    for name, data in result["confirmation"]["by_group"]["family"].items():
        if data.get("n", 0) == 0:
            lines.append(f"| {name} | 0 | - | - |")
            continue
        lines.append(f"| {name} | {data['n']} | "
                     f"{data['classical']['clipped_mae_cp']} | "
                     f"{data['residual']['clipped_mae_cp']} |")
    lines += [
        "",
        f"Single evaluation run. No training, no width/seed re-selection, no "
        f"threshold change, no game re-selection. "
        f"{result['outcome']['next_step']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="confirmation dataset (evaluation only)")
    parser.add_argument("--n3b-dataset", type=Path, required=True)
    parser.add_argument("--confirm-source", type=Path, required=True)
    parser.add_argument("--n3b-sources", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "data/s6/models"
                        / "s6-n3d-residual-w16-s20260818.pt")
    parser.add_argument("--classical-cache", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3d-classical-cache-confirm01.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "results/s6"
                        / "s6-n3d-residual-confirmation.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    provenance = residual.script_provenance(REPO, Path(__file__))
    engine_sha = expect(residual.sha256_file(args.engine),
                        EXPECTED_ENGINE_BINARY_SHA256, "engine binary sha256")

    source_info = verify_confirm_source(args.confirm_source)
    verify_report = verify_labeled_dataset(args.dataset)

    confirm_family_map, confirm_manifest_shas = probe.build_family_map(
        [str(args.confirm_source)])
    confirm = diag.load_prepared(args.dataset, args.engine, confirm_family_map)
    expect(confirm["data"]["manifest"]["dataset_id"], CONFIRM_DATASET_ID,
           "confirmation dataset_id")
    for record in confirm["records"]:
        if record["source_id"] != CONFIRM_SOURCE_ID:
            fail(f"unexpected source_id {record['source_id']} in confirmation "
                 f"dataset")
    families = {family for split in confirm["prepared"].values()
                for family in split["source_families"]}
    expect(families, EXPECTED_CONFIRM_FAMILIES, "confirmation family set")
    teacher = verify_teacher(confirm["data"]["teacher_manifest"])

    n3b = probe.load_dataset(args.n3b_dataset)
    expect(n3b["dataset_sha"], EXPECTED_N3B_DATASET_SHA256, "N3B dataset sha")

    identity, eligible_records = identity_audit(
        confirm, n3b["records"], Path(source_info["pgn_path"]),
        [Path(path) for path in args.n3b_sources])
    print(f"identity audit: eligible={identity['positions']['eligible_positions']} "
          f"retained={identity['positions']['retained_fraction']} "
          f"key_intersection={identity['game_fingerprints']['intersection']}",
          flush=True)

    cache = diag.build_classical_cache(
        confirm, args.engine, args.classical_cache, engine_sha)
    expect(cache["header"]["dataset_sha256"], confirm["data"]["dataset_sha"],
           "confirmation cache dataset sha")
    classical = cache["values"]

    model, metadata = residual.load_canonical_checkpoint(args.checkpoint)
    expect(metadata["dataset_sha256"], EXPECTED_N3B_DATASET_SHA256,
           "checkpoint training dataset sha")
    expect(metadata["classical_cache_sha256"],
           EXPECTED_N3C_CLASSICAL_CACHE_SHA256, "checkpoint cache sha")
    expect(metadata["engine_binary_sha256"], EXPECTED_ENGINE_BINARY_SHA256,
           "checkpoint engine sha")

    # Evaluation split = ALL eligible confirmation rows, every split pooled.
    # Nothing here trains, so no confirmation row can reach early stopping.
    eval_split = probe.prepare_split(
        confirm["exported"], eligible_records, confirm["data"]["labels"])
    diag.add_family_metadata(eval_split, confirm_family_map)
    eval_residual = residual.residual_targets(eval_split, classical)

    with tempfile.TemporaryDirectory(prefix="s6-n3d-roundtrip-") as temp:
        roundtrip = checkpoint_roundtrip(model, eval_residual, Path(temp))
    if not roundtrip["passed"]:
        print(f"checkpoint roundtrip FAILED: {roundtrip}", flush=True)

    classical_cp = residual.classical_predictions_cp(eval_split, classical)
    residual_cp, residual_loss = residual.residual_predictions_cp(
        model, eval_residual, classical)
    deltas = residual.abs_error_deltas(
        residual_cp, classical_cp, eval_split["raw_target_cp"])
    overall = residual.comparison(
        residual_cp, classical_cp, eval_split["raw_target_cp"])
    bootstrap = residual.paired_bootstrap(deltas)
    by_group = residual.grouped_comparison(
        eval_split, residual_cp, classical_cp)

    confirmation = {
        "evaluation_positions": len(eval_split["pids"]),
        "splits_pooled": ["train", "validation", "holdout"],
        "role": "evaluation only; no confirmation record entered training or "
                "early stopping",
        "trained_here": False,
        "residual_loss_on_residual_target": round(residual_loss, 6),
        "overall": overall,
        "paired_bootstrap": bootstrap,
        "by_group": by_group,
        "residual_prediction_stats": probe.pred_stats(residual_cp),
        "classical_prediction_stats": probe.pred_stats(classical_cp),
    }
    finite_violations = all_finite(
        {"confirmation": confirmation, "identity_audit": identity})
    gate_evaluation = evaluate_gates(
        overall, bootstrap, by_group, identity, True, roundtrip,
        finite_violations)
    passed = gate_evaluation["all_passed"]

    result = {
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "authorization": "docs/s6/s6-n3c-closure-and-n3d-authorization.md",
        "provenance": provenance,
        "bindings": {
            "engine_binary_sha256": engine_sha,
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_official_sha256": source_info["archive_official_sha256"],
            "confirmation_source_id": CONFIRM_SOURCE_ID,
            "confirmation_source_family": CONFIRM_SOURCE_FAMILY,
            "confirmation_selection_seed": CONFIRM_SEED,
            "confirmation_pgn_sha256": source_info["pgn_sha256"],
            "confirmation_source_manifest_sha256":
                source_info["manifest_sha256"],
            "selector_script_sha256": source_info["selector_script_sha256"],
            "selected_fingerprints_sha256":
                source_info["selected_fingerprints_sha256"],
            "exclude_fingerprints_sha256":
                source_info["exclude_fingerprints_sha256"],
            "confirmation_dataset_id": CONFIRM_DATASET_ID,
            "confirmation_dataset_sha256": confirm["data"]["dataset_sha"],
            "confirmation_labels_sha256": confirm["data"]["labels_sha"],
            "confirmation_cache_path": str(args.classical_cache),
            "confirmation_cache_sha256": cache["sha256"],
            "confirmation_cache_header": cache["header"],
            "teacher_binary_sha256": teacher["binary_sha256"],
            "n3b_dataset_sha256": n3b["dataset_sha"],
            "n3b_labels_sha256": n3b["labels_sha"],
            "source_manifests": confirm_manifest_shas,
            "source_id_to_family": confirm_family_map,
        },
        "confirmation_source": source_info,
        "teacher": teacher,
        "dataset_verification": verify_report,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": residual.sha256_file(args.checkpoint),
            "metadata": metadata,
            "loaded_from_disk": True,
            "trained_in_this_run": False,
            "roundtrip": roundtrip,
        },
        "config": {
            "cpu_only": True, "cuda_used": False,
            "torch_num_threads": torch.get_num_threads(),
            "width": residual.RESIDUAL_WIDTH,
            "seed": residual.RESIDUAL_SEED,
            "target_mode": residual.TARGET_MODE,
            "target_formula": residual.RESIDUAL_TARGET_FORMULA,
            "inference_formula": residual.RESIDUAL_INFERENCE_FORMULA,
            "clip_cp": probe.CLIP_CP, "target_scale": probe.TARGET_SCALE,
        },
        "identity_audit": identity,
        "confirmation": confirmation,
        "gate_evaluation": gate_evaluation,
        "outcome": {
            "single_evaluation_run": True,
            "trained_in_this_run": False,
            "width_or_seed_reselected": False,
            "thresholds_changed": False,
            "games_reselected": False,
            "n3b_dataset_relabeled": False,
            "existing_splits_used_as_gate": False,
            "next_step": (
                "bench-only residual artifact parity/cost may begin in the "
                "NEXT round; no runtime, search, exporter or Arena work"
                if passed else
                "NNUE residual sub-branch ends; do not retune or re-run"),
        },
    }
    out_sha = residual.write_json(args.out, result)
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(f"results written to {args.out} sha256={out_sha}", flush=True)
    print(f"markdown written to {md_path}", flush=True)
    print(f"status: {result['status']}", flush=True)
    print(f"verdict: {result['verdict']}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
