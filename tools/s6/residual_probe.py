#!/usr/bin/env python3
"""S6-N3D shared residual-probe library.

Frozen residual contract, authorized by
`docs/s6/s6-n3c-closure-and-n3d-authorization.md`:

    architecture  40960 x 16 shared table, shared 16-dim accumulator bias,
                  ReLU via clamp-min(0), head concat(own, opponent) 32 -> 1
    width         16
    seed          20260818
    target        clamp(teacher_cp_stm - base_eval_stm, +-2000) / 1000
    inference     base_eval_stm + residual * 1000

Nothing here re-tunes width, seed, optimizer, target, or any threshold. The
learning internals are reused verbatim: `train_nnue_probe.train_probe` (best
state restore + early stopping on validation only),
`train_nnue_probe.bind_run_provenance` (clean tracked tree + disk == HEAD
blob), `train_nnue_probe.clipped_metrics`, and the already-tested
`run_n3c_probe_diagnostics.residual_split` / classical-cache builder. This
module adds only the canonical-checkpoint format, the paired-bootstrap
statistic, and the grouped classical-vs-residual comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_n3c_probe_diagnostics as diag  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

TARGET_MODE = "classical_residual"
RESIDUAL_WIDTH = 16
RESIDUAL_SEED = 20260818
RESIDUAL_TARGET_FORMULA = (
    "clamp(teacher_cp_stm - base_eval_stm, -2000, 2000) / 1000")
RESIDUAL_INFERENCE_FORMULA = "base_eval_stm + residual * 1000"

# Deterministic paired bootstrap (frozen before any confirmation label existed).
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_RESAMPLES = 10000
CI_PERCENTILES = (2.5, 97.5)

PHASE_BUCKET_NAMES = ("high", "mid", "low", "zero")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clip_cp(value: float) -> float:
    return max(-probe.CLIP_CP, min(probe.CLIP_CP, float(value)))


def abs_cp_bucket(raw_target_cp: float) -> str:
    """Bucket key by RAW |teacher cp|, matching probe.clipped_metrics."""
    magnitude = abs(float(raw_target_cp))
    for lo, hi in probe.CP_BUCKETS:
        if magnitude >= lo and (hi is None or magnitude < hi):
            return f"{lo}-{hi if hi else 'inf'}"
    raise SystemExit(
        f"PIPELINE_FAILURE: no abs-cp bucket for target {raw_target_cp}")


def script_provenance(repo: Path, script_path: Path) -> dict:
    """Clean tracked tree + committed trainer blob + committed OWN blob.

    `probe.bind_run_provenance` binds the run to the committed trainer; this
    additionally byte-compares the calling script against its HEAD blob, so a
    measurement can only ever be produced by a committed script on a clean
    commit.
    """
    clean = probe.bind_run_provenance(repo)
    script_path = Path(script_path).resolve()
    relative = script_path.relative_to(repo)
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
        capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"PIPELINE_FAILURE: cannot read committed blob for {relative}")
    if script_path.read_bytes() != proc.stdout:
        raise SystemExit(
            f"PIPELINE_FAILURE: disk {relative} differs from HEAD blob")
    return {
        **clean,
        "script_relpath": str(relative),
        "script_sha256": sha256_file(script_path),
        "committed_script_blob_sha256": hashlib.sha256(proc.stdout).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Model / target plumbing (frozen width and seed)
# ---------------------------------------------------------------------------

def build_residual_model(width: int = RESIDUAL_WIDTH,
                         seed: int = RESIDUAL_SEED) -> probe.NnueProbe:
    """CPU-only frozen-width probe; make_model already rejects non-CPU."""
    return diag.make_model(width, seed)


def residual_targets(prepared: dict, classical: dict[str, float]) -> dict:
    """Attach the frozen residual target; reuses the tested N3C path."""
    return diag.residual_split(prepared, classical)


def classical_predictions_cp(split: dict,
                             classical: dict[str, float]) -> list[float]:
    return [float(classical[pid]) for pid in split["pids"]]


def residual_predictions_cp(model: probe.NnueProbe, split: dict,
                            classical: dict[str, float]) -> tuple[list[float], float]:
    """Final cp predictions under `base_eval_stm + residual * 1000`."""
    predictions, loss = probe.evaluate_split(model, split)
    return ([float(classical[pid]) + float(value) * probe.TARGET_SCALE
             for pid, value in zip(split["pids"], predictions.tolist())],
            loss)


# ---------------------------------------------------------------------------
# Paired per-position statistics
# ---------------------------------------------------------------------------

def abs_error_deltas(residual_cp: list[float], classical_cp: list[float],
                     raw_target_cp: list[float]) -> list[float]:
    """`abs(residual - teacher) - abs(classical - teacher)`, clipped pair.

    Both predictions and the raw teacher target are clamped to +-CLIP_CP
    exactly as `probe.clipped_metrics` does, so the mean of these deltas
    equals `residual_clipped_mae - classical_clipped_mae` by construction.
    Negative means the residual model is closer to the teacher.
    """
    if not (len(residual_cp) == len(classical_cp) == len(raw_target_cp)):
        raise SystemExit("PIPELINE_FAILURE: paired delta length mismatch")
    deltas = []
    for residual, classical, target in zip(residual_cp, classical_cp,
                                           raw_target_cp):
        clipped_target = clip_cp(target)
        delta = (abs(clip_cp(residual) - clipped_target)
                 - abs(clip_cp(classical) - clipped_target))
        if not math.isfinite(delta):
            raise SystemExit("PIPELINE_FAILURE: non-finite paired delta")
        deltas.append(delta)
    return deltas


def paired_bootstrap(deltas: list[float], seed: int = BOOTSTRAP_SEED,
                     resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Deterministic paired bootstrap over per-position deltas.

    One `rng.integers(0, n, size=n)` draw per resample, in order, so the
    result depends only on (deltas, seed, resamples) and never on batching.
    """
    if not deltas:
        raise SystemExit("PIPELINE_FAILURE: paired bootstrap on empty deltas")
    if resamples <= 0:
        raise SystemExit("PIPELINE_FAILURE: paired bootstrap resamples <= 0")
    values = np.asarray(deltas, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise SystemExit("PIPELINE_FAILURE: non-finite delta in bootstrap")
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        means[index] = values[rng.integers(0, n, size=n)].mean()
    lower, upper = (float(value) for value in
                    np.percentile(means, list(CI_PERCENTILES)))
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise SystemExit("PIPELINE_FAILURE: non-finite bootstrap CI")
    return {
        "n": int(n),
        "mean_delta_cp": round(float(values.mean()), 6),
        "median_delta_cp": round(float(np.median(values)), 6),
        "positions_improved": int((values < 0).sum()),
        "positions_worsened": int((values > 0).sum()),
        "positions_unchanged": int((values == 0).sum()),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "bootstrap_rng": "numpy.random.default_rng",
        "ci_percentiles": list(CI_PERCENTILES),
        "ci_lower_cp": round(lower, 6),
        "ci_upper_cp": round(upper, 6),
    }


def delta_summary(deltas: list[float]) -> dict:
    """Mean/median only (no bootstrap); used for small subgroups."""
    return {
        "n": len(deltas),
        "mean_delta_cp": round(statistics.mean(deltas), 6) if deltas else None,
        "median_delta_cp": round(statistics.median(deltas), 6)
        if deltas else None,
    }


# ---------------------------------------------------------------------------
# Grouped classical-vs-residual comparison
# ---------------------------------------------------------------------------

def comparison(residual_cp: list[float], classical_cp: list[float],
               raw_target_cp: list[float]) -> dict:
    """Classical vs classical+residual clipped metrics plus paired deltas."""
    classical_metrics = probe.clipped_metrics(classical_cp, raw_target_cp)
    residual_metrics = probe.clipped_metrics(residual_cp, raw_target_cp)
    classical_mae = classical_metrics["clipped_mae_cp"]
    residual_mae = residual_metrics["clipped_mae_cp"]
    deltas = abs_error_deltas(residual_cp, classical_cp, raw_target_cp)
    return {
        "n": len(raw_target_cp),
        "classical": classical_metrics,
        "residual": residual_metrics,
        "mae_delta_cp": round(residual_mae - classical_mae, 6),
        "mae_improvement_fraction": round(
            (classical_mae - residual_mae) / classical_mae, 6)
        if classical_mae else None,
        "rmse_delta_cp": round(residual_metrics["clipped_rmse_cp"]
                               - classical_metrics["clipped_rmse_cp"], 6),
        "paired_delta": delta_summary(deltas),
    }


def _group_masks(split: dict, kind: str) -> dict[str, list[bool]]:
    if kind == "family":
        families = sorted(set(split.get("source_families", [])))
        return {family: [value == family
                         for value in split["source_families"]]
                for family in families}
    if kind == "phase":
        buckets = [probe.phase_bucket(int(phase)) for phase in split["phases"]]
        return {name: [value == name for value in buckets]
                for name in PHASE_BUCKET_NAMES}
    if kind == "abs_cp":
        keys = [abs_cp_bucket(target) for target in split["raw_target_cp"]]
        return {f"{lo}-{hi if hi else 'inf'}":
                [key == f"{lo}-{hi if hi else 'inf'}" for key in keys]
                for lo, hi in probe.CP_BUCKETS}
    raise SystemExit(f"PIPELINE_FAILURE: unknown group kind {kind}")


def grouped_comparison(split: dict, residual_cp: list[float],
                       classical_cp: list[float],
                       kinds: tuple[str, ...] = ("family", "phase", "abs_cp")
                       ) -> dict:
    """Per-group comparison over family, phase bucket and |teacher CP| bucket.

    Empty groups are reported as `{"n": 0}` rather than dropped, so a reader
    can never mistake an absent group for an unreported one.
    """
    raw_target = split["raw_target_cp"]
    result: dict[str, dict] = {}
    for kind in kinds:
        result[kind] = {}
        for name, mask in _group_masks(split, kind).items():
            indices = [i for i, keep in enumerate(mask) if keep]
            if not indices:
                result[kind][name] = {"n": 0}
                continue
            result[kind][name] = comparison(
                [residual_cp[i] for i in indices],
                [classical_cp[i] for i in indices],
                [raw_target[i] for i in indices])
    return result


# ---------------------------------------------------------------------------
# Canonical checkpoint format
# ---------------------------------------------------------------------------

def canonical_checkpoint_payload(model: probe.NnueProbe, *, width: int,
                                 seed: int, dataset_sha256: str,
                                 labels_sha256: str,
                                 classical_cache_sha256: str,
                                 engine_binary_sha256: str,
                                 best_epoch: int, best_val_loss: float,
                                 trainer_git_sha: str,
                                 trainer_blob_sha256: str,
                                 dataset_id: str) -> dict:
    return {
        "state_dict": model.state_dict(),
        "target_mode": TARGET_MODE,
        "target_formula": RESIDUAL_TARGET_FORMULA,
        "inference_formula": RESIDUAL_INFERENCE_FORMULA,
        "architecture": {
            "inputs": probe.NNUE_INPUTS,
            "width": width,
            "activation": "relu",
            "head": "concat(own,opp) width*2->1 linear",
        },
        "width": width,
        "seed": seed,
        "clip_cp": probe.CLIP_CP,
        "target_scale": probe.TARGET_SCALE,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "labels_sha256": labels_sha256,
        "classical_cache_sha256": classical_cache_sha256,
        "engine_binary_sha256": engine_binary_sha256,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "trainer_git_sha": trainer_git_sha,
        "trainer_blob_sha256": trainer_blob_sha256,
    }


REQUIRED_CHECKPOINT_KEYS = (
    "state_dict", "target_mode", "target_formula", "inference_formula",
    "architecture", "width", "seed", "clip_cp", "target_scale",
    "dataset_sha256", "labels_sha256", "classical_cache_sha256",
    "engine_binary_sha256", "best_epoch", "best_val_loss",
    "trainer_git_sha", "trainer_blob_sha256",
)

# The architecture block is compared field-by-field, not just on width: a
# checkpoint that silently changed activation or head wiring would otherwise
# load and score, producing numbers that no longer describe the frozen probe.
EXPECTED_ARCHITECTURE = {
    "inputs": probe.NNUE_INPUTS,
    "width": RESIDUAL_WIDTH,
    "activation": "relu",
    "head": "concat(own,opp) width*2->1 linear",
}


def load_canonical_checkpoint(path: Path, *, width: int = RESIDUAL_WIDTH,
                              seed: int = RESIDUAL_SEED,
                              expected_sha256: str | None = None
                              ) -> tuple[probe.NnueProbe, dict]:
    """Load and fully validate a canonical residual checkpoint.

    `expected_sha256` is checked BEFORE the torch payload is deserialized, so a
    substituted or corrupted file is rejected without ever being unpickled.
    Every frozen contract field is then verified exactly - target mode and both
    formulas, clip/scale, and the whole architecture block - because a
    checkpoint that merely has the right width can still describe a different
    model.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"PIPELINE_FAILURE: checkpoint missing {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint sha256 {actual_sha256} != expected "
            f"{expected_sha256}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS
               if key not in checkpoint]
    if missing:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint {path} missing metadata {missing}")
    for key, expected in (("target_mode", TARGET_MODE),
                          ("target_formula", RESIDUAL_TARGET_FORMULA),
                          ("inference_formula", RESIDUAL_INFERENCE_FORMULA),
                          ("clip_cp", probe.CLIP_CP),
                          ("target_scale", probe.TARGET_SCALE),
                          ("seed", seed),
                          ("width", width)):
        if checkpoint[key] != expected:
            raise SystemExit(
                f"PIPELINE_FAILURE: checkpoint {key} {checkpoint[key]!r} != "
                f"expected {expected!r}")
    architecture = checkpoint["architecture"]
    if not isinstance(architecture, dict):
        raise SystemExit("PIPELINE_FAILURE: checkpoint architecture not a dict")
    expected_architecture = {**EXPECTED_ARCHITECTURE, "width": width}
    if architecture != expected_architecture:
        raise SystemExit(
            f"PIPELINE_FAILURE: checkpoint architecture {architecture!r} != "
            f"expected {expected_architecture!r}")
    model = build_residual_model(width, seed)
    model.load_state_dict(checkpoint["state_dict"])
    metadata = {key: value for key, value in checkpoint.items()
                if key != "state_dict"}
    metadata["checkpoint_sha256"] = actual_sha256
    return model, metadata


def write_json(path: Path, payload: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return sha256_file(path)
