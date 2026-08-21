#!/usr/bin/env python3
"""S6-N3E cheap calibration candidates (the null hypothesis for N3D's gain).

N3D showed `classical + NNUE residual` beats raw classical on unseen games,
but the published prediction statistics are also consistent with the NNUE
having learned a nearly CONSTANT correction of about +41.38 cp - i.e. that
this engine's classical evaluation is simply pessimistic relative to a
16384-node Stockfish 18 teacher. If a one-parameter constant reproduces the
gain, no NNUE runtime work is justified.

This module fits a ladder of deliberately parameter-poor calibrators so the
NNUE has to beat the best cheap explanation, not just raw classical:

    identity       correction = 0                       0 parameters
    global_bias    correction = b                       1
    global_affine  correction = u*x + b                 2
    phase_bias     correction = b[phase]                4
    phase_affine   correction = u[phase]*x + b[phase]   8

with `x = base_eval_stm / 1000` and fitting target
`z = clamp(teacher_cp_stm - base_eval_stm, -2000, 2000) / 1000`. Corrections
are in the same scaled units the NNUE emits, so cp is `correction * 1000` and
the calibrated prediction is `base_eval_stm + correction * 1000` - formed
exactly like the NNUE hybrid, making the comparison apples to apples.

Fitting is deterministic and RNG-free: CPU float64, zero initialization,
full-batch `torch.optim.LBFGS` with `strong_wolfe`, `max_iter=500`, SmoothL1
`beta=0.1`. Parameters come only from the N3B train split; the candidate is
chosen only by N3B validation SmoothL1. Holdout and the N3D confirmation set
never take part in fitting or selection.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_nnue_probe as probe  # noqa: E402

# Frozen fitting protocol. No randomness anywhere in this module.
DTYPE = torch.float64
LBFGS_MAX_ITER = 500
LBFGS_LINE_SEARCH = "strong_wolfe"
LOSS_BETA = probe.LOSS_BETA  # 0.1, same loss the probe was trained under
TIE_TOLERANCE = 1e-9

PHASE_BUCKETS = ("high", "mid", "low", "zero")

# Selection preference order: fewer parameters first. On a validation tie
# within TIE_TOLERANCE the earlier entry wins, so a cheap explanation is never
# displaced by an equally good expensive one.
CANDIDATE_ORDER = ("identity", "global_bias", "global_affine",
                   "phase_bias", "phase_affine")
CANDIDATE_PARAMETER_COUNT = {
    "identity": 0,
    "global_bias": 1,
    "global_affine": 2,
    "phase_bias": len(PHASE_BUCKETS),
    "phase_affine": 2 * len(PHASE_BUCKETS),
}


def fail(message: str) -> None:
    raise SystemExit(f"PIPELINE_FAILURE: {message}")


def _finite(values: torch.Tensor, label: str) -> torch.Tensor:
    if not bool(torch.isfinite(values).all()):
        fail(f"non-finite {label}")
    return values


def calibration_inputs(split: dict, classical: dict[str, float]) -> dict:
    """Build (x, z, phase_index) tensors for one split.

    x is the classical evaluation in scaled units, z the frozen residual
    target, and phase_index selects the per-phase parameter row.
    """
    pids = split["pids"]
    phases = split.get("phases")
    if phases is None:
        fail("calibration split has no phases")
    if len(phases) != len(pids):
        fail("calibration split phase/pid length mismatch")
    base: list[float] = []
    target: list[float] = []
    index: list[int] = []
    for pid, phase, teacher in zip(pids, phases, split["raw_target_cp"]):
        if pid not in classical:
            fail(f"classical cache missing position {pid}")
        base_cp = float(classical[pid])
        residual_cp = float(teacher) - base_cp
        clipped = max(-probe.CLIP_CP, min(probe.CLIP_CP, residual_cp))
        base.append(base_cp / probe.TARGET_SCALE)
        target.append(clipped / probe.TARGET_SCALE)
        index.append(PHASE_BUCKETS.index(probe.phase_bucket(int(phase))))
    return {
        "x": _finite(torch.tensor(base, dtype=DTYPE), "calibration x"),
        "z": _finite(torch.tensor(target, dtype=DTYPE), "calibration z"),
        "phase_index": torch.tensor(index, dtype=torch.long),
        "n": len(pids),
    }


def _forward(name: str, parameters: dict[str, torch.Tensor],
             data: dict) -> torch.Tensor:
    """Correction in scaled units for one candidate."""
    x = data["x"]
    if name == "identity":
        return torch.zeros_like(x)
    if name == "global_bias":
        return parameters["b"].expand_as(x)
    if name == "global_affine":
        return parameters["u"] * x + parameters["b"]
    index = data["phase_index"]
    if name == "phase_bias":
        return parameters["b"].index_select(0, index)
    if name == "phase_affine":
        return (parameters["u"].index_select(0, index) * x
                + parameters["b"].index_select(0, index))
    fail(f"unknown calibrator {name}")


def _zero_parameters(name: str) -> dict[str, torch.Tensor]:
    """Zero initialization; a zero-init calibrator starts AT identity."""
    if name == "identity":
        return {}
    if name in ("global_bias",):
        return {"b": torch.zeros(1, dtype=DTYPE, requires_grad=True)}
    if name == "global_affine":
        return {"u": torch.zeros(1, dtype=DTYPE, requires_grad=True),
                "b": torch.zeros(1, dtype=DTYPE, requires_grad=True)}
    if name == "phase_bias":
        return {"b": torch.zeros(len(PHASE_BUCKETS), dtype=DTYPE,
                                 requires_grad=True)}
    if name == "phase_affine":
        return {"u": torch.zeros(len(PHASE_BUCKETS), dtype=DTYPE,
                                 requires_grad=True),
                "b": torch.zeros(len(PHASE_BUCKETS), dtype=DTYPE,
                                 requires_grad=True)}
    fail(f"unknown calibrator {name}")


def smooth_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(prediction, target,
                                              beta=LOSS_BETA)


def require_all_phases(data: dict, label: str) -> dict[str, int]:
    """Every phase bucket must be represented, else per-phase fits are blind."""
    counts = {name: int((data["phase_index"] == i).sum())
              for i, name in enumerate(PHASE_BUCKETS)}
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        fail(f"{label} is missing phase bucket(s) {missing}; per-phase "
             f"calibrators cannot be fitted")
    return counts


def fit_calibrators(train: dict) -> dict[str, dict]:
    """Fit every candidate on the TRAIN split only. Deterministic, no RNG.

    Returns name -> {parameters (plain floats), parameter_count, train_loss}.
    """
    require_all_phases(train, "calibration train split")
    fitted: dict[str, dict] = {}
    for name in CANDIDATE_ORDER:
        parameters = _zero_parameters(name)
        if parameters:
            optimizer = torch.optim.LBFGS(
                list(parameters.values()), max_iter=LBFGS_MAX_ITER,
                line_search_fn=LBFGS_LINE_SEARCH)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = smooth_l1(_forward(name, parameters, train),
                                 train["z"])
                loss.backward()
                return loss

            optimizer.step(closure)
        with torch.no_grad():
            for key, tensor in parameters.items():
                if not bool(torch.isfinite(tensor).all()):
                    fail(f"calibrator {name} produced non-finite parameter "
                         f"{key}")
            loss = float(smooth_l1(_forward(name, parameters, train),
                                   train["z"]))
        if not math.isfinite(loss):
            fail(f"calibrator {name} produced non-finite train loss")
        fitted[name] = {
            "name": name,
            "parameter_count": CANDIDATE_PARAMETER_COUNT[name],
            "parameters": {key: [round(v, 10) for v in
                                 tensor.detach().reshape(-1).tolist()]
                           for key, tensor in parameters.items()},
            "phase_order": list(PHASE_BUCKETS),
            "train_smooth_l1": round(loss, 12),
        }
    return fitted


def _tensors_from(calibrator: dict) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(values, dtype=DTYPE)
            for key, values in calibrator["parameters"].items()}


def predict_calibrator(calibrator: dict, split: dict,
                       classical: dict[str, float]) -> list[float]:
    """Calibrated prediction in cp: `base_eval_stm + correction * 1000`."""
    data = calibration_inputs(split, classical)
    with torch.no_grad():
        correction = _forward(calibrator["name"], _tensors_from(calibrator),
                              data)
        _finite(correction, f"{calibrator['name']} correction")
        predicted = (data["x"] + correction) * probe.TARGET_SCALE
    return [float(value) for value in predicted.tolist()]


def correction_cp(calibrator: dict, split: dict,
                  classical: dict[str, float]) -> list[float]:
    """Correction alone, in cp."""
    data = calibration_inputs(split, classical)
    with torch.no_grad():
        correction = _forward(calibrator["name"], _tensors_from(calibrator),
                              data)
    return [float(value) * probe.TARGET_SCALE for value in correction.tolist()]


def candidate_loss(calibrator: dict, data: dict) -> float:
    with torch.no_grad():
        loss = float(smooth_l1(
            _forward(calibrator["name"], _tensors_from(calibrator), data),
            data["z"]))
    if not math.isfinite(loss):
        fail(f"calibrator {calibrator['name']} non-finite selection loss")
    return loss


def validation_select_calibrator(fitted: dict[str, dict],
                                 validation: dict) -> dict:
    """Select on VALIDATION SmoothL1 only; ties go to fewer parameters.

    Candidates are scanned in CANDIDATE_ORDER (ascending parameter count) and a
    later candidate must beat the incumbent by more than TIE_TOLERANCE to
    displace it, so an equally good expensive model never wins.
    """
    require_all_phases(validation, "calibration validation split")
    losses: dict[str, float] = {}
    for name in CANDIDATE_ORDER:
        if name not in fitted:
            fail(f"calibrator {name} was not fitted")
        losses[name] = candidate_loss(fitted[name], validation)
    selected = CANDIDATE_ORDER[0]
    for name in CANDIDATE_ORDER[1:]:
        if losses[name] < losses[selected] - TIE_TOLERANCE:
            selected = name
    return {
        "selected": selected,
        "selection_rule": (
            "minimum N3B validation SmoothL1; ties within "
            f"{TIE_TOLERANCE} resolved toward fewer parameters in order "
            + " -> ".join(CANDIDATE_ORDER)),
        "tie_tolerance": TIE_TOLERANCE,
        "validation_smooth_l1": {name: round(value, 12)
                                 for name, value in losses.items()},
        "parameter_counts": dict(CANDIDATE_PARAMETER_COUNT),
        "selected_parameter_count": CANDIDATE_PARAMETER_COUNT[selected],
        "fitted_on": "N3B train split only",
        "selected_on": "N3B validation split only",
        "holdout_or_confirmation_used": False,
    }
