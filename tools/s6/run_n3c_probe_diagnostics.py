#!/usr/bin/env python3
"""S6-N3C controlled NNUE generalization diagnostics.

This is a CPU-only, measurement-only matrix around the frozen S6-N1 probe.
It does not change the formal N3B checkpoint, dataset, or engine sources.

Controls:
  A  replay the old N1 dataset/checkpoint with the current trainer;
  B  compare old N1 and new N3B on identity-filtered N3B evaluation rows;
  C  mixed-family width scan;
  D  family-isolated training;
  E  classical-evaluation residual targets.

Every trained diagnostic model is saved to a temporary checkpoint, reloaded
from disk, and only then evaluated.  Validation selects early stopping and any
reported diagnostic width; holdout is never used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset as bd  # noqa: E402
import train_nnue_probe as probe  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
SEEDS = (20260818, 20260819, 20260820)
MIXED_WIDTHS = (4, 8, 16, 32)
RESIDUAL_WIDTHS = (8, 16)
EXPECTED_OLD_CHECKPOINT_SHA = (
    "6bfdba6d7d9cc034d55d8bfe433ebb3b0d6f48d78afa2351f3ef465ac9003a66")
OLD_REFERENCE_HOLDOUT_MAE = 141.59
CLEAR_IMPROVEMENT = 0.05
MIN_SEEDS_SAME_DIRECTION = 2
FAR_WORSE = 0.05


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_ids(ids: list[str]) -> str:
    return sha256_bytes("\n".join(sorted(ids)).encode("utf-8"))


def finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"PIPELINE_FAILURE: non-finite {label}")
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return round(ordered[lo], 3)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return round(value, 3)


def target_statistics(records: list[dict], labels: dict[str, dict]) -> dict:
    raw: list[float] = []
    null_cp = 0
    buckets = {"0-100": 0, "100-300": 0, "300-1000": 0, "1000-inf": 0}
    for record in records:
        cp = labels[record["position_id"]].get("teacher_cp_stm")
        if cp is None:
            null_cp += 1
            continue
        value = finite(cp, f"teacher_cp_stm/{record['position_id']}")
        raw.append(value)
        magnitude = abs(value)
        if magnitude < 100:
            buckets["0-100"] += 1
        elif magnitude < 300:
            buckets["100-300"] += 1
        elif magnitude < 1000:
            buckets["300-1000"] += 1
        else:
            buckets["1000-inf"] += 1
    if raw:
        mean = statistics.mean(raw)
        std = statistics.pstdev(raw)
    else:
        mean = std = None
    return {
        "raw_records": len(records),
        "usable_records": len(raw),
        "null_cp_records": null_cp,
        "null_cp_rate": round(null_cp / len(records), 6) if records else 0.0,
        "teacher_target_cp": {
            "mean": round(mean, 3) if mean is not None else None,
            "std": round(std, 3) if std is not None else None,
            "p10": percentile(raw, 0.10),
            "p50": percentile(raw, 0.50),
            "p90": percentile(raw, 0.90),
        },
        "abs_cp_buckets": {
            key: {"n": count,
                  "rate": round(count / len(raw), 6) if raw else 0.0}
            for key, count in buckets.items()
        },
    }


def add_family_metadata(split: dict, family_map: dict[str, str]) -> None:
    try:
        split["source_families"] = [family_map[sid]
                                     for sid in split["source_ids"]]
    except KeyError as exc:
        raise SystemExit(f"PIPELINE_FAILURE: unknown source_id {exc.args[0]}") from exc


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    out = {"train": [], "validation": [], "holdout": []}
    for record in records:
        if record.get("split") not in out:
            raise SystemExit(
                f"PIPELINE_FAILURE: unsupported split {record.get('split')}")
        out[record["split"]].append(record)
    return out


def load_prepared(dataset_dir: Path, engine: Path,
                  family_map: dict[str, str]) -> dict:
    data = probe.load_dataset(dataset_dir)
    records = data["records"]
    splits = split_records(records)
    exported = probe.export_all_features(engine, records)
    prepared = {
        name: probe.prepare_split(exported, rows, data["labels"])
        for name, rows in splits.items()
    }
    for split in prepared.values():
        add_family_metadata(split, family_map)
    return {
        "dir": str(dataset_dir),
        "data": data,
        "records": records,
        "splits": splits,
        "exported": exported,
        "prepared": prepared,
    }


def feature_statistics(exported: dict[str, dict], records: list[dict]) -> dict:
    frequency: Counter[int] = Counter()
    for record in records:
        feature = exported[record["position_id"]]
        frequency.update(int(i) for i in feature["white"])
        frequency.update(int(i) for i in feature["black"])
    values = list(frequency.values())
    return {
        "positions": len(records),
        "union_unique": len(frequency),
        "total_activations": sum(values),
        "singleton_features": sum(v == 1 for v in values),
        "features_with_activation_le5": sum(v <= 5 for v in values),
    }


def group_records(records: list[dict], labels: dict[str, dict],
                  family_map: dict[str, str], kind: str,
                  value: str) -> tuple[list[dict], list[dict]]:
    all_rows = []
    usable = []
    for record in records:
        if kind == "family":
            match = family_map[record["source_id"]] == value
        else:
            match = probe.phase_bucket(int(record.get("phase", 0))) == value
        if not match:
            continue
        all_rows.append(record)
        if labels[record["position_id"]].get("teacher_cp_stm") is not None:
            usable.append(record)
    return all_rows, usable


def grouped_coverage(dataset: dict, family_map: dict[str, str]) -> dict:
    """Report null rates and per-group sparse-feature diagnostics for N3B."""
    records = dataset["records"]
    labels = dataset["data"]["labels"]
    exported = dataset["exported"]
    groups = {
        "family": sorted(set(family_map.values())),
        "phase": ["high", "mid", "low", "zero"],
    }
    result = {}
    for kind, names in groups.items():
        result[kind] = {}
        for name in names:
            all_rows, usable_rows = group_records(
                records, labels, family_map, kind, name)
            train_all, train_usable = group_records(
                dataset["splits"]["train"], labels, family_map, kind, name)
            val_all, val_usable = group_records(
                dataset["splits"]["validation"], labels, family_map, kind, name)
            train_features = feature_statistics(exported, train_usable)
            if train_usable and val_usable:
                coverage = probe.coverage_for_split(
                    exported, val_usable,
                    {int(i) for row in train_usable
                     for i in (exported[row["position_id"]]["white"]
                               + exported[row["position_id"]]["black"])})
                unseen_rate = coverage["unseen_rate"]
            else:
                unseen_rate = None
            result[kind][name] = {
                "raw_records": len(all_rows),
                "usable_records": len(usable_rows),
                "null_cp_records": len(all_rows) - len(usable_rows),
                "null_cp_rate": round(
                    (len(all_rows) - len(usable_rows)) / len(all_rows), 6)
                if all_rows else 0.0,
                "split_counts": {
                    "train": {"raw": len(train_all), "usable": len(train_usable)},
                    "validation": {"raw": len(val_all), "usable": len(val_usable)},
                },
                "train_feature_statistics": train_features,
                "validation_unseen_rate": unseen_rate,
            }
    return result


def make_model(width: int, seed: int) -> probe.NnueProbe:
    if width <= 0:
        raise SystemExit(f"PIPELINE_FAILURE: invalid width {width}")
    torch.manual_seed(seed)
    model = probe.NnueProbe(width=width)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise SystemExit("PIPELINE_FAILURE: diagnostic model is not CPU")
    return model


def save_checkpoint(model: probe.NnueProbe, path: Path, *, width: int,
                    seed: int, dataset_sha: str, target: str) -> str:
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": {"inputs": probe.NNUE_INPUTS, "width": width,
                         "activation": "relu", "head": "width*2->1 linear"},
        "seed": seed,
        "dataset_sha256": dataset_sha,
        "target": target,
        "best_epoch": None,
    }, path)
    return sha256_file(path)


def load_checkpoint(path: Path, width: int, seed: int) -> tuple[probe.NnueProbe, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", {})
    if architecture.get("inputs") != probe.NNUE_INPUTS:
        raise SystemExit(f"PIPELINE_FAILURE: checkpoint inputs mismatch {path}")
    if architecture.get("width") != width:
        raise SystemExit(f"PIPELINE_FAILURE: checkpoint width mismatch {path}")
    model = make_model(width, seed)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def score_model(model: probe.NnueProbe, split: dict,
                classical_cache: dict[str, float] | None = None,
                residual: bool = False) -> dict:
    predictions, loss = probe.evaluate_split(model, split)
    nnue_cp = [float(value) * probe.TARGET_SCALE for value in predictions.tolist()]
    if residual:
        if classical_cache is None:
            raise SystemExit("PIPELINE_FAILURE: residual score has no classical cache")
        final_cp = [classical_cache[pid] + value
                    for pid, value in zip(split["pids"], nnue_cp)]
        model_metrics = probe.clipped_metrics(final_cp, split["raw_target_cp"])
        model_metrics["prediction_mode"] = "classical_plus_residual"
    else:
        model_metrics = probe.clipped_metrics(nnue_cp, split["raw_target_cp"])
        model_metrics["prediction_mode"] = "absolute"
    return {
        "loss": round(loss, 6),
        "metrics": model_metrics,
    }


def train_reload_score(train: dict, validation: dict, holdout: dict,
                       width: int, seed: int, dataset_sha: str,
                       checkpoint_dir: Path, target: str,
                       classical_cache: dict[str, float] | None = None,
                       residual: bool = False) -> dict:
    model = make_model(width, seed)
    training = probe.train_probe(model, train, validation, seed=seed)
    path = checkpoint_dir / f"{target}-w{width}-s{seed}.pt"
    checkpoint_sha = save_checkpoint(
        model, path, width=width, seed=seed,
        dataset_sha=dataset_sha, target=target)
    loaded, checkpoint = load_checkpoint(path, width, seed)
    scored_val = score_model(loaded, validation, classical_cache, residual)
    scored_holdout = score_model(loaded, holdout, classical_cache, residual)
    if abs(scored_val["loss"] - training["best_val_loss"]) > 1e-6:
        raise SystemExit(
            f"PIPELINE_FAILURE: roundtrip validation loss {scored_val['loss']} "
            f"!= best {training['best_val_loss']}")
    return {
        "status": "PASS",
        "width": width,
        "seed": seed,
        "target": target,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": str(path),
        "training": {key: value for key, value in training.items()
                     if key != "train_losses"},
        "roundtrip_validation_loss": scored_val["loss"],
        "validation": scored_val,
        "holdout": scored_holdout,
        "model": loaded,
        "checkpoint": checkpoint,
    }


def public_run(run: dict) -> dict:
    return {key: value for key, value in run.items()
            if key not in {"model", "checkpoint"}}


def state_comparison(left: dict, right: dict) -> dict:
    keys = sorted(set(left) | set(right))
    tensors = {}
    exact = True
    max_abs = 0.0
    for key in keys:
        if key not in left or key not in right:
            tensors[key] = {"present_both": False}
            exact = False
            continue
        a = left[key].detach().cpu()
        b = right[key].detach().cpu()
        if a.shape != b.shape:
            tensors[key] = {"present_both": True, "shape_equal": False}
            exact = False
            continue
        delta = float((a - b).abs().max().item()) if a.numel() else 0.0
        tensors[key] = {
            "present_both": True,
            "shape_equal": True,
            "exact_equal": bool(torch.equal(a, b)),
            "max_abs_delta": delta,
        }
        exact = exact and torch.equal(a, b)
        max_abs = max(max_abs, delta)
    return {"exact_equal": exact, "max_abs_delta": max_abs, "tensors": tensors}


def metrics_delta(left: dict, right: dict) -> float:
    return round(right["metrics"]["clipped_mae_cp"]
                 - left["metrics"]["clipped_mae_cp"], 3)


def control_a(n1: dict, old_checkpoint: Path, checkpoint_dir: Path) -> dict:
    old_sha = sha256_file(old_checkpoint)
    if old_sha != EXPECTED_OLD_CHECKPOINT_SHA:
        raise SystemExit(
            f"PIPELINE_FAILURE: old checkpoint SHA {old_sha} != "
            f"{EXPECTED_OLD_CHECKPOINT_SHA}")
    old_model, old_ckpt = load_checkpoint(old_checkpoint, probe.WIDTH, probe.SEED)
    replay = train_reload_score(
        n1["prepared"]["train"], n1["prepared"]["validation"],
        n1["prepared"]["holdout"], probe.WIDTH, probe.SEED,
        n1["data"]["dataset_sha"], checkpoint_dir, "control-a-replay")
    old_val = score_model(old_model, n1["prepared"]["validation"])
    old_holdout = score_model(old_model, n1["prepared"]["holdout"])
    replay_val = replay["validation"]
    replay_holdout = replay["holdout"]
    holdout_delta = metrics_delta(old_holdout, replay_holdout)
    def prediction_delta(split: dict) -> float:
        left, _ = probe.evaluate_split(old_model, split)
        right, _ = probe.evaluate_split(replay["model"], split)
        delta = (left - right).abs() * probe.TARGET_SCALE
        return round(float(delta.max().item()), 6) if len(delta) else 0.0
    return {
        "status": "PASS" if abs(holdout_delta) <= 0.1 else "FAIL",
        "gate": {"holdout_mae_abs_delta_le_0.1_cp": abs(holdout_delta) <= 0.1},
        "old_checkpoint_sha256": old_sha,
        "old_checkpoint_metadata": {
            key: value for key, value in old_ckpt.items() if key != "state_dict"
        },
        "reference_old_holdout_clipped_mae_cp": OLD_REFERENCE_HOLDOUT_MAE,
        "old_checkpoint": {
            "validation": old_val,
            "holdout": old_holdout,
        },
        "replay": public_run(replay),
        "replay_vs_old": {
            "validation_mae_delta_cp": metrics_delta(old_val, replay_val),
            "holdout_mae_delta_cp": holdout_delta,
            "validation_prediction_max_abs_delta_cp": prediction_delta(
                n1["prepared"]["validation"]),
            "holdout_prediction_max_abs_delta_cp": prediction_delta(
                n1["prepared"]["holdout"]),
        },
        "state_tensors": state_comparison(
            old_model.state_dict(),
            replay["model"].state_dict()),
    }


def identity_filter(eval_records: list[dict], train_sets: dict[str, list[dict]],
                    label: str) -> tuple[list[dict], dict]:
    train_pids = {name: {row["position_id"] for row in rows}
                  for name, rows in train_sets.items()}
    train_games = {name: {row["source_game_id"] for row in rows}
                   for name, rows in train_sets.items()}
    eval_pids = {row["position_id"] for row in eval_records}
    eval_games = {row["source_game_id"] for row in eval_records}
    game_overlaps = {
        name: len(eval_games & games) for name, games in train_games.items()}
    if any(game_overlaps.values()):
        raise SystemExit(
            f"PIPELINE_FAILURE: {label} source_game_id overlap {game_overlaps}")
    excluded = sorted(eval_pids & set().union(*train_pids.values()))
    filtered = [row for row in eval_records
                if row["position_id"] not in set(excluded)]
    return filtered, {
        "raw_eval_positions": len(eval_records),
        "excluded_positions": len(excluded),
        "eligible_positions": len(filtered),
        "excluded_ids_sha256": sha256_ids(excluded),
        "train_position_counts": {name: len(ids)
                                   for name, ids in train_pids.items()},
        "train_source_game_counts": {name: len(ids)
                                      for name, ids in train_games.items()},
        "source_game_id_overlap": game_overlaps,
        "selection_uses_labels_or_predictions": False,
        "policy": "exclude-position-id-overlap-before-inference",
    }


def predictor_table(split: dict, old_model: probe.NnueProbe,
                    new_model: probe.NnueProbe,
                    classical_cache: dict[str, float]) -> dict:
    old_pred, _ = probe.evaluate_split(old_model, split)
    new_pred, _ = probe.evaluate_split(new_model, split)
    old_cp = [float(value) * probe.TARGET_SCALE for value in old_pred.tolist()]
    new_cp = [float(value) * probe.TARGET_SCALE for value in new_pred.tolist()]
    classical_cp = [classical_cache[pid] for pid in split["pids"]]
    return {
        "n": len(split["pids"]),
        "old_n1": probe.clipped_metrics(old_cp, split["raw_target_cp"]),
        "new_n3b": probe.clipped_metrics(new_cp, split["raw_target_cp"]),
        "classical": probe.clipped_metrics(classical_cp, split["raw_target_cp"]),
    }


def group_tables(split: dict, old_model: probe.NnueProbe,
                 new_model: probe.NnueProbe,
                 classical_cache: dict[str, float]) -> dict:
    result = {}
    for kind, names in {
        "family": sorted(set(split.get("source_families", []))),
        "phase": ["high", "mid", "low", "zero"],
    }.items():
        result[kind] = {}
        for name in names:
            mask = [(family == name if kind == "family"
                     else probe.phase_bucket(phase) == name)
                    for family, phase in zip(
                        split.get("source_families", []), split["phases"])]
            sub = probe.slice_rows(split, mask)
            if not sub["pids"]:
                continue
            result[kind][name] = predictor_table(
                sub, old_model, new_model, classical_cache)
    return result


def control_b(current: dict, n1: dict, n3b_checkpoint: Path,
              current_classical: dict[str, float]) -> dict:
    old_model, _ = load_checkpoint(
        Path("data/s6/models/s6-n1-probe.pt"), probe.WIDTH, probe.SEED)
    new_model, new_ckpt = load_checkpoint(n3b_checkpoint, probe.WIDTH, probe.SEED)
    train_sets = {
        "n1": n1["splits"]["train"],
        "n3b": current["splits"]["train"],
    }
    result = {
        "status": "PASS",
        "n3b_checkpoint_sha256": sha256_file(n3b_checkpoint),
        "n3b_checkpoint_metadata": {
            key: value for key, value in new_ckpt.items() if key != "state_dict"
        },
        "splits": {},
    }
    for name in ("validation", "holdout"):
        rows, audit = identity_filter(
            current["splits"][name], train_sets, f"control-b/{name}")
        split = probe.prepare_split(
            current["exported"], rows, current["data"]["labels"])
        add_family_metadata(split, current["family_map"])
        result["splits"][name] = {
            "identity_audit": audit,
            "predictors": predictor_table(
                split, old_model, new_model, current_classical),
            "by_group": group_tables(
                split, old_model, new_model, current_classical),
        }
    return result


def aggregate_runs(runs: list[dict], split: str = "validation") -> dict:
    good = [run for run in runs if run.get("status") == "PASS"]
    values = [run[split]["metrics"]["clipped_mae_cp"] for run in good]
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "std": round(statistics.pstdev(values), 3),
        "median": round(statistics.median(values), 3),
    }


def run_config(fn, *args, **kwargs) -> dict:
    try:
        return fn(*args, **kwargs)
    except SystemExit as exc:
        return {"status": "FAIL", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def control_c(current: dict, checkpoint_dir: Path) -> dict:
    runs = []
    for width in MIXED_WIDTHS:
        for seed in SEEDS:
            runs.append(run_config(
                train_reload_score,
                current["prepared"]["train"], current["prepared"]["validation"],
                current["prepared"]["holdout"], width, seed,
                current["data"]["dataset_sha"], checkpoint_dir, "control-c-mixed"))
    summaries = {str(width): {
        "validation": aggregate_runs(
            [run for run in runs if run.get("width") == width]),
        "holdout": aggregate_runs(
            [run for run in runs if run.get("width") == width], "holdout"),
    } for width in MIXED_WIDTHS}
    width32 = summaries["32"]["validation"]["mean"]
    selected_width = None
    if width32 is not None:
        valid = [(int(width), data["validation"]["mean"])
                 for width, data in summaries.items()
                 if data["validation"]["mean"] is not None]
        if valid:
            selected_width = min(valid, key=lambda item: item[1])[0]
    narrower = {}
    for width in MIXED_WIDTHS:
        if width == 32 or width32 is None:
            continue
        width_runs = {run["seed"]: run for run in runs
                      if run.get("width") == width and run.get("status") == "PASS"}
        base_runs = {run["seed"]: run for run in runs
                     if run.get("width") == 32 and run.get("status") == "PASS"}
        same_direction = sum(
            width_runs[seed]["validation"]["metrics"]["clipped_mae_cp"]
            < base_runs[seed]["validation"]["metrics"]["clipped_mae_cp"]
            for seed in SEEDS if seed in width_runs and seed in base_runs)
        mean = summaries[str(width)]["validation"]["mean"]
        narrower[str(width)] = {
            "mean_improvement_fraction": round((width32 - mean) / width32, 6)
            if mean is not None else None,
            "at_least_5_percent_better":
                mean is not None and mean <= width32 * (1 - CLEAR_IMPROVEMENT),
            "seeds_better_than_width32": same_direction,
            "at_least_2_of_3_seeds_better":
                same_direction >= MIN_SEEDS_SAME_DIRECTION,
        }
    return {
        "status": "PASS" if all(run.get("status") == "PASS" for run in runs)
        else "FAIL",
        "runs": [public_run(run) for run in runs],
        "summaries_by_width": summaries,
        "validation_selected_width": selected_width,
        "selection_rule": "minimum three-seed validation MAE mean only",
        "narrower_vs_width32": narrower,
    }


def family_rows(dataset: dict, family: str) -> dict[str, list[dict]]:
    return {
        name: [row for row in dataset["splits"][name]
               if dataset["family_map"][row["source_id"]] == family]
        for name in ("train", "validation", "holdout")
    }


def score_single_model(model: probe.NnueProbe, split: dict,
                       classical_cache: dict[str, float]) -> dict:
    pred, loss = probe.evaluate_split(model, split)
    cp = [float(value) * probe.TARGET_SCALE for value in pred.tolist()]
    classical = [classical_cache[pid] for pid in split["pids"]]
    return {
        "loss": round(loss, 6),
        "nnue": probe.clipped_metrics(cp, split["raw_target_cp"]),
        "classical": probe.clipped_metrics(classical, split["raw_target_cp"]),
    }


def control_d(current: dict, n1: dict, checkpoint_dir: Path,
              current_classical: dict[str, float],
              legacy_classical: dict[str, float]) -> dict:
    mixed = {}
    for run in current["control_c"]["runs"]:
        if run.get("status") != "PASS" or run.get("width") != 32:
            continue
        mixed[run["seed"]] = run
    output = {"status": "PASS", "families": {}}
    for family in ("arena", "lichess-standard-rated-v1"):
        rows = family_rows(current, family)
        prepared = {
            name: probe.prepare_split(
                current["exported"], rows[name], current["data"]["labels"])
            for name in rows
        }
        for split in prepared.values():
            add_family_metadata(split, current["family_map"])
        legacy_rows = n1["splits"]["holdout"]
        filtered_legacy, legacy_audit = identity_filter(
            legacy_rows, {"family_train": rows["train"]},
            f"control-d/{family}/legacy")
        legacy_split = probe.prepare_split(
            n1["exported"], filtered_legacy, n1["data"]["labels"])
        add_family_metadata(legacy_split, n1["family_map"])
        family_runs = []
        for seed in SEEDS:
            run = run_config(
                train_reload_score, prepared["train"], prepared["validation"],
                prepared["holdout"], probe.WIDTH, seed,
                current["data"]["dataset_sha"], checkpoint_dir,
                f"control-d-{family}")
            if run.get("status") == "PASS":
                run["legacy"] = score_single_model(
                    run["model"], legacy_split, legacy_classical)
                mixed_run = mixed.get(seed)
                if mixed_run is not None:
                    mixed_model = load_checkpoint(
                        Path(mixed_run["checkpoint_path"]), probe.WIDTH, seed)[0]
                    run["mixed_width32_same_seed"] = {
                        "holdout": score_single_model(
                            mixed_model, prepared["holdout"], current_classical),
                        "legacy": score_single_model(
                            mixed_model, legacy_split, legacy_classical),
                    }
                    run["family_minus_mixed_holdout_mae_delta_cp"] = round(
                        run["holdout"]["metrics"]["clipped_mae_cp"]
                        - run["mixed_width32_same_seed"]["holdout"]["nnue"]
                        ["clipped_mae_cp"], 3)
            family_runs.append(run)
        output["families"][family] = {
            "identity_filtered_legacy": legacy_audit,
            "runs": [public_run(run) for run in family_runs],
            "validation": aggregate_runs(family_runs),
            "holdout": aggregate_runs(family_runs, "holdout"),
            "legacy": {
                "mean": round(statistics.mean([
                    run["legacy"]["nnue"]["clipped_mae_cp"]
                    for run in family_runs if run.get("status") == "PASS"]), 3)
                if any(run.get("status") == "PASS" for run in family_runs)
                else None,
            },
        }
        if any(run.get("status") != "PASS" for run in family_runs):
            output["status"] = "FAIL"
    return output


def build_classical_cache(dataset: dict, engine: Path, cache_path: Path,
                          engine_sha: str) -> dict:
    usable = [row for row in dataset["records"]
              if dataset["data"]["labels"][row["position_id"]]
              .get("teacher_cp_stm") is not None]
    ids = [row["position_id"] for row in usable]
    if len(set(ids)) != len(ids):
        raise SystemExit("PIPELINE_FAILURE: duplicate usable position_id for cache")
    values = {}
    for index, row in enumerate(usable, 1):
        value = finite(probe.classical_eval_stm(engine, row["fen"]),
                       f"base_eval_stm/{row['position_id']}")
        if row["position_id"] in values:
            raise SystemExit("PIPELINE_FAILURE: duplicate cache position_id")
        values[row["position_id"]] = value
        if index % 1000 == 0:
            print(f"classical cache: {index}/{len(usable)}", flush=True)
    payload = {
        "header": {
            "schema": 1,
            "dataset_sha256": dataset["data"]["dataset_sha"],
            "engine_binary_sha256": engine_sha,
            "target": "base_eval_stm",
            "usable_position_count": len(usable),
        },
        "values": values,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                          encoding="utf-8")
    return validate_classical_cache(cache_path, dataset, engine_sha)


def validate_classical_cache(cache_path: Path, dataset: dict,
                             engine_sha: str) -> dict:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    header = payload.get("header", {})
    if header.get("dataset_sha256") != dataset["data"]["dataset_sha"]:
        raise SystemExit("PIPELINE_FAILURE: classical cache dataset SHA mismatch")
    if header.get("engine_binary_sha256") != engine_sha:
        raise SystemExit("PIPELINE_FAILURE: classical cache engine SHA mismatch")
    usable_ids = {
        row["position_id"] for row in dataset["records"]
        if dataset["data"]["labels"][row["position_id"]]
        .get("teacher_cp_stm") is not None
    }
    values = payload.get("values", {})
    if set(values) != usable_ids:
        raise SystemExit("PIPELINE_FAILURE: classical cache position-id set mismatch")
    for pid, value in values.items():
        finite(value, f"classical_cache/{pid}")
    return {
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "header": header,
        "values": {pid: float(value) for pid, value in values.items()},
    }


def residual_split(prepared: dict, classical_cache: dict[str, float]) -> dict:
    out = dict(prepared)
    residual_targets = []
    for pid, teacher in zip(prepared["pids"], prepared["raw_target_cp"]):
        base = finite(classical_cache[pid], f"classical_cache/{pid}")
        residual = finite(teacher - base, f"residual/{pid}")
        residual_targets.append(
            max(-probe.CLIP_CP, min(probe.CLIP_CP, residual))
            / probe.TARGET_SCALE)
    out["target"] = torch.tensor(residual_targets, dtype=torch.float32)
    out["residual_raw_target_cp"] = [
        finite(teacher - classical_cache[pid], f"residual/{pid}")
        for pid, teacher in zip(prepared["pids"], prepared["raw_target_cp"])]
    return out


def control_e(current: dict, checkpoint_dir: Path,
              current_classical: dict[str, float]) -> dict:
    residual_prepared = {
        name: residual_split(current["prepared"][name], current_classical)
        for name in ("train", "validation", "holdout")
    }
    runs = []
    for width in RESIDUAL_WIDTHS:
        for seed in SEEDS:
            runs.append(run_config(
                train_reload_score, residual_prepared["train"],
                residual_prepared["validation"], residual_prepared["holdout"],
                width, seed, current["data"]["dataset_sha"], checkpoint_dir,
                "control-e-residual", current_classical, True))
    for run in runs:
        if run.get("status") == "PASS":
            run["classical_validation"] = probe.clipped_metrics(
                [current_classical[pid]
                 for pid in current["prepared"]["validation"]["pids"]],
                current["prepared"]["validation"]["raw_target_cp"])
            run["classical_holdout"] = probe.clipped_metrics(
                [current_classical[pid]
                 for pid in current["prepared"]["holdout"]["pids"]],
                current["prepared"]["holdout"]["raw_target_cp"])
    summaries = {str(width): {
        "validation": aggregate_runs(
            [run for run in runs if run.get("width") == width]),
        "holdout": aggregate_runs(
            [run for run in runs if run.get("width") == width], "holdout"),
        "classical_validation_mae_cp": round(statistics.mean([
            run["classical_validation"]["clipped_mae_cp"]
            for run in runs if run.get("width") == width
            and run.get("status") == "PASS"]), 3)
        if any(run.get("width") == width and run.get("status") == "PASS"
               for run in runs) else None,
        "classical_holdout_mae_cp": round(statistics.mean([
            run["classical_holdout"]["clipped_mae_cp"]
            for run in runs if run.get("width") == width
            and run.get("status") == "PASS"]), 3)
        if any(run.get("width") == width and run.get("status") == "PASS"
               for run in runs) else None,
    } for width in RESIDUAL_WIDTHS}
    selected_width = min(
        (width for width in RESIDUAL_WIDTHS
         if summaries[str(width)]["validation"]["mean"] is not None),
        key=lambda width: summaries[str(width)]["validation"]["mean"],
        default=None)
    return {
        "status": "PASS" if all(run.get("status") == "PASS" for run in runs)
        else "FAIL",
        "runs": [public_run(run) for run in runs],
        "summaries_by_width": summaries,
        "validation_selected_width": selected_width,
        "selection_rule": "minimum three-seed validation residual MAE mean only",
    }


def mean_predictor_metric(control: dict, split: str,
                          predictor: str) -> float | None:
    values = []
    for run in control.get("runs", []):
        if run.get("status") == "PASS":
            values.append(run[split]["metrics"]["clipped_mae_cp"])
    return round(statistics.mean(values), 3) if values else None


def clear_improvement(values: list[tuple[float, float]]) -> dict:
    if not values:
        return {"mean_improvement_fraction": None,
                "at_least_5_percent_better": False,
                "seeds_better": 0,
                "at_least_2_of_3_seeds_better": False}
    improved = sum(new < mixed for new, mixed in values)
    mean_new = statistics.mean(new for new, _ in values)
    mean_mixed = statistics.mean(mixed for _, mixed in values)
    fraction = (mean_mixed - mean_new) / mean_mixed if mean_mixed else None
    return {
        "mean_improvement_fraction": round(fraction, 6)
        if fraction is not None else None,
        "at_least_5_percent_better":
            fraction is not None and fraction >= CLEAR_IMPROVEMENT,
        "seeds_better": improved,
        "at_least_2_of_3_seeds_better": improved >= MIN_SEEDS_SAME_DIRECTION,
    }


def interpretation(result: dict) -> dict:
    a = result.get("control_a", {})
    signals = {
        "TRAINER_REGRESSION": a.get("status") == "FAIL",
        "OVERPARAMETERIZED_SPARSE_TABLE": False,
        "FAMILY_INTERFERENCE": False,
        "DISTRIBUTION_REPRESENTATION_GAP": False,
        "RESIDUAL_PATH_PROMISING": False,
        "CURRENT_NNUE_REPRESENTATION_NOT_VIABLE": False,
    }
    c = result.get("control_c", {})
    for data in c.get("narrower_vs_width32", {}).values():
        if (data.get("at_least_5_percent_better")
                and data.get("at_least_2_of_3_seeds_better")):
            signals["OVERPARAMETERIZED_SPARSE_TABLE"] = True
    d = result.get("control_d", {})
    family_improvements = []
    for family_data in d.get("families", {}).values():
        values = []
        for run in family_data.get("runs", []):
            mixed = run.get("mixed_width32_same_seed", {})
            if run.get("status") == "PASS" and mixed:
                values.append((
                    run["holdout"]["nnue"]["clipped_mae_cp"],
                    mixed["holdout"]["nnue"]["clipped_mae_cp"]))
        family_improvements.append(clear_improvement(values))
    signals["FAMILY_INTERFERENCE"] = any(
        item["at_least_5_percent_better"]
        and item["at_least_2_of_3_seeds_better"] for item in family_improvements)
    b = result.get("control_b", {})
    gap_checks = []
    for split_data in b.get("splits", {}).values():
        metrics = split_data.get("predictors", {})
        classical = metrics.get("classical", {}).get("clipped_mae_cp")
        if classical is None:
            continue
        gap_checks.extend([
            metrics.get("old_n1", {}).get("clipped_mae_cp")
            >= classical * (1 + FAR_WORSE),
            metrics.get("new_n3b", {}).get("clipped_mae_cp")
            >= classical * (1 + FAR_WORSE),
        ])
    signals["DISTRIBUTION_REPRESENTATION_GAP"] = bool(gap_checks) and all(gap_checks)
    e = result.get("control_e", {})
    selected = e.get("validation_selected_width")
    if selected is not None:
        summary = e.get("summaries_by_width", {}).get(str(selected), {})
        residual_val = summary.get("validation", {}).get("mean")
        residual_hold = summary.get("holdout", {}).get("mean")
        classical_val = summary.get("classical_validation_mae_cp")
        classical_hold = summary.get("classical_holdout_mae_cp")
        signals["RESIDUAL_PATH_PROMISING"] = (
            None not in (residual_val, residual_hold, classical_val, classical_hold)
            and residual_val < classical_val and residual_hold < classical_hold)
    absolute_failures = []
    if e.get("status") == "PASS" and selected is not None:
        summary = e["summaries_by_width"][str(selected)]
        absolute_failures = [
            summary["validation"]["mean"]
            >= summary["classical_validation_mae_cp"],
            summary["holdout"]["mean"]
            >= summary["classical_holdout_mae_cp"],
        ]
    signals["CURRENT_NNUE_REPRESENTATION_NOT_VIABLE"] = bool(
        absolute_failures) and all(absolute_failures) and not signals[
            "RESIDUAL_PATH_PROMISING"]
    return {
        "fixed_thresholds": {
            "clear_improvement_fraction": CLEAR_IMPROVEMENT,
            "min_seeds_same_direction": MIN_SEEDS_SAME_DIRECTION,
            "distribution_gap_worse_fraction": FAR_WORSE,
        },
        "signals": signals,
        "family_improvement_checks": family_improvements,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# S6-N3C - NNUE Generalization Diagnostics", "",
        f"STATUS: **{result['status']}**", "",
        f"VERDICT: **{result['verdict']}**", "",
        "## Provenance", "", "```text",
        f"run git: {result['provenance']['run_git_sha']}",
        f"diagnostics script sha256: {result['provenance']['diagnostics_script_sha256']}",
        f"N3B checkpoint sha256: {result['provenance']['n3b_checkpoint_sha256']}",
        "```", "", "## Dataset Statistics", "",
        "| dataset | raw | usable | null CP | target mean | target std | p10 | p50 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in result["dataset_statistics"].items():
        target = data["teacher_target_cp"]
        lines.append(
            f"| {name} | {data['raw_records']} | {data['usable_records']} | "
            f"{data['null_cp_records']} | {target['mean']} | {target['std']} | "
            f"{target['p10']} | {target['p50']} | {target['p90']} |")
    lines += ["", "## Controls", "", "| control | status | detail |", "|---|---|---|"]
    for name in ("control_a", "control_b", "control_c", "control_d", "control_e"):
        control = result.get(name, {"status": "NOT_RUN"})
        detail = ""
        if name == "control_a":
            detail = str(control.get("replay_vs_old", {}).get(
                "holdout_mae_delta_cp", control.get("error", "")))
        elif name == "control_c":
            detail = f"selected width={control.get('validation_selected_width')}"
        elif name == "control_e":
            detail = f"selected width={control.get('validation_selected_width')}"
        elif "error" in control:
            detail = control["error"]
        lines.append(f"| {name[-1].upper()} | {control.get('status')} | {detail} |")
    lines += ["", "### Control C Runs", "",
              "| width | seed | status | validation MAE | holdout MAE |", 
              "|---:|---:|---|---:|---:|"]
    for run in result.get("control_c", {}).get("runs", []):
        lines.append(
            f"| {run.get('width', '-')} | {run.get('seed', '-')} | {run.get('status')} | "
            f"{run.get('validation', {}).get('metrics', {}).get('clipped_mae_cp', '-')} | "
            f"{run.get('holdout', {}).get('metrics', {}).get('clipped_mae_cp', '-')} |")
    lines += ["", "### Control D Runs", "",
              "| family | seed | status | validation MAE | holdout MAE | legacy MAE |", 
              "|---|---:|---|---:|---:|---:|"]
    for family, data in result.get("control_d", {}).get("families", {}).items():
        for run in data.get("runs", []):
            lines.append(
                f"| {family} | {run.get('seed', '-')} | {run.get('status')} | "
                f"{run.get('validation', {}).get('metrics', {}).get('clipped_mae_cp', '-')} | "
                f"{run.get('holdout', {}).get('metrics', {}).get('clipped_mae_cp', '-')} | "
                f"{run.get('legacy', {}).get('nnue', {}).get('clipped_mae_cp', '-')} |")
    lines += ["", "### Control E Runs", "",
              "| width | seed | status | residual validation MAE | residual holdout MAE |", 
              "|---:|---:|---|---:|---:|"]
    for run in result.get("control_e", {}).get("runs", []):
        lines.append(
            f"| {run.get('width', '-')} | {run.get('seed', '-')} | {run.get('status')} | "
            f"{run.get('validation', {}).get('metrics', {}).get('clipped_mae_cp', '-')} | "
            f"{run.get('holdout', {}).get('metrics', {}).get('clipped_mae_cp', '-')} |")
    lines += ["", "## Interpretation Signals", "", "| signal | value |", "|---|---|"]
    for signal, value in result["interpretation"]["signals"].items():
        lines.append(f"| {signal} | {value} |")
    lines += ["", "All configurations are retained in the JSON record; no holdout-based model selection was performed.", ""]
    return "\n".join(lines)


def diagnostics_script_provenance(repo: Path) -> dict:
    clean = probe.bind_run_provenance(repo)
    path = Path(__file__).resolve()
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{path.relative_to(repo)}"],
        capture_output=True)
    if proc.returncode != 0 or path.read_bytes() != proc.stdout:
        raise SystemExit("PIPELINE_FAILURE: disk diagnostics script differs from HEAD blob")
    return {
        **clean,
        "diagnostics_script_sha256": sha256_file(path),
        "committed_diagnostics_blob_sha256": sha256_bytes(proc.stdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--legacy-dataset", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--legacy-checkpoint", type=Path,
                        default=Path("data/s6/models/s6-n1-probe.pt"))
    parser.add_argument("--n3b-checkpoint", type=Path,
                        default=Path("data/s6/models/s6-n3b-multisource-probe.pt"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/s6/s6-n3c-generalization-diagnostics.json"))
    parser.add_argument("--classical-cache", type=Path,
                        default=Path("results/s6/s6-n3c-classical-cache.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    provenance = diagnostics_script_provenance(repo)
    engine_sha = sha256_file(args.engine)
    family_map, source_manifest_shas = probe.build_family_map(
        [str(path) for path in args.sources])
    current = load_prepared(args.dataset, args.engine, family_map)
    current["family_map"] = family_map
    current["source_manifest_shas"] = source_manifest_shas
    # N3B exact family gate is repeated before diagnostics consume the data.
    probe.attach_source_families(current["prepared"], family_map)
    n1 = load_prepared(args.legacy_dataset, args.engine, family_map)
    n1["family_map"] = family_map
    n1["source_manifest_shas"] = source_manifest_shas

    result = {
        "status": "RUNNING",
        "verdict": "CLOUD_VERDICT_PENDING",
        "provenance": {
            **provenance,
            "engine_binary_sha256": engine_sha,
            "n3b_checkpoint_sha256": sha256_file(args.n3b_checkpoint),
            "n3b_checkpoint_path": str(args.n3b_checkpoint),
            "source_id_to_family": family_map,
            "source_manifests": source_manifest_shas,
        },
        "config": {
            "cpu_only": True,
            "cuda_used": False,
            "seeds": list(SEEDS),
            "mixed_widths": list(MIXED_WIDTHS),
            "residual_widths": list(RESIDUAL_WIDTHS),
            "optimizer": "AdamW",
            "lr": probe.LR,
            "weight_decay": probe.WEIGHT_DECAY,
            "batch_size": probe.BATCH_SIZE,
            "max_epochs": probe.MAX_EPOCHS,
            "patience": probe.PATIENCE,
            "loss": "SmoothL1",
            "loss_beta": probe.LOSS_BETA,
            "target_clip_cp": probe.CLIP_CP,
            "target_scale": probe.TARGET_SCALE,
            "clear_improvement_fraction": CLEAR_IMPROVEMENT,
            "min_seeds_same_direction": MIN_SEEDS_SAME_DIRECTION,
        },
        "dataset_statistics": {
            "N1": target_statistics(n1["records"], n1["data"]["labels"]),
            "N3B": target_statistics(current["records"], current["data"]["labels"]),
        },
        "n3b_group_diagnostics": grouped_coverage(current, family_map),
        "controls": ["A", "B", "C", "D", "E"],
    }
    with tempfile.TemporaryDirectory(prefix="s6-n3c-checkpoints-") as temp:
        checkpoint_dir = Path(temp)
        result["control_a"] = run_config(
            control_a, n1, args.legacy_checkpoint, checkpoint_dir)
        if result["control_a"].get("status") != "PASS":
            result["status"] = "CONTROL_A_FAIL_STOPPED"
            result["stop_reason"] = "TRAINER_REGRESSION"
        else:
            # Control E owns the persistent cache; B and D use the same values.
            cache = build_classical_cache(
                current, args.engine, args.classical_cache, engine_sha)
            current_classical = cache["values"]
            result["classical_cache"] = {
                key: value for key, value in cache.items() if key != "values"
            }
            legacy_cache_path = checkpoint_dir / "legacy-classical-cache.json"
            legacy_cache = build_classical_cache(
                n1, args.engine, legacy_cache_path, engine_sha)
            legacy_classical = legacy_cache["values"]
            result["control_b"] = run_config(
                control_b, current, n1, args.n3b_checkpoint, current_classical)
            result["control_c"] = control_c(current, checkpoint_dir)
            current["control_c"] = result["control_c"]
            result["control_d"] = run_config(
                control_d, current, n1, checkpoint_dir,
                current_classical, legacy_classical)
            result["control_e"] = control_e(
                current, checkpoint_dir, current_classical)
            result["status"] = "DIAGNOSTICS_COMPLETE"
    result["interpretation"] = interpretation(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Remove internal paths/models before writing; all model-derived values are
    # already present in public_run records.
    result["outcome"] = {
        "all_configurations_reported": True,
        "holdout_used_for_selection": False,
        "n3b_checkpoint_modified": False,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_markdown(result) + "\n", encoding="utf-8")
    print(f"results written to {args.out}", flush=True)
    print(f"markdown written to {md_path}", flush=True)
    print(f"verdict: {result['verdict']}", flush=True)
    return 0 if result["status"] == "DIAGNOSTICS_COMPLETE" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
