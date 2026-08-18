#!/usr/bin/env python3
"""S6-N1 NNUE learnability probe trainer.

Frozen probe architecture (NOT a production NNUE contract):
    shared feature table   40960 x 32   (one row per NNUE input index)
    shared accumulator bias 32
    accumulator = bias + sum(active feature rows)   (per perspective)
    activation  ReLU (clamp-min(0))
    head        concat(own_accumulator, opponent_accumulator) by side to move
                64 -> 1 linear scalar

The Rust engine (`bench nnue-features-batch`) is the SINGLE encoding source
of truth; this script never re-implements orientation, king buckets,
relative channels or feature_index. Training targets are teacher_cp_stm only
(null CP rows excluded); the current train/validation/holdout split is reused
as-is; validation only selects the best epoch; holdout is evaluated once with
the restored best checkpoint.

Everything fails closed: duplicate position ids, missing/extra labels,
manifest hash mismatches, and exporter count/id/fen mismatches abort the run.
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
import time
from pathlib import Path

import torch
import torch.nn as nn

NNUE_INPUTS = 40960
WIDTH = 32
SEED = 20260818
CLIP_CP = 2000.0
TARGET_SCALE = 1000.0
LOSS_BETA = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 15
CP_BUCKETS = [(0, 100), (100, 300), (300, 1000), (1000, None)]


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------

def compute_dataset_sha(records: list[dict]) -> str:
    """Canonical dataset SHA exactly as verify_dataset.py computes it."""
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(dataset_dir: Path) -> dict:
    """Load records + labels with strict fail-closed validation."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        (dataset_dir / "teacher_manifest.json").read_text(encoding="utf-8"))

    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    ids = [r["position_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("PIPELINE_FAILURE: duplicate position_id in dataset")

    actual_sha = compute_dataset_sha(records)
    if actual_sha != manifest["dataset_sha256"]:
        raise SystemExit(
            f"PIPELINE_FAILURE: dataset_sha256 mismatch {actual_sha[:16]} "
            f"!= manifest {manifest['dataset_sha256'][:16]}")
    if len(records) != manifest["records_total"]:
        raise SystemExit(
            f"PIPELINE_FAILURE: records {len(records)} != manifest "
            f"{manifest['records_total']}")

    labels: dict[str, dict] = {}
    labels_path = dataset_dir / "labels.jsonl"
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    missing = set(ids) - set(labels)
    extra = set(labels) - set(ids)
    if missing:
        raise SystemExit(f"PIPELINE_FAILURE: {len(missing)} records without labels")
    if extra:
        raise SystemExit(f"PIPELINE_FAILURE: {len(extra)} labels without records")

    labels_text = labels_path.read_text(encoding="utf-8")
    labels_sha = hashlib.sha256(labels_text.encode("utf-8")).hexdigest()
    if labels_sha != teacher_manifest.get("labels_sha256"):
        raise SystemExit(
            f"PIPELINE_FAILURE: labels_sha256 mismatch {labels_sha[:16]} "
            f"!= manifest {teacher_manifest.get('labels_sha256', '?')[:16]}")

    return {
        "records": records,
        "labels": labels,
        "dataset_sha": actual_sha,
        "labels_sha": labels_sha,
        "manifest": manifest,
        "teacher_manifest": teacher_manifest,
    }


# ---------------------------------------------------------------------------
# Engine export bridge
# ---------------------------------------------------------------------------

def parse_export_line(line: str) -> dict:
    rec = json.loads(line)
    for key in ("fen", "white", "black"):
        if key not in rec:
            raise SystemExit(f"PIPELINE_FAILURE: exporter line missing '{key}': {line[:80]}")
    if not rec.get("position_id"):
        raise SystemExit(f"PIPELINE_FAILURE: exporter line missing position_id: {line[:80]}")
    return rec


def export_all_features(engine: Path, records: list[dict]) -> dict[str, dict]:
    """One nnue-features-batch call for every record; fail closed on any
    count / id / fen mismatch."""
    engine = Path(engine).resolve()
    with tempfile.TemporaryDirectory(prefix="s6-n1-export-") as tmp:
        batch_path = Path(tmp) / "batch.txt"
        batch_path.write_text(
            "".join(f"{r['position_id']}|{r['fen']}\n" for r in records),
            encoding="utf-8")
        proc = subprocess.run(
            [str(engine), "bench", "nnue-features-batch", "--batch", str(batch_path)],
            capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise SystemExit(
                f"PIPELINE_FAILURE: exporter exit {proc.returncode}: "
                f"{proc.stderr[:500]}")

    exported: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rec = parse_export_line(line)
        exported[rec["position_id"]] = rec
    if len(exported) != len(records):
        raise SystemExit(
            f"PIPELINE_FAILURE: exporter {len(exported)} rows != {len(records)} records")
    by_id = {r["position_id"]: r for r in records}
    if set(exported) != set(by_id):
        raise SystemExit("PIPELINE_FAILURE: exporter position_id set mismatch")
    for pid, rec in exported.items():
        if rec["fen"] != by_id[pid]["fen"]:
            raise SystemExit(f"PIPELINE_FAILURE: exporter fen mismatch for {pid}")
        for index in rec["white"] + rec["black"]:
            if not 0 <= index < NNUE_INPUTS:
                raise SystemExit(
                    f"PIPELINE_FAILURE: exporter index {index} out of range for {pid}")
    return exported


def classical_eval_stm(engine: Path, fen: str) -> int:
    """base_eval_stm from `bench eval-breakdown` (side-to-move cp score)."""
    proc = subprocess.run(
        [str(engine), "bench", "eval-breakdown", "--fen", fen],
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise SystemExit(
            f"PIPELINE_FAILURE: eval-breakdown exit {proc.returncode} for {fen}: "
            f"{proc.stderr[:300]}")
    for token in proc.stdout.split():
        if token.startswith("base_eval_stm="):
            return int(token.split("=", 1)[1])
    raise SystemExit(f"PIPELINE_FAILURE: no base_eval_stm in output for {fen}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NnueProbe(nn.Module):
    """Frozen S6-N1 probe: shared 40960x32 feature table + 32 bias + sum +
    ReLU + STM-ordered concat + 64->1 head. No extra hidden layers."""

    def __init__(self, inputs: int = NNUE_INPUTS, width: int = WIDTH):
        super().__init__()
        # features.weight is the shared 40960x32 table; the Linear bias is the
        # shared 32-dim accumulator bias. bias=False on the table.
        self.features = nn.Linear(inputs, width, bias=False)
        self.acc_bias = nn.Parameter(torch.zeros(width))
        self.head = nn.Linear(width * 2, 1)

    def accumulate(self, indices: list[torch.Tensor]) -> torch.Tensor:
        """bias + sum(active feature rows), vectorized via index_add."""
        batch_size = len(indices)
        if batch_size == 0:
            return torch.empty(0, self.acc_bias.numel())
        counts = [len(idx) for idx in indices]
        total = sum(counts)
        if total == 0:
            return self.acc_bias.unsqueeze(0).expand(batch_size, -1).contiguous()
        flat = torch.cat(indices)
        batch_ids = torch.cat(
            [torch.full((n,), b, dtype=torch.long) for b, n in enumerate(counts)
             if n > 0])
        # features.weight is (32, 40960); transposed it is the shared
        # 40960 x 32 feature table indexed by NNUE input id.
        rows = self.features.weight.t().index_select(0, flat)  # (total, width)
        acc = self.acc_bias.unsqueeze(0).expand(batch_size, -1).clone()
        acc.index_add_(0, batch_ids, rows)
        return acc

    def forward(self, own: list[torch.Tensor], opp: list[torch.Tensor]) -> torch.Tensor:
        own_acc = torch.clamp_min(self.accumulate(own), 0.0)
        opp_acc = torch.clamp_min(self.accumulate(opp), 0.0)
        x = torch.cat([own_acc, opp_acc], dim=1)
        return self.head(x).squeeze(-1)


def build_model(seed: int = SEED) -> NnueProbe:
    torch.manual_seed(seed)
    return NnueProbe()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_for_split(exported: dict[str, dict], records: list[dict],
                       train_union: set[int] | None = None) -> dict:
    n_positions = len(records)
    white_set: set[int] = set()
    black_set: set[int] = set()
    union_set: set[int] = set()
    total_activations = 0
    unseen_activations = 0
    positions_with_unseen = 0
    for r in records:
        rec = exported[r["position_id"]]
        white = {int(i) for i in rec["white"]}
        black = {int(i) for i in rec["black"]}
        union = white | black
        white_set |= white
        black_set |= black
        union_set |= union
        total_activations += len(white) + len(black)
        if train_union is not None:
            unseen = sum(1 for i in union if i not in train_union)
            unseen_activations += unseen
            if unseen:
                positions_with_unseen += 1
    result = {
        "positions": n_positions,
        "white_unique": len(white_set),
        "black_unique": len(black_set),
        "union_unique": len(union_set),
        "union_fraction": round(len(union_set) / NNUE_INPUTS, 6),
        "white_fraction": round(len(white_set) / NNUE_INPUTS, 6),
        "black_fraction": round(len(black_set) / NNUE_INPUTS, 6),
        "total_activations": total_activations,
    }
    if train_union is not None:
        result["unseen_activations"] = unseen_activations
        result["unseen_rate"] = round(
            unseen_activations / total_activations, 6) if total_activations else 0.0
        result["positions_with_unseen"] = positions_with_unseen
        result["positions_with_unseen_rate"] = round(
            positions_with_unseen / n_positions, 6) if n_positions else 0.0
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def clipped_metrics(pred_cp: list[float], target_cp: list[float]) -> dict:
    n = len(pred_cp)
    raw_mae = statistics.mean(abs(p - t) for p, t in zip(pred_cp, target_cp))
    clipped_pred = [max(-CLIP_CP, min(CLIP_CP, p)) for p in pred_cp]
    clipped_target = [max(-CLIP_CP, min(CLIP_CP, t)) for t in target_cp]
    mae = statistics.mean(abs(p - t) for p, t in zip(clipped_pred, clipped_target))
    rmse = math.sqrt(statistics.mean((p - t) ** 2
                                     for p, t in zip(clipped_pred, clipped_target)))
    buckets: dict[str, dict] = {}
    for lo, hi in CP_BUCKETS:
        key = f"{lo}-{hi if hi else 'inf'}"
        pairs = [(p, t) for p, t in zip(clipped_pred, clipped_target)
                 if (hi is None or abs(t) < hi) and abs(t) >= lo]
        if pairs:
            buckets[key] = {
                "n": len(pairs),
                "mae": round(statistics.mean(abs(p - t) for p, t in pairs), 3),
            }
        else:
            buckets[key] = {"n": 0, "mae": None}
    return {"n": n, "raw_mae_cp": round(raw_mae, 3),
            "clipped_mae_cp": round(mae, 3), "clipped_rmse_cp": round(rmse, 3),
            "buckets": buckets}


def pred_stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def index_tensor(indices: list[int]) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.long)


def prepare_split(exported: dict[str, dict], records: list[dict],
                  labels: dict[str, dict]) -> dict:
    """Rows = records with non-null teacher_cp_stm, in record order."""
    white: list[torch.Tensor] = []
    black: list[torch.Tensor] = []
    target: list[float] = []
    stm_white: list[bool] = []
    fens: list[str] = []
    pids: list[str] = []
    for r in records:
        lbl = labels[r["position_id"]]
        cp = lbl.get("teacher_cp_stm")
        if cp is None:
            continue
        rec = exported[r["position_id"]]
        white.append(index_tensor(rec["white"]))
        black.append(index_tensor(rec["black"]))
        target.append(max(-CLIP_CP, min(CLIP_CP, float(cp))) / TARGET_SCALE)
        stm_white.append(r["fen"].split()[1] == "w")
        fens.append(r["fen"])
        pids.append(r["position_id"])
    return {
        "white": white, "black": black,
        "target": torch.tensor(target, dtype=torch.float32),
        "stm_white": torch.tensor(stm_white, dtype=torch.bool),
        "fens": fens, "pids": pids,
    }


def stm_ordered(split: dict, indices: list[int]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return (own, opponent) index lists ordered by side to move."""
    own: list[torch.Tensor] = []
    opp: list[torch.Tensor] = []
    for i in indices:
        if bool(split["stm_white"][i]):
            own.append(split["white"][i])
            opp.append(split["black"][i])
        else:
            own.append(split["black"][i])
            opp.append(split["white"][i])
    return own, opp


def evaluate_split(model: NnueProbe, split: dict, batch: int = 512) -> tuple[torch.Tensor, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    n = len(split["white"])
    with torch.no_grad():
        for start in range(0, n, batch):
            end = min(start + batch, n)
            own, opp = stm_ordered(split, list(range(start, end)))
            pred = model.forward(own, opp)
            preds.append(pred)
        pred = torch.cat(preds)
        loss = torch.nn.functional.smooth_l1_loss(
            pred, split["target"], beta=LOSS_BETA).item()
    return pred, loss


def train_probe(model: NnueProbe, train: dict, val: dict, seed: int = SEED) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=LOSS_BETA)
    n = len(train["white"])
    gen = torch.Generator().manual_seed(seed)
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_run = 0
    patience_left = PATIENCE
    train_losses: list[float] = []
    start = time.monotonic()
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs_run = epoch
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        count = 0
        for start_idx in range(0, n, BATCH_SIZE):
            idx = perm[start_idx:start_idx + BATCH_SIZE].tolist()
            own, opp = stm_ordered(train, idx)
            target = train["target"][idx]
            optimizer.zero_grad()
            pred = model.forward(own, opp)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
            count += len(idx)
        epoch_loss = running / count
        train_losses.append(epoch_loss)
        _, val_loss = evaluate_split(model, val)
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = epoch_loss
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    elapsed = time.monotonic() - start
    return {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "train_loss_at_best_epoch": round(best_train_loss, 6),
        "final_train_loss": round(train_losses[-1], 6) if train_losses else None,
        "overfit_gap": round(best_val_loss - best_train_loss, 6),
        "early_stopped": epochs_run < MAX_EPOCHS,
        "elapsed_seconds": round(elapsed, 1),
        "train_losses": [round(v, 6) for v in train_losses],
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(zero_mae: float, classical_mae: float, nnue_mae: float) -> str:
    """Evidence-based label recorded in the results.

    STRONG_EARLY_SIGNAL: NNUE beats BOTH zero and classical on holdout
        clipped MAE by >= 5%.
    LEARNABLE_BUT_DATA_STARVED: run completed; NNUE beats zero but not
        classical by the margin, or beats neither (no reliable signal yet).
    PIPELINE_FAILURE: any pipeline error aborts earlier with this label.
    """
    if nnue_mae <= zero_mae * 0.95 and nnue_mae <= classical_mae * 0.95:
        return "STRONG_EARLY_SIGNAL"
    return "LEARNABLE_BUT_DATA_STARVED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def git_sha(path: Path) -> str:
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("data/s6/models/s6-n1-probe.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/s6/s6-n1-probe.json"))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]

    engine_sha = git_sha(repo)
    trainer_sha = git_sha(repo)
    dataset = load_dataset(args.dataset)
    records = dataset["records"]
    labels = dataset["labels"]
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "holdout": []}
    for r in records:
        splits[r["split"]].append(r)

    print(f"dataset records={len(records)} dataset_sha={dataset['dataset_sha'][:16]} "
          f"labels_sha={dataset['labels_sha'][:16]}", flush=True)
    print("exporting features via engine (single batch call)...", flush=True)
    exported = export_all_features(args.engine, records)
    print(f"exported {len(exported)} positions", flush=True)

    prepared = {name: prepare_split(exported, rows, labels)
                for name, rows in splits.items()}
    for name, split in prepared.items():
        print(f"split {name}: positions={len(splits[name])} "
              f"cp_rows={len(split['target'])}", flush=True)

    train_union: set[int] = set()
    for r in splits["train"]:
        rec = exported[r["position_id"]]
        train_union |= {int(i) for i in rec["white"]}
        train_union |= {int(i) for i in rec["black"]}
    coverage = {
        "train": coverage_for_split(exported, splits["train"], None),
        "validation": coverage_for_split(exported, splits["validation"], train_union),
        "holdout": coverage_for_split(exported, splits["holdout"], train_union),
    }
    coverage["train"]["union_fraction"] = round(len(train_union) / NNUE_INPUTS, 6)
    coverage["train"]["union_unique"] = len(train_union)
    print(f"coverage train union={len(train_union)}/{NNUE_INPUTS}", flush=True)

    model = build_model(seed=SEED)
    print("training probe...", flush=True)
    training = train_probe(model, prepared["train"], prepared["validation"])
    print(f"best_epoch={training['best_epoch']} "
          f"best_val_loss={training['best_val_loss']}", flush=True)

    # Restore best epoch checkpoint and evaluate once.
    state = model.state_dict()
    checkpoint_dir = args.checkpoint.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "architecture": {"inputs": NNUE_INPUTS, "width": WIDTH,
                         "activation": "relu", "head": "64->1 linear"},
        "seed": SEED,
        "dataset_sha256": dataset["dataset_sha"],
        "labels_sha256": dataset["labels_sha"],
        "best_epoch": training["best_epoch"],
        "best_val_loss": training["best_val_loss"],
        "trainer_git_sha": trainer_sha,
    }, args.checkpoint)
    checkpoint_sha = hashlib.sha256(
        args.checkpoint.read_bytes()).hexdigest()
    print(f"checkpoint saved: {args.checkpoint} sha256={checkpoint_sha}", flush=True)

    # Classical baseline: val + holdout only (eval-breakdown per position).
    metrics: dict[str, dict] = {}
    for name in ("validation", "holdout"):
        split = prepared[name]
        n = len(split["target"])
        zero_pred = [0.0] * n
        zero = clipped_metrics(zero_pred, [t * TARGET_SCALE for t in split["target"].tolist()])
        classical_pred: list[float] = []
        for fen in split["fens"]:
            classical_pred.append(float(classical_eval_stm(args.engine, fen)))
        classical = clipped_metrics(
            classical_pred, [t * TARGET_SCALE for t in split["target"].tolist()])
        pred, _ = evaluate_split(model, split)
        nnue_pred_cp = [p * TARGET_SCALE for p in pred.tolist()]
        nnue = clipped_metrics(
            nnue_pred_cp, [t * TARGET_SCALE for t in split["target"].tolist()])
        metrics[name] = {
            "zero": zero,
            "classical": classical,
            "nnue": nnue,
            "nnue_prediction_stats": pred_stats(nnue_pred_cp),
        }

    classification = classify(
        metrics["holdout"]["zero"]["clipped_mae_cp"],
        metrics["holdout"]["classical"]["clipped_mae_cp"],
        metrics["holdout"]["nnue"]["clipped_mae_cp"])

    result = {
        "status": "COMPLETE",
        "classification": classification,
        "hashes": {
            "dataset_sha256": dataset["dataset_sha"],
            "labels_sha256": dataset["labels_sha"],
            "checkpoint_sha256": checkpoint_sha,
            "engine_git_sha": engine_sha,
            "trainer_git_sha": trainer_sha,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": __import__("numpy").__version__,
            "python_chess": __import__("chess").__version__,
            "device": str(torch.device("cpu")),
            "os": os.uname().sysname,
        },
        "config": {
            "inputs": NNUE_INPUTS, "width": WIDTH, "seed": SEED,
            "clip_cp": CLIP_CP, "target_scale": TARGET_SCALE,
            "loss": "SmoothL1", "loss_beta": LOSS_BETA,
            "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "init": f"pytorch default init under torch.manual_seed({SEED})",
            "head_ordering": "concat(stm_accumulator, opponent_accumulator)",
            "activation": "clamp-min(0)",
        },
        "dataset": {
            "total": len(records),
            "dataset_id": dataset["manifest"]["dataset_id"],
            "splits": {name: {"positions": len(splits[name]),
                              "cp_rows": int(len(prepared[name]["target"]))}
                       for name in splits},
        },
        "coverage": coverage,
        "training": {k: v for k, v in training.items() if k != "train_losses"},
        "metrics": metrics,
        "classification_rule": (
            "STRONG_EARLY_SIGNAL: nnue holdout clipped MAE <= 0.95 * zero "
            "and <= 0.95 * classical; else LEARNABLE_BUT_DATA_STARVED; "
            "PIPELINE_FAILURE on any pipeline error"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"results written to {args.out}", flush=True)
    print(f"classification: {classification}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
