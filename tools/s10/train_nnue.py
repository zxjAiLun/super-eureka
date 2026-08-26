#!/usr/bin/env python3
"""S10-A Production NNUE Foundation Training Harness.

Architecture (Production 128-32-32-1 baseline with ClippedReLU):
    - Sparse Feature Transformer: NNUE_INPUTS -> 128 (with bias)
    - Activation: ClippedReLU(0.0, 1.0)
    - Perspective Accumulators: Concat [stm_acc, nstm_acc] -> 256
    - Dense Hidden 1: Linear(256, 32) -> ClippedReLU(0.0, 1.0)
    - Dense Hidden 2: Linear(32, 32)  -> ClippedReLU(0.0, 1.0)
    - Output Head:    Linear(32, 1)   -> cp prediction (scaled by TARGET_SCALE)

The Rust engine (`bench nnue-features-batch --feature-set v1|v2`) is the SINGLE
encoding source of truth.

Fail-Closed Rules:
    - Dataset manifest SHA256 verification
    - Teacher manifest / labels.jsonl SHA256 verification
    - Position ID uniqueness and exact matching across records, labels, and exporter
    - Missing/extra labels rejection
    - FEN consistency check
    - Feature index bounds check
    - Usable position_id sets must be identical across all training runs for the same dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

NNUE_INPUTS_V1 = 40960
NNUE_INPUTS_V2 = 22528

CLIP_CP = 2000.0
TARGET_SCALE = 1000.0
LOSS_BETA = 0.1
DEFAULT_LR = 1e-3
DEFAULT_WD = 1e-5
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 15


class ClippedReLU(nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)


class NnueModel(nn.Module):
    """Production NNUE architecture: 128-32-32-1."""

    def __init__(self, num_inputs: int, ft_width: int = 128):
        super().__init__()
        self.num_inputs = num_inputs
        self.ft_width = ft_width

        # Feature Transformer (sparse embedding table + accumulator bias)
        self.ft_weights = nn.Embedding(num_inputs, ft_width)
        self.ft_bias = nn.Parameter(torch.zeros(ft_width))

        # Dense evaluation network
        self.act1 = ClippedReLU(0.0, 1.0)
        self.l1 = nn.Linear(ft_width * 2, 32)
        self.act2 = ClippedReLU(0.0, 1.0)
        self.l2 = nn.Linear(32, 32)
        self.act3 = ClippedReLU(0.0, 1.0)
        self.out = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        # Uniform init for embedding table
        nn.init.uniform_(self.ft_weights.weight, -0.01, 0.01)
        nn.init.zeros_(self.ft_bias)
        nn.init.kaiming_uniform_(self.l1.weight, nonlinearity="relu")
        nn.init.zeros_(self.l1.bias)
        nn.init.kaiming_uniform_(self.l2.weight, nonlinearity="relu")
        nn.init.zeros_(self.l2.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        stm_indices: torch.Tensor,
        stm_offsets: torch.Tensor,
        nstm_indices: torch.Tensor,
        nstm_offsets: torch.Tensor,
    ) -> torch.Tensor:
        # Fast vectorized sparse bag-of-features lookup via nn.functional.embedding_bag
        stm_acc = (
            nn.functional.embedding_bag(
                stm_indices, self.ft_weights.weight, stm_offsets, mode="sum"
            )
            + self.ft_bias
        )
        nstm_acc = (
            nn.functional.embedding_bag(
                nstm_indices, self.ft_weights.weight, nstm_offsets, mode="sum"
            )
            + self.ft_bias
        )

        # Apply ClippedReLU to accumulator output [stm_acc, nstm_acc] -> 256
        acc_act = self.act1(torch.cat([stm_acc, nstm_acc], dim=1))

        # Forward dense layers
        h1 = self.act2(self.l1(acc_act))
        h2 = self.act3(self.l2(h1))
        out = self.out(h2)
        return out.view(-1)


# ---------------------------------------------------------------------------
# Dataset Integrity & Fail-Closed Loading
# ---------------------------------------------------------------------------

def compute_dataset_sha(records: list[dict]) -> str:
    canonical = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"FAIL CLOSED: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    teacher_manifest_path = dataset_dir / "teacher_manifest.json"
    if not teacher_manifest_path.exists():
        raise SystemExit(f"FAIL CLOSED: missing {teacher_manifest_path}")
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))

    records: list[dict] = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    ids = [r["position_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("FAIL CLOSED: duplicate position_id in dataset")

    actual_sha = compute_dataset_sha(records)
    if actual_sha != manifest["dataset_sha256"]:
        raise SystemExit(
            f"FAIL CLOSED: dataset_sha256 mismatch {actual_sha} != {manifest['dataset_sha256']}"
        )
    if len(records) != manifest["records_total"]:
        raise SystemExit(
            f"FAIL CLOSED: record count {len(records)} != manifest {manifest['records_total']}"
        )

    labels: dict[str, dict] = {}
    labels_path = dataset_dir / "labels.jsonl"
    if not labels_path.exists():
        raise SystemExit(f"FAIL CLOSED: missing {labels_path}")
    labels_text = labels_path.read_text(encoding="utf-8")
    for line in labels_text.splitlines():
        if line.strip():
            rec = json.loads(line)
            pid = rec["position_id"]
            if pid in labels:
                raise SystemExit(f"FAIL CLOSED: duplicate position_id in labels: {pid}")
            labels[pid] = rec

    missing = set(ids) - set(labels)
    extra = set(labels) - set(ids)
    if missing:
        raise SystemExit(f"FAIL CLOSED: {len(missing)} records without labels")
    if extra:
        raise SystemExit(f"FAIL CLOSED: {len(extra)} labels without records")

    labels_sha = hashlib.sha256(labels_text.encode("utf-8")).hexdigest()
    if labels_sha != teacher_manifest.get("labels_sha256"):
        raise SystemExit(
            f"FAIL CLOSED: labels_sha256 mismatch {labels_sha} != {teacher_manifest.get('labels_sha256')}"
        )

    return {
        "records": records,
        "labels": labels,
        "dataset_sha": actual_sha,
        "labels_sha": labels_sha,
        "manifest": manifest,
        "teacher_manifest": teacher_manifest,
    }


def export_features_from_engine(
    engine: Path, records: list[dict], feature_set: str
) -> dict[str, dict]:
    engine = Path(engine).resolve()
    with tempfile.TemporaryDirectory(prefix="s10-export-") as tmp:
        batch_path = Path(tmp) / "batch.txt"
        batch_path.write_text(
            "".join(f"{r['position_id']}|{r['fen']}\n" for r in records),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                str(engine),
                "bench",
                "nnue-features-batch",
                "--batch",
                str(batch_path),
                "--feature-set",
                feature_set,
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"FAIL CLOSED: exporter exit code {proc.returncode}: {proc.stderr[:500]}"
            )

    max_dim = NNUE_INPUTS_V1 if feature_set == "v1" else NNUE_INPUTS_V2
    exported: dict[str, dict] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pid = rec["position_id"]
        if pid in exported:
            raise SystemExit(f"FAIL CLOSED: duplicate position_id in export: {pid}")
        for idx in rec["white"] + rec["black"]:
            if idx < 0 or idx >= max_dim:
                raise SystemExit(
                    f"FAIL CLOSED: feature index {idx} out of range [0, {max_dim}) in export"
                )
        exported[pid] = rec

    if len(exported) != len(records):
        raise SystemExit(
            f"FAIL CLOSED: exporter count {len(exported)} != input records {len(records)}"
        )

    for r in records:
        pid = r["position_id"]
        if pid not in exported:
            raise SystemExit(f"FAIL CLOSED: missing exported record for {pid}")
        if exported[pid]["fen"] != r["fen"]:
            raise SystemExit(
                f"FAIL CLOSED: fen mismatch for {pid}: '{exported[pid]['fen']}' != '{r['fen']}'"
            )

    return exported


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class EncodedSplit:
    def __init__(self, items: list[dict]):
        self.items = items
        self.n = len(items)

        # Pre-compute flattened arrays and offsets for fast batched embedding_bag
        stm_indices = []
        stm_offsets = []
        nstm_indices = []
        nstm_offsets = []
        targets = []
        raw_cps = []

        cur_stm_off = 0
        cur_nstm_off = 0

        for it in items:
            stm_offsets.append(cur_stm_off)
            nstm_offsets.append(cur_nstm_off)

            stm_indices.extend(it["stm"])
            nstm_indices.extend(it["nstm"])

            cur_stm_off += len(it["stm"])
            cur_nstm_off += len(it["nstm"])

            targets.append(it["target_scaled"])
            raw_cps.append(it["target_cp"])

        self.stm_indices = torch.tensor(stm_indices, dtype=torch.long)
        self.stm_offsets = torch.tensor(stm_offsets, dtype=torch.long)
        self.nstm_indices = torch.tensor(nstm_indices, dtype=torch.long)
        self.nstm_offsets = torch.tensor(nstm_offsets, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.raw_cps = torch.tensor(raw_cps, dtype=torch.float32)


def train_and_eval(
    dataset_dir: Path,
    engine_bin: Path,
    feature_set: str,
    seed: int,
    output_dir: Path,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WD,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    device_name: str | None = None,
    allow_holdout: bool = False,
) -> dict:
    start_time = time.time()
    num_inputs = NNUE_INPUTS_V1 if feature_set == "v1" else NNUE_INPUTS_V2

    # Determinism / Device setup
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # 1. Load dataset with fail-closed validation
    ds = load_dataset(dataset_dir)
    records = ds["records"]
    labels = ds["labels"]

    # 2. Filter usable records (teacher_cp_stm not None)
    usable = []
    for r in records:
        lbl = labels[r["position_id"]]
        cp = lbl.get("teacher_cp_stm")
        if cp is not None:
            usable.append((r, lbl))

    if not usable:
        raise SystemExit("FAIL CLOSED: 0 usable records with teacher_cp_stm")

    # 3. Export features from Rust engine
    exported = export_features_from_engine(engine_bin, [r for r, _ in usable], feature_set)

    # 4. Prepare split items
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "holdout": []}
    for r, lbl in usable:
        pid = r["position_id"]
        split = r["split"]
        exp = exported[pid]
        target_cp = max(-CLIP_CP, min(CLIP_CP, float(lbl["teacher_cp_stm"])))
        target_scaled = target_cp / TARGET_SCALE

        stm_is_white = r["fen"].split()[1] == "w"
        stm_feats = exp["white"] if stm_is_white else exp["black"]
        nstm_feats = exp["black"] if stm_is_white else exp["white"]

        item = {
            "position_id": pid,
            "white": exp["white"],
            "black": exp["black"],
            "stm": stm_feats,
            "nstm": nstm_feats,
            "target_scaled": target_scaled,
            "target_cp": target_cp,
        }
        splits[split].append(item)

    train_items = splits["train"]
    val_items = splits["validation"]
    holdout_items = splits["holdout"]

    # Feature coverage diagnostics
    train_feat_counts = torch.zeros(num_inputs, dtype=torch.int64)
    for item in train_items:
        for idx in item["white"]:
            train_feat_counts[idx] += 1
        for idx in item["black"]:
            train_feat_counts[idx] += 1

    train_observed_unique = int((train_feat_counts > 0).sum().item())
    singleton_count = int((train_feat_counts == 1).sum().item())
    le5_count = int(((train_feat_counts > 0) & (train_feat_counts <= 5)).sum().item())

    # Validation unseen diagnostics
    val_total_activations = 0
    val_unseen_activations = 0
    val_positions_with_unseen = 0

    for item in val_items:
        pos_has_unseen = False
        for idx in item["white"]:
            val_total_activations += 1
            if train_feat_counts[idx] == 0:
                val_unseen_activations += 1
                pos_has_unseen = True
        for idx in item["black"]:
            val_total_activations += 1
            if train_feat_counts[idx] == 0:
                val_unseen_activations += 1
                pos_has_unseen = True
        if pos_has_unseen:
            val_positions_with_unseen += 1

    val_unseen_activation_rate = (
        (val_unseen_activations / val_total_activations) if val_total_activations > 0 else 0.0
    )
    val_positions_with_unseen_rate = (
        (val_positions_with_unseen / len(val_items)) if len(val_items) > 0 else 0.0
    )

    # Encode splits
    val_encoded = EncodedSplit(val_items)
    val_stm_ind = val_encoded.stm_indices.to(device)
    val_stm_off = val_encoded.stm_offsets.to(device)
    val_nstm_ind = val_encoded.nstm_indices.to(device)
    val_nstm_off = val_encoded.nstm_offsets.to(device)
    val_targets = val_encoded.targets.to(device)
    val_raw_cps = val_encoded.raw_cps.to(device)

    # 5. Initialize Model & Optimizer
    model = NnueModel(num_inputs=num_inputs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=LOSS_BETA)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0

    g = torch.Generator()
    g.manual_seed(seed)

    history = []
    n_train = len(train_items)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=g).tolist()
        train_loss_accum = 0.0

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = perm[start_idx:end_idx]
            batch_items = [train_items[i] for i in batch_indices]
            b_enc = EncodedSplit(batch_items)

            stm_ind = b_enc.stm_indices.to(device)
            stm_off = b_enc.stm_offsets.to(device)
            nstm_ind = b_enc.nstm_indices.to(device)
            nstm_off = b_enc.nstm_offsets.to(device)
            targets = b_enc.targets.to(device)

            optimizer.zero_grad()
            preds = model(stm_ind, stm_off, nstm_ind, nstm_off)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * len(batch_items)

        train_loss = train_loss_accum / n_train

        # Fast vectorized validation pass
        model.eval()
        with torch.no_grad():
            preds = model(val_stm_ind, val_stm_off, val_nstm_ind, val_nstm_off)
            val_loss = criterion(preds, val_targets).item()
            pred_cp = preds * TARGET_SCALE
            val_mae = torch.mean(torch.abs(pred_cp - val_raw_cps)).item()

        epoch_rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
        }
        history.append(epoch_rec)

        # Checkpoint strictly by validation SmoothL1 loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Restore best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    model.eval()

    # Re-evaluate validation set with best checkpoint to verify restored loss/MAE parity
    with torch.no_grad():
        preds = model(val_stm_ind, val_stm_off, val_nstm_ind, val_nstm_off)
        restored_val_loss = criterion(preds, val_targets).item()
        pred_cp = preds * TARGET_SCALE
        restored_val_mae = torch.mean(torch.abs(pred_cp - val_raw_cps)).item()

    assert abs(restored_val_loss - best_val_loss) < 1e-5, "Restored val loss parity failure"

    # Holdout evaluation (Stage 2 strictly only)
    holdout_metrics = None
    if allow_holdout:
        holdout_encoded = EncodedSplit(holdout_items)
        h_stm_ind = holdout_encoded.stm_indices.to(device)
        h_stm_off = holdout_encoded.stm_offsets.to(device)
        h_nstm_ind = holdout_encoded.nstm_indices.to(device)
        h_nstm_off = holdout_encoded.nstm_offsets.to(device)
        h_targets = holdout_encoded.targets.to(device)
        h_raw_cps = holdout_encoded.raw_cps.to(device)

        with torch.no_grad():
            preds = model(h_stm_ind, h_stm_off, h_nstm_ind, h_nstm_off)
            h_loss = criterion(preds, h_targets).item()
            pred_cp = preds * TARGET_SCALE
            h_mae = torch.mean(torch.abs(pred_cp - h_raw_cps)).item()

        holdout_metrics = {
            "records_evaluated": len(holdout_items),
            "holdout_loss": h_loss,
            "holdout_mae": h_mae,
        }

    # Parameter footprint calculation
    ft_param_count = num_inputs * 128 + 128
    dense_param_count = (256 * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
    total_param_count = ft_param_count + dense_param_count
    ft_fp32_bytes = ft_param_count * 4
    total_fp32_bytes = total_param_count * 4

    elapsed_s = time.time() - start_time

    summary = {
        "feature_set": feature_set,
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "dataset_sha256": ds["dataset_sha"],
        "labels_sha256": ds["labels_sha"],
        "records_usable": len(usable),
        "usable_split_counts": {k: len(v) for k, v in splits.items()},
        "architecture": {
            "num_inputs": num_inputs,
            "ft_width": 128,
            "dense_layers": [256, 32, 32, 1],
            "ft_param_count": ft_param_count,
            "dense_param_count": dense_param_count,
            "total_param_count": total_param_count,
            "ft_fp32_bytes": ft_fp32_bytes,
            "total_fp32_bytes": total_fp32_bytes,
            "ft_fp32_mib": ft_fp32_bytes / (1024 * 1024),
            "total_fp32_mib": total_fp32_bytes / (1024 * 1024),
        },
        "coverage_diagnostics": {
            "train_observed_unique": train_observed_unique,
            "train_observed_ratio": train_observed_unique / num_inputs,
            "singleton_count": singleton_count,
            "singleton_ratio": singleton_count / train_observed_unique if train_observed_unique > 0 else 0.0,
            "le5_count": le5_count,
            "le5_ratio": le5_count / train_observed_unique if train_observed_unique > 0 else 0.0,
            "val_unseen_activation_rate": val_unseen_activation_rate,
            "val_positions_with_unseen_rate": val_positions_with_unseen_rate,
        },
        "training": {
            "best_epoch": best_epoch,
            "total_epochs": len(history),
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "elapsed_seconds": elapsed_s,
        },
        "holdout_observed": holdout_metrics is not None,
        "holdout_metrics": holdout_metrics,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary artifact
    summary_path = output_dir / f"training_summary_{feature_set}_s{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Save best checkpoint model
    checkpoint_path = output_dir / f"checkpoint_{feature_set}_s{seed}.pt"
    torch.save(
        {
            "summary": summary,
            "model_state_dict": best_state_dict,
        },
        checkpoint_path,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="S10 Production NNUE Training Harness")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--engine", type=Path, required=True, help="Path to eureka engine binary")
    parser.add_argument("--feature-set", choices=["v1", "v2"], required=True, help="Feature set representation")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow-holdout", action="store_true", help="Stage 2 only: evaluate holdout split")

    args = parser.parse_args()
    summary = train_and_eval(
        dataset_dir=args.dataset,
        engine_bin=args.engine,
        feature_set=args.feature_set,
        seed=args.seed,
        output_dir=args.output,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device_name=args.device,
        allow_holdout=args.allow_holdout,
    )
    print(f"Training completed for {args.feature_set} seed {args.seed}: best_epoch={summary['training']['best_epoch']} best_val_mae={summary['training']['best_val_mae']:.3f} cp (elapsed: {summary['training']['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
