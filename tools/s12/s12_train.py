#!/usr/bin/env python3
"""S12: Bullet-style output-buckets NNUE for Eureka (first recipe).

Adapted from Bullet 629ee500 examples/progression/2_output_buckets.rs:
  - dual-perspective FT (V2+R12 inputs, 256 lanes/perspective)
  - SCReLU activation: clamp(x, 0, 1)^2
  - ONE linear output layer 2*256 -> 8 buckets, selected per position
    by Bullet's material-count bucket: (popcount(occ) - 2) / 4
    (count includes kings; 8 buckets; u8 integer division)
  - score head: eval_cp = material_cp_stm + 1000 * network_output
  - loss: MSE over sigmoid(eval_cp / 400) vs sigmoid(teacher_cp / 400)
    (both stm-perspective; teacher_cp already clipped +-2000 upstream)

Kept from the existing S10/S11 stack (via train_nnue):
  dataset loading/provenance, feature export, material export, splits,
  EncodedSplit, determinism. NOT copied: the old 32-32-1 tail, the old
  SmoothL1-on-scaled-cp loss, the old bucket routing.

Budget (first run): single seed, max 20 epochs OR 1 GPU-hour
(whichever first), AdamW + cosine LR decay, checkpoint by the NEW
validation loss (sigmoid MSE).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

import importlib.util as _ilu

_HERE = Path(__file__).parent
_spec = _ilu.spec_from_file_location(
    "_tn", str(_HERE.parent / "s10" / "train_nnue.py"))
_tn = _ilu.module_from_spec(_spec)
import sys as _sys

_sys.modules.setdefault("_tn", _tn)
_spec.loader.exec_module(_tn)

EncodedSplit = _tn.EncodedSplit
export_features_from_engine = _tn.export_features_from_engine
export_material_from_engine = _tn.export_material_from_engine
load_dataset = _tn.load_dataset
TARGET_SCALE = _tn.TARGET_SCALE          # 1000.0 — residual output unit
CLIP_CP = _tn.CLIP_CP                    # 2000.0

NNUE_INPUTS_V2R12 = _tn.NNUE_INPUTS_V2R12
NNUE_V2R6_REL_BASE = _tn.NNUE_V2R6_REL_BASE

NUM_OUTPUT_BUCKETS = 8
BUCKET_DIVISOR = 4  # ceil(32/8) — Bullet MaterialCount::<8>
SCORE_SCALE = 400.0  # sigmoid temperature for the score loss


def output_bucket(board_piece_count: int) -> int:
    """Bullet MaterialCount::<8>: (occ_count - 2) / 4, count includes
    kings. occ in [2, 32] -> bucket in [0, 7]."""
    return (board_piece_count - 2) // BUCKET_DIVISOR


class Screlu(torch.nn.Module):
    """SCReLU: clamp(x, 0, 1)^2 (float training space)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0.0, 1.0).square()


class S12Model(torch.nn.Module):
    """FT(256) x2 perspectives -> SCReLU -> Linear(512, 8) -> bucket select.

    Output = single scalar per position (the selected bucket's logit).
    eval_cp = material_cp_stm + TARGET_SCALE * output.
    """

    def __init__(self, num_inputs: int, ft_width: int = 256):
        super().__init__()
        self.num_inputs = num_inputs
        self.ft_width = ft_width
        self.ft_weights = torch.nn.Embedding(num_inputs, ft_width)
        self.ft_bias = torch.nn.Parameter(torch.zeros(ft_width))
        self.act = Screlu()
        # ONE linear output layer over the concatenated (stm, nstm)
        # activations, one column per output bucket.
        self.l1 = torch.nn.Linear(2 * ft_width, NUM_OUTPUT_BUCKETS)
        self._init_weights()

    def forward(self, stm_indices, stm_offsets, nstm_indices,
                nstm_offsets, buckets):
        stm_acc = (
            torch.nn.functional.embedding_bag(
                stm_indices, self.ft_weights.weight, stm_offsets,
                mode="sum")
            + self.ft_bias
        )
        nstm_acc = (
            torch.nn.functional.embedding_bag(
                nstm_indices, self.ft_weights.weight, nstm_offsets,
                mode="sum")
            + self.ft_bias
        )
        hidden = self.act(torch.cat([stm_acc, nstm_acc], dim=1))
        logits = self.l1(hidden)                      # [N, 8]
        out = logits.gather(
            1, buckets.view(-1, 1)).view(-1)          # select one bucket
        return out

    def _init_weights(self):
        torch.nn.init.uniform_(self.ft_weights.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.ft_bias)
        torch.nn.init.kaiming_uniform_(self.l1.weight,
                                       nonlinearity="relu")
        torch.nn.init.zeros_(self.l1.bias)


def train_s12(
    dataset_dir: Path,
    engine_bin: Path,
    seed: int,
    output_dir: Path,
    feature_set: str = "v2r12",
    ft_width: int = 256,
    lr: float = 1e-3,
    final_lr_frac: float = 0.3 ** 5,     # Bullet: 0.001 * 0.3^5
    weight_decay: float = 1e-5,
    batch_size: int = 1024,
    max_epochs: int = 20,
    max_gpu_seconds: float = 3600.0,
    patience: int = 6,
    device_name: str | None = None,
) -> dict:
    import chess as _chess

    assert feature_set == "v2r12", "S12 first recipe: V2+R12 inputs"
    assert ft_width == 256, "S12 first recipe: FT256"
    num_inputs = NNUE_INPUTS_V2R12

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    t_start = time.time()

    # ---- data (existing pipeline, zero new labels) ----
    ds = load_dataset(dataset_dir)
    records = ds["records"]
    labels = ds["labels"]
    usable = []
    for r in records:
        lbl = labels[r["position_id"]]
        if lbl.get("teacher_cp_stm") is not None:
            usable.append((r, lbl))
    export_records = [r for r, _ in usable]
    exported = export_features_from_engine(
        engine_bin, export_records, "v2r12")
    material_stm = export_material_from_engine(
        engine_bin, export_records)

    splits: dict[str, list[dict]] = {"train": [], "validation": []}
    _bucket_cache: dict[str, int] = {}

    def _bucket_of_fen(fen: str) -> int:
        b = _bucket_cache.get(fen)
        if b is None:
            board = _chess.Board(fen)
            b = output_bucket(
                len(board.piece_map()))
            _bucket_cache[fen] = b
        return b

    for r, lbl in usable:
        pid = r["position_id"]
        if r["split"] not in ("train", "validation"):
            continue
        exp = exported[pid]
        stm_is_white = r["fen"].split()[1] == "w"
        target_cp = max(-CLIP_CP, min(CLIP_CP,
                                      float(lbl["teacher_cp_stm"])))
        splits[r["split"]].append({
            "position_id": pid,
            "white": exp["white"],
            "black": exp["black"],
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
            # EncodedSplit reads these two keys:
            "target_scaled": target_cp / TARGET_SCALE,
            "target_cp": target_cp,
            "material_cp_stm": float(material_stm[pid]),
            "bucket": _bucket_of_fen(r["fen"]),
        })

    train_items = splits["train"]
    val_items = splits["validation"]
    val_encoded = EncodedSplit(val_items)
    val_stm_ind = val_encoded.stm_indices.to(device)
    val_stm_off = val_encoded.stm_offsets.to(device)
    val_nstm_ind = val_encoded.nstm_indices.to(device)
    val_nstm_off = val_encoded.nstm_offsets.to(device)
    val_buckets = val_encoded.buckets.to(device)
    # val targets for BOTH losses: raw cp (teacher, clipped) + material
    val_teacher_cp = val_encoded.raw_cps.to(device)
    val_material = torch.tensor(
        [it["material_cp_stm"] for it in val_items],
        dtype=torch.float32, device=device)

    # ---- model / optimizer / schedule ----
    model = S12Model(num_inputs=num_inputs, ft_width=ft_width).to(device)
    # R12 sidecar rows zero-init (S11-A fairness convention)
    with torch.no_grad():
        model.ft_weights.weight[NNUE_V2R6_REL_BASE:].zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    steps_per_epoch = (len(train_items) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(
            final_lr_frac,
            0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
        ),
    )

    def score_loss(eval_cp: torch.Tensor,
                   teacher_cp: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(eval_cp / SCORE_SCALE)
        tgt = torch.sigmoid(teacher_cp / SCORE_SCALE)
        return torch.mean((pred - tgt) ** 2)

    g = torch.Generator()
    g.manual_seed(seed)
    n_train = len(train_items)
    history = []
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    global_step = 0
    budget_hit = None

    for epoch in range(1, max_epochs + 1):
        if time.time() - t_start > max_gpu_seconds:
            budget_hit = "gpu_time"
            print(f"budget stop before epoch {epoch}: "
                  f"{time.time() - t_start:.0f}s > {max_gpu_seconds:.0f}s",
                  flush=True)
            break
        model.train()
        perm = torch.randperm(n_train, generator=g).tolist()
        accum = 0.0
        for start_idx in range(0, n_train, batch_size):
            if time.time() - t_start > max_gpu_seconds:
                budget_hit = "gpu_time"
                break
            batch_items = [train_items[i]
                           for i in perm[start_idx:start_idx + batch_size]]
            b = EncodedSplit(batch_items)
            stm_ind = b.stm_indices.to(device)
            stm_off = b.stm_offsets.to(device)
            nstm_ind = b.nstm_indices.to(device)
            nstm_off = b.nstm_offsets.to(device)
            buckets = b.buckets.to(device)
            teacher_cp = b.raw_cps.to(device)
            material = torch.tensor(
                [it["material_cp_stm"] for it in batch_items],
                dtype=torch.float32, device=device)

            optimizer.zero_grad()
            out = model(stm_ind, stm_off, nstm_ind, nstm_off, buckets)
            eval_cp = material + TARGET_SCALE * out
            loss = score_loss(eval_cp, teacher_cp)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            accum += loss.item() * len(batch_items)
        if budget_hit == "gpu_time":
            break
        train_loss = accum / n_train

        # validation: NEW loss + composed MAE (for reporting continuity)
        model.eval()
        with torch.no_grad():
            out = model(val_stm_ind, val_stm_off, val_nstm_ind,
                        val_nstm_off, val_buckets)
            eval_cp = val_material + TARGET_SCALE * out
            v_loss = score_loss(eval_cp, val_teacher_cp).item()
            v_mae = torch.mean(
                torch.abs(eval_cp - val_teacher_cp)).item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": v_loss, "val_mae": v_mae})
        print(f"epoch {epoch}: train {train_loss:.6f} val {v_loss:.6f} "
              f"val_mae {v_mae:.1f} lr "
              f"{scheduler.get_last_lr()[0]:.2e} "
              f"({time.time() - t_start:.0f}s)", flush=True)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_mae = v_mae
            best_epoch = epoch
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                budget_hit = budget_hit or "patience"
                break

    model.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # final val pass with the restored best checkpoint
    with torch.no_grad():
        out = model(val_stm_ind, val_stm_off, val_nstm_ind,
                    val_nstm_off, val_buckets)
        eval_cp = val_material + TARGET_SCALE * out
        restored_loss = score_loss(eval_cp, val_teacher_cp).item()
        restored_mae = torch.mean(
            torch.abs(eval_cp - val_teacher_cp)).item()
    assert abs(restored_loss - best_val_loss) < 1e-6, \
        "restored checkpoint val-loss parity failure"

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_s12_{feature_set}_s{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "summary": {
            "schema": "s12-bullet-recipe",
            "feature_set": feature_set,
            "seed": seed,
            "ft_width": ft_width,
            "num_inputs": num_inputs,
            "output_buckets": NUM_OUTPUT_BUCKETS,
            "bucket_rule": "(popcount_occ_incl_kings - 2) // 4",
            "activation": "screlu",
            "head": "linear(512, 8) bucket-select",
            "score": "eval_cp = material_cp_stm + 1000 * out",
            "loss": f"mse(sigmoid(eval_cp/{SCORE_SCALE}), "
                    f"sigmoid(teacher_cp/{SCORE_SCALE}))",
            "target_mode": "material-residual",
            "dataset_sha256": ds["dataset_sha"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "budget_hit": budget_hit,
            "elapsed_seconds": time.time() - t_start,
            "steps": global_step,
        },
    }, ckpt_path)

    summary = {
        "schema": "s12_training_summary",
        "checkpoint": str(ckpt_path),
        "feature_set": feature_set,
        "seed": seed,
        "ft_width": ft_width,
        "num_inputs": num_inputs,
        "output_buckets": NUM_OUTPUT_BUCKETS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "budget_hit": budget_hit,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "history": history,
    }
    (output_dir / f"training_summary_s12_{feature_set}_s{seed}.json"
     ).write_text(json.dumps(summary, indent=1) + "\n",
                  encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "best_epoch", "best_val_loss", "best_val_mae", "budget_hit",
        "elapsed_seconds")}, indent=1))
    return summary


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--engine", type=Path,
                    default=Path(r"target\release\eureka.exe"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--max-gpu-seconds", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_s12(args.dataset, args.engine, args.seed, args.out,
              max_epochs=args.max_epochs,
              max_gpu_seconds=args.max_gpu_seconds,
              batch_size=args.batch_size, lr=args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
