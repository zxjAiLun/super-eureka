#!/usr/bin/env python3
"""S14-B trainer: the S12-R0 recipe VERBATIM, only the training data changes.

Frozen recipe (identical to S12-R0 / tools/s12/s12_train.py):
  * V2+R12 inputs (23296), FT256, SCReLU, 8 MaterialCount output buckets,
    material-residual head (eval_cp = material_cp_stm + 1000 * out).
  * CP-only loss: MSE( sigmoid(eval_cp/400), sigmoid(teacher_cp/400) ).
    NO WDL / result blend (wdl_proportion is not a knob here).
  * batch 1024, seed 20260908, AdamW lr 1e-3 wd 1e-5, cosine decay.
  * R12 relation sidecar rows zero-init (S11-A fairness convention).

What is DIFFERENT from S12-R0 (this is the whole S14 experiment):
  * The training pool is DUAL-TRACK:
      - canonical CP corpus TRAIN split (779,590 positions with a
        teacher_cp_stm label), built through the exact S12 engine-export
        path so every feature/material/bucket is bit-identical to S12; +
      - fresh Fishtest LTC CP positions (4,220,410) encoded by
        tools/s14/s14_encode.py (the engine's own feature/material
        exporters -> one semantic source), read straight from
        data/s14/encoded/encoded-*.jsonl.gz.
    779,590 + 4,220,410 = 5,000,000 presentations-per-pass.
  * Budget is measured in PRESENTATIONS, not epochs: stop at
    >= --presentations (default 20,000,000 ~= 4 passes) OR
    >= --max-seconds (default 3600 = 1 GPU hour), whichever first.
    total_steps for the cosine schedule is planned from --presentations.
  * MODEL SELECTION uses ONLY the original Y16 corpus validation split
    (97,528 CP-labeled positions) -- never Fishtest data -- so we do not
    silently overfit the new label distribution. Lowest sigmoid-loss
    checkpoint wins (evaluated every --val-every steps + once at the end).

Outputs an S12-format checkpoint (schema "s12-bullet-recipe") so
tools/s12/s12_export.py can quantize it to a v5 artifact with no changes.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
import time
from array import array
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"tools\s12")
import s12_train  # noqa: E402  (S12Model, output_bucket, re-exported symbols)
import chess  # noqa: E402

S12Model = s12_train.S12Model
output_bucket = s12_train.output_bucket
EncodedSplit = s12_train.EncodedSplit
export_features_from_engine = s12_train.export_features_from_engine
export_material_from_engine = s12_train.export_material_from_engine
load_dataset = s12_train.load_dataset
TARGET_SCALE = s12_train.TARGET_SCALE            # 1000.0
CLIP_CP = s12_train.CLIP_CP                       # 2000.0
SCORE_SCALE = s12_train.SCORE_SCALE               # 400.0
NNUE_INPUTS_V2R12 = s12_train.NNUE_INPUTS_V2R12   # 23296
NNUE_V2R6_REL_BASE = s12_train.NNUE_V2R6_REL_BASE  # 22528
NUM_OUTPUT_BUCKETS = s12_train.NUM_OUTPUT_BUCKETS  # 8


# ---------------------------------------------------------------------------
# Flat, memory-frugal pool of sparse feature indices.
# A single Python dict per position (5M of them, each with two int lists)
# would cost ~20 GB; instead we keep one contiguous int32 buffer of all
# stm indices + a CSR-style offset array (and likewise for nstm), plus
# per-position material/bucket/cp scalars. ~2 GB for the whole 5M pool.
# ---------------------------------------------------------------------------
class FlatPool:
    def __init__(self):
        self._stm_chunks: list[np.ndarray] = []
        self._nstm_chunks: list[np.ndarray] = []
        self._stm_len = array("i")
        self._nstm_len = array("i")
        self._material = array("f")
        self._bucket = array("b")
        self._cp = array("f")
        self.n = 0

    def add(self, stm_idx: np.ndarray, nstm_idx: np.ndarray,
            material: float, bucket: int, cp: float):
        self._stm_chunks.append(stm_idx)
        self._nstm_chunks.append(nstm_idx)
        self._stm_len.append(len(stm_idx))
        self._nstm_len.append(len(nstm_idx))
        self._material.append(float(material))
        self._bucket.append(int(bucket))
        self._cp.append(float(cp))
        self.n += 1

    def add_bulk(self, stm_flat: np.ndarray, stm_len: np.ndarray,
                 nstm_flat: np.ndarray, nstm_len: np.ndarray,
                 material, bucket, cp):
        """Append a whole shard at once (arrays already parsed)."""
        self._stm_chunks.append(stm_flat)
        self._nstm_chunks.append(nstm_flat)
        self._stm_len.extend(stm_len.tolist())
        self._nstm_len.extend(nstm_len.tolist())
        self._material.extend(material)
        self._bucket.extend(bucket)
        self._cp.extend(cp)
        self.n += len(stm_len)

    def finalize(self, max_index: int):
        stm_flat = np.concatenate(self._stm_chunks).astype(np.int32, copy=False)
        nstm_flat = np.concatenate(self._nstm_chunks).astype(np.int32,
                                                             copy=False)
        # fail closed on any out-of-range sparse index
        if stm_flat.size:
            lo, hi = int(stm_flat.min()), int(stm_flat.max())
            if lo < 0 or hi >= max_index:
                raise SystemExit(
                    f"FAIL CLOSED: stm feature index out of range "
                    f"[{lo},{hi}] not in [0,{max_index})")
        if nstm_flat.size:
            lo, hi = int(nstm_flat.min()), int(nstm_flat.max())
            if lo < 0 or hi >= max_index:
                raise SystemExit(
                    f"FAIL CLOSED: nstm feature index out of range "
                    f"[{lo},{hi}] not in [0,{max_index})")
        stm_len = np.array(self._stm_len, dtype=np.int64)
        nstm_len = np.array(self._nstm_len, dtype=np.int64)
        stm_off = np.zeros(self.n + 1, dtype=np.int64)
        nstm_off = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(stm_len, out=stm_off[1:])
        np.cumsum(nstm_len, out=nstm_off[1:])
        assert int(stm_off[-1]) == stm_flat.size, (stm_off[-1], stm_flat.size)
        assert int(nstm_off[-1]) == nstm_flat.size
        self.stm_flat = torch.from_numpy(stm_flat)          # int32 [T]
        self.nstm_flat = torch.from_numpy(nstm_flat)        # int32 [T]
        self.stm_off = torch.from_numpy(stm_off)            # int64 [N+1]
        self.nstm_off = torch.from_numpy(nstm_off)          # int64 [N+1]
        self.material = torch.from_numpy(
            np.array(self._material, dtype=np.float32))
        self.bucket = torch.from_numpy(
            np.array(self._bucket, dtype=np.int64))
        self.cp = torch.from_numpy(
            np.array(self._cp, dtype=np.float32))
        # release the builders
        self._stm_chunks = self._nstm_chunks = None
        self._stm_len = self._nstm_len = None
        self._material = self._bucket = self._cp = None
        self.total_stm = int(stm_flat.size)
        self.total_nstm = int(nstm_flat.size)


def _gather_bag(flat_int32: torch.Tensor, off: torch.Tensor,
                idx: torch.Tensor):
    """CSR ragged-gather: return (indices, offsets) for embedding_bag over
    the positions selected by `idx` (a 1D long tensor). Fully vectorized."""
    starts = off[idx]
    lens = off[idx + 1] - starts
    total = int(lens.sum().item())
    out_off = torch.zeros(idx.numel(), dtype=torch.long)
    if idx.numel() > 1:
        torch.cumsum(lens[:-1], 0, out=out_off[1:])
    base = torch.repeat_interleave(starts - out_off, lens)
    seg = base + torch.arange(total, dtype=torch.long)
    batch_indices = flat_int32[seg].long()
    return batch_indices, out_off


# ---------------------------------------------------------------------------
# Track 1: canonical CP corpus (exact S12 engine-export path).
# ---------------------------------------------------------------------------
def build_canonical(dataset_dir: Path, engine_bin: Path, pool: FlatPool):
    """Append the canonical TRAIN split to `pool` and return the
    validation items (list of dicts for EncodedSplit) + dataset sha."""
    ds = load_dataset(dataset_dir)
    records = ds["records"]
    labels = ds["labels"]
    usable = [(r, labels[r["position_id"]]) for r in records
              if labels[r["position_id"]].get("teacher_cp_stm") is not None]
    export_records = [r for r, _ in usable]
    exported = export_features_from_engine(engine_bin, export_records, "v2r12")
    material_stm = export_material_from_engine(engine_bin, export_records)

    _bucket_cache: dict[str, int] = {}

    def bucket_of(fen: str) -> int:
        b = _bucket_cache.get(fen)
        if b is None:
            b = output_bucket(len(chess.Board(fen).piece_map()))
            _bucket_cache[fen] = b
        return b

    val_items: list[dict] = []
    n_train = 0
    for r, lbl in usable:
        split = r["split"]
        if split not in ("train", "validation"):
            continue
        pid = r["position_id"]
        exp = exported[pid]
        stm_is_white = r["fen"].split()[1] == "w"
        stm = exp["white"] if stm_is_white else exp["black"]
        nstm = exp["black"] if stm_is_white else exp["white"]
        target_cp = max(-CLIP_CP, min(CLIP_CP, float(lbl["teacher_cp_stm"])))
        bucket = bucket_of(r["fen"])
        if split == "train":
            pool.add(np.asarray(stm, dtype=np.int32),
                     np.asarray(nstm, dtype=np.int32),
                     float(material_stm[pid]), bucket, target_cp)
            n_train += 1
        else:  # validation -> Y16 model-selection set (canonical only)
            val_items.append({
                "stm": stm,
                "nstm": nstm,
                "target_scaled": target_cp / TARGET_SCALE,
                "target_cp": target_cp,
                "material_cp_stm": float(material_stm[pid]),
                "bucket": bucket,
            })
    del exported, material_stm
    return val_items, ds["dataset_sha"], n_train


# ---------------------------------------------------------------------------
# Track 2: fresh Fishtest CP (pre-encoded shards).
# ---------------------------------------------------------------------------
def load_fishtest(encoded_glob: str, pool: FlatPool, limit: int = 0) -> int:
    shards = sorted(glob.glob(encoded_glob))
    if not shards:
        raise SystemExit(f"FAIL CLOSED: no encoded shards match {encoded_glob}")
    added = 0
    for sp in shards:
        stm_strs: list[str] = []
        nstm_strs: list[str] = []
        stm_len: list[int] = []
        nstm_len: list[int] = []
        material: list[float] = []
        bucket: list[int] = []
        cp: list[float] = []
        with gzip.open(sp, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                s, n = r["stm"], r["nstm"]
                stm_strs.append(s)
                nstm_strs.append(n)
                stm_len.append(s.count(",") + 1)
                nstm_len.append(n.count(",") + 1)
                material.append(float(r["m"]))
                bucket.append(int(r["b"]))
                cp.append(float(r["cp"]))
                added += 1
                if limit and added >= limit:
                    break
        if stm_strs:
            # bulk-parse: one big np.fromstring per shard parses the CSV at
            # C-speed with only the joined string as intermediate (no
            # tens-of-millions-of-substrings list -> low peak memory).
            stm_flat = np.fromstring(",".join(stm_strs), dtype=np.int32,
                                     sep=",")
            nstm_flat = np.fromstring(",".join(nstm_strs), dtype=np.int32,
                                      sep=",")
            pool.add_bulk(stm_flat, np.asarray(stm_len, dtype=np.int64),
                          nstm_flat, np.asarray(nstm_len, dtype=np.int64),
                          material, bucket, cp)
        if limit and added >= limit:
            break
    return added


def score_loss(eval_cp: torch.Tensor, teacher_cp: torch.Tensor) -> torch.Tensor:
    """Identical to s12_train.score_loss."""
    pred = torch.sigmoid(eval_cp / SCORE_SCALE)
    tgt = torch.sigmoid(teacher_cp / SCORE_SCALE)
    return torch.mean((pred - tgt) ** 2)


def train_s14(
    dataset_dir: Path,
    encoded_glob: str,
    engine_bin: Path,
    seed: int,
    output_dir: Path,
    target_presentations: int = 20_000_000,
    max_seconds: float = 3600.0,
    batch_size: int = 1024,
    lr: float = 1e-3,
    final_lr_frac: float = 0.3 ** 5,
    weight_decay: float = 1e-5,
    val_every: int = 500,
    device_name: str | None = None,
    limit_fishtest: int = 0,
) -> dict:
    t_start = time.time()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # ---- data ----
    pool = FlatPool()
    print("[data] building canonical CP corpus track (engine export)...",
          flush=True)
    val_items, dataset_sha, n_canon = build_canonical(
        dataset_dir, engine_bin, pool)
    print(f"[data] canonical train={n_canon} val={len(val_items)}", flush=True)
    print(f"[data] loading Fishtest encoded shards from {encoded_glob}...",
          flush=True)
    n_fish = load_fishtest(encoded_glob, pool, limit=limit_fishtest)
    print(f"[data] fishtest positions={n_fish}", flush=True)
    pool.finalize(max_index=NNUE_INPUTS_V2R12)
    n_pool = pool.n
    print(f"[data] POOL total={n_pool} (canonical {n_canon} + fishtest "
          f"{n_fish}); stm_idx={pool.total_stm} nstm_idx={pool.total_nstm} "
          f"avg_feat/pos={(pool.total_stm)/max(1,n_pool):.1f}", flush=True)

    # validation (canonical Y16 only) -> resident on device
    val = EncodedSplit(val_items)
    v_stm_ind = val.stm_indices.to(device)
    v_stm_off = val.stm_offsets.to(device)
    v_nstm_ind = val.nstm_indices.to(device)
    v_nstm_off = val.nstm_offsets.to(device)
    v_buckets = val.buckets.to(device)
    v_teacher_cp = val.raw_cps.to(device)
    v_material = torch.tensor([it["material_cp_stm"] for it in val_items],
                              dtype=torch.float32, device=device)

    # ---- model / optimizer / schedule (S12-R0 verbatim) ----
    model = S12Model(num_inputs=NNUE_INPUTS_V2R12, ft_width=256).to(device)
    with torch.no_grad():
        model.ft_weights.weight[NNUE_V2R6_REL_BASE:].zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    planned_total_steps = max(1, math.ceil(target_presentations / batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(
            final_lr_frac,
            0.5 * (1 + math.cos(math.pi * step / planned_total_steps)),
        ),
    )
    print(f"[train] planned_total_steps={planned_total_steps} "
          f"batch={batch_size} device={device_name}", flush=True)

    def evaluate() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            out = model(v_stm_ind, v_stm_off, v_nstm_ind, v_nstm_off,
                        v_buckets)
            eval_cp = v_material + TARGET_SCALE * out
            v_loss = score_loss(eval_cp, v_teacher_cp).item()
            v_mae = torch.mean(torch.abs(eval_cp - v_teacher_cp)).item()
        model.train()
        return v_loss, v_mae

    g = torch.Generator()
    g.manual_seed(seed)
    history: list[dict] = []
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_step = -1
    best_state = None
    global_step = 0
    presentations = 0
    passes = 0
    budget_hit = None
    model.train()

    def maybe_checkpoint(step: int):
        nonlocal best_val_loss, best_val_mae, best_step, best_state
        v_loss, v_mae = evaluate()
        history.append({"step": step, "presentations": presentations,
                        "val_loss": v_loss, "val_mae": v_mae,
                        "lr": scheduler.get_last_lr()[0]})
        improved = v_loss < best_val_loss
        if improved:
            best_val_loss, best_val_mae, best_step = v_loss, v_mae, step
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"  step {step} pres {presentations} val {v_loss:.6f} "
              f"val_mae {v_mae:.1f} lr {scheduler.get_last_lr()[0]:.2e} "
              f"({time.time() - t_start:.0f}s)"
              f"{'  <= best' if improved else ''}", flush=True)

    while True:
        if presentations >= target_presentations:
            budget_hit = budget_hit or "presentations"
            break
        if time.time() - t_start >= max_seconds:
            budget_hit = budget_hit or "wall_clock"
            break
        perm = torch.randperm(n_pool, generator=g)
        passes += 1
        for start in range(0, n_pool, batch_size):
            if presentations >= target_presentations:
                budget_hit = budget_hit or "presentations"
                break
            if time.time() - t_start >= max_seconds:
                budget_hit = budget_hit or "wall_clock"
                break
            idx = perm[start:start + batch_size]
            stm_ind, stm_off = _gather_bag(pool.stm_flat, pool.stm_off, idx)
            nstm_ind, nstm_off = _gather_bag(pool.nstm_flat, pool.nstm_off,
                                             idx)
            stm_ind = stm_ind.to(device)
            stm_off = stm_off.to(device)
            nstm_ind = nstm_ind.to(device)
            nstm_off = nstm_off.to(device)
            buckets = pool.bucket[idx].to(device)
            material = pool.material[idx].to(device)
            teacher_cp = pool.cp[idx].to(device)

            optimizer.zero_grad()
            out = model(stm_ind, stm_off, nstm_ind, nstm_off, buckets)
            eval_cp = material + TARGET_SCALE * out
            loss = score_loss(eval_cp, teacher_cp)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            presentations += int(idx.numel())

            if global_step == 1 or global_step % val_every == 0:
                maybe_checkpoint(global_step)
        if budget_hit:
            break

    # final eval (guarantees best_state exists and captures the last state)
    maybe_checkpoint(global_step)

    assert best_state is not None, "no validation checkpoint captured"
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    restored_loss, restored_mae = evaluate()
    assert abs(restored_loss - best_val_loss) < 1e-6, \
        f"restored val-loss parity failure {restored_loss} != {best_val_loss}"

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_s14_v2r12_s{seed}.pt"
    elapsed = time.time() - t_start
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "summary": {
            "schema": "s12-bullet-recipe",   # export-compatible
            "experiment": "s14-data-supply",
            "feature_set": "v2r12",
            "seed": seed,
            "ft_width": 256,
            "num_inputs": NNUE_INPUTS_V2R12,
            "output_buckets": NUM_OUTPUT_BUCKETS,
            "bucket_rule": "(popcount_occ_incl_kings - 2) // 4",
            "activation": "screlu",
            "head": "linear(512, 8) bucket-select",
            "score": "eval_cp = material_cp_stm + 1000 * out",
            "loss": (f"mse(sigmoid(eval_cp/{SCORE_SCALE:.0f}), "
                     f"sigmoid(teacher_cp/{SCORE_SCALE:.0f}))"),
            "target_mode": "material-residual",
            "wdl_proportion": 0.0,
            "data_pool": {
                "canonical_train": n_canon,
                "fishtest": n_fish,
                "total": n_pool,
                "canonical_dataset_sha256": dataset_sha,
                "encoded_glob": encoded_glob,
            },
            "budget": {
                "target_presentations": target_presentations,
                "max_seconds": max_seconds,
                "planned_total_steps": planned_total_steps,
            },
            "model_selection_val": "canonical-Y16-validation-only",
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "budget_hit": budget_hit,
            "elapsed_seconds": elapsed,
            "steps": global_step,
            "presentations": presentations,
            "passes": passes,
        },
    }, ckpt_path)

    summary = {
        "schema": "s14_training_summary",
        "checkpoint": str(ckpt_path),
        "feature_set": "v2r12",
        "seed": seed,
        "data_pool": {"canonical_train": n_canon, "fishtest": n_fish,
                      "total": n_pool},
        "budget_hit": budget_hit,
        "target_presentations": target_presentations,
        "presentations": presentations,
        "steps": global_step,
        "passes": passes,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "elapsed_seconds": round(elapsed, 1),
        "device": device_name,
        "history": history,
    }
    (output_dir / f"training_summary_s14_v2r12_s{seed}.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "best_step", "best_val_loss", "best_val_mae", "budget_hit",
        "presentations", "steps", "passes", "elapsed_seconds")}, indent=1),
        flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path(r"data\s10\s10-eval-v2-1m01"))
    ap.add_argument("--encoded-glob", type=str,
                    default=r"data\s14\encoded\encoded-*.jsonl.gz")
    ap.add_argument("--engine", type=Path,
                    default=Path(r"target\release\eureka.exe"))
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--presentations", type=int, default=20_000_000)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--limit-fishtest", type=int, default=0,
                    help="smoke only: cap Fishtest positions loaded (0=all)")
    args = ap.parse_args()
    train_s14(
        args.dataset, args.encoded_glob, args.engine, args.seed, args.out,
        target_presentations=args.presentations, max_seconds=args.max_seconds,
        batch_size=args.batch_size, lr=args.lr, val_every=args.val_every,
        limit_fishtest=args.limit_fishtest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
