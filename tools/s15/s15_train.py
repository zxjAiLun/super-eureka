#!/usr/bin/env python3
"""S15 trainer: identical recipe to S14, but reading the binary cache.

The recipe is UNCHANGED from S14 (and therefore from S12-R0):
  V2+R12 inputs (23296), FT256, SCReLU, 8 MaterialCount output buckets,
  material-residual head (eval_cp = material_cp_stm + 1000 * out),
  CP-only MSE( sigmoid(eval_cp/400), sigmoid(teacher_cp/400) ),
  batch 1024, AdamW lr 1e-3 wd 1e-5, cosine decay, R12 sidecar zero-init.
  Model selection on the canonical Y16 validation split ONLY.

What changes is the I/O path only: the pool is assembled from binary cache
shards (`tools/s15/nnue_cache.py`) instead of gzip'd JSONL, and the
canonical track is read from its own cache instead of being re-exported
through the engine on every run.

Equivalence is verifiable: with the same seed and the same pool contents
in the same order (canonical train first, then Fishtest in shard order),
the step-1 validation loss must equal the S14 run's logged 0.017029.

Usage:
  python tools/s15/s15_train.py --out data/s15/run --seed 20260915 \
      --presentations 60000000 --max-seconds 3600
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"tools\s14")
sys.path.insert(0, r"tools\s15")
import s14_train  # noqa: E402  (model, loss, gather -- frozen recipe)
import nnue_cache  # noqa: E402

S12Model = s14_train.S12Model
EncodedSplit = s14_train.EncodedSplit
score_loss = s14_train.score_loss
_gather_bag = s14_train._gather_bag
TARGET_SCALE = s14_train.TARGET_SCALE
SCORE_SCALE = s14_train.SCORE_SCALE
NNUE_INPUTS_V2R12 = s14_train.NNUE_INPUTS_V2R12
NNUE_V2R6_REL_BASE = s14_train.NNUE_V2R6_REL_BASE
NUM_OUTPUT_BUCKETS = s14_train.NUM_OUTPUT_BUCKETS

FEATURE_SET = "v2r12"


def val_items_from_shard(path: Path) -> tuple[list[dict], dict]:
    d = nnue_cache.read_shard(path)
    h = d["header"]
    if h["feature_set"] != FEATURE_SET:
        raise SystemExit(
            f"FAIL CLOSED: val cache declares {h['feature_set']}")
    stm_flat, nstm_flat = d["stm_flat"], d["nstm_flat"]
    stm_off, nstm_off = d["stm_off"], d["nstm_off"]
    items = []
    for i in range(int(h["n"])):
        cp = float(d["cp"][i])
        items.append({
            "stm": stm_flat[stm_off[i]:stm_off[i + 1]].tolist(),
            "nstm": nstm_flat[nstm_off[i]:nstm_off[i + 1]].tolist(),
            "target_scaled": cp / TARGET_SCALE,
            "target_cp": cp,
            "material_cp_stm": float(d["material"][i]),
            "bucket": int(d["bucket"][i]),
        })
    return items, h


def train(args) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    t_load = time.time()
    canon_train = args.canonical_dir / "train.ekc"
    canon_val = args.canonical_dir / "validation.ekc"
    if not canon_train.exists() or not canon_val.exists():
        raise SystemExit(
            f"FAIL CLOSED: canonical cache missing under {args.canonical_dir} "
            f"(run tools/s15/s15_cache_canonical.py first)")

    fish_paths: list[str] = []
    for g in args.cache_glob:
        fish_paths.extend(sorted(glob.glob(g)))
    if not fish_paths:
        raise SystemExit(f"FAIL CLOSED: no cache shards match {args.cache_glob}")

    # Pool order mirrors S14: canonical train first, then Fishtest in
    # deterministic shard order.
    paths = [str(canon_train)] + fish_paths
    pool = nnue_cache.load_pool(paths, max_index=NNUE_INPUTS_V2R12,
                                device="cpu")
    load_s = time.time() - t_load
    print(f"[data] POOL total={pool.n} shards={len(paths)} "
          f"(canonical train + {len(fish_paths)} fishtest) "
          f"stm_idx={pool.total_stm} avg_feat/pos="
          f"{pool.total_stm/max(1,pool.n):.1f} load={load_s:.1f}s", flush=True)

    val_items, vh = val_items_from_shard(canon_val)
    val = EncodedSplit(val_items)
    v_stm_ind = val.stm_indices.to(device)
    v_stm_off = val.stm_offsets.to(device)
    v_nstm_ind = val.nstm_indices.to(device)
    v_nstm_off = val.nstm_offsets.to(device)
    v_buckets = val.buckets.to(device)
    v_teacher_cp = val.raw_cps.to(device)
    v_material = torch.tensor([it["material_cp_stm"] for it in val_items],
                              dtype=torch.float32, device=device)
    print(f"[data] val items={val.n} (canonical Y16 only)", flush=True)

    n_pool = pool.n
    stm_flat = pool.stm_flat
    stm_off = pool.stm_off
    nstm_flat = pool.nstm_flat
    nstm_off = pool.nstm_off
    material_all = pool.material
    bucket_all = pool.bucket
    cp_all = pool.cp

    model = S12Model(num_inputs=NNUE_INPUTS_V2R12, ft_width=args.ft_width)
    model = model.to(device)
    with torch.no_grad():
        model.ft_weights.weight[NNUE_V2R6_REL_BASE:].zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    planned_total_steps = max(1, math.ceil(args.presentations /
                                          args.batch_size))
    final_lr_frac = 0.3 ** 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(final_lr_frac,
                         0.5 * (1 + math.cos(math.pi * step /
                                             planned_total_steps))))
    print(f"[train] planned_total_steps={planned_total_steps} "
          f"batch={args.batch_size} device={device}", flush=True)

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = model(v_stm_ind, v_stm_off, v_nstm_ind, v_nstm_off,
                        v_buckets)
            eval_cp = v_material + TARGET_SCALE * out
            loss = score_loss(eval_cp, v_teacher_cp).item()
            mae = torch.mean(torch.abs(eval_cp - v_teacher_cp)).item()
        model.train()
        return loss, mae

    g = torch.Generator()
    g.manual_seed(args.seed)
    history: list[dict] = []
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_step = -1
    best_state = None
    global_step = 0
    presentations = 0
    passes = 0
    budget_hit = None
    t_start = time.time()
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
              f"({time.time()-t_start:.0f}s)"
              f"{'  <= best' if improved else ''}", flush=True)

    while True:
        if presentations >= args.presentations:
            budget_hit = budget_hit or "presentations"
            break
        if time.time() - t_start >= args.max_seconds:
            budget_hit = budget_hit or "wall_clock"
            break
        perm = torch.randperm(n_pool, generator=g)
        passes += 1
        for start in range(0, n_pool, args.batch_size):
            if presentations >= args.presentations:
                budget_hit = budget_hit or "presentations"
                break
            if time.time() - t_start >= args.max_seconds:
                budget_hit = budget_hit or "wall_clock"
                break
            idx = perm[start:start + args.batch_size]
            s_ind, s_off = _gather_bag(stm_flat, stm_off, idx)
            n_ind, n_off = _gather_bag(nstm_flat, nstm_off, idx)
            s_ind, s_off = s_ind.to(device), s_off.to(device)
            n_ind, n_off = n_ind.to(device), n_off.to(device)
            buckets = bucket_all[idx].to(device)
            material = material_all[idx].to(device)
            teacher_cp = cp_all[idx].to(device)

            optimizer.zero_grad()
            out = model(s_ind, s_off, n_ind, n_off, buckets)
            eval_cp = material + TARGET_SCALE * out
            loss = score_loss(eval_cp, teacher_cp)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            presentations += int(idx.numel())

            if global_step == 1 or global_step % args.val_every == 0:
                maybe_checkpoint(global_step)
        if budget_hit:
            break

    maybe_checkpoint(global_step)
    assert best_state is not None, "no validation checkpoint captured"
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    restored_loss, _ = evaluate()
    assert abs(restored_loss - best_val_loss) < 1e-6, \
        f"restored val-loss parity failure {restored_loss} != {best_val_loss}"

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / f"checkpoint_s15_v2r12_s{args.seed}.pt"
    elapsed = time.time() - t_start
    torch.save({
        "model_state_dict": {k: v.cpu()
                             for k, v in model.state_dict().items()},
        "summary": {
            "schema": "s12-bullet-recipe",
            "experiment": "s15-data-expansion",
            "feature_set": FEATURE_SET,
            "seed": args.seed,
            "ft_width": args.ft_width,
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
            "data_pool": {"total": n_pool,
                          "fishtest_shards": len(fish_paths),
                          "cache_globs": list(args.cache_glob),
                          "pool_load_seconds": round(load_s, 1)},
            "budget": {"target_presentations": args.presentations,
                       "max_seconds": args.max_seconds,
                       "planned_total_steps": planned_total_steps},
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
        "schema": "s15_training_summary",
        "checkpoint": str(ckpt_path),
        "feature_set": FEATURE_SET,
        "seed": args.seed,
        "data_pool": {"total": n_pool, "fishtest_shards": len(fish_paths),
                      "pool_load_seconds": round(load_s, 1)},
        "budget_hit": budget_hit,
        "target_presentations": args.presentations,
        "presentations": presentations,
        "steps": global_step,
        "passes": passes,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
        "history": history,
    }
    (args.out / f"training_summary_s15_v2r12_s{args.seed}.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "best_step", "best_val_loss", "best_val_mae", "budget_hit",
        "presentations", "steps", "passes", "elapsed_seconds")}, indent=1),
        flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-glob", action="append", default=None,
                    help="repeatable; Fishtest cache shard glob")
    ap.add_argument("--canonical-dir", type=Path,
                    default=Path(r"data\s15\cache-canonical"))
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--presentations", type=int, default=60_000_000)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--ft-width", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    if not args.cache_glob:
        args.cache_glob = [r"data\s15\cache\*.ekc"]
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
