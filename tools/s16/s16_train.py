#!/usr/bin/env python3
"""S16 trainer: FT256 + a TRAINING-ONLY shared feature table.

Factorisation (Bullet-style shared factoriser; NOT a low-rank decomposition):

    V2 feature i has effective weight  W_eff[i] = W[i] + F[i % 704]

  * `W` = existing per-king-position table   [22528, 256]  (kept)
  * `F` = new cross-king-position shared table [ 704, 256]  (added)

`704 = 11 * 64`, so `i % 704` recovers exactly `channel * 64 + oriented_square`
from the ALREADY-oriented, already-mirrored V2 index
(`src/engine/nnue.rs::v2_feature_index = (bucket*11 + channel)*64 + sq`).
We reuse the existing index; board coordinates are NOT re-interpreted.

Frozen scope guards (owner):
  * R12 rows [22528, 23296) get NO shared term -- they train independently.
  * Both perspectives share ONE `F`; no shared bias is added.
  * FT width stays 256; export stays a single [23296, 256] table.
  * New training params: 704 * 256 = 180,224 (~3% of the base FT table).

Batch efficiency: `embedding_bag(mode="sum")` is additive over the bag, so

    sum_j ( W[i_j] + F[i_j % 704] )  ==  sum_j W[i_j]  +  sum_j F[i_j % 704]

=> two bags, added. No expanded [23296, 256] table is ever materialised.
R12 entries must contribute zero shared weight, so `F` has 705 rows and the
sentinel row 704 is pinned to zero and used for every `i >= 22528`.

Usage:
  python tools/s16/s16_train.py \
      --cache-glob "data/s15/cache/*.ekc" \
      --cache-glob "data/s15/cache-new/*.ekc" \
      --canonical-dir data/s15/cache-canonical \
      --presentations 400000000 --max-seconds 3600 --seed 20260915 \
      --out data/s16/run
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r"tools\s14")
sys.path.insert(0, r"tools\s15")
import s14_train  # noqa: E402
import s15_train  # noqa: E402
import nnue_cache  # noqa: E402

S12Model = s14_train.S12Model
_gather_bag = s14_train._gather_bag
score_loss = s14_train.score_loss
EncodedSplit = s14_train.EncodedSplit
val_items_from_shard = s15_train.val_items_from_shard

NNUE_INPUTS_V2R12 = s14_train.NNUE_INPUTS_V2R12      # 23296
V2_BASE = s14_train.NNUE_V2R6_REL_BASE               # 22528
TARGET_SCALE = s14_train.TARGET_SCALE
NUM_OUTPUT_BUCKETS = s14_train.NUM_OUTPUT_BUCKETS
FEATURE_SET = "v2r12"

SHARED_PERIOD = 11 * 64        # 704
SHARED_ROWS = SHARED_PERIOD + 1   # +1 sentinel
SENTINEL = SHARED_PERIOD       # 704


class S16Model(S12Model):
    """S12Model + zero-init shared table applied to V2 rows only.

    Two invariants this class must maintain (both were violated by an earlier
    revision of this file, and the export guard caught it):

      1. R12 rows must receive NO shared contribution. `_shared_indices`
         routes them to the sentinel row, so the sentinel row must be zero.
      2. The sentinel row must stay zero for the WHOLE run. `ft_shared` is one
         [705, 256] Parameter, so AdamW would otherwise apply weight decay and
         gradient updates to row 704 like any other row -- silently feeding a
         spurious shared term to every R12 feature for the entire run.
         `zero_sentinel_()` is called after every optimizer step.
    """

    def __init__(self, num_inputs: int, ft_width: int = 256):
        super().__init__(num_inputs, ft_width)
        if num_inputs != NNUE_INPUTS_V2R12:
            raise SystemExit("PIPELINE_FAILURE: S16 expects V2R12 inputs")
        if V2_BASE % SHARED_PERIOD != 0:
            raise SystemExit("PIPELINE_FAILURE: 22528 % 704 != 0")
        self.ft_shared = torch.nn.Parameter(
            torch.zeros(SHARED_ROWS, ft_width))
        self.zero_sentinel_()

    @torch.no_grad()
    def zero_sentinel_(self) -> None:
        """Re-pin the sentinel row to zero (call after every opt step)."""
        self.ft_shared[SENTINEL].zero_()

    @torch.no_grad()
    def sentinel_absmax(self) -> float:
        return float(self.ft_shared[SENTINEL].abs().max())

    def _shared_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.where(indices < V2_BASE,
                           indices.remainder(SHARED_PERIOD),
                           torch.full_like(indices, SENTINEL))

    def forward(self, stm_indices, stm_offsets, nstm_indices,
                nstm_offsets, buckets):
        stm_acc = (
            torch.nn.functional.embedding_bag(
                stm_indices, self.ft_weights.weight, stm_offsets,
                mode="sum")
            + torch.nn.functional.embedding_bag(
                self._shared_indices(stm_indices), self.ft_shared,
                stm_offsets, mode="sum")
            + self.ft_bias
        )
        nstm_acc = (
            torch.nn.functional.embedding_bag(
                nstm_indices, self.ft_weights.weight, nstm_offsets,
                mode="sum")
            + torch.nn.functional.embedding_bag(
                self._shared_indices(nstm_indices), self.ft_shared,
                nstm_offsets, mode="sum")
            + self.ft_bias
        )
        hidden = self.act(torch.cat([stm_acc, nstm_acc], dim=1))
        logits = self.l1(hidden)
        return logits.gather(1, buckets.view(-1, 1)).view(-1)

    @torch.no_grad()
    def merged_ft_table(self) -> torch.Tensor:
        """EXPORT-time semantic: W + repeat(F[:704], 32); R12 rows untouched."""
        merged = self.ft_weights.weight.detach().clone()
        reps = V2_BASE // SHARED_PERIOD
        merged[:V2_BASE] = merged[:V2_BASE] + self.ft_shared[:SHARED_PERIOD].repeat(reps, 1)
        return merged

    @torch.no_grad()
    def merged_forward(self, stm_indices, stm_offsets, nstm_indices,
                       nstm_offsets, buckets):
        merged = self.merged_ft_table()
        stm_acc = (torch.nn.functional.embedding_bag(
            stm_indices, merged, stm_offsets, mode="sum") + self.ft_bias)
        nstm_acc = (torch.nn.functional.embedding_bag(
            nstm_indices, merged, nstm_offsets, mode="sum") + self.ft_bias)
        hidden = self.act(torch.cat([stm_acc, nstm_acc], dim=1))
        logits = self.l1(hidden)
        return logits.gather(1, buckets.view(-1, 1)).view(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-glob", action="append", default=[])
    ap.add_argument("--canonical-dir", type=Path,
                    default=Path(r"data\s15\cache-canonical"))
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--presentations", type=int, default=400_000_000)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--ft-width", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ft_width != 256:
        raise SystemExit("PIPELINE_FAILURE: S16 freezes FT width at 256")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ---------------- data: byte-for-byte the S15 pool order -------------
    t_load = time.time()
    canon_train = args.canonical_dir / "train.ekc"
    canon_val = args.canonical_dir / "validation.ekc"
    if not canon_train.exists() or not canon_val.exists():
        raise SystemExit(f"FAIL CLOSED: canonical cache missing under "
                         f"{args.canonical_dir}")
    fish_paths: list[str] = []
    for g in args.cache_glob:
        fish_paths.extend(sorted(glob.glob(g)))
    if not fish_paths:
        raise SystemExit("FAIL CLOSED: no cache shards match --cache-glob")

    paths = [str(canon_train)] + fish_paths
    pool = nnue_cache.load_pool(paths, max_index=NNUE_INPUTS_V2R12,
                                device="cpu")
    load_s = time.time() - t_load
    print(f"[data] POOL total={pool.n} shards={len(paths)} "
          f"(canonical train + {len(fish_paths)} fishtest) "
          f"stm_idx={pool.total_stm} avg_feat/pos="
          f"{pool.total_stm/max(1,pool.n):.1f} load={load_s:.1f}s", flush=True)

    val_items, _vh = val_items_from_shard(canon_val)
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
    stm_flat, stm_off = pool.stm_flat, pool.stm_off
    nstm_flat, nstm_off = pool.nstm_flat, pool.nstm_off
    material_all, bucket_all, cp_all = pool.material, pool.bucket, pool.cp

    # ---------------- model ------------------------------------------
    model = S16Model(num_inputs=NNUE_INPUTS_V2R12, ft_width=args.ft_width)
    model = model.to(device)
    with torch.no_grad():
        model.ft_weights.weight[V2_BASE:].zero_()   # R12 sidecar zero-init
        model.ft_shared.zero_()                     # shared table zero-init

    n_shared = SHARED_PERIOD * args.ft_width
    n_base = NNUE_INPUTS_V2R12 * args.ft_width
    print(f"[model] FT width {args.ft_width} | base FT params {n_base:,} | "
          f"shared params {n_shared:,} ({100.0*n_shared/n_base:.2f}% of base)",
          flush=True)
    print(f"[model] shared applies to rows [0,{V2_BASE}); R12 "
          f"[{V2_BASE},{NNUE_INPUTS_V2R12}) has no shared term", flush=True)

    # ---- verification item 1: factorised vs merged float eval ----------
    nsc = min(4096, pool.n)
    ridx = torch.arange(nsc)
    rs_i, rs_o = _gather_bag(stm_flat, stm_off, ridx)
    rn_i, rn_o = _gather_bag(nstm_flat, nstm_off, ridx)
    rs_i, rs_o = rs_i.to(device), rs_o.to(device)
    rn_i, rn_o = rn_i.to(device), rn_o.to(device)
    rb = bucket_all[ridx].to(device)
    model.eval()
    with torch.no_grad():
        a = model(rs_i, rs_o, rn_i, rn_o, rb)
        b = model.merged_forward(rs_i, rs_o, rn_i, rn_o, rb)
        merge_diff = (a - b).abs().max().item()
    r12_rows = int((rs_i >= V2_BASE).sum().item() + (rn_i >= V2_BASE).sum().item())
    print(f"[check] factorised vs merged float eval: n={nsc} "
          f"max|diff|={merge_diff:.3e} (R12 rows in sample: {r12_rows})",
          flush=True)
    if not (merge_diff < 1e-5):
        raise SystemExit("PIPELINE_FAILURE: merge self-check failed")

    # shared table must be inert at init
    print(f"[check] shared table all-zero at init: "
          f"{bool((model.ft_shared == 0).all())}", flush=True)
    model.train()

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
          f"batch={args.batch_size} device={args.device}", flush=True)

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
            bs = idx.numel()
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
            # INVARIANT: the sentinel row must remain exactly zero, otherwise
            # every R12 feature silently receives a shared contribution.
            model.zero_sentinel_()
            scheduler.step()

            global_step += 1
            presentations += bs
            if global_step % args.val_every == 0:
                maybe_checkpoint(global_step)

    maybe_checkpoint(global_step)
    assert best_state is not None, "no validation checkpoint captured"
    # the sentinel invariant must hold in the SAVED state too
    if model.sentinel_absmax() != 0.0:
        raise SystemExit(
            "PIPELINE_FAILURE: sentinel row non-zero at save time "
            f"(absmax={model.sentinel_absmax():.3e}); R12 rows would have "
            "received an untrained shared contribution")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    v_loss, _ = evaluate()
    assert abs(v_loss - best_val_loss) < 1e-6, \
        f"restored val-loss parity failure {v_loss} != {best_val_loss}"

    elapsed = time.time() - t_start
    ckpt_path = args.out / f"checkpoint_s16_v2r12_s{args.seed}.pt"
    torch.save({
        "model_state_dict": {k: v.cpu()
                             for k, v in model.state_dict().items()},
        "summary": {
            "schema": "s16-shared-factoriser",
            "experiment": "s16-training-time-shared-ft",
            "feature_set": FEATURE_SET,
            "seed": args.seed,
            "ft_width": args.ft_width,
            "num_inputs": NNUE_INPUTS_V2R12,
            "shared_period": SHARED_PERIOD,
            "shared_rows": SHARED_PERIOD,
            "shared_sentinel_row": SENTINEL,
            "shared_params": n_shared,
            "base_ft_params": n_base,
            "shared_frac_of_base": round(100.0 * n_shared / n_base, 3),
            "shared_applies_to": f"[0,{V2_BASE})",
            "r12_rows_untouched": f"[{V2_BASE},{NNUE_INPUTS_V2R12})",
            "merge_rule": "merged[:22528] = W + repeat(F, 32); merged[22528:] = R12",
            "merge_selfcheck_max_abs_diff": merge_diff,
            "output_buckets": NUM_OUTPUT_BUCKETS,
            "activation": "screlu",
            "head": "linear(512, 8) bucket-select",
            "score": "eval_cp = material_cp_stm + 1000 * out",
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

    (args.out / f"training_history_s16_s{args.seed}.json").write_text(
        json.dumps(history) + "\n", encoding="utf-8")
    summary_out = {
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "budget_hit": budget_hit,
        "presentations": presentations,
        "steps": global_step,
        "passes": passes,
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.out / f"training_summary_s16_s{args.seed}.json").write_text(
        json.dumps(summary_out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary_out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
