#!/usr/bin/env python3
"""S17 trainer: FT512 (V2+R12, SCReLU, 8 material buckets, material residual).

This is the S12/S15 recipe with the FT width raised 256 -> 512 and NOTHING
else changed:

  * no shared feature table (S16's factoriser is NOT inherited -- a lower
    validation loss there does not earn it a place here);
  * same feature set (V2 + R12 relation rows), same head (SCReLU -> one
    selected material bucket), same target (material residual / CP-only);
  * same data: the existing 21M-position pool, reused as-is.

The only structural difference from FT256 is the lane count, which changes
the parameter count and the dense-head width:

    FT params  = 23296 * ft_width
    l1 weights = 8 * (2 * ft_width)

Usage:
    python tools/s17/s17_train.py --out data/s17/run \
        --presentations 400000000 --max-seconds 3600 --seed 20260917
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s12"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s14"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s15"))

import s12_train  # noqa: E402
import s14_train  # noqa: E402
import s15_train  # noqa: E402
import nnue_cache  # noqa: E402

S12Model = s12_train.S12Model
NUM_OUTPUT_BUCKETS = s12_train.NUM_OUTPUT_BUCKETS
TARGET_SCALE = s12_train.TARGET_SCALE                    # 1000.0
score_loss = s14_train.score_loss
_gather_bag = s14_train._gather_bag
EncodedSplit = s14_train.EncodedSplit
NNUE_INPUTS_V2R12 = s14_train.NNUE_INPUTS_V2R12          # 23296
NNUE_V2R6_REL_BASE = s14_train.NNUE_V2R6_REL_BASE        # 22528
val_items_from_shard = s15_train.val_items_from_shard


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-glob", action="append", default=[])
    ap.add_argument("--canonical-dir", type=Path,
                    default=Path("data/s15/cache-canonical"))
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--presentations", type=int, default=400_000_000)
    ap.add_argument("--max-seconds", type=float, default=3600.0)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--ft-width", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.ft_width not in (256, 512):
        raise SystemExit(f"unsupported ft-width {args.ft_width}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- data (reuse the existing pool; no conversion, no relabelling) ----
    globs = args.cache_glob or [
        "data/s15/cache/*.ekc", "data/s15/cache-new/*.ekc",
        "data/s15/cache-canonical/train.ekc"]
    paths = []
    for g in globs:
        paths.extend(sorted(Path(".").glob(g)))
    if not paths:
        raise SystemExit("no cache shards matched")
    t0 = time.time()
    pool = nnue_cache.load_pool(paths, max_index=NNUE_INPUTS_V2R12,
                                device="cpu")
    load_s = time.time() - t0
    print(f"[data] POOL total={pool.n} shards={len(paths)} "
          f"load={load_s:.1f}s", flush=True)

    canon_val = sorted(Path(args.canonical_dir).glob("*validation*.ekc"))
    if not canon_val:
        raise SystemExit(f"no validation shard under {args.canonical_dir}")
    val_items, _vh = val_items_from_shard(canon_val[0])
    val = EncodedSplit(val_items)
    print(f"[data] val items={val.n} (canonical Y16 only)", flush=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = S12Model(num_inputs=NNUE_INPUTS_V2R12,
                     ft_width=args.ft_width).to(dev)
    # R12 rows are zero at init (same as S15): they train from zero.
    with torch.no_grad():
        model.ft_weights.weight[NNUE_V2R6_REL_BASE:].zero_()
    n_ft = NNUE_INPUTS_V2R12 * args.ft_width
    n_l1 = NUM_OUTPUT_BUCKETS * 2 * args.ft_width
    print(f"[model] FT width {args.ft_width} | FT params {n_ft:,} "
          f"| l1 params {n_l1:,} | no shared table", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    planned_total_steps = max(1, args.presentations // args.batch_size)
    print(f"[train] planned_total_steps={planned_total_steps} "
          f"batch={args.batch_size} device={dev}", flush=True)

    pool_idx = torch.arange(pool.n)
    step = 0
    pres = 0
    passes = 0
    t_start = time.time()
    best_val = math.inf
    best_step = -1
    best_state = None
    best_mae = math.inf

    def lr_at(s: int) -> float:
        """Cosine to `final_lr_frac`, aligned to planned_total_steps."""
        frac = min(1.0, s / planned_total_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        final = args.lr * (0.3 ** 5)
        return final + (args.lr - final) * cos

    # Validation tensors hoisted once (canonical Y16 only), exactly as in
    # the S15 trainer.
    v_material = torch.tensor([it["material_cp_stm"] for it in val_items],
                              dtype=torch.float32, device=dev)
    v_stm_ind = val.stm_indices.to(dev)
    v_stm_off = val.stm_offsets.to(dev)
    v_nstm_ind = val.nstm_indices.to(dev)
    v_nstm_off = val.nstm_offsets.to(dev)
    v_buckets = val.buckets.to(dev)
    v_teacher_cp = val.raw_cps.to(dev)

    def evaluate() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            out = model(v_stm_ind, v_stm_off, v_nstm_ind, v_nstm_off,
                        v_buckets)
            eval_cp = v_material + TARGET_SCALE * out
            loss = score_loss(eval_cp, v_teacher_cp).item()
            mae = torch.mean(torch.abs(eval_cp - v_teacher_cp)).item()
        model.train()
        return loss, mae

    shuffle = torch.Generator().manual_seed(args.seed)
    # Batch source: permute ONCE per pass and walk it in slices, rather
    # than re-permuting the whole pool every step. Permuting 20M indices
    # per step dominates the step time (~50x slower) and buys nothing.
    perm = None
    cursor = 0
    while True:
        if pres >= args.presentations:
            print("[train] budget_hit presentations", flush=True)
            break
        if time.time() - t_start >= args.max_seconds:
            print("[train] budget_hit wall_clock", flush=True)
            break
        if perm is None or cursor + args.batch_size > perm.numel():
            perm = torch.randperm(pool.n, generator=shuffle)
            cursor = 0
            passes += 1
        idx = perm[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        s_i, s_o = _gather_bag(pool.stm_flat, pool.stm_off, idx)
        n_i, n_o = _gather_bag(pool.nstm_flat, pool.nstm_off, idx)
        b = pool.bucket[idx].to(dev)
        y = pool.cp[idx].float().to(dev)
        mat = pool.material[idx].float().to(dev)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        out = model(s_i.to(dev), s_o.to(dev), n_i.to(dev), n_o.to(dev), b)
        pred = mat + TARGET_SCALE * out
        loss = score_loss(pred, y)
        loss.backward()
        opt.step()

        step += 1
        pres += args.batch_size
        if step % 5000 == 0:
            print(f"  step {step} pres {pres} passes {passes} "
                  f"loss {float(loss):.6f} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

        if step % args.val_every == 0:
            vl, vm = evaluate()
            tag = ""
            if vl < best_val:
                best_val = vl
                best_mae = vm
                best_step = step
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                tag = "  <= best"
            if step % 5000 == 0 or tag:
                print(f"  step {step} pres {pres} val {vl:.7f} "
                      f"mae {vm:.1f} lr {lr_at(step):.2e} "
                      f"({time.time()-t_start:.0f}s){tag}", flush=True)

    elapsed = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = args.out / f"checkpoint_s17_ft{args.ft_width}_s{args.seed}.pt"
    summary = {
        "schema": "s12-bullet-recipe",
        "stage": "s17-ft512" if args.ft_width == 512 else "s17-ft256",
        "seed": args.seed,
        "ft_width": args.ft_width,
        "num_inputs": NNUE_INPUTS_V2R12,
        "shared_table": False,
        "best_step": best_step,
        "best_val_loss": best_val,
        "best_val_mae": best_mae,
        "presentations": pres,
        "steps": step,
        "passes": pres / pool.n,
        "planned_total_steps": planned_total_steps,
        "elapsed_seconds": elapsed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    torch.save({"model_state_dict": model.state_dict(),
                "summary": summary}, ckpt_path)
    info = dict(summary, checkpoint=str(ckpt_path))
    (args.out / "summary.json").write_text(
        json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
