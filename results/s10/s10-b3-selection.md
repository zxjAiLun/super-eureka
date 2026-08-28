# S10-B3 Stage 1 — Frozen Selection (three-seed V2 training on 300k)

**Status: STAGE 1 COMPLETE — SELECTION FROZEN**
**Holdout NOT evaluated at this point.** `holdout_observed_before_selection = false`

## Frozen inputs

- dataset: `s10-eval-v1-300k01` (SHA `503b47b6...076b3`)
- labels: 300k Stockfish-18 teacher labels (SHA `bcd49da1...16116f`)
- teacher provenance: verified against frozen S10-B2 contract in
  `train_nnue.py::verify_teacher_contract()` (engine, binary SHA, nodes=16384,
  Threads=1/Hash=64/MultiPV=1/UCI_ShowWDL=true, audit ok / fresh-second-pass /
  1000 checked / 0 mismatches)
- feature set: v2 (22528 sparse inputs)
- environment: CUDA (RTX 4060 Laptop), torch 2.13.0+cu130, deterministic
  algorithms on, CUBLAS_WORKSPACE_CONFIG=:4096:8, exporter
  `target/release/eureka` (SHA `d6649098...d02abe`)
- seeds: 20260818, 20260819, 20260820 (same seeds as S10-A)
- recipe (unchanged, frozen): 128-32-32-1, AdamW lr=1e-3 wd=1e-5,
  SmoothL1 beta=0.1, batch=256, max_epochs=100, patience=15, checkpoint by
  minimum validation SmoothL1

## Usable data (mate-only positions excluded, per frozen recipe)

```
records_total:               300000
records with teacher_cp_stm: 292322
records with mate only:        7678  (2.56%)

usable:    train 233823 | validation 29270 | holdout 29229
excluded:  train   6177 | validation   730 | holdout   771
```

Exclusion rate 2.56% is small; no anomalous data loss.

## Three-seed validation results

```
seed      best_epoch  total  best_val_loss  best_val_mae_cp  time
20260818      6        21    0.12383734     165.3246         82.1s
20260819      6        21    0.12414581     165.5692         82.5s
20260820      6        21    0.12495897     166.4741         84.2s

median val MAE: 165.5692 cp
min/max:        165.3246 / 166.4741 cp  (spread 1.1495 cp)
median best epoch: 6
median training time: 82.5 s
```

All three seeds converged normally, no NaN/Inf, no abnormal drift
(spread ~1.15 cp on ~165 cp).

## Selection rule (frozen, version 2)

```
production candidate = seed with minimum best_val_mae
                     = 20260818
                     = checkpoint_v2_s20260818.pt
                     = SHA256 d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7
```

Per-seed checkpoint selection remains minimum validation SmoothL1
(unchanged from S10-A).

## Coverage diagnostics (selected seed, train 233823)

```
train_observed_unique:            18012 / 22528  (79.95%)
singleton count / rate:           1550 / 8.61%
<=5 count / rate:                 4733 / 26.28%

val unseen activation rate:       0.0178%
val positions with unseen rate:   0.509%
```

Comparison against S10-A (21k dataset, descriptive only):

```
                          S10-A (21k)     S10-B3 (300k)
val unseen activation     0.511%          0.0178%   (29x lower)
positions with unseen     10.128%         0.509%    (20x lower)
median val MAE            238.85 cp       165.57 cp
```

Coverage improved dramatically with the 300k dataset, exactly the expected
signal of the 12x training-data increase.

## Blindness chronology

1. B3-0 trainer hardening committed (teacher-contract gate, Stage-1 holdout
   exclusion, preflight mode).
2. B3-1 preflight run committed.
3. Three training runs completed with `allow_holdout=False`; holdout split
   was never exported, never encoded, never evaluated
   (`holdout_observed: false` in all three summaries; usable holdout count 0).
4. This selection artifact frozen and committed BEFORE any holdout
   evaluation.
