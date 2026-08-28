# S10-B3 Closeout — V2 300k Training + Blind Holdout

**Status: CLOSED / PASS**

## Single production candidate (frozen)

```
selected seed:            20260818
selected checkpoint:      data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt
checkpoint SHA256:        d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7

dataset SHA256:           503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3
labels SHA256:            bcd49da1ece75a15591e135d5bcf6d036608b1759d6a00e639f3e344e516116f

architecture:             22528 sparse inputs -> FT 128 -> concat 256 -> 32 -> 32 -> 1
                          (ClippedReLU, AdamW lr=1e-3 wd=1e-5, SmoothL1 beta=0.1,
                           batch=256, max_epochs=100, patience=15)
feature_set:              v2
```

Checkpoint `.pt` is local-only (data/s10/ is gitignored); its SHA256 is
recorded here and in s10-b3-selection.json.

## Results

```
selected validation loss: 0.12383734   MAE: 165.3246 cp   (best_epoch 6/21)
selected holdout loss:    0.12451087   MAE: 166.0762 cp   (29229 positions)

holdout / validation ratio: 1.0045   (gate <= 1.15)  -> PASS
delta:                      +0.7515 cp
```

Three-seed summary (validation only, selection by min best_val_mae):

```
seed      best_epoch  best_val_loss  best_val_mae_cp  time    checkpoint SHA256
20260818      6        0.12383734     165.3246         82.1s   d59ad852...   SELECTED
20260819      6        0.12414581     165.5692         82.5s   5b40ffc2...
20260820      6        0.12495897     166.4741         84.2s   bb8bbad4...

median val MAE 165.5692 cp | spread 1.1495 cp | median best epoch 6 | median time 82.5 s
All seeds converged normally, no NaN/Inf, no abnormal drift.
```

## Coverage telemetry (selected seed; S10-A 21k reference, descriptive only)

```
                              S10-A (21k)     S10-B3 (300k)
train observed unique         ~ (not in ref)  18012 / 22528 (79.95%)
singleton rate                -               8.61%
<=5 rate                      -               26.28%
val unseen activation rate    0.511%          0.0178%    (29x better)
val positions with unseen     10.128%         0.509%     (20x better)
median val MAE                238.85 cp       165.57 cp
```

The 12x training-data growth produced exactly the expected coverage
improvement — no anomalies.

## Usable data accounting

```
records_total 300000 | with teacher_cp_stm 292322 | mate-only excluded 7678 (2.56%)
usable:     train 233823  validation 29270  holdout 29229
mate-only:  train   6177  validation   730  holdout   771
```

Mate-target redesign was explicitly out of scope for B3 (mate-only
positions excluded per frozen S10-A recipe).

## Environment identity

```
device:      cuda — NVIDIA GeForce RTX 4060 Laptop GPU (all three seeds)
torch:       2.13.0+cu130   CUDA 13.0
deterministic_algorithms: true    CUBLAS_WORKSPACE_CONFIG=:4096:8
exporter:    target/release/eureka (SHA256 d6649098fe5f47b335454a54cddb57c64afcda157d20cf8a279e4c2f23d02abe)
python env:  .venv-s10 (uv cache-linked; torch wheel identical to S10-A version string)
teacher gate: verify_teacher_contract() fail-closed on engine/binary SHA/nodes/
              options/labeled_positions/audit/labels SHA vs frozen S10-B2 contract
```

## Holdout blindness chronology

```
ad2fb64  B2 closeout (labels published)
   ↓
trainer hardening (teacher gate, stage-1 holdout exclusion, preflight)
three runs with allow_holdout=False:
   holdout never exported, never encoded, never evaluated
   (usable_split_counts.holdout = 0, holdout_observed = false)
   ↓
bb6383c  Stage-1 selection commit (min best_val_mae -> seed 20260818)
   ↓  ONLY AFTER selection was committed:
eval_nnue_checkpoint.py loaded the frozen checkpoint (SHA-verified, its
embedded dataset/labels SHA verified, holdout_observed=false enforced),
exported ONLY holdout features, performed zero optimizer steps, and
evaluated the single selected checkpoint:
   holdout MAE 166.0762 cp  ->  ratio 1.0045 <= 1.15  ->  PASS
```

## Offline gate verdict

```
3 seeds normal convergence ........ PASS (no NaN/Inf, spread 1.15 cp)
holdout <= 1.15 x validation ...... PASS (1.0045)
coverage vs S10-A ................ dramatically improved (expected)
```

**No Arena in B3** — NNUE is not yet integrated into the Rust production
eval; next stage is S10-B4/C: FP32 checkpoint -> Rust inference integration
-> quantization / model format -> NNUE-only eval path -> NPS -> Arena.
