# S6-N1 — NNUE Learnability Probe

STATUS: **COMPLETE — LEARNABLE_BUT_DATA_STARVED**

## Provenance

```text
dataset:    s6-eval-v1-core-shard01, 5919 records (train 4776 / val 500 / holdout 643)
dataset SHA: 3a3483fd46fd5a570c4c62b7d93378efc80eafbab43ec155db5ac5894fbc6a9d
labels SHA:  78dd8d52a34d1dd10a5d09cb3295be8f3a91a495d808fbd8b0cb68d31d668aa5
checkpoint:  data/s6/models/s6-n1-probe.pt (local-only)
checkpoint SHA256: 1a0f2d3b0010ff0eb8d0689d47e963ec7070e3dc17d3dab0feadfdfd6b307810
engine git:  deeb828cca9561192c364f9f3f813661e77f110a (trainer source commit)
verify_dataset: VERIFY_PASS
```

## Environment & config

```text
python 3.12.13, torch 2.13.0+cpu, numpy 2.5.2, python-chess 1.11.2, device cpu
architecture: 40960x32 shared table + 32 bias + sum + relu + STM concat + 64->1
init: pytorch default under torch.manual_seed(20260818)
loss: SmoothL1 beta=0.1; target = clip(teacher_cp_stm, +-2000)/1000
AdamW lr=1e-3 wd=1e-5, batch 256, max 100 epochs, patience 15
```

## Training

```text
rows: train 4752 / val 497 / holdout 642 (null-CP excluded)
epochs run: 18 (early stopped), best epoch 3
best val loss 0.100804, train loss at best epoch 0.076280
final train loss 0.015629 -> overfit gap 0.024524 (best epoch), strong late overfit
elapsed: 8.0 s
```

## Feature coverage

| split | white unique | black unique | union | union/40960 | unseen rate |
|---|---:|---:|---:|---:|---:|
| train | 3979 | 3616 | 5100 | 12.45% | - |
| validation | 1230 | 1259 | 1821 | 4.45% | 1.63% (144/500 positions) |
| holdout | 1510 | 1438 | 2189 | 5.34% | 1.67% (201/643 positions) |

Only ~5100 of 40960 features are ever active; 4752 training rows cover
~1.06 rows per unique feature.

## Metrics (clipped +-2000cp MAE / RMSE)

### Validation (497)

| predictor | raw MAE | clipped MAE | clipped RMSE |
|---|---:|---:|---:|
| zero | 159.79 | 159.79 | 219.48 |
| classical CurrentFinal | 168.61 | 168.61 | 242.75 |
| NNUE probe | 158.82 | 158.82 | 212.38 |

### Holdout (642)

| predictor | raw MAE | clipped MAE | clipped RMSE |
|---|---:|---:|---:|
| zero | 151.04 | 151.04 | 205.49 |
| classical CurrentFinal | 149.72 | 149.72 | 227.67 |
| NNUE probe | 154.39 | 154.39 | 208.00 |

NNUE prediction stats (val/holdout): mean 14.9/18.7 cp, std 152/146 cp,
min -523/-496, max 536/474.

## Classification

**LEARNABLE_BUT_DATA_STARVED** — the probe learns (train loss collapses,
validation NNUE beats both zero and classical), but the 4752-row dataset
cannot support a 40960x32 feature table: holdout NNUE does not beat zero or
classical, unseen-feature rate is low, and the model overfits almost
immediately (best epoch 3). This is a data-scale verdict, not a pipeline
failure: export, join, coverage, and metric plumbing all passed.

## Next options (direction only)

- enlarge training data before re-probing, or
- shrink the probe (lower-width / subset encoding) on the same data to
  isolate whether signal survives at realistic data scale.
