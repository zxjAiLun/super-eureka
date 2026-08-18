# S6-N1 — NNUE Learnability Probe (Repair 2)

STATUS: **MEASUREMENT_COMPLETE / CLOUD_VERDICT_PENDING**

```text
supersedes: 98bcdf01a7080325beb655c6f33b57f99806619f
  reason:    invalid measurement — evaluated the FINAL epoch state instead of
             the best-epoch state (98bcdf0 kept as historical record)
```

## Provenance

```text
dataset:    s6-eval-v1-core-shard01, 5919 records (train 4776 / val 500 / holdout 643)
dataset SHA: 3a3483fd46fd5a570c4c62b7d93378efc80eafbab43ec155db5ac5894fbc6a9d
labels SHA:  78dd8d52a34d1dd10a5d09cb3295be8f3a91a495d808fbd8b0cb68d31d668aa5
checkpoint:  data/s6/models/s6-n1-probe.pt (local-only)
checkpoint SHA256: 6bfdba6d7d9cc034d55d8bfe433ebb3b0d6f48d78afa2351f3ef465ac9003a66
engine git:  849682ee93e51c85d828cbb099591101c506ddc9 (fix commit; trainer source)
engine binary SHA256: d7e5a78c559dd0e6cb7ce14140b4ebc9d7148abe408bd5691dd6595d8472c86c
verify_dataset: VERIFY_PASS
```

## Environment & frozen config (unchanged)

```text
python 3.12.13, torch 2.13.0+cpu, numpy 2.5.2, python-chess 1.11.2, device cpu
architecture: 40960x32 shared table + 32 bias + sum + relu + STM concat + 64->1
init: pytorch default under torch.manual_seed(20260818)   seed=20260818
loss: SmoothL1 beta=0.1; target = clip(teacher_cp_stm, +-2000)/1000
AdamW lr=1e-3 wd=1e-5, batch 256, max 100 epochs, patience 15
```

## Best-state restoration (P1 verified)

```text
best_state_restored:              true
best epoch:                       3 (of 18 run, early stopped)
best_val_loss:                    0.100804
restored_validation_loss:         0.100804  (== best, strict tolerance)
checkpoint_roundtrip_validation_loss: 0.100804  (fresh model loaded from disk)
train_loss_at_best_epoch:         0.076280
final_train_loss:                 0.015629 (NOT the evaluated state)
overfit_gap (best epoch):         0.024524
elapsed:                          4.2 s
```

All reported metrics use the DISK-LOADED best-epoch checkpoint.

## Usable-row coverage (null-CP rows excluded)

| split | usable | total activations | unseen activations | unseen rate | positions w/ unseen |
|---|---:|---:|---:|---:|---:|
| train | 4752 | 208908 | - | - | - |
| validation | 497 | 22030 | 349 | 1.584% | 146 (29.4%) |
| holdout | 642 | 27544 | 459 | 1.666% | 201 (31.3%) |

Train activation frequency (real counts, not inference):

```text
union unique features:    5019 / 40960 (12.25%)
unobserved features:      35941
mean activations/feature: 41.6
median:                   4
p10 / p90:                1 / 52
singleton features:       1221
features with <=5 activations: 2861
```

## Metrics (vs RAW teacher CP; clipped metrics clamp both sides to +-2000)

### Validation (497)

| predictor | raw MAE | clipped MAE | clipped RMSE |
|---|---:|---:|---:|
| zero | 159.79 | 159.79 | 219.48 |
| classical CurrentFinal | 168.61 | 168.61 | 242.75 |
| NNUE probe (best ckpt) | 141.50 | 141.50 | 194.73 |

### Holdout (642)

| predictor | raw MAE | clipped MAE | clipped RMSE |
|---|---:|---:|---:|
| zero | 151.04 | 151.04 | 205.49 |
| classical CurrentFinal | 149.72 | 149.72 | 227.67 |
| NNUE probe (best ckpt) | 141.59 | 141.59 | 191.14 |

### Signals (clipped MAE deltas)

| split | NNUE vs zero | NNUE vs classical |
|---|---:|---:|
| validation | -18.28 cp (-11.44%) | -27.11 cp (-16.08%) |
| holdout | -9.45 cp (-6.26%) | -8.14 cp (-5.43%) |

Holdout bucket MAE (raw |teacher| magnitude): 0-100: 71.7; 100-300: 139.1;
300-1000: 350.9; >=1000: no rows.

NNUE prediction stats (val/holdout): mean 11.0/12.9 cp, std 143/142 cp.

## Verdict

**CLOUD_VERDICT_PENDING** — measurement only. The best-epoch probe now
outperforms both the zero predictor and CurrentFinal classical on validation
AND holdout clipped MAE (holdout -5.4% vs classical), with checkpoint
provenance verified end-to-end. Data remains sparse (5019/40960 features,
2861 features with <=5 activations); whether this counts as a strong early
signal is left to review.
