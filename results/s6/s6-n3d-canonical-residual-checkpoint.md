# S6-N3D - Canonical Residual Checkpoint (width 16, seed 20260818)

STATUS: **CANONICAL_CHECKPOINT_COMPLETE**

ROLE: **canonical residual checkpoint; existing-split report is diagnostic only and is NOT a confirmation gate**

## Provenance

```text
run git:            ab6aef921a838594616d0d8099e5453b2a26a70a
trainer blob:       2a7f5a31814e53df90cde61585af90c3c411e9e743ceefee149d3b43ce04accb
this script blob:   dbdd8bae5641c020ec89995e96f58f9c023f07c808cbff1b13b70f7410eb0cec
engine binary:      05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66
dataset:            5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af
labels:             e6f036f426db8a5fffc6c28baa6ae5333b0fe441bd9eec13f56d4dda989896d9
classical cache:    c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727
teacher binary:     6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9
```

## Canonical checkpoint

```text
path:      /media/bailan/DISK/AUbuntuProject/project/chessenginedemo/data/s6/models/s6-n3d-residual-w16-s20260818.pt
sha256:    5033d47cb101d96057e13aae9d3819d48fa8079e90bda8eae8cd935ac1006c55
target:    classical_residual (clamp(teacher_cp_stm - base_eval_stm, -2000, 2000) / 1000)
inference: base_eval_stm + residual * 1000
width:     16
seed:      20260818
best epoch/loss: 1 / 0.117962
roundtrip validation loss: 0.117962
```

## N3C control-E reproduction

| split | reference MAE | this run MAE | delta |
|---|---:|---:|---:|
| validation | 159.685 | 159.685 | 0.0 |
| holdout | 159.071 | 159.071 | 0.0 |

reproduction within 0.001 cp: **True**

## Exploratory comparison on EXISTING N3B splits

Diagnostic only - NOT a confirmation gate (same game pool as the residual target).

| split | n | classical MAE | residual MAE | improvement | classical RMSE | residual RMSE |
|---|---:|---:|---:|---:|---:|---:|
| validation | 1955 | 167.588 | 159.685 | 4.7157% | 242.686 | 229.905 |
| holdout | 2000 | 170.036 | 159.071 | 6.4486% | 242.526 | 227.747 |

### Paired per-position absolute-error delta

`abs(residual - teacher) - abs(classical - teacher)`; negative is better.

| split | n | mean | median | improved | worsened | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| validation | 1955 | -7.902286 | -5.682241 | 1085 | 870 | [-10.35199, -5.416735] |
| holdout | 2000 | -10.965772 | -8.830359 | 1146 | 854 | [-13.372268, -8.51167] |

### By source family

| split | group | n | classical MAE | residual MAE | delta | mean paired delta |
|---|---|---:|---:|---:|---:|---:|
| validation | arena | 840 | 186.892 | 178.812 | -8.08 | -8.079328 |
| validation | lichess-standard-rated-v1 | 1115 | 153.045 | 145.276 | -7.769 | -7.768909 |
| holdout | arena | 1007 | 166.677 | 156.941 | -9.736 | -9.736485 |
| holdout | lichess-standard-rated-v1 | 993 | 173.443 | 161.231 | -12.212 | -12.212391 |

### By phase bucket

| split | group | n | classical MAE | residual MAE | delta | mean paired delta |
|---|---|---:|---:|---:|---:|---:|
| validation | high | 673 | 144.663 | 136.133 | -8.53 | -8.529205 |
| validation | mid | 712 | 187.388 | 177.319 | -10.069 | -10.068271 |
| validation | low | 531 | 159.354 | 154.631 | -4.723 | -4.722923 |
| validation | zero | 39 | 313.821 | 312.991 | -0.83 | -0.82906 |
| holdout | high | 672 | 142.705 | 128.196 | -14.509 | -14.509124 |
| holdout | mid | 752 | 196.827 | 183.522 | -13.305 | -13.304972 |
| holdout | low | 516 | 154.636 | 151.237 | -3.399 | -3.399152 |
| holdout | zero | 60 | 272.817 | 265.781 | -7.036 | -7.035194 |

### By |teacher CP| bucket

| split | group | n | classical MAE | residual MAE | delta | mean paired delta |
|---|---|---:|---:|---:|---:|---:|
| validation | 0-100 | 684 | 95.433 | 96.068 | 0.635 | 0.635531 |
| validation | 100-300 | 528 | 142.223 | 132.677 | -9.546 | -9.546086 |
| validation | 300-1000 | 730 | 247.951 | 233.573 | -14.378 | -14.377824 |
| validation | 1000-inf | 13 | 481.538 | 454.806 | -26.732 | -26.732885 |
| holdout | 0-100 | 701 | 87.17 | 85.949 | -1.221 | -1.220332 |
| holdout | 100-300 | 517 | 151.468 | 136.212 | -15.256 | -15.255939 |
| holdout | 300-1000 | 775 | 255.001 | 238.091 | -16.91 | -16.909904 |
| holdout | 1000-inf | 7 | 433.143 | 421.201 | -11.942 | -11.942291 |

Caveat carried from the N3C closure: early stopping selects a very low best epoch, so this candidate is close to a learned shrinkage of the classical score rather than a deep learned correction.

