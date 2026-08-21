# S6-N3D - Independent Residual Confirmation

STATUS: **RESIDUAL_CONFIRMATION_PASS**

VERDICT: **RESIDUAL_CONFIRMED_AWAITING_RUNTIME_REVIEW**

## Provenance and bindings

```text
run git:              54e5d908f2e2c917cc04e30573b4c9ca6ae34db0
runner blob:          8b1b698ddb3b4fa64ad31ed97734c38eb3d124db7a7f7e49b6238c7eab2a8079
trainer blob:         2a7f5a31814e53df90cde61585af90c3c411e9e743ceefee149d3b43ce04accb
canonical checkpoint: 5033d47cb101d96057e13aae9d3819d48fa8079e90bda8eae8cd935ac1006c55
engine binary:        05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66
archive (official):   68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7
confirm PGN:          50e88f13ddb721c2e58a5d0fb3a6be17d92d244057c30f551284503e686c95a7
confirm dataset:      3deff6a4a5cbafcdceb02b2b2c3d06ea0cd061e127cb66f24be4d2bc81d2c43d
confirm labels:       e1c25844fd785d46625f6a2a24edaa1a2e8fbd2863f57edfc3f3769723e8edfb
teacher binary:       6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9
confirm cache:        126f7c82a5dfb29dbb4750b6a979da16652dca2bd850d1b8551c9361b3e5b169
selected keys sha:    da296dc88dfc77f74b4018428d7fb4b70f570f73ca331c8ffb86cab6dfae03a4
exclude keys sha:     5e9616dfd3f7c122be0b884a24683525e0a66cec81476ce1d2d188aec3a60b59
```

## Identity audit vs the FULL N3B dataset

| check | value |
|---|---|
| confirmation games | 1400 |
| N3B source games | 3572 |
| game fingerprint intersection | 0 |
| raw records | 7156 |
| usable records | 6983 |
| excluded (position_id overlap) | 4 |
| eligible positions | 6979 |
| retained fraction | 0.999427 |

## Construction attempts and the authorized sample-size adjustment

Adjustment reason: **pre-metric construction precondition shortfall**. Metrics observed before adjustment: **False**.

| attempt | games | raw | usable | excluded | eligible | retained | status |
|---|---:|---:|---:|---:|---:|---:|---|
| g1000 | 1000 | 5099 | 4978 | 2 | 4976 | 0.999598 | CONSTRUCTION_INSUFFICIENT |
| g1400 | 1400 | 7156 | 6983 | 4 | 6979 | 0.999427 | CONSTRUCTION_SUFFICIENT |

Enlargement: 1000 -> 1400 games, one fixed enlargement, all scientific parameters frozen. 1000 of 1000 superseded games retained, 400 added.

| enlargement check | pass |
|---|---|
| all_superseded_games_retained | PASS |
| superseded_is_ordered_subsequence | PASS |
| added_game_count_exact | PASS |
| added_disjoint_from_retained | PASS |
| no_duplicate_fingerprints | PASS |

Contiguous-prefix identity is deliberately NOT asserted: long-stratum cap int(N/3) scales with games-per-month, so the 1000-game run saturated its long stratum at selection index 747 and dropped long candidates the 1400-game run retains; order-preserving subsequence containment is the provable form.

## Confirmation result (single run, disk-loaded checkpoint)

| predictor | n | clipped MAE | clipped RMSE |
|---|---:|---:|---:|
| classical | 6979 | 162.142 | 231.462 |
| classical + residual | 6979 | 153.895 | 218.473 |

MAE improvement: **5.0863%** (-8.247 cp)

### Paired per-position absolute-error delta

`abs(residual - teacher) - abs(classical - teacher)`, negative is better.

```text
n                6979
mean             -8.247828 cp
median           -8.303619 cp
improved         3918
worsened         3061
unchanged        0
bootstrap        numpy default_rng(seed=20260821), 10000 resamples
95% CI           [-9.564065, -6.935074] cp
```

## Gates

| # | gate | pass |
|---:|---|---|
| 1 | overall_clipped_mae_improvement | PASS |
| 2 | paired_abs_error_delta_bootstrap_ci_upper_below_zero | PASS |
| 3 | residual_clipped_rmse_not_worse | PASS |
| 4 | phase_bucket_no_regression | PASS |
| 5 | abs_teacher_cp_bucket_no_regression | PASS |
| 6 | integrity | PASS |

### By phase bucket

| group | n | classical MAE | residual MAE | regression | evaluated |
|---|---:|---:|---:|---:|---|
| high | 2784 | 131.121 | 122.933 | -0.062446 | yes |
| mid | 2872 | 184.379 | 174.044 | -0.056053 | yes |
| low | 1245 | 168.775 | 165.148 | -0.02149 | yes |
| zero | 78 | 344.744 | 337.458 | - | no |

### By |teacher CP| bucket

| group | n | classical MAE | residual MAE | regression | evaluated |
|---|---:|---:|---:|---:|---|
| 0-100 | 2507 | 89.346 | 86.746 | -0.0291 | yes |
| 100-300 | 2020 | 149.539 | 140.1 | -0.063121 | yes |
| 300-1000 | 2443 | 246.227 | 233.175 | -0.053008 | yes |
| 1000-inf | 9 | 444.667 | 434.452 | - | no |

### By source family

| group | n | classical MAE | residual MAE |
|---|---:|---:|---:|
| lichess-standard-rated-v1 | 6979 | 162.142 | 153.895 |

Single evaluation run. No training, no width/seed re-selection, no threshold change, no game re-selection. bench-only residual artifact parity/cost may begin in the NEXT round; no runtime, search, exporter or Arena work

