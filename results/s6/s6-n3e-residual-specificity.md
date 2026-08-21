# S6-N3E - Residual Specificity Audit

STATUS: **CHEAP_CALIBRATION_SUFFICIENT**

AUTHORIZATION: **NNUE_RUNTIME_NOT_AUTHORIZED**

This is a post-confirmation mechanism audit opened AFTER N3D was accepted. It does not rewrite or supersede any N3D artifact.

## Question

N3D's gain is consistent with a near-constant **+41.3827 cp** global correction. Can a parameter-poor calibrator reproduce it?

## Provenance and bindings

```text
run git:              c439ef89b087cccf7014256f57eadacae41045a5
runner blob:          cb1a0e05e25b1c8899d6988f74aadf5cd0e8d59e0bd7a6bd7cd0fd2765d77f41
canonical checkpoint: 5033d47cb101d96057e13aae9d3819d48fa8079e90bda8eae8cd935ac1006c55
engine binary:        05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66
N3B dataset:          5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af
N3C cache:            c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727
N3D dataset:          3deff6a4a5cbafcdceb02b2b2c3d06ea0cd061e127cb66f24be4d2bc81d2c43d
N3D cache:            126f7c82a5dfb29dbb4750b6a979da16652dca2bd850d1b8551c9361b3e5b169
```

## Calibration ladder

Fitted on **N3B train split only**, selected on **N3B validation split only**. Selected: **phase_affine** (8 parameters).

| candidate | parameters | N3B train SmoothL1 | N3B validation SmoothL1 |
|---|---:|---:|---:|
| identity | 0 | 0.126896482486 | 0.126286534527 |
| global_bias | 1 | 0.123440450273 | 0.12265033089 |
| global_affine | 2 | 0.122790390012 | 0.122376927604 |
| phase_bias | 4 | 0.123127547354 | 0.122535030124 |
| phase_affine **<- selected** | 8 | 0.118830543295 | 0.118973049864 |

Selected calibrator parameters (cp where applicable):

```json
{
  "name": "phase_affine",
  "parameter_count": 8,
  "parameters": {
    "b": [
      0.036717418,
      0.0503747198,
      0.0217028031,
      -0.000464891
    ],
    "u": [
      -0.2613823636,
      -0.1061941736,
      -0.0117859984,
      1.5102040829
    ]
  },
  "phase_order": [
    "high",
    "mid",
    "low",
    "zero"
  ],
  "train_smooth_l1": 0.118830543295
}
```

## Predictor comparison (clipped MAE / RMSE, cp)

| split | n | raw classical | selected calibrator | NNUE hybrid |
|---|---:|---:|---:|---:|
| N3B validation | 1955 | 167.588 / 242.686 | 160.738 / 228.804 | 159.685 / 229.905 |
| N3B holdout | 2000 | 170.036 / 242.526 | 162.985 / 226.835 | 159.071 / 227.747 |
| N3D confirmation | 6979 | 162.142 / 231.462 | 155.103 / 215.082 | 153.895 / 218.473 |

### Every calibrator candidate (clipped MAE, cp)

| split | identity | global_bias | global_affine | phase_bias | phase_affine | hybrid |
|---|---:|---:|---:|---:|---:|---:|
| N3B validation | 167.588 | 164.694 | 164.327 | 164.483 | 160.738 | 159.685 |
| N3B holdout | 170.036 | 165.619 | 165.832 | 165.354 | 162.985 | 159.071 |
| N3D confirmation | 162.142 | 157.4 | 157.16 | 157.134 | 155.103 | 153.895 |

### Mechanism: what the correction actually looks like

| split | correction mean | std | p10 | p50 | p90 | corr vs classical | calibrator captures |
|---|---:|---:|---:|---:|---:|---:|---:|
| N3B validation | 39.2578 | 46.8502 | -14.0733 | 33.2254 | 102.5379 | 0.001324 | 86.68% |
| N3B holdout | 40.9494 | 46.7525 | -14.353 | 36.7959 | 101.6862 | -0.007261 | 64.30% |
| N3D confirmation | 41.3827 | 46.3602 | -13.9328 | 38.47 | 102.5067 | -0.046391 | 85.35% |

### Paired per-position delta vs the selected calibrator

`abs(hybrid - teacher) - abs(calibrator - teacher)` on the N3D confirmation set; negative favours the NNUE.

```text
n                6979
mean             -1.208211 cp
median           -2.165429 cp
improved         3636
worsened         3343
unchanged        0
bootstrap        numpy default_rng(seed=20260821), 10000 resamples
95% CI           [-2.494612, 0.073826] cp
```

## Gates

| # | gate | pass |
|---:|---|---|
| 1 | hybrid_not_worse_than_calibrator_on_n3b_validation_and_holdout | PASS |
| 2 | confirmation_mae_improvement_over_calibrator | FAIL |
| 3 | paired_abs_error_delta_bootstrap_ci_upper_below_zero | FAIL |
| 4 | hybrid_rmse_not_above_calibrator | FAIL |
| 5 | phase_bucket_no_regression_vs_calibrator | PASS |
| 6 | abs_teacher_cp_bucket_no_regression_vs_calibrator | FAIL |
| 7 | provenance_cache_checkpoint_finite_integrity | PASS |

### By phase bucket (hybrid vs selected calibrator)

| group | n | calibrator MAE | hybrid MAE | regression | evaluated |
|---|---:|---:|---:|---:|---|
| high | 2784 | 126.526 | 122.933 | -0.028397 | yes |
| mid | 2872 | 174.995 | 174.044 | -0.005434 | yes |
| low | 1245 | 167.222 | 165.148 | -0.012403 | yes |
| zero | 78 | - | - | - | no |

### By |teacher CP| bucket (hybrid vs selected calibrator)

| group | n | calibrator MAE | hybrid MAE | regression | evaluated |
|---|---:|---:|---:|---:|---|
| 0-100 | 2507 | 77.686 | 86.746 | 0.116623 | yes |
| 100-300 | 2020 | 143.145 | 140.1 | -0.021272 | yes |
| 300-1000 | 2443 | 243.35 | 233.175 | -0.041812 | yes |
| 1000-inf | 9 | - | - | - | no |

## Outcome

The cheap calibrator `phase_affine` (8 parameters) already captures 85.35% of the NNUE's gain over raw classical, and the NNUE does not clear the specificity gates. The measured N3D improvement is adequately explained by a cheap recalibration of the classical evaluation.

NNUE residual runtime sub-branch CLOSED. No exporter and no Rust runtime file is created. Production baseline bde9085 stays unchanged. The selected calibrator's parameters are recorded above for whatever the cloud decides to do with them.

