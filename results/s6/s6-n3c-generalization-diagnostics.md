# S6-N3C - NNUE Generalization Diagnostics

STATUS: **DIAGNOSTICS_COMPLETE**

VERDICT: **CLOUD_VERDICT_PENDING**

## Provenance

```text
run git: 9ecc87fe95c03c02b56d88acaaa83f54001b784b
diagnostics script sha256: 591d4f31bb094664f2a25f128bbf3dcd6fdba075759c3517890d92a2f1dbb01a
N3B checkpoint sha256: b56e0c336bfc761cddd5dccffd4636a4fdaf3ea10f668029073294436b62a7db
```

## Dataset Statistics

| dataset | raw | usable | null CP | target mean | target std | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | 5919 | 5891 | 28 | 18.332 | 220.528 | -248.0 | 17.0 | 287.0 |
| N3B | 21531 | 20542 | 989 | 25.289 | 362.424 | -456.0 | 12.0 | 494.0 |

## Controls

| control | status | detail |
|---|---|---|
| A | PASS | 0.0 |
| B | PASS |  |
| C | PASS | selected width=32 |
| D | PASS |  |
| E | PASS | selected width=16 |

### Control C Runs

| width | seed | status | validation MAE | holdout MAE |
|---:|---:|---|---:|---:|
| 4 | 20260818 | PASS | 247.945 | 249.325 |
| 4 | 20260819 | PASS | 248.085 | 247.849 |
| 4 | 20260820 | PASS | 245.348 | 245.245 |
| 8 | 20260818 | PASS | 245.692 | 245.657 |
| 8 | 20260819 | PASS | 245.914 | 244.716 |
| 8 | 20260820 | PASS | 245.624 | 246.664 |
| 16 | 20260818 | PASS | 246.936 | 245.926 |
| 16 | 20260819 | PASS | 246.598 | 244.462 |
| 16 | 20260820 | PASS | 246.234 | 243.671 |
| 32 | 20260818 | PASS | 246.942 | 245.526 |
| 32 | 20260819 | PASS | 245.619 | 241.699 |
| 32 | 20260820 | PASS | 243.689 | 242.984 |

### Control D Runs

| family | seed | status | validation MAE | holdout MAE | legacy MAE |
|---|---:|---|---:|---:|---:|
| arena | 20260818 | PASS | 279.257 | 259.792 | 171.099 |
| arena | 20260819 | PASS | 276.305 | 261.936 | 169.83 |
| arena | 20260820 | PASS | 278.392 | 257.267 | 167.258 |
| lichess-standard-rated-v1 | 20260818 | PASS | 226.675 | 240.362 | 157.579 |
| lichess-standard-rated-v1 | 20260819 | PASS | 227.277 | 240.521 | 161.586 |
| lichess-standard-rated-v1 | 20260820 | PASS | 226.943 | 239.621 | 148.531 |

### Control E Runs

| width | seed | status | residual validation MAE | residual holdout MAE |
|---:|---:|---|---:|---:|
| 8 | 20260818 | PASS | 159.798 | 156.79 |
| 8 | 20260819 | PASS | 160.629 | 158.938 |
| 8 | 20260820 | PASS | 159.552 | 157.012 |
| 16 | 20260818 | PASS | 159.685 | 159.071 |
| 16 | 20260819 | PASS | 159.713 | 158.964 |
| 16 | 20260820 | PASS | 159.123 | 158.779 |

## Interpretation Signals

| signal | value |
|---|---|
| TRAINER_REGRESSION | False |
| OVERPARAMETERIZED_SPARSE_TABLE | False |
| FAMILY_INTERFERENCE | False |
| DISTRIBUTION_REPRESENTATION_GAP | True |
| RESIDUAL_PATH_PROMISING | True |
| CURRENT_NNUE_REPRESENTATION_NOT_VIABLE | False |

All configurations are retained in the JSON record; no holdout-based model selection was performed.

