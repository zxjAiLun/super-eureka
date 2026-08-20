# S6-N3A — Independent-Source Data Pilot

STATUS: **DATA_PILOT_PASS**

## Dataset

```text
dataset_id:   s6-eval-v1-multisource-pilot01
records:      21531
dataset SHA:  5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af
rebuild SHA:  5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af
verify --allow-unlabeled rc: 0
```

## Composition

| family | records | share |
|---|---:|---:|
| arena | 10317 | 47.9% |
| lichess-standard-rated-v1 | 11214 | 52.1% |

| phase bucket | records |
|---|---:|
| high | 7099 |
| low | 5679 |
| mid | 8215 |
| zero | 538 |

| split | records |
|---|---:|
| holdout | 2112 |
| train | 17362 |
| validation | 2057 |

## Feature coverage (Rust nnue-features-batch)

| split | positions | union unique | union/40960 | unseen act. | unseen rate | pos w/ unseen |
|---|---:|---:|---:|---:|---:|---:|
| train | 17362 | 17077 | 0.416919 | - | - | - |
| validation | 2057 | 7065 | 0.172485 | 686 | 0.010015 | 392 |
| holdout | 2112 | 6976 | 0.170313 | 592 | 0.008327 | 360 |

## Pilot hard gate

| check | pass |
|---|---|
| records_total >= 10000 | PASS |
| families include arena + lichess-standard-rated-v1 | PASS |
| each family share in [0.30, 0.70] | PASS |
| low+zero >= 10% | PASS |
| train feature union >= 6500 | PASS |
| verify_dataset --allow-unlabeled pass | PASS |
| second rebuild dataset_sha256 identical | PASS |

**DATA_PILOT_PASS**

