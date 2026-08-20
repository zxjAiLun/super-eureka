# S6-N3B — Multisource NNUE Learnability Probe

STATUS: **MEASUREMENT_COMPLETE / CLOUD_VERDICT_PENDING**

## Provenance

```text
dataset: s6-eval-v1-multisource-pilot01  sha 5501240e9fd30414
labels sha e6f036f426db8a5f
teacher: Stockfish 18  nodes 16384  sha 6b087694916228c9
audit: fresh-second-pass  checked 1000  mismatches 0
checkpoint sha 66f03a5ef0c26df3  epoch 3
legacy checkpoint sha 6bfdba6d7d9cc034 (old, bound) vs new 66f03a5e (n3b)
engine sha 05b822b49940a740  git 9320cbc3c658
```

## Overall Metrics (clipped ±2000 MAE)

| split | zero | classical | new_n3b | new vs classical | new vs old |
|---|---:|---:|---:|---:|---:|
| validation | 264.804 | 167.588 | 246.942 | 79.354 (47.351%) | - |
| holdout | 263.784 | 170.036 | 245.526 | 75.49 (44.396%) | - |
| legacy_holdout (eligible 635/643) | 151.222 | 150.213 | 164.133 | 13.92 | 22.206 |

## By Source Family (validation/holdout)

### validation
| family | n | zero | classical | new |
|---|---:|---:|---:|---:|
| arena | 840 | 298.304 | 186.892 | 274.887 |
| lichess-standard-rated-v1 | 1115 | 239.567 | 153.045 | 225.89 |

### holdout
| family | n | zero | classical | new |
|---|---:|---:|---:|---:|
| arena | 1007 | 274.887 | 166.677 | 255.297 |
| lichess-standard-rated-v1 | 993 | 252.525 | 173.443 | 235.617 |

## By Phase

### validation
| phase | n | zero | classical | new |
|---|---:|---:|---:|---:|
| high | 673 | 154.89 | 144.663 | 154.536 |
| mid | 712 | 295.371 | 187.388 | 277.354 |
| low | 531 | 354.311 | 159.354 | 313.247 |
| zero | 39 | 384.821 | 313.821 | 383.57 |

### holdout
| phase | n | zero | classical | new |
|---|---:|---:|---:|---:|
| high | 672 | 168.557 | 142.705 | 156.396 |
| mid | 752 | 286.507 | 196.827 | 273.826 |
| low | 516 | 345.676 | 154.636 | 312.439 |
| zero | 60 | 341.267 | 272.817 | 313.644 |

## Coverage

```text
{
  "train": {
    "positions": 16587,
    "white_unique": 12893,
    "black_unique": 12760,
    "union_unique": 16306,
    "union_fraction": 0.398096,
    "white_fraction": 0.314771,
    "black_fraction": 0.311523,
    "total_activations": 572226,
    "activation_frequency": {
      "total_activations": 572226,
      "observed_unique_features": 16306,
      "unobserved_features": 24654,
      "mean_activations_per_feature": 35.093,
      "median_activations_per_feature": 3,
      "p10_activations_per_feature": 1,
      "p90_activations_per_feature": 42,
      "singleton_features": 4397,
      "features_with_activation_le5": 9963
    }
  },
  "validation": {
    "positions": 1955,
    "white_unique": 4650,
    "black_unique": 4647,
    "union_unique": 6786,
    "union_fraction": 0.165674,
    "white_fraction": 0.113525,
    "black_fraction": 0.113452,
    "total_activations": 66514,
    "unseen_activations": 673,
    "unseen_rate": 0.010118,
    "unseen_white_activations": 353,
    "unseen_black_activations": 320,
    "unseen_union_unique": 566,
    "unseen_union_unique_rate": 0.013818,
    "positions_with_unseen": 370,
    "positions_with_unseen_rate": 0.189258
  },
  "holdout": {
    "positions": 2000,
    "white_unique": 4727,
    "black_unique": 4507,
    "union_unique": 6623,
    "union_fraction": 0.161694,
    "white_fraction": 0.115405,
    "black_fraction": 0.110034,
    "total_activations": 68740,
    "unseen_activations": 503,
    "unseen_rate": 0.007317,
    "unseen_white_activations": 253,
    "unseen_black_activations": 250,
    "unseen_union_unique": 409,
    "unseen_union_unique_rate": 0.009985,
    "positions_with_unseen": 310,
    "positions_with_unseen_rate": 0.155
  }
}
```

## Legacy Overlap Audit

```text
{
  "raw_position_id_overlap": 7,
  "usable_position_id_overlap": 7,
  "source_game_id_overlap": 0,
  "excluded_usable_positions": 7,
  "excluded_ids_sha256": "a18a5179eb9ee65226f92aa7ff91ea5b9a10173ef330f693c75cc95a1e334fbb",
  "eligible_positions": 635,
  "retained_fraction": 0.9891,
  "policy": "exclude-exact-position-overlap-before-inference",
  "selection_uses_labels_or_predictions": false,
  "source_game_overlap_required_zero": true
}
```

