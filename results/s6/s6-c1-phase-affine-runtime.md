# S6-C1 - Phase-Affine Runtime Parity

STATUS: **PHASE_AFFINE_RUNTIME_PARITY_PASS**

## Bindings

```text
run git:        91ca15cdeb0ac5696f71be44dc58c63af0f5abfe
engine binary:  7535ec09414eed2ae3fab230cf8f111d25667921a80ebcc4f7a08a38311fc4ad
N3E result:     4a1fb32d76cbd69d0cee5be02ed6f4cfd5177020b172ba8a08ecce84e982f627
N3B dataset:    5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af
N3D dataset:    3deff6a4a5cbafcdceb02b2b2c3d06ea0cd061e127cb66f24be4d2bc81d2c43d
N3C cache:      c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727
N3D cache:      126f7c82a5dfb29dbb4750b6a979da16652dca2bd850d1b8551c9361b3e5b169
```

## Frozen constants (derived from the N3E result)

```text
phase order    ['high', 'mid', 'low', 'zero']
scale          1000000
factor         [738618, 893806, 988214, 2510204]
bias_scaled_cp [36717418, 50374720, 21702803, -464891]
```

## Parity

| split | positions | exact fixed-point | base_cp vs cache | max quant | mean quant |
|---|---:|---|---|---:|---:|
| N3B usable | 20542 | exact | equal | 0.499917 | 0.250518 |
| N3D eligible | 6979 | exact | equal | 0.499917 | 0.248209 |

## Static metric reproduction (N3D confirmation set)

| metric | N3E record | Rust runtime | drift | budget |
|---|---:|---:|---:|---:|
| clipped MAE | 155.103 | 155.105 | 0.002 | 0.05 |
| clipped RMSE | 215.082 | 215.089 | 0.007 | 0.05 |

## Microbench (median of 5)

```text
positions        512
iterations       200
base ns/eval     285.5513
candidate ns/eval 288.463
ratio            1.010197 (gate <= 1.1)
```

## Gates

| gate | pass |
|---|---|
| constants_derived_from_n3e_result | PASS |
| labels_aligned_with_n3e_result | PASS |
| n3b_exact_fixed_point_parity | PASS |
| n3d_exact_fixed_point_parity | PASS |
| base_cp_unchanged_vs_committed_caches | PASS |
| n3b_quantization_within_budget | PASS |
| n3d_quantization_within_budget | PASS |
| static_metrics_reproduced | PASS |
| microbench_feasible | PASS |

Arena baseline profile: `current-final`; candidate profile: `current-final-phase-affine`. Same binary, evaluator dispatch is the only difference.

