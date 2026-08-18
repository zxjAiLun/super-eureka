# S6-N2 — Bench-only NNUE Runtime Parity & Cost

STATUS: **PARITY_PASS / COST_MEASURED**

## Provenance

```text
checkpoint SHA256:   6bfdba6d7d9cc034d55d8bfe433ebb3b0d6f48d78afa2351f3ef465ac9003a66
artifact SHA256:     aa39f52a85ead6b776b5b2bf973e06a8e993a6380387897f17e22cd9f25a27b0
engine binary SHA256: 05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66
engine git:          3b44954cc13556e011f995be314cd23133bbeb7e (feat commit; code source)
dataset SHA256:      3a3483fd46fd5a570c4c62b7d93378efc80eafbab43ec155db5ac5894fbc6a9d
labels SHA256:       78dd8d52a34d1dd10a5d09cb3295be8f3a91a495d808fbd8b0cb68d31d668aa5
```

## Artifact (S6-N2 probe format, bench-only, not production)

```text
magic "EUNN1F32", version 1, inputs 40960, width 32, target_scale 1000.0
header 56 B + payload: features [40960][32] input-major, acc_bias [32],
head.weight [64] own-then-opponent, head.bias [1]  -> total 5243324 B
offsets (canonical, from artifact_offsets()): header 0, features 56,
acc_bias 5242936, head 5243064, head.bias 5243320, end 5243324
checkpoint SHA embedded raw in header; loader rejects any length/shape/
finite/trailing mismatch.
```

## Parity gate (Python checkpoint vs Rust artifact, 5891 CP rows)

```text
rows:            5891 / 5891
NaN/Inf:         0
sign mismatch:   0
max  abs diff:   0.000119 cp   (gate <= 0.1)
mean abs diff:   0.000012 cp   (gate <= 0.01)
p50 / p95 / p99: 0.000009 / 0.000034 / 0.000054 cp
```

The Rust full-refresh inference is numerically consistent with the disk
checkpoint across every CP-labelled position; residual differences are pure
float32 summation order (~1e-5 cp).

## Microbench (5 rounds, 6 positions, 2000 iterations/round)

| metric | median ns/call |
|---|---:|
| feature extraction (both perspectives) | 121.75 |
| full full-refresh NNUE (incl. extraction) | 376.99 |
| classical evaluate | 122.16 |

```text
NNUE / classical ratio:      3.086
feature-extraction share of NNUE: 0.323
checksum:                    82793484820000
```

Raw per-round values are recorded in `results/s6/s6-n2-runtime-probe.json`.
Cost-only measurement: no promotion gate; no accumulator is justified or
implemented from this data alone.
