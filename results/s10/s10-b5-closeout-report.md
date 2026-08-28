# S10-B5 Closeout — Quantized NNUE Numerical Gate

**Status: CLOSED / PASS — all gates green, PTQ only (no QAT needed)**

**Repair 1 (post-review)**: calibration tool/artifact realigned to the
frozen v3 scheme (previously recorded a rejected early scheme); Rust
loader now fail-closed on embedded source SHAs AND recomputes proven i32
MAC bounds from the actual payload at load time; consistency tests pin
calibration == layout == exporter constants. Numerical results unchanged.

## Frozen quantization scheme (EUNN2Q01 v1, "scheme v3")

All power-of-two shifts, zero float operations at inference time:

```
FT:       q_w = round(w * 2^12) i16, q_b = round(b * 2^12) i32
          acc = q_b + sum(q_w rows), i32
          PROVEN bound: |q_b|max + 31*32767 = 1,016,721 << 2^31
Dense:    (l1/l2/out) identical pattern, NO input requantization:
          q_w = round(w * 2^12) i16 (l1 1387 / l2 2967 / out 2224 max)
          q_b = round(b * 2^24) i32 (z_int units = float * 2^24)
          z_int = q_b + sum(q_w * a)   [a = accumulator-precision input]
          PROVEN bounds: l1 1.45e9 (1.5x), l2 3.9e8 (5.5x), out 2.9e8 (7.4x)
Activation: clamp(x, 0, QA=4096) integer ClippedReLU
Layer out: shift_round(z_int, 12) -> A units, clamp(0, 4096)
Output:   raw = shift_round(z_out, 12); cp = raw / 2^12 * 1000 (final float)
Rounding: quantize round-half-away; shift round-half-away; saturation clamps
```

Design iteration (recorded for provenance): scheme v1 (i8 dense weights)
collapsed — the l1 scale 2^7/4096 quantized all weights to 0. Scheme v2
(i16 weights 2^16 + dense-input requantization d=(a+8)>>4) had amplified
per-layer rounding (gate2 mean 1.90cp, p95 7.7cp — FAIL). Scheme v3 removed
the input requantization entirely (MAC at accumulator precision, weights
2^12) which keeps every i32 MAC provably in range and passes all gates.

## Artifacts

```
quantized artifact:  data/s10/b3/seed-20260818/nnue-v2-q01.bin
SHA256:              b51a79b19999aeed974c2279eef60b01f890248c7d006cbe3d504cc7c0f28b9a
size:                5786544 bytes (5.52 MiB; FP32 artifact 11.57 MB -> 50%)
deterministic:       export A == export B == deployed artifact (byte-identical)
source checkpoint:   d59ad852... (frozen B3 candidate, verified at export)
source FP32 artifact: 9bf7addd... (frozen B4 artifact, verified at export)
layout:              results/s10/s10-b5-quantized-layout.json
```

## Gate results (results/s10/s10-b5-gates.json)

```
Gate 1  Python integer reference vs Rust integer runtime (bit-exact raw):
        1000 positions (train, dataset order) — 0 mismatches            PASS

Gate 2  quantized vs frozen FP32 (10000 non-holdout positions):
        mean  0.1933 cp   (<= 1.0)                                     PASS
        p50   0.1456 cp
        p95   0.5408 cp   (<= 2.0)                                     PASS
        p99   0.8240 cp   (<= 4.0)                                     PASS
        max   1.5767 cp

Gate 3  validation teacher-MAE delta (29270 positions):
        FP32  165.3246 cp
        quant 165.3312 cp
        delta +0.0066 cp   (<= +1.0)                                   PASS

Holdout used anywhere in B5: NO (corpus train-only; validation for MAE only)
active_features_v2 touched: NO (Rust runtime reuses the single source of truth)
```

## Implementation

```
exporter:      tools/s10/export_quantized.py (fail-closed on frozen
               checkpoint/FP32-artifact SHA, tensor shapes, i32 bounds)
reference:     tools/s10/integer_reference.py (pure-integer Python
               semantics: the single source of truth for the scheme)
calibration:   tools/s10/calibrate_quantization.py (imports scheme
               constants from the exporter; 10k-position activation
               telemetry + proven bounds matching the layout artifact)
gates:         tools/s10/gates_quantized.py (gates 1-3, frozen identities)
Rust runtime:  src/engine/nnue_v2q_runtime.rs (bench-only; loader fails
               closed on magic/version/dims/shifts, frozen source
               checkpoint + FP32 artifact SHAs, and recomputed proven
               i32 MAC bounds from the actual payload)
bench:         nnue-v2q-probe / nnue-v2q-probe-batch (JSONL with raw_output
               for bit-exact comparison)
engine SHA256: see gates artifact (rebuilt in Repair 1; predictions
               unchanged: startpos raw 190 -> 46.387 cp)
```

## Tests

```
cargo test --release:  388 passed (378 prior + 10 nnue_v2q incl. loader
                       source-SHA / proven-bound rejection tests)
tools/s10 (python):    40 passed (incl. 6 calibration-layout-exporter
                       consistency tests)
startpos sanity:       Rust raw 190 -> 46.387 cp (FP32 46.445 cp)
```

## Verdict

PTQ quality makes QAT unnecessary: quantization costs +0.0066 cp validation
MAE (0.004%) and <= 1.58 cp worst-case per-position deviation. The integer
network is a faithful, provably overflow-free representation of the frozen
FP32 production function.

```
deferred (unchanged): incremental accumulator (S10-C1), eval/search wiring
(S10-C2), NPS benchmark (S10-C3), Arena (S10-D)
```
