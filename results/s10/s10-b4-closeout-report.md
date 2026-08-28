# S10-B4 Closeout — FP32 Rust Inference Parity Bridge

**Status: CLOSED / PASS**

## Goal

Prove that the frozen S10-B3 production checkpoint's function is reproduced
by a Rust full-refresh implementation, position by position, before any
quantization / accumulator / search integration.

## Frozen candidate

```
checkpoint:  data/s10/b3/seed-20260818/checkpoint_v2_s20260818.pt
SHA256:      d59ad8525c06abe80307bffb121ff497a36e94b191c3c9bb3c8f31e5cce550c7
seed:        20260818
architecture: 22528 sparse inputs -> FT 128 -> concat [stm, nstm] 256
              -> 32 -> 32 -> 1, ClippedReLU(0,1), target scale 1000
val MAE:     165.3246 cp | holdout MAE: 166.0762 cp (from B3)
```

## Artifacts

```
EUNN2F32 artifact:  data/s10/b3/seed-20260818/nnue-v2-f32.bin (11572156 bytes)
artifact SHA256:    9bf7adddf7b3b44affa5e26d2276b13d74566191a4eb4d0090fbde5a7afbc9fc
format:             magic "EUNN2F32" v1, LE, inputs=22528, ft_width=128,
                    target_scale=1000, checkpoint SHA embedded;
                    ft_weights[22528][128], ft_bias[128],
                    l1_w[32][256], l1_b[32], l2_w[32][32], l2_b[32],
                    out_w[1][32], out_b[1]; all floats finite-validated
exporter:           tools/s10/export_nnue_v2.py (fail-closed on checkpoint
                    SHA, seed, dataset/labels SHA, holdout_observed=false,
                    tensor keys/shapes/finiteness)
```

## Rust runtime

```
module:      src/engine/nnue_v2_runtime.rs (bench-only, NOT wired into
             search/eval/UCI; no quantization, no incremental accumulator)
subcommands: bench nnue-v2-probe --model <bin> --fen <fen>
             bench nnue-v2-probe-batch --model <bin> --batch <file>
inference:   full refresh per call — FT accumulate per perspective
             (bias + sum of active V2 feature rows, single source of truth
             nnue::active_features_v2), ClippedReLU(0,1) on concat,
             256->32->32->1 with ClippedReLU, scaled * 1000
engine SHA256: f005811bda2c73f8833915787dc7fcc777b8a6b84803f752ff357c5d60bfb41b
```

## Parity result (results/s10/s10-b4-parity.json)

```
corpus:      first 1000 non-holdout usable records (all train) of
             s10-eval-v1-300k01, dataset order; 0 holdout positions
python side: frozen checkpoint loaded into NnueModel, eval mode, no_grad;
             features from the same engine exporter (single source of truth)
rust side:   nnue-v2-probe-batch over the same ordered corpus

max_abs_error_cp:   0.000153
mean_abs_error_cp:  0.0000283
gate:               <= 0.01 cp
PASS
```

The residual error is float operation-ordering noise (Python embedding_bag
sums features in a different order than the Rust row loop), three orders of
magnitude below the gate and five below 1 cp.

## Tests

```
Rust:   378 lib tests pass (6 new: valid load + prediction, bad
        magic/version/dims, truncated/trailing, NaN payload,
        stm/nstm ordering, position immutability)
Python: tools/s10 suite 34/34 (B3 tests unchanged)
```

## Not done here (deferred by design)

```
quantization ................. S10-B5
incremental accumulator ...... S10-C
production eval wiring ....... S10-C
NPS measurement .............. S10-C
Arena .......................  S10-D
```
