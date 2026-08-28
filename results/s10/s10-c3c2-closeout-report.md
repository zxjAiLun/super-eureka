# S10-C3-C2 Closeout — AVX2 L1 Dense Kernel: ACCEPTED

**Status: CLOSED / PASS — candidate ACCEPTED (decision gate: keep_avx2_then_arena)**

## What was done

The final bounded runtime optimization before Arena: an explicit
`_mm256_madd_epi16` AVX2 kernel for the L1 dense layer ONLY (256→32),
keeping the OUTPUT-MAJOR weight layout (the input-major transpose was
rejected by C3-C1). One vector accumulator per output — no
32-accumulator register pressure.

```
per output o:
    vacc = 0 (8 x i32 lanes)
    for i in 0..256 step 16:
        vacc += _mm256_madd_epi16(w[o][i:i+16], a16[i:i+16])
    z = bias[o] + horizontal_sum(vacc)
    a1[o] = clamp(shift_round(z))
```

- Activations (clamped to [0, 4096]) packed to i16 once per eval; the
  madd does 16 i16xi16→8 i32 pair-products per instruction.
- Runtime dispatch: `is_x86_feature_detected!("avx2")` ONCE at model
  load → `L1Backend::{Scalar, Avx2}` frozen on the model; the eval path
  pays one plain branch. No `-C target-cpu=native`; scalar fallback
  always exists (`force_scalar_l1` feature forces it for A/B builds).
- Bit-exactness argument: every product (|w|<=1387, |a|<=4096) and every
  lane partial sum stays far inside i32 under the loader's proven
  1.455e9 bound; integer addition has no intermediate rounding, so lane
  re-grouping cannot change the final integer.

## Correctness gates (all PASS before performance judgment)

```
artifact SHA unchanged                          b51a79b1...
startpos raw = 190                              unchanged
AVX2 vs scalar Rust (forced-scalar build):
  frozen 10k C3-A corpus: 10,000/10,000 raw     BIT-EXACT
Python integer reference vs AVX2 binary:
  1,000 positions                               BIT-EXACT
FullRefresh vs Incremental search parity smoke  exact
cargo test --release                            417 pass
```

## Performance gate (same-session paired real search, 3 arms)

```
A  = current binary, CurrentFinal (Eval2)
C0 = frozen baseline binary aa9ec116... (2eae99c incremental)
C1 = current binary, incremental NNUE (AVX2)

24 FEN x 12 rounds x 200k nodes x cold 64 MiB TT, balanced 6-permutation
rotation; C1/C0 tree identity fail-closed per run (nodes/qnodes/score/
bestmove/PV); baseline SHA hashed and verified at harness startup
(Repair-1 hardening).

                    median    p25      p75       n
C1/C0               1.4981   1.4575   1.5430    288
C1/A                1.2418   1.0298   1.3166    288
C0/A                0.8405   0.6889   0.8737    288   (session calibration;
                                                          matches 0.8200/0.8206)

raw NPS medians:    A 265.0k | C0 230.0k | C1 342.7k
identity:           288/288 exact, 0 failures
UCI spot check:     startpos depth 9, 388-395k NPS sustained
```

## Decision (frozen rules)

```
C1/C0 = 1.498 > 1.00  →  keep_avx2_then_arena
```

The AVX2 kernel takes the incremental NNUE from 0.82x Eval2 to **1.24x
Eval2** — the NNUE arm is now FASTER than the production classical
evaluator while carrying a 300k-teacher-trained network. Per the frozen
plan this is the LAST runtime optimization candidate; next stop is
S10-D Arena with `CurrentFinalNnueV2QIncremental + AVX2`.

## Provenance note (Repair 1, prior commit 86b27dc)

The C3-B official artifact and harness were restored verbatim from
2eae99c; the candidate harness is now separate (search_nps_c3c.py) and
fails closed on the baseline binary SHA at startup.
