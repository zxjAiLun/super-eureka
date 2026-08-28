# S10-C3-C1 Closeout — L1 Input-Major Layout Candidate: REJECTED

**Status: CLOSED — candidate REJECTED and REVERTED (gate: <5% improvement,
real-search paired measurement showed a ~9-10% REGRESSION)**

## What was tried

The review-frozen first candidate: load-time transpose of L1 weights to
input-major `[256][32]`, dense forward with input-outer / 32-output-inner
loops. Accumulation order per output strictly preserved (i=0..255), so
the raw output is bit-exact — which was verified.

## Correctness gates (all PASSED before any performance judgment)

```
frozen artifact SHA unchanged                 b51a79b1...  PASS
Python integer reference vs optimized Rust
  1000 positions raw_output bit-exact                        PASS
startpos raw = 190 unchanged                                 PASS
Full vs Incremental search parity smoke (3 FENs, 9 fields)   PASS
cargo test --release 416 pass                                PASS
```

## Performance evidence

### Microbench (isolated, hot-cache loops) — MISLEADING

```
session of record (17a23fd):  dense_forward 2782.8 ns
candidate session:            dense_forward 2497.5 ns   (-10.3%)
baseline binary TODAY:        dense_forward 2499.0 ns   (SAME as candidate)
```

The baseline binary re-measured today shows the same "improvement" — the
microbench delta was machine-state drift between sessions, not the
transpose. Hot-cache component loops are NOT a valid proxy here.

### Real-search paired measurement (decisive)

Same machine, same protocol, both binaries interleaved in the same loop
(4-arm C3-B rerun with C0 = frozen baseline binary
aa9ec116..., 24 FEN × 12 rounds × 200k nodes × 64 MiB cold TT):

```
C/C0 (optimized / baseline incremental)  median 0.9125   p25 0.9001
                                                                    p75 0.9249
C0/A (baseline / Eval2)                  median 0.8206   (matches the
                                                    original 0.8200 —
                                                    baseline intact)
C/A (optimized / Eval2)                  median 0.7500
B/C tree identity across BOTH binaries: 288/288 exact
```

Direct confirmation on a single high-branching fixture (interleaved
old-new-new-old ×8, 100k nodes):

```
new/old elapsed ratio = 1.108  (candidate ~10% SLOWER)
```

## Verdict per the frozen gate

The candidate is REJECTED: real-search paired throughput got WORSE by
~9-10%, not better. The optimization was fully reverted; the release
binary is byte-identical to the 2eae99c baseline again (SHA
aa9ec116... re-verified; startpos raw 190; cargo 416 pass).

## What was learned (the actual value of this run)

1. The input-major inner-32 loop is a real-world LOSS on this machine —
   plausibly 32 live i32 accumulators in the inner loop defeat the
   compiler's register allocation vs the output-major form's single
   accumulator, and/or the hot-cache microbench flattered a layout that
   interleaves worse with real search working sets.
2. Microbench sessions on this machine drift by ~10% — future gates MUST
   use paired same-session comparisons (as C/C0 did here), never
   cross-session absolute ns/op.
3. Baseline reproducibility is excellent: C0/A = 0.8206 vs the original
   0.8200 across sessions.

## Next bounded candidate (per review escalation path)

Explicit x86_64 AVX2 widening MAC for L1 ONLY — same narrow scope
(256→32 L1, nothing else), same bit-exact gate, and measured ONLY via
paired same-session real-search ratios (C1/C0), with microbench numbers
reported as diagnostics, never as the gate.
