# S10-C3-A Closeout — Microcost + Real-tree Update Telemetry

**Status: CLOSED / PASS (measurement only; zero optimization)**

## Frozen protocol

```
corpus:            data/s10/b3/c3a-corpus-10k.txt
                   10,000 non-holdout usable positions (dataset order,
                   train only), position_id|fen lines
corpus sha256:     6eaa7f8fabcbb2a4eb9bf3b5f8f86decc89175ce9524261c6e3a4686f198b611
transition corpus: 10,000 deterministic legal-playout transitions
                   (xorshift seed 0xc3a07a11...), split ordinary vs
                   own-king
rounds:            16 measured per component (2 warmup), median primary
model:             b51a79b1... (frozen quantized artifact)
FEN parse / model load / fixture precompute: all OUTSIDE timed regions
anti-DCE:          std::hint::black_box on every result
command:           bench nnue-v2q-cost --model <bin> --batch <file> --rounds 16
artifact:          results/s10/s10-c3a-microcost.json
```

## Component microcost (10k ops/round, median ns/op)

```
component                          median_ns/op   p25        p75
eval2_total                          3452.6      3412       3493
v2_feature_extract (2 persp)          663.9       660        674
ft_accumulate (2 persp)              1144.0      1135       1154
dense_forward                        2782.8      2714       2789
nnue_full_total                      4431.1      4418       4454
delta_prepare                           6.4         6          7
ordinary_accumulator_update           123.4       123        124
king_refresh_accumulator_update       209.7       193        206
dense_from_accumulator               2862.9      2789       2885
incremental_edge_plus_eval           3051.8      3006       3072
```

Notes:
- `nnue_full_total` (4431) < feature(664)+FT(1144)+dense(2783) = 4591 —
  cache/locality makes the composed path cheaper than the sum; this is why
  the total is measured directly, not summed.
- `king_refresh` here = ONE perspective full-refresh + the OTHER
  perspective's delta (the C2A update rule), measured on own-king
  transitions only.
- The random-playout transition corpus is king-heavy (9528/10000 king
  moves — kings move often in sparse random positions); that is exactly
  why ordinary/king are reported SEPARATELY and why the real-search rates
  below are the authoritative workload weights.

## Real search-tree telemetry (8 representative fixtures, incremental)

```
artifact:    results/s10/s10-c3a-search-tree-telemetry.json
pushes:      285,311   == pops (perfectly balanced)
null_pushes: 19
real move edges: 285,292
max depth:   103

king_refresh_rate            = full_refreshes / real_move_edges = 20.53%
FT row updates per real move = delta_updates / real_move_edges = 4.156
null move rate               = null_pushes / pushes             = 0.007%
```

## Cost model synthesis (pre-optimization baseline)

Per real move edge in a real search tree:

```
80% ordinary edges:   ~123 ns update    + dense at eval time
20% king edges:       ~210 ns update    + dense at eval time
avg update cost       ≈ 0.795*123 + 0.205*210 ≈ 141 ns/edge
+ delta prepare       ≈   6 ns
+ 4.16 FT row add/sub per edge (both perspectives combined)
```

But the DOMINANT cost is the dense forward at every static eval:

```
dense forward ≈ 2783–2863 ns   vs   accumulator update ≈ 141 ns/edge
NNUE full total 4431 ns        vs   Eval2 3453 ns
```

**Headline findings (no optimization performed):**

1. The incremental accumulator update is CHEAP (~141 ns/move average,
   only ~3-5x the delta-prepare cost) — C2A's move-aware design works.
2. The dense forward is ~63% of the full NNUE eval and is paid at EVERY
   static eval in BOTH arms (full and incremental) — it dominates.
3. NNUE full-refresh total (4431 ns) is currently ~1.28x the production
   Eval2 (3453 ns). The incremental eval path saves the feature+FT work
   (664+1144 = 1808 ns) per eval vs full refresh, but still pays the
   ~2783 ns dense + ~141 ns/edge amortized updates.
4. King refresh costs 20.5% of edges but only ~1.7x an ordinary update —
   the C2A single-perspective refresh is not a major penalty.

Search setup (model load + root accumulator) is one-time per run and is
excluded from all steady-state numbers above (per C3-0 review guidance).

## Explicitly NOT done here

```
SIMD / accumulator layout / frame-copy avoidance / lazy accumulators:
deferred to a bounded optimization pass ONLY IF C3-B shows the NNUE
arms are clearly slower than Eval2 (decision gate frozen in the C3-B
review: C/A << 1 → optimize; C/A ≈ 1 or > 1 → Arena directly).
```
