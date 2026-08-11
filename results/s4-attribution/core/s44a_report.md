# S4.4A — Post-Promotion Core Re-Attribution

ATTRIBUTION ONLY — no optimization candidate.

## Production identity

```text
production source: 26604c425625d69e5b7e7b967db8926f4da01b8a
feat(search): promote legality fast path into current-final
artifact:          20260811-26604c4-linux-x86_64
binary SHA:        f0e8f91a3a0828a158672cecdf7859dbd9a3c9bac36b965bdcc90db31b51189d
HEAD at measurement: dc3c023 (records-only commits after 26604c4;
                      `git diff --stat 26604c4..HEAD -- src/` = empty)
```

## Benchmark contract

```text
profile:  current-final (uses_legality_fast = true, promoted)
corpus:   30-position S4 core-attribution corpus
limit:    fixed depth 6
TT:       16 MB, cold per position
threads:  1
sampler:  sparse 256/1 call-granular sampling
repeat:   3 full corpus repetitions
no forced-root, no target-root, no selectivity diagnostics
```

## Throughput (per repetition + aggregate)

| rep | nodes | wall s | NPS |
|---|---|---|---|
| 1 | 3,637,042 | 15.97 | 227,768 |
| 2 | 3,637,042 | 16.32 | 222,855 |
| 3 | 3,637,042 | 15.91 | 228,604 |
| all | 10,911,126 | 48.20 | 226,381 |

All three repetitions are bit-identical in nodes (3,637,042 each;
identical to the S4.3A run — same tree, same limit, cold TT).

## Top-level wall buckets (sparse sampled, mean-based share of elapsed)

| rank | bucket | share of elapsed | calls | samples |
|---|---|---|---|---|
| movegen_full_legal | 59.9% | 16,840,467 | 65,571 |
| movegen_tactical | 10.4% | 8,029,668 | 31,293 |
| SEE/qSEE | 10.1% | 6,436,323 | 25,086 |
| move_ordering | 9.4% | 10,887,774 | 42,411 |
| eval | 7.0% | 9,730,197 | 37,902 |
| movegen_evasion | 4.7% | 954,714 | 3,762 |
| TT_probe_store | 2.7% | 7,836,024 | 30,534 |
| movegen_has_any | 1.7% | 1,485,123 | 5,820 |
| accounted sum | 105.9% | | |
| other / unattributed | 0.0% | | |

> Note: mean-based sparse sampling over-accounts on this host
> (accounted sum 105.9%; descheduled sampled windows inflate the mean).
> The ranking and the within-bucket splits (same sampled calls) are robust;
> absolute shares carry ~6% sampling uncertainty from this effect.

## Full-legal sub-attribution (inside `movegen_full_legal`)

Call-granular sparse phases (pseudo gen / check-state / pin scan) are
measured directly; the per-move loop is split by exact counters x
microbench per-op costs (in-band per-move wall timing would be the same
order of magnitude as the measured work and would pollute the sampler).

| component | share of full-legal bucket | ns per full-legal call |
|---|---|---|
| pseudo move generation | 37.5% | 643 |
| king/check-state setup (in-check test) | 5.1% | 88 |
| absolute_pin_mask slider-ray pin scan | 6.6% | 113 |
| loop + rest (measured residual) | 50.8% | 871 |
|   of which fast accepts (model) | 2.8% | 47 |
|   of which fallback probes (model) | 5.2% | 89 |
|   of which in-check probes (model) | 5.1% | 87 |
|   other (model residual) | 37.8% | 648 |

Cost-model split explains 26% of the measured
loop+rest; the residual is sampled-window inflation plus unmodeled
loop bookkeeping (iterator, branch, Vec growth). The model is a lower
bound for the per-op costs (microbench fast-accept path is compiler-friendly).

## Traversal counters (normalized)

```text
total search nodes                            10,911,126
full legal generator calls                   16,840,467  (1.54/node)
full-legal pseudo moves                      602,479,188  (55.2/node)
fast accepts                                 467,640,114  (77.6% of full-legal pseudo)
fallback probes (non-check)                  43,941,288  (7.3% of full-legal pseudo)
fallback probes (in-check branch)            45,448,893  (7.5%)
fallback probes (total, make/unmake)         89,390,181  (14.8%)
absolute_pin_mask calls                      15,510,450  (0.92/full-legal call)
in-check full-legal calls                    1,330,017  (7.9%)

legality probes by generator (make/unmake):
  full legal     89,390,181 / 89,390,181
  tactical       18,442,056 / 18,442,056
  evasion        33,921,018 / 33,921,018
  has-any        1,527,267 / 1,527,267
  search edges   10,906,689 / 10,906,689
  probe make total per node                  13.13
  search-edge make per node                  1.00
```

## Cost model (microbench, ns per op, corpus-wide median)

```text
legacy probe (make->attack->unmake)  32.2 ns
make+unmake pair                     18.1 ns
is_square_attacked                   16.0 ns
fast accept (eligibility+push)       1.7 ns
```

## Historical comparison (S4.3A, pre-promotion)

| bucket | S4.3A | S4.4A |
|---|---|---|
| movegen_full_legal | 63.7% | 59.9% |
| movegen_tactical | 7.6% | 10.4% |
| movegen_evasion | 2.7% | 4.7% |
| movegen_has_any | 1.2% | 1.7% |
| SEE/qSEE | 7.1% | 10.1% |
| move_ordering | 6.8% | 9.4% |
| eval | 5.2% | 7.0% |
| TT_probe_store | 2.0% | 2.7% |

```text
S4.3A: nodes 3,637,042, wall 18.84 s, NPS ~193k, probes 203.6M (56.0/node)
S4.4A: nodes 3,637,042, wall 16.07 s, NPS ~226,381, probes 143.3M (13.1/node)
Cross-run wall/NPS comparison is indicative only (same binary profile,
same host, same corpus; host load differs across measurement sessions).
```

## Answers (S4.4A)

- **A. New largest wall bucket:** `movegen_full_legal` (59.9% of elapsed)

- **B. `movegen_full_legal` still dominant:** yes, 59.9%.

- **C. Inside full legal generation:** the legality probes collapsed
  (56.0 -> 13.1 make/unmake per node; probes are now ~15% of the full-legal
  bucket). The measured phases show pseudo move generation as the largest
  component (~37.5% of the bucket), then the per-move loop residual
  (~50.8%, mostly sampled-window inflation + loop bookkeeping; fast accepts
  ~2.8%, fallback probes ~10.2%), with the pin scan small
  (~6.6%).

- **D. Tactical/evasion/has-any:** tactical grew to 10.4%
  (from 7.6%) and still legacy-probes every pseudo move (18,442,056 probes);
  evasion 4.7%, has-any 1.7% — isolated
  fast-path candidates are plausible for tactical, but none exceeds the
  full-legal generator.

- **E. SEE/qSEE and ordering:** 10.1% and 9.4% grew
  (7.1/6.8 before) but neither overtook movegen.

- **F. Next target:** see decision block below.

## Decision (one preferred + at most one secondary)

**Preferred: pseudo move generation inside `movegen_full_legal`** — the
largest measured component of the still-dominant bucket (generation is
called 1.54x/node, ~602M pseudo moves across the run, and pseudo-gen is the
only in-bucket phase whose wall time is measured directly rather than
residual). A piece-list / bitboard attack generator (or per-piece-type
direct generation) is the cleanest next cut.

**Secondary: `movegen_tactical` fast path** — 10.4% of elapsed (up from 7.6%),
the fastest-growing bucket, and it still performs the legacy probe on every
pseudo move (18,442,056 probes). The promoted full-legal safety theorem
transplants directly (same eligibility/fallback rules on the tactical move set).

Not indicated: pin representation / pin-mask reuse (pin scan is ~6.6% of
the full-legal bucket); SEE or ordering work (both below 11%); a new TT
layout (2.6%).

