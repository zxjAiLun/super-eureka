# S4.3A — Core Wall-Time Attribution summary

Method: bench-only deterministic sparse sampling (1/256 calls) on coarse
hot-path operations, over the 30-position S4.0A corpus (depth 6, cold 16MB TT,
1 thread, release CurrentFinal). External sampling profiler unavailable on this
host (wpr without WPA, no VS/perf), so the sampled-timing fallback is the
primary measurement; the movegen microbench is supporting evidence only.

## Ranked wall-time buckets (extrapolated from samples; share of total elapsed)

| rank | bucket | share | note |
|---|---|---|---|
| 1 | movegen_legal (full legal generation incl. legality filtering) | **63.7%** | dominant |
| 2 | movegen_tactical (qsearch tactical movegen) | 7.6% | |
| 3 | SEE / qSEE pruning | 7.1% | |
| 4 | move ordering (negamax + qsearch stable sorts) | 6.8% | |
| 5 | eval (static evaluate) | 5.2% | |
| 6 | movegen_evasion | 2.7% | |
| 7 | TT probe + store | 2.0% | |
| 8 | movegen_has_any (stalemate probes) | 1.2% | |
| - | other / unattributed | 3.6% | SearchPath bookkeeping + control |
| - | **movegen total** | **~75.2%** | 63.7+7.6+2.7+1.2 |

## Legality-filter vs recursive traversal split

- legality probe make/node: **56.0**
- legality probe unmake/node: **56.0**
- search-edge make/node: **1.0**, search-edge unmake/node: **1.0**
- **legality = 98.2% of all make/unmake** -> the ~120 make+unmake per node
  (S4.0A) is almost entirely legality filtering, NOT recursive traversal.

## Movegen sub-attribution (microbench, supporting evidence, ns per op)

| class | pseudo moves | pseudo gen | make+unmake | is_sq_attacked | filter/move |
|---|---|---|---|---|---|
| opening | 20 | 141 | 21 | 16 | 46 |
| normal mid | 35 | 156 | 22 | 14 | 28 |
| tactical | 48 | 249 | 15 | 15 | 28 |
| closed | 35 | 138 | 17 | 14 | 35 |
| KQK endgame | 26 | 75 | 17 | 17 | 37 |

Per movegen call the legality filter is ~7x pseudo generation
(~30ns/move x 35-48 pseudo moves vs ~150ns pseudo gen): the make -> king-attack
test -> unmake per pseudo move dominates movegen_legal.

## Profiling overhead

Timing on vs off on 5 corpus positions at depth 6: -0.5% (within noise, well
under the 5% limit at 1/256 sampling).

## Interpretation

- movegen (75.2%) with legality filtering (the 56 probe-make/node, 98.2% of
  make/unmake) is THE dominant wall-time cost, far above the 25-30% bar.
- eval (5.2%), ordering (6.8%), SEE (7.1%), TT (2.0%) are minor by comparison.
- The old ~120 make/unmake per node intuition is CONFIRMED as a wall-time
  reality: it is legality filtering, not recursive traversal.
- Candidate selection is data-driven, not a forced bitboard rewrite: the
  cheapest high-leverage fix is the legality test path
  (make -> is_square_attacked -> unmake per pseudo move).

## ONE recommended S4.3B candidate: LEGALITY_FILTER

Target: reduce the per-pseudo-move legality test cost in legal move
generation (and its qsearch variants). Options for S4.3B (not implemented):
attack-table / precomputed king-danger caching, or legality-by-construction
generation (pinned-piece filtering before the per-move make test). The
microbench isolates the make+attack+unmake triple as the unit to attack.
