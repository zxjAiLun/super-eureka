# S7.1A — Lazy Qsearch Materialization (VERDICT: REJECTED)

## Result

**Negative.** `current-final-qsearch-lazy` is ~2-3% *slower* than
`current-final`, with no depth change. The candidate is NOT promoted.

## Evidence

| gate | result |
|---|---|
| exact tree (400 S7 + 30 S4 entries) | **PASS** — 0 mismatches (nodes/score/bestmove/PV identical) |
| throughput (550 paired reps, depth 6) | aggregate **+3.5%** wall, median paired **+2.8%**, 164 favorable / 372 unfavorable |
| depth uplift (1000ms / 3000ms) | completed depth unchanged (6 / 7); ~3% fewer nodes at fixed wall |

## Root cause

The lazy flow replaces the current "tactical movegen + (rare) has-any
fallback" with "has-any on *every* non-check node". Measured on a startpos
depth-8 search:

```
current-final:         legal_move_generations  3,887,040
current-final-qsearch-lazy: legal_move_generations  4,126,062  (+239k)
  qsearch_lazy_has_any_probes         1,003,933   (full-legal movegen)
  qsearch_lazy_standpat_cutoffs_before_movegen 374,148  (avoided tactical movegen)
```

`has_any_legal_move_profiled` is a **full-legal** move generation, which is
more expensive than the tactical-only move generation it defers. The current
design already calls `has_any` only when the tactical list is empty; the lazy
design calls it unconditionally, so the "saved" tactical work is cheaper than
the "added" has-any work.

## Implication for S7.1B

Stand-pat-before-movegen does not pay for itself while the stalemate probe
is a full-legal movegen. S7.1B (conservative delta/SEE futility pruning)
is a tree-changing node-reduction lane and is not blocked by this result;
it attacks the ~48-70% of non-check qsearch nodes that currently
stand-pat-cutoff by *pruning their capture tails* rather than deferring
their materialization.
