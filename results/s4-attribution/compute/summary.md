# S4.0A Compute Attribution summary

Diagnostic only; no candidate, no Elo decision.

- positions: 30
- budgets (ms): [100, 500, 1000, 3000]
- profile: current-final, cold TT, 16 MB, 1 thread

| budget_ms | depth | nodes | nps | ebf_approx | qsearch_ratio | eval/node | movegen/node | make-unmake/node | tt_hit | tt_cutoff | lmr_rsrch | null_rsrch | asp_retry/iter | futility |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 4.0 | 17614.0 | 179720.5 | 11.242 | 0.864 | 0.907 | 2.614 | 116.722 | 0.232 | 0.002 | 0.000 | 0.000 | 0.000 | 1830.0 |
| 500 | 5.0 | 89190.5 | 182019.0 | 9.646 | 0.832 | 0.901 | 2.526 | 113.665 | 0.216 | 0.002 | 0.004 | 0.406 | 0.000 | 21572.5 |
| 1000 | 6.0 | 175366.5 | 179136.5 | 7.484 | 0.815 | 0.896 | 2.479 | 114.695 | 0.240 | 0.001 | 0.007 | 0.484 | 0.000 | 29111.0 |
| 3000 | 7.0 | 520784.0 | 177136.5 | 6.898 | 0.835 | 0.909 | 2.584 | 121.801 | 0.142 | 0.001 | 0.011 | 0.401 | 0.000 | 273203.5 |

## Key observations (raw sums, 3000 ms budget, all 30 positions x 3 repeats)

- NPS is flat across budgets (~177k–182k): per-node cost is stable and consistent.
- Median depth only grows 4 → 7 across 100 ms → 3000 ms; effective branching
  factor (nodes^(1/depth)) is ~6.9–11.2, i.e. very high. Depth growth is limited
  by **tree size / EBF**, not by an unstable node cost.
- **make/unmake ≈ 114–122 per node** (middlegame ~104–125; endgames only ~36–43).
  This is the dominant per-node operation: pseudo-legal move generation followed
  by make→check→unmake legality filtering.
- **qsearch ≈ 80–86% of all nodes**; qsearch SEE prune rate ~0.42.
- **TT hit rate only ~0.15–0.24**, TT cutoff ~0.1%; yet stores happen on ~99% of
  probes and ~13% of probes are rejected for depth. TT is under-utilized.
- LMR research rate ~0.40–0.48 at longer budgets (many reductions are re-searched);
  null-move fail-highs are all re-verified (researches == fail_highs).
- aspiration retries are negligible on the median.

## Preliminary top-3 compute bottleneck hypotheses (S4.0A, evidence-ordered)

1. **Legal-move legality filtering** (pseudo → make/check/unmake): ~114–122
   make-unmake per node, the dominant per-node CPU cost (Core node cost).
2. **qsearch dominates the tree** (~80% of nodes) together with a high EBF
   (~7–11): the search tree is far larger than a selective engine should be.
3. **TT under-utilization** (hit ~15–24%, cutoff ~0.1%, stores on ~99% of probes):
   weak transposition reuse inflates the tree.

Overall this looks like a **mixed** result: the most concrete, cheaply attributable
node-cost culprit is movegen legality filtering (make/unmake), while the binding
constraint on completed depth is tree size (EBF) and qsearch share. Both point away
from evaluation being the primary S4.1 target; the first cut leans toward Core
node-cost (movegen/legality) with Search guidance (ordering / EBF) close behind.
