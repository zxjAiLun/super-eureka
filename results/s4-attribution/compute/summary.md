# S4.0A Compute Attribution summary

Diagnostic only; no candidate, no Elo decision.

- positions: 30
- budgets (ms): [100, 500, 1000, 3000]
- profile: current-final, cold TT, 16 MB, 1 thread

| budget_ms | depth | nodes | nps | nodes_depth_proxy | qsearch_ratio | eval/node | movegen/node | make-unmake/node | tt_hit | tt_cutoff/elig | tt_dep_rej/hit | tt_bound_rej/elig | lmr_rsrch | null_rsrch | asp/iter | futility | growth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 4.0 | 17614.0 | 179720.5 | 11.242 | 0.864 | 0.91 | 2.61 | 117 | 0.232 | 0.064 | 0.712 | 0.936 | 0.000 | 0.000 | 0.000 | 1830.0 | 4.58 |
| 500 | 5.0 | 89190.5 | 182019.0 | 9.646 | 0.832 | 0.90 | 2.53 | 114 | 0.216 | 0.055 | 0.869 | 0.945 | 0.004 | 0.406 | 0.000 | 21572.5 | 4.58 |
| 1000 | 6.0 | 175366.5 | 179136.5 | 7.484 | 0.815 | 0.90 | 2.48 | 115 | 0.240 | 0.061 | 0.902 | 0.939 | 0.007 | 0.484 | 0.000 | 29111.0 | 4.58 |
| 3000 | 7.0 | 520784.0 | 177136.5 | 6.898 | 0.835 | 0.91 | 2.58 | 122 | 0.142 | 0.042 | 0.918 | 0.958 | 0.011 | 0.401 | 0.000 | 273203.5 | 4.58 |

## Key observations

- NPS is flat across budgets (~177k-182k): per-node cost is stable.
- Median depth only grows 4 -> 7 across 100 ms -> 3000 ms. The rough `nodes_depth_proxy` (nodes^(1/depth)) is ~7-11; it is a coarse proxy that includes iterative-deepening, qsearch and aborted deeper-iteration work, and is NOT comparable to another engine's reported branching factor, so it is not used here to prove 'excessive EBF'.
- make/unmake ~= 114-122 per node in middlegames (much lower, ~36-43, with far higher NPS in low-material endgames). Counters measure operation frequency, not CPU wall-time share, so this is the strongest *measured* Core-cost suspect, not a proven dominant CPU cost.
- qsearch ~= 80-86% of all nodes (~5 qsearch nodes per main-search node at 3000 ms). Strongest Search-side signal; not yet proof qsearch is mis-designed.
- TT: hit rate ~0.15-0.24, cutoffs ~0.1% of eligible hits. `tt_stores/tt_probes ~= 1` alone is not evidence of a problem. Distinguish: depth-rejected entries (this is why hit rate is low) vs depth-eligible entries whose bounds rarely cut.
- LMR re-search is low (~0-1.1%); null-move fail-highs are all re-verified (~40-48% null research rate). S4.0A does NOT show an LMR re-search explosion.
- Aspiration retries are negligible on the median.

## Preliminary top-3 compute bottleneck hypotheses (evidence-ordered)

1. Legal-move legality filtering (pseudo -> make/check/unmake): ~114-122 make/unmake per node, the strongest measured Core-cost suspect.
2. qsearch dominates the tree (~80% of nodes) together with a high `nodes_depth_proxy`: the tree is large relative to a more selective engine.
3. TT under-utilization: hit rate 15-24%, cutoffs ~0.1%; depth-rejected hits dominate. Transposition reuse is weak.

Overall this is a mixed result: the cheapest, clearly-attributable node-cost culprit is movegen legality filtering; the binding constraint on completed depth is tree size (qsearch share and high branching proxy). Evaluation quality remains unmeasured by S4.0A.
