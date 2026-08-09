# S4.1cA — Non-PV Selectivity Attribution report

S4.1 (RootHistory) and S4.1b (RootPrevScore) are CLOSED / REJECTED. This phase
attributed the SEARCH_LIKE cohort to specific non-PV selective mechanisms using
the existing bench-only diagnostics in NORMAL (unforced) search.

Cohort: 54 strict SEARCH_LIKE positions (SEARCH_LIKE + teacher quiet + root
rank >= 8); stricter-49 subset (delta[d7] >= -30). No --forced-root, no
--target-root on any run (no diagnostic ordering). Fixed depth 5 and 6
(depth 7 was disproportionately expensive and showed no signal at d5/d6, so it
was skipped per plan).

## Phase A: individual selective mechanisms (normal search, per config vs baseline)

### Depth 5 (null and futility do not fire yet; LMR active)
| config | top1 | top3 | median loss | recovered/54 | >50cp improve | >100cp regress |
|---|---|---|---|---|---|---|
| baseline | 1 | 5 | 77.0 | - | - | - |
| no-lmr | 1 | 5 | 74.5 | 0 | 1 | 0 |
| no-futility | 1 | 5 | 77.0 | 0 | 0 | 0 |
| no-null | 1 | 5 | 77.0 | 0 | 0 | 0 |
| no-qsee | 1 | 5 | 74.5 | 0 | 0 | 0 |

### Depth 6 (all four mechanisms active)
| config | top1 | top3 | median loss | recovered/54 | recovered/stricter49 | >50cp improve | >100cp regress | median nodes |
|---|---|---|---|---|---|---|---|---|
| baseline | 4 | 5 | 75.5 | - | - | - | - | 105909 |
| no-lmr | 5 | 6 | 74.5 | **1** | 1 | 0 | 0 | 130677 |
| no-futility | 4 | 5 | 75.5 | 0 | 0 | 0 | 0 | 155063 |
| no-null | 4 | 5 | 75.5 | 0 | 0 | 0 | 0 | 103697 |
| no-qsee | 4 | 5 | 75.5 | **1** | 1 | 0 | 0 | 172744 |

None of the four mechanisms explains the cohort: recovery is 0-1/54 at depth 6
even though all mechanisms were active (LMR reductions 5246, qSEE tests 6.77M,
null attempts 1362).

## Phase B: root-full-window diagnostic (every root move full-window, no root
scout + conditional re-search; non-root PVS/LMR/futility/null/qSEE unchanged)

| depth | top1 | top3 | median loss | recovered/54 | >50cp improve | >100cp regress | median nodes |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 5 | 77.0 | 0 | 0 | 0 | 31121 |
| 6 | 2 | 4 | 73.0 | **1** | 4 | 0 | 71724 |

Root PVS/PV-context is not the causal lever either: full-window root recovers
1/54 (top1/top3 actually dip slightly at d6: 4->2 top1, 5->4 top3), though
median loss improves a little (75.5->73.0) with 4 positions improving >50cp.

## Interpretation

- LMR is now exonerated on QUALITY grounds at depth 5-6 as well (not just the
  S4.0A re-search/compute exoneration): disabling it recovers 1/54.
- Neither root ordering (S4.1/S4.1b, 3/54), nor any single non-PV mechanism
  (0-1/54), nor the root PVS window context (1/54) explains SEARCH_LIKE.
- The forced-root "PV treatment" hypothesis is weakened: even giving every
  root move a full window at equal depth does not surface the teacher moves.
- Remaining hypotheses (in order): (1) mechanism INTERACTIONS (e.g. LMR +
  futility + ordering jointly); (2) deeper depths (d7+, where LMR reductions
  grow and null/futility bite harder) - not measurable here within the 40s
  bench timeout; (3) the EVAL_LIKE residual (11 quiet positions, flat
  ~-80cp at depth 3..7) which was and remains the only consistently positive
  attribution signal in S4.0B.

No feature is removed from production based on this diagnostic. The 54-position
SEARCH_LIKE cohort is a correlation (root rank + quiet + forced-root
equivalence), not a proven causal lever for any single mechanism.

Artifacts: results/s4-attribution/quality/s41c_attribution.jsonl (depths 5/6,
all configs, per-position counters aggregated).
