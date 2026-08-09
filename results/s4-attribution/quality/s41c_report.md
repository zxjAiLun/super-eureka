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

### Depth 5 (LMR, futility and qSEE all ACTIVE; null move inactive)
At d5 the raw artifact already records ~604,914 futility prunes and disabling
futility raises median nodes from ~27.4k to ~38.4k, so futility demonstrably
fires at d5. Only null move does not fire at d5 (it needs depth >= 5 in a
null-window node and first appears at d6).

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

## Phase B: root-full-window — full-window / PV-context STRESS diagnostic
(not a clean isolation)

Implementation-wise the diagnostic only forces `ChildWindow::Full` at the root
gate, but search-behavior-wise the downstream PV/non-PV topology is NOT
equivalent: d6 counters show null attempts 1362 -> 0 and LMR reductions
5246 -> 40569 while median nodes fall 105909 -> 71724. The code gate for
non-root PVS/LMR/futility/null/qSEE is untouched, but their actual activation
behavior changes materially. So Phase B demonstrates only:

> giving ALL root candidates stronger full-window/PV-like treatment still does
> not recover the cohort (1/54).

It does NOT strictly prove that the root PVS window itself is individually
exonerated. No further repair of Phase B was performed (enough negative
evidence already exists).

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
  (0-1/54), nor the full-window/PV-context stress (1/54) explains SEARCH_LIKE.
- The 54-position SEARCH_LIKE label should now be read as
  "search-context-sensitive disagreement" (SEARCH_CONTEXT_LIKE): the
  classification is about forced-vs-normal search behavior differing, not
  evidence of a specific search-heuristic bug.
- Remaining hypotheses (deferred): mechanism INTERACTIONS; d7+ selectivity
  (unmeasurable within the 40s bench timeout at d7); the EVAL_LIKE residual
  (11 quiet positions, flat ~-80cp at depth 3..7) -> handed to S4.2A.

No feature is removed from production based on this diagnostic.

Artifacts: results/s4-attribution/quality/s41c_attribution.jsonl (depths 5/6,
all configs, per-position counters aggregated).
