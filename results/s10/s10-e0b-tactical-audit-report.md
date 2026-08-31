# S10-E0b Closeout — Tactical Instability Audit (search/NNUE coupling)

**Status: CLOSED / PASS (post Repair 1) — evaluator weakness, not search-coupling; 1M data-scale probe GO**

## Motivation

D1 rejected the 300k NNUE candidate (12.02% score, ~ -346 Elo) with
visibly weak shallow play. Before burning 1M teacher labels, audit
whether the search's eval-dependent selectivity (null move / futility)
amplifies the NNUE's ~165 cp static error, or whether the evaluator is
simply weak.

## Setup

400 deterministic validation positions (seed 2026083004, never holdout),
fixed depth 7, three arms:

```
A = CurrentFinal HCE                    (production search + HCE)
B = NNUE + full CurrentFinal search     (the exact D1 candidate)
C = NNUE conservative                   (B with null move + futility
                                         disabled; bench-only --diag)
```

Per-depth bestmove/score parsed from the engine's own iterative
`info depth` lines. Blunder detector: after the audited move, enumerate
opponent captures with a recapture-aware material net.

## Results

```
                          A (HCE)   B (NNUE)  C (NNUE conservative)
final bm blunders            35        43          42
depth-1 bm blunders          28        68          68
teacher_bestmove_agreement  200       174         174
root_score_flips (expl.)     1         19          19
final A-vs-B agreement              188/400
final A-vs-C agreement              187/400
```

`teacher_bestmove_agreement` is an exact-match rate against the SF18
16k-node searched bestmove — an information-rich proxy, NOT an
SF-evaluation regret metric.

### Repair 1: diagnostic-intervention telemetry

The original closeout did not show that arm B's searches actually
exercised null move / futility. Repair 1 re-ran the same 400 positions /
seed / depth for B and C, harvesting the bench harness's own selectivity
counters (`results/s10/s10-e0b-repair1-telemetry.json`):

```
B totals (400 searches):   null_move_attempts   36,410  (301/400 positions)
                           null_move_fail_highs   9,626
                           futility_pruned   38,494,594  (307/400 positions)
                           nodes            118,566,619
C totals:                  null/futility       0 / 0    (0/400 positions)
                           nodes            170,084,616
paired:                    bestmove agreement   385/400 (96.25%)
                           nodes ratio C/B     median 1.288  (p25 1.00, p75 1.676)
                           nodes changed        307/400
```

All three frozen gates hold:

1. **B exercised the interventions at scale** — 36k null attempts and
   38.5M futility prunes across 75%+ of positions (not a case of the
   diagnostic never firing);
2. **disabling them materially changed the workload** — C searched 1.29x
   median more nodes, with 307/400 positions changing their node count;
3. **B/C final bestmoves stayed 96.25% identical** — removing that much
   selectivity did NOT rescue the candidate's choices.

### Forced-root refutation check

Of B's 68 shallow material-hang candidates, 63 were explicitly refuted
by the engine's own deeper forced-root PV (the counter-capture appears
in the PV); the remaining 5 were hand-verified as detector
false-positives (fair queen trades mislabeled by the one-ply SEE
approximation, or the engine choosing a different defense while the
capture was searched). **No evidence of a missed tactical capture was
found.** This is a forced-root PV check — it does not prove the depth-1
qsearch itself searched those captures (PV membership is only the
principal line).

## Verdict (frozen decision rules)

```
B blunders a lot, C clearly fixes it   -> NOT the case (C == B, telemetry
                                          proves the intervention fired)
B and C blunder equally                -> evaluator is weak  <- MATCH
A also blunders at the same rate       -> partially (lower rate)
```

**The D1 rejection is an evaluator-strength problem.** With the Repair-1
telemetry confirming the diagnostic actually fired at scale and still
changed nothing, the search-coupling hypothesis is demoted: the
CurrentFinal search strategy is not what sank the NNUE candidate. Per
the frozen rule: **1M data-scale probe GO, architecture unchanged.**

## Metric honesty notes (Repair 1)

* `root_score_flips` is EXPLORATORY: it measures the ROOT score drop
  across a bestmove flip, not the frozen per-move forced-root reversal
  (which would re-search the abandoned move at the deeper depth). It is
  excluded from the authoritative verdict.
* `forced_root_refutation` is a PV-membership check, not a qsearch
  trace; the claim is "no evidence found", not "qsearch proven correct".
* `teacher_bestmove_agreement` is an exact-match rate, not an SF regret
  metric.
* The one-ply SEE approximation over-labels fair trades; two detector
  bugs found during the live audit are now pinned by fixture tests
  (`tools/s10/test_e0_tactical_audit.py`, 13 tests).

## Artifacts

```
tools/s10/e0_tactical_audit.py       3-arm runner + blunder detector
tools/s10/e0_tactical_final.py       final aggregation pass
tools/s10/e0_telemetry_repair.py     Repair-1 B/C selectivity telemetry
tools/s10/test_e0_tactical_audit.py  parser/detector fixture tests
results/s10/s10-e0-tactical-audit.json
results/s10/s10-e0-tactical-final.json
results/s10/s10-e0b-repair1-telemetry.json
```
