# S10-E0b Closeout — Tactical Instability Audit (search/NNUE coupling)

**Status: CLOSED / PASS — evaluator weakness, not search-coupling; 1M data-scale probe GO**

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
opponent captures with a recapture-aware material net (fixed two
detector bugs during the audit: inverted recapture side, king valuation).

## Results

```
                       A (HCE)   B (NNUE)  C (NNUE conservative)
final bm blunders         35        43          42      (>=300cp net hangs)
depth-1 bm blunders       28        68          68
teacher agreement        200       174         174      (final bm == SF18 16k bm)
reversals (>=200cp flip)   1        19          19
final A-vs-B agreement            188/400
final A-vs-C agreement            187/400
```

Key observations:

1. **C ≈ B on every metric** (43 vs 42 final blunders, 174 vs 174
   teacher agreement, 19 vs 19 reversals). Disabling null move and
   futility changes NOTHING for the NNUE arm — the search's
   eval-dependent selectivity is NOT amplifying the static error.
2. **A also blunders** (35 final, 28 shallow): depth-7 bullet-search
   blunders are common to both evaluators — part is normal tactical
   horizon, part is the depth/TC budget.
3. **B's excess over A is real but moderate**: +8 final blunders,
   +40 depth-1 blunders, -26 teacher agreement, +18 reversals. This is
   the signature of a noisier static evaluator, not of a broken search.
4. **qsearch hard check**: of 68 shallow-blunder moves, the engine's own
   deeper search refuted (contained the counter-capture in its forced PV)
   63; the 5 "missed" cases were hand-verified to be (a) fair queen
   trades mislabeled by the one-ply SEE approximation, or (b) the engine
   choosing a different defense on the PV while the capture was
   searched. No case of qsearch genuinely skipping a winning capture.

## Verdict (frozen decision rules)

```
B blunders a lot, C clearly fixes it   -> NOT the case (C == B)
B and C blunder equally                -> evaluator is weak  ← MATCH
A also blunders at the same rate       -> partially (lower rate)
```

**The D1 rejection is an evaluator-strength problem.** The CurrentFinal
search strategy is not systematically misleading the NNUE evaluator;
its selectivity is calibrated well enough that removing it does not
rescue the candidate. Per the frozen rule: **1M data-scale probe GO,
architecture unchanged.**

## Detector honesty notes

* PV-membership is a weak "saw" proxy (only the principal line is
  visible); the arm-level counts above are the authoritative metric.
* The one-ply SEE approximation over-labels fair trades; two detector
  bugs were found and fixed DURING the audit (recapture side inversion,
  king valuation) — final numbers are post-fix.
* 165 cp MAE + these counts are consistent: an evaluator with that much
  static noise will (a) misrank quiet moves at depth 1 (68 blunders) and
  (b) disagree with a 16k-node teacher search more often (174 vs 200).

## Artifacts

```
tools/s10/e0_tactical_audit.py      3-arm runner + blunder detector
tools/s10/e0_tactical_final.py      final aggregation pass
results/s10/s10-e0-tactical-audit.json
results/s10/s10-e0-tactical-final.json
```
