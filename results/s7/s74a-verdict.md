# S7.4A — LMR-on-Null-Window Candidate Verdict

BASE: `6cf45063ef4c8eb2db03069149327d843fe47fba` (implementation commit `d50729a`)

## Status

- S7.3 observation implementation: APPROVED (data used with repairs)
- S7.3 diagnosis: `SELECTIVITY_TOO_CONSERVATIVE`, precise mechanism
  `LMR suppressed by null-window/PVS coupling`
- S7.4A candidate: `current-final-lmr-null-window`
  (CurrentFinal + apply EXISTING LMR policy to null-window nodes;
  NO threshold/formula changes, no LMP, no null-move changes)

## Mechanism summary (S7.3 repair values)

- `late_move_reduction()` proposed R1 on 232,058 depth>=4 quiets; only
  26,982 were actually reduced (11.6%) because `pvs_child_window()`
  returns `ChildWindow::Full` when the caller is already null-window,
  discarding the reduction.
- S7.4A routes those moves through reduced null-window search with
  full-depth fail-high verification.

## Fixed-depth node gate (80 S7 corpus, cold 16MB, threads 1)

| depth | baseline nodes | candidate nodes | reduction | verdict |
|---|---|---|---|---|
| 6  | 9,941,996  | 5,785,742  | **-41.805%** | STRONG_MECHANISM |
| 7  | 43,104,125 | 21,810,945 | **-49.399%** | STRONG_MECHANISM |
| 8 (20-pos subset) | 50,168,172 | 12,286,278 | **-75.510%** | STRONG_MECHANISM |

Predeclared gate: <= -15% STRONG. All depths qualify.

The d8 subset exercises the existing R2 rule (remaining depth >= 7,
move index >= 8) which S7.3 could not reach at root d6/d7; at d8 the
reduction compounds through the tree (-75.5%) with LMR proposed 143,892 /
applied 134,876 (93.7%).

- LMR proposed at d6: 65,835; applied on null-window: 58,195 (88.4%)
- LMR proposed at d7: 151,971; applied on null-window: 133,011 (87.5%)
- Zero-reduction rows (6/80 at d7) are sparse piece/pawn endgames where
  the existing `non_pawn_material >= 4` guard never proposes LMR —
  identical trees are the intended conservative behavior, not a bug.
- d8 subset (first 20 positions, exercising R2): see below.

## Teacher challenge (S6 corpus, 178 rows / 176 evaluated at depth 6; 2 rows excluded as engine errors, fixed-depth)

- baseline teacher-bestmove matches: 33
- candidate teacher-bestmove matches: **34** (+1)
- cp divergence >= 100 / 300 / 500: **0 / 0 / 0**
- 15 positions with small score differences, all < 100cp
- 3 mate-labelled positions: candidate identical to baseline
  (same mate scores, same bestmoves); no mate transitions
- baseline-only matches: 0; candidate-only matches: 1

## Search stability gate (d6 -> d7 on 80 S7 corpus)

| metric | baseline | candidate |
|---|---|---|
| bestmove changes      | 19 | **15** |
| >=200cp reversals     |  1 | 1 |
| cp <-> mate transitions | 0 | 0 |

Candidate is not less stable; bestmove flips actually decrease.

## Production invariance

- current-final, 30 S4, depth 6, before/after implementation:
  nodes/score/bestmove/PV **30/30 exact match**.
- fmt / clippy -D warnings / debug tests / release tests: clean.

## Fixed-wall depth gate (80 S7 positions, interleaved A/B)

| movetime | median depth A→B | mean depth A→B | median seldepth A→B | gained | lost |
|---|---|---|---|---|---|
| 1000 ms | 7 → 7 | 6.775 → 7.362 | 17 → 18 | **44 / 80** | **0** |
| 3000 ms | 7 → **9** | 7.513 → 8.512 | 18 → 19.5 | **62 / 80** | **0** |

Qualification minimum was 10/80 gained with gained > lost; the candidate
gains completed depth on 55% (1s) to 78% (3s) of positions and loses
depth on none. Median completed depth at 3s rises a full two plies.

## Verdict

**QUALIFIED_FOR_ARENA — STRONG.**

1. Fixed-depth node reduction: **-41.8% (d6) / -49.4% (d7) / -75.5% (d8 subset)** — far beyond the 15% STRONG threshold.
2. Fixed-wall depth evidence: decisively positive (44/80 and 62/80 gained, 0 lost; 3s median depth +2).
3. Teacher challenge: not worse — matches 33 → 34, zero >=100cp divergences, mate-labelled positions identical.
4. No mate/tactical regression: no cp<->mate transitions, mate scores unchanged; search-stability gate equal-or-better (bestmove flips 19 → 15).
5. Production CurrentFinal unchanged: 30/30 S4 depth-6 exact (nodes/score/bestmove/PV).
6. fmt / clippy -D warnings / debug + release tests: clean.

Per the predeclared contract: STOP after offline evidence. No promotion,
no LMR threshold tuning, no LMP, no null-move changes, no forcing
extensions, no Arena run before review.
