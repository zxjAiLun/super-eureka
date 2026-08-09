# S4.1b — Root Previous-Iteration Score Ordering: local gate report

Candidate: `current-final-root-prev-score` (SearchProfile::CurrentFinalRootPrevScore).
Baseline: CurrentFinal. S4.1 Root Quiet History Ordering is CLOSED (REJECTED).

## Candidate behavior

Exactly CurrentFinal plus, after each completed iteration:
- previous best stays root index 0;
- only the remaining QUIET root slots are sorted by the previous completed
  iteration's `RootIteration.move_scores` (descending, stable);
- tactical slots unchanged; no history/killer/static-eval/threat signal;
  no PVS or re-search changes; no move added/dropped.

`move_scores` is treated as a scheduling signal only (mix of full candidate
values and scout fail-lows), never as an exact evaluation.

CLI: `--profile current-final-root-prev-score` (UCI + bench). Production default
stays CurrentFinal. Not combined with the RootHistory candidate.

## Strict cohort (54 = SEARCH_LIKE + quiet + root rank >= 8; stricter 49)

| budget | top1 cf/ps | top3 cf/ps | median loss cf/ps | delta (improve/regress/unchanged) | median depth cf/ps |
|---|---|---|---|---|---|
| 100 | 4/4 | 10/11 | 72.0/72.0 | 1/1/52 | 4.0/4.0 |
| 500 | 0/1 | 2/3 | 79.5/81.5 | 2/1/51 | 5.0/5.0 |
| 1000 | 0/1 | 0/2 | 84.5/81.5 | 3/1/50 | 6.0/6.0 |

The mechanism win is real but again weak and late: 3/54 improves at 1000 ms
(median loss -3 cp), 1-2 top1/top3 lifts at 500/1000 ms. This is NOT materially
beyond the RootHistory effect (3/54 at 1000 ms) that closed S4.1.

## Broad regression (150 untouched)

| budget | top1 cf/ps | top3 cf/ps | median loss cf/ps | delta (improve/regress/unchanged) |
|---|---|---|---|---|
| 100 | 64/64 | 109/113 | 0.0/0.0 | 1/2/137 |
| 500 | 66/65 | 115/115 | 0.0/0.0 | 0/1/139 |
| 1000 | 70/68 | 122/117 | 0.0/0.0 | 0/4/136 |

Slightly negative at 1000 ms (top3 -5, 4 regresses): not neutral/positive.

## Fixed-depth node A/B (depth 6, cold)

nodes cf vs ps: 56175/56175, 175806/175806, 202475/202475, 73281/73297.
Per-node cost and tree size are effectively identical -> compute cost neutral.

## Mechanism observations

The previous-iteration root scores mostly carry scout fail-low upper bounds and
the previous best's value. The missed quiet moves scored poorly in the previous
iteration too (that is why they were missed), so ranking by previous scores
mostly re-orders the also-rans; the previous-best lift already captures the
strongest part of the signal. The tiny top1/top3 lifts confirm the signal has
some scheduling value, but not enough to move the cohort.

## Verdict: REJECTED

Acceptance gate requires (a) cohort improvement materially beyond RootHistory's
3/54, (b) broad neutral-or-positive, (c) compute neutral-or-better. Observed:
(a) not met (same 3/54 at 1000 ms), (b) not met (small broad top3 regression at
1000 ms), (c) met (neutral). S4.1b is CLOSED; no features stacked onto it.

Profile + unit tests kept for historical reproducibility (candidate-only, never
the UCI default). Root-quiet reordering by either history or previous scores
does not surface the unknown-good quiet moves S4.0B identified; the next S4.1
hypothesis needs a different lever (not root quiet reordering).
