# S4.1 — Root Quiet History Ordering: local gate report

Candidate: `current-final-root-history` (SearchProfile::CurrentFinalRootHistory).
Baseline: CurrentFinal. S4.0 closed APPROVED (direction: Search guidance).

## Candidate behavior

Exactly CurrentFinal plus, after each completed iteration:
- previous best stays at root index 0;
- only the remaining QUIET root moves are sorted by
  `history[color][from][to]` descending, stable;
- no root killers, no static-eval ordering, no history-update changes,
  no tactical reordering, no move added/dropped.

CLI: `--profile current-final-root-history` (UCI + bench). Production default
stays CurrentFinal.

## Strict SEARCH_LIKE cohort

- primary: SEARCH_LIKE + teacher quiet + root rank >= 8 -> **54**
- stricter: + delta[d7] >= -30 cp -> **49** (subset of primary)

## Baseline vs candidate (teacher metrics, 100/500/1000 ms)

### Cohort (54)
| budget | top1 cf/rh | top3 cf/rh | median loss cf/rh | delta (improve/regress/unchanged) | median depth cf/rh |
|---|---|---|---|---|---|
| 100 | 3/3 | 9/9 | 73.5/73.5 | 0/0/54 | 4.0/4.0 |
| 500 | 0/0 | 1/1 | 79.5/79.5 | 0/1/53 | 5.5/6.0 |
| 1000 | 0/1 | 1/2 | 84.5/79.5 | 3/1/50 | 6.0/6.0 |

The mechanism win is real but weak and late: only at 1000 ms (3 improves,
median loss -5 cp). At 100-500 ms, where the effect was predicted strongest,
top1/top3 and median loss are identical.

### Broad regression sample (150 untouched positions, disagreements excluded)
| budget | top1 cf/rh | top3 cf/rh | median loss cf/rh | delta (improve/regress/unchanged) |
|---|---|---|---|---|
| 100 | 63/64 | 111/109 | 0.5/1.0 | 0/2/138 |
| 500 | 68/68 | 119/117 | 0.0/0.0 | 0/1/139 |
| 1000 | 70/70 | 122/120 | 0.0/0.0 | 0/1/139 |

Essentially neutral; slight top-3 dip (2-3 positions, within noise).

## S4.0A compute A/B (30 positions, 500/1000/3000 ms, repeat 3)

Fixed-depth deterministic check (depth 6, cold):
- nodes cf vs rh: 175806/176192, 202475/199485, 56175/56302, 73281/73281
- per-node NPS delta: -1.8% / +3.8% -> effectively NEUTRAL (no added per-node
  cost; the candidate adds no evaluation or move generation).

Time-limited run (500/1000/3000): depth 5/5, 6/5, 6.5/6 and NPS -/+ deltas are
abort-phase + machine-state noise, not a systematic regression (fixed-depth
node counts and per-node cost are equal).

## Root-order mechanism observations

The mechanism barely fires: the missed quiet moves are exactly the ones that
NEVER failed high in the baseline search, so their history scores are ~0, and
sorting by history is a near no-op relative to the baseline order. The moves
with high history are already surfaced by the previous-best lift. Root history
feedback cannot surface unknown-good quiet moves - it only reorders moves that
already proved themselves.

## Verdict: REJECTED

Acceptance gate requires a clear improvement on the strict SEARCH_LIKE cohort
plus no broad regression. The candidate shows only 3/54 improves at 1000 ms
(none at 100-500 ms), a slight broad top-3 dip, and no depth/quality gain that
would justify promotion. Per the gate, Root Quiet History Ordering is CLOSED;
no additional heuristics are stacked onto it.

The profile and unit tests remain for historical reproducibility (candidate-only,
bench-selectable, never the UCI default). Next S4.1 hypothesis should target a
signal that can surface unknown quiet moves (history feedback cannot), e.g.
root ordering by a broader/computed signal or search-allocation changes.
