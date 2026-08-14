# S7.2 Move Ordering Attribution (OBSERVATION ONLY)

- profile: current-final, cold 16MB TT, threads 1
- corpus: 80 S7 (d6,7) + 30 S4 (d6)
- classification: **ORDERING_NOT_PRIMARY**

## Headline
- first-move cutoff: 80.304%
- moves before cutoff: mean 1.425 median 1 p90 2
- late cutoff >=5 searched: 1.864%
- late cutoff >=9 searched: 0.279%
- fail-low all-moves nodes: 1911854 (mean searched 14.683)

## Cutoff category share / success rate (%)
- tt_hash: share 26.784 success 83.544
- promotion: share 0.339 success None
- capture: share 45.143 success None
- killer0: share 24.901 success 59.343
- killer1: share 1.409 success 15.417
- history_quiet: share 1.248 success 0.877
- other_quiet: share 0.175 success 0.059

## Quiet opportunity
- quiet searched 26813899 / available 219912912, cutoffs 2261315 (rate 8.433%)
- quiet cutoff global-index histogram: {'0': 53.412, '1': 21.024, '2_3': 20.728, '4_7': 4.067, '8_15': 0.534, '16p': 0.235}
- quiet cutoff quiet-rank histogram: {'r0': 94.683, 'r1': 3.079, 'r2_3': 1.069, 'r4_7': 0.651, 'r8_15': 0.318, 'r16p': 0.2}

## History buckets: cutoff rate (%)
- le0: 0.067
- 1_15: 3.465
- 16_63: 7.953
- 64_255: 17.281
- 256p: 54.271

## Killers / TT
- killer0 searched 2567989 cutoff rate 78.143%
- killer1 searched 559267 cutoff rate 28.528%
- TT hash searched 1961989 cutoff rate 83.544%

## LMR quiet interaction
- {'reduced_fail_low': 28743, 'reduced_research': 1023, 'reduced_eventual_cutoff': 1278}

## Depth split (cutoffs / late>=5 / fail-low / quiet searched / quiet cutoffs)
- d1: 4208812 / 88833 / 1568862 / 21097542 / 1624038
- d2: 1573211 / 19715 / 246599 / 3978061 / 535394
- d3: 242334 / 4236 / 85282 / 1404616 / 72721
- d4_5: 92798 / 1219 / 10978 / 329120 / 28021
- d6_7: 2620 / 49 / 133 / 4560 / 1141
- d8p: 0 / 0 / 0 / 0 / 0

## Interpretation (predeclared rule, section M)

Classification: **ORDERING_NOT_PRIMARY**.

1. Early cutoff dominance: 80.3% of beta cutoffs occur on the first searched
   move; only 1.86% occur after >=5 searched moves and 0.28% after >=9.
   The "ordering cost" pool is small at every depth bucket, including d4-5
   (1219 late of 92798 cutoffs, 1.3%).
2. The dominant work source is NOT late cutoffs: 1.91M all-moves fail-low
   nodes searching a mean of ~14.7 moves each with no cutoff at all. These
   are genuine fail-low nodes, not ordering failures, and must not be
   conflated with late-cutoff waste.
3. Existing signals are healthy where they fire: TT hash cutoff rate 83.5%,
   killer0 78.1% (killer1 28.5%), and the history gradient is strong and
   monotonic (0.07% -> 54.3% cutoff rate from <=0 to 256+). Quiet cutoffs
   land at quiet-rank 0 in 94.7% of cases, i.e. history already places the
   cutoff-producing quiet move first.
4. Per the predeclared decision rule this rules out a Search 2.0
   MovePicker/history candidate as the next depth lever. The evidence
   points at all-moves fail-low work, i.e. forcing/selective-depth
   techniques (null-move already present; future: verified null extensions,
   selective search of forcing lines), not re-ordering.

Caveat: the background "first-cutoff ~84% vs strong engines 90%+" note is
recorded only as context; it was NOT used as a formal threshold.
