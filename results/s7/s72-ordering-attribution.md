# S7.2 Move Ordering Attribution (OBSERVATION ONLY)

- profile: current-final, cold 16MB TT, threads 1
- corpus: 80 S7 (d6,7) + 30 S4 (d6)
- classification: **ORDERING_NOT_PRIMARY**

## Headline
- first-move cutoff: 80.304%
- moves before cutoff: mean 1.425 median 1 p90 2
- late cutoff >=5 searched: 1.864%
- late cutoff >=9 searched: 0.279%
- no-beta-cutoff move-loop nodes: 1911854 (mean searched 14.683)

## Cutoff category share / success rate (%)
- tt_hash: share 26.784 success 83.544
- promotion: share 0.339 success n/a (no searched-opportunity denominator)
- capture: share 45.143 success n/a (no searched-opportunity denominator)
- killer0: share 24.901 success 78.143 (opportunity rate, see Killers/TT)
- killer1: share 1.409 success 28.528 (opportunity rate, see Killers/TT)
- history_quiet: share 1.248 success n/a (bucket rates below)
- other_quiet: share 0.175 success n/a (bucket rates below)

NOTE (P2 fix): only TT hash, killer0/killer1, and the history-score buckets
have a consistent opportunity denominator. The category table has precedence
TT > promotion > capture > killer0 > killer1 > history > other quiet, so a
cutoff move that is BOTH the TT hash move and killer0 is counted only in the
TT category, while killer0's searched denominator still includes it; the
category-share rows therefore must not be read as strict category success
rates. The 59.343%/15.417% killer-category figures from the raw JSON suffer
exactly this overlap and are superseded by the opportunity rates below.

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

## Depth split (cutoffs / late>=5 / no-beta-cutoff / quiet searched / quiet cutoffs)
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
2. The dominant work source is NOT late cutoffs: 1.91M no-beta-cutoff
   move-loop nodes search a mean of ~14.7 moves each (~28.1M of the ~36.7M
   total searched moves, ~76%). P2-1 wording fix: "no-beta-cutoff" is the
   accurate name — this population includes PV/wide-window nodes, nodes with
   partial alpha improvement, and nodes where some moves were futility-pruned;
   it is NOT strictly "all-moves fail-low". Better ordering could still raise
   alpha earlier in such nodes and affect later PVS/futility behavior; what
   the data rules out is only the classic "a cutoff move exists but is ranked
   too late" failure mode.
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

Caveat (P2 methodology note): the aggregate totals mix 80 S7 positions
(depth 6+7) with the 30 S4 positions (depth 6). The S7 corpus (A1) already
contains those 30 S4 positions, so their d6 contribution is double-weighted
in the totals. This does not affect the verdict: the depth-split shows
late-cutoff is only ~1.3-1.9% even in the d4-7 buckets that would remain
untouched by de-duplication, far from any ORDERING_MAJOR threshold. Future
attribution harnesses should keep "diagnostic aggregate = 80 unique S7
corpus only" and treat the S4 30 as a separate performance appendix.
