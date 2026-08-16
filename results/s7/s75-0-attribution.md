# S7.5-0 — Forcing Opportunity Attribution

STATUS: **OBSERVATION COMPLETE**

## Contract

```text
production baseline:   990aed6 (PRODUCTION_PROFILE = CurrentFinal)
observation build:     8aab036 (instrumentation only)
binary SHA256:         f27a378bf0c8d56f5734b3417158cb5ea53b6a50156e99fa479671612dac393c
G0 invariance:         30 S4 depth 6, vs production 990aed6
                       30/30 exact nodes/qsearch_nodes/score/bestmove/PV/seldepth
corpora:               80 S7 x d6/d7 + 120 R2 x d8 = 280 completed rows
search semantics:      ZERO CHANGE
```

## Aggregate funnels

### Main search

| metric | S7 d6 | S7 d7 | R2 d8 |
|---|---:|---:|---:|
| positions | 80 | 80 | 120 |
| actual main nodes | 836,986 | 3,169,128 | 12,432,578 |
| main in-check nodes | 224,626 | 893,654 | 5,298,801 |
| single-evasion raw | 19,123 | 68,321 | 656,616 |
| single-evasion depth 1 | 3,663 | 12,700 | 124,079 |
| single-evasion depth 2+ | 15,460 | 55,621 | 532,537 |
| single-evasion depth 3+ | 316 | 869 | 10,485 |
| chain length 1 | 19,123 | 68,321 | 656,616 |
| chain length 2 | 0 | 0 | 0 |
| chain length 3+ | 0 | 0 | 0 |
| checking edges searched | 224,594 | 893,620 | 5,298,788 |
| check child evasions 1 | 19,123 | 68,321 | 656,616 |
| check child evasions 2 | 40,550 | 156,123 | 1,052,046 |
| check child evasions 3+ | 164,921 | 669,176 | 3,590,126 |
| depth1 nodes | 607,620 | 2,352,641 | 8,665,775 |
| depth1 in-check | 41,224 | 137,506 | 1,043,027 |
| depth1 single-evasion | 3,663 | 12,700 | 124,079 |
| depth1 entered from checking edge | 41,222 | 137,504 | 1,043,026 |

### Qsearch

| metric | S7 d6 | S7 d7 | R2 d8 |
|---|---:|---:|---:|
| q nodes | 4,949,100 | 18,642,999 | 72,040,134 |
| q in-check nodes | 493,195 | 1,981,988 | 9,993,636 |
| q single-evasion raw | 38,882 | 185,400 | 1,276,077 |
| q single-evasion qply0 | 14,521 | 52,064 | 496,035 |
| q single-evasion qply1+ | 24,361 | 133,336 | 780,042 |
| q checking edges | 320,115 | 1,268,404 | 6,052,426 |
| q check child evasions 1 | 38,882 | 185,400 | 1,276,077 |
| q check child evasions 2 | 89,174 | 412,034 | 2,189,948 |
| q check child evasions 3+ | 365,139 | 1,384,554 | 6,527,611 |

## Key observations

1. **Single-evasion opportunity is real and heavily concentrated at the
   depth-1 horizon.**
   - 18.6-19.2% of main single-evasion nodes are `remaining depth == 1`.
   - That is the exact population where today the forced line drops into
     qsearch one ply earlier than an extension would.
   - R2 d8 contributes the largest absolute population: 124,079 depth-1
     single-evasion nodes.

2. **Consecutive single-evasion chains are essentially absent in these
   corpora.**
   - chain length 2 / 3+ counts are zero across all 280 rows.
   - This supports `S75A_FORCING_BUDGET = 2` as more than sufficient for
     the first candidate; there is no observed multi-link forced chain
     demanding a larger budget.

3. **Checking-edge funnel gives S7.5B a concrete target.**
   - Main-search checks with exactly 2 evasions:
     `7.6-8.5%` of all main in-check children (by construction equal to
     single-evasion share for evasions=1).
   - Checks with 3+ evasions are the large majority (67-75%), so the
     `evasions == 2` first candidate is naturally bounded.
   - Main and qsearch distributions are consistent, but qsearch carries
     roughly 2x the in-check population of main search.

4. **Qsearch is the larger forcing reservoir.**
   - q in-check / main in-check ratio: 2.19x (d6), 2.22x (d7), 1.89x (R2 d8).
   - S7.5A is deliberately main-search only. If it succeeds, this dataset
     supports a later, separately gated discussion of qsearch forcing
     interaction; it must NOT be silently folded into S7.5A.

5. **No extra legal generation was used for attribution.**
   - Every checking child was classified from the legal list the normal
     search already generated at that node.

## Budget recommendation (for review)

```text
S75A_FORCING_BUDGET = 2
```

Justification: no chain longer than 1 was observed in any corpus; budget 2
already covers the observed single-evasion structure with one spare unit.
This is a recommendation, not a frozen parameter: final freeze belongs to
the S7.5A pre-implementation review.

## S7.5A status

```text
S7.5-0:  OBSERVATION COMPLETE
S7.5A:   NO-GO until reviewer freezes budget/gates from this data
