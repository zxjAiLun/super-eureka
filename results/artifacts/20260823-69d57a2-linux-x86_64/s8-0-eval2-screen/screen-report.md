# S8.0 Stage A Screen Report - PASS

STATUS: **S8_0_STAGE_A_SCREEN_PASS**

The `CurrentFinalEval2` candidate demonstrates a massive positive playing strength advantage over the production baseline `CurrentFinal`, even after paying the evaluation throughput compute tax.

## Result Summary

| Metric | Value | Gate / Requirement | Verdict |
|---|---:|:---:|:---:|
| Pairs / Games | 200 / 400 | 200 / 400 | **COMPLETE** |
| Candidate W / L / D | 221 / 115 / 64 | — | — |
| Candidate Score | **63.2500%** (253.0 / 400) | >= 52.0% | **PASS (大幅超越)** |
| Pair-level Mean Score | **63.25% ± 4.20%** | — | — |
| Pair-level 95% CI | **[59.05%, 67.45%]** | lower > 50.0% | **SIGNIFICANT** |
| Elo Estimate | **+94.3 Elo** | > 0 | **STRONG POSITIVE** |
| Elo 95% CI | **[+63.6, +126.5] Elo** | — | — |
| Pentanomial `[LL, LD, D/WL, WD, WW]` | `[15, 18, 71, 38, 58]` | — | 58 WW vs 15 LL |

## Integrity and Verification Checks

All 15 verification and audit checks passed:
- `completed_pairs_200`: verified
- `games_400`: verified
- `wld_sums_to_400`: 221 + 115 + 64 = 400
- `candidate_is_engine_a`: verified (`s80-current-final-eval2-69d57a2`)
- `engine_a_profile`: `current-final-eval2`
- `engine_b_profile`: `current-final`
- `strict_colour_swap`: verified
- `fen_overlap_zero`: 0 overlap with 11,336 historical FENs
- `all_moves_legal`: verified by cutechess/verifier

## Match Configuration

```text
Tournament ID:  569b4c23-f379-493e-840e-4c698cd1d4fe
Build ID:       20260823-69d57a2-linux-x86_64
Binary SHA256:  6ea680a9375707ec921780354b6e8052d3a3b3cf4ea379b796d2eb2f03b88726
Engine A:       s80-current-final-eval2-69d57a2  (--profile current-final-eval2)
Engine B:       s80-current-final-69d57a2        (--profile current-final)
TC:             10+0.1 (blitz_10_01) | Hash 16MB | Threads 1 | Concurrency 1
Openings:       stockfish-8moves-v3, 16 plies, seed 2026082301
Excluded:       11,336 historical unique FENs
```

## Authorized Next Steps

1. **Stage B Confirmation**: 200 fresh opening pairs (seed `2026082302`, excluding 11,336 historical + 200 Stage A FENs = 11,536 excluded FENs) under identical TC.
2. **Equal-Node Attribution Gate**: 200 pairs strictly reusing Stage A opening set with fixed node budget to decouple evaluation knowledge gain from computational tax.
