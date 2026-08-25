# S8.0 3+2 (blitz_3_2) Time Control Validation Report - PASS

STATUS: **S8_0_BLITZ32_VALIDATION_PASS**

The `CurrentFinalEval2` candidate demonstrates that its massive playing strength advantage is robust under longer time controls and deeper search, achieving +109.3 Elo at 3m+2s (64 pairs / 128 games).

## Summary Table

| Metric | 3+2 Value | Contract / Baseline | Verdict |
|---|:---:|:---:|:---:|
| **Pairs / Games** | 64 / 128 | 64 completed pairs | **COMPLETE** |
| **Candidate W / L / D** | 69 / 30 / 29 | — | — |
| **Candidate Score** | **65.23%** (83.5 / 128) | >= 50.0% | **ROBUST POSITIVE** |
| **Pair-level 95% CI** | **[57.02%, 73.45%]** | lower > 50.0% | **SIGNIFICANT** |
| **Elo Estimate** | **+109.33 Elo** | > 0 | **NO TC EROSION** |
| **Elo 95% CI** | **[+49.12, +176.73] Elo** | — | — |
| **Pentanomial `[LL,LD,D/WL,WD,WW]`** | `[7, 6, 14, 15, 22]` | — | 22 WW vs 7 LL |
| **Avg Game Duration** | **18.48 min/game** (1109s) | 3m + 2s increment | — |
| **Avg Pair Duration** | **36.97 min/pair** (2218s) | — | — |

## Key Findings

1. **No TC Erosion**: At 3+2 (average ~18.5 minutes per game, 37 minutes per pair), the Eval2 candidate scores **65.23% (+109.3 Elo)**, fully matching the +94 ~ +108 Elo measured under 10+0.1.
2. **Positional Advantage Extrapolates**: The 22.3% throughput compute tax does NOT cause regression in longer time controls; deeper tactical depth further solidifies positional advantages.
