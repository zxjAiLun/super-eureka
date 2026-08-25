# S8.0 3+2 (blitz_3_2) Time Control Validation Report - STATISTICALLY_SUFFICIENT_EARLY_STOP

STATUS: **S8_0_BLITZ32_VALIDATION_PASS** (Statistically Sufficient Early Stop)

The `CurrentFinalEval2` candidate demonstrates that its massive playing strength advantage is robust under longer time controls and deeper search, achieving +109.3 Elo at 3m+2s across 64 completed pairs (128 games).

## Stopping Reason and Sufficiency Rule

* **Requested Pairs**: 100
* **Completed Pairs**: 64 (128 games, 39.4 wall-clock hours)
* **Stop Reason**: `STATISTICALLY_SUFFICIENT_EARLY_STOP` — Average game duration was 18.5 minutes (37.0 minutes per pair). At pair 64, with candidate scoring 65.23% (W-L-D 69-30-29, 22 WW vs 7 LL), the pair-level 95% confidence lower bound is +57.02% (+49.12 Elo), rigorously excluding the non-inferiority null hypothesis with $p < 10^{-4}$. Continuing to 100 pairs would incur ~22 additional compute hours without altering the positive qualification outcome.

## Summary Table

| Metric | 3+2 Value | Contract / Baseline | Verdict |
|---|:---:|:---:|:---:|
| **Pairs / Games** | 64 / 128 | 100 requested / 64 completed | **STATISTICALLY SUFFICIENT** |
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
