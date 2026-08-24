# S8.0 Formal Pentanomial SPRT Report - ACCEPT_H1

STATUS: **SPRT_ACCEPT_H1**

The `CurrentFinalEval2` candidate officially passes the formal pentanomial SPRT against `CurrentFinal` baseline at pair 299, qualifying for production promotion.

## Match & SPRT Summary

| Metric | Formal SPRT Value | Contract / Bounds | Verdict |
|---|:---:|:---:|:---:|
| **Status** | **SPRT_ACCEPT_H1** | `H0: 0.0 Elo, H1: +8.0 Elo` | **PROMOTION QUALIFIED** |
| **Pairs / Games** | 299 / 598 | Max 3,000 pairs | **TERMINATED (Pair 299)** |
| **Candidate W / L / D** | 323 / 202 / 73 | — | — |
| **Candidate Score** | **60.12%** | >= 50.0% | **SIGNIFICANT** |
| **Pair-level 95% CI** | **[56.52%, 63.71%]** | lower > 50.0% | **PASS** |
| **Elo Estimate** | **+71.28 Elo** | > +8.0 Elo ($H_1$) | **ACCEPTED H1** |
| **Elo 95% CI** | **[+45.57, +97.80] Elo** | — | — |
| **Pentanomial `[LL,LD,D/WL,WD,WW]`** | `[32, 24, 118, 41, 84]` | — | 84 WW vs 32 LL |
| **Candidate Nodes Ratio** | **77.85% (-22.15%)** | — | 916M vs 1,177M nodes |
| **Throughput Tax (NPS)** | **-22.41%** | — | 203,204 vs 261,906 NPS |

## Integrity & Provenance

```text
Tournament ID:  2c9e0d90-effb-4c70-bea5-de151beb3111
Build ID:       20260823-69d57a2-linux-x86_64
Binary SHA256:  6ea680a9375707ec921780354b6e8052d3a3b3cf4ea379b796d2eb2f03b88726
Engine A:       s80-current-final-eval2-69d57a2  (--profile current-final-eval2)
Engine B:       s80-current-final-69d57a2        (--profile current-final)
TC:             10+0.1 (blitz_10_01) | Hash 16MB | Threads 1 | Concurrency 1
Openings:       stockfish-8moves-v3, 16 plies, seed 2026082303
Excluded:       11,736 historical unique FENs (overlap verified 0)
```
