# S8.0 Stage B Confirmation Report - PASS

STATUS: **S8_0_STAGE_B_CONFIRMATION_PASS**

The `CurrentFinalEval2` candidate independently replicates its massive playing strength advantage on 200 fresh opening pairs (seed `2026082302`, 11,536 excluded FENs).

## Summary Table

| Metric | Stage A (Screen) | Stage B (Confirmation) | Pooled (A + B) |
|---|:---:|:---:|:---:|
| **Pairs / Games** | 200 / 400 | 200 / 400 | **400 / 800** |
| **Candidate W / L / D** | 221 / 115 / 64 | 230 / 109 / 61 | **451 / 224 / 125** |
| **Candidate Score** | **63.25%** | **65.12%** | **64.19%** |
| **Pair-level 95% CI** | [59.05%, 67.45%] | [60.99%, 69.26%] | **[61.27%, 67.11%]** |
| **Elo Estimate** | **+94.32 Elo** | **+108.49 Elo** | **+101.37 Elo** |
| **Elo 95% CI** | [+63.62, +126.54] | [+77.93, +140.77] | **[+79.60, +123.94]** |
| **Pentanomial `[LL,LD,D/WL,WD,WW]`** | `[15, 18, 71, 38, 58]` | `[13, 15, 71, 40, 61]` | **`[28, 33, 142, 78, 119]`** |
| **Candidate Nodes Ratio** | 77.85% (-22.15%) | 78.13% (-21.87%) | **77.99% (-22.01%)** |
| **Throughput Tax (NPS)** | -22.29% | -22.15% | **-22.22%** |

## Key Findings

1. **Robust Independent Replication**:
   Stage B achieved 65.12% score (+108.49 Elo), matching Stage A's 63.25% score (+94.32 Elo). The combined 400-pair evidence establishes an Elo advantage of **+101.37 Elo (95% CI: [+79.60, +123.94])**.
2. **Deterministic Throughput Tax**:
   Candidate nodes ratio is consistently ~78.0% (effective NPS ~205k vs ~263k, tax: 22.2%), proving that the positional evaluation knowledge gain overwhelmingly overcomes the search speed tax.
