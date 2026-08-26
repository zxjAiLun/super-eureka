# S9-B2: Leave-One-Out (LOO) Feature Family Attribution Report

**Date**: 2026-08-25  
**Binary SHA-256**: `45d1a895ebc8a6643dce3454d7b421111cfeaab2f5e55537c8851ffd318933ba`  
**Build ID**: `20260825-96d1a69-linux-x86_64` (Git SHA: `96d1a69b2d884b3f78703d8c87c973dff9eb7830`)  
**Evaluation Protocol**: Fixed-sample screening (128 pairs / 256 games per family, TC `10+0.1`, Hash 32MB, Threads 1, Concurrency 1)  
**Common Opening Design**: 6 LOO tournaments evaluated against identical 128 opening indices derived from `stockfish-8moves-v3` with fixed random seed `2026082501`.  
**Attribution Metric**: $\Delta\text{Elo}_X = \text{Elo}(\text{Full}) - \text{Elo}(\text{NoX})$ (Engine A = `CurrentFinal`, Engine B = `CurrentFinalNoX`).

> **Methodological Note on Screening Sample Size ($N=128$ pairs)**:  
> As expected for an exploratory $N=128$ screening design, the 95% Elo confidence intervals for all 6 individual families currently span zero. Point estimates are reported strictly to rank and prioritize candidate hypotheses for targeted confirmatory testing (S9-C), not as definitive asymptotic valuations.

---

## 1. Executive Summary & Attribution Matrix

| Family ($X$) | Wins | Losses | Draws | Score (%) | Pentanomial `[0-2, 0.5-1.5, 1-1, 1.5-0.5, 2-0]` | $\Delta\text{Elo}_X$ (Full − NoX) | 95% Pair CI (Score / Elo) | Removable Cost (`median_ns`) | Cost Tier | Efficiency Proxy ($\Delta\text{Elo}/\text{ns}$) | Screening Finding / Next Step |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Piece Activity** | 113 | 96 | 47 | 53.32% | `[19, 19, 44, 18, 28]` | **+23.11** | `[47.58%, 59.06%]` / `[-16.80, +63.63]` | 162.4 ns | High | $+0.142$ | **Highest positive point estimate**. Major feature prior candidate for S10 NNUE. |
| **Pawn Structure** | 115 | 103 | 38 | 52.34% | `[16, 18, 57, 12, 25]` | **+16.30** | `[47.01%, 57.68%]` / `[-20.83, +53.80]` | 78.9 ns | Medium | **$+0.207$** | **Strong positive point estimate at moderate cost**. Core baseline knowledge. |
| **Mobility** | 108 | 102 | 46 | 51.17% | `[19, 22, 45, 18, 24]` | **+8.14** | `[45.58%, 56.76%]` / `[-30.79, +47.29]` | 109.8 ns | Med-High | $+0.074$ | Modest positive point estimate; candidate for weight retention. |
| **Rook Activity** | 107 | 102 | 47 | 50.98% | `[18, 15, 57, 20, 18]` | **+6.79** | `[45.84%, 56.12%]` / `[-28.99, +42.71]` | 78.0 ns | Medium | $+0.087$ | Modest positive point estimate; low removable overhead. |
| **King Safety** | 103 | 104 | 49 | 49.80% | `[19, 23, 45, 22, 19]` | **-1.36** | `[44.41%, 55.20%]` / `[-39.00, +36.25]` | 168.9 ns | High | $-0.008$ | **Expensive, unresolved marginal value at this sample size**. Motivates a throughput-optimization hypothesis (+3.6% observed removal NPS). |
| **Development / Space** | 95 | 106 | 55 | 47.85% | `[23, 19, 50, 18, 18]` | **-14.94** | `[42.42%, 53.29%]` / `[-53.11, +22.88]` | 44.4 ns | Low | **$-0.336$** | **Strongest negative candidate in screening**. Full−NoX point estimate is $-14.94$ Elo; requires dedicated SPRT confirmation (S9-C1). |

---

## 2. Key Screening Findings & Hypotheses

1. **Strongest Negative Candidate (`Development / Space`, Full−NoX Point Estimate $-14.94\text{ Elo}$)**:
   * `CurrentFinalNoDevelopmentSpace` scored **106 wins to 95 wins** (55 draws) against `CurrentFinal` in this common-opening sample ($47.85\%$ score for Full).
   * Micro-benchmarking shows a 44.4 ns removable evaluator cost, though end-to-end tournament NPS did not show a net speedup in this run ($-1.84\%$).
   * **Actionable Hypothesis**: The combined development/space heuristic may introduce static biases that interact poorly with search pruning. This motivates a dedicated confirmatory SPRT (S9-C1) and sub-component decomposition (`Development` vs `Space` in S9-C2).

2. **Core Positive Knowledge Priors (`Piece Activity` & `Pawn Structure`)**:
   * `Piece Activity` (+23.11 Elo point estimate) and `Pawn Structure` (+16.30 Elo point estimate) represent the primary positive drivers of evaluation quality in the current HCE configuration.
   * `Pawn Structure` exhibits the highest compute-to-Elo efficiency proxy ($+0.207\text{ Elo/ns}$).

3. **King Safety Throughput / Value Hypothesis**:
   * King Safety carries the largest single-family compute footprint (**168.9 ns** across 12 fixtures), while its point estimate in this 128-pair sample is neutral (**-1.36 Elo**).
   * Removing King Safety showed a +3.61% increase in search NPS. This motivates testing whether King Safety can be simplified or replaced without strength degradation, making it a natural target for S10 NNUE representation.

---

## 3. Provenance & Artifact Integrity

* **Source Tournaments**:
  * `pawn-structure`: `19e75dd8-1379-4bb6-bd00-621a9b48d18b`
  * `mobility`: `931b4f90-1dce-4f9d-a865-9a891b9bc12e`
  * `piece-activity`: `952b2a48-5af9-435e-a190-893442d202f1`
  * `rook-activity`: `d9fdc528-35b7-4f10-b9bb-1c2b182ca104`
  * `development-space`: `7a7e4948-ce43-40af-8eb7-d2cd076da33f`
  * `king-safety`: `1b560e21-c77f-412b-b398-a88918936312`
* **Raw Artifacts**: Full `summary.json`, `combined.pgn`, and `artifact-manifest.json` archived in respective family subdirectories under `results/artifacts/20260825-96d1a69-linux-x86_64/s9-b2-loo-attribution/`.
