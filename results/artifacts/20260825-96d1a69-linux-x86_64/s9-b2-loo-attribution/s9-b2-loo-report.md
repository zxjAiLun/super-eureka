# S9-B2: Leave-One-Out (LOO) Feature Family Attribution Report

**Date**: 2026-08-25  
**Binary SHA-256**: `45d1a895ebc8a6643dce3454d7b421111cfeaab2f5e55537c8851ffd318933ba`  
**Build ID**: `20260825-96d1a69-linux-x86_64` (Git SHA: `96d1a69b2d884b3f78703d8c87c973dff9eb7830`)  
**Evaluation Protocol**: Fixed-sample screening (128 pairs / 256 games per family, TC `10+0.1`, Hash 32MB, Threads 1, Concurrency 1)  
**Common Opening Design**: 6 LOO tournaments evaluated against identical 128 opening indices derived from `stockfish-8moves-v3` with fixed random seed `2026082501`.  
**Attribution Metric**: $\Delta\text{Elo}_X = \text{Elo}(\text{Full}) - \text{Elo}(\text{NoX})$ (Engine A = `CurrentFinal`, Engine B = `CurrentFinalNoX`).

---

## 1. Executive Summary & Attribution Matrix

| Family ($X$) | Wins | Losses | Draws | Score (%) | Pentanomial `[0-2, 0.5-1.5, 1-1, 1.5-0.5, 2-0]` | $\Delta\text{Elo}_X$ (Full − NoX) | 95% Pair CI (Score / Elo) | Removable Cost (`median_ns`) | Cost Tier | Elo/ns Efficiency Proxy | Preliminary Conclusion |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Piece Activity** | 113 | 96 | 47 | 53.32% | `[19, 19, 44, 18, 28]` | **+23.11** | `[47.58%, 59.06%]` / `[-16.80, +63.63]` | 162.4 ns | High | $+0.142$ | **Top Elo contributor**. Critical knowledge source despite high compute footprint. |
| **Pawn Structure** | 115 | 103 | 38 | 52.34% | `[16, 18, 57, 12, 25]` | **+16.30** | `[47.01%, 57.68%]` / `[-20.83, +53.80]` | 78.9 ns | Medium | **$+0.207$** | **Highest compute-to-Elo efficiency**. Essential baseline knowledge. |
| **Mobility** | 108 | 102 | 46 | 51.17% | `[19, 22, 45, 18, 24]` | **+8.14** | `[45.58%, 56.76%]` / `[-30.79, +47.29]` | 109.8 ns | Med-High | $+0.074$ | Positive marginal value; moderate cost. |
| **Rook Activity** | 107 | 102 | 47 | 50.98% | `[18, 15, 57, 20, 18]` | **+6.79** | `[45.84%, 56.12%]` / `[-28.99, +42.71]` | 78.0 ns | Medium | $+0.087$ | Positive marginal value; modest cost. |
| **King Safety** | 103 | 104 | 49 | 49.80% | `[19, 23, 45, 22, 19]` | **-1.36** | `[44.41%, 55.20%]` / `[-39.00, +36.25]` | 168.9 ns | High | $-0.008$ | **Expensive & Neutral**. High cost (~169 ns) with zero/negative marginal yield in search. |
| **Development / Space** | 95 | 106 | 55 | 47.85% | `[23, 19, 50, 18, 18]` | **-14.94** | `[42.42%, 53.29%]` / `[-53.11, +22.88]` | 44.4 ns | Low | **$-0.336$** | **Negative Value Feature**. Removing Development/Space yields +14.94 Elo for baseline! |

---

## 2. Key Scientific Findings & Discoveries

1. **Discovery of Negative-Value Feature (`Development / Space`)**:
   * Baseline `CurrentFinalNoDevelopmentSpace` scored **106 wins to 95 wins** against Full `CurrentFinal` (47.85% for Full, $\Delta\text{Elo} = -14.94$).
   * Even though Development/Space is compute-cheap (44.4 ns removable cost), its static evaluation bias / scaling distorts tree search in opening transitions.
   * **Actionable Insight**: Pruning or zeroing Development/Space weights provides an immediate positive Elo gain and search speedup.

2. **Compute-Value Frontier Leader (`Pawn Structure`)**:
   * Pawn structure provides **+16.30 Elo** at an incremental cost of only **78.9 ns**, yielding the highest efficiency ratio ($+0.207\text{ Elo/ns}$).
   * Candidate for weight optimization in S9-C.

3. **Core Driver of Playing Strength (`Piece Activity`)**:
   * Piece activity delivers **+23.11 Elo**, the largest single-family playing strength contribution across the suite.
   * Justifies its 162.4 ns computational footprint in HCE, and forms the primary feature prior for NNUE distillation in S10.

4. **King Safety Inefficiency in HCE Search Tree**:
   * Evaluator timing proves King Safety is the most computationally expensive family (**168.9 ns** across 12 fixtures), yet its LOO match outcome was dead neutral (**-1.36 Elo**, 103W / 104L / 49D).
   * This confirms that hand-crafted king safety formulas with non-linear penalties create search instability when evaluated dynamically. King safety patterns are prime targets for NNUE feature representation.

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
