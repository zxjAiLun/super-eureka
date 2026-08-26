# S10-A: NNUE Representation Gate & Foundation Report

## 1. Executive Summary
- **Primary Decision**: **`SELECT_V2`** (HalfKAv2_hm-inspired horizontal king-mirroring representation, 22,528 sparse inputs, 31 active features on start position).
- **Primary Non-Inferiority Gate**: Passed. 3-seed median validation MAE for V2 is **238.85 cp** vs **236.23 cp** for V1 ($+1.11\% \le +2.00\%$ threshold).
- **Secondary Engineering Evidence**:
  - **Parameter Footprint**: 45.0% reduction in Feature Transformer weights ($5,243,008 \rightarrow 2,883,712$ parameters, saving $9.00\text{ MiB}$ FP32 memory).
  - **Feature Activation Generalization**: Validation unseen feature activation rate cut in half ($1.01\% \rightarrow 0.51\%$).
  - **Positions with Unseen Features**: Reduced by 46.5% ($18.93\% \rightarrow 10.13\%$).
  - **Rust Feature Extraction Cost**: $537.30\text{ ns/pos}$ for V2 vs $480.99\text{ ns/pos}$ for V1 (pure sparse extraction without search overhead).
- **Blind-Holdout Integrity**:
  - Stage 1 selection artifact `s10-a-representation-selection.json` was committed with SHA-256 `474b82664d9aa3ba2aeb3252b5469b10ba73a9c32080dc054f4fc4f1bf77b437` *before* observing any holdout metrics.
  - Stage 2 confirmatory holdout evaluated only the 3 winning V2 seeds, achieving a median holdout MAE of **230.68 cp** (outperforming validation MAE).
- **Semantic Anti-Drift**: Production classical evaluator and search files (`src/engine/eval.rs`, `src/engine/search.rs`) maintain **0 diff**.

---

## 2. Representation Architecture Comparison

| Feature Metric | NnueFeatureSetV1 (S6 Legacy Control) | NnueFeatureSetV2 (S10 HalfKAv2_hm) | Relative Delta |
| :--- | :--- | :--- | :--- |
| **King Conditioning** | 64 oriented king squares | 32 horizontally mirrored king squares (files e–h) | $-50.0\%$ |
| **Piece Channels** | 10 (Own/Opp P, N, B, R, Q) | 11 (Own/Opp P, N, B, R, Q + Opp King) | $+1\text{ channel}$ |
| **Startpos Active Count** | 30 features / perspective | 31 features / perspective | $+1\text{ feature}$ |
| **Total Sparse Dimension** | $64 \times 10 \times 64 = 40,960$ | $32 \times 11 \times 64 = 22,528$ | **$-45.0\%$** |
| **FT Parameter Count** | $5,243,008$ | $2,883,712$ | **$-45.0\%$** |
| **FT FP32 Memory** | $20.00\text{ MiB}$ | $11.00\text{ MiB}$ | **$-9.00\text{ MiB}$** |
| **Total Model Parameters** | $5,252,513$ | $2,893,217$ | **$-44.9\%$** |
| **Extraction Cost (Rust)** | $480.99\text{ ns/pos}$ (MAD $4.30$) | $537.30\text{ ns/pos}$ (MAD $7.71$) | $+56.31\text{ ns}$ ($+11.7\%$) |

---

## 3. Stage 1 Bake-Off Results (S6-N3B Pilot Dataset)

### Validation MAE (Centipawns)
All models trained with identical hyperparameters: PyTorch 2.13, CUDA 13.0, RTX 4060 Laptop GPU, AdamW (`lr=1e-3`, `wd=1e-5`), SmoothL1 (`beta=0.1`), batch size 256, max epochs 100, patience 15. Best checkpoints selected strictly by validation SmoothL1 loss.

| Random Seed | V1 Val Loss | V1 Val MAE (cp) | Best Epoch | V2 Val Loss | V2 Val MAE (cp) | Best Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20260818` | 0.053229 | 236.23 | 44 | 0.055107 | 239.85 | 53 |
| `20260819` | 0.054366 | 237.91 | 21 | 0.054944 | 238.85 | 11 |
| `20260820` | 0.053427 | 235.73 | 32 | 0.054170 | 237.51 | 28 |
| **Median** | **0.053427** | **236.23** | — | **0.054944** | **238.85** | — |
| **Mean** | **0.053674** | **236.62** | — | **0.054740** | **238.74** | — |

**Primary Gate Verification**:
$$\frac{\text{median\_val\_MAE(V2)}}{\text{median\_val\_MAE(V1)}} = \frac{238.8546}{236.2257} = 1.01113 \le 1.0200 \implies \mathbf{PASS}$$

### Coverage & Generalization Diagnostics
- **Training Unique Features**: V1 saw 16,306 / 40,960 features ($39.81\%$). V2 saw 11,515 / 22,528 features ($51.11\%$).
- **Validation Unseen Activation Rate**:
  - V1: $1.012\%$ of active feature lookups had never been seen during training.
  - V2: $0.511\%$ of active feature lookups had never been seen during training (**$49.5\%$ reduction**).
- **Validation Positions with $\ge 1$ Unseen Feature**:
  - V1: $18.93\%$
  - V2: $10.13\%$ (**$46.5\%$ reduction**).

---

## 4. Stage 2 Confirmatory Holdout Evaluation (V2)

Holdout split consists of 2,154 positions with non-null `teacher_cp_stm`. Evaluated strictly once across all 3 V2 seeds.

| Winning Model | Validation MAE (cp) | Confirmatory Holdout MAE (cp) | Holdout Smooth L1 Loss |
| :--- | :--- | :--- | :--- |
| `checkpoint_v2_s20260818.pt` | 239.85 | 230.68 | 0.186516 |
| `checkpoint_v2_s20260819.pt` | 238.85 | 233.11 | 0.188604 |
| `checkpoint_v2_s20260820.pt` | 237.51 | 230.42 | 0.186084 |
| **Median** | **238.85** | **230.68** | **0.186516** |
| **Mean** | **238.74** | **231.40** | **0.187068** |

---

## 5. Artifact & Provenance Manifest

- **Git Branch**: `s10/nnue-production-foundation`
- **Base Commit**: `3dae2fa`
- **Stage 1 Decision Artifact**: `results/s10/s10-a-representation-selection.json` (`SHA-256: 474b82664d9aa3ba2aeb3252b5469b10ba73a9c32080dc054f4fc4f1bf77b437`)
- **Stage 2 Holdout Artifact**: `results/s10/s10-a-holdout.json`
- **Dataset SHA-256**: `5c90b07481c70e7e1b5f10682ba7d4c82c3c907a974b71be0e657bfb7b9f34fb`
- **Labels SHA-256**: `e3057e93dcad1a704f0ddbf5f6d7285a8dfda1d2729a6e11894d03e5c9a0937c`
- **Engine Binary**: `./target/release/eureka`
- **Python Harness**: `tools/s10/train_nnue.py` (validated by `tools/s10/test_train_nnue.py`)
