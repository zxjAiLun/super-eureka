# S10-A Deterministic Training Replay & Reproducibility Audit

## 1. Executive Summary
- **Audit Outcome**: **`PASS_CONFIRMED`**.
- **Determinism Contract Enforcement**:
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` set prior to CUDA initialization.
  - `torch.use_deterministic_algorithms(True, warn_only=False)` active (fails closed on non-deterministic operations).
  - `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` enforced.
- **Primary Non-Inferiority Gate**: **PASSED**.
  - Deterministic 3-seed median validation MAE for V2 is **238.85 cp** vs **236.23 cp** for V1 ($+1.113\% \le +2.00\%$ threshold).
- **Secondary Engineering Evidence**: Reconfirmed. 45.0% Feature Transformer footprint reduction, 49.5% reduction in unseen feature activation rate, and 46.5% reduction in positions containing unseen features.
- **Holdout Integrity**: No holdout split was accessed or evaluated during this replay (`allow_holdout=False`).

---

## 2. Replay Validation MAE Matrix

All 6 training runs executed under strict deterministic mode on NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 13.0, cuDNN 9.2, PyTorch 2.13.0+cu130):

| Seed | V1 Val Loss | V1 Val MAE (cp) | Best Epoch | V2 Val Loss | V2 Val MAE (cp) | Best Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `20260818` | 0.053229 | 236.2257 | 44 | 0.055107 | 239.8546 | 53 |
| `20260819` | 0.054366 | 237.9063 | 21 | 0.054944 | 238.8546 | 11 |
| `20260820` | 0.053427 | 235.7326 | 32 | 0.054170 | 237.5076 | 28 |
| **Median** | **0.053427** | **236.2257** | — | **0.054944** | **238.8546** | — |
| **Mean** | **0.053674** | **236.6215** | — | **0.054740** | **238.7389** | — |

$$\text{Delta} = \frac{238.8546 - 236.2257}{236.2257} = +1.113\% \le +2.00\% \implies \mathbf{PASS}$$

---

## 3. Closeout Errata & Scientific Calibration

The historical artifacts (`results/s10/s10-a-representation-selection.json` and `results/s10/s10-a-holdout.json`) are preserved immutably. The following calibrations apply to the closeout interpretation:

1. **Feature Extraction Cost**: V2 trades an estimated $+11.7\%$ full-refresh sparse extraction cost ($537.30\text{ ns/pos}$ vs $480.99\text{ ns/pos}$) for a $-45.0\%$ Feature Transformer parameter reduction ($5.24\text{M} \rightarrow 2.88\text{M}$) and cut-in-half validation unseen feature activation rate. This cost will be governed by incremental accumulator updates in S10-C.
2. **Holdout Evaluation**: The 3-seed median holdout MAE of $230.68\text{ cp}$ confirms that the selected V2 representation exhibited no post-selection generalization collapse on the untouched holdout split.
