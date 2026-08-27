# S10-B1 Source Capacity Audit Report

## 1. Audit Objective
Verify whether existing local source collections (`data/s6/sources/*`) can construct the full $300,000$ position multi-source dataset (`s10-eval-v1-300k01`) under frozen S6.0 / S10-B rules:
- $300,000$ exact records (240k train / 30k val / 30k holdout)
- Phase quotas: 75k high / 135k mid / 60k low / 30k zero
- Sampling: Deterministic hash top-$k$, min ply 12, max ply 160, **max 8 positions per game**
- Diversity: $\ge 2$ source families, largest family share $\le 70\%$

---

## 2. Source Capacity Audit Results

| Source Collection | Source Family | Total Games | Max Sampled Positions (@ 8/game) |
| :--- | :--- | :--- | :--- |
| `data/s6/sources/*.pgn` | Arena | 1,572 | 12,576 |
| `data/s6/sources/lichess-standard-rated-v1` | Lichess | 2,000 | 16,000 |
| `data/s6/sources/lichess-standard-rated-confirm-v1-g1400` | Lichess | 1,400 | 11,200 |
| **Total Available Local Sources** | — | **4,972 games** | **39,776 positions** |

---

## 3. Bottleneck Analysis & Fail-Closed Statement

- **Required Total Positions**: $300,000$
- **Maximum Feasible Yield**: $39,776$ positions
- **Deficit**: $260,224$ positions ($32,528$ additional games needed)
- **Status**: **`FAIL_CLOSED_REPORTED`**. Per frozen instructions, we do not duplicate samples, relax sampling parameters, or downscale the dataset target without authorization. Generating the full $300\text{k}$ dataset requires extracting $\ge 35,000$ games from the source archives.
