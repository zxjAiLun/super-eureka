# S10-B1B Implementation Plan: Production Source Expansion & Exact 300k Dataset Gate

## 1. Goal & Architectural Boundaries
- Overcome the source capacity and family composition bottleneck discovered in S10-B1A without modifying sampling contracts or downscaling quotas.
- Expand available raw games from 4,972 to $\approx 81,500$ games ($\approx 652,000$ potential sample slots, a $\approx 2.17\times$ safety buffer) across **3 distinct source families**:
  1. `lichess-standard-rated`: Expand to $\approx 50,000$ high-quality games from verified monthly archives (`lichess-standard-rated-v1`, `lichess-standard-rated-v2`, `lichess-standard-rated-v3`).
  2. `lichess-broadcast`: Extract $\approx 30,000$ offline/official tournament broadcast games (CC BY-SA 4.0) from verified broadcast monthly archives (`lichess-broadcast-v1`).
  3. `arena`: Retain 1,572 historical engine tournament games.
- Note: Multiple extractions from standard rated games will be assigned to the single `lichess-standard-rated` family (no artificial family splitting to bypass the 70% share cap).
- Produce the exact $300,000$ record dataset `s10-eval-v1-300k01` using `build_dataset.py` with 2-run independent rebuild SHA verification before any teacher labeling.

---

## 2. Source Expansion Architecture & Quota Matrix

### A. Raw Source Pool Composition
| Source Identity | Family Name | Target Games | Theoretical Sample Capacity (@ 8/game) | Expected Share |
| :--- | :--- | :--- | :--- | :--- |
| `lichess-standard-rated-*` | `lichess-standard-rated` | $50,000$ | $400,000$ slots | $\le 70\%$ cap |
| `lichess-broadcast-v1` | `lichess-broadcast` | $30,000$ | $240,000$ slots | $\ge 25\%$ |
| `arena-*` | `arena` | $1,572$ | $12,576$ slots | $\approx 2\text{--}4\%$ |
| **Total Expanded Pool** | **3 Families** | **$81,572$ games** | **$652,576$ slots** | **$2.17\times$ Buffer** |

### B. Binding Constraint Protection & Diagnostic Tooling
- Build a dedicated diagnostic tool `tools/s10/analyze_source_pool.py` that parses the candidate sources and profiles:
  1. Total legal, non-check, non-terminal position counts
  2. True unique FEN yields
  3. Phase distribution ($zero \in [0, 0]$, $low \in [1, 7]$, $mid \in [8, 17]$, $high \in [18, 24]$)
  4. Cross-split game partition feasibility and family share balance
- If any phase quota (e.g. $zero=30,000$) or family ceiling ($70\%$) cannot be met, the tool will explicitly report the exact binding constraint.

---

## 3. Execution Steps

1. **Broadcast Family Extraction (`lichess-broadcast-v1`)**:
   - Extract 30,000 games from 2026-05, 2026-06, 2026-07 broadcast archives (10k games/month) with in-flight SHA-256 validation.
   - Publish atomically to `data/s6/sources/lichess-broadcast-v1`.
2. **Standard-Rated Family Expansion**:
   - Extract additional rated games using verified chunked streaming from official Lichess standard rated archives, saving to `data/s6/sources/lichess-standard-rated-v2`.
3. **Pool Feasibility & Binding Constraint Audit**:
   - Run `tools/s10/analyze_source_pool.py` to confirm exact phase and family feasibility.
4. **300k Dataset Construction (`s10-eval-v1-300k01`)**:
   - Run `tools/s6/build_dataset.py` with `--sampling-version 2 --final-mode --enforce-family-mix`.
   - Perform independent Build A and Build B to verify identical dataset SHA-256.
   - Run `tools/s6/verify_dataset.py --allow-unlabeled`.
5. **Closeout & Commit**:
   - Commit dataset manifests, source manifests, audit reports, and tooling to `s10/nnue-production-foundation`.
