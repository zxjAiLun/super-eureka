# S10-B1B Implementation Plan: Production Source Expansion & Exact 300k Dataset Gate (Repair 1)

## 1. Goal & Architectural Boundaries
- Overcome the source capacity and family composition deficit discovered in S10-B1A without modifying sampling contracts or downscaling quotas.
- Expand available raw games from 4,972 to $\approx 81,572$ games ($\approx 652,000$ potential sample slots, a $\approx 2.17\times$ safety buffer) across **3 canonical source families**:
  1. `lichess-standard-rated-v1` (Canonical standard rated family): Expand to $\approx 50,000$ total games by adding `lichess-standard-rated-v2` ($46,600$ new games extracted from 2026-07 archive via `tools/s6/lichess_select.py`).
  2. `lichess-broadcast` (Canonical broadcast family): Extract $30,000$ games ($10,000$/month from 2026-05, 2026-06, 2026-07 archives via `tools/s10/extract_broadcast.py` under CC BY-SA 4.0).
  3. `arena` (Historical engine tournaments): Retain $1,572$ games across classical and smoke test suites.
- Strict family continuity: New standard-rated extractions will explicitly carry `source_family: "lichess-standard-rated-v1"` to preserve scientific identity and strictly obey the $\le 70\%$ family ceiling without artificial family splitting.
- Produce the exact $300,000$ record dataset `s10-eval-v1-300k01` using `build_dataset.py` with 2-run independent rebuild SHA verification before any teacher labeling.

---

## 2. Canonical Source Pool Composition Matrix
| Source Key | Source ID | Source Family | Target Games | Theoretical Slots (@ 8/game) | Expected Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `lichess-standard-rated-v1` | `lichess-standard-rated-v1` | `lichess-standard-rated-v1` | $2,000$ | $16,000$ | Shared Family Pool |
| `lichess-standard-rated-confirm-v1-g1400` | `lichess-standard-rated-confirm-v1-g1400` | `lichess-standard-rated-v1` | $1,400$ | $11,200$ | Shared Family Pool |
| `lichess-standard-rated-v2` | `lichess-standard-rated-v2` | `lichess-standard-rated-v1` | $46,600$ | $372,800$ | Shared Family Pool ($\le 70\%$ total) |
| `lichess-broadcast-v1` | `lichess-broadcast-v1` | `lichess-broadcast` | $30,000$ | $240,000$ | $\ge 25\%$ |
| `arena-*` (aggregate catalog) | Various (`sf-smoke*`, `s43*`, etc.) | `arena` | $1,572$ | $12,576$ | $\approx 2\text{--}4\%$ |
| **Total Expanded Pool** | **5 Catalogs** | **3 Canonical Families** | **$81,572$ games** | **$652,576$ slots** | **$2.17\times$ Capacity Buffer** |

---

## 3. Repair 1 Hardened Technical Contracts

### A. Broadcast Extractor Contract (`tools/s10/extract_broadcast.py`)
1. **Streaming SHA Lifecycle**: Verified upstream SHA-256 in flight while decompression stream is active and before closing resources.
2. **Immutable Publication**: Existing destination directory causes immediate FAIL CLOSED (never overwrites published sources).
3. **Whole-Month Deterministic Top-K**: All candidates in an archive are evaluated and ranked via `sha256(seed || month || canonical_fingerprint)`. Top-K smallest ranks are retained, ensuring exact order-independence.
4. **Stratum Balancing**: Explicitly partitions and selects $\approx 33\%$ long games ($\ge 80$ plies) and $\approx 67\%$ short games ($\ge 40$ plies).
5. **Unified Fingerprint Serialization**: Reuses `tools/s10/source_identity.py` matching the S6 canonical JSON contract.
6. **Strict Exclusion Fail-Closed**: Missing `--exclude-pgn` path triggers FAIL CLOSED; manifest enforces `fingerprint_intersection_count == 0`.

### B. Profiler Contract (`tools/s10/analyze_source_pool.py`)
- Simulates exact builder pipeline (top-8 per game, global FEN4 `position_id` deduplication, deterministic game splitting).
- Profiles and audits the **12 exact target cells**:
  - `train`: `high` ($60\text{k}$), `mid` ($108\text{k}$), `low` ($48\text{k}$), `zero` ($24\text{k}$)
  - `validation`: `high` ($7.5\text{k}$), `mid` ($13.5\text{k}$), `low` ($6\text{k}$), `zero` ($3\text{k}$)
  - `holdout`: `high` ($7.5\text{k}$), `mid` ($13.5\text{k}$), `low` ($6\text{k}$), `zero` ($3\text{k}$)
- Reports post-deduplication `family x split x phase` candidate matrix with exact shortfall margins.

---

## 4. Execution Roadmap
1. **Unit Test Gate**: Run full synthetic test suite (`tools/s10/test_extract_broadcast.py`) to verify all 8 Repair 1 criteria.
2. **Broadcast Extraction (`lichess-broadcast-v1`)**: Extract 30,000 games across 2026-05, 2026-06, 2026-07.
3. **Standard-Rated Expansion (`lichess-standard-rated-v2`)**: Extract 46,600 games from 2026-07 standard rated archive using `tools/s6/lichess_select.py`.
4. **Feasibility Profiling**: Run `tools/s10/analyze_source_pool.py` across all 5 source directories.
5. **300k Dataset Rebuild & Verification**: Run `build_dataset.py` in FINAL mode with 2-run independent bit-identical SHA verification (`s10-eval-v1-300k01`).
