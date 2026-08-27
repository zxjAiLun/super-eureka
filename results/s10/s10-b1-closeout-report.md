# S10-B1 Dataset Construction Closeout Report

## Summary & Objectives
- **Milestone**: S10-B1 Production Dataset Construction (`s10-eval-v1-300k01`).
- **Feature Set**: Frozen `NnueFeatureSetV2` (22,528 sparse inputs, 32 king buckets $\times$ 11 piece channels $\times$ 64 squares with horizontal mirroring, 31 active features on startpos).
- **Target Position Quota**: Exactly 300,000 canonical chess positions.
- **Split Distribution**:
  - `train`: 240,000 positions (80%)
  - `validation`: 30,000 positions (10%)
  - `holdout`: 30,000 positions (10%)
- **Phase Stratification**:
  - `high` [18, 24]: 75,000 positions (25%)
  - `mid` [8, 17]: 135,000 positions (45%)
  - `low` [1, 7]: 60,000 positions (20%)
  - `zero` [0, 0]: 30,000 positions (10%)
- **Deduplication & Integrity**: Canonical FEN4 deduplication (`<piece_placement> <active_color> <castling_rights> <en_passant>`), zero cross-split position leakage, zero cross-split game leakage.

---

## Source Corpus
The source pool consists of 234,972 parsed games across 24 source inputs:
1. `lichess-standard-rated-v1` (2,000 games)
2. `lichess-standard-rated-confirm-v1-g1400` (1,400 games)
3. `lichess-standard-rated-v2` (70,000 standard rated games from 2026-07)
4. `lichess-standard-rated-v3` (100,000 long games $\ge 80$ plies from 2026-07)
5. `lichess-standard-rated-v4` (60,000 long games $\ge 80$ plies from 2026-06)
6. Arena selfplay & tournament suites (1,572 games across 19 sub-sources)

### Extraction & Integrity Verification
- Upstream standard-rated archives streamed directly from `https://database.lichess.org/standard/`.
- Official upstream SHA-256 verified against published checksums.
- Deduplication against prior batches enforced via deterministic game fingerprints.

---

## 3-Tier Capacity Profiling
Execution of `tools/s10/analyze_source_pool.py` verified the entire pool across all 12 core split $\times$ phase cells:

| Split | Phase | Target Quota | Available in Pool | Margin | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **train** | high | 60,000 | 372,405 | +312,405 | **PASS** |
| **train** | mid | 108,000 | 429,819 | +321,819 | **PASS** |
| **train** | low | 48,000 | 298,184 | +250,184 | **PASS** |
| **train** | zero | 24,000 | 26,543 | +2,543 | **PASS** |
| **validation** | high | 7,500 | 46,140 | +38,640 | **PASS** |
| **validation** | mid | 13,500 | 54,064 | +40,564 | **PASS** |
| **validation** | low | 6,000 | 36,875 | +30,875 | **PASS** |
| **validation** | zero | 3,000 | 3,151 | +151 | **PASS** |
| **holdout** | high | 7,500 | 46,471 | +38,971 | **PASS** |
| **holdout** | mid | 13,500 | 52,926 | +39,426 | **PASS** |
| **holdout** | low | 6,000 | 36,954 | +30,954 | **PASS** |
| **holdout** | zero | 3,000 | 3,345 | +345 | **PASS** |

- **Tier 1 (Raw Post-Dedup)**: 1,860,066 positions
- **Tier 2 (Global Stratified)**: 1,406,877 positions
- **Tier 3 (Exact FINAL Selected)**: 300,000 positions (100.00% target match)
- **Feasibility Verdict**: **PASS** (Zero cell deficits across all 12 cells).

---

## Bit-Identical Rebuild Verification (2-Run Determinism)
Two independent build runs were executed using `tools/s6/build_dataset.py`:
- **Build A Output SHA-256**: `503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3`
- **Build B Output SHA-256**: `503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3`
- **Verification**: `Build A SHA-256 == Build B SHA-256` (**BIT-IDENTICAL MATCH**).

Independent dataset audit via `tools/s6/verify_dataset.py --allow-unlabeled`:
- `board.is_valid` audit: 300,000 / 300,000 **PASS**
- Split isolation (position & game): **PASS**
- Phase buckets & quotas: **PASS**
- Overall Audit Result: **VERIFY_PASS**

Canonical dataset artifact installed at: `data/s10/s10-eval-v1-300k01/`

---

## Production Invariants & Code Base Health
- `cargo test --lib`: 372 / 372 tests passing (0 failures).
- Python test suites (`test_extract_broadcast.py`, `test_analyze_source_pool.py`): 10 / 10 passing (0 failures).
- Search and evaluation engines (`src/engine/eval.rs`, `src/engine/search.rs`): 0 diff against frozen baseline `3dae2fa`.
- Ready for S10-B2 Stockfish-18 teacher labeling.

---

## Wall-Clock Telemetry
- **S10-B1 total wall-clock time**: 370 minutes 38 seconds (6 hours 10 minutes 38 seconds).

---

## Post-Approval Provenance Repair (P1)
A post-hoc provenance audit found that the published `source-manifest.json` files for `lichess-standard-rated-v2/v3/v4` recorded a hardcoded filters block (`mainline_plies_min=40`, `long_stratum_plies_min=80`, `long_fraction=1/3`) and a hardcoded selection threshold text (`< 0x05`) instead of the actual effective extraction arguments. For v3/v4 this is provably inconsistent with the outputs (both are 100% long games, impossible under the recorded defaults).

- **Impact**: Documentation/provenance only. The PGNs, their SHA-256 identities, and the frozen 300k dataset (`503b47b6...`) are unaffected.
- **Repair**: `tools/s6/lichess_select.py` now serializes `args.min_plies`, `args.long_min_plies`, `args.long_fraction`, and `args.accept_byte` into the manifest filters block and formats the actual accept-byte threshold into the selection text; targeted tests (`ManifestEffectiveArgumentsTests`) assert non-default arguments round-trip exactly.
- **Historical parameters**: The actual v2/v3/v4 command lines were not recoverable from any execution log. Per the no-guessing rule, partial provenance is recorded in `results/s10/s10-b1-source-extraction-provenance.json` with the immutable PGN SHA-256s as the authoritative source identities.
- **Non-blocking P2 fixes**: stale family-gate wording updated in `tools/s6/build_dataset.py --help` (family mix is opt-in via `--enforce-family-mix`) and the `tools/s10/analyze_source_pool.py` docstring (telemetry-only unless `--enforce-family-mix`).
- **Explicitly NOT done**: no re-extraction of v2/v3/v4, no rebuild of `s10-eval-v1-300k01`; dataset SHA `503b47b6...` remains frozen.
