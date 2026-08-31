# S10-E1 Closeout — Nested 1M Dataset Construction

**Status: CLOSED / PASS — dataset `s10-eval-v2-1m01` frozen, E2 labeling unblocked**

```
dataset_id:     s10-eval-v2-1m01
records_total:  1,000,000
dataset_sha256: 4eda8fe1d241d418071888dec661b00fa8bd6000c07abce2421cec668ec53de0
splits:         train 800,000 / validation 100,000 / holdout 100,000
phase buckets:  high 250,000 / mid 450,000 / low 200,000 / zero 100,000
verify_dataset: VERIFY_PASS (--allow-unlabeled; labels are E2)
rebuild:        second independent build bit-identical (all 4 shards +
                dataset SHA equal)
nested:         all 300,000 frozen 300k records present, position_id and
                split unchanged (0 missing, 0 split changes)
```

## Pipeline

### Pool reconstruction (parallel + resumable)

* `e1_pool_extract.py`: per-source parallel candidate extraction with an
  immutable per-source cache (`data/s10/e1-pool-cache/<id>.jsonl` +
  manifest binding source_id, PGN SHA, sampling contract, candidate
  SHA). 24 B1 sources -> 1,879,773 candidates.
* `e1_pool_merge.py`: global merge + GLOBAL position_id dedup (19,707
  duplicates removed -> 1,860,066 unique), then the 12-cell capacity
  matrix. First profile found the exact zero-phase shortfall:
  train 53,457 / validation 6,849 / holdout 6,655 (matches the old B1
  capacity report).

### v5 source (zero-phase expansion)

* The only zero-phase lever is LONG games; the July 2026 local archive
  (27.1 GB, official lichess SHA `68738b1c…` verified on disk) was used
  — no network download.
* `e1_fast_select_lib.py`: header-gated fast selector. Line-oriented
  streaming state machine; header-only gates (Result/Elo>=1800 both/no
  BOT/TC base>=180s + Site-hash gate) run before any move-tree parse;
  a raw token ply-counter gates long games; full python-chess parse
  only for accepted games. **Equivalence vs the original
  lichess_select.py loop: 300/300 accepted games identical (key,
  fingerprint, plies) at 64x speedup.**
* Two bugs found during equivalence bring-up are pinned by the run:
  a header regex missing PGN bracket literals (caught by 0 accepted in
  1M games) and an initial buffer-rescan design that was 2x SLOWER than
  the original (rewritten as the line state machine).
* `e1_extract_v5.py`: v5 = 350,000 games, ALL >= 100 plies
  (long_fraction 1.0), accept_byte 0x1F, seed 20260830, every B1
  source game excluded by fingerprint (8,178 excluded candidates).
  A first attempt with accept 0x05 yielded only 58k long games
  (~14.7k zero positions, insufficient); the accept byte is the
  sampling knob and was widened to 0x1F with target 350k. Final v5
  pool: 2,796,000 candidates, PGN 1.36 GB.

### Capacity (after v5)

All 12 cells OK; zero cells: train 90,936 / validation 11,574 /
holdout 11,172 (targets 80k/10k/10k).

### Nested build (`e1_build_1m.py`)

* Frozen 300k preserved verbatim (old shard lines copied byte-for-byte;
  verified 0 missing / 0 split changes).
* Expansion 700k, seed 2026083002, deterministic top-K by ply priority:
  * high/mid/low cells drawn ONLY from the old B1 pool (v5 excluded —
    source composition for these cells is unchanged);
  * zero cells: old pool first, v5 only for the shortfall (train
    53,457 / validation 6,849 / holdout 6,655 — exactly the deficits).
* `verify_dataset.py`: records, uniqueness, eligibility re-validation
  (1,000,000/1,000,000 board.is_valid PASS), cross-split position AND
  game isolation, phase buckets vs manifest — all PASS.
* Second independent build: 4 shards bit-identical, dataset SHA equal.

## Artifacts

```
tools/s10/e1_pool_extract.py       per-source parallel extractor + cache
tools/s10/e1_pool_merge.py         global dedup + 12-cell capacity
tools/s10/e1_fast_select_lib.py    header-gated fast selector lib
tools/s10/e1_selector_equivalence.py  300/300 equivalence harness
tools/s10/e1_extract_v5.py         v5 production extraction
tools/s10/e1_build_1m.py           nested 1M builder
data/s10/e1-pool-cache/            immutable per-source caches (25)
data/s6/sources/lichess-standard-rated-v5/   v5 source (350k long games)
data/s10/s10-eval-v2-1m01/         the frozen 1M dataset
results/s10/s10-e1-pool-capacity-v2.json
```
