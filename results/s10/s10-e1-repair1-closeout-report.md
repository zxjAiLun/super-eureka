# S10-E1 Repair 1 Closeout — Nested 1M Dataset (B1-rule exact reconstruction)

**Status: CLOSED / PASS — dataset `s10-eval-v2-1m01` (1504803a...) frozen;
E2 SF18 1M labeling unblocked. Supersedes the pre-repair `4eda8fe1...`,
which used the exploratory `SEED_1M / ply_priority` extension rule.**

```
dataset_id:     s10-eval-v2-1m01
records_total:  1,000,000
dataset_sha256: 1504803a179e7601149862aa47ba6d3ed77977d976397bc780ec92d0d8880560
splits:         train 800,000 / validation 100,000 / holdout 100,000
phase buckets:  high 250,000 / mid 450,000 / low 200,000 / zero 100,000
verify_dataset: VERIFY_PASS (1,000,000/1,000,000 is_valid; split isolation
                by position AND game; 12-cell matrix exact; shard SHAs
                exact — the new Repair-1 verifier checks)
rebuild:        second independent build bit-identical (4 shards +
                dataset SHA equal)
nested parent:  reconstructed under the FULL frozen B1 chain and matched
                300,000/300,000 (position_id, split, core fields), 0
                mismatches; the frozen 300k records are written VERBATIM
```

## Repair 1 root cause

The first repair attempt (pre-repair `4eda8fe1...`) reconstructed the
parent by sorting each split x phase cell of the raw deduped pool by
`position_id` — but the frozen B1 chain has FOUR steps, and the
reconstruction silently skipped step 3:

```
1. per game: (ply_priority(game_id, ply, DATASET_SEED), ply) top-8
2. global dedup by (position_id, source_game_id)
3. GLOBAL phase-quota stratification: per bucket target =
   round(n_pre * PHASE_QUOTAS) {high .25, mid .45, low .20, zero .10},
   sort by position_id, take the bucket quota
   (1,860,066 unique old-pool -> 1,406,877 stratified)
4. final split x phase downsample: sort by position_id, take the cell
   quota (= the frozen 300k, verified 300000/300000 exact)
```

Without step 3, each cell's position_id prefix matched only ~50% of the
frozen parent (180,174 mismatches) — the frozen 300k is a subset of the
STRATIFIED pool, not of the raw pool. Repair 1 reproduces the full
chain: the parent is reconstructed from the stratified pool, and the
+700k extension continues position_id order on the stratified pool's
remaining candidates (high/mid/low: old B1 sources only; zero: old pool
first, then v5 for exactly the shortfall).

```
extension:      633,039 from the stratified old pool + 66,961 v5
                (zero cells: old-pool remainder 25,897 total, then v5
                shortfall train 53,457 / validation 6,849 / holdout 6,655)
```

## Verifier hardening (Repair 1, fail-close additions)

`tools/s6/verify_dataset.py` now additionally enforces, when the
manifest carries the fields:

* `splits` — actual per-split record counts must match;
* `phase_split_counts` — the full 12-cell split x phase matrix exact;
* `shard_hashes` — every shard file re-hashed (list and dict forms).

`tools/s6/build_dataset.py` staging now computes shard SHAs over the
exact bytes written (write_bytes + sha256(data)), removing the
text-vs-bytes ambiguity that made the staged verify fail.

## Artifacts

```
tools/s10/e1_build_1m.py            nested 1M builder (full B1 chain)
tools/s6/verify_dataset.py          +12-cell / split-counts / shard-SHA checks
tools/s6/build_dataset.py           staged shard SHA over written bytes
data/s10/s10-eval-v2-1m01/          the frozen 1M dataset (4 shards)
data/s10/e1-pool-cache/             immutable per-source caches (25)
data/s6/sources/lichess-standard-rated-v5/   v5 source (350k long games)
results/s10/s10-e1-pool-capacity-v2.json
```

## Next

```
E2  SF18 1M labeling (unchanged teacher contract: SF18 6b087694...,
    16384 nodes, Threads 1, Hash 64, MultiPV 1, UCI_ShowWDL true,
    ucinewgame per position, full 1M re-label, checkpoint/resume,
    fresh second-process audit 1000 / 0 mismatches) — GO.
    PLUS the free parent-300k oracle: after labeling, the 300,000
    nested parent positions must reproduce the old B2 labels
    field-for-field (teacher_cp_stm / teacher_mate / teacher_bestmove /
    teacher_wdl_stm), 300000/300000 exact.
E3  1M MATERIAL-RESIDUAL training (F1 recipe frozen verbatim).
```
