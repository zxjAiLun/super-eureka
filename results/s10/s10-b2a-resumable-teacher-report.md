# S10-B2A Closeout Report: Resumable Stockfish-18 Teacher Labeling

## Summary
`tools/s6/label_teacher.py` now supports crash-consistent, fail-closed
checkpoint/resume of teacher labeling without changing any frozen teacher,
audit, or publication semantics.

- **Base commit**: `476d383e0ed889d439e42b65aec32a29a797c517`
- **Head commit**: see `git log -1` on this change
- **Frozen dataset**: `s10-eval-v1-300k01`, SHA-256
  `503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3`
  (NOT rebuilt, NOT modified)

## Frozen teacher contract (unchanged)
- Stockfish 18, binary SHA-256
  `6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9`
- `go nodes 16384`; Threads=1, Hash=64, MultiPV=1, UCI_ShowWDL=true,
  Syzygy disabled (default)
- per-position `ucinewgame` + `position fen` + `go nodes`
- score/WDL side-to-move convention
- teacher death: respawn + retry same position once; second failure aborts
- final audit: fresh Stockfish process, deterministic first N positions,
  exact field-by-field equality (never tolerance)
- final `labels.jsonl` serialization: position_id-sorted (unchanged)

## Checkpoint design
- `labels.partial.jsonl`: append-only journal in DATASET RECORD ORDER
  (partial line i == dataset record i; enables trivial prefix validation).
- `teacher-progress.json`: the CHECKPOINT COMMIT RECORD; written atomically
  (tmp + fsync + os.replace) every `--checkpoint-interval` (default 500)
  completed positions. Binds:
  - dataset identity: `dataset_id`, `dataset_sha256`, `record_count`,
    `ordered_position_id_sha256` (sha256 of `"\n".join(pids)`, no trailing
    newline — frozen serialization, test-locked)
  - teacher identity: `teacher_binary_sha256`, `teacher_nodes`,
    `teacher_options` (canonical key set)
  - committed prefix: `completed_count`, `partial_size_bytes`,
    `partial_labels_sha256` (incremental SHA-256 state, no full re-hash)
- Checkpoint protocol per interval: append lines -> flush -> fsync partial ->
  atomic progress update. The final tail (interval not a divisor) gets a
  forced final checkpoint so `completed_count == len(records)` pre-audit.

## Resume semantics (fail-closed)
| Case | State | Action |
|---|---|---|
| 1 | no progress, no partial | fresh run |
| 2 | both present | full validation + resume |
| 3 | progress, no partial | FAIL CLOSED (committed journal lost) |
| 4 | partial, no progress | FAIL CLOSED (orphan journal) |
| 5 | partial bytes > committed size | truncate uncommitted tail, resume |
| 6 | partial bytes < committed size | FAIL CLOSED (committed data lost) |

Validation chain: progress schema -> dataset identity -> teacher identity ->
committed-prefix SHA -> JSONL parse (line count == completed_count) ->
line-by-line position_id == records[i] (rejects duplicate / unknown /
reordered / non-prefix) -> LABEL_FIELDS present + nodes == 16384.

## Publication
Unchanged: after final checkpoint, fresh-process audit PASS ->
`labels.jsonl.tmp` (fsync) -> `teacher_manifest.json.tmp` (fsync) ->
`os.replace(labels)` -> `os.replace(teacher_manifest)` LAST (commit point).
`teacher_manifest.json` gains resume telemetry: `enabled`,
`checkpoint_interval`, `ordered_position_id_sha256`, `resume_count`,
`checkpoint_schema_version`. After successful publish, partial/progress are
best-effort deleted (unlink failure is a warning only). A dataset with only
partial/progress present remains UNLABELLED for `verify_dataset.py`
(test-proven). Final labels + stale partial together FAIL CLOSED.

## Test matrix (`tools/s6/test_label_teacher_resume.py`, 20 tests)
| ID | Scenario | Result |
|---|---|---|
| T1 | fresh run 17 positions, interval 5; publishes + cleans up | PASS |
| T2 | interrupt at 8, resume; final labels BYTE-IDENTICAL to uninterrupted | PASS |
| T3 | uncommitted crash tail (2 extra lines) truncated; resumed; byte-identical | PASS |
| T4 | partial shorter than committed size | FAIL CLOSED |
| T5 | committed byte flipped (SHA mismatch) | FAIL CLOSED |
| T6 | progress dataset_sha256 tampered | FAIL CLOSED |
| T7 | ordered PID hash mismatch (reordered records, same SHA/count) | FAIL CLOSED |
| T8 | teacher binary SHA mismatch | FAIL CLOSED |
| T9 | nodes / Threads / Hash mismatch | FAIL CLOSED |
| T10 | duplicate PID in committed partial | FAIL CLOSED |
| T11 | unknown PID in committed partial | FAIL CLOSED |
| T12 | reordered PID prefix (A C instead of A B) | FAIL CLOSED |
| T13 | orphan partial (no progress) | FAIL CLOSED |
| T14 | orphan progress (no partial) | FAIL CLOSED |
| T15 | interrupted run leaves NO labels.jsonl / teacher_manifest.json | PASS |
| T16 | audit mismatch: no publication, partial/progress retained | PASS |
| T17 | partial-only dataset: verify_dataset FAILS unlabeled-mode, passes --allow-unlabeled | PASS |
| T18 | final labels stay pid-sorted | PASS |
| T19 | ordered-PID SHA serialization frozen | PASS |
| T20 | teacher death at position 3: respawn + retry recovers, byte-identical | PASS |

## Real Stockfish smoke (native, frozen binary SHA `6b08...`)
1. 12-position copy of the frozen 300k dataset, interval 5, audit-n 6:
   checkpoints 5/10/12 -> fresh-process audit PASS -> published.
2. Independent duplicate run: labels byte-identical.
3. 1000-position copy, interval 500: SIGKILL at ~25s left
   `completed_count=500` + partial, no final artifacts; re-running the
   SAME command resumed from 500/1000, fresh-process 1000-position audit
   PASS, published. Final labels SHA `d91736681700f80de8b56892235f39a156316a04dbc13508bb2f7b894c798c05`
   BYTE-IDENTICAL to the uninterrupted reference run.

## Regression gates
- `tools/s6/test_label_teacher_resume.py`: 20/20 PASS
- `tools/s6/test_label_teacher.py` (existing audit tests): 6/6 PASS
- `tools/s6/test_verify_dataset.py` + `tools/s6/test_build_dataset.py`: 18/18 PASS
- `cargo test --lib`: 372/372 PASS
- `src/engine/eval.rs`, `src/engine/search.rs`, `src/engine/nnue.rs`: no diff
  (B2A touches only `tools/s6/label_teacher.py` + new test file)
- `NnueFeatureSetV2` untouched; frozen dataset SHA unchanged

## Production usage (S10-B2)
```text
python tools/s6/label_teacher.py \
    --dataset data/s10/s10-eval-v1-300k01 \
    --native \
    --checkpoint-interval 500
```
After any interruption, re-run the exact same command; the tool validates
provenance, discards any uncommitted tail, and resumes from the last
committed checkpoint. At 300000/300000 it runs the fresh-process
1000-position exact replay audit and publishes `labels.jsonl` then
`teacher_manifest.json` (last, commit point).

## S10-B2A Verdict: CLOSED / PASS
All 13 acceptance conditions met (fresh labeling; interrupt+resume byte
identity; crash-tail truncation; corruption fail-closed; dataset/teacher
identity fail-closed; PID validation fail-closed; partial never publishes;
audit semantics unchanged; final serialization unchanged; frozen Stockfish
settings; frozen dataset SHA; real Stockfish smoke; regression gates).

## Repair 1 (post-review): frozen-dataset preflight integrity gate
Review found the pipeline trusted `dataset_manifest.json` without
re-verifying the local dataset bytes (the 300k shards are local-only, not
in Git). Added `preflight_dataset()` in `tools/s6/label_teacher.py`,
running BEFORE any Teacher process is instantiated, before any partial
file is created or mutated, and before resume validation:

- dataset_id non-empty string; dataset_sha256 64 lowercase hex
- `manifest.records_total == len(records)`
- recomputed canonical dataset SHA (exact builder/verify_dataset
  serialization) `== manifest.dataset_sha256`
- position_id uniqueness

A mutated local shard now fails closed even when `teacher-progress.json`
is internally valid, and the partial/progress files are left untouched.
Also locked the checkpoint interval into the resume contract (resume with
a different `--checkpoint-interval` fails closed; production rule is to
re-run the exact same command).

New tests (T21–T25 + interval lock, 27 total):
| ID | Scenario | Result |
|---|---|---|
| T21 | manifest records_total mismatch | FAIL CLOSED, 0 Teacher instances |
| T22 | mutated record FEN, manifest unchanged (SHA mismatch) | FAIL CLOSED, 0 Teacher instances |
| T23 | removed record line | FAIL CLOSED, 0 Teacher instances |
| T23b | post-checkpoint dataset mutation on RESUME | FAIL CLOSED, partial/progress untouched |
| T24 | duplicate position_id in dataset (SHA consistent) | FAIL CLOSED |
| T25 | valid dataset passes preflight, pipeline unchanged | PASS |
| P2 | checkpoint_interval mismatch on resume | FAIL CLOSED; same interval resumes |

Verified against the real frozen dataset: preflight recomputes
`503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3` over
300000 records. Real Stockfish 12-position smoke after the repair:
byte-identical labels. Regression: 27/27 resume tests, 24/24 existing
teacher/verify/build tests, `cargo test --lib` 372/372.
