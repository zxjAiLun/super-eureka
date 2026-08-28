# S10-B2 Closeout — Real 300k Stockfish-18 Teacher Labeling

**Status: CLOSED / PASS**

## Execution

Frozen command (executed verbatim, only interpreter name `python`→`python3`):

```bash
python3 tools/s6/label_teacher.py \
    --dataset data/s10/s10-eval-v1-300k01 \
    --native \
    --checkpoint-interval 500
```

Run log: `results/s10/s10-b2-labeling-run.log`

## Frozen identity

```
dataset_id:      s10-eval-v1-300k01
dataset_sha256:  503b47b6a6fb33f3248e0f15d69de67fcd4334bdefce174767b720910a9076b3
teacher binary:  ~/sf18 (sha256 6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9)
UCI identity:    Stockfish 18 / the Stockfish developers (see AUTHORS file)
nodes:           16384 (labeling_mode: go nodes)
Threads:         1
Hash:            64
MultiPV:         1
UCI_ShowWDL:     true
syzygy:          disabled (default)
```

## Resume behavior (B2A infrastructure exercised in production)

- Run began from a prior interrupted run's state: 37500 committed positions
  with a 440-record uncommitted crash tail (90668 bytes).
- Preflight PASS (dataset_id/SHA/records_total/PID checks) before any teacher
  process was created.
- Uncommitted tail discarded (by design, untrusted); labeling resumed from
  checkpoint 37500/300000.
- `resume_count: 1`, `checkpoint_interval: 500`.

## Results

```
labeled_positions: 300000
labels_sha256:     bcd49da1ece75a15591e135d5bcf6d036608b1759d6a00e639f3e344e516116f
audit:
  mode:       fresh-second-pass (new Stockfish process after labeling engine destroyed)
  checked:    1000
  mismatches: 0
  ok:         true
  sample_position_id_sha256: ad908ad99a489486f162910be71fb2a1223be9f59fe45f4a877501199ea8aad0
```

Audit line from run log: `audit [fresh-second-pass]: PASS (1000 checked, 0 mismatches)`

## Final verification (no --allow-unlabeled)

```bash
python3 tools/s6/verify_dataset.py --dataset data/s10/s10-eval-v1-300k01
```

```
board.is_valid audit: 300000/300000 PASS
records: 300000  splits: {'train': 240000, 'validation': 30000, 'holdout': 30000}
phase buckets: {'high': 75000, 'mid': 135000, 'low': 60000, 'zero': 30000}
VERIFY_PASS
```

## Wall-clock telemetry

- Process start: 2026-08-28 12:43 (+0800)
- Final artifact publish (teacher_manifest.json): 2026-08-28 15:00:19 (+0800)
- Total wall-clock (including resume-discard, 262500 labeled positions, and
  1000-position fresh-second-pass audit): **2h 17m 20s**
- Effective labeling rate ≈ 25.6 positions/s at nodes=16384 / Threads=1

## Cleanup state

```
labels.partial.jsonl:  REMOVED (clean)
teacher-progress.json: REMOVED (clean)
labels.jsonl:          published (61793432 bytes, 300000 lines)
teacher_manifest.json: published LAST (1488 bytes)
```

## Git policy

300k `labels.jsonl` is NOT committed (`.gitignore` covers `data/s10/`).
`teacher_manifest.json` key fields, labels SHA, verify result, and
wall-clock telemetry are archived in this report under `results/s10/`.
