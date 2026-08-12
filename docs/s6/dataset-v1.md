# S6.0 Dataset v1 — s6-eval-v1-core-shard01

STATUS: **NON-FINAL shard** (final id `s6-eval-v1-core-300k` is gated on a
second independent source; see dataset_manifest.json `not_final_reason`).

## Identity

```text
dataset_id:   s6-eval-v1-core-shard01
seed:         20260812
schema:       1
target:       300,000 (final); this shard: 5,919 unique records
splits:       train 4,776 / validation 500 / holdout 643
sources:      18 Arena historical tournament PGNs (1,572 games), single
              source family — second independent source required for FINAL
dataset_sha256: 3a3483fd46fd5a57...
```

## Construction rules (mirror of the builder)

- sampling: hash(source_game_id, ply, seed), min ply 12, max ply 160,
  max 8 positions per game, ~50% ply acceptance;
- eligibility: legal, both kings, non-terminal, side to move has a legal
  move, side to move NOT in check;
- identity: canonical FEN = first four FEN fields;
  position_id = sha256(canonical_fen4);
- split: per game (deterministic hash, 80/10/10); each game in exactly one
  split; cross-split FEN duplicates removed (deterministic keep);
- dedup: 875 duplicates removed (12,534 -> 11,659 unique, quota-trimmed to
  5,919);
- phase: sum of piece weights N/B=1 R=2 Q=4 clamped to 24; quota targets
  25/45/20/10 (high/mid/low/zero); this shard is high/mid-heavy and reports
  the shortfall explicitly (zero/pawn-endgame and low buckets need the
  second source);
- storage: deterministic JSONL shards; the manifest's dataset_sha256 is the
  hash of the canonical representation (sorted JSON lines concatenation).

## Labels

- teacher: Stockfish 18, binary sha256 6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9,
  Threads=1 Hash=64 MultiPV=1 UCI_ShowWDL=true Syzygy disabled,
  `go nodes 16384` per position, `ucinewgame` before each position;
- labels.jsonl keyed by position_id (side-to-move perspective);
- determinism audit: 1,000 positions re-labeled, 0 mismatches (PASS);
- 28 mate labels present (excluded from cp regression by the benchmark).

## CurrentFinal holdout baseline (results/s6/baseline_metrics.json)

```text
current-final material+pst: MAE 149.7, RMSE 227.7, median 98,
Pearson 0.416, Spearman 0.525, sign 67.0%, Texel logistic 0.734
dormant E2 (diagnostic):   MAE 148.7, RMSE 230.1, median 97,
Pearson 0.477, Spearman 0.586, sign 71.2%, Texel logistic 0.745
```

The learned evaluator (S6.2+) must beat these on the same holdout.
