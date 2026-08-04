# S3-PROMOTION — frozen final candidate launcher

Status: **PREPARED — NOT STARTED**

This document records the final promotion configuration. It does not claim a
promotion result and it does not change the default `Current` profile.

## Frozen engine

```text
engine-code commit: 91347775906f3f5d3730c9e9596037493429776d
engine SHA-256:     b4bf0c3e73158bf3f5c072aa863aca671721275efc5b2e6d5354cf53a0fd0933
candidate profile:  current-final
baseline profile:   current
```

`tools/run_s3_promotion.py` refuses to run if the worktree is dirty, the
engine source differs from the frozen commit, the binary hash differs, either
UCI identity probe reports the wrong profile, or the manager provenance is not
the pinned cutechess-cli 1.5.1 binary.

The default mode is `--dry-run`; starting the manager requires an explicit
`--run`.

## New opening selection

The promotion opening set is a deterministic, disjoint 500-position slice of
the pinned source book:

```text
source: tests/data/openings/d1.14-openings-v1.epd
source lines: 1000..1499
positions: 500 unique non-terminal positions
slice SHA-256: 2c5a8e02c8b62ff934a70706e605f7be3eb8c6a36c11e701d802be2c785264c7
```

The earlier D1.14 partial run consumed only the first 413 sequential source
positions, and S3-FINAL consumed only the first 50. S2.1 and S2.2 used fixed
position manifests rather than this opening source. The selection contract is
committed in [`s3-promotion-openings-v1.json`](../../tests/data/openings/s3-promotion-openings-v1.json).

## Match contract

The launcher writes the exact command before starting cutechess:

```text
candidate first: CurrentFinal vs Current
time control:    10+0.1
Hash:            16 MB
Threads:         1
concurrency:     1
maximum:         1,000 games / 500 complete pairs
pairing:         sequential openings, repeat=2, strict color swap
SPRT:            H0=+20 Elo, H1=+60 Elo, alpha=0.05, beta=0.05
```

The launcher records engine and manager hashes, both UCI identities, the
runtime opening hash, the complete argv, and stdout/stderr/PGN paths in the
run manifest. It also re-checks the engine hash, manager hash, and Git tip
after the manager exits.

No S3-PROMOTION games have been started by this preparation change. A later
run must independently verify the completed PGN, pair/color contract, legal
moves, timeouts, crashes, and the manager's SPRT decision before any
promotion discussion. `Current` is not replaced automatically.
