# S3-PROMOTION — frozen final candidate launcher

Status: **COMPLETE — H1 ACCEPTED; CURRENT UNCHANGED**

This document records the final promotion configuration and its completed
candidate-first SPRT result. It does not automatically replace the default
`Current` profile.

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
the pinned cutechess-cli 1.5.1 binary with SHA-256
`8889f9582dc688c567704cf083f6025baf77f791cde903698c70b3420caf5d7e`.

The CLI requires an explicit mode. `--dry-run` performs a non-persistent
preflight in a system temporary directory; starting the manager requires an
explicit `--run`.

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

## Completed promotion run

The frozen launcher was run once from the clean `d138ab9` worktree. Cutechess
stopped after a valid SPRT boundary rather than reaching the 1,000-game cap:

```text
games:             106
complete pairs:     53
candidate:          62 wins / 28 losses / 16 draws
candidate score:    66.038%
candidate colors:   53 White / 53 Black
SPRT:               H1 accepted
manager return:     0
```

The manager stdout reported:

```text
Elo difference: 115.5 +/- 64.6, LOS: 100.0 %
SPRT: llr 3.03, lbound -2.94, ubound 2.94 - H1 was accepted
```

These Elo and LOS values are descriptive manager output; the formal result is
the explicit `H1 accepted` boundary. The early stop is not treated as a
1,000-game result. Both colors were represented equally, and no manager
stdout failure token, illegal move, timeout, forfeit, or crash was found.

The independent verifier [`verify_s3_promotion.py`](../../tools/verify_s3_promotion.py)
replayed all 106 PGN games, checked all moves for legality, confirmed the
first 53 runtime openings occurred in order with strict color reversal, and
matched the manager W/D/L line to the PGN. It also rechecked the frozen engine
and cutechess SHA-256 values and the unchanged engine source lineage. Its
canonical output is [`verification.json`](../../results/s3-promotion/run-001/verification.json).

The complete run artifacts are [`manifest.json`](../../results/s3-promotion/run-001/manifest.json),
[`command.txt`](../../results/s3-promotion/run-001/command.txt),
[`openings.epd`](../../results/s3-promotion/run-001/openings.epd),
[`manager.stdout.log`](../../results/s3-promotion/run-001/manager.stdout.log),
[`manager.stderr.log`](../../results/s3-promotion/run-001/manager.stderr.log),
and [`match.pgn`](../../results/s3-promotion/run-001/match.pgn).

This result qualifies `CurrentFinal` for a separately reviewed production
promotion commit, but `Current` remains unchanged in this artifact commit.
