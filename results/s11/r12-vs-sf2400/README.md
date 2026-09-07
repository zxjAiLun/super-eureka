# S11 · R12-inc vs Stockfish Elo-2400 — Arena Match Setup

**Status: READY — match to be created on the website by the user.**
(2026-09-08; all server-side registration complete and verified)

## What is deployed

**Engine (candidate arm)** — Eureka R12 incremental NNUE, the current
strongest NNUE model per the S11 program:

```
build_id     20260908-562e77c-s11b4b-r12inc-8eacd0c1
preset_id    s11b4b-r12inc-562e77c  (display: "Eureka R12-inc 562e77c")
git_sha      562e77cee9d772b1b3d70018aea6491e545e25ca  (branch
             s10/nnue-production-foundation; clean git-archive build,
             EUREKA_GIT_* provenance env, WSL rustc 1.94.1)
binary       f716680104890317db5c9b48857c59a9dda92501278db190d71b1b5e25b94c54
model        models/nnue-v2-q01-material-r12-v4.bin
             sha256 8eacd0c1a20cae0fd7cb5bd5a96ecbe00917a49ec1e801f0cec71e66298b6619
             (R12 seed 20260819 checkpoint, v4 format)
profile      current-final-nnue-v2q-material-r12-inc
command_args ["--profile", "current-final-nnue-v2q-material-r12-inc",
              "--nnue-model", "/opt/chessarena/builds/<build_id>/models/
              nnue-v2-q01-material-r12-v4.bin"]
```

**Opponent (baseline arm)** — already registered on the site:

```
preset_id    stockfish-limited-2400  (display: "Stockfish Limited 2400")
build        stockfish-18-avx2-linux-x86_64
options      {UCI_LimitStrength: true, UCI_Elo: 2400}
```

## Evidence trail for this deployment

- Engine identity verified server-side as the chessarena worker user:
  `id name Eureka v0.1.0-dev+562e77ce`,
  `info string profile current-final-nnue-v2q-material-r12-inc`,
  `info string network nnue-v2r12-incremental`.
- Binary SHA verified twice (local build + on-server re-hash; matches
  manifest = DB = on-disk).
- `probe_build_capabilities.py` backfilled the UCI options schema
  (3 options; required by the capability-aware match runtime).
- Model quality/parity provenance: results/s10/s11-b4b-*.json
  (tree identity 0 mismatch; paired NPS 0.9488).

## Why this engine is "the strongest NNUE model"

S11 chain: R12 sidecar = largest task-aligned offline gains of the
program (lockbox pairwise +10.4~10.9pp, regret -47.7~-60.2cp, val MAE
127.2 vs E3 138.6; S11-A verdict). B4-B made it production-speed
(NPS 0.9488 vs E3 V2 baseline; B2 fresh-reference was 0.5805).

## Local smoke (Windows, before server deployment)

4 games 10+0.1 vs SF18 UCI_Elo=2400 through cutechess: 2-2. PGN and
command archived in this directory (smoke.pgn, command.txt). The local
background match was superseded by this site deployment and stopped.

## Suggested match settings (user's choice on the site)

- Engine A: Eureka R12-inc 562e77c; Engine B: Stockfish Limited 2400
- TC: blitz_10_01 (10s+0.1s, the S3 house standard) or rapid_5_3
- Games: >= 400 for a usable Elo estimate; 1000 for a tight CI
  (score-based estimate; this is a rating measurement, not an SPRT)

## Interpretation guide

- Eureka's CCRL-style anchor is unknown for the NNUE line; prior HCE
  CurrentFinal beat SF-limited-1800-class engines comfortably, and the
  S8 Eval2 SPRT (+71.3 Elo vs pre-Eval2) set the HCE baseline.
- The S10-E3/F1/G1 V2-NNUE candidates were each REJECTED vs
  CurrentFinal HCE (SPRT H0). R12's offline task-aligned gains are the
  first to beat E3 across every lockbox metric — this match is the
  first live-strength readout of that line, and of the
  relation-sidecar representation in general.
