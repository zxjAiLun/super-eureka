# S10-F1-D1 Deployment & Opening Provenance (frozen)

**Status: RUNNING — AUTHORITATIVE strength evidence (no interim interpretation)**

Frozen before significant results existed (snapshot taken at
completed_pairs = 2 of 1000; tournament creation 2026-08-31T14:38:40Z).

## Tournament

```
tournament_id    0c70792f-67b5-4022-8dbf-2e8378619a88
name             S10-F1 D1 material-residual NNUE same-binary
experiment_id    s10-f1-d1-nnue-v2q-material-300k01 (stage: confirmation)
time_control     bullet_1_0
SPRT             pentanomial, logistic, elo0=0, elo1=+10, alpha=beta=0.05,
                 max_pairs=1000
arena_elo        disabled
```

## Engines (same binary; the ONLY variable is the evaluator)

```
engine_a (candidate)
  preset_id       s10f1-formal-db6c459-material-77404d04
  build_id        20260831-db6c459-s10f1-material-77404d04
  profile         current-final-nnue-v2q-material
  command_args    ["--profile", "current-final-nnue-v2q-material",
                   "--nnue-model", "/opt/chessarena/builds/
                   20260831-db6c459-s10f1-material-77404d04/models/
                   nnue-v2q-material-300k01.bin"]
  uci_options     {}
  git_sha         db6c459155af1dcbe5764c1c970d6845576204ad
  binary_sha256   2f8ef9028dad2b4da83ae7ffdfa2ee48a4283c75d5f9a91bd62d8464aecfb8f3
  model_sha256    77404d04c8d00af2156df1c3f8152156ee2497f220c9dfa902d40b0ab468d613
                  (manifest = DB registration = on-disk file, three-way match)
  model target_mode material_residual (EUNN2Q01 v2 semantic header)

engine_b (baseline)
  preset_id       s10f1-formal-db6c459-currentfinal
  build_id        20260831-db6c459-s10f1-material-77404d04   (same build)
  profile         current-final
  command_args    ["--profile", "current-final"]
  uci_options     {}
  git_sha         db6c459155af1dcbe5764c1c970d6845576204ad   (same)
  binary_sha256   2f8ef9028dad2b4da83ae7ffdfa2ee48a4283c75d5f9a91bd62d8464aecfb8f3
                  (same; no model argv)
```

Provenance note: commit db6c459 also carries the S10-D GUI/UCI
evaluator-delivery changes (NnueMode/EvalFile) from the parallel
workstream. Both arms use this exact same binary; the startup profile
fixes the evaluator, so the UCI NnueMode option is inert for these
arms. History was NOT rewritten.

## Opening sample (fresh evidence)

```
opening_set      stockfish-8moves-v3
book sha256      5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e
format/plies     pgn / 16
seed             2026083101
indices          1000, all unique
indices_sha256   c5a6aee41a2f457174230c1c16ff650d7f45942d939c776905a2d8814b9b9cef
                  (sha256 of the JSON array in s10-f1-d1-frozen-snapshot.json)

excluded         1004 starting FENs = ALL 1000 S10-D1 indices
                  ("S10-D1 NNUE V2Q same-binary confirmation",
                   seed 2026083001) + the 4 A5 smoke indices [8, 4, 11, 6]
excluded_fens_sha256
                  384d5c3dcaad0e407a01eb60926e2deb659f56ff4db2cfa35b3d88f82f6068d3
                  (sha256 of the sorted JSON array of 1004 normalized FENs)
verified         snapshot indices overlap with old-D1 indices = 0
                  snapshot indices overlap with smoke indices  = 0
```

The 1000 frozen indices are in
`results/s10/s10-f1-d1-frozen-snapshot.json` (the authoritative copy is
the tournament's config_snapshot in the Arena DB; this file is the
pre-results provenance capture).

## Deployment chain

```
WSL Ubuntu build  rustc 1.94.1 (e408947bf 2026-03-25), db6c459 clean src
tarball           build-20260831-db6c459-s10f1-material-77404d04.tar.gz
install           arena-deploy build-install (SHA-verified), DB id 15
capability probe  5 profiles recognized; UCI handshake 3 options;
                  uci id Eureka v0.1.0-dev+db6c4591
both arms         launched live on the server before formal start
                  (both reached uciok with the correct profiles)
```

## Historical reference (not part of this experiment)

```
S10-D1 pure NNUE: 104 pairs, 18W/176L/14D = 12.02% score, SPRT_ACCEPT_H0
```

## Interpretation paths (frozen)

```
ACCEPT_H1         -> blitz_3_2 64 pairs confirmation; then 1M residual
ACCEPT_H0 @35-45% -> material anchor rescued a large part of the -346 Elo
                     collapse but not all; decide 1M residual vs capacity
ACCEPT_H0 @10-20% -> do not burn 1M; representation/capacity first
```

No interim interpretation. Escalation only on SPRT_ACCEPT_H1 /
SPRT_ACCEPT_H0 / MAX_PAIRS or an infrastructure anomaly (attempt > 1,
failed/invalid pair, engine crash/timeout, worker restart, model
re-hash failure, opening/snapshot drift, binary provenance mismatch,
server outage).
