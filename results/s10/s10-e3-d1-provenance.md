# S10-E3-D1 Deployment & Opening Provenance (frozen)

**Status: RUNNING — AUTHORITATIVE (no interim interpretation)**

Frozen at completed_pairs = 0 of 1000.

## Tournament

```
tournament_id    21d96a29-06ff-4458-b6f0-e7678ac34c43
name             S10-E3 D1 1M material-residual NNUE same-binary
experiment_id    s10-e3-d1-nnue-v2q-material-1m01 (stage: confirmation)
time_control     bullet_1_0
SPRT             pentanomial logistic, elo0=0, elo1=+10, alpha=beta=0.05,
                 max_pairs=1000
arena_elo        disabled
```

## Engines (same binary; only the evaluator differs)

```
engine_a (candidate)
  preset_id       s10e3-formal-bffb53c-material-1m-daddd085
  build_id        20260902-bffb53c-s10e3-1m-daddd085
  profile         current-final-nnue-v2q-material
  command_args    ["--profile", "current-final-nnue-v2q-material",
                   "--nnue-model", "/opt/chessarena/builds/
                   20260902-bffb53c-s10e3-1m-daddd085/models/
                   nnue-v2q-material-1m.bin"]
  uci_options     {}
  git_sha         bffb53c2a78fe48c8f484804dd1ea096f62c5f9a
  binary_sha256   41d64a9d508c27430db462c84c3d4e5eec55b3bf0ae62e04937e62a30f1b51d0
  model_sha256    daddd085d7260362d40a53ee4f1643a0b96125d36e53591babade99152104de4
                  (manifest = DB = on-disk, three-way match)
  model target_mode material_residual; trained on s10-eval-v2-1m01
                  (Windows SF18 teacher labels 5765fedb...);
                  val composed MAE 138.578 / holdout 138.693

engine_b (baseline)
  preset_id       s10e3-formal-bffb53c-currentfinal
  build_id        20260902-bffb53c-s10e3-1m-daddd085   (same build)
  profile         current-final
  command_args    ["--profile", "current-final"]        (no model argv)
  uci_options     {}
```

Same-binary note: between db6c459 (F1-D1) and bffb53c the ONLY engine
source change is `src/engine/nnue_v2q_runtime.rs` (S10-E3 i64 L1 MAC);
search, CurrentFinal eval, and UCI dispatch untouched.

Runtime telemetry (non-blocking): startpos 200k-node search
228,550 NPS (AVX2, i64-widened L1), ~3% below the F1-era incremental
reference — no anomaly. Accumulator audit on the 1M artifact: 1200
transitions, 0 lane/output mismatches.

## Opening sample (fresh evidence)

```
opening_set      stockfish-8moves-v3 (sha 5835239f...)
format/plies     pgn / 16
seed             2026090201
indices          1000, unique
indices_sha256   21535466216e2a796365956c27da1c95aad483bd0a3e713158d392c49def8060
excluded         2004 FENs = old S10-D1 (1000, seed 2026083001)
                  + F1-D1 (1000, seed 2026083101) + A5 smoke (4)
verified         overlap with old D1 = 0; F1-D1 = 0; A5 = 0
                  (checked post-creation against the frozen snapshot)
startup smoke    candidate + baseline launched live on the server;
                 candidate loaded model SHA verified, baseline clean
```

## Result paths (frozen)

```
ACCEPT_H1            -> 3+2 64 pairs; then production promotion talk
ACCEPT_H0 > 40.76%   -> data scale helps but insufficient; capacity/width next
ACCEPT_H0 ~ 40%      -> offline MAE gain did not convert; representation first
MAX_PAIRS            -> frozen statistical evidence, no threshold edits
```

Escalation only on terminal SPRT/MAX_PAIRS or infra anomaly.
