# S10-C1 Closeout — Incremental Accumulator Correctness

**Status: CLOSED / PASS — bit-exact, zero tolerance**

## Question answered

> From a parent position's full-refresh accumulator, updating by the board
> changes of one `make_move`, are the White/Black 128-lane integer
> accumulators BIT-EXACT with a fresh full refresh of the child position?

**Yes — across 9,974 transitions and 2,553,344 lanes: zero mismatches.**

## Design (frozen)

```
NnueV2Accumulator: always White/Black perspective (never STM/NSTM);
                   the dense forward swaps by side-to-move.

update_accumulator(before, after):
  per perspective:
    own king square changed -> FULL REFRESH that perspective
    (V2 own-king conditioning controls orientation, horizontal mirror,
     and bucket; a king move can flip the whole perspective's mirror
     regime — no bucket-lookalike shortcuts)
    otherwise -> 64-square board diff:
      before-piece feature row -> subtract
      after-piece feature row  -> add
    (opponent king participates normally as channel 10)

single feature formula: nnue::v2_feature_for_piece() — the SAME helper
now backs active_features_v2() (full refresh) and the incremental deltas;
no third copy of the V2 encoding exists.

evaluate_raw(pos) = full_accumulator(pos) -> evaluate_raw_from_accumulator:
one fact path for the dense network.
```

The 64-square diff deliberately trades performance for correctness
(C1 scope); Move-aware dirty-piece extraction is C2/C3 work.

## Gate result (results/s10/s10-c1-incremental-accumulator.json)

```
games:                    100 deterministic random legal playouts
plies max per game:       100
transitions_checked:      9,974
accumulator lanes checked 2,553,344
white lane mismatches:    0
black lane mismatches:    0
raw_output_mismatches:    0

delta updates:            40,423 feature applications
full refreshes:           1,051 (own-king moves)
mirror boundary king moves: 247 (a-d <-> e-h regime flips)

move flags covered:
  normal 9,277 | double_pawn_push 667 | en_passant 1
  king_castle 6 | queen_castle 1 | promotion 22

passed: true
```

Command: `bench nnue-v2q-accumulator-audit --model <bin> --games 100
--plies 100` (deterministic xorshift, fixed seed 0x5989d57...).

## Test matrix (src/engine/nnue_v2q_runtime.rs, all in cargo test)

```
c1_quiet_pawn_capture_sequence         e4/d5/exd5 playout chain
c1_en_passant_three_square_change      d4xe3 e.p.; 3 changed squares;
                                       exactly 6 delta applications, 0 refreshes
c1_castling_both_sides                 white O-O and black O-O-O; the mover's
                                       perspective refreshes (1), the other deltas
c1_promotions_quiet_and_capture        Q/N quiet + Q capture promotions
c1_own_king_move_triggers_perspective_refresh
c1_horizontal_mirror_boundary_king_moves
                                       d<->e crossings + within-regime controls
c1_make_unmake_branch                  every legal move of a castling-rich FEN:
                                       make -> incremental == fresh; unmake ->
                                       parent snapshot == parent full refresh
c1_deterministic_random_legal_playout_100_games
                                       100 games x <=100 plies, every ply exact

nnue::v2_feature_for_piece_matches_active_features_v2
                                       no-drift: per-piece formula == set of
                                       active_features_v2, 9 FENs x 2 perspectives
nnue::v2_feature_for_piece_excludes_own_king
```

## Environment

```
engine SHA256: ab5bd16b7ceb0e5b09e8db3a42108ac28c1dd40c921be416496f53463758f2b8
frozen quantized artifact: b51a79b1... (unchanged, loads, startpos raw 190)
cargo test --release: 401 passed (21 nnue_v2q incl. 8 C1 tests + 2 no-drift)
```

## Explicitly out of scope (deferred)

```
performance / NPS ......................... C2/C3 (64-square diff is NOT the
                                            production update path)
search-stack integration .................. C2
Move-aware dirty-piece extraction ......... C2
Arena ..................................... D
```
