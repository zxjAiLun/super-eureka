# S10-C2A Closeout — Move-Aware Dirty Accumulator

**Status: CLOSED / PASS**

## Scope

Replace the C1 64-square board-scan updater (correctness oracle, retained)
with a fixed-size, zero-heap move description for the production search hot
path — verified A==B==C against the C1 reference and full refresh at every
transition. No search wiring, no perf claims (C2B/C3).

## API (src/engine/nnue_v2q_runtime.rs)

```rust
pub struct NnueMoveDelta {          // fixed-size, zero heap
    removed: [Option<(Square, Piece)>; 2],
    added:   [Option<(Square, Piece)>; 2],
    moved_king: Option<Color>,
}

model.prepare_move_delta(&parent_pos, &mv) -> NnueMoveDelta
model.update_accumulator_for_move(&mut acc, &delta, &child_pos) -> UpdateStats
model.full_accumulator_for_perspective(&pos, perspective) -> [i32; 128]
```

Key properties:

- `prepare_move_delta` covers all move kinds with at most 2 removed + 2
  added entries (quiet/double-push, capture, EP incl. the captured pawn's
  distinct square, both castles incl. rook squares, promotions incl.
  capture-promotion). No Vec allocation, no 64-square scan.
- `update_accumulator_for_move` needs only the CHILD position: for every
  perspective whose own king did not move, the king context is identical
  parent/child, so old and new feature indices are both computed with the
  child's context. The hot path never copies the parent board.
- Own-king moves (incl. castling) refresh exactly ONE perspective via
  `full_accumulator_for_perspective` (C1 refreshed both); the opponent
  perspective still deltas the king through channel 10.
- Null-move semantics verified: the fixed White/Black accumulator is
  bit-identical across a side-to-move flip; only the dense STM/NSTM
  ordering changes.

## Verification

```
A == B == C at every transition:
  A = move-aware update (production path)
  B = C1 64-square reference update (live oracle)
  C = full refresh

dirty-squares(Move) == actual changed squares (board diff)   [test-only scan]

100-game deterministic audit (same seed as C1):
  transitions_checked:        9,974
  accumulator_lanes_checked:  2,553,344
  white/black lane mismatches 0
  raw_output_mismatches:      0
  reference_64sq_mismatches:  0     (A==B live oracle)
  flags: normal 9277, dpb 667, ep 1, O-O 6, O-O-O 1, promo 22
  mirror_boundary_king_moves: 247
  full_refreshes: 1,051  delta_updates: 40,423
```

Artifact: results/s10/s10-c2a-move-aware-audit.json
Command: `bench nnue-v2q-accumulator-audit --model <bin> --games 100 --plies 100`

## Tests (cargo 405 pass, +4 C2A)

```
c2a_dirty_squares_match_board_diff        Move-derived dirty set == board
                                          diff (64-scan exists ONLY here)
c2a_move_aware_matches_reference_and_full_refresh
                                          A==B==C on every legal move of 7
                                          scenario FENs (>100 transitions)
c2a_null_move_leaves_accumulator_bit_identical
c2a_move_aware_100_game_playout_with_oracle
                                          100 games, C1 oracle at every ply
```

## Untouched

```
frozen artifact b51a79b1...  (loads; startpos raw 190 unchanged)
C1 64-square updater         (kept as test/reference oracle)
search / UCI / eval wiring   (C2B)
performance claims           (C3)
```
