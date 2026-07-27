# EVAL 1A — Tapered evaluation and King PST

Status: **IMPLEMENTED — local gates passing; independent review pending**
Scope: evaluation only. This record does not authorize search pruning,
mobility, pawn-structure, or king-safety work.

## Contract

`pub fn evaluate(pos: &Position) -> i32` remains the public entry point. The
score is still from the side-to-move perspective: positive means good for the
side to move and flipping only the side to move negates the score. Evaluation
is read-only and does not call `make_move` on the caller's position.

EVAL 1A retains the M2.4 material values and non-king PST tables. Those tables
are used in both score lanes, so their old semantics do not depend on the
phase. Only the King receives new phase-specific PST terms.

## Score lanes and phase

Each piece contributes a signed middlegame (`mg`) and endgame (`eg`) score.
The phase is non-pawn material only, with the following weights:

| Piece | Phase weight |
|---|---:|
| Pawn | 0 |
| King | 0 |
| Knight | 1 |
| Bishop | 1 |
| Rook | 2 |
| Queen | 4 |

The phase is clamped to `MAX_PHASE = 24`; the normal starting position is
therefore phase 24 and a king-and-pawn ending is phase 0. The final score is:

```text
score = (mg * phase + eg * (24 - phase)) / 24
```

The implementation uses a 64-bit intermediate for the multiplication and
retains Rust's integer-toward-zero division semantics.

## King tables

Both tables are a1-first (`a1 == index 0`) and are vertically mirrored for
Black. Positive middlegame values prefer a sheltered home-rank King; positive
endgame values prefer centralisation.

```text
KING_MG_PST, ranks 1..8:
  20  30  10   0   0  10  30  20
  20  20   0   0   0   0  20  20
 -10 -20 -20 -20 -20 -20 -20 -10
 -20 -30 -30 -40 -40 -30 -30 -20
 -30 -40 -40 -50 -50 -40 -40 -30
 -30 -40 -40 -50 -50 -40 -40 -30
 -30 -40 -40 -50 -50 -40 -40 -30
 -30 -40 -40 -50 -50 -40 -40 -30

KING_EG_PST, ranks 1..8:
 -50 -40 -30 -20 -20 -30 -40 -50
 -30 -20 -10   0   0 -10 -20 -30
 -30 -10  20  30  30  20 -10 -30
 -30 -10  30  40  40  30 -10 -30
 -30 -10  30  40  40  30 -10 -30
 -30 -10  20  30  30  20 -10 -30
 -30 -20 -10   0   0 -10 -20 -30
 -50 -40 -30 -20 -20 -30 -40 -50
```

## Tests and intermediate lineage

The EVAL 1A proof is in `tests/eval1a.rs` and the existing PST regressions:

- phase counts use non-pawn material only;
- interpolation locks both phase extremes and the midpoint;
- the King tables are distinct;
- King shelter and endgame centralisation have directional regressions;
- side symmetry, read-only evaluation, and prior PST/mirror behavior remain
  covered;
- evaluation and FEN restoration are checked across every legal
  `make_move`/`unmake_move` edge in a castling position.

The approved pre-EVAL M4.0 record remains
`startpos = 1149 / b1c3 / cp 50` and
`queen-win = 963 / e4a4 / cp 890` in the historical benchmark documents. On
the current EVAL 1A tree with the M4.0 reference search profile, the
corresponding regression is `1149 / b1c3 / cp 50` and
`969 / e4a4 / cp 990`. The queen-win fixture contains extra material, so it is
not an exact KQK/KRK position and is not changed by EVAL 1B. Later Current-
profile SEE/pruning measurements are recorded separately and do not rewrite
the historical M4.2 documents.

## Non-goals

This milestone does not implement mobility, pawn structure, general king
safety, KQK/KRK mop-up, SEE, qsearch changes, aspiration windows, null move,
LMR, futility, bitboards, NNUE, or time-management changes.
