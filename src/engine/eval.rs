//! Evaluation.
//!
//! Returns the position's value from the side-to-move's perspective
//! (positive = good for the side to move). The score combines the
//! piece material (each piece's `PieceType::value()`) with
//! piece-square-table bonuses. The King contributes material only,
//! since its PST is all zeros, until a tapered evaluation adds a
//! real King PST later.
//!
//! Planned additions (one at a time, each validated by self-play):
//!   - bishop pair bonus
//!   - mobility
//!   - pawn structure (doubled/isolated/passed)
//!   - king safety
//!   - endgame-phase weighting (tapered eval, real King PST)
//!
use crate::chess::position::Position;
use crate::chess::types::{Color, PieceType};

/// Piece-square tables, a1-first (index 0 = a1 ... 63 = h8), one row per
/// rank (rank 1 first). Values are from Tomasz Michniewski's "Simplified
/// Evaluation Function"; the public tables start at a8 (rank 8 first), so
/// they were reversed rank-wise to match this engine's `a1 = 0`. Black
/// pieces reuse the same tables via a vertical mirror (`sq ^ 56`, see
/// `pst_idx`).
///
/// These are locked by `docs/specs/m2.4-piece-square-evaluation.md` —
/// do NOT re-tune them here.
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
    // a1 .. h1  (rank 1)
         0,   0,   0,   0,   0,   0,   0,   0,
    // a2 .. h2  (rank 2)
         5,  10,  10, -20, -20,  10,  10,   5,
    // a3 .. h3  (rank 3)
         5,  -5, -10,   0,   0, -10,  -5,   5,
    // a4 .. h4  (rank 4)
         0,   0,   0,  20,  20,   0,   0,   0,
    // a5 .. h5  (rank 5)
         5,   5,  10,  25,  25,  10,   5,   5,
    // a6 .. h6  (rank 6)
        10,  10,  20,  30,  30,  20,  10,  10,
    // a7 .. h7  (rank 7)
        50,  50,  50,  50,  50,  50,  50,  50,
    // a8 .. h8  (rank 8)
         0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    // a1 .. h1
        -50, -40, -30, -30, -30, -30, -40, -50,
    // a2 .. h2
        -40, -20,   0,   5,   5,   0, -20, -40,
    // a3 .. h3
        -30,   5,  10,  15,  15,  10,   5, -30,
    // a4 .. h4
        -30,   0,  15,  20,  20,  15,   0, -30,
    // a5 .. h5
        -30,   5,  15,  20,  20,  15,   5, -30,
    // a6 .. h6
        -30,   0,  10,  15,  15,  10,   0, -30,
    // a7 .. h7
        -40, -20,   0,   0,   0,   0, -20, -40,
    // a8 .. h8
        -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    // a1 .. h1
        -20, -10, -10, -10, -10, -10, -10, -20,
    // a2 .. h2
        -10,   5,   0,   0,   0,   0,   5, -10,
    // a3 .. h3
        -10,  10,  10,  10,  10,  10,  10, -10,
    // a4 .. h4
        -10,   0,  10,  10,  10,  10,   0, -10,
    // a5 .. h5
        -10,   5,   5,  10,  10,   5,   5, -10,
    // a6 .. h6
        -10,   0,   5,  10,  10,   5,   0, -10,
    // a7 .. h7
        -10,   0,   0,   0,   0,   0,   0, -10,
    // a8 .. h8
        -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
    // a1 .. h1
          0,   0,   0,   5,   5,   0,   0,   0,
    // a2 .. h2
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a3 .. h3
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a4 .. h4
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a5 .. h5
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a6 .. h6
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a7 .. h7
          5,  10,  10,  10,  10,  10,  10,   5,
    // a8 .. h8
          0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    // a1 .. h1
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    // a2 .. h2
        -10,   0,   5,   0,   0,   0,   0, -10,
    // a3 .. h3
        -10,   5,   5,   5,   5,   5,   0, -10,
    // a4 .. h4
          0,   0,   5,   5,   5,   5,   0,  -5,
    // a5 .. h5
         -5,   0,   5,   5,   5,   5,   0,  -5,
    // a6 .. h6
        -10,   0,   5,   5,   5,   5,   0, -10,
    // a7 .. h7
        -10,   0,   0,   0,   0,   0,   0, -10,
    // a8 .. h8
        -20, -10, -10,  -5,  -5, -10, -10, -20,
];

/// King PST is all zeros for now (a tapered eval with a real King PST is a
/// later milestone). The King therefore contributes material only.
const KING_PST: [i32; 64] = [0; 64];

/// Map a piece type to its piece-square table. An exhaustive `match` (not
/// `table[piece_type as usize]`) so the compiler forces every new piece
/// kind to be handled explicitly instead of silently indexing the wrong row
/// if the `PieceType` enum order ever changes.
fn pst_for(pt: PieceType) -> &'static [i32; 64] {
    match pt {
        PieceType::Pawn => &PAWN_PST,
        PieceType::Knight => &KNIGHT_PST,
        PieceType::Bishop => &BISHOP_PST,
        PieceType::Rook => &ROOK_PST,
        PieceType::Queen => &QUEEN_PST,
        PieceType::King => &KING_PST,
    }
}

/// Square index into a piece-square table. White uses the square directly.
/// Black mirrors it vertically: `sq ^ 56` flips the rank bits (0b111000)
/// while leaving the file unchanged, so the same table serves both colors.
#[inline]
fn pst_idx(sq: usize, color: Color) -> usize {
    if color == Color::Black {
        sq ^ 56
    } else {
        sq
    }
}

/// Static evaluation from the side-to-move's perspective: material plus
/// piece-square-table bonuses. Read-only — it never mutates `pos`.
pub fn evaluate(pos: &Position) -> i32 {
    let mut score = 0;
    for sq in 0..64usize {
        if let Some(p) = pos.board[sq] {
            let sign = if p.color == pos.side { 1 } else { -1 };
            let material = p.piece_type.value();
            let bonus = pst_for(p.piece_type)[pst_idx(sq, p.color)];
            score += sign * (material + bonus);
        }
    }
    score
}
