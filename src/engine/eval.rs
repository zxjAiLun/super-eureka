//! Evaluation.
//!
//! Returns the position's value from the side-to-move's perspective
//! (positive = good for the side to move). EVAL 1A keeps the existing
//! material and non-king PST values, but evaluates them through separate
//! middlegame/endgame score lanes and interpolates them by non-pawn phase.
//! The King now has distinct middlegame safety and endgame-centralisation
//! PSTs. This module deliberately does not add mobility, pawn structure,
//! king-safety features, or search pruning.
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

/// Middlegame King PST, a1-first. Safe shelter squares near the home rank
/// are preferred while central exposure is penalised.
#[rustfmt::skip]
const KING_MG_PST: [i32; 64] = [
    // a1 .. h1
        20,  30,  10,   0,   0,  10,  30,  20,
    // a2 .. h2
        20,  20,   0,   0,   0,   0,  20,  20,
    // a3 .. h3
       -10, -20, -20, -20, -20, -20, -20, -10,
    // a4 .. h4
       -20, -30, -30, -40, -40, -30, -30, -20,
    // a5 .. h5
       -30, -40, -40, -50, -50, -40, -40, -30,
    // a6 .. h6
       -30, -40, -40, -50, -50, -40, -40, -30,
    // a7 .. h7
       -30, -40, -40, -50, -50, -40, -40, -30,
    // a8 .. h8
       -30, -40, -40, -50, -50, -40, -40, -30,
];

/// Endgame King PST, a1-first. Centralisation is rewarded symmetrically;
/// the table is intentionally independent from the middlegame shelter table.
#[rustfmt::skip]
const KING_EG_PST: [i32; 64] = [
    // a1 .. h1
       -50, -40, -30, -20, -20, -30, -40, -50,
    // a2 .. h2
       -30, -20, -10,   0,   0, -10, -20, -30,
    // a3 .. h3
       -30, -10,  20,  30,  30,  20, -10, -30,
    // a4 .. h4
       -30, -10,  30,  40,  40,  30, -10, -30,
    // a5 .. h5
       -30, -10,  30,  40,  40,  30, -10, -30,
    // a6 .. h6
       -30, -10,  20,  30,  30,  20, -10, -30,
    // a7 .. h7
       -30, -20, -10,   0,   0, -10, -20, -30,
    // a8 .. h8
       -50, -40, -30, -20, -20, -30, -40, -50,
];

const MAX_PHASE: i32 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Score {
    mg: i32,
    eg: i32,
}

impl Score {
    const fn same(value: i32) -> Score {
        Score {
            mg: value,
            eg: value,
        }
    }
}

/// Map a piece type to its piece-square table. An exhaustive `match` (not
/// `table[piece_type as usize]`) so the compiler forces every new piece
/// kind to be handled explicitly instead of silently indexing the wrong row
/// if the `PieceType` enum order ever changes.
fn pst_score(pt: PieceType, idx: usize) -> Score {
    match pt {
        PieceType::Pawn => Score::same(PAWN_PST[idx]),
        PieceType::Knight => Score::same(KNIGHT_PST[idx]),
        PieceType::Bishop => Score::same(BISHOP_PST[idx]),
        PieceType::Rook => Score::same(ROOK_PST[idx]),
        PieceType::Queen => Score::same(QUEEN_PST[idx]),
        PieceType::King => Score {
            mg: KING_MG_PST[idx],
            eg: KING_EG_PST[idx],
        },
    }
}

fn material_score(pt: PieceType) -> Score {
    Score::same(pt.value())
}

fn phase_weight(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn | PieceType::King => 0,
        PieceType::Knight | PieceType::Bishop => 1,
        PieceType::Rook => 2,
        PieceType::Queen => 4,
    }
}

/// Non-pawn material phase: 24 is the normal starting position and 0 is a
/// pure king/pawn ending. Promotions are clamped so the interpolation range
/// remains stable even when a position contains extra heavy pieces.
pub(crate) fn game_phase(pos: &Position) -> i32 {
    let phase: i32 = pos
        .board
        .iter()
        .filter_map(|piece| *piece)
        .map(|piece| phase_weight(piece.piece_type))
        .sum();
    phase.min(MAX_PHASE)
}

#[inline]
fn interpolate(mg: i32, eg: i32, phase: i32) -> i32 {
    ((mg as i64 * phase as i64 + eg as i64 * (MAX_PHASE - phase) as i64) / MAX_PHASE as i64) as i32
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
    let mut mg = 0;
    let mut eg = 0;
    for sq in 0..64usize {
        if let Some(p) = pos.board[sq] {
            let sign = if p.color == pos.side { 1 } else { -1 };
            let material = material_score(p.piece_type);
            let positional = pst_score(p.piece_type, pst_idx(sq, p.color));
            mg += sign * (material.mg + positional.mg);
            eg += sign * (material.eg + positional.eg);
        }
    }
    interpolate(mg, eg, game_phase(pos))
}

#[cfg(test)]
mod tests {
    use super::{game_phase, interpolate, KING_EG_PST, KING_MG_PST, MAX_PHASE};
    use crate::chess::fen::parse_fen;

    #[test]
    fn phase_counts_non_pawn_material_only() {
        let start = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let king_pawn = parse_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").unwrap();
        let queen_ending = parse_fen("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1").unwrap();

        assert_eq!(game_phase(&start), MAX_PHASE);
        assert_eq!(game_phase(&king_pawn), 0);
        assert_eq!(game_phase(&queen_ending), 4);
    }

    #[test]
    fn interpolation_has_locked_phase_extremes() {
        assert_eq!(interpolate(100, 200, MAX_PHASE), 100);
        assert_eq!(interpolate(100, 200, 0), 200);
        assert_eq!(interpolate(100, 200, MAX_PHASE / 2), 150);
    }

    #[test]
    fn king_tables_are_distinct() {
        assert_ne!(KING_MG_PST[4], KING_EG_PST[4]);
        assert_ne!(KING_MG_PST[28], KING_EG_PST[28]);
    }
}
