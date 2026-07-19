//! C1 — automatic insufficient-material draw (FIDE Articles 9.6 / 9.7).
//!
//! This module implements ONLY the automatic insufficient-material draw.
//! Claimable draws (fifty-move, threefold) and prospective-claim edges
//! are NOT part of C1 and must NOT be pre-buried here: there is no
//! `FiftyMoveClaim` / `ThreefoldClaim`, no `ZobristKey` / `GameState`
//! dependency, and no `claim_available_*` helper. Those land in C2 / C3.

use crate::chess::position::Position;
use crate::chess::types::*;

/// Why a position is (or is not) drawn. C1 covers only the automatic
/// insufficient-material draw, so the claimable variants are deliberately
/// absent — adding them here would silently pre-implement a later milestone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawReason {
    InsufficientMaterial,
}

/// Classify the current position. C1 takes only `&Position` — no game
/// history / Zobrist key — because automatic insufficient material needs
/// nothing beyond the board itself.
pub(crate) fn classify_draw(pos: &Position) -> Option<DrawReason> {
    is_insufficient_material(pos).then_some(DrawReason::InsufficientMaterial)
}

/// FIDE 9.6 / 9.7: a position is drawn by insufficient material when
/// no series of legal moves can possibly deliver checkmate. The locked
/// predicate:
///
///   1. any pawn / rook / queen present  -> NOT insufficient (mate possible);
///   2. exactly two kings and at most one minor -> insufficient
///      (K vs K, K+N vs K, K+B vs K);
///   3. exactly two kings, no knights, at least one bishop, and ALL bishops
///      occupy one square color -> insufficient (incl. arbitrary same-color
///      counts, e.g. K + three same-color bishops);
///   4. otherwise NOT insufficient (opposite-color bishops keep mating
///      potential; K+NN / K+BN / K+N vs K+N / any pawn-rook-queen
///      position do not).
pub(crate) fn is_insufficient_material(pos: &Position) -> bool {
    let board = pos.board();

    let mut kings = 0u32;
    let mut knights = 0u32;
    let mut bishops = 0u32;
    // Bitmask of distinct bishop square-colors seen: bit 0 = even
    // (file+rank)%2 == 0, bit 1 = odd. One bit set => all same color.
    let mut bishop_colors = 0u32;
    let mut pawn_rook_queen = false;

    for sq in 0u8..64u8 {
        if let Some(piece) = board[sq as usize] {
            match piece.piece_type {
                PieceType::King => kings += 1,
                PieceType::Knight => knights += 1,
                PieceType::Bishop => {
                    bishops += 1;
                    bishop_colors |= 1u32 << ((file_of(sq) + rank_of(sq)) % 2);
                }
                PieceType::Pawn | PieceType::Rook | PieceType::Queen => {
                    pawn_rook_queen = true;
                }
            }
        }
    }

    // Rule 1: any pawn / rook / queen means a mate is possible.
    if pawn_rook_queen {
        return false;
    }

    // Only kings (plus possibly minors) remain.
    if kings == 2 {
        // Rule 2: two kings and at most one minor piece.
        if knights + bishops <= 1 {
            return true;
        }
        // Rule 3: two kings, only bishops, all on a single square color.
        if knights == 0 && bishops > 0 && bishop_colors.count_ones() == 1 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;

    /// Assert `is_insufficient_material` and `classify_draw` agree, and that
    /// both report `expected` for `fen`.
    fn assert_insufficient(fen: &str, expected: bool) {
        let pos = parse_fen(fen).expect("valid FEN");
        assert_eq!(
            is_insufficient_material(&pos),
            expected,
            "is_insufficient_material({})",
            fen
        );
        assert_eq!(
            classify_draw(&pos).is_some(),
            expected,
            "classify_draw({})",
            fen
        );
        if expected {
            assert_eq!(
                classify_draw(&pos),
                Some(DrawReason::InsufficientMaterial),
                "classify_draw must be InsufficientMaterial when drawn"
            );
        }
    }

    // --- Rule 2: two kings and <= 1 minor ---

    #[test]
    fn k_vs_k_is_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/K6k w - - 0 1", true);
    }

    #[test]
    fn k_plus_knight_vs_k_is_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KN5k w - - 0 1", true);
    }

    #[test]
    fn k_plus_bishop_vs_k_is_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KB5k w - - 0 1", true);
    }

    // --- Rule 3: two kings, same-color bishops (any count) ---

    #[test]
    fn two_bishops_same_color_is_insufficient() {
        // White B b1 (light) + B d1 (light); both bishops share color.
        assert_insufficient("8/8/8/8/8/8/8/KB1B3k w - - 0 1", true);
    }

    #[test]
    fn three_bishops_same_color_is_insufficient() {
        // Spec fixture: K + three same-color bishops vs K.
        assert_insufficient("7k/8/8/8/8/8/8/KB1B1B2 w - - 0 1", true);
    }

    // --- Rule 4: NOT insufficient ---

    #[test]
    fn opposite_color_bishops_are_not_insufficient() {
        // White B b1 (odd color) + B c1 (even color): opposite colors.
        assert_insufficient("8/8/8/8/8/8/8/KBB4k w - - 0 1", false);
    }

    #[test]
    fn pawn_present_is_not_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KP5k w - - 0 1", false);
    }

    #[test]
    fn rook_present_is_not_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KR5k w - - 0 1", false);
    }

    #[test]
    fn queen_present_is_not_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KQ5k w - - 0 1", false);
    }

    #[test]
    fn k_plus_two_knights_is_not_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KNN4k w - - 0 1", false);
    }

    #[test]
    fn k_plus_bishop_and_knight_is_not_insufficient() {
        assert_insufficient("8/8/8/8/8/8/8/KBN4k w - - 0 1", false);
    }

    #[test]
    fn k_knight_vs_k_knight_is_not_insufficient() {
        // White K+N vs Black K+N (each side has a knight).
        assert_insufficient("8/8/8/8/8/8/8/KN3nk1 w - - 0 1", false);
    }
}
