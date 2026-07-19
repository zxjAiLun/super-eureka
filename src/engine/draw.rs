//! Automatic dead-position draw (FIDE Article 5.2.2).
//!
//! This module implements the automatic insufficient-material draw (C1) and
//! the fifty-move claimable draw (C2). The fifty-move claim is a FIDE Article
//! 9.3 claimable draw: a `0`-score OPTION for the side to move, not a forced
//! terminal. The 75-move automatic draw is out of scope.
//!
//! Repetition / threefold (C3) is deliberately NOT yet present: there is no
//! `ThreefoldClaim`, no `ZobristKey` / `GameState` dependency, and no
//! `path.keys()`/history argument. Those land in C3.

use crate::chess::position::Position;
use crate::chess::types::*;

/// Why a position is (or is not) drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawReason {
    /// Dead position: automatic, terminal, returns 0 immediately.
    InsufficientMaterial,
    /// FIDE fifty-move claim by the side to move: a 0-score OPTION, not a
    /// forced terminal.
    FiftyMoveClaim,
}

/// Classify the current position. The classifier delegates the fifty-move
/// branch to `claim_available_now` so that helper stays live in non-test
/// builds (it would otherwise be dead code). C2 takes only `&Position` — no
/// game history / Zobrist key — because the fifty-move rule needs nothing
/// beyond the board's halfmove clock.
pub(crate) fn classify_draw(pos: &Position) -> Option<DrawReason> {
    if is_insufficient_material(pos) {
        return Some(DrawReason::InsufficientMaterial);
    }
    if claim_available_now(pos) {
        return Some(DrawReason::FiftyMoveClaim);
    }
    None
}

/// FIDE Article 5.2.2: a position is drawn by insufficient material when
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

/// Fifty-move rule: 100 half-moves (50 full moves) without a pawn move
/// or capture. (The 75-move mandatory draw is out of scope.)
pub(crate) fn is_fifty_move_draw(pos: &Position) -> bool {
    pos.halfmove_clock() >= 100
}

/// A draw the side to move can claim RIGHT NOW (FIDE 9.3.2): the position is
/// itself a claimable draw. Used at a NODE ENTRY to give the side to move a
/// `0` floor. C2 covers only the fifty-move claim.
pub(crate) fn claim_available_now(pos: &Position) -> bool {
    is_fifty_move_draw(pos)
}

/// A candidate move, once made, will produce a position the MOVER can claim
/// (FIDE 9.2.1 / 9.3.1: a player may declare an *intended* move and claim
/// before executing it). `child` is the position AFTER `make_move` +
/// `push_child`. This claim belongs to the PARENT node's mover, NOT the
/// child's side to move. C2 covers only the fifty-move claim.
pub(crate) fn claim_available_by_intended_move(child: &Position) -> bool {
    is_fifty_move_draw(child)
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

    // --- C1 cleanup: both-sides same-color bishop (one per side) ---

    #[test]
    fn both_sides_same_color_bishops_are_insufficient() {
        // White B b1 and Black B g8 both on dark squares (parity odd).
        assert_insufficient("6bk/8/8/8/8/8/8/KB6 w - - 0 1", true);
    }

    // --- C2: fifty-move draw ---

    fn assert_fifty(fen: &str, expected: bool) {
        let pos = parse_fen(fen).expect("valid FEN");
        assert_eq!(
            is_fifty_move_draw(&pos),
            expected,
            "is_fifty_move_draw({})",
            fen
        );
    }

    #[test]
    fn fifty_move_threshold() {
        assert_fifty("7k/8/8/8/8/8/8/K7 w - - 99 1", false);
        assert_fifty("7k/8/8/8/8/8/8/K7 w - - 100 1", true);
        assert_fifty("7k/8/8/8/8/8/8/K7 w - - 101 1", true);
    }

    #[test]
    fn classify_fifty_priority_and_reason() {
        // Insufficient material takes precedence over fifty-move.
        let insuff = parse_fen("8/8/8/8/8/8/8/K6k w - - 100 1").unwrap();
        assert_eq!(
            classify_draw(&insuff),
            Some(DrawReason::InsufficientMaterial)
        );

        // Pure fifty-move (sufficient material): FiftyMoveClaim.
        let fifty = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 100 1").unwrap();
        assert_eq!(classify_draw(&fifty), Some(DrawReason::FiftyMoveClaim));

        // Not a draw.
        let normal = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1").unwrap();
        assert_eq!(classify_draw(&normal), None);
    }

    #[test]
    fn claim_available_now_and_intended() {
        let now = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 100 1").unwrap();
        assert!(claim_available_now(&now));
        assert!(claim_available_by_intended_move(&now));
        let not_yet = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1").unwrap();
        assert!(!claim_available_now(&not_yet));
        assert!(!claim_available_by_intended_move(&not_yet));
    }
}
