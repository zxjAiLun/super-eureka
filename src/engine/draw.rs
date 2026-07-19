//! Automatic dead-position draw (FIDE Article 5.2.2).
//!
//! This module implements the automatic insufficient-material draw (C1), the
//! fifty-move claimable draw (C2, FIDE Article 9.3), and the
//! threefold-repetition claimable draw (C3, FIDE Article 9.2). Both C2 and C3
//! are claimable draws: a `0`-score OPTION for the side to move, not a forced
//! terminal. The 75-move automatic draw, fivefold automatic draw, twofold
//! cycle cutoff, and UCI ClaimDraw are all out of scope.

use crate::chess::game::GameState;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::chess::zobrist::ZobristKey;

/// Why a position is (or is not) drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawReason {
    /// Dead position: automatic, terminal, returns 0 immediately.
    InsufficientMaterial,
    /// FIDE fifty-move claim by the side to move: a 0-score OPTION, not a
    /// forced terminal.
    FiftyMoveClaim,
    /// FIDE threefold-repetition claim by the side to move: a 0-score OPTION,
    /// not a forced terminal. Detected against the full search-line Zobrist
    /// keys (`path.keys()`), not just the board.
    ThreefoldClaim,
}

/// Classify the current position against the full search-line Zobrist history.
/// Order is fixed: insufficient material (automatic) first, then the
/// claimable draws (fifty-move, then threefold). The classifier delegates the
/// claim branch to `claim_available_now` so that helper stays live in non-test
/// builds (it would otherwise be dead code).
pub(crate) fn classify_draw(pos: &Position, keys: &[ZobristKey]) -> Option<DrawReason> {
    if is_insufficient_material(pos) {
        return Some(DrawReason::InsufficientMaterial);
    }
    if claim_available_now(pos, keys) {
        if is_fifty_move_draw(pos) {
            return Some(DrawReason::FiftyMoveClaim);
        }
        return Some(DrawReason::ThreefoldClaim);
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

/// Threefold repetition (FIDE Article 9.2.2): the current position's Zobrist
/// key has appeared at least THREE times in the full search line `keys`.
///
/// Strictly `>= 3`:
///   `[key, key]`          -> false
///   `[key, key, key]`     -> true
/// This deliberately does NOT treat the second occurrence as a draw (that is
/// a twofold cycle cutoff, which is out of scope).
pub(crate) fn is_threefold_repetition(pos: &Position, keys: &[ZobristKey]) -> bool {
    let key = pos.zobrist_key();
    keys.iter().filter(|&&k| k == key).count() >= 3
}

/// A draw the side to move can claim RIGHT NOW: the position is itself a
/// claimable draw. Used at a NODE ENTRY to give the side to move a `0` floor.
/// Covers both claims:
///   - fifty-move claim (FIDE Article 9.3.2), and
///   - threefold-repetition claim (FIDE Article 9.2.2).
pub(crate) fn claim_available_now(pos: &Position, keys: &[ZobristKey]) -> bool {
    is_fifty_move_draw(pos) || is_threefold_repetition(pos, keys)
}

/// A candidate move, once made, will produce a position the MOVER can claim:
/// a player may declare an *intended* move and claim before executing it.
/// Covers two claim kinds:
///   - fifty-move intended claim (FIDE Article 9.3.1), and
///   - threefold-repetition intended claim (FIDE Article 9.2.1).
///
/// `child` is the position AFTER `make_move` + `push_child`, and `child_keys`
/// is the search line AFTER that push (so `child`'s key is its last element).
/// This claim belongs to the PARENT node's mover, NOT the child's side to
/// move.
pub(crate) fn claim_available_by_intended_move(
    child: &Position,
    child_keys: &[ZobristKey],
) -> bool {
    is_fifty_move_draw(child) || is_threefold_repetition(child, child_keys)
}

/// Convenience helper over a live `GameState`: is the current position a
/// threefold-repetition claim? Reuses `is_threefold_repetition` directly
/// (no copied filter/count) over the game's real key history.
pub fn is_threefold_in_game(gs: &GameState) -> bool {
    is_threefold_repetition(gs.position(), gs.key_history())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::{parse_fen, to_fen};
    use crate::chess::movegen::generate_legal_moves;
    use crate::chess::types::move_to_uci;
    use crate::chess::types::{Move, START_FEN};

    /// Local test helper: locate a legal move by UCI string in `pos`.
    fn find_move(pos: &Position, uci: &str) -> Move {
        generate_legal_moves(&mut pos.clone())
            .into_iter()
            .find(|m| move_to_uci(*m) == uci)
            .unwrap_or_else(|| panic!("no legal move {}", uci))
    }

    /// Local test helper: the piece-placement + side fields of a FEN (the
    /// board identity, ignoring move/halfmove counters which legitimately
    /// advance during a game).
    fn board_and_side(fen: &str) -> &str {
        let end = fen.find(' ').map_or(fen.len(), |i| i);
        let after_side = fen[end + 1..]
            .find(' ')
            .map_or(fen.len(), |j| end + 1 + j);
        &fen[..after_side]
    }

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
            classify_draw(&pos, &[pos.zobrist_key()]).is_some(),
            expected,
            "classify_draw({})",
            fen
        );
        if expected {
            assert_eq!(
                classify_draw(&pos, &[pos.zobrist_key()]),
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
            classify_draw(&insuff, &[insuff.zobrist_key()]),
            Some(DrawReason::InsufficientMaterial)
        );

        // Pure fifty-move (sufficient material): FiftyMoveClaim.
        let fifty = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 100 1").unwrap();
        assert_eq!(
            classify_draw(&fifty, &[fifty.zobrist_key()]),
            Some(DrawReason::FiftyMoveClaim)
        );

        // Not a draw.
        let normal = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1").unwrap();
        assert_eq!(classify_draw(&normal, &[normal.zobrist_key()]), None);
    }

    #[test]
    fn claim_available_now_and_intended() {
        let now = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 100 1").unwrap();
        assert!(claim_available_now(&now, &[now.zobrist_key()]));
        assert!(claim_available_by_intended_move(&now, &[now.zobrist_key()]));
        let not_yet = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1").unwrap();
        assert!(!claim_available_now(&not_yet, &[not_yet.zobrist_key()]));
        assert!(!claim_available_by_intended_move(
            &not_yet,
            &[not_yet.zobrist_key()]
        ));
    }

    // --- C3: strict threefold predicate ---

    #[test]
    fn threefold_strict_two_vs_three() {
        let pos = parse_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let key = pos.zobrist_key();
        assert!(
            !is_threefold_repetition(&pos, &[key, key]),
            "two occurrences are NOT a draw"
        );
        assert!(
            is_threefold_repetition(&pos, &[key, key, key]),
            "three occurrences ARE a draw"
        );
    }

    #[test]
    fn threefold_counts_only_current_key() {
        let pos = parse_fen("7k/8/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let key = pos.zobrist_key();
        let other = !key;
        assert!(
            !is_threefold_repetition(&pos, &[other, key, other, key]),
            "two occurrences (interleaved) are NOT a draw"
        );
        assert!(
            is_threefold_repetition(&pos, &[key, other, key, other, key]),
            "three occurrences (interleaved) ARE a draw"
        );
    }

    #[test]
    fn classify_threefold_reason_and_priority() {
        // A current position whose search line already contains it twice more.
        let pos = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 99 1").unwrap();
        let key = pos.zobrist_key();
        // Not yet a draw (only one occurrence, not fifty-move).
        assert_eq!(classify_draw(&pos, &[key]), None);

        // Current key appears a third time -> ThreefoldClaim.
        assert_eq!(
            classify_draw(&pos, &[key, key, key]),
            Some(DrawReason::ThreefoldClaim)
        );

        // Insufficient material still wins precedence.
        let insuff = parse_fen("8/8/8/8/8/8/8/K6k w - - 99 1").unwrap();
        let ik = insuff.zobrist_key();
        assert_eq!(
            classify_draw(&insuff, &[ik, ik, ik]),
            Some(DrawReason::InsufficientMaterial)
        );
    }

    // --- C3: GameState knight-shuffle threefold ---

    #[test]
    fn game_state_knight_shuffle_threefold() {
        use crate::chess::game::GameState;
        let moves = [
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
        ];
        let startpos_key = parse_fen(START_FEN).unwrap().zobrist_key();
        let mut gs = GameState::startpos();
        for m in moves {
            let mv = find_move(gs.position(), m);
            gs.push_known_legal_move(mv);
        }
        // We are back at the startpos; its key appears 3 times (ply 0, 4, 8).
        assert_eq!(
            board_and_side(&to_fen(gs.position())),
            board_and_side(START_FEN),
            "back at startpos board (piece placement + side)"
        );
        assert_eq!(gs.current_key(), startpos_key, "back at startpos key");
        assert_eq!(
            gs.key_history()
                .iter()
                .filter(|&&k| k == gs.current_key())
                .count(),
            3
        );
        assert!(is_threefold_in_game(&gs), "startpos key appears 3 times");
    }

    #[test]
    fn game_state_knight_shuffle_twofold_not_draw() {
        use crate::chess::game::GameState;
        let moves = ["g1f3", "g8f6", "f3g1", "f6g8"];
        let startpos_key = parse_fen(START_FEN).unwrap().zobrist_key();
        let mut gs = GameState::startpos();
        for m in moves {
            let mv = find_move(gs.position(), m);
            gs.push_known_legal_move(mv);
        }
        assert_eq!(
            board_and_side(&to_fen(gs.position())),
            board_and_side(START_FEN),
            "back at startpos board (piece placement + side)"
        );
        assert_eq!(gs.current_key(), startpos_key, "back at startpos key");
        // Back at startpos; its key appears only 2 times (ply 0 and 4).
        assert_eq!(
            gs.key_history()
                .iter()
                .filter(|&&k| k == gs.current_key())
                .count(),
            2
        );
        assert!(
            !is_threefold_in_game(&gs),
            "startpos key appears only 2 times: not a draw"
        );
    }
}
