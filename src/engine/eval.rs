//! Evaluation (Phase 4 will expand this).
//!
//! First version: pure material balance, returned from the side-to-move's
//! perspective. This is enough for the Phase-3 search to "think" and for
//! Perft-free sanity checks.
//!
//! Planned additions (one at a time, each validated by self-play):
//!   - piece-square tables
//!   - bishop pair bonus
//!   - mobility
//!   - pawn structure (doubled/isolated/passed)
//!   - king safety
//!   - endgame-phase weighting

use crate::chess::position::Position;

pub fn evaluate(pos: &Position) -> i32 {
    let mut score = 0;
    for sq in 0..64usize {
        if let Some(p) = pos.board[sq] {
            let v = p.piece_type.value();
            if p.color == pos.side {
                score += v;
            } else {
                score -= v;
            }
        }
    }
    score
}
