//! Search — the first "thinking" version (Phase 3).
//!
//! Order of escalation (per the roadmap, do NOT skip ahead):
//!   1. Negamax            (done here)
//!   2. Alpha-Beta pruning (done here)
//!   3. Iterative deepening (done here, at the root)
//!   4. Principal variation  (TODO)
//!   5. Quiescence search   (TODO, Phase 6)
//!   6. Move ordering       (TODO, Phase 6)
//!   7. Transposition table (TODO, Phase 5)
//!
//! This version is slow but intentionally simple and correct.

use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::engine::eval::evaluate;

pub const MATE: i32 = 1_000_000;

/// Negamax with alpha-beta. Returns the score from the side-to-move's view.
pub fn negamax(pos: &mut Position, depth: u32, ply: u32, mut alpha: i32, beta: i32) -> i32 {
    // Terminal-node check MUST run before the depth==0 evaluation. A position
    // that is checkmate or stalemate is scored by its game-theoretic value,
    // never by the material count at the search horizon. Otherwise the engine
    // would happily "evaluate" a position it is already mated in, or score a
    // forced stalemate as a normal material balance.
    let moves = generate_legal_moves(pos);
    if moves.is_empty() {
        if pos.is_in_check(pos.side) {
            // Being checkmated: prefer the *latest* possible mate (smaller |score|),
            // so a mate delivered sooner is always preferred over a later one.
            return -(MATE - ply as i32);
        }
        return 0; // stalemate
    }

    if depth == 0 {
        return evaluate(pos);
    }

    let mut best = i32::MIN + 1000;
    for m in moves {
        let undo = pos.make_move(m);
        let score = -negamax(pos, depth - 1, ply + 1, -beta, -alpha);
        pos.unmake_move(undo);
        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break; // beta cutoff
        }
    }
    best
}

fn score_to_uci(score: i32) -> String {
    if score > MATE - 1000 {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score < -(MATE - 1000) {
        format!("mate {}", -((MATE + score + 1) / 2))
    } else {
        format!("cp {}", score)
    }
}

/// Iterative deepening from depth 1 up to `max_depth`.
/// Returns the best move found and its score, and prints `info` lines.
pub fn search_best_move(pos: &mut Position, max_depth: u32) -> Option<(Move, i32)> {
    let max_depth = max_depth.max(1);
    if generate_legal_moves(pos).is_empty() {
        return None;
    }

    let mut best_move: Option<Move> = None;
    let mut best_score = 0;

    for depth in 1..=max_depth {
        let mut alpha = i32::MIN + 1000;
        let beta = i32::MAX - 1000;
        let moves = generate_legal_moves(pos);
        let mut local_best = moves[0];
        let mut local_score = i32::MIN + 1000;

        for m in moves {
            let undo = pos.make_move(m);
            let score = -negamax(pos, depth - 1, 1, -beta, -alpha);
            pos.unmake_move(undo);
            if score > local_score {
                local_score = score;
                local_best = m;
            }
            if local_score > alpha {
                alpha = local_score;
            }
        }

        best_move = Some(local_best);
        best_score = local_score;
        println!(
            "info depth {} score {} pv {}",
            depth,
            score_to_uci(best_score),
            move_to_uci(local_best)
        );
    }

    best_move.map(|m| (m, best_score))
}
