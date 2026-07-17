//! Search — the first "thinking" version (Phase 3), now interruptible.
//!
//! Order of escalation (per the roadmap, do NOT skip ahead):
//!   1. Negamax            (done)
//!   2. Alpha-Beta pruning (done)
//!   3. Iterative deepening (done, at the root)
//!   4. Principal variation  (TODO, Milestone 2)
//!   5. Quiescence search   (TODO, Milestone 2)
//!   6. Move ordering       (TODO, Milestone 2)
//!   7. Transposition table (TODO, later)
//!
//! Milestone 1.1 adds the interruptibility plumbing that later milestones
//! (M1.2 threads, M1.3 time control) build on:
//!   - `SearchLimits` describes *what* to search (depth / nodes / time / infinite).
//!   - `SearchContext` carries the live, shared abort state (stop flag,
//!     node counter, hard deadline). Its fields are shared across the future
//!     search thread, hence the atomics.
//!   - The search can be aborted at any node; on abort it unwinds every
//!     made move so the root position is never left corrupted.
//!   - `search_best_move` keeps the last *fully completed* iteration's best
//!     move, so being stopped mid-deeper-search never loses a valid result.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::engine::eval::evaluate;

pub const MATE: i32 = 1_000_000;

/// What the caller wants the search to do. Fields other than `depth`
/// and `nodes` are data carriers for later milestones: `movetime` /
/// `infinite` are consumed by M1.3 time control, not here.
#[derive(Clone, Default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime: Option<Duration>,
    pub infinite: bool,
}

/// Live, *shared* state for one search run. `stop` and `nodes` are
/// atomic because M1.2 will run the search on its own thread while the
/// UCI main thread flips `stop` and reads `nodes`. We keep them atomic
/// now so the same struct works unchanged once threading lands.
pub struct SearchContext {
    pub stop: Arc<AtomicBool>,
    pub start: Instant,
    pub hard_deadline: Option<Instant>,
    pub nodes: AtomicU64,
}

impl SearchContext {
    pub fn new(stop: Arc<AtomicBool>) -> Self {
        SearchContext {
            stop,
            start: Instant::now(),
            hard_deadline: None,
            nodes: AtomicU64::new(0),
        }
    }
}

/// Return value of a search attempt. `Stopped` means the caller should
/// unwind and fall back to a previous result rather than treat the partial
/// score as a real answer.
pub enum SearchResult {
    Score(i32),
    Stopped,
}

/// Honour any externally-set abort condition. Returns true if the search
/// should stop *now* (before making another move).
fn should_abort(ctx: &SearchContext, limits: &SearchLimits) -> bool {
    if ctx.stop.load(Ordering::Relaxed) {
        return true;
    }
    if let Some(max_nodes) = limits.nodes {
        if ctx.nodes.load(Ordering::Relaxed) >= max_nodes {
            return true;
        }
    }
    if let Some(deadline) = ctx.hard_deadline {
        if Instant::now() >= deadline {
            return true;
        }
    }
    false
}

/// Negamax with alpha-beta. Returns `None` if the search was asked to
/// abort. A `None` is a directive to unwind *immediately*: the caller
/// must undo the move it made in THIS node and propagate `None` upward.
/// We never leave the position with a move applied when returning `None`.
pub fn negamax(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<i32> {
    // Account for this node and check the abort conditions BEFORE we touch
    // the board. If we bail out here we have made no move, so the board
    // is already clean for whoever called us.
    ctx.nodes.fetch_add(1, Ordering::Relaxed);
    if should_abort(ctx, limits) {
        return None;
    }

    // Terminal-node check MUST run before the depth==0 evaluation. A position
    // that is checkmate or stalemate is scored by its game-theoretic value,
    // never by the material count at the search horizon.
    let moves = generate_legal_moves(pos);
    if moves.is_empty() {
        if pos.is_in_check(pos.side) {
            // Checkmated: prefer the *latest* possible mate (smaller |score|),
            // so a mate delivered sooner is always preferred over a later one.
            return Some(-(MATE - ply as i32));
        }
        return Some(0); // stalemate
    }

    if depth == 0 {
        return Some(evaluate(pos));
    }

    let mut best = i32::MIN + 1000;
    for m in moves {
        let undo = pos.make_move(m);
        let child = negamax(pos, depth - 1, ply + 1, -beta, -alpha, ctx, limits);
        match child {
            Some(s) => {
                let score = -s;
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
            None => {
                // Abort: undo our move and unwind immediately.
                pos.unmake_move(undo);
                return None;
            }
        }
    }
    Some(best)
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

/// Search one root ply to `depth`. On success returns the best move
/// found alongside a `Score`. On abort (`Stopped`) all made moves
/// have been unmade and `pos` is left exactly as it was on entry;
/// the move is `None` in that case.
fn root_search(
    pos: &mut Position,
    depth: u32,
    root_moves: &[Move],
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> (Option<Move>, SearchResult) {
    let mut best_score = i32::MIN + 1000;
    let mut best_move: Option<Move> = None;
    let mut alpha = i32::MIN + 1000;
    let beta = i32::MAX - 1000;

    for &m in root_moves {
        let undo = pos.make_move(m);
        let child = negamax(pos, depth - 1, 1, -beta, -alpha, ctx, limits);
        match child {
            Some(s) => {
                let score = -s;
                pos.unmake_move(undo);
                if score > best_score {
                    best_score = score;
                    best_move = Some(m);
                }
                if best_score > alpha {
                    alpha = best_score;
                }
                // No beta cutoff at the root: we want real scores for every
                // root move so move ordering stays meaningful.
            }
            None => {
                pos.unmake_move(undo);
                return (None, SearchResult::Stopped);
            }
        }
    }

    match best_move {
        Some(mv) => (Some(mv), SearchResult::Score(best_score)),
        None => (None, SearchResult::Stopped),
    }
}

/// Iterative deepening from depth 1 up to the configured limit.
///
/// Returns the best move of the last *fully completed* iteration and its
/// score, or a legal fallback move if we were stopped before any iteration
/// finished. The root position is never left corrupted, no matter where the
/// abort lands.
pub fn search_best_move(
    pos: &mut Position,
    limits: &SearchLimits,
    ctx: &SearchContext,
) -> Option<(Move, i32)> {
    let max_depth = limits.depth.unwrap_or(4).max(1);
    let mut root_moves = generate_legal_moves(pos);
    if root_moves.is_empty() {
        return None; // already terminal (checkmate / stalemate)
    }
    // Stable fallback: the first legal move. Used if we never complete a
    // single iteration (e.g. stopped before depth 1 finishes).
    let fallback = root_moves[0];
    // Best result of the last fully completed iteration.
    let mut completed: Option<(Move, i32)> = None;

    let mut depth = 1u32;
    while depth <= max_depth {
        match root_search(pos, depth, &root_moves, ctx, limits) {
            (Some(mv), SearchResult::Score(sc)) => {
                completed = Some((mv, sc));
                // Move-ordering hook for the next iteration (cheap; real
                // ordering heuristics land in Milestone 2).
                if let Some(idx) = root_moves.iter().position(|m| *m == mv) {
                    root_moves.swap(0, idx);
                }
                println!(
                    "info depth {} score {} pv {}",
                    depth,
                    score_to_uci(sc),
                    move_to_uci(mv)
                );
                depth += 1;
            }
            (_, SearchResult::Stopped) => break,
            // `Score` is only ever produced together with `Some(mv)` in
            // `root_search`, so `(None, Score)` is unreachable; the
            // compiler still requires the arm to be listed.
            (None, SearchResult::Score(_)) => unreachable!(),
        }
        // Re-check at the loop level too, in case the abort tripped between
        // iterations or a node-limit was exactly met at an iteration boundary.
        if should_abort(ctx, limits) {
            break;
        }
    }

    completed.or(Some((fallback, 0)))
}
