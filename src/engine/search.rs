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

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::engine::eval::evaluate;
use crate::engine::time::TimeBudget;

pub const MATE: i32 = 1_000_000;

/// Maximum quiescence ply. A "check → evasion → check → ..." sequence has no
/// natural depth bound (there is no repetition / 50-move handling in the
/// search yet), so this cap guarantees termination. It is a *safety* limit,
/// not a strength-tuning knob: at the cap we still detect checkmate /
/// stalemate first, then fall back to the static evaluation without
/// recursing further.
pub const MAX_QPLY: u32 = 32;

/// What the caller wants the search to do.
///
/// Time control is *not* here: `movetime` / clock fields are parsed into a
/// `TimeBudget` (soft/hard deadlines) carried on `SearchContext` instead, so
/// the search core never mixes "how much time" with "what to search".
/// "Infinite" (iterate until `stop` / an external deadline) is expressed by
/// the *absence* of a depth cap, a node cap, and a hard deadline — there is
/// deliberately no `infinite: bool` flag, so there is a single source of
/// truth for "keep deepening". The UCI layer encodes `go infinite` as
/// `SearchLimits { depth: None, nodes: None }` plus a `TimeBudget` whose
/// deadlines are both `None`.
#[derive(Clone, Default)]
pub struct SearchLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
}

/// Live, *shared* state for one search run. `stop` and `nodes` are
/// atomic because the search runs on its own thread (M1.2) while the UCI
/// main thread flips `stop` and reads `nodes`. The deadlines come from the
/// M1.3 time budget:
///   - `hard_deadline` is checked at every node entry (immediate unwind);
///   - `soft_deadline` is checked only between completed iterations (don't
///     start a deeper one). It is intentionally *not* read by `try_enter_node`.
pub struct SearchContext {
    pub stop: Arc<AtomicBool>,
    pub start: Instant,
    pub soft_deadline: Option<Instant>,
    pub hard_deadline: Option<Instant>,
    pub nodes: AtomicU64,
}

impl SearchContext {
    /// No time limits — used by tests and by depth/nodes/infinite searches
    /// that have no clock.
    pub fn new(stop: Arc<AtomicBool>) -> Self {
        SearchContext {
            stop,
            start: Instant::now(),
            soft_deadline: None,
            hard_deadline: None,
            nodes: AtomicU64::new(0),
        }
    }

    /// With a precomputed time budget (soft + hard deadlines).
    pub fn with_budget(stop: Arc<AtomicBool>, budget: TimeBudget) -> Self {
        SearchContext {
            stop,
            start: Instant::now(),
            soft_deadline: budget.soft_deadline,
            hard_deadline: budget.hard_deadline,
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

/// The outcome of a (possibly aborted) search run.
///
/// `score` is `None` when no full iteration completed — we were stopped
/// before even depth 1 finished, or the position had no legal move (in
/// which case `search_best_move` returns `None` upstream instead). A
/// fabricated `0` is deliberately avoided: a `0` would be misread by the
/// M1.3 `info ... score cp 0` line as "the engine thinks the position
/// is dead equal" when in fact no real evaluation exists yet.
///
/// `completed_depth` is `0` and `stopped` is `true` when we aborted
/// before the first iteration finished; otherwise `completed_depth` is the
/// depth of the last fully completed iteration.
pub struct SearchOutcome {
    pub best_move: Move,
    pub score: Option<i32>,
    pub completed_depth: u32,
    pub stopped: bool,
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

/// Atomically acquire the right to search *one* node, honouring the
/// external stop flag and the hard deadline. Returns `true` if the node may
/// be searched, `false` if the search must abort *before* touching the board.
///
/// This replaces the old "increment the counter, then check" sequence,
/// which under-counted by one (`nodes N` only ever processed N-1 nodes) and
/// also counted a node that was never actually searched (a preset stop
/// incremented the counter to 1 though zero nodes were evaluated). The
/// atomic `fetch_update` below makes the node quota exact: the counter is
/// only ever bumped when a node is genuinely about to be searched, and it
/// can never exceed the budget — even if several search workers later share
/// the same `SearchContext` (M1.2+).
fn try_enter_node(ctx: &SearchContext, limits: &SearchLimits) -> bool {
    if ctx.stop.load(Ordering::Relaxed) {
        return false;
    }
    if let Some(deadline) = ctx.hard_deadline {
        if Instant::now() >= deadline {
            return false;
        }
    }
    match limits.nodes {
        Some(limit) => ctx
            .nodes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                if current < limit {
                    Some(current + 1)
                } else {
                    None
                }
            })
            .is_ok(),
        None => {
            ctx.nodes.fetch_add(1, Ordering::Relaxed);
            true
        }
    }
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
    // Acquire the right to search this node *before* touching the board.
    // `try_enter_node` checks the external stop flag, the hard deadline, and
    // the node budget atomically, so bailing out here leaves the board
    // exactly as we found it (no move applied, nothing to unmake).
    if !try_enter_node(ctx, limits) {
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
        // Leaf: hand off to quiescence so pending captures / promotions are
        // resolved before we trust a static score (this is the cure for the
        // horizon effect). THIS node was already counted by `try_enter_node`
        // above, so we call the `_entered` variant, which does NOT re-enter
        // (and therefore does not re-count) the node — every position is
        // counted exactly once.
        return quiescence_entered(pos, ply, 0, alpha, beta, ctx, limits);
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

/// Is `m` a "tactical" move — one that quiescence must resolve?
///
/// Tactical = any capture (target square occupied), an en-passant capture
/// (the captured pawn is NOT on the target square, so "target occupied"
/// would miss it), or ANY promotion — including a *quiet* promotion like
/// `e7e8q` onto an empty square. Judging tacticalness by "target occupied"
/// alone would silently drop en passant and quiet promotions.
fn is_tactical(pos: &Position, m: Move) -> bool {
    matches!(m.flag, MoveFlag::EnPassant | MoveFlag::Promotion(_))
        || pos.board[m.to as usize].is_some()
}

/// Quiescence search that acquires (counts) a node first. This is the entry
/// point for the *recursive* calls made from within quiescence itself. The
/// depth-0 leaf in `negamax` instead calls [`quiescence_entered`] directly,
/// because that node has already been counted — keeping node accounting in
/// exactly one place per position.
pub fn quiescence(
    pos: &mut Position,
    ply: u32,
    qply: u32,
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<i32> {
    if !try_enter_node(ctx, limits) {
        return None;
    }
    quiescence_entered(pos, ply, qply, alpha, beta, ctx, limits)
}

/// The quiescence body, for a node the caller has ALREADY counted.
///
/// Correctness rules (M2.1 — pure quiescence, nothing more):
///  1. **In check ⇒ no stand-pat.** A static evaluation is meaningless while
///     the king is attacked, so we search *every* legal evasion — quiet king
///     moves, blocks and non-capturing interpositions included — never just
///     captures.
///  2. **Not in check ⇒ still detect stalemate.** We generate all legal
///     moves so an empty list is scored `0` (stalemate), instead of being
///     mistaken for "no captures, so stand-pat".
///  3. **Tactical set** = captures + en passant + promotions (`is_tactical`).
///  4. **`MAX_QPLY` cap** guarantees termination. At the cap we still
///     detect mate / stalemate first (handled above). A *non-check* node then
///     stands pat with fail-hard bounds; an *in-check* node must NOT stand
///     pat, so it delegates to `search_final_evasion_ply` (one ply of
///     evasions, no recursion) — never a raw static eval with the king
///     still attacked.
///  5. **Fail-hard alpha-beta**, matching `negamax`. On abort we return
///     `None`, having unmade any move so the board is left untouched.
fn quiescence_entered(
    pos: &mut Position,
    ply: u32,
    qply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<i32> {
    let in_check = pos.is_in_check(pos.side);

    // Generate all legal moves once. Correctness-first: this is what lets us
    // score checkmate / stalemate exactly. (A later optimisation may generate
    // only captures on the common not-in-check path.)
    let legal = generate_legal_moves(pos);
    if legal.is_empty() {
        // Terminal: checkmate (prefer the latest mate, smaller |score|) or
        // stalemate. Same convention as `negamax`.
        return Some(if in_check { -(MATE - ply as i32) } else { 0 });
    }

    // Termination cap. Mate / stalemate were handled above; now stop
    // recursing. Two cases:
    //   - NOT in check: stand pat, but honour fail-hard (a raw
    //     `evaluate(pos)` could fall below `alpha` or above `beta` and
    //     break the contract the rest of the tree relies on).
    //   - IN check: we MUST still search the evasions — a static eval
    //     with the king still attacked is meaningless, and returning it
    //     would re-introduce exactly the stand-pat-on-check bug this
    //     branch is meant to bound. `search_final_evasion_ply` searches
    //     exactly one ply of evasions with no further recursion, so a
    //     would-be-cyclic check chain terminates (no repetition detection
    //     exists yet, so we do not let it recurse).
    if qply >= MAX_QPLY {
        if !in_check {
            let stand_pat = evaluate(pos);
            if stand_pat >= beta {
                return Some(beta);
            }
            return Some(alpha.max(stand_pat));
        }
        return search_final_evasion_ply(pos, ply, alpha, beta, &legal, ctx, limits);
    }

    // Decide which moves to search.
    let tactical: Vec<Move> = if in_check {
        // Rule 1: under check, search ALL evasions, no stand-pat.
        legal
    } else {
        // Rule 2 (stalemate) already handled. Stand-pat is the lower bound:
        // the side to move is never forced to make a capture.
        let stand_pat = evaluate(pos);
        if stand_pat >= beta {
            return Some(beta);
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }
        legal.into_iter().filter(|m| is_tactical(pos, *m)).collect()
    };

    for m in tactical {
        let undo = pos.make_move(m);
        let child = quiescence(pos, ply + 1, qply + 1, -beta, -alpha, ctx, limits);
        match child {
            Some(s) => {
                let score = -s;
                pos.unmake_move(undo);
                if score >= beta {
                    return Some(beta); // fail-hard cutoff
                }
                if score > alpha {
                    alpha = score;
                }
            }
            None => {
                // Abort: undo our move and unwind immediately, leaving the
                // board exactly as we found it.
                pos.unmake_move(undo);
                return None;
            }
        }
    }
    Some(alpha)
}

/// Emergency cap handler for a position where the side to move is *in check*
/// at `MAX_QPLY`.
///
/// We must NOT stand pat — a static eval with the king still attacked is
/// meaningless, and returning it would re-introduce the stand-pat-on-check
/// bug. But we also must not recurse into a possibly-cyclic check chain
/// (there is no repetition / 50-move detection in the search yet), so we
/// search exactly one ply of evasions here with no further quiescence
/// recursion. That guarantees termination:
///   - every child move is made, scored, and unmade (board left intact);
///   - a terminal child (the opponent is checkmated / stalemated by our
///     evasion) is scored by its game-theoretic value;
///   - a non-terminal child is approximated by its static eval — the same
///     safe cap estimate we would otherwise have used, but now derived from
///     the *searched* evasion rather than the illegal resting position;
///   - stop / node-budget / hard-deadline are honoured at every child
///     entry (`try_enter_node`), and a None abort leaves the board untouched
///     because it is returned before any move is made.
///
/// This is a deliberate, incomplete stopgap: the full fix is repetition /
/// 50-move detection, which lands in a later milestone. Until then this at
/// least never evaluates a position with the king in check as if it were a
/// quiet leaf.
fn search_final_evasion_ply(
    pos: &mut Position,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    legal: &[Move],
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<i32> {
    for &m in legal {
        // Honour stop / hard-deadline / node-budget before touching the
        // board (same contract as the recursive `quiescence` entry). If we
        // cannot acquire a node we abort — but no move has been made yet, so
        // the board is already intact.
        if !try_enter_node(ctx, limits) {
            return None;
        }

        let undo = pos.make_move(m);

        // `legal` came from `generate_legal_moves`, so this evasion is legal:
        // the opponent is NOT attacking our king here. Score the child:
        //   - opponent has no move & is in check -> we delivered mate;
        //   - opponent has no move & not in check -> stalemate (0);
        //   - otherwise approximate with the static eval (safe cap estimate).
        let child_in_check = pos.is_in_check(pos.side);
        let child_legal = generate_legal_moves(pos);
        let score = if child_legal.is_empty() {
            if child_in_check {
                MATE - (ply as i32 + 1)
            } else {
                0
            }
        } else {
            -evaluate(pos)
        };

        pos.unmake_move(undo);

        if score >= beta {
            return Some(beta); // fail-hard cutoff
        }
        if score > alpha {
            alpha = score;
        }
    }
    Some(alpha)
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

/// Iterative deepening.
///
/// Termination semantics (M1.3 unifies them):
///   - `depth` set: stop once that depth completes (the *only* natural end).
///   - `nodes` set: stop when the node budget is exhausted (mid-iteration
///     abort; we keep the last fully completed iteration).
///   - time budget: `soft_deadline` is checked only *between* completed
///     iterations (don't start a deeper one); `hard_deadline` is checked at
///     every node entry (immediate unwind). `soft` is intentionally NOT
///     checked per-node, or soft/hard would be indistinguishable.
///   - `infinite` / no limit at all: keep deepening until `stop` or a
///     deadline. There is no longer a hidden depth-4 cap.
///
/// Returns the best move of the last *fully completed* iteration, or a
/// legal fallback if we were stopped before any iteration finished. The
/// root position is never left corrupted, no matter where the abort lands.
pub fn search_best_move(
    pos: &mut Position,
    limits: &SearchLimits,
    ctx: &SearchContext,
) -> Option<SearchOutcome> {
    let mut root_moves = generate_legal_moves(pos);
    if root_moves.is_empty() {
        return None; // already terminal (checkmate / stalemate)
    }
    // Stable fallback: the first legal move. Used if we never complete a
    // single iteration (e.g. stopped before depth 1 finishes).
    let fallback = root_moves[0];
    // Best result of the last fully completed iteration.
    let mut completed: Option<(Move, i32)> = None;
    let mut completed_depth: u32 = 0;
    let mut stopped = false;

    let mut depth = 1u32;
    loop {
        // A configured depth cap is the only *natural* end. With only nodes
        // or only time we keep deepening until the budget/deadline fires.
        if let Some(max_depth) = limits.depth {
            if depth > max_depth {
                break; // stopped stays false: we finished the requested depth
            }
        }

        match root_search(pos, depth, &root_moves, ctx, limits) {
            (Some(mv), SearchResult::Score(sc)) => {
                completed = Some((mv, sc));
                completed_depth = depth;
                // Move-ordering hook for the next iteration (cheap; real
                // ordering heuristics land in Milestone 2).
                if let Some(idx) = root_moves.iter().position(|m| *m == mv) {
                    root_moves.swap(0, idx);
                }
                // Standard UCI info: nodes from the atomic counter, time
                // from the search start, nps = nodes*1000/ms. nps is guarded
                // against time == 0 (no divide-by-zero) and computed in u128
                // to avoid overflow on huge node counts. Only completed
                // iterations emit info; an aborted depth 1 emits nothing.
                let nodes = ctx.nodes.load(Ordering::Relaxed);
                let elapsed_ms = ctx.start.elapsed().as_millis();
                let nps = if elapsed_ms > 0 {
                    let nps128 = nodes as u128 * 1000 / elapsed_ms;
                    // Saturate instead of truncating on a >u64::MAX result
                    // (theoretical only, but free to be correct).
                    if nps128 > u64::MAX as u128 {
                        u64::MAX
                    } else {
                        nps128 as u64
                    }
                } else {
                    0
                };
                println!(
                    "info depth {} score {} nodes {} time {} nps {} pv {}",
                    depth,
                    score_to_uci(sc),
                    nodes,
                    elapsed_ms,
                    nps,
                    move_to_uci(mv)
                );
                // The search runs on its own thread; flush after every
                // `info` so a GUI sees progress immediately.
                let _ = std::io::stdout().flush();

                // soft deadline: checked only between completed iterations.
                // If it has fired, keep this iteration's result and do NOT
                // start a deeper one — a partial deeper iteration could blow
                // the clock for no guaranteed gain.
                if let Some(sd) = ctx.soft_deadline {
                    if Instant::now() >= sd {
                        stopped = true;
                        break;
                    }
                }
                // hard deadline / external stop / node budget: stop now.
                if should_abort(ctx, limits) {
                    stopped = true;
                    break;
                }
                // saturating_add prevents a theoretical u32 overflow at
                // absurd depths (never reached in practice).
                depth = depth.saturating_add(1);
            }
            (_, SearchResult::Stopped) => {
                stopped = true;
                break;
            }
            // `Score` is only ever produced together with `Some(mv)` in
            // `root_search`, so `(None, Score)` is unreachable; the
            // compiler still requires the arm to be listed.
            (None, SearchResult::Score(_)) => unreachable!(),
        }
    }

    Some(SearchOutcome {
        best_move: completed.map(|(m, _)| m).unwrap_or(fallback),
        // No completed iteration => no real score; report `None` rather
        // than a fabricated 0 that M1.3 would misreport as "equal".
        score: completed.map(|(_, s)| s),
        completed_depth,
        stopped,
    })
}
