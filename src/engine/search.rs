//! Search — the first "thinking" version (Phase 3), now interruptible.
//!
//! Order of escalation (per the roadmap, do NOT skip ahead):
//!   1. Negamax            (done)
//!   2. Alpha-Beta pruning (done)
//!   3. Iterative deepening (done, at the root)
//!   4. Principal variation  (TODO, Milestone 2)
//!   5. Quiescence search   (done, M2.1 — correctness-only)
//!   6. Move ordering       (done, M2.2 — basic MVV-LVA)
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
    let mut moves = generate_legal_moves(pos);
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

    // M2.2: try the most forcing moves first so alpha-beta cutoffs fire
    // earlier. Pure reorder — no move is dropped; for a full fixed-depth
    // search the minimax value is preserved (the visited node set and the
    // node count may still differ, because cutoffs land at different points).
    order_moves(pos, &mut moves);

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

/// Lexicographic move-ordering key for alpha-beta: higher key = searched
/// first. The tuple components are compared in order, so the key *is* a
/// strict MVV-LVA ranking.
///
/// Component 0 is the category and establishes the coarse rank: promotion
/// (2) sits above every plain capture (1), which sits above a quiet move
/// (0). This guarantees a king capture (attacker value 20000) still lands
/// in category 1, strictly above every quiet move (category 0). The old
/// `victim * 10 - attacker` key let a king capture score below 0 and be
/// sorted after quiet moves, which was wrong.
///
/// Component 1 is the victim value: within captures, the most valuable
/// victim is tried first. Component 2 is `-attacker`: within equal victims,
/// the cheapest attacker is tried first (a smaller attacker value yields a
/// larger -attacker and sorts earlier).
///
/// The key is used by every node that iterates a move list -- negamax, the
/// quiescence body, and the qply-cap evasion handler -- so forcing lines
/// (capturing the most valuable piece with the least valuable attacker,
/// promotions, and en passant) are tried first. This is a reordering only:
/// it never drops a move. Because alpha-beta cutoffs fire at different
/// points depending on move order, the visited node set and the node count
/// can change between orderings; what is preserved for a full, fixed-depth,
/// uninterrupted search is the legal move set and the final minimax value.
/// Under a node / time / deadline / stop limit even the returned depth and
/// best move may differ -- that is expected, not a bug.
///
/// Explicit victim values: en passant captures a pawn that is NOT on the
/// to-square (it sits one rank behind, on the same file), so a naive "is
/// the target occupied?" test would score it 0; we value it explicitly as
/// a pawn. Promotion is ranked above every plain capture (a freshly
/// promoted queen is worth more than the best single capture), with a
/// capturing promotion ranked above a quiet one.
fn move_order_key(pos: &Position, m: Move) -> (u8, i32, i32) {
    match m.flag {
        MoveFlag::Promotion(pt) => {
            let victim = pos.board[m.to as usize]
                .map(|p| p.piece_type.value())
                .unwrap_or(0);
            (2, pt.value(), victim)
        }
        MoveFlag::EnPassant => {
            // The captured pawn lives one rank behind the to-square.
            (1, PieceType::Pawn.value(), -PieceType::Pawn.value())
        }
        _ if pos.board[m.to as usize].is_some() => {
            let victim = pos.board[m.to as usize].unwrap().piece_type.value();
            let attacker = pos.board[m.from as usize].unwrap().piece_type.value();
            (1, victim, -attacker)
        }
        _ => (0, 0, 0),
    }
}

/// Reorder `moves` in place: highest [`move_order_key`] first.
///
/// The sort is stable on equal keys (ties keep their generation order,
/// broken by original index), which keeps the root fallback and tie-breaking
/// deterministic. It never adds or removes a move — `order_moves` is a pure
/// permutation of the input.
fn order_moves(pos: &Position, moves: &mut [Move]) {
    let mut indexed: Vec<((u8, i32, i32), usize, Move)> = moves
        .iter()
        .enumerate()
        .map(|(i, &m)| (move_order_key(pos, m), i, m))
        .collect();
    // Descending key, then ascending original index (stable, deterministic).
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for (i, (_, _, m)) in indexed.into_iter().enumerate() {
        moves[i] = m;
    }
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
///     stands pat with fail-hard bounds. An *in-check* node must NOT stand
///     pat, so it delegates to `search_final_evasion_ply`, which searches one
///     ply of the *current* position's evasions with no further recursion.
///     Caveat (documented, not a bug): an evasion that itself delivers a
///     *new* non-terminal check yields a child whose king is also in check;
///     that child is approximated by `-evaluate(child)` at the safety cap
///     rather than fully resolved. Strict resolution of the counter-check
///     chain needs repetition / 50-move detection and is deferred.
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
    let mut legal = generate_legal_moves(pos);
    if legal.is_empty() {
        // Terminal: checkmate (prefer the latest mate, smaller |score|) or
        // stalemate. Same convention as `negamax`.
        return Some(if in_check { -(MATE - ply as i32) } else { 0 });
    }

    // M2.2: order the legal list once. The in-check branch below
    // searches *all* evasions (reordered, still every move); the
    // not-in-check branch filters to tactical moves (reordered, since
    // `filter` preserves order); and the qply-cap handler receives this
    // already-reordered `legal`. Pure reorder — no move is dropped.
    order_moves(pos, &mut legal);

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
/// 50-move detection, which lands in a later milestone. What it guarantees
/// TODAY: the side to move (the node that is genuinely in check) is never
/// scored by a raw static eval of its own attacked position — its evasions
/// are always searched. What it does NOT (yet) guarantee: an evasion that
/// itself gives check produces a child whose king is also in check; that
/// child is approximated by `-evaluate(child)` at the safety cap. That is a
/// KNOWN, labelled approximation — not a "quiet-leaf stand-pat" (the child
/// is still reached through a real evasion, just not resolved further), and
/// not a correctness invariant.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;
    use crate::chess::move_to_uci;
    use crate::chess::movegen::generate_legal_moves;
    use std::collections::BTreeSet;

    /// White queen e4, black queen a4, black pawn h4 (all on rank 4,
    /// neither blocking the other), kings off the rank. White to move: two
    /// same-attacker captures of different victim value, plus quiet moves.
    const MVV_POS: &str = "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1";

    fn find_move(pos: &Position, uci: &str) -> Move {
        generate_legal_moves(&mut pos.clone())
            .into_iter()
            .find(|m| move_to_uci(*m) == uci)
            .unwrap_or_else(|| panic!("move {} not legal", uci))
    }

    /// Index of `uci` within the ordered legal-move list.
    fn ordered_index(pos: &Position, uci: &str) -> usize {
        let mut legal = generate_legal_moves(&mut pos.clone());
        order_moves(pos, &mut legal);
        legal
            .iter()
            .position(|m| move_to_uci(*m) == uci)
            .unwrap_or_else(|| panic!("move {} not present after ordering", uci))
    }

    #[test]
    fn order_key_mvv_lva_same_attacker() {
        let pos = parse_fen(MVV_POS).unwrap();
        let take_queen = find_move(&pos, "e4a4");
        let take_pawn = find_move(&pos, "e4h4");
        // Capturing the queen (victim 900) outranks capturing the pawn (100).
        assert!(
            move_order_key(&pos, take_queen) > move_order_key(&pos, take_pawn),
            "capturing the queen must outrank capturing the pawn (MVV)"
        );
        // Verify the victim component directly.
        assert_eq!(move_order_key(&pos, take_queen).1, 900);
        assert_eq!(move_order_key(&pos, take_pawn).1, 100);
    }

    #[test]
    fn order_key_en_passant_uses_pawn_victim() {
        // White pawn f5, black pawn g5 (just double-pushed), ep target g6.
        let pos = parse_fen("7k/8/8/5Pp1/8/8/8/4K3 w - g6 0 1").unwrap();
        let ep = find_move(&pos, "f5g6");
        assert!(matches!(ep.flag, MoveFlag::EnPassant));
        // The captured pawn is one rank behind the to-square: value it
        // explicitly as a pawn -> (1, 100, -100), never a quiet (0,0,0).
        assert_eq!(move_order_key(&pos, ep), (1, 100, -100));
        let push = find_move(&pos, "f5f6");
        assert_eq!(
            move_order_key(&pos, push),
            (0, 0, 0),
            "quiet push is category 0"
        );
        assert!(
            move_order_key(&pos, ep) > move_order_key(&pos, push),
            "en passant (capture) must outrank a quiet push"
        );
    }

    #[test]
    fn order_key_promotion_outranks_capture() {
        // White pawn e7: quiet promotion e8, and capturing-promotion e7xd8.
        let pos = parse_fen("3p3k/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let quiet_promo = find_move(&pos, "e7e8q");
        let cap_promo = find_move(&pos, "e7d8q");
        assert!(matches!(quiet_promo.flag, MoveFlag::Promotion(_)));
        assert!(matches!(cap_promo.flag, MoveFlag::Promotion(_)));
        // Both are category 2 (promotion), above every plain capture (cat 1).
        assert_eq!(move_order_key(&pos, quiet_promo).0, 2);
        assert_eq!(move_order_key(&pos, cap_promo).0, 2);
        // A capturing promotion outranks a quiet one.
        assert!(
            move_order_key(&pos, cap_promo) > move_order_key(&pos, quiet_promo),
            "capturing promotion must outrank quiet promotion"
        );
        // Even a quiet promotion outranks the best possible plain capture
        // (a queen captured by a pawn would key (1, 900, -100)).
        assert!(
            move_order_key(&pos, quiet_promo) > (1, 900, -100),
            "promotion must outrank every plain capture"
        );
    }

    /// STRICT MVV-LVA: when victims differ, the more valuable victim is
    /// tried first, regardless of who is attacking. Queen-takes-bishop
    /// (victim 330) must precede pawn-takes-knight (victim 320) even
    /// though the pawn attacker is cheaper — the old `victim*10 - attacker`
    /// key reversed this (2400 vs 3100).
    #[test]
    fn order_key_strict_mvv_by_victim() {
        // Black bishop a7, black knight c3; white queen a1, white pawn b2.
        let pos = parse_fen("6k1/b7/8/8/8/2n5/1P6/Q6K w - - 0 1").unwrap();
        let _qxb = find_move(&pos, "a1a7"); // queen takes bishop (330)
        let _pxn = find_move(&pos, "b2c3"); // pawn takes knight (320)
        assert!(
            ordered_index(&pos, "a1a7") < ordered_index(&pos, "b2c3"),
            "queen-takes-bishop (victim 330) must precede pawn-takes-knight (victim 320)"
        );
        assert_eq!(move_order_key(&pos, _qxb).1, 330);
        assert_eq!(move_order_key(&pos, _pxn).1, 320);
    }

    /// STRICT MVV-LVA: when victims are equal, the cheaper attacker is
    /// tried first. A pawn capturing the victim must precede a rook doing
    /// so (both key (1, 100, ...), but the pawn's -attacker is larger).
    #[test]
    fn order_key_same_victim_cheaper_attacker_first() {
        // Black pawn a3; white rook a1 and white pawn b2 can both take it.
        let pos = parse_fen("6k1/8/8/8/8/p7/1P6/R6K w - - 0 1").unwrap();
        let _rxa3 = find_move(&pos, "a1a3"); // rook takes pawn
        let _bxa3 = find_move(&pos, "b2a3"); // pawn takes pawn
        assert!(
            ordered_index(&pos, "b2a3") < ordered_index(&pos, "a1a3"),
            "pawn (cheaper attacker) must precede rook when the victim is equal"
        );
        assert_eq!(move_order_key(&pos, _rxa3).1, 100);
        assert_eq!(move_order_key(&pos, _bxa3).1, 100); // equal victims
    }

    /// A king capture is still a capture (category 1) and must sort before
    /// every quiet move (category 0). The old `victim*10 - attacker` key made
    /// a king capture (e.g. 900*10 - 20000 = -11000) score below 0 and
    /// sort AFTER quiet moves — fixed by the category component.
    #[test]
    fn order_key_king_capture_before_quiet() {
        // White king e5 can capture the black pawn d5; a quiet pawn push exists.
        let pos = parse_fen("6k1/8/8/3pK3/8/8/4P3/R7 w - - 0 1").unwrap();
        let _kxd5 = find_move(&pos, "e5d5"); // king takes pawn
        let _push = find_move(&pos, "e2e3"); // quiet pawn push
        assert!(
            ordered_index(&pos, "e5d5") < ordered_index(&pos, "e2e3"),
            "a king capture must precede a quiet move"
        );
        assert_eq!(
            move_order_key(&pos, _kxd5).0,
            1,
            "king capture is a capture"
        );
        assert_eq!(move_order_key(&pos, _push).0, 0, "quiet push is category 0");
    }

    #[test]
    fn order_moves_preserves_set_and_partitions_captures() {
        let pos = parse_fen(MVV_POS).unwrap();
        let mut legal = generate_legal_moves(&mut pos.clone());
        let count_before = legal.len();
        let before: BTreeSet<String> = legal.iter().map(|m| move_to_uci(*m)).collect();

        order_moves(&pos, &mut legal);

        // No move dropped or duplicated.
        assert_eq!(legal.len(), count_before, "ordering must not change count");
        let after: BTreeSet<String> = legal.iter().map(|m| move_to_uci(*m)).collect();
        assert_eq!(after, before, "ordering must not drop or add moves");

        // Every capture precedes every quiet move (captures-first partition).
        let mut seen_quiet = false;
        for &m in &legal {
            if is_tactical(&pos, m) {
                assert!(!seen_quiet, "a capture appeared after a quiet move");
            } else {
                seen_quiet = true;
            }
        }
    }
}
