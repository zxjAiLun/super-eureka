//! Search — the first "thinking" version (Phase 3), now interruptible.
//!
//! Order of escalation (per the roadmap, do NOT skip ahead):
//!   1. Negamax            (done)
//!   2. Alpha-Beta pruning (done)
//!   3. Iterative deepening (done, at the root)
//!   4. Principal variation  (done, M2.3)
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
    /// The principal variation of the last *fully completed* iteration,
    /// rooted at `best_move` (so `pv[0] == best_move`). Empty when no
    /// iteration completed (we were stopped before depth 1 finished) — a
    /// fabricated PV is deliberately avoided, matching `score`'s rationale.
    pub pv: Vec<Move>,
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

/// Triangular principal-variation storage: one row per global `ply`,
/// indexed by the search's global ply. Row `ply` holds the PV *through*
/// this node — the move made here followed by the child's PV. Rows are
/// grown only when needed (never shrunk), so a node writes only its own
/// row; a sibling's tail and the previous iteration's tail are never read
/// once a parent copies just the child row it actually returned.
#[derive(Default)]
struct PvTable {
    lines: Vec<Vec<Move>>,
}

impl PvTable {
    /// Grow to at least `rows` rows, but never shrink. A `resize` that
    /// received a smaller value would silently drop existing rows.
    fn ensure_rows(&mut self, rows: usize) {
        if self.lines.len() < rows {
            self.lines.resize_with(rows, Vec::new);
        }
    }

    /// Ensure a single `index` (and thus every row below it) exists.
    fn ensure_index(&mut self, index: usize) {
        let rows = index.checked_add(1).expect("PV table index overflow");
        self.ensure_rows(rows);
    }

    /// Clear the row for `ply`. Call after `try_enter_node` succeeds and
    /// before any terminal / stand-pat early return, so a node that never
    /// improves alpha never inherits a stale sibling tail.
    fn clear_at(&mut self, ply: u32) {
        let i = ply as usize;
        self.ensure_index(i);
        self.lines[i].clear();
    }

    /// Set `lines[ply]` to `mv` followed by `lines[ply+1]` (the child's
    /// PV), reusing the parent's capacity and without cloning the child row.
    fn set_from_child(&mut self, ply: u32, mv: Move) {
        let i = ply as usize;
        let child_ply = ply.checked_add(1).expect("PV ply overflow");
        let ci = child_ply as usize;
        self.ensure_index(ci);
        let (parents, children) = self.lines.split_at_mut(ci);
        let parent = &mut parents[i];
        let child = &children[0];
        parent.clear();
        parent.push(mv);
        parent.extend_from_slice(child);
    }

    /// Set `lines[ply]` to a single move (used by `search_final_evasion_ply`,
    /// which searches exactly one ply with no recursion, so there is no child
    /// PV to append). Writes only the current row — no child expansion.
    fn set_single(&mut self, ply: u32, mv: Move) {
        let i = ply as usize;
        self.ensure_index(i);
        self.lines[i].clear();
        self.lines[i].push(mv);
    }
}

/// A fully completed root iteration: its score and its principal variation
/// (rooted at the best move). Returned by [`root_search`]; `None` means
/// the iteration was aborted before completion.
struct RootIteration {
    score: i32,
    pv: Vec<Move>,
}

/// Negamax with alpha-beta. Returns `None` if the search was asked to
/// abort. A `None` is a directive to unwind *immediately*: the caller
/// must undo the move it made in THIS node and propagate `None` upward.
/// We never leave the position with a move applied when returning `None`.
pub fn negamax(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<i32> {
    // Public entry: PV lives only inside a real search, so a caller that
    // wants just the score gets a throwaway table that is discarded on
    // return. PV tracking never changes the score.
    let mut pv = PvTable::default();
    negamax_impl(pos, depth, ply, alpha, beta, ctx, limits, &mut pv)
}

/// Private search body. Identical to the public [`negamax`], but threads a
/// [`PvTable`] so the principal variation can be recorded. `clear_at(ply)`
/// runs right after the node is acquired and before any terminal / stand-pat
/// early return, so a node that never improves alpha never inherits a stale
/// sibling tail. When a child improves the score we record the move *before*
/// checking the beta cutoff, so the cut-off move is still captured.
///
/// The 8-argument shape mirrors the public 7-arg [`negamax`] entry plus the
/// live [`PvTable`] this milestone threads through the search — collapsing
/// them into a struct would only obscure the one-to-one mapping, so we keep
/// the explicit form and silence the arg-count lint deliberately.
#[allow(clippy::too_many_arguments)]
fn negamax_impl(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
) -> Option<i32> {
    // Acquire the right to search this node *before* touching the board.
    if !try_enter_node(ctx, limits) {
        return None;
    }
    // Clear our row now (after entry, before any early return).
    pv.clear_at(ply);

    // Terminal-node check MUST run before the depth==0 evaluation.
    let mut moves = generate_legal_moves(pos);
    if moves.is_empty() {
        if pos.is_in_check(pos.side) {
            return Some(-(MATE - ply as i32));
        }
        return Some(0); // stalemate -> empty PV (row already cleared)
    }

    if depth == 0 {
        // Leaf: hand off to quiescence. THIS node was already counted by
        // `try_enter_node` above, so we call the `_entered` variant, which
        // does NOT re-count it. The same PV table is passed down.
        return quiescence_entered_impl(pos, ply, 0, alpha, beta, ctx, limits, pv);
    }

    // M2.2: try the most forcing moves first so alpha-beta cutoffs fire
    // earlier. Pure reorder — no move is dropped; for a full fixed-depth
    // search the minimax value is preserved.
    order_moves(pos, &mut moves);

    let mut best = i32::MIN + 1000;
    for m in moves {
        let undo = pos.make_move(m);
        let child = negamax_impl(pos, depth - 1, ply + 1, -beta, -alpha, ctx, limits, pv);
        match child {
            Some(s) => {
                let score = -s;
                pos.unmake_move(undo);
                // Record the new best PV *before* the cutoff check, so the
                // cut-off move (which is the best we have) is captured.
                if score > best {
                    best = score;
                    pv.set_from_child(ply, m);
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
/// promoted queen is worth more than the best single capture). For the same
/// promoted piece, a capturing promotion ranks above a quiet promotion —
/// the key does NOT make an arbitrary capturing promotion outrank an
/// arbitrary quiet one (e.g. a quiet queen promotion `(2,900,0)` outranks
/// a knight-promotion capture `(2,320,900)`).
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
/// depth-0 leaf in `negamax` instead calls [`quiescence_entered_impl`] directly,
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
    // Public entry: throwaway PV table, discarded on return.
    let mut pv = PvTable::default();
    quiescence_impl(pos, ply, qply, alpha, beta, ctx, limits, &mut pv)
}

/// Recursive quiescence entry: acquires (counts) the node, then hands off to
/// the body ([`quiescence_entered_impl`]). This is the variant called by the
/// quiescence body for its own recursion — it carries the live [`PvTable`].
///
/// 8 args = the public 7-arg [`quiescence`] entry plus the live [`PvTable`];
/// kept explicit (see [`negamax_impl`] for the rationale).
#[allow(clippy::too_many_arguments)]
fn quiescence_impl(
    pos: &mut Position,
    ply: u32,
    qply: u32,
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
) -> Option<i32> {
    if !try_enter_node(ctx, limits) {
        return None;
    }
    quiescence_entered_impl(pos, ply, qply, alpha, beta, ctx, limits, pv)
}

/// The quiescence body, for a node the caller has ALREADY counted.
/// Threads a [`PvTable`] so the tactical principal variation is recorded.
///
/// 8 args = the public 7-arg [`quiescence`] entry plus the live [`PvTable`];
/// kept explicit (see [`negamax_impl`] for the rationale).
#[allow(clippy::too_many_arguments)]
/// `clear_at` runs first (the node is already entered by the caller), so a
/// terminal or stand-pat node leaves an empty row. A cut-off move is
/// recorded *before* returning the fail-hard beta, so the tactical PV is
/// never truncated.
///
/// Correctness rules (M2.1 — pure quiescence, nothing more): in check ⇒
/// no stand-pat (search every evasion); not in check ⇒ still detect
/// stalemate and stand-pat with fail-hard bounds; tactical set = captures +
/// en passant + promotions; the `MAX_QPLY` cap delegates an in-check node
/// to `search_final_evasion_ply` (one ply, no recursion); fail-hard
/// alpha-beta matching `negamax_impl`, returning `None` (board intact) on
/// abort.
fn quiescence_entered_impl(
    pos: &mut Position,
    ply: u32,
    qply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
) -> Option<i32> {
    // Node already entered by the caller: clear the row before any return.
    pv.clear_at(ply);

    let in_check = pos.is_in_check(pos.side);

    // Generate all legal moves once. Correctness-first: this is what lets us
    // score checkmate / stalemate exactly.
    let mut legal = generate_legal_moves(pos);
    if legal.is_empty() {
        return Some(if in_check { -(MATE - ply as i32) } else { 0 });
    }

    // M2.2: order the legal list once. Pure reorder — no move is dropped.
    order_moves(pos, &mut legal);

    // Termination cap.
    if qply >= MAX_QPLY {
        if !in_check {
            let stand_pat = evaluate(pos);
            if stand_pat >= beta {
                return Some(beta);
            }
            return Some(alpha.max(stand_pat));
        }
        return search_final_evasion_ply(pos, ply, alpha, beta, &legal, ctx, limits, pv);
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
        let child = quiescence_impl(pos, ply + 1, qply + 1, -beta, -alpha, ctx, limits, pv);
        match child {
            Some(s) => {
                let score = -s;
                pos.unmake_move(undo);
                // IMPORTANT: record the cut-off move BEFORE returning the
                // fail-hard beta, so the tactical PV captures it.
                if score >= beta {
                    pv.set_from_child(ply, m);
                    return Some(beta);
                }
                if score > alpha {
                    alpha = score;
                    pv.set_from_child(ply, m);
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
///
/// 8 args = the public entry shape (pos/ply/alpha/beta/ctx/limits) plus the
/// pre-generated `legal` slice and the live [`PvTable`]; kept explicit (see
/// [`negamax_impl`] for the rationale).
#[allow(clippy::too_many_arguments)]
fn search_final_evasion_ply(
    pos: &mut Position,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    legal: &[Move],
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
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

        // Record the cut-off move BEFORE returning the fail-hard beta.
        if score >= beta {
            pv.set_single(ply, m);
            return Some(beta);
        }
        if score > alpha {
            pv.set_single(ply, m);
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

/// Search one root ply to `depth`, returning the completed iteration (its
/// score and full principal variation) or `None` if aborted. The PV table is
/// (re)allocated per call, sized to this iteration's depth plus the quiescence
/// cap — never to `limits.depth`, so an absurd `go depth` cannot trigger a
/// huge one-shot allocation.
fn root_search(
    pos: &mut Position,
    depth: u32,
    root_moves: &[Move],
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<RootIteration> {
    let mut best_score = i32::MIN + 1000;
    let mut best_move: Option<Move> = None;
    let mut alpha = i32::MIN + 1000;
    let beta = i32::MAX - 1000;

    let mut pv = PvTable::default();
    // Capacity for this iteration: ply indices 0..=depth+MAX_QPLY. `+2`
    // keeps one spare row beyond the theoretical maximum (root at ply 0, the
    // deepest qsearch node at ply depth+MAX_QPLY).
    let rows = (depth as usize)
        .checked_add(MAX_QPLY as usize)
        .and_then(|n| n.checked_add(2))
        .expect("PV table size overflow");
    pv.ensure_rows(rows);

    for &m in root_moves {
        let undo = pos.make_move(m);
        let child = negamax_impl(pos, depth - 1, 1, -beta, -alpha, ctx, limits, &mut pv);
        match child {
            Some(s) => {
                let score = -s;
                pos.unmake_move(undo);
                if score > best_score {
                    best_score = score;
                    best_move = Some(m);
                    // Record the root PV: this move followed by the child's
                    // PV (which `negamax_impl` wrote into `pv.lines[1]`).
                    pv.set_from_child(0, m);
                }
                if best_score > alpha {
                    alpha = best_score;
                }
                // No beta cutoff at the root: real scores for every root move.
            }
            None => {
                pos.unmake_move(undo);
                return None; // aborted
            }
        }
    }

    best_move.map(|_| RootIteration {
        score: best_score,
        pv: std::mem::take(&mut pv.lines[0]),
    })
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
    let mut completed: Option<RootIteration> = None;
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
            Some(iter) => {
                let RootIteration { score, pv } = iter;
                completed_depth = depth;
                // Move-ordering hook for the next iteration (cheap; real
                // ordering heuristics land in Milestone 2).
                if let Some(idx) = root_moves.iter().position(|m| *m == pv[0]) {
                    root_moves.swap(0, idx);
                }
                // Standard UCI info: nodes from the atomic counter, time
                // from the search start, nps = nodes*1000/ms. nps is guarded
                // against time == 0 (no divide-by-zero) and computed in u128
                // to avoid overflow on huge node counts. Only completed
                // iterations emit info; an aborted depth 1 emits nothing.
                // The PV is the full principal variation of this iteration.
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
                let pv_str = pv
                    .iter()
                    .map(|m| move_to_uci(*m))
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "info depth {} score {} nodes {} time {} nps {} pv {}",
                    depth,
                    score_to_uci(score),
                    nodes,
                    elapsed_ms,
                    nps,
                    pv_str
                );
                // The search runs on its own thread; flush after every
                // `info` so a GUI sees progress immediately.
                let _ = std::io::stdout().flush();

                // Keep the completed iteration (pv moved in).
                completed = Some(RootIteration { score, pv });

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
            None => {
                stopped = true;
                break;
            }
        }
    }

    let best_move = completed.as_ref().map(|it| it.pv[0]).unwrap_or(fallback);
    // No completed iteration => no real score; report `None` rather
    // than a fabricated 0 that M1.3 would misreport as "equal".
    let score = completed.as_ref().map(|it| it.score);
    // PV mirrors `score`: empty when no iteration completed, so a
    // fallback move is never dressed up as a real principal variation.
    let pv = completed.map(|it| it.pv).unwrap_or_default();
    Some(SearchOutcome {
        best_move,
        score,
        completed_depth,
        stopped,
        pv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;
    use crate::chess::move_to_uci;
    use crate::chess::movegen::generate_legal_moves;
    use std::collections::BTreeSet;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

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

    /// M2.3 test 3 (private impl): a depth-0 search (leaf → quiescence)
    /// of a quiet-promotion position must produce a PV that contains the
    /// promotion `e7e8q`. The public `negamax` discards its PV, so we
    /// call the private `_impl` and inspect the table directly.
    #[test]
    fn negamax_impl_qsearch_pv_contains_promotion() {
        // White pawn e7, quiet promotion e7e8=Q; White Ka1, Black Kh8.
        let pos = parse_fen("7k/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let mut pv = PvTable::default();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
        )
        .expect("not stopped");

        // The promotion must appear somewhere in the recorded PV.
        let promo = find_move(&pos, "e7e8q");
        assert!(
            pv.lines[0].contains(&promo),
            "qsearch PV must contain the quiet promotion e7e8q (pv={:?})",
            pv.lines[0]
        );
        // Sanity: a promoted queen is worth far more than a pawn.
        assert!(score >= 800, "promotion should beat a pawn, got {}", score);
    }
}
