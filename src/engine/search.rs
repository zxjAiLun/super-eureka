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

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::chess::zobrist::{recompute_zobrist, ZobristKey};
use crate::engine::draw::{
    claim_available_by_intended_move, classify_draw, is_insufficient_material, DrawReason,
};
use crate::engine::eval::evaluate;
use crate::engine::time::TimeBudget;

pub const MATE: i32 = 1_000_000;

/// Maximum quiescence ply. A "check → evasion → check → ..." sequence has no
/// natural depth bound, so this cap guarantees termination. It is a *safety*
/// limit, not a repetition/fifty-move substitute: M3.1 draw handling
/// (fifty-move claim and threefold-repetition claim) is already implemented
/// and applied at every node, but an unresolved checking/tactical chain still
/// needs a hard ply cap to terminate. At the cap we still detect checkmate /
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
    /// rooted at `best_move`. Empty when no iteration completed (we were
    /// stopped before depth 1 finished) — a fabricated PV is deliberately
    /// avoided, matching `score`'s rationale. `best_move` is the source
    /// of truth for the root move and never depends on `pv` being non-empty.
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

/// Edge score a PARENT assigns to a child that has no legal moves.
/// `child_in_check == true`  -> child is checkmated -> parent delivered
///   mate -> POSITIVE `MATE - (parent_ply + 1)`.
/// `child_in_check == false` -> child is stalemated -> draw -> `0`.
/// (NOTE: this is the PARENT's perspective; never the child's negative score.)
fn terminal_child_score_for_parent(child_in_check: bool, parent_ply: u32) -> i32 {
    if child_in_check {
        MATE - (parent_ply as i32 + 1)
    } else {
        0
    }
}

/// Result of a single manual child probe. A `Continue` child has already
/// consumed its one node via the probe's `try_enter_node`, so it recurses
/// into the ENTERED body (never the counting entry, which would double-count).
#[derive(Debug, PartialEq)]
enum ChildProbe {
    Terminal(i32),
    IntendedClaim,
    Continue,
}

/// Shared manual-child probe used by the negamax, qsearch, and root move
/// edges. It performs the exact-once node accounting (the only `try_enter_node`
/// for this child), clears the child PV row, and classifies the child as
/// terminal / intended-claim / normal-continue — exactly per spec §5.6.1.
///
/// The caller MUST have already done `make_move` + `push_child` so `pos` is
/// the child and `child_keys` is the full search line ending in the child's
/// key. The intended-claim check (the PARENT mover's fifty-move or threefold
/// claim on this move) reads that history.
///
/// It does NOT make/push/pop/unmake the move: the calling edge owns the
/// `make_move` + `push_child` before the call and the `pop` + `unmake_move`
/// after. A `None` return means the node budget / stop / deadline was
/// exhausted and the caller must restore its own state and propagate `None`.
fn probe_child_draw(
    pos: &mut Position,
    child_keys: &[ZobristKey],
    child_ply: u32,
    parent_ply: u32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
) -> Option<ChildProbe> {
    // Exactly ONE node acquisition for this child.
    if !try_enter_node(ctx, limits) {
        return None;
    }
    // The probe is the sole owner of the child PV row's initial clear.
    pv.clear_at(child_ply);

    let child_legal = generate_legal_moves(pos);
    if child_legal.is_empty() {
        return Some(ChildProbe::Terminal(terminal_child_score_for_parent(
            pos.is_in_check(pos.side),
            parent_ply,
        )));
    }
    // Prospective (intended) fifty-move OR threefold claim belongs to the
    // PARENT mover, evaluated after the move on the extended search line.
    if claim_available_by_intended_move(pos, child_keys) {
        return Some(ChildProbe::IntendedClaim);
    }
    Some(ChildProbe::Continue)
}

/// Fixed, deterministic 64-bit mixing with domain separation (SplitMix64).
/// Maps a (key, count) pair to a u64 token for the repetition-signature XOR.
fn repetition_token(key: ZobristKey, count: usize) -> u64 {
    let count64 = u64::try_from(count).expect("usize fits u64 on supported 32/64-bit targets");

    let mut z = key ^ count64.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z
}

/// The search's own view of the Zobrist-key history during a search.
///
/// Seeded from the caller's real UCI `GameState` history (via
/// `search_best_move_with_history`), it is a *private* stack the search
/// extends with the key of every child position it descends into and
/// contracts on the way back up. It is deliberately separate from the
/// `GameState.key_history` it was cloned from: the search may push and
/// pop freely without ever touching the caller's history.
///
/// Invariant (M3.0, §16.8): for every `make_move` there is exactly
/// one `push_child` (called *after* the move, so the pushed key is the
/// *child's* key, never the parent's or a stale value) and one `pop`
/// after the matching `unmake_move`. Because the two are always paired,
/// the stack length returns to `root_len` at the end of the search —
/// whether it completed a depth, hit a node budget, a preset stop, a
/// qsearch abort, or the emergency-evasion cap. The caller's history is
/// therefore never mutated by the search.
///
/// M3.2 extension: `counts` tracks how many times each Zobrist key appears
/// in the path, and `repetition_signature` is a commutative XOR multiset
/// over all (key, count) pairs for count > 0.
#[derive(Clone)]
pub(crate) struct SearchPath {
    history: Vec<ZobristKey>,
    counts: HashMap<ZobristKey, usize>,
    repetition_signature: u64,
}

impl SearchPath {
    /// Build from a caller-supplied history (the `GameState` keys).
    /// Scans the input once to build counts and the XOR signature.
    pub(crate) fn new(history: Vec<ZobristKey>) -> Self {
        let mut counts = HashMap::new();
        let mut signature = 0u64;

        for &key in &history {
            let old = counts.get(&key).copied().unwrap_or(0);
            if old > 0 {
                signature ^= repetition_token(key, old);
            }
            let new = old + 1;
            counts.insert(key, new);
            signature ^= repetition_token(key, new);
        }

        SearchPath {
            history,
            counts,
            repetition_signature: signature,
        }
    }

    /// Current stack length (root length at search entry).
    pub(crate) fn len(&self) -> usize {
        self.history.len()
    }

    /// The full current stack (root first, current last). Read-only.
    /// Used by tests to assert post-search restoration; the non-test lib
    /// build has no caller, hence the allow.
    #[allow(dead_code)]
    pub(crate) fn keys(&self) -> &[ZobristKey] {
        &self.history
    }

    /// Last key on the stack (current node).
    #[allow(dead_code)]
    pub(crate) fn last(&self) -> Option<&ZobristKey> {
        self.history.last()
    }

    /// The commutative repetition signature for the full path.
    #[allow(dead_code)]
    pub(crate) fn repetition_signature(&self) -> u64 {
        self.repetition_signature
    }

    /// How many times `key` appears in the current path.
    #[allow(dead_code)]
    pub(crate) fn occurrences(&self, key: ZobristKey) -> usize {
        self.counts.get(&key).copied().unwrap_or(0)
    }

    /// Record the child key after a `make_move`. `child` is the position
    /// *after* the move, so its `zobrist_key` is the child's key.
    /// Updates the occurrence count and the XOR repetition signature.
    pub(crate) fn push_child(&mut self, child: &Position) {
        let key = child.zobrist_key();

        let old = self.counts.get(&key).copied().unwrap_or(0);
        if old > 0 {
            self.repetition_signature ^= repetition_token(key, old);
        }

        let new = old + 1;
        self.counts.insert(key, new);
        self.repetition_signature ^= repetition_token(key, new);

        self.history.push(key);
    }

    /// Undo a `push_child` (paired with the matching `unmake_move`).
    /// Restores the occurrence count and the XOR repetition signature.
    pub(crate) fn pop(&mut self) {
        let key = *self.history.last().expect("pop from empty SearchPath");

        let new = self.counts[&key];
        self.repetition_signature ^= repetition_token(key, new);

        if new == 1 {
            self.counts.remove(&key);
        } else {
            let old = new - 1;
            self.counts.insert(key, old);
            self.repetition_signature ^= repetition_token(key, old);
        }

        self.history.pop();
    }

    /// Defensive safety net: restore to the root length by popping
    /// individual entries. Each pop updates counts and the signature,
    /// so the path is fully consistent after restoration.
    pub(crate) fn restore_root(&mut self, root_len: usize) {
        debug_assert!(root_len >= 1);
        debug_assert!(root_len <= self.history.len());

        while self.history.len() > root_len {
            self.pop();
        }

        debug_assert_eq!(self.history.len(), root_len);
    }

    /// Rebuild the repetition signature from scratch by re-scanning
    /// history.  Used only in test helpers; production uses incremental
    /// updates.
    #[cfg(test)]
    fn rebuild_signature(&self) -> u64 {
        let mut counts: HashMap<ZobristKey, usize> = HashMap::new();
        let mut sig = 0u64;

        for &key in &self.history {
            let old = counts.get(&key).copied().unwrap_or(0);
            if old > 0 {
                sig ^= repetition_token(key, old);
            }
            let new = old + 1;
            counts.insert(key, new);
            sig ^= repetition_token(key, new);
        }

        sig
    }
}

/// A fully completed root iteration: its score and its principal variation
/// (rooted at the best move). Returned by [`root_search`]; `None` means
/// the iteration was aborted before completion.
struct RootIteration {
    score: i32,
    /// The best move of this completed iteration. Carried explicitly so the
    /// final `SearchOutcome.best_move` is derived from a real field rather
    /// than from `pv[0]` (which could be empty for a draw / non-PV
    /// outcome and would panic on a `.unwrap()`).
    best_move: Move,
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
    // Thin history view: a single-element root key. This caller has no
    // real game history, so the search still threads a `SearchPath`
    // (keeping the private impl one shape) but knows nothing before root.
    let mut path = SearchPath::new(vec![pos.zobrist_key()]);
    let root_len = path.len();
    let r = negamax_impl(
        pos, depth, ply, alpha, beta, ctx, limits, &mut pv, &mut path,
    );
    path.restore_root(root_len);
    r
}

/// Private search entry. Acquires (counts) exactly one node, then hands off
/// to the body ([`negamax_entered_impl`]). Every recursive child goes through
/// [`probe_child_draw`] (which itself calls `try_enter_node` once) and recurses
/// into `negamax_entered_impl`, so node accounting stays in exactly one place
/// per position — a child is never counted twice.
#[allow(clippy::too_many_arguments)]
fn negamax_impl(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    path: &mut SearchPath,
) -> Option<i32> {
    // Acquire the right to search this node *before* touching the board.
    if !try_enter_node(ctx, limits) {
        return None;
    }
    negamax_entered_impl(pos, depth, ply, alpha, beta, ctx, limits, pv, path)
}

/// The negamax body, for a node the caller has ALREADY counted. Threads a
/// [`PvTable`] so the principal variation is recorded. `clear_at(ply)` runs
/// right after the node is acquired and before any terminal / stand-pat early
/// return, so a node that never improves alpha never inherits a stale sibling
/// tail. When a child improves the score we record the move *before* checking
/// the beta cutoff, so the cut-off move is still captured.
///
/// The 8-argument shape mirrors the public 7-arg [`negamax`] entry plus the
/// live [`PvTable`] this milestone threads through the search — collapsing
/// them into a struct would only obscure the one-to-one mapping, so we keep
/// the explicit form and silence the arg-count lint deliberately.
#[allow(clippy::too_many_arguments)]
fn negamax_entered_impl(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    path: &mut SearchPath,
) -> Option<i32> {
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

    // Draw rules. Terminal (mate / stalemate) already returned above, so it
    // takes precedence. C2: the fifty-move claim is a 0-score FLOOR, not a
    // forced terminal — a node with a winning move still returns the win.
    let mut best = i32::MIN + 1000;
    if let Some(reason) = classify_draw(pos, path.keys()) {
        match reason {
            DrawReason::InsufficientMaterial => return Some(0), // automatic
            DrawReason::FiftyMoveClaim | DrawReason::ThreefoldClaim => {
                if 0 >= beta {
                    return Some(beta);
                }
                alpha = alpha.max(0);
                best = best.max(0);
                // fall through to the normal move loop below
            }
        }
    }

    if depth == 0 {
        // Leaf: hand off to quiescence. THIS node was already counted by the
        // entry above, so we call the `_entered` variant, which does NOT
        // re-count it. The same PV table is passed down.
        return quiescence_entered_impl(pos, ply, 0, alpha, beta, ctx, limits, pv, path);
    }

    // M2.2: try the most forcing moves first so alpha-beta cutoffs fire
    // earlier. Pure reorder — no move is dropped.
    order_moves(pos, &mut moves);

    for m in moves {
        let undo = pos.make_move(m);
        path.push_child(pos);

        // Manual child probe: try_enter_node called EXACTLY ONCE here.
        let probe = match probe_child_draw(pos, path.keys(), ply + 1, ply, ctx, limits, pv) {
            Some(p) => p,
            None => {
                path.pop();
                pos.unmake_move(undo);
                return None;
            }
        };

        let score = match probe {
            ChildProbe::Terminal(s) => s, // mate/stalemate edge, parent perspective
            ChildProbe::IntendedClaim => 0, // mover claims on this intended move
            ChildProbe::Continue => {
                // Manual probe already spent the single node. Recurse into the
                // ENTERED body — NEVER negamax_impl (that would double-count).
                // Handle a deeper abort EXPLICITLY: it must still pop + unmake
                // THIS edge before propagating None (no `?` before cleanup).
                match negamax_entered_impl(
                    pos,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    ctx,
                    limits,
                    pv,
                    path,
                ) {
                    Some(s) => -s,
                    None => {
                        path.pop();
                        pos.unmake_move(undo);
                        return None;
                    }
                }
            }
        };

        path.pop();
        pos.unmake_move(undo);

        // Record the new best PV *before* the cutoff check, so the cut-off
        // move (which is the best we have) is captured.
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
    // Thin history view: a single-element root key. This caller has
    // no real game history, so the search still threads a `SearchPath`
    // (keeping the private impl one shape) but knows nothing before root.
    let mut path = SearchPath::new(vec![pos.zobrist_key()]);
    let root_len = path.len();
    let r = quiescence_impl(pos, ply, qply, alpha, beta, ctx, limits, &mut pv, &mut path);
    path.restore_root(root_len);
    r
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
    path: &mut SearchPath,
) -> Option<i32> {
    if !try_enter_node(ctx, limits) {
        return None;
    }
    quiescence_entered_impl(pos, ply, qply, alpha, beta, ctx, limits, pv, path)
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
    path: &mut SearchPath,
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

    // Draw rules. Terminal (mate / stalemate) already returned above, so it
    // takes precedence. C2: the fifty-move claim is a 0-score FLOOR, not a
    // forced terminal — qsearch must still find a winning capture, so we do
    // NOT return 0 here; we only apply the floor to alpha and continue with
    // the stand-pat / capture / evasion loop below.
    if let Some(reason) = classify_draw(pos, path.keys()) {
        match reason {
            DrawReason::InsufficientMaterial => return Some(0), // automatic
            DrawReason::FiftyMoveClaim | DrawReason::ThreefoldClaim => {
                if 0 >= beta {
                    return Some(beta);
                }
                alpha = alpha.max(0);
                // fall through to the stand-pat + capture/evasion loop
            }
        }
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
        return search_final_evasion_ply(pos, ply, alpha, beta, &legal, ctx, limits, pv, path);
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
        path.push_child(pos);

        // Manual child probe: try_enter_node called EXACTLY ONCE here.
        let probe = match probe_child_draw(pos, path.keys(), ply + 1, ply, ctx, limits, pv) {
            Some(p) => p,
            None => {
                path.pop();
                pos.unmake_move(undo);
                return None;
            }
        };

        let score = match probe {
            ChildProbe::Terminal(s) => s, // mate/stalemate edge, parent perspective
            ChildProbe::IntendedClaim => 0, // mover claims on this intended move
            ChildProbe::Continue => {
                // Manual probe already spent the single node. Recurse into the
                // ENTERED qsearch variant — NEVER quiescence_impl (double count).
                // Handle a deeper abort EXPLICITLY before cleanup.
                match quiescence_entered_impl(
                    pos,
                    ply + 1,
                    qply + 1,
                    -beta,
                    -alpha,
                    ctx,
                    limits,
                    pv,
                    path,
                ) {
                    Some(s) => -s,
                    None => {
                        path.pop();
                        pos.unmake_move(undo);
                        return None;
                    }
                }
            }
        };

        path.pop();
        pos.unmake_move(undo);

        // IMPORTANT: record the cut-off move BEFORE returning the fail-hard
        // beta, so the tactical PV captures it.
        if score >= beta {
            pv.set_from_child(ply, m);
            return Some(beta);
        }
        if score > alpha {
            alpha = score;
            pv.set_from_child(ply, m);
        }
    }
    Some(alpha)
}

/// Emergency cap handler for a position where the side to move is *in check*
/// at `MAX_QPLY`.
///
/// We must NOT stand pat — a static eval with the king still attacked is
/// meaningless, and returning it would re-introduce the stand-pat-on-check
/// bug. But we also must not recurse into a possibly-cyclic check chain, so
/// we search exactly one ply of evasions here with no further quiescence
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
/// This is a deliberate, labelled safety cap: M3.1 draw handling (fifty-move
/// and threefold-repetition claims) is already implemented and applied at the
/// current in-check node, but an unresolved checking/tactical chain still
/// needs a hard ply cap to terminate. What it guarantees: the side to move
/// (the node that is genuinely in check) is never
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
    path: &mut SearchPath,
) -> Option<i32> {
    // Automatic insufficient-material draw for the in-check node at the qply
    // cap: a single ply cannot add material, so the position stays a draw; we
    // return 0 immediately. The CURRENT node's fifty-move or threefold claim
    // floor is the caller's responsibility (`quiescence_entered_impl`), so we
    // must NOT early-return on a claim here (that would suppress the evasions
    // the deeper search relies on).
    if is_insufficient_material(pos) {
        return Some(0);
    }

    for &m in legal {
        // Honour stop / hard-deadline / node-budget before touching the
        // board (same contract as the recursive `quiescence` entry). If we
        // cannot acquire a node we abort — but no move has been made yet, so
        // the board is already intact.
        if !try_enter_node(ctx, limits) {
            return None;
        }

        let undo = pos.make_move(m);
        path.push_child(pos);

        // `legal` came from `generate_legal_moves`, so this evasion is legal:
        // the opponent is NOT attacking our king here. Score the child:
        //   - terminal FIRST: opponent has no move & is in check -> mate; no
        //     move & not in check -> stalemate (0);
        //   - then draw: automatic insufficient material -> 0 (dead position);
        //     fifty-move or threefold intended claim -> 0 (mover secures draw);
        //   - otherwise approximate with the static eval (safe cap estimate).
        let child_in_check = pos.is_in_check(pos.side);
        let child_legal = generate_legal_moves(pos);
        let score = if child_legal.is_empty() {
            terminal_child_score_for_parent(child_in_check, ply)
        } else if is_insufficient_material(pos)
            || claim_available_by_intended_move(pos, path.keys())
        {
            // Draw: automatic dead position, or the mover's intended fifty-move
            // or threefold claim on this evasion — both secure 0 (no real win).
            0
        } else {
            -evaluate(pos)
        };

        path.pop();
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
#[allow(clippy::too_many_arguments)]
fn root_search(
    pos: &mut Position,
    depth: u32,
    root_moves: &[Move],
    root_claimable: bool,
    claim_fallback: Move,
    ctx: &SearchContext,
    limits: &SearchLimits,
    path: &mut SearchPath,
) -> Option<RootIteration> {
    // A claimable root (the side to move may claim right now) has a 0 floor:
    // the root value can never drop below 0, because the mover need not move.
    // We start from 0 so a losing/equal move cannot drag the root below the
    // claim; a move that scores > 0 truly beats the claim and is reported.
    // When not claimable, the root starts from the normal fail-soft floor.
    let mut best_score = if root_claimable { 0 } else { i32::MIN + 1000 };
    let mut best_move: Option<Move> = None;
    let mut alpha = if root_claimable { 0 } else { i32::MIN + 1000 };
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
        path.push_child(pos);

        // Manual child probe: try_enter_node called EXACTLY ONCE here.
        let probe = match probe_child_draw(pos, path.keys(), 1, 0, ctx, limits, &mut pv) {
            Some(p) => p,
            None => {
                path.pop();
                pos.unmake_move(undo);
                return None;
            }
        };

        // Terminal (mate / stalemate) FIRST — never overridden by a claim.
        let child = match probe {
            ChildProbe::Terminal(s) => Some(s),
            ChildProbe::IntendedClaim => Some(0), // mover claims on this intended move
            ChildProbe::Continue => {
                // Manual probe already spent the single node. Recurse into the
                // ENTERED body — NEVER negamax_impl (double count). Handle a
                // deeper abort EXPLICITLY before cleanup.
                match negamax_entered_impl(
                    pos,
                    depth - 1,
                    1,
                    -beta,
                    -alpha,
                    ctx,
                    limits,
                    &mut pv,
                    path,
                ) {
                    Some(s) => Some(-s),
                    None => {
                        path.pop();
                        pos.unmake_move(undo);
                        return None; // aborted (deeper recursion)
                    }
                }
            }
        };

        path.pop();
        let score = match child {
            Some(s) => s,
            None => unreachable!(),
        };
        pos.unmake_move(undo);

        if score > best_score {
            best_score = score;
            best_move = Some(m);
            // Record the root PV: this move followed by the child's PV.
            pv.set_from_child(0, m);
        }
        if best_score > alpha {
            alpha = best_score;
        }
        // No beta cutoff at the root: real scores for every root move.
    }

    // A claimable root with no real move beating 0 returns the claim itself
    // as a COMPLETED iteration: score 0, the stable fallback (protocol
    // placeholder, NOT a found 0-score line), empty PV. best_move stays None
    // so this branch fires instead of the `best_move.map` below.
    if root_claimable && best_move.is_none() {
        return Some(RootIteration {
            score: 0,
            best_move: claim_fallback,
            pv: Vec::new(),
        });
    }

    best_move.map(|bm| RootIteration {
        score: best_score,
        best_move: bm,
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
/// Public entry (unchanged signature). Builds a single-root history so
/// the search still threads a `SearchPath`, then delegates to the
/// history-aware implementation. Existing callers (and their tests)
/// keep compiling.
pub fn search_best_move(
    pos: &mut Position,
    limits: &SearchLimits,
    ctx: &SearchContext,
) -> Option<SearchOutcome> {
    let mut path = SearchPath::new(vec![pos.zobrist_key()]);
    let root_len = path.len();
    let r = search_best_move_impl(pos, limits, ctx, &mut path);
    path.restore_root(root_len);
    r
}

/// History-aware entry used by the UCI layer, which passes the real
/// `GameState` key history. The search extends this with its own
/// `SearchPath` (cloned from `game_history`) but never mutates the
/// caller's `GameState`.
///
/// Contract (debug-checked): `game_history` is non-empty and its last
/// element equals the current position's Zobrist key.
pub(crate) fn search_best_move_with_history(
    pos: &mut Position,
    game_history: &[ZobristKey],
    limits: &SearchLimits,
    ctx: &SearchContext,
) -> Option<SearchOutcome> {
    debug_assert!(!game_history.is_empty());
    debug_assert_eq!(game_history.last(), Some(&pos.zobrist_key()));
    // Derived-state invariant (spec hardening): the position's cached Zobrist
    // key must match a fresh recomputation. Checking only `history.last() ==
    // pos.zobrist_key()` is insufficient — both could be holding the same
    // stale key.
    debug_assert_eq!(pos.zobrist_key(), recompute_zobrist(pos));
    let mut path = SearchPath::new(game_history.to_vec());
    let root_len = path.len();
    let r = search_best_move_impl(pos, limits, ctx, &mut path);
    path.restore_root(root_len);
    r
}

/// Shared search body. Threads `path` through every recursion so the
/// search-line Zobrist keys are recorded (for M3.1 repetition /
/// M3.2 TT). The public `search_best_move` and the UCI-facing
/// `search_best_move_with_history` are thin wrappers around this.
fn search_best_move_impl(
    pos: &mut Position,
    limits: &SearchLimits,
    ctx: &SearchContext,
    path: &mut SearchPath,
) -> Option<SearchOutcome> {
    let mut root_moves = generate_legal_moves(pos);
    if root_moves.is_empty() {
        return None; // already terminal (checkmate / stalemate)
    }
    // Stable fallback: the first legal move. Used if we never complete a
    // single iteration (e.g. stopped before depth 1 finishes).
    let fallback = root_moves[0];

    // Root draw handling. The automatic insufficient-material draw is a
    // direct return (score 0, stable fallback, empty PV). The fifty-move and
    // threefold claims are 0-score OPTIONS: they do NOT early-return — we
    // still search for a winning move and only fall back to the claim if no
    // real line beats 0 (see the `root_claimable` branch after the loop).
    match classify_draw(pos, path.keys()) {
        Some(DrawReason::InsufficientMaterial) => {
            return Some(SearchOutcome {
                best_move: fallback,
                score: Some(0),
                completed_depth: 0,
                stopped: false,
                pv: Vec::new(),
            });
        }
        Some(DrawReason::FiftyMoveClaim) | Some(DrawReason::ThreefoldClaim) => {
            // Continue the depth loop below; the claim floor is honoured by
            // `negamax_entered_impl` / `quiescence_entered_impl`. We only note
            // that the root itself is a claim so a pre-depth-1 abort can still
            // report the claim instead of `None`.
        }
        None => {}
    }
    let root_claimable = matches!(
        classify_draw(pos, path.keys()),
        Some(DrawReason::FiftyMoveClaim) | Some(DrawReason::ThreefoldClaim)
    );

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

        match root_search(
            pos,
            depth,
            &root_moves,
            root_claimable,
            fallback,
            ctx,
            limits,
            path,
        ) {
            Some(iter) => {
                let RootIteration {
                    score,
                    best_move,
                    pv,
                } = iter;
                completed_depth = depth;
                // Move-ordering hook for the next iteration (cheap; real
                // ordering heuristics land in Milestone 2). Driven by the
                // explicit `best_move` field, never by `pv[0]`.
                if let Some(idx) = root_moves.iter().position(|m| *m == best_move) {
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

                // Keep the completed iteration (pv moved in); `best_move`
                // is carried explicitly so the final outcome never derives
                // the root move from `pv[0]`.
                completed = Some(RootIteration {
                    score,
                    best_move,
                    pv,
                });

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

    // Derive the root best move from the explicit `best_move` field of the
    // last completed iteration — never from `pv[0]`, which can be empty
    // (a draw / non-PV outcome) and would panic on a `.unwrap()`.
    let best_move = completed
        .as_ref()
        .map(|it| it.best_move)
        .unwrap_or(fallback);
    // No completed iteration => no real score. If the root itself is a
    // fifty-move OR threefold claim we still report the claim floor (0, stable
    // fallback, empty PV) rather than `None` — but ONLY when no real iteration
    // ran.
    if completed.is_none() && root_claimable {
        return Some(SearchOutcome {
            best_move: fallback,
            score: Some(0),
            completed_depth: 0,
            stopped: true,
            pv: Vec::new(),
        });
    }
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
    use crate::chess::fen::{parse_fen, to_fen};
    use crate::chess::game::GameState;
    use crate::chess::move_to_uci;
    use crate::chess::movegen::generate_legal_moves;
    use crate::chess::types::START_FEN;
    use crate::chess::zobrist::recompute_zobrist;
    use crate::engine::draw::is_threefold_repetition;
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
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
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
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);

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

    // ===== §16.8 SearchPath invariants =====
    //
    // The search threads a `SearchPath` (a clone of the caller's
    // real `GameState` history) and must restore its root length on
    // every exit: a completed depth, a node-budget abort, a preset
    // stop, a qsearch abort, and an emergency-evasion abort. The
    // caller's history must never be mutated, the root position must be
    // fully restored, and every pushed key must equal the *child*
    // position's key.

    /// `push_child` records the child's key (never the parent's or a
    /// stale value), and `pop` returns the stack to the root length.
    #[test]
    fn search_path_push_child_records_child_key() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        assert_eq!(path.len(), 1);
        let mv = find_move(&pos, "e2e4");
        let mut child = pos;
        child.make_move(mv);
        path.push_child(&child);
        assert_eq!(path.len(), 2);
        // Pushed key is the child's key, not the parent's.
        assert_eq!(path.keys()[1], child.zobrist_key());
        assert_ne!(path.keys()[1], pos.zobrist_key());
        // And it must match a fresh recomputation (incremental == recompute).
        assert_eq!(path.keys()[1], recompute_zobrist(&child));
        path.pop();
        assert_eq!(path.len(), 1);
        assert_eq!(path.keys(), &[pos.zobrist_key()][..]);
    }

    /// Helper: run the history-aware search and assert the `SearchPath`
    /// length is restored to the root length (== input history) and equals
    /// the input history exactly (no search-line residue).
    /// Drives the REAL private search with a path we own, so the assertions
    /// below inspect the actual stack the search push/pops — not a discarded
    /// local. Critically we do NOT call `restore_root()` before asserting:
    /// that would mask a push/pop imbalance and let the test pass spuriously.
    fn search_history_checked(
        pos: Position,
        history: Vec<ZobristKey>,
        limits: SearchLimits,
        ctx: &SearchContext,
    ) -> Option<SearchOutcome> {
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();

        let mut path = SearchPath::new(history.clone());
        let root_len = path.len();

        let out = search_best_move_impl(&mut p, &limits, ctx, &mut path);

        // Root length restored on every exit.
        assert_eq!(path.len(), root_len, "SearchPath root length not restored");
        // No search-line residue: path == input history.
        assert_eq!(
            path.keys(),
            history.as_slice(),
            "SearchPath must equal input history"
        );
        // The root Position itself is left exactly as found.
        assert_eq!(to_fen(&p), before_fen, "root Position FEN not restored");
        assert_eq!(
            p.zobrist_key(),
            before_key,
            "root Position Zobrist key not restored"
        );
        out
    }

    #[test]
    fn search_path_restores_root_on_completed_depth() {
        let pos = parse_fen(START_FEN).unwrap();
        let history = vec![pos.zobrist_key()];
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(2),
            ..Default::default()
        };
        let out = search_history_checked(pos, history, limits, &ctx);
        assert!(out.is_some(), "depth-2 search must complete");
        assert_eq!(
            to_fen(&pos),
            to_fen(&parse_fen(START_FEN).unwrap()),
            "root position must be fully restored"
        );
    }

    #[test]
    fn search_path_restores_root_on_node_budget_abort() {
        let pos = parse_fen(START_FEN).unwrap();
        let history = vec![pos.zobrist_key()];
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        // Tiny node budget: aborts mid-search, never completes a depth.
        let limits = SearchLimits {
            nodes: Some(3),
            ..Default::default()
        };
        let out = search_history_checked(pos, history, limits, &ctx);
        assert!(
            out.is_some() && out.unwrap().stopped,
            "node-budget abort returns a stopped result"
        );
    }

    #[test]
    fn search_path_restores_root_on_preset_stop() {
        let pos = parse_fen(START_FEN).unwrap();
        let history = vec![pos.zobrist_key()];
        // Preset stop: the search must abort immediately.
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(true)));
        let limits = SearchLimits::default();
        let out = search_history_checked(pos, history, limits, &ctx);
        assert!(
            out.is_some() && out.unwrap().stopped,
            "preset stop returns a stopped result"
        );
    }

    /// The caller's `GameState` history is never mutated by the search:
    /// UCI hands the thread a *clone* and `into_search_parts` moves the
    /// history out of that clone; the live `GameState` stays put.
    #[test]
    fn search_path_does_not_mutate_caller_history() {
        let mut gs = GameState::startpos();
        // Apply a couple of real moves so the history grows.
        let wm = gs
            .legal_moves()
            .into_iter()
            .find(|m| move_to_uci(*m) == "e2e4")
            .unwrap();
        gs.apply_legal_move(wm).unwrap();
        let bm = gs.legal_moves().into_iter().next().unwrap();
        gs.apply_legal_move(bm).unwrap();
        let original = gs.key_history().to_vec();
        assert!(original.len() >= 3, "history must have grown");

        // UCI-style hand-off: clone the GameState, split it, search the clone.
        let (mut pos, history) = gs.clone().into_search_parts();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(2),
            ..Default::default()
        };
        let out = search_best_move_with_history(&mut pos, &history, &limits, &ctx);
        assert!(out.is_some(), "depth-2 search must complete");

        // Live GameState untouched.
        assert_eq!(
            gs.key_history(),
            &original[..],
            "search must not mutate the caller's GameState history"
        );
    }

    // Direct white-box checks for the qsearch / emergency-evasion abort
    // branches: `try_enter_node` fails at the top (before any make_move),
    // so no push happens and the stack length stays at root.

    #[test]
    fn search_path_restores_root_on_qsearch_abort() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut p = pos;
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(true)));
        let limits = SearchLimits::default();
        let out = quiescence_impl(
            &mut p,
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        assert!(out.is_none(), "preset stop must abort quiescence");
        path.restore_root(root_len);
        assert_eq!(
            path.len(),
            root_len,
            "root length must restore on qsearch abort"
        );
    }

    #[test]
    fn search_path_restores_root_on_emergency_evasion_abort() {
        // White king e1 in check from a black rook on e8; white to move.
        let pos = parse_fen("4r1k1/8/8/8/8/R7/8/4K3 w - - 0 1").unwrap();
        let mut p = pos;
        let legal = generate_legal_moves(&mut p);
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(true)));
        let limits = SearchLimits::default();
        let out = search_final_evasion_ply(
            &mut p,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &legal,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        assert!(out.is_none(), "preset stop must abort emergency evasion");
        path.restore_root(root_len);
        assert_eq!(
            path.len(),
            root_len,
            "root length must restore on emergency-evasion abort"
        );
    }

    #[test]
    fn search_path_emergency_evasion_completes_and_restores() {
        // White king e1 in check from a black rook on e8; white to move.
        // With no stop it completes (one ply of evasions) and restores.
        let fen = "4r1k1/8/8/8/8/R7/8/4K3 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let mut p = pos;
        let legal = generate_legal_moves(&mut p);
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let out = search_final_evasion_ply(
            &mut p,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &legal,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        assert!(
            out.is_some(),
            "emergancy evasion must complete when not stopped"
        );
        path.restore_root(root_len);
        assert_eq!(
            path.len(),
            root_len,
            "root length must restore after emergency evasion completes"
        );
        assert_eq!(
            to_fen(&p),
            to_fen(&parse_fen(fen).unwrap()),
            "root position must be fully restored"
        );
    }

    /// Real mid-search abort for quiescence: the root qsearch node is entered
    /// (consuming the only budgeted node), at least one tactical capture is
    /// made + pushed, then the *child* qsearch recursion is denied a node and
    /// aborts. This verifies that after a genuine push, the lower layer's
    /// abort still pops the stack and restores the root Position.
    #[test]
    fn search_path_restores_root_after_qsearch_mid_abort() {
        // White queen e4 can capture black queen a4 (e4a4): a real tactical
        // move, so a make + push definitely happens before the abort.
        let fen = "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();

        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();

        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        // Exactly one node: the root qsearch enters, the first child recursion
        // is denied, forcing an abort *after* a push.
        let limits = SearchLimits {
            nodes: Some(1),
            ..Default::default()
        };
        let out = quiescence_impl(
            &mut p,
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        assert!(
            out.is_none(),
            "qsearch must abort when no child node is available"
        );
        assert_eq!(
            path.len(),
            root_len,
            "SearchPath root length not restored on qsearch mid-abort"
        );
        assert_eq!(
            path.keys(),
            &[before_key],
            "SearchPath must equal root key after qsearch mid-abort"
        );
        assert_eq!(
            to_fen(&p),
            before_fen,
            "root Position FEN not restored on qsearch mid-abort"
        );
        assert_eq!(
            p.zobrist_key(),
            before_key,
            "root Position Zobrist key not restored on qsearch mid-abort"
        );
    }

    /// Real mid-abort for emergency evasion: the first evasion is fully made,
    /// pushed, scored, popped and unmade (consuming the only budgeted node),
    /// then the *second* evasion's `try_enter_node` is denied and the search
    /// aborts. Verifies that a genuine push + pop pair still balances when the
    /// budget runs out on a later sibling.
    #[test]
    fn search_path_restores_root_after_emergency_evasion_mid_abort() {
        // White king e1 in check from a black rook on e8; several evasions
        // (king moves + Ra3-e3 block) so the second sibling is reachable.
        let fen = "4r1k1/8/8/8/8/R7/8/4K3 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();

        let legal = generate_legal_moves(&mut p);
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();

        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        // Exactly one node: the first evasion enters and completes its make /
        // push / score / pop / unmake; the second sibling is denied.
        let limits = SearchLimits {
            nodes: Some(1),
            ..Default::default()
        };
        let out = search_final_evasion_ply(
            &mut p,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &legal,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        assert!(
            out.is_none(),
            "emergency evasion must abort when the second sibling is denied a node"
        );
        assert_eq!(
            path.len(),
            root_len,
            "SearchPath root length not restored on emergency-evasion mid-abort"
        );
        assert_eq!(
            path.keys(),
            &[before_key],
            "SearchPath must equal root key after emergency-evasion mid-abort"
        );
        assert_eq!(
            to_fen(&p),
            before_fen,
            "root Position FEN not restored on emergency-evasion mid-abort"
        );
        assert_eq!(
            p.zobrist_key(),
            before_key,
            "root Position Zobrist key not restored on emergency-evasion mid-abort"
        );
    }

    // ===== C1: automatic insufficient-material draw =====

    /// A K vs K position, searched directly, must score 0 (draw) — the
    /// automatic insufficient-material check fires before any depth search.
    #[test]
    fn negamax_impl_k_vs_k_is_zero() {
        let pos = parse_fen("8/8/8/8/8/8/8/K6k w - - 0 1").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
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
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(score, 0, "K vs K is drawn by insufficient material");
    }

    /// Quiescence of a K vs K position must also return 0 — the same
    /// automatic check runs inside `quiescence_entered_impl`.
    #[test]
    fn quiescence_k_vs_k_is_zero() {
        let pos = parse_fen("8/8/8/8/8/8/8/K6k w - - 0 1").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = quiescence_impl(
            &mut pos.clone(),
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(score, 0, "qsearch K vs K is drawn by insufficient material");
    }

    /// Terminal (checkmate) must be detected and scored as a mate, never as
    /// a 0 draw. A checkmated position cannot also be FIDE-insufficient
    /// (a forced mate implies sufficient material), so the priority is shown
    /// by the mate score being non-zero here while the insufficient tests
    /// above return exactly 0. Code ordering: the terminal check precedes
    /// the `is_insufficient_material` short-circuit in every search node.
    #[test]
    fn negamax_terminal_checkmate_priority() {
        // Black Kh8, White Kf7, White Rh1: Black is checkmated.
        let pos = parse_fen("7k/5K2/8/8/8/8/8/7R b - - 0 1").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
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
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(
            score,
            -(MATE),
            "checkmate is scored as a mate, not a 0 draw"
        );
    }

    /// Root insufficient material: legal moves exist, so the search must NOT
    /// return None; it returns a draw outcome with a stable legal best move,
    /// score 0, empty PV, completed_depth 0, and stopped == false.
    #[test]
    fn root_insufficient_material_returns_draw() {
        let pos = parse_fen("8/8/8/8/8/8/8/K6k w - - 0 1").unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_best_move(&mut pos.clone(), &limits, &ctx).expect("outcome");

        assert_eq!(out.score, Some(0), "insufficient material scores 0");
        assert_eq!(out.completed_depth, 0, "no iteration is searched");
        assert!(!out.stopped, "not stopped");
        assert!(out.pv.is_empty(), "empty PV for a drawn root");

        // best_move must be a legal root move.
        let legal: BTreeSet<String> = generate_legal_moves(&mut pos.clone())
            .iter()
            .map(|m| move_to_uci(*m))
            .collect();
        assert!(
            legal.contains(&move_to_uci(out.best_move)),
            "best_move must be a legal root move"
        );
    }

    /// Root with no legal move (checkmate) still returns None — the
    /// terminal check precedes the insufficient-material short-circuit.
    #[test]
    fn root_no_legal_move_still_none() {
        // Black Kh8, White Kf7, White Rh1: Black is checkmated (no moves).
        let pos = parse_fen("7k/5K2/8/8/8/8/8/7R b - - 0 1").unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_best_move(&mut pos.clone(), &limits, &ctx);
        assert!(
            out.is_none(),
            "a root with no legal move returns None (terminal precedence)"
        );
    }

    /// SearchPath / Position restoration holds for an insufficient-material
    /// root (which short-circuits before any iteration).
    #[test]
    fn search_path_restores_root_on_insufficient_material() {
        let pos = parse_fen("8/8/8/8/8/8/8/K6k w - - 0 1").unwrap();
        let history = vec![pos.zobrist_key()];
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_history_checked(pos, history, limits, &ctx);
        let out = out.expect("insufficient-material root returns an outcome");
        assert_eq!(out.score, Some(0));
        assert!(out.pv.is_empty());
    }

    // ===== C1/C2/C3: draw rules (insufficient material / fifty-move / threefold) =====

    /// Make `m` on `pos`, push the child, run the manual probe, then restore
    /// the board and return the probe result (so each test controls cleanup).
    #[allow(clippy::too_many_arguments)]
    fn probe_move(
        pos: &mut Position,
        m: Move,
        child_ply: u32,
        parent_ply: u32,
        ctx: &SearchContext,
        limits: &SearchLimits,
        pv: &mut PvTable,
        path: &mut SearchPath,
    ) -> Option<ChildProbe> {
        let undo = pos.make_move(m);
        path.push_child(pos);
        let r = probe_child_draw(pos, path.keys(), child_ply, parent_ply, ctx, limits, pv);
        path.pop();
        pos.unmake_move(undo);
        r
    }

    /// Drive exactly ONE root move `m` through the same edge-scoring path as
    /// `negamax_entered_impl`, returning the (parent-perspective) edge score.
    /// Used to assert the score of a specific intended-claim edge in isolation
    /// (the full negamax returns the best of ALL legal moves).
    #[allow(clippy::too_many_arguments)]
    fn score_one_move_edge(
        pos: &mut Position,
        m: Move,
        history: Vec<ZobristKey>,
        depth: u32,
    ) -> i32 {
        let before_fen = to_fen(pos);
        let before_key = pos.zobrist_key();
        let root_keys = history.clone();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(history);
        let undo = pos.make_move(m);
        path.push_child(pos);
        let probe = probe_child_draw(pos, path.keys(), 1, 0, &ctx, &limits, &mut pv);
        let score = match probe {
            Some(ChildProbe::Terminal(s)) => s,
            Some(ChildProbe::IntendedClaim) => 0, // mover claims on this move
            Some(ChildProbe::Continue) => {
                let s = negamax_entered_impl(
                    pos,
                    depth,
                    1,
                    i32::MIN + 1000,
                    i32::MAX - 1000,
                    &ctx,
                    &limits,
                    &mut pv,
                    &mut path,
                )
                .expect("not stopped");
                -s
            }
            None => unreachable!("test probe cannot abort (unbounded limits)"),
        };
        path.pop();
        pos.unmake_move(undo);
        // Self-verify the edge fully restored the board + path.
        assert_eq!(to_fen(pos), before_fen, "edge restored FEN");
        assert_eq!(pos.zobrist_key(), before_key, "edge restored key");
        assert_eq!(path.keys(), &root_keys[..], "edge restored path");
        score
    }

    /// §N.9: a single quiet evasion that pushes the halfmove clock 99→100 is
    /// an intended fifty-move claim, scored exactly 0 by the mover. The probe
    /// returns `IntendedClaim` for that edge.
    #[test]
    fn intended_fifty_move_claim_probe_is_intended() {
        // White Ke1, Black Ke3, rook e2 (check), bishop h3. e1d1 is the only
        // quiet legal evasion; it pushes halfmove to 100.
        let pos = parse_fen("8/8/8/8/8/4k2b/4r3/4K3 w - - 99 50").unwrap();
        let mut p = pos;
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);

        let m = find_move(&p, "e1d1");
        let probe = probe_move(&mut p, m, 1, 0, &ctx, &limits, &mut pv, &mut path);
        assert_eq!(
            probe,
            Some(ChildProbe::IntendedClaim),
            "e1d1 edge must be an intended fifty-move claim"
        );
    }

    /// §N.9: the e1d1 edge scores exactly 0 when driven through negamax.
    #[test]
    fn intended_fifty_move_claim_edge_scores_zero() {
        let pos = parse_fen("8/8/8/8/8/4k2b/4r3/4K3 w - - 99 50").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            1,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(
            score, 0,
            "e1d1 intended-claim edge must score exactly 0 from White's view"
        );
    }

    /// §N.4: the losing side to move at halfmove==100 returns >= 0 (claim
    /// floor), never a forced loss.
    #[test]
    fn current_node_fifty_claim_floor_for_losing_side() {
        let pos = parse_fen("7k/8/8/8/8/8/8/KQ6 b - - 100 50").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            2,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert!(
            score >= 0,
            "losing side at halfmove==100 must keep the claim floor (>= 0), got {}",
            score
        );
    }

    /// §N.4: the winning side to move still finds the mate (score > 0), the
    /// claim floor does not replace a win. Mate-in-1 with halfmove==100.
    #[test]
    fn current_node_fifty_claim_allows_win() {
        // White Kg6, Qg5; Black Kh8. Qg7# is mate-in-1.
        let pos = parse_fen("7k/8/6K1/6Q1/8/8/8/8 w - - 100 50").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            1,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert!(
            score > 0,
            "winning side at halfmove==100 must still find the mate, got {}",
            score
        );
    }

    /// §N.4 / terminal precedence: a checkmate at halfmove==100 scores the
    /// mate, not the claim floor 0.
    #[test]
    fn checkmate_priority_over_fifty_claim() {
        // Black to move, in check from a1-rook, no escape, halfmove==100.
        let pos = parse_fen("k7/2K5/8/8/8/8/8/R7 b - - 100 50").unwrap();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            1,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(
            score,
            -(MATE),
            "checkmate must outrank the fifty-move claim (mate score, not 0)"
        );
    }

    /// §N.13: a manual probe consumes exactly one node-counter tick, whether
    /// the edge is a Terminal, IntendedClaim, or Continue.
    #[test]
    fn manual_probe_node_delta_is_one() {
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();

        // IntendedClaim edge: halfmove 99 -> 100 on the e1d1 evasion.
        let pos = parse_fen("8/8/8/8/8/4k2b/4r3/4K3 w - - 99 50").unwrap();
        let mut p = pos;
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let before = ctx.nodes.load(Ordering::Relaxed);
        let m = find_move(&p, "e1d1");
        let _ = probe_move(&mut p, m, 1, 0, &ctx, &limits, &mut pv, &mut path);
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed) - before,
            1,
            "intended-claim probe must consume exactly one node"
        );

        // Continue edge: a normal quiet position (startpos) at depth.
        let pos2 = parse_fen(START_FEN).unwrap();
        let mut p2 = pos2;
        let mut pv2 = PvTable::default();
        let mut path2 = SearchPath::new(vec![p2.zobrist_key()]);
        let before2 = ctx.nodes.load(Ordering::Relaxed);
        let m2 = find_move(&p2, "e2e4");
        let _ = probe_move(&mut p2, m2, 1, 0, &ctx, &limits, &mut pv2, &mut path2);
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed) - before2,
            1,
            "continue probe must consume exactly one node"
        );
    }

    /// §N.12(b): a probe that succeeds (Continue) but whose deeper entered
    /// recursion aborts must still restore the board + path at THIS edge.
    #[test]
    fn negamax_deeper_abort_restores_state() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();

        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        // Exactly one node: the root manual probe enters, the first child
        // `negamax_entered_impl` recursion is denied -> deeper abort.
        let limits = SearchLimits {
            nodes: Some(2),
            ..Default::default()
        };
        let r = negamax_impl(
            &mut p,
            3,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        assert!(r.is_none(), "deeper abort must propagate None");
        assert_eq!(path.len(), root_len, "path restored after deeper abort");
        assert_eq!(
            path.keys(),
            &[before_key],
            "path equals root key after deeper abort"
        );
        assert_eq!(
            to_fen(&p),
            before_fen,
            "position restored after deeper abort"
        );
        assert_eq!(
            p.zobrist_key(),
            before_key,
            "key restored after deeper abort"
        );
    }

    /// §N.14 + root claim placeholder: when no real move beats the root claim
    /// floor, the completed iteration reports score 0, an EMPTY PV, and the
    /// stable fallback (protocol placeholder — NOT a found 0-score line).
    #[test]
    fn manual_edge_pv_is_single_move() {
        // Root is a fifty-move claim for the side to move; every root move is
        // also an intended claim on its own (child halfmove stays >= 100), so
        // the best edge is 0 and no real line beats the floor.
        let pos = parse_fen("7k/8/8/8/8/8/8/KQ6 b - - 100 50").unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(1),
            ..Default::default()
        };
        let out = search_best_move(&mut pos.clone(), &limits, &ctx).expect("outcome");
        // Claim floor 0 with a stable fallback move.
        assert_eq!(out.score, Some(0), "claim floor score");
        assert_eq!(out.completed_depth, 1, "a real iteration completed");
        // The claim placeholder MUST have an empty PV.
        assert!(
            out.pv.is_empty(),
            "root claim PV is empty, got {:?}",
            out.pv
        );
        // best_move is the stable fallback (first legal root move), a protocol
        // placeholder — it must be legal.
        let legal: BTreeSet<String> = generate_legal_moves(&mut pos.clone())
            .iter()
            .map(|m| move_to_uci(*m))
            .collect();
        let fallback_uci = move_to_uci(generate_legal_moves(&mut pos.clone())[0]);
        assert!(legal.contains(&move_to_uci(out.best_move)));
        assert_eq!(
            move_to_uci(out.best_move),
            fallback_uci,
            "fallback is stable"
        );
    }

    /// Root fifty-move claim does not early-return: a winning root move is
    /// still searched and reported with its real score / PV. Mate-in-1.
    #[test]
    fn root_fifty_claim_still_searches_win() {
        let pos = parse_fen("7k/8/6K1/6Q1/8/8/8/8 w - - 100 50").unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(1),
            ..Default::default()
        };
        let out = search_best_move(&mut pos.clone(), &limits, &ctx).expect("outcome");
        assert!(
            out.score.unwrap() > 0,
            "root fifty-move claim must not suppress a winning move"
        );
        // A real winning line is reported with a non-empty PV rooted at the
        // best move.
        assert!(!out.pv.is_empty(), "winning line has a PV");
        assert_eq!(out.pv[0], out.best_move, "PV is rooted at best_move");
    }

    /// Root fifty-move claim, aborted before any iteration completes, still
    /// reports the claim (score 0, fallback, empty PV, stopped) — not None.
    #[test]
    fn root_fifty_claim_abort_reports_claim() {
        let pos = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 100 50").unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        // Only one node: aborts during the very first child probe.
        let limits = SearchLimits {
            nodes: Some(1),
            ..Default::default()
        };
        let out = search_best_move(&mut pos.clone(), &limits, &ctx).expect("outcome");
        assert_eq!(out.score, Some(0), "aborted root claim still reports 0");
        assert!(out.stopped, "aborted search is stopped");
        assert!(out.pv.is_empty(), "no PV when aborted before depth 1");
        let legal: BTreeSet<String> = generate_legal_moves(&mut pos.clone())
            .iter()
            .map(|m| move_to_uci(*m))
            .collect();
        assert!(legal.contains(&move_to_uci(out.best_move)));
    }

    /// §N.9 via the ROOT move loop: a single quiet evasion that pushes
    /// halfmove 99→100 is an intended fifty-move claim scored 0 at the root.
    #[test]
    fn root_intended_fifty_claim_edge_is_zero() {
        let pos = parse_fen("8/8/8/8/8/4k2b/4r3/4K3 w - - 99 50").unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(1),
            ..Default::default()
        };
        let out = search_best_move(&mut pos.clone(), &limits, &ctx).expect("outcome");
        // The e1d1 edge is exactly 0 (intended fifty-move claim), so the root
        // reports score 0 for that edge.
        assert_eq!(out.score, Some(0), "root e1d1 intended-claim edge is 0");
        let legal: BTreeSet<String> = generate_legal_moves(&mut pos.clone())
            .iter()
            .map(|m| move_to_uci(*m))
            .collect();
        assert!(legal.contains(&move_to_uci(out.best_move)));
    }

    /// §N.9 via QSEARCH: the e1d1 quiet evasion reaches quiescence, is an
    /// intended fifty-move claim, scores 0, and leaves the position restored.
    #[test]
    fn qsearch_intended_fifty_claim_edge_is_zero() {
        let pos = parse_fen("8/8/8/8/8/4k2b/4r3/4K3 w - - 99 50").unwrap();
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = quiescence_impl(
            &mut p,
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(score, 0, "qsearch e1d1 intended-claim edge is 0");
        assert_eq!(to_fen(&p), before_fen, "position restored");
        assert_eq!(p.zobrist_key(), before_key, "key restored");
    }

    /// §N.9 via final-evasion: `search_final_evasion_ply` scores the e1d1
    /// evasion as 0 (intended claim) and restores the position.
    #[test]
    fn final_evasion_intended_fifty_claim_is_zero() {
        let pos = parse_fen("8/8/8/8/8/4k2b/4r3/4K3 w - - 99 50").unwrap();
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();
        let legal = generate_legal_moves(&mut p);
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = search_final_evasion_ply(
            &mut p,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &legal,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert_eq!(score, 0, "final-evasion e1d1 intended-claim edge is 0");
        assert_eq!(to_fen(&p), before_fen, "position restored");
        assert_eq!(p.zobrist_key(), before_key, "key restored");
    }

    // ===== C3: threefold-repetition draw =====

    /// §C3: the losing side to move when its OWN key appears a third time in
    /// the search line keeps the claim floor (score >= 0), never a forced loss.
    #[test]
    fn current_node_threefold_floor_for_losing_side() {
        let pos = parse_fen("7k/8/8/8/8/8/8/KQ6 b - - 0 1").unwrap();
        let mut pv = PvTable::default();
        let key = pos.zobrist_key();
        // The current position's key occurs 3 times on the line.
        let mut path = SearchPath::new(vec![key, key, key]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            2,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert!(
            score >= 0,
            "losing side with own key thrice must keep the claim floor (>= 0), got {}",
            score
        );
    }

    /// §C3: the winning side to move still finds the mate when its key appears
    /// a third time — the threefold claim is a 0 floor, NOT a forced draw.
    #[test]
    fn current_node_threefold_allows_win() {
        // White Kg6, Qg5; Black Kh8. Qg7# is mate-in-1.
        let pos = parse_fen("7k/8/6K1/6Q1/8/8/8/8 w - - 0 1").unwrap();
        let mut pv = PvTable::default();
        let key = pos.zobrist_key();
        let mut path = SearchPath::new(vec![key, key, key]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            1,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert!(
            score > 0,
            "winning side with own key thrice must still find the mate, got {}",
            score
        );
        // PV must contain the real winning move.
        assert!(!pv.lines[0].is_empty(), "winning PV must contain a move");
    }

    /// §C3: TWO occurrences of the current key are NOT a draw — the search
    /// takes the normal (non-claim) path. We assert the predicate and that
    /// `classify_draw` does not return ThreefoldClaim.
    #[test]
    fn threefold_two_occurrences_not_a_draw() {
        let pos = parse_fen("7k/8/8/8/8/8/8/KQ6 w - - 0 1").unwrap();
        let key = pos.zobrist_key();
        assert!(
            !is_threefold_repetition(&pos, &[key, key]),
            "two occurrences are NOT threefold"
        );
        assert_ne!(
            classify_draw(&pos, &[key, key]),
            Some(DrawReason::ThreefoldClaim),
            "two occurrences must not classify as ThreefoldClaim"
        );
        // The search must run normally (not floor at 0). Use a clearly winning
        // position so the normal path returns a positive, non-claim score.
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![key, key]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            1,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert!(
            score > 0,
            "two occurrences: normal winning search, got {}",
            score
        );
    }

    /// §C3: an intended threefold claim on a quiet move. The parent is NOT yet
    /// a repetition, but after making `g1f3` the child key appears a third time
    /// on the extended line, so the edge is an `IntendedClaim` and scores 0.
    #[test]
    fn intended_threefold_claim_edge_scores_zero() {
        // Parent = startpos (key once). Quiet move g1f3.
        let pos = parse_fen(START_FEN).unwrap();
        let parent_key = pos.zobrist_key();
        let m = find_move(&pos, "g1f3");
        let mut child = pos;
        child.make_move(m);
        let child_key = child.zobrist_key();
        // Pre-move line: child_key twice (older reps) + parent_key.
        let mut path = SearchPath::new(vec![child_key, child_key, parent_key]);

        assert_eq!(path.keys().last(), Some(&parent_key));
        assert_eq!(pos.zobrist_key(), parent_key);

        let mut pv = PvTable::default();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();

        // Drive the edge through the manual probe (make + push done here).
        {
            let mut p = pos;
            let undo = p.make_move(m);
            path.push_child(&p);
            assert_eq!(path.keys().last(), Some(&child_key));
            assert_eq!(p.zobrist_key(), child_key);
            assert!(
                claim_available_by_intended_move(&p, path.keys()),
                "after g1f3 the child is an intended threefold claim"
            );
            let probe = probe_child_draw(&mut p, path.keys(), 1, 0, &ctx, &limits, &mut pv);
            assert_eq!(
                probe,
                Some(ChildProbe::IntendedClaim),
                "g1f3 edge must be an intended threefold claim"
            );
            path.pop();
            p.unmake_move(undo);
            // Explicit restoration: FEN + key restored, and the search path is
            // back to exactly its pre-move state.
            assert_eq!(to_fen(&p), to_fen(&pos), "manual probe restored FEN");
            assert_eq!(p.zobrist_key(), parent_key, "manual probe restored key");
            assert_eq!(
                path.keys(),
                &[child_key, child_key, parent_key][..],
                "manual probe restored path"
            );
        }

        // The parent edge (mover's perspective) for the SINGLE move g1f3 must
        // score the claim as 0. Driving the whole-node negamax would return the
        // best of ALL root moves (another move may win), so we score this one
        // edge in isolation through the same probe->edge path.
        let edge = score_one_move_edge(
            &mut pos.clone(),
            m,
            vec![child_key, child_key, parent_key],
            1,
        );
        assert_eq!(edge, 0, "intended-threefold g1f3 edge scores 0");
        // The intended move is a real PV move on this edge, not the root
        // placeholder empty PV. Re-derive it by making the move + probing.
        let mut pv2 = PvTable::default();
        let mut path2 = SearchPath::new(vec![child_key, child_key, parent_key]);
        let probe2 = probe_move(
            &mut pos.clone(),
            m,
            1,
            0,
            &ctx,
            &limits,
            &mut pv2,
            &mut path2,
        );
        assert_eq!(
            probe2,
            Some(ChildProbe::IntendedClaim),
            "g1f3 edge is an intended threefold claim (single move)"
        );
    }

    /// §C3: a root that is NOT itself a claim, searching ONLY the quiet move
    /// `g1f3` whose child key appears a third time, must treat it as a real
    /// intended-threefold edge (not a root claim placeholder): the completed
    /// iteration is exactly `score 0`, `best_move g1f3`, `pv [g1f3]`.
    #[test]
    fn root_intended_threefold_via_single_move() {
        let pos = parse_fen(START_FEN).unwrap();
        let parent_key = pos.zobrist_key();
        let m = find_move(&pos, "g1f3");
        let mut child = pos;
        child.make_move(m);
        let child_key = child.zobrist_key();

        // Parent is not yet a repetition, so root_claimable = false.
        assert_ne!(
            classify_draw(&pos, &[child_key, child_key, parent_key]),
            Some(DrawReason::ThreefoldClaim),
            "parent itself is not a claim"
        );

        let mut path = SearchPath::new(vec![child_key, child_key, parent_key]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(2),
            ..Default::default()
        };
        let iter = root_search(
            &mut pos.clone(),
            2,
            &[m],
            false, // parent is NOT itself a claim
            m,     // fallback (also g1f3 here)
            &ctx,
            &limits,
            &mut path,
        )
        .expect("completed iteration");
        path.restore_root(3);

        assert_eq!(iter.score, 0, "intended-threefold root edge scores 0");
        assert_eq!(iter.best_move, m, "best_move is the intended claim move");
        assert_eq!(
            iter.pv,
            vec![m],
            "intended-threefold root PV is exactly [g1f3]"
        );
    }

    /// §C3: a checkmate child must precede an intended threefold claim. Using
    /// the mate-in-1 fixture, after the mating move the child key appears a
    /// third time AND the child is checkmated: the probe returns Terminal (a
    /// positive mate score), never IntendedClaim.
    #[test]
    fn mate_precedence_over_threefold_intended_claim() {
        // White Kg6, Qg5; Black Kh8. Qg7# is mate-in-1.
        let pos = parse_fen("7k/8/6K1/6Q1/8/8/8/8 w - - 0 1").unwrap();
        // Discover the actual mate-in-1 move (the queen cannot pass through its
        // own king, so we don't hardcode the square).
        let m = generate_legal_moves(&mut pos.clone())
            .into_iter()
            .find(|mm| {
                let mut child = pos;
                child.make_move(*mm);
                child.is_in_check(child.side) && generate_legal_moves(&mut child).is_empty()
            })
            .expect("a mate-in-1 move exists");
        let parent_key = pos.zobrist_key();
        let mut child = pos;
        child.make_move(m);
        let child_key = child.zobrist_key();
        assert!(child.is_in_check(child.side));
        assert!(
            generate_legal_moves(&mut child).is_empty(),
            "child is checkmated"
        );

        // Pre-move line: child_key twice + parent_key (so after push it's thrice).
        let mut path = SearchPath::new(vec![child_key, child_key, parent_key]);
        let mut pv = PvTable::default();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();

        let mut p = pos;
        let undo = p.make_move(m);
        path.push_child(&p);
        let probe = probe_child_draw(&mut p, path.keys(), 1, 0, &ctx, &limits, &mut pv);
        assert_eq!(
            probe,
            Some(ChildProbe::Terminal(MATE - 1)),
            "checkmate child must precede the threefold intended claim"
        );
        path.pop();
        p.unmake_move(undo);
        assert_eq!(to_fen(&p), to_fen(&pos), "mate probe restored FEN");
        assert_eq!(p.zobrist_key(), parent_key, "mate probe restored key");
        assert_eq!(
            path.keys(),
            &[child_key, child_key, parent_key][..],
            "mate probe restored path"
        );

        // Full PV: the mating move is found.
        let mut pv2 = PvTable::default();
        let mut path2 = SearchPath::new(vec![child_key, child_key, parent_key]);
        let root_len = path2.len();
        let ctx2 = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits2 = SearchLimits::default();
        let score = negamax_impl(
            &mut pos.clone(),
            1,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx2,
            &limits2,
            &mut pv2,
            &mut path2,
        )
        .expect("not stopped");
        path2.restore_root(root_len);
        assert!(
            score > 0,
            "mate precedence: winning line still returned, got {}",
            score
        );
        assert!(pv2.lines[0].contains(&m), "PV contains the mating move");
    }

    /// §C3 / root: a losing side whose key appears a third time in the real
    /// history claims -> score 0, stable legal fallback, empty PV.
    #[test]
    fn root_threefold_losing_claims_zero() {
        // Black to move, down a queen (losing side). Its key on the line 3x.
        let pos0 = parse_fen("7k/8/8/8/8/8/8/KQ6 b - - 0 1").unwrap();
        let key = pos0.zobrist_key();
        let history = vec![key, key, key];
        // The stable fallback is the first legal root move, captured BEFORE any
        // move-ordering swap inside the search.
        let initial_root_move = generate_legal_moves(&mut pos0.clone())[0];
        let mut pos = pos0;
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let out =
            search_best_move_with_history(&mut pos, &history, &limits, &ctx).expect("outcome");
        assert_eq!(out.score, Some(0), "root threefold losing side scores 0");
        assert!(out.pv.is_empty(), "root threefold claim PV is empty");
        assert!(
            out.completed_depth >= 1,
            "a completed iteration reported the claim"
        );
        assert_eq!(
            out.best_move, initial_root_move,
            "stable fallback equals the initial first legal root move"
        );
    }

    /// §C3 / root: a winning mate-in-1 whose key appears a third time still
    /// returns the real mate (score > 0, non-empty PV containing the move).
    #[test]
    fn root_threefold_winning_still_wins() {
        let pos0 = parse_fen("7k/8/6K1/6Q1/8/8/8/8 w - - 0 1").unwrap();
        let key = pos0.zobrist_key();
        // History: key appears 3 times (startpos, after two null-ish reps).
        let history = vec![key, key, key];
        let mut pos = pos0;
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let out =
            search_best_move_with_history(&mut pos, &history, &limits, &ctx).expect("outcome");
        assert!(
            out.score.unwrap() > 0,
            "root threefold winning side still wins"
        );
        assert!(!out.pv.is_empty(), "root threefold winning PV non-empty");
        assert_eq!(
            out.pv[0], out.best_move,
            "root threefold winning PV[0] == best_move"
        );
    }

    /// §C3 / root: a terminal (checkmate) root is never masked by threefold —
    /// the public API still returns None.
    #[test]
    fn root_terminal_not_masked_by_threefold() {
        // Black Kh8, White Kf7, White Rh1: Black is checkmated (no moves).
        let pos0 = parse_fen("7k/5K2/8/8/8/8/8/7R b - - 0 1").unwrap();
        let key = pos0.zobrist_key();
        let history = vec![key, key, key];
        let mut pos = pos0;
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let out = search_best_move_with_history(&mut pos, &history, &limits, &ctx);
        assert!(
            out.is_none(),
            "terminal root returns None, not masked by threefold"
        );
    }

    /// §N.12(b) qsearch deeper abort: root entry succeeds, a child probe
    /// succeeds and returns `Continue`, the entered child qsearch then tries
    /// a grandchild probe that FAILS the node budget. The abort must unwind
    /// both edges (pop + unmake each) and fully restore the root position.
    ///
    /// Control flow with `nodes: Some(2)`:
    ///   1. `quiescence_impl` root entry acquires node #1.
    ///   2. White `Qe4xa4`: make + push, `probe_child_draw` acquires node #2
    ///      and returns `Continue`.
    ///   3. Enters child `quiescence_entered_impl` (a real recursion).
    ///   4. Black `Ra8xa4` is a tactical reply; its grandchild probe tries to
    ///      acquire node #3, which fails (budget 2).
    ///   5. The grandchild abort unwinds: Rxa4 edge pops+unmakes, the None
    ///      propagates to the Qxa4 edge, which also pops+unmakes, and None
    ///      reaches the root. Board + path + key fully restored.
    #[test]
    fn qsearch_deeper_abort_restores_state() {
        let fen = "r6k/8/8/8/q3Q3/8/8/4K3 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let mut p = pos;
        let before_fen = to_fen(&p);
        let before_key = p.zobrist_key();

        // Verify the fixture really forces the intended two-step chain.
        assert!(
            generate_legal_moves(&mut p.clone())
                .iter()
                .any(|m| move_to_uci(*m) == "e4a4"),
            "fixture must allow White Qxa4"
        );
        {
            let mut q = p;
            let undo = q.make_move(find_move(&q, "e4a4"));
            let has_raxa4 = generate_legal_moves(&mut q)
                .iter()
                .any(|m| move_to_uci(*m) == "a8a4");
            q.unmake_move(undo);
            assert!(has_raxa4, "after Qxa4, Black must have tactical Ra8xa4");
        }

        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let root_len = path.len();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            nodes: Some(2),
            ..Default::default()
        };
        let r = quiescence_impl(
            &mut p,
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
        );
        // Root entry (#1) + child probe (#2) succeeded; grandchild probe (#3)
        // failed -> deeper abort propagates None.
        assert!(r.is_none(), "qsearch deeper abort must propagate None");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            2,
            "exactly nodes 1 (root) + 2 (child probe) were acquired"
        );
        assert_eq!(
            path.len(),
            root_len,
            "path restored after qsearch deeper abort"
        );
        assert_eq!(
            path.keys(),
            &[before_key],
            "path equals root key after qsearch deeper abort"
        );
        assert_eq!(to_fen(&p), before_fen, "position restored");
        assert_eq!(p.zobrist_key(), before_key, "key restored");
    }

    /// White-box `root_search` with a claimable root where the only searched
    /// move RESETS the halfmove clock (so it is NOT an intended claim and its
    /// search value is negative). The root claim floor must hold: the
    /// completed iteration reports score 0, the stable fallback, empty PV.
    #[test]
    fn root_claim_floor_holds_when_all_moves_reset_halfmove() {
        // White to move at halfmove 100, with a pawn move available. The pawn
        // move resets halfmove to 0 (not an intended claim) and is losing for
        // White (Black is up a queen). `root_claimable` should keep the root
        // value at 0.
        let fen = "4k3/3q4/8/8/8/8/4P3/K7 w - - 100 50";
        let pos = parse_fen(fen).unwrap();
        let mut p = pos;

        // Verify the chosen move resets the halfmove clock.
        let pm = find_move(&p, "e2e4");
        let undo = p.make_move(pm);
        assert_eq!(
            p.halfmove_clock(),
            0,
            "pawn move must reset the halfmove clock (not an intended claim)"
        );
        p.unmake_move(undo);

        // Control the root move list to ONLY the pawn move, so the search
        // cannot fall back on an intended-claim edge.
        let root_moves = vec![pm];
        let fallback = pm;
        let mut path = SearchPath::new(vec![p.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(2),
            ..Default::default()
        };
        let iter = root_search(
            &mut p,
            2,
            &root_moves,
            true, // root_claimable
            fallback,
            &ctx,
            &limits,
            &mut path,
        )
        .expect("completed iteration");
        assert_eq!(iter.score, 0, "root claim floor holds at 0");
        assert_eq!(
            iter.best_move, fallback,
            "claim placeholder is the stable fallback"
        );
        assert!(iter.pv.is_empty(), "claim placeholder PV is empty");
    }

    // ===== M3.2 SearchPath: counts, repetition_signature =====

    /// Constructor with duplicate keys builds correct counts, occurrences, and signature.
    #[test]
    fn search_path_constructor_with_duplicates() {
        // Two distinct keys, one repeated three times.
        let keys = vec![100, 200, 100, 200, 100];
        let path = SearchPath::new(keys);

        assert_eq!(path.occurrences(100), 3);
        assert_eq!(path.occurrences(200), 2);
        assert_eq!(path.occurrences(300), 0);

        // Signature must match a fresh rebuild.
        assert_eq!(path.repetition_signature(), path.rebuild_signature());
    }

    /// Same multiset, different order -> same signature.
    #[test]
    fn search_path_same_multiset_diff_order_same_signature() {
        let a = vec![1, 2, 1, 2, 3];
        let b = vec![2, 1, 3, 2, 1];
        let p1 = SearchPath::new(a);
        let p2 = SearchPath::new(b);
        assert_eq!(
            p1.repetition_signature(),
            p2.repetition_signature(),
            "same multiset must produce same XOR signature"
        );
    }

    /// Different occurrence count -> different signature.
    #[test]
    fn search_path_different_count_differs() {
        let a = vec![1, 1];
        let b = vec![1];
        let p1 = SearchPath::new(a);
        let p2 = SearchPath::new(b);
        assert_ne!(
            p1.repetition_signature(),
            p2.repetition_signature(),
            "different occurrence counts must produce different signatures"
        );
    }

    /// Single push_child and pop restores all three fields.
    #[test]
    fn search_path_push_pop_restores_all() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let before_keys = path.keys().to_vec();
        let before_sig = path.repetition_signature();

        let mv = find_move(&pos, "e2e4");
        let mut child = pos;
        child.make_move(mv);
        path.push_child(&child);

        // After push: len increased, signature changed.
        assert_eq!(path.len(), 2);
        assert_ne!(path.repetition_signature(), before_sig);
        assert_eq!(path.occurrences(child.zobrist_key()), 1);

        path.pop();

        // After pop: fully restored.
        assert_eq!(path.len(), 1);
        assert_eq!(path.keys(), &before_keys[..]);
        assert_eq!(path.repetition_signature(), before_sig);
        assert_eq!(path.occurrences(child.zobrist_key()), 0);
    }

    /// Nested depth-3 push/pop fully restores.
    #[test]
    fn search_path_nested_push_pop_three() {
        // e2e4 e7e5 g1f3
        let pos = parse_fen(START_FEN).unwrap();
        let root_key = pos.zobrist_key();
        let mut path = SearchPath::new(vec![root_key]);
        let orig_keys = path.keys().to_vec();
        let orig_sig = path.repetition_signature();

        let mut p = pos;

        let moves = ["e2e4", "e7e5", "g1f3"];
        for &uci in &moves {
            let m = find_move(&p, uci);
            let undo = p.make_move(m);
            path.push_child(&p);
            let _ = undo;
        }
        assert_eq!(path.len(), 4);

        for _ in 0..3 {
            path.pop();
        }

        assert_eq!(path.len(), 1);
        assert_eq!(path.keys(), &orig_keys[..]);
        assert_eq!(path.repetition_signature(), orig_sig);
        // Root key has occurrence count 1 after full restore.
        assert_eq!(path.occurrences(root_key), 1);
    }

    /// restore_root from a deep child reverts all fields.
    #[test]
    fn search_path_restore_root_deep() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let orig_sig = path.repetition_signature();

        let mut p = pos;
        let moves = ["e2e4", "e7e5", "g1f3", "b8c6"];
        for &uci in &moves {
            let m = find_move(&p, uci);
            let undo = p.make_move(m);
            path.push_child(&p);
            let _ = undo;
        }
        assert_eq!(path.len(), 5);

        path.restore_root(1);
        assert_eq!(path.len(), 1);
        // Counts should reflect only the root key.
        assert_eq!(path.occurrences(pos.zobrist_key()), 1);
        assert_eq!(path.repetition_signature(), orig_sig);
    }

    /// Abort-style: push child, restore to intermediate root_len, then continue.
    #[test]
    fn search_path_abort_style_restore() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);

        let mut p = pos;

        // Push e2e4, e7e5
        for &uci in &["e2e4", "e7e5"] {
            let m = find_move(&p, uci);
            let undo = p.make_move(m);
            path.push_child(&p);
            let _ = undo;
        }
        let sig_after_two = path.repetition_signature();

        // Push g1f3 (abort target will restore_before this)
        let m = find_move(&p, "g1f3");
        let undo = p.make_move(m);
        path.push_child(&p);
        let _ = undo;
        assert_eq!(path.len(), 4);

        // Abort back to depth 2 (root + 2 children)
        path.restore_root(3);
        assert_eq!(path.len(), 3);
        assert_eq!(path.repetition_signature(), sig_after_two);

        // Continue with a different third push
        let m = find_move(&p, "d7d5");
        let undo = p.make_move(m);
        path.push_child(&p);
        let _ = undo;
        assert_eq!(path.len(), 4);
        // Signature must differ from the original depth-3 path (different key).
        assert_ne!(path.repetition_signature(), sig_after_two);
    }

    /// Clone preserves all three fields.
    #[test]
    fn search_path_clone_preserves_fields() {
        let keys = vec![1, 2, 1, 3, 2];
        let path = SearchPath::new(keys);
        let cloned = path.clone();
        assert_eq!(cloned.keys(), path.keys());
        assert_eq!(cloned.repetition_signature(), path.repetition_signature());
        assert_eq!(cloned.occurrences(1), path.occurrences(1));
        assert_eq!(cloned.occurrences(2), path.occurrences(2));
    }

    /// Fresh rebuild of signature equals incremental signature.
    #[test]
    fn search_path_rebuild_equals_incremental() {
        let keys = vec![10, 20, 10, 30, 20, 10];
        let path = SearchPath::new(keys);
        assert_eq!(
            path.repetition_signature(),
            path.rebuild_signature(),
            "incremental signature must equal a fresh scan"
        );
    }
}
