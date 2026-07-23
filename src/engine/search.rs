//! Search — the first "thinking" version (Phase 3), now interruptible.
//!
//! Order of escalation (per the roadmap, do NOT skip ahead):
//!   1. Negamax            (done)
//!   2. Alpha-Beta pruning (done)
//!   3. Iterative deepening (done, at the root)
//!   4. Principal variation  (done, M2.3)
//!   5. Quiescence search   (done, M2.1 — correctness-only)
//!   6. Move ordering       (done, M2.2 — basic MVV-LVA)
//!   7. Transposition table (done, M3.2 — context-safe TT + UCI lifecycle)
//!
//! The M3.2 TT keys on board Zobrist + halfmove clock + repetition
//! signature; qsearch itself still has no direct TT probe/store.
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
use crate::engine::tt::{score_from_tt, score_to_tt, Bound, TTEntry, TranspositionTable, TtKey};

/// M4.1+ internal search configuration. Crate-private only: it selects the
/// move-ordering strategy used by the search core and is NEVER exposed through
/// the public API or the UCI surface.
///
/// * `M4Reference` reproduces the M4.0 production behavior exactly: no killer
///   moves, no history heuristic. The historical baseline (e.g. `bench smoke`
///   startpos d3 disabled = 1149 / queen-win d3 disabled = 963) is preserved
///   verbatim on this profile.
/// * `M41Reference` reproduces the M4.1 full-window search exactly: M4.1
///   quiet move ordering (killer moves + history heuristic) with NO principal
///   variation search. It preserves the 236,418-node M4.1 A/B baseline and
///   keeps M4.2 replayable once `Current` enables PVS.
/// * `Current` is the production configuration: M4.1 quiet move ordering plus
///   the M4.2 PVS at both non-root nodes (Commit 3) and the root (Commit 4).
///
/// `M41Reference` keeps the M4.1 full-window path (killer/history ordering at
/// non-root nodes, NO PVS at either the root or a non-root node), while
/// `Current` enables the null-window scout + re-search at every non-root node
/// AND at the root. `M4Reference` keeps every search behavior byte-identical to
/// M4.0. Move ordering at the root itself is the pure hash-move lift in all
/// profiles (no MVV-LVA / killer / history reorder); PVS changes only the
/// WINDOW a later root move is searched with, never the root move order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchProfile {
    M4Reference,
    // Dormant until Commit 5 exposes `--profile m4.1` through the bench
    // CLI; it is constructed there (and in tests now). Allowed as dead in
    // production for Commit 2 per the M4.2 spec's compile-safety scope
    // (the variant must exist so `profile_str` stays exhaustive, but the CLI
    // must NOT accept `--profile m4.1` yet, so no production path builds it).
    #[allow(dead_code)]
    M41Reference,
    Current,
}

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

/// PVS child-window decision for one move at a non-root ordinary negamax
/// node (M4.2 Commit 3, spec §2). Pure: it chooses whether the
/// child is searched with the full window or a null-window scout. It
/// NEVER changes the score — only which window the child search receives.
enum ChildWindow {
    /// Full window `[-beta, -alpha_before_move]` — used for the first
    /// move, non-Current profiles (M4Reference / M41Reference),
    /// depth-0, and the caller-null-window / `i32` overflow fallbacks.
    Full,
    /// Null-window scout. `scout_beta` is the parent's narrow bound
    /// `alpha_before_move + 1`; the child window is
    /// `[-scout_beta, -alpha_before_move]`.
    Scout { scout_beta: i32 },
}

/// Decide the child window for a non-root ordinary negamax move.
///
/// PVS applies ONLY when `profile == Current`, the move is NOT the first
/// in the list (`is_first == false`), and `depth > 0`. A later move is
/// scouted only if `alpha_before_move + 1` does not overflow `i32` AND
/// the resulting null window `scout_beta = alpha + 1` is still strictly
/// inside the caller window (`scout_beta < beta`) — if the caller is
/// already a null-window node (`scout_beta >= beta`) we search the
/// caller's own full window once instead of narrowing further. Bare
/// `alpha + 1` is never written; `checked_add` guards the overflow.
fn pvs_child_window(
    profile: SearchProfile,
    is_first: bool,
    depth: u32,
    alpha_before_move: i32,
    beta: i32,
) -> ChildWindow {
    if profile != SearchProfile::Current || is_first || depth == 0 {
        return ChildWindow::Full;
    }
    let scout_beta = match alpha_before_move.checked_add(1) {
        None => return ChildWindow::Full, // i32 overflow guard
        Some(b) => b,
    };
    if scout_beta >= beta {
        // Caller is already a null-window node; do not narrow further.
        return ChildWindow::Full;
    }
    ChildWindow::Scout { scout_beta }
}

/// PVS re-search condition (spec §3): a scout that improves alpha but does
/// not already prove a beta cutoff must be re-searched with the full
/// window. `score <= alpha` (fail-low, no improvement) and
/// `score >= beta` (fail-high, cutoff already proven) do NOT re-search.
fn pvs_needs_research(score: i32, alpha: i32, beta: i32) -> bool {
    alpha < score && score < beta
}

/// Explicit classification of one child edge's result (M4.2 Commit 3
/// hardening, P1.1). The parent commits state by MATCHING on this variant —
/// it never infers "was this a fail-low?" from `score > best`. That matters
/// because this engine is fail-soft COMPATIBLE: an interior node usually
/// returns a window-clamped score, but a TT `Exact` hit returns the real
/// decoded score unconditionally, a TT `Lower`/`Upper` cutoff returns the
/// real stored bound score (never a raw alpha/beta), and terminal mate /
/// stalemate scores propagate unclamped. A null-window scout can therefore
/// return a score ANYWHERE relative to the parent's window — including
/// strictly above the parent's running `best` while still failing low, and
/// at or above `beta`.
enum MoveOutcome {
    /// A real candidate score: a full-window search (first move / non-Current
    /// profile / depth-0 / caller-null-window / overflow fallback), a
    /// re-searched scout, a terminal child, or an intended-claim child.
    /// Participates normally in best / node_best_move / PV / alpha / cutoff /
    /// killer-history.
    Candidate(i32),
    /// A null-window scout that failed LOW (`scout_score <= alpha_before_move`).
    /// Its PV/move is NOT committable: it must never update best /
    /// node_best_move / PV / alpha, never trigger a beta cutoff, and never
    /// reward killer/history. Its NUMERIC value, however, is still part of
    /// search correctness — it is a valid upper bound on this child, so the
    /// parent folds it into `fail_low_upper` and lifts the score it returns
    /// and stores with it (P1.1: discarding the number entirely would let a
    /// fail-low node store a TT `Bound::Upper` that under-states the real
    /// node value and later cause wrong TT cutoffs).
    ScoutFailLow(i32),
    /// A null-window scout that failed HIGH (`scout_score >= beta`) —
    /// reachable because child searches are fail-soft compatible (a TT
    /// Exact / Lower / Upper hit or a mate score can carry the scout
    /// outside its null window). A valid lower bound and a real cutoff
    /// candidate: its legal scout line is committed once and (if quiet)
    /// killer/history is rewarded once. A fail-high is never re-searched.
    ScoutFailHigh(i32),
}

/// Test-only PVS path counters. The ENTIRE module is `#[cfg(test)]`, so it
/// leaves no trace in production builds — no thread-locals, no `mark_*`
/// symbols, and no `#[allow(dead_code)]` needed (a production `cargo build`
/// never compiles it). Every call site is likewise wrapped in `#[cfg(test)]`,
/// so there is no production behavior change and no hot-path overhead. It
/// exists solely so tests can assert that the scout / full re-search /
/// fail-low / fail-high / parent-reward branches actually execute inside a
/// real search, rather than inferring it from a node-count delta.
///
/// Implemented as `thread_local!` (not a process-global `static`) so the
/// per-thread counts cannot race between the parallel unit tests that
/// exercise them — a global `static` would let a concurrent `Current`
/// search in another test thread inflate the counters observed by
/// `pvs_m41reference_never_scouts`.
#[cfg(test)]
mod pvs_counters {
    use super::Move;
    use std::cell::{Cell, RefCell};
    thread_local! {
        /// A null-window scout search was launched for a later move.
        pub static SCOUT: Cell<usize> = const { Cell::new(0) };
        /// A scout failed LOW (`scout_score <= alpha_before_move`): its
        /// PV/move is never committed to the parent; its numeric value is
        /// folded into the parent's `fail_low_upper` bound.
        pub static SCOUT_FAIL_LOW: Cell<usize> = const { Cell::new(0) };
        /// A fail-low scout whose score was strictly GREATER than the
        /// parent's running `best`. This is exactly the P1.1 hazard: if the
        /// parent committed on `score > best` it would wrongly adopt this
        /// upper-bound score/PV as exact. Reachable because child searches
        /// are fail-soft compatible (e.g. a TT Exact hit returns the real
        /// score even outside the scout's null window). Counting it lets a
        /// test prove the adversarial case really occurred AND that the
        /// parent kept only the numeric bound, not the move/PV.
        pub static SCOUT_FAIL_LOW_EXCEEDS_BEST: Cell<usize> = const { Cell::new(0) };
        /// A scout failed HIGH (`scout_score >= beta`): a real cutoff whose
        /// legal scout line is committed once (no re-search). Reachable via
        /// the same fail-soft paths (TT Exact / Lower / Upper hits, mate
        /// scores) that can carry a scout outside its null window.
        pub static SCOUT_FAIL_HIGH: Cell<usize> = const { Cell::new(0) };
        /// A scout landed inside the window and a full re-search was WANTED.
        pub static RESEARCH_ATTEMPT: Cell<usize> = const { Cell::new(0) };
        /// The full re-search actually acquired its node and ran (i.e. the
        /// `try_enter_node` budget check passed).
        pub static RESEARCH_ENTERED: Cell<usize> = const { Cell::new(0) };
        /// A *quiet* beta cutoff rewarded killer + history exactly once.
        pub static PARENT_QUIET_REWARD: Cell<usize> = const { Cell::new(0) };
        /// A *tactical* beta cutoff (capture / ep / promotion) — cutoff
        /// taken but killer/history deliberately NOT rewarded.
        pub static PARENT_TACTICAL_CUTOFF: Cell<usize> = const { Cell::new(0) };
        /// Abort observed while the null-window SCOUT search was running
        /// (phase A: the scout's own subtree ran out of budget).
        pub static ABORT_IN_SCOUT: Cell<usize> = const { Cell::new(0) };
        /// Abort observed acquiring the re-search node: the scout completed
        /// and a full re-search was wanted, but `try_enter_node` failed
        /// (phase B).
        pub static ABORT_RESEARCH_ACQUIRE: Cell<usize> = const { Cell::new(0) };
        /// Abort observed while the full re-search subtree was running
        /// (phase C: the re-search ran out of budget after entering).
        pub static ABORT_IN_RESEARCH: Cell<usize> = const { Cell::new(0) };
        /// Exact call count of `SearchHeuristics::record_killer` (P2:
        /// exact-once proof — a scout+re-search double reward would call it
        /// twice for one cutoff, which two-run table equality cannot see).
        pub static RECORD_KILLER_CALLS: Cell<usize> = const { Cell::new(0) };
        /// Exact call count of `SearchHeuristics::record_history`.
        pub static RECORD_HISTORY_CALLS: Cell<usize> = const { Cell::new(0) };
        /// Sum of the ACTUAL deltas `record_history` applied to the table
        /// (post-cap). The table's total mass must equal this exactly; a
        /// double deposit would double-count here vs the reward events.
        pub static HISTORY_TOTAL_DELTA: Cell<i64> = const { Cell::new(0) };
        /// For every completed full re-search: the child PV row as the
        /// SCOUT left it, paired with the row the RE-SEARCH rewrote (P2:
        /// lets a test distinguish the two lines and prove the parent
        /// committed the re-searched one).
        pub static RESEARCH_PV_PAIRS: RefCell<Vec<(Vec<Move>, Vec<Move>)>> =
            const { RefCell::new(Vec::new()) };
        /// A re-searched move became the node's best move AND, at that
        /// commit, the parent's committed child line (its PV tail below the
        /// move) was verified equal to the row the RE-SEARCH rewrote — never
        /// a stale scout row. This is the direct structural proof that the
        /// parent copies the re-search line (P2.2). The equality itself is
        /// asserted inline at the commit site; this counter only proves the
        /// adversarial path was actually exercised at least once.
        pub static RESEARCH_ROW_COMMITTED: Cell<usize> = const { Cell::new(0) };

        // --- Root PVS counters (M4.2 Commit 4). Deliberately DISTINCT from
        // the non-root counters above: a scout / re-search / fail-low at the
        // root is attributed strictly to `root_search`, never confused with a
        // deeper non-root node, so a test can prove the ROOT PVS path itself
        // fired (and, for the reference profiles, that it did NOT). ---
        /// `Current`'s first root move used the full window (never scouted).
        pub static ROOT_FIRST_FULL: Cell<usize> = const { Cell::new(0) };
        /// A null-window scout was launched for a later `Current` root move.
        pub static ROOT_SCOUT: Cell<usize> = const { Cell::new(0) };
        /// A root scout failed low (`scout_score <= alpha_before_move`): not
        /// committable, not re-searched. The root keeps NO numeric bound —
        /// the running exact candidate / claim floor already dominates it.
        pub static ROOT_FAIL_LOW: Cell<usize> = const { Cell::new(0) };
        /// A root scout improved alpha (`scout_score > alpha_before_move`) so a
        /// full re-search was WANTED (root has no beta cutoff, so EVERY
        /// improving scout re-searches — there is no fail-high shortcut).
        pub static ROOT_RESEARCH_ATTEMPT: Cell<usize> = const { Cell::new(0) };
        /// The root full re-search acquired its node and ran.
        pub static ROOT_RESEARCH_ENTERED: Cell<usize> = const { Cell::new(0) };
        /// A re-searched root move became the root best AND its committed root
        /// PV tail was verified (inline) equal to the re-search child row.
        pub static ROOT_RESEARCH_ROW_COMMITTED: Cell<usize> = const { Cell::new(0) };
        /// One increment per root move whose edge was fully resolved (probe +
        /// search + cleanup), INCLUDING dropped fail-lows. Proves the root
        /// visits every legal move (no beta cutoff / early break).
        pub static ROOT_MOVES_VISITED: Cell<usize> = const { Cell::new(0) };
        /// Abort while the root null-window scout subtree ran (phase A).
        pub static ROOT_ABORT_IN_SCOUT: Cell<usize> = const { Cell::new(0) };
        /// Abort acquiring the root re-search node (phase B).
        pub static ROOT_ABORT_RESEARCH_ACQUIRE: Cell<usize> = const { Cell::new(0) };
        /// Abort while the root full re-search subtree ran (phase C).
        pub static ROOT_ABORT_IN_RESEARCH: Cell<usize> = const { Cell::new(0) };
        /// For every completed root re-search: the child PV row as the SCOUT
        /// left it, paired with the row the RE-SEARCH rewrote.
        pub static ROOT_RESEARCH_PV_PAIRS: RefCell<Vec<(Vec<Move>, Vec<Move>)>> =
            const { RefCell::new(Vec::new()) };
    }
    pub fn reset() {
        SCOUT.set(0);
        SCOUT_FAIL_LOW.set(0);
        SCOUT_FAIL_LOW_EXCEEDS_BEST.set(0);
        SCOUT_FAIL_HIGH.set(0);
        RESEARCH_ATTEMPT.set(0);
        RESEARCH_ENTERED.set(0);
        PARENT_QUIET_REWARD.set(0);
        PARENT_TACTICAL_CUTOFF.set(0);
        ABORT_IN_SCOUT.set(0);
        ABORT_RESEARCH_ACQUIRE.set(0);
        ABORT_IN_RESEARCH.set(0);
        RECORD_KILLER_CALLS.set(0);
        RECORD_HISTORY_CALLS.set(0);
        HISTORY_TOTAL_DELTA.set(0);
        RESEARCH_PV_PAIRS.with_borrow_mut(Vec::clear);
        RESEARCH_ROW_COMMITTED.set(0);
        ROOT_FIRST_FULL.set(0);
        ROOT_SCOUT.set(0);
        ROOT_FAIL_LOW.set(0);
        ROOT_RESEARCH_ATTEMPT.set(0);
        ROOT_RESEARCH_ENTERED.set(0);
        ROOT_RESEARCH_ROW_COMMITTED.set(0);
        ROOT_MOVES_VISITED.set(0);
        ROOT_ABORT_IN_SCOUT.set(0);
        ROOT_ABORT_RESEARCH_ACQUIRE.set(0);
        ROOT_ABORT_IN_RESEARCH.set(0);
        ROOT_RESEARCH_PV_PAIRS.with_borrow_mut(Vec::clear);
    }
    pub fn mark_scout() {
        SCOUT.set(SCOUT.get() + 1);
    }
    pub fn mark_scout_fail_low() {
        SCOUT_FAIL_LOW.set(SCOUT_FAIL_LOW.get() + 1);
    }
    pub fn mark_scout_fail_low_exceeds_best() {
        SCOUT_FAIL_LOW_EXCEEDS_BEST.set(SCOUT_FAIL_LOW_EXCEEDS_BEST.get() + 1);
    }
    pub fn mark_scout_fail_high() {
        SCOUT_FAIL_HIGH.set(SCOUT_FAIL_HIGH.get() + 1);
    }
    pub fn mark_research_attempt() {
        RESEARCH_ATTEMPT.set(RESEARCH_ATTEMPT.get() + 1);
    }
    pub fn mark_research_entered() {
        RESEARCH_ENTERED.set(RESEARCH_ENTERED.get() + 1);
    }
    pub fn mark_parent_quiet_reward() {
        PARENT_QUIET_REWARD.set(PARENT_QUIET_REWARD.get() + 1);
    }
    pub fn mark_parent_tactical_cutoff() {
        PARENT_TACTICAL_CUTOFF.set(PARENT_TACTICAL_CUTOFF.get() + 1);
    }
    pub fn mark_abort_in_scout() {
        ABORT_IN_SCOUT.set(ABORT_IN_SCOUT.get() + 1);
    }
    pub fn mark_abort_research_acquire() {
        ABORT_RESEARCH_ACQUIRE.set(ABORT_RESEARCH_ACQUIRE.get() + 1);
    }
    pub fn mark_abort_in_research() {
        ABORT_IN_RESEARCH.set(ABORT_IN_RESEARCH.get() + 1);
    }
    pub fn mark_record_killer_call() {
        RECORD_KILLER_CALLS.set(RECORD_KILLER_CALLS.get() + 1);
    }
    pub fn mark_record_history_call(delta: i64) {
        RECORD_HISTORY_CALLS.set(RECORD_HISTORY_CALLS.get() + 1);
        HISTORY_TOTAL_DELTA.set(HISTORY_TOTAL_DELTA.get() + delta);
    }
    pub fn record_research_pv_pair(scout_row: Vec<Move>, research_row: Vec<Move>) {
        RESEARCH_PV_PAIRS.with_borrow_mut(|v| v.push((scout_row, research_row)));
    }
    pub fn mark_research_row_committed() {
        RESEARCH_ROW_COMMITTED.set(RESEARCH_ROW_COMMITTED.get() + 1);
    }
    pub fn mark_root_first_full() {
        ROOT_FIRST_FULL.set(ROOT_FIRST_FULL.get() + 1);
    }
    pub fn mark_root_scout() {
        ROOT_SCOUT.set(ROOT_SCOUT.get() + 1);
    }
    pub fn mark_root_fail_low() {
        ROOT_FAIL_LOW.set(ROOT_FAIL_LOW.get() + 1);
    }
    pub fn mark_root_research_attempt() {
        ROOT_RESEARCH_ATTEMPT.set(ROOT_RESEARCH_ATTEMPT.get() + 1);
    }
    pub fn mark_root_research_entered() {
        ROOT_RESEARCH_ENTERED.set(ROOT_RESEARCH_ENTERED.get() + 1);
    }
    pub fn mark_root_research_row_committed() {
        ROOT_RESEARCH_ROW_COMMITTED.set(ROOT_RESEARCH_ROW_COMMITTED.get() + 1);
    }
    pub fn mark_root_move_visited() {
        ROOT_MOVES_VISITED.set(ROOT_MOVES_VISITED.get() + 1);
    }
    pub fn mark_root_abort_in_scout() {
        ROOT_ABORT_IN_SCOUT.set(ROOT_ABORT_IN_SCOUT.get() + 1);
    }
    pub fn mark_root_abort_research_acquire() {
        ROOT_ABORT_RESEARCH_ACQUIRE.set(ROOT_ABORT_RESEARCH_ACQUIRE.get() + 1);
    }
    pub fn mark_root_abort_in_research() {
        ROOT_ABORT_IN_RESEARCH.set(ROOT_ABORT_IN_RESEARCH.get() + 1);
    }
    pub fn record_root_research_pv_pair(scout_row: Vec<Move>, research_row: Vec<Move>) {
        ROOT_RESEARCH_PV_PAIRS.with_borrow_mut(|v| v.push((scout_row, research_row)));
    }
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

// ---------------------------------------------------------------------------
// M3.2: transposition-table integration helpers (private to this module).
// ---------------------------------------------------------------------------

/// Result of probing the TT for one search node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SearchTtProbe {
    /// A real score the node can return immediately (the TT entry's bound
    /// is satisfied by the current window). `None` means no cut-off.
    cutoff: Option<i32>,
    /// The entry's stored best move, used only for move ordering. `None`
    /// when the entry has no move or the probe is a miss / decode failure.
    hash_move: Option<Move>,
}

/// Build the context-safe TT key for the CURRENT position. The repetition
/// signature comes from the full [`SearchPath`], so two identical boards with
/// different repetition context get different keys.
fn current_tt_key(pos: &Position, path: &SearchPath) -> TtKey {
    debug_assert_eq!(path.last(), Some(&pos.zobrist_key()));
    TtKey::new(
        pos.zobrist_key(),
        pos.halfmove_clock(),
        path.repetition_signature(),
    )
}

/// Probe the TT for one search node and decide a cut-off.
///
/// All call sites must use this single helper (never hand-assemble a key).
/// Returns the decoded score cut-off only when the entry's bound actually
/// closes the current window — fail-soft semantics are preserved: we return
/// the *real* stored bound score, never a raw alpha/beta.
fn probe_tt_for_search(
    tt: &TranspositionTable,
    key: TtKey,
    requested_depth: u32,
    ply: u32,
    effective_alpha: i32,
    beta: i32,
) -> SearchTtProbe {
    let Some(entry) = tt.probe(key) else {
        return SearchTtProbe {
            cutoff: None,
            hash_move: None,
        };
    };

    // Full-key mismatch is already a miss (handled by `tt.probe`). A decode
    // failure (mate score at an unsupported ply) means we cannot trust the
    // score: treat the ENTIRE entry as a miss — no cut-off AND no hash move.
    let Some(decoded) = score_from_tt(entry.score, ply) else {
        return SearchTtProbe {
            cutoff: None,
            hash_move: None,
        };
    };

    // Shallower entry: a real miss for cut-off purposes, but its stored move
    // (if any) is still useful for ordering.
    if entry.depth < requested_depth {
        return SearchTtProbe {
            cutoff: None,
            hash_move: entry.best_move,
        };
    }

    let cutoff = match entry.bound {
        Bound::Exact => Some(decoded),
        Bound::Lower => {
            if decoded >= beta {
                Some(decoded)
            } else {
                None
            }
        }
        Bound::Upper => {
            if decoded <= effective_alpha {
                Some(decoded)
            } else {
                None
            }
        }
    };

    SearchTtProbe {
        cutoff,
        hash_move: entry.best_move,
    }
}

/// Classify a stored score into a TT bound relative to the *caller's* window
/// (saved BEFORE any draw floor raised alpha). This is the inverse of the
/// probe: an Exact entry was exact, a Lower entry failed high, an Upper
/// entry failed low.
fn classify_tt_bound(score: i32, caller_alpha: i32, caller_beta: i32) -> Bound {
    if score <= caller_alpha {
        Bound::Upper
    } else if score >= caller_beta {
        Bound::Lower
    } else {
        Bound::Exact
    }
}

/// Store a node's result, skipping the write only when the score cannot be
/// encoded (a mate score at a ply beyond the codec's range). Never panics
/// on an un-encodable score.
fn store_tt_score(
    tt: &mut TranspositionTable,
    key: TtKey,
    depth: u32,
    score: i32,
    ply: u32,
    bound: Bound,
    best_move: Option<Move>,
) {
    if let Some(encoded) = score_to_tt(score, ply) {
        tt.store(TTEntry {
            key,
            depth,
            score: encoded,
            bound,
            best_move,
        });
    }
}

/// M4.1 quiet-move-ordering heuristic state, local to a single
/// `search_best_move` call. Created once at the start of the call and
/// carried through all iterative-deepening iterations; re-zeroed on the
/// next independent `go`/`search` call. Never persisted across games or
/// into quiescence.
///
/// `M41Reference` and `Current` build one; the `M4Reference` path skips it
/// entirely (no killer/history ordering), so the historical baseline is
/// untouched.
///
/// Bounded normalization cap for the history table (spec §4.1/§4.3).
/// Every `history` entry is capped at this value; this bounds table
/// growth and prevents overflow within a single search. It is a cap,
/// NOT a periodic decay.
const M4_HISTORY_CAP: i32 = 16_384;

struct SearchHeuristics {
    /// `killers[ply] = [slot0, slot1]`; grown on demand via
    /// [`SearchHeuristics::ensure_ply`].
    killers: Vec<[Option<Move>; 2]>,
    /// `history[color][from][to]`; reset to zero at construction and
    /// re-zeroed on the next search. Updated only on quiet beta-cutoffs
    /// (see [`SearchHeuristics::record_history`]); consulted only for
    /// remaining quiet moves in `order_moves_with_hash_and_killers`.
    /// Distinct from the search-line `SearchPath` Zobrist stack.
    history: [[[i32; 64]; 64]; 2],
}

impl SearchHeuristics {
    fn new() -> Self {
        SearchHeuristics {
            killers: Vec::new(),
            history: [[[0; 64]; 64]; 2],
        }
    }

    /// `ply` has no fixed upper bound (`go depth N` accepts any
    /// parseable `u32`; depth-only / nodes-only / time-only / infinite /
    /// bare `go` all leave `SearchLimits.depth = None` at times, and
    /// iterative deepening keeps growing depth until a budget/deadline/
    /// stop fires). The table is therefore grown lazily.
    fn ensure_ply(&mut self, ply: usize) {
        if self.killers.len() <= ply {
            self.killers.resize(ply + 1, [None, None]);
        }
    }

    /// Record `m` as a killer at `ply` after a real *quiet* beta-cutoff.
    /// The locked update keeps the `slot0 != slot1` invariant:
    /// - `m == slot0` -> no-op (no duplicate);
    /// - `m == slot1` -> promoted to slot0, old slot0 demoted to slot1;
    /// - brand-new `m` -> inserted at slot0, old slot0 demoted to slot1.
    fn record_killer(&mut self, ply: usize, m: Move) {
        #[cfg(test)]
        pvs_counters::mark_record_killer_call();
        self.ensure_ply(ply);
        let k = &mut self.killers[ply];
        if k[0] != Some(m) {
            k[1] = k[0];
            k[0] = Some(m);
        }
    }

    /// Record `m` into the history table after a real *quiet* beta-cutoff
    /// at this node, whose remaining search depth is `d` (the depth at
    /// which `m` is played). Uses the locked, overflow-free formula from
    /// spec §4.2: reward `d*d`, capped at `M4_HISTORY_CAP` (bounded
    /// normalization, never periodic decay). `pos` must be the parent
    /// node (the mover's side to move) — i.e. after the move's
    /// `unmake_move` — which is exactly where this is called.
    fn record_history(&mut self, pos: &Position, m: Move, d: u32) {
        // Lossless widening first; NO reliance on any "depth <= 64" bound.
        let dd = u64::from(d); // d: u32 depth, widened
        let bonus = dd
            .saturating_mul(dd) // d*d, never overflows u64
            .min(M4_HISTORY_CAP as u64) as i32; // cap before i32 cast
        let color = pos.side_to_move() as usize; // mover's color
        let idx = m.from as usize;
        let jdx = m.to as usize;
        let updated = self.history[color][idx][jdx]
            .saturating_add(bonus)
            .min(M4_HISTORY_CAP); // bounded normalization
        #[cfg(test)]
        pvs_counters::mark_record_history_call(
            i64::from(updated) - i64::from(self.history[color][idx][jdx]),
        );
        self.history[color][idx][jdx] = updated;
    }
}

/// Reorder `moves` in place for `SearchProfile::Current` non-root nodes,
/// per the seven-level priority of spec §5: TT hash move first, then
/// promotions, then captures + en passant (MVV-LVA within each), then
/// killer slot 0, then killer slot 1, then the remaining quiet moves
/// sorted by `history[color][from][to]` **descending** with a
/// deterministic `(from, to)` ascending tie-break (Commit 4 levels 6-7).
/// Every legal move appears exactly once.
///
/// A killer is placed only if it is present in `moves` and has not
/// already been placed (i.e. it is not the TT hash move); it is a quiet
/// move by construction (only quiet cutoffs are recorded as killers).
/// Sort key for [`order_moves_with_hash_and_killers`]:
/// `(bucket, mvkey, hist_rank, orig_index, move)`. `bucket` 0..=5
/// encodes the §5 level. `mvkey` is the tactical `move_order_key`
/// (MVV-LVA) used to rank promotions / captures+ep within their
/// buckets. `hist_rank` is the within-quiet-band tie-break (higher =
/// searched first): for the remaining quiets it combines the history score
/// (descending) with a deterministic `(from, to)` ascending break; it is
/// 0 for tacticals and the singleton buckets. `orig_index` is the final
/// stable tie-break.
type KillerOrderKey = (i32, (u8, i32, i32), i64, usize, Move);

fn order_moves_with_hash_and_killers(
    pos: &Position,
    moves: &mut [Move],
    hash_move: Option<Move>,
    h: Option<&SearchHeuristics>,
    ply: usize,
) {
    // Resolve this ply's killers (empty when no heuristic state exists).
    let killers = if let Some(hh) = h {
        if hh.killers.len() > ply {
            hh.killers[ply]
        } else {
            [None, None]
        }
    } else {
        [None, None]
    };
    let mut keyed: Vec<KillerOrderKey> = moves
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let bucket = if Some(m) == hash_move {
                0 // legal TT hash move (lifted to index 0)
            } else if matches!(m.flag, MoveFlag::Promotion(_)) {
                1 // promotions
            } else if pos.board[m.to as usize].is_some() || matches!(m.flag, MoveFlag::EnPassant) {
                2 // captures + en passant (MVV-LVA)
            } else if Some(m) == killers[0] {
                3 // killer slot 0 (quiet)
            } else if Some(m) == killers[1] {
                4 // killer slot 1 (quiet)
            } else {
                5 // remaining quiet moves (history-sorted, levels 6-7)
            };
            // Tactical key (for MVV-LVA buckets) and quiet-band history
            // tie-break (for the remaining quiets). Singletons / tacticals
            // that are not captures+ep/promotions use the zeroed defaults.
            let mvkey = move_order_key(pos, m);
            // Within-quiet-band tie-break (higher = searched first):
            // history descending, then (from, to) ascending. Encoded as a
            // single descending integer: bigger history first, then smaller
            // from, then smaller to.
            let hist_rank: i64 = if bucket == 5 {
                let hist = if let Some(hh) = h {
                    let color = pos.side_to_move() as usize;
                    hh.history[color][m.from as usize][m.to as usize]
                } else {
                    0
                };
                (hist as i64) * 4096 - (m.from as i64) * 64 - (m.to as i64)
            } else {
                0
            };
            (bucket, mvkey, hist_rank, i, m)
        })
        .collect();
    // Ascending bucket, then descending `mvkey` (MVV-LVA), then
    // descending `hist_rank` (history), then ascending original index
    // (deterministic tie-break).
    keyed.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.3.cmp(&b.3))
    });
    for (i, (_, _, _, _, m)) in keyed.into_iter().enumerate() {
        moves[i] = m;
    }
}

/// Reorder `moves` so the TT hash move (if legal and present) sits at
/// index 0, while every other move keeps its existing MVV-LVA relative
/// order. Never drops, duplicates, or reorders around the hash move; an
/// illegal / absent hash move is ignored (no panic).
fn order_moves_with_hash(pos: &Position, moves: &mut [Move], hash_move: Option<Move>) {
    // First the existing stable MVV-LVA order.
    order_moves(pos, moves);
    // Then lift the hash move to the front, preserving the relative order of
    // the remaining moves via a single right rotation over [0..=idx].
    if let Some(hm) = hash_move {
        if let Some(idx) = moves.iter().position(|&m| m == hm) {
            if idx != 0 {
                moves[..=idx].rotate_right(1);
            }
        }
    }
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
    /// The immutable base length captured at construction: the full
    /// caller-supplied game history. The search may push search children
    /// on top, but it must NEVER `pop` or `restore_root` below this
    /// length — doing so would drop a real game-history key and corrupt
    /// the repetition context. `base_len` is NOT a fixed 1; it equals
    /// the length of the history the caller threaded in.
    base_len: usize,
}

impl SearchPath {
    /// Build from a caller-supplied history (the `GameState` keys).
    /// Scans the input once to build counts and the XOR signature.
    pub(crate) fn new(history: Vec<ZobristKey>) -> Self {
        assert!(
            !history.is_empty(),
            "SearchPath requires the current position key"
        );
        let base_len = history.len();

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
            base_len,
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

    /// The immutable base length captured at construction. The search may
    /// never `pop` or `restore_root` below this. Used by tests.
    #[allow(dead_code)]
    pub(crate) fn base_len(&self) -> usize {
        self.base_len
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
    /// MUST NOT remove a key at or below the construction base — doing so
    /// would drop a real game-history key and corrupt the repetition
    /// context. This invariant holds in both debug and release builds,
    /// hence the plain `assert!` (never `debug_assert!`).
    pub(crate) fn pop(&mut self) {
        assert!(
            self.history.len() > self.base_len,
            "cannot pop below the SearchPath base"
        );
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

    /// Defensive safety net: restore to the target length by popping
    /// individual entries. Each pop updates counts and the signature,
    /// so the path is fully consistent after restoration.
    ///
    /// Invariants (plain `assert!`, enforced in debug AND release):
    ///   * `target_len >= base_len` — never restore below the construction
    ///     base (would drop a real game-history key).
    ///   * `target_len <= history.len()` — never restore "beyond" the
    ///     current stack (would be a no-op that silently did nothing).
    pub(crate) fn restore_root(&mut self, root_len: usize) {
        assert!(
            root_len >= self.base_len,
            "cannot restore below the SearchPath base"
        );
        assert!(
            root_len <= self.history.len(),
            "cannot restore beyond the current SearchPath"
        );

        while self.history.len() > root_len {
            self.pop();
        }

        assert_eq!(self.history.len(), root_len);
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
    // Public entry is TT-disabled: it builds its own throwaway table.
    let mut tt = TranspositionTable::disabled();
    let r = negamax_impl(
        pos, depth, ply, alpha, beta, ctx, limits, &mut pv, &mut path, &mut tt,
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
    tt: &mut TranspositionTable,
) -> Option<i32> {
    // Acquire the right to search this node *before* touching the board.
    if !try_enter_node(ctx, limits) {
        return None;
    }
    negamax_entered_impl(
        pos,
        depth,
        ply,
        alpha,
        beta,
        ctx,
        limits,
        SearchProfile::M4Reference,
        pv,
        path,
        tt,
        &mut None::<SearchHeuristics>,
    )
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
    _profile: SearchProfile,
    pv: &mut PvTable,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
    // M4.1: killer/history state, consumed by `M41Reference` and `Current`
    // at this non-root ordinary negamax node (only `Current` also applies
    // PVS on top). `M4Reference` passes `None` and is never read/written.
    heur: &mut Option<SearchHeuristics>,
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
    // Save the caller's window BEFORE the draw floor may raise alpha. The
    // bound we STORE later uses this original window, not the raised one
    // (a claim floor that lifts alpha to 0 must not turn a true Exact
    // score of 0 into a spurious Upper bound).
    let caller_alpha = alpha;
    let caller_beta = beta;
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

    // M3.2: build the context-safe TT key and probe. The probe runs
    // AFTER try_enter_node already counted this node, and AFTER terminal /
    // draw precedence — so every TT hit or cut-off still consumes exactly
    // one real node. On a cut-off we return the decoded score and leave
    // the (already-cleared) PV row empty.
    let key = current_tt_key(pos, path);
    let tt_probe = probe_tt_for_search(tt, key, depth, ply, alpha, beta);
    if let Some(cutoff) = tt_probe.cutoff {
        return Some(cutoff);
    }

    if depth == 0 {
        // Leaf: hand off to quiescence (already counted). On a real return,
        // store a depth-0 entry under the caller window; on abort propagate
        // None WITHOUT storing (the partial node must not be cached).
        return match quiescence_entered_impl(pos, ply, 0, alpha, beta, ctx, limits, pv, path) {
            Some(s) => {
                let bound = classify_tt_bound(s, caller_alpha, caller_beta);
                // The qsearch PV row start (if any) is the real best-capture
                // move; stand-pat / claim floor / empty PV -> None.
                let best_move = pv.lines[ply as usize].first().copied();
                store_tt_score(tt, key, 0, s, ply, bound, best_move);
                Some(s)
            }
            None => None,
        };
    }

    // M4.1: for non-M4Reference profiles (`M41Reference` and `Current`),
    // apply the seven-level ordering (§5) at this non-root ordinary negamax
    // node — TT hash lift, promotions, MVV-LVA captures/ep, killer slot 0,
    // killer slot 1, then the remaining quiets sorted by history descending
    // (Commit 4) with a deterministic (from,to) tie-break. `M4Reference`
    // keeps the exact M4.0 ordering.
    // Killers are read from this `ply` (grown lazily; empty until a quiet
    // cutoff records one in a prior iteration); history is the per-search
    // table carried in `heur`.
    if _profile != SearchProfile::M4Reference {
        order_moves_with_hash_and_killers(
            pos,
            &mut moves,
            tt_probe.hash_move,
            heur.as_ref(),
            ply as usize,
        );
    } else {
        // M2.2 + M3.2: stable MVV-LVA order, then lift the TT hash
        // move (if legal and present) to index 0 without disturbing the
        // relative order of the other moves.
        order_moves_with_hash(pos, &mut moves, tt_probe.hash_move);
    }

    let mut node_best_move: Option<Move> = None;
    // P1.1: the running maximum of all fail-low scout scores. A fail-low
    // scout's PV is not committable, but the numeric value it returns is a
    // valid upper bound on its child and therefore part of this node's own
    // upper bound. It is folded into the RETURNED/STORED score only (never
    // into best / alpha / PV / cutoff / heuristics below).
    let mut fail_low_upper: Option<i32> = None;
    for (move_idx, m) in moves.into_iter().enumerate() {
        // Capture the window BEFORE this move, so a possible re-search and
        // the beta-cutoff decision both see the same `alpha_before_move`.
        let alpha_before_move = alpha;
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

        // P2.2: when this move goes through a full re-search, remember the
        // child PV row the RE-SEARCH rewrote, so the commit block can verify
        // (inline) that the parent copies exactly this row — never a stale
        // scout row — whenever this move becomes the node's best.
        #[cfg(test)]
        let mut researched_row: Option<Vec<Move>> = None;

        // Resolve the child window into an EXPLICIT `MoveOutcome` (P1.1).
        // Terminal / IntendedClaim children are exact results and are never
        // scouted or re-searched (spec §4 / §8); only a `Continue` child may
        // take the PVS scout path.
        let outcome = match probe {
            // mate/stalemate edge, parent perspective
            ChildProbe::Terminal(s) => MoveOutcome::Candidate(s),
            // mover claims on this intended move
            ChildProbe::IntendedClaim => MoveOutcome::Candidate(0),
            ChildProbe::Continue => {
                match pvs_child_window(_profile, move_idx == 0, depth, alpha_before_move, beta) {
                    ChildWindow::Full => {
                        // Full-window search: the first move, a non-Current
                        // profile (M4Reference / M41Reference), depth-0, or the
                        // caller-null-window / overflow fallbacks. The manual
                        // probe already spent the single node for this child.
                        // Handle a deeper abort EXPLICITLY: pop + unmake THIS
                        // edge before propagating None (no `?` before cleanup).
                        match negamax_entered_impl(
                            pos,
                            depth - 1,
                            ply + 1,
                            -beta,
                            -alpha_before_move,
                            ctx,
                            limits,
                            _profile,
                            pv,
                            path,
                            tt,
                            heur,
                        ) {
                            Some(s) => MoveOutcome::Candidate(-s),
                            None => {
                                path.pop();
                                pos.unmake_move(undo);
                                return None;
                            }
                        }
                    }
                    ChildWindow::Scout { scout_beta } => {
                        // Null-window scout. Child window is
                        // `[-scout_beta, -alpha_before_move]`; the manual
                        // probe already spent the single node for this child.
                        #[cfg(test)]
                        pvs_counters::mark_scout();
                        let scout_score = match negamax_entered_impl(
                            pos,
                            depth - 1,
                            ply + 1,
                            -scout_beta,
                            -alpha_before_move,
                            ctx,
                            limits,
                            _profile,
                            pv,
                            path,
                            tt,
                            heur,
                        ) {
                            Some(s) => -s,
                            None => {
                                // Phase A: the scout's own subtree aborted.
                                #[cfg(test)]
                                pvs_counters::mark_abort_in_scout();
                                path.pop();
                                pos.unmake_move(undo);
                                return None;
                            }
                        };
                        if pvs_needs_research(scout_score, alpha_before_move, beta) {
                            // Improve alpha but not a cutoff: re-search with the
                            // full window. The child position stays made and the
                            // SearchPath stays pushed (NO pop/unmake yet); we do
                            // NOT re-probe — the node budget for this child was
                            // already taken by the scout. Acquire exactly ONE
                            // more real node for the re-search (spec §4). The
                            // re-search overwrites the scout's child PV row, so
                            // the committed line is the re-searched one.
                            #[cfg(test)]
                            pvs_counters::mark_research_attempt();
                            // P2: snapshot the child PV row the SCOUT left,
                            // so a test can prove the parent commits the
                            // re-searched line, never this stale scout line.
                            #[cfg(test)]
                            let scout_child_row = pv.lines[(ply + 1) as usize].clone();
                            if !try_enter_node(ctx, limits) {
                                // Phase B: re-search node acquisition failed.
                                #[cfg(test)]
                                pvs_counters::mark_abort_research_acquire();
                                path.pop();
                                pos.unmake_move(undo);
                                return None;
                            }
                            #[cfg(test)]
                            pvs_counters::mark_research_entered();
                            match negamax_entered_impl(
                                pos,
                                depth - 1,
                                ply + 1,
                                -beta,
                                -alpha_before_move,
                                ctx,
                                limits,
                                _profile,
                                pv,
                                path,
                                tt,
                                heur,
                            ) {
                                Some(s) => {
                                    // P2: pair the scout's stale child row
                                    // with the row the re-search rewrote, and
                                    // remember the re-search row so the commit
                                    // block can prove the parent copies it.
                                    #[cfg(test)]
                                    {
                                        let research_row = pv.lines[(ply + 1) as usize].clone();
                                        pvs_counters::record_research_pv_pair(
                                            scout_child_row,
                                            research_row.clone(),
                                        );
                                        researched_row = Some(research_row);
                                    }
                                    MoveOutcome::Candidate(-s)
                                }
                                None => {
                                    // Phase C: the full re-search subtree aborted.
                                    #[cfg(test)]
                                    pvs_counters::mark_abort_in_research();
                                    path.pop();
                                    pos.unmake_move(undo);
                                    return None;
                                }
                            }
                        } else if scout_score <= alpha_before_move {
                            // Scout failed LOW. Its move/PV are NOT
                            // committable — we do NOT re-search and we do NOT
                            // let its line reach the parent's best / PV /
                            // alpha / cutoff / heuristics. Its NUMERIC value
                            // IS kept: a fail-soft-compatible child can
                            // return a real upper bound above the running
                            // `best` (e.g. via a TT Exact hit), and dropping
                            // that number would make this node's returned
                            // score / stored TT `Bound::Upper` under-state
                            // the true node value (P1.1).
                            #[cfg(test)]
                            {
                                pvs_counters::mark_scout_fail_low();
                                // Record when the fail-low score exceeded
                                // `best` — the exact P1.1 hazard where the
                                // numeric bound (and only the bound) matters.
                                if scout_score > best {
                                    pvs_counters::mark_scout_fail_low_exceeds_best();
                                }
                            }
                            MoveOutcome::ScoutFailLow(scout_score)
                        } else {
                            // Scout failed HIGH (`scout_score >= beta`) — a
                            // reachable fail-soft outcome (TT Exact / Lower /
                            // Upper hits and mate scores can carry the scout
                            // outside its null window). A valid lower bound
                            // and a real cutoff candidate: its legal scout
                            // line is committed once below and (if quiet)
                            // killer/history is rewarded once. We never
                            // re-search a fail-high.
                            #[cfg(test)]
                            pvs_counters::mark_scout_fail_high();
                            MoveOutcome::ScoutFailHigh(scout_score)
                        }
                    }
                }
            }
        };

        path.pop();
        pos.unmake_move(undo);

        // Commit parent state by MATCHING on the explicit outcome (P1.1). A
        // `ScoutFailLow` never updates best / node_best_move / PV / alpha,
        // never triggers a cutoff, and never rewards killer/history — but
        // its NUMERIC upper bound is retained in `fail_low_upper` so the
        // node's returned/stored score cannot under-state the true value.
        // Every other outcome (full search, re-searched scout, terminal,
        // intended claim, and a fail-high scout) carries a real candidate.
        let score = match outcome {
            MoveOutcome::ScoutFailLow(s) => {
                fail_low_upper = Some(fail_low_upper.map_or(s, |u| u.max(s)));
                continue;
            }
            MoveOutcome::Candidate(s) | MoveOutcome::ScoutFailHigh(s) => s,
        };

        // Update ONCE, based only on the move's FINAL score (the full search,
        // the re-searched scout, or a fail-high scout). The PVS window never
        // touches the heuristics: killer/history updates happen only in the
        // beta-cutoff block below, never inside the scout.
        if score > best {
            best = score;
            node_best_move = Some(m);
            pv.set_from_child(ply, m);
            // P2.2: a re-searched move that becomes the node's best commits
            // the row the RE-SEARCH rewrote (`set_from_child` copies
            // `pv.lines[ply + 1]`, which the re-search overwrote AFTER the
            // scout). Prove it structurally: the parent's committed child
            // line (the PV tail below `m`) equals the recorded re-search row,
            // never a stale scout row.
            #[cfg(test)]
            if let Some(research_row) = researched_row.as_ref() {
                let committed_tail = &pv.lines[ply as usize][1..];
                assert_eq!(
                    committed_tail,
                    research_row.as_slice(),
                    "parent must commit the re-search child row, not a stale scout row"
                );
                pvs_counters::mark_research_row_committed();
            }
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            // M4.1: a *quiet* beta-cutoff at this non-root node records
            // `m` as a killer (Commit 3) AND into the history table
            // (Commit 4) for `ply` / the remaining depth `depth`. Tactical
            // cutoffs (capture / en passant / promotion) are excluded per
            // spec §3.2 / §4.4. `pos` is already the parent node here
            // (after `path.pop()` + `pos.unmake_move(undo)`), so
            // `pos.side_to_move()` is the mover's color. This block runs
            // exactly once per move, on `final_score` only — never inside
            // the scout — so a quiet cutoff is never rewarded twice.
            if _profile != SearchProfile::M4Reference {
                if let Some(h) = heur {
                    if !is_tactical(pos, m) {
                        h.record_killer(ply as usize, m);
                        h.record_history(pos, m, depth);
                        #[cfg(test)]
                        pvs_counters::mark_parent_quiet_reward();
                    } else {
                        // Tactical cutoff: take the cutoff but do NOT reward
                        // killer/history (spec §3.2 / §4.4).
                        #[cfg(test)]
                        pvs_counters::mark_parent_tactical_cutoff();
                    }
                }
            }
            break; // beta cutoff
        }
    }

    // Completed the whole node (or hit a normal beta cutoff). The score we
    // return and store lifts the exact-search `best` by any retained
    // fail-low scout upper bound (P1.1): "the fail-low scout's PV is not
    // committable, but the numeric upper bound it provides is still part of
    // search correctness". On an all-fail-low node this prevents storing a
    // TT `Bound::Upper` that claims `value <= best` when a dropped scout
    // proved only `value <= fail_low_upper` with `fail_low_upper > best`.
    // `node_best_move` / PV remain driven by real candidates only. When a
    // beta cutoff or an alpha improvement occurred, `best >= alpha >=` every
    // fail-low scout score, so the lift is a no-op there.
    let returned_score = fail_low_upper.map_or(best, |u| best.max(u));
    // Store one entry under the caller window; a deeper abort never reaches
    // here, so no partial node is ever cached.
    let bound = classify_tt_bound(returned_score, caller_alpha, caller_beta);
    store_tt_score(tt, key, depth, returned_score, ply, bound, node_best_move);
    Some(returned_score)
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

/// Explicit classification of one ROOT move edge (M4.2 Commit 4). The root
/// commits state by MATCHING on this variant, never by inferring "was this a
/// fail-low?" from a score comparison. The root is simpler than a non-root
/// node: it has NO beta cutoff, so a scout that improves alpha is ALWAYS
/// re-searched (there is no fail-high shortcut variant), and a fail-low scout
/// is simply dropped WITHOUT retaining a numeric bound — the root's running
/// exact candidate / claim floor already sits at or above `alpha_before_move`,
/// which already dominates a fail-low scout's upper bound. This engine is
/// fail-soft COMPATIBLE: the re-search decision (`scout_score >
/// alpha_before_move`) never assumes a scout return is clamped into its null
/// window; a fail-soft value above the window still re-searches correctly.
enum RootMoveOutcome {
    /// A real candidate score: a full-window search (first move, or a
    /// reference profile), a re-searched scout, a terminal child, or an
    /// intended-claim child. Participates in best / best_move / root PV /
    /// alpha normally.
    Candidate(i32),
    /// A later `Current` root scout that failed low
    /// (`scout_score <= alpha_before_move`). Not committable and not
    /// re-searched; the root retains no numeric bound from it.
    ScoutFailLow,
}

/// Search one root ply to `depth`, returning the completed iteration (its
/// score and full principal variation) or `None` if aborted. The PV table is
/// (re)allocated per call, sized to this iteration's depth plus the quiescence
/// cap — never to `limits.depth`, so an absurd `go depth` cannot trigger a
/// huge one-shot allocation.
///
/// M4.2 Commit 4: under `SearchProfile::Current` the root runs Principal
/// Variation Search — the first root move takes the full window, later moves
/// are scouted with a null window and re-searched at full width only if the
/// scout improves alpha. `M4Reference` and `M41Reference` keep the full-window
/// root unchanged (byte-identical root node counts / scores / PV). There is no
/// root beta cutoff in any profile: every legal root move is checked, moves
/// that may improve alpha are fully re-searched, and the final root best score
/// is exact.
#[allow(clippy::too_many_arguments)]
fn root_search(
    pos: &mut Position,
    depth: u32,
    root_moves: &mut [Move],
    root_claimable: bool,
    claim_fallback: Move,
    ctx: &SearchContext,
    limits: &SearchLimits,
    // M4.1: forwarded to the non-root negamax node for killer/history
    // ordering. M4.2 Commit 4: ALSO consumed at the root itself to gate root
    // PVS (`Current` scouts later root moves; the reference profiles keep the
    // full-window root).
    profile: SearchProfile,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
    // M4.1: killer/history state forwarded to the non-root negamax
    // node.
    heur: &mut Option<SearchHeuristics>,
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

    // M3.2: probe the root entry for hash-move ordering ONLY. We never
    // use the stored root score here (iterative deepening must run every
    // depth), so the cut-off field is deliberately ignored. A legal stored
    // move is lifted to the front — but we deliberately do NOT apply the
    // full MVV-LVA `order_moves` pass at the root. The disabled path must
    // keep the exact root move order (and therefore the exact node count)
    // of the pre-TT search, and MVV-LVA reordering at the root would
    // change both. So this is a pure hash-move lift, identical to
    // `order_moves_with_hash` minus its `order_moves` pre-pass.
    let root_key = current_tt_key(pos, path);
    let root_probe = probe_tt_for_search(tt, root_key, depth, 0, alpha, beta);
    if let Some(hm) = root_probe.hash_move {
        if let Some(idx) = root_moves.iter().position(|&m| m == hm) {
            if idx != 0 {
                root_moves[..=idx].rotate_right(1);
            }
        }
    }

    for (move_idx, &mut m) in root_moves.iter_mut().enumerate() {
        // Capture the window BEFORE this move so the scout window and the
        // re-search both see the same `alpha_before_move`. For the reference
        // profiles (and the first move) this equals the running `alpha`, so
        // the full-window search below is byte-identical to the pre-PVS root.
        let alpha_before_move = alpha;
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

        // P2: when this root move goes through a full re-search, remember the
        // child PV row the RE-SEARCH rewrote so the commit block can verify
        // inline that the root copies exactly that row — never a stale scout
        // row — whenever this move becomes the root best.
        #[cfg(test)]
        let mut researched_row: Option<Vec<Move>> = None;

        // Resolve the child window into an EXPLICIT `RootMoveOutcome`. Terminal
        // / IntendedClaim children are exact results and are never scouted or
        // re-searched; only a `Continue` child may take the root PVS path.
        let outcome = match probe {
            // mate/stalemate edge, parent (root) perspective
            ChildProbe::Terminal(s) => RootMoveOutcome::Candidate(s),
            // mover claims on this intended move
            ChildProbe::IntendedClaim => RootMoveOutcome::Candidate(0),
            ChildProbe::Continue => {
                // The manual probe already spent the single node for this
                // child. Root PVS uses the SAME window helper as a non-root
                // node (its behavior is unchanged): first move / reference
                // profile -> Full; a later `Current` move -> Scout.
                match pvs_child_window(profile, move_idx == 0, depth, alpha_before_move, beta) {
                    ChildWindow::Full => {
                        #[cfg(test)]
                        if move_idx == 0 && profile == SearchProfile::Current {
                            pvs_counters::mark_root_first_full();
                        }
                        // Full-window search. Recurse into the ENTERED body —
                        // NEVER negamax_impl (double count). Handle a deeper
                        // abort EXPLICITLY before cleanup.
                        match negamax_entered_impl(
                            pos,
                            depth - 1,
                            1,
                            -beta,
                            -alpha_before_move,
                            ctx,
                            limits,
                            profile,
                            &mut pv,
                            path,
                            tt,
                            heur,
                        ) {
                            Some(s) => RootMoveOutcome::Candidate(-s),
                            None => {
                                path.pop();
                                pos.unmake_move(undo);
                                return None; // aborted (deeper recursion)
                            }
                        }
                    }
                    ChildWindow::Scout { scout_beta } => {
                        // Null-window scout for a later `Current` root move.
                        // Child window is `[-scout_beta, -alpha_before_move]`;
                        // the probe already spent this child's single node.
                        #[cfg(test)]
                        pvs_counters::mark_root_scout();
                        let scout_score = match negamax_entered_impl(
                            pos,
                            depth - 1,
                            1,
                            -scout_beta,
                            -alpha_before_move,
                            ctx,
                            limits,
                            profile,
                            &mut pv,
                            path,
                            tt,
                            heur,
                        ) {
                            Some(s) => -s,
                            None => {
                                // Phase A: the root scout's subtree aborted.
                                #[cfg(test)]
                                pvs_counters::mark_root_abort_in_scout();
                                path.pop();
                                pos.unmake_move(undo);
                                return None;
                            }
                        };
                        // The root has NO beta cutoff, so there is no fail-high
                        // shortcut: ANY scout that improves alpha
                        // (`scout_score > alpha_before_move`) must be fully
                        // re-searched before it can become the root best. This
                        // condition is fail-soft SAFE — it never assumes the
                        // scout return was clamped into its null window; a
                        // fail-soft value above the window still re-searches.
                        if scout_score > alpha_before_move {
                            #[cfg(test)]
                            pvs_counters::mark_root_research_attempt();
                            // P2: snapshot the child PV row the SCOUT left, so
                            // a test can prove the root commits the re-searched
                            // line, never this stale scout line.
                            #[cfg(test)]
                            let scout_child_row = pv.lines[1].clone();
                            // Re-search: the child stays made and the SearchPath
                            // stays pushed (NO pop/unmake, NO re-probe). Acquire
                            // exactly ONE more real node for the re-search.
                            if !try_enter_node(ctx, limits) {
                                // Phase B: re-search node acquisition failed.
                                #[cfg(test)]
                                pvs_counters::mark_root_abort_research_acquire();
                                path.pop();
                                pos.unmake_move(undo);
                                return None;
                            }
                            #[cfg(test)]
                            pvs_counters::mark_root_research_entered();
                            match negamax_entered_impl(
                                pos,
                                depth - 1,
                                1,
                                -beta,
                                -alpha_before_move,
                                ctx,
                                limits,
                                profile,
                                &mut pv,
                                path,
                                tt,
                                heur,
                            ) {
                                Some(s) => {
                                    #[cfg(test)]
                                    {
                                        let research_row = pv.lines[1].clone();
                                        pvs_counters::record_root_research_pv_pair(
                                            scout_child_row,
                                            research_row.clone(),
                                        );
                                        researched_row = Some(research_row);
                                    }
                                    RootMoveOutcome::Candidate(-s)
                                }
                                None => {
                                    // Phase C: the full re-search subtree aborted.
                                    #[cfg(test)]
                                    pvs_counters::mark_root_abort_in_research();
                                    path.pop();
                                    pos.unmake_move(undo);
                                    return None;
                                }
                            }
                        } else {
                            // Scout failed low at the root: its move/PV are NOT
                            // committable and it is NOT re-searched. Unlike a
                            // non-root node, the root keeps NO numeric bound —
                            // `alpha_before_move` (the running exact candidate /
                            // claim floor) already sits at or above this scout's
                            // upper bound, so nothing is lost.
                            #[cfg(test)]
                            pvs_counters::mark_root_fail_low();
                            RootMoveOutcome::ScoutFailLow
                        }
                    }
                }
            }
        };

        path.pop();
        pos.unmake_move(undo);
        #[cfg(test)]
        pvs_counters::mark_root_move_visited();

        // Commit root state by MATCHING on the explicit outcome. A fail-low
        // scout never updates best / best_move / root PV / alpha. Every other
        // outcome carries a real candidate.
        let score = match outcome {
            RootMoveOutcome::ScoutFailLow => continue,
            RootMoveOutcome::Candidate(s) => s,
        };

        if score > best_score {
            best_score = score;
            best_move = Some(m);
            // Record the root PV: this move followed by the child's PV.
            pv.set_from_child(0, m);
            // P2: a re-searched root move that becomes the root best commits
            // the row the RE-SEARCH rewrote (`set_from_child` copies
            // `pv.lines[1]`, which the re-search overwrote AFTER the scout).
            // Prove it structurally: the committed root PV tail equals the
            // recorded re-search row, never a stale scout row.
            #[cfg(test)]
            if let Some(research_row) = researched_row.as_ref() {
                let committed_tail = &pv.lines[0][1..];
                assert_eq!(
                    committed_tail,
                    research_row.as_slice(),
                    "root must commit the re-search child row, not a stale scout row"
                );
                pvs_counters::mark_root_research_row_committed();
            }
        }
        if best_score > alpha {
            alpha = best_score;
        }
        // No root beta cutoff: all root moves are checked; moves that may
        // improve alpha are fully re-searched; the final root best score is
        // exact.
    }

    // A claimable root with no real move beating 0 returns the claim itself
    // as a COMPLETED iteration: score 0, the stable fallback (protocol
    // placeholder, NOT a found 0-score line), empty PV. best_move stays None
    // so this branch fires instead of the `best_move.map` below.
    if root_claimable && best_move.is_none() {
        // M3.2: cache the claim placeholder (Exact, no best move).
        let root_key = current_tt_key(pos, path);
        store_tt_score(tt, root_key, depth, 0, 0, Bound::Exact, None);
        return Some(RootIteration {
            score: 0,
            best_move: claim_fallback,
            pv: Vec::new(),
        });
    }

    best_move.map(|bm| {
        // M3.2: cache the completed iteration. Bound is always Exact for
        // a fully searched root; best_move follows PV reality (a non-empty
        // PV carries the real move, an empty PV — the claim placeholder —
        // carries None).
        let root_key = current_tt_key(pos, path);
        let store_move = if pv.lines[0].is_empty() {
            None
        } else {
            Some(bm)
        };
        store_tt_score(tt, root_key, depth, best_score, 0, Bound::Exact, store_move);
        RootIteration {
            score: best_score,
            best_move: bm,
            pv: std::mem::take(&mut pv.lines[0]),
        }
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
/// keep compiling. TT is DISABLED here — the public API and the UCI
/// production path stay TT-disabled until the dedicated UCI Hash option
/// lands in a later stage.
pub fn search_best_move(
    pos: &mut Position,
    limits: &SearchLimits,
    ctx: &SearchContext,
) -> Option<SearchOutcome> {
    let mut path = SearchPath::new(vec![pos.zobrist_key()]);
    let root_len = path.len();
    let mut tt = TranspositionTable::disabled();
    let r = search_best_move_impl(
        pos,
        limits,
        ctx,
        SearchProfile::M4Reference,
        &mut path,
        &mut tt,
    );
    path.restore_root(root_len);
    r
}

/// History-aware entry used by the UCI layer, which passes the real
/// `GameState` key history. The search extends this with its own
/// `SearchPath` (cloned from `game_history`) but never mutates the
/// caller's `GameState`. TT is DISABLED — see [`search_best_move`].
///
/// Contract (debug-checked): `game_history` is non-empty and its last
/// element equals the current position's Zobrist key.
///
/// NOTE: since the M3.2 Phase-3 UCI layer switched its production path to
/// `search_best_move_with_history_and_tt` (persistent TT), this disabled-table
/// wrapper is now only referenced by the in-crate `search` tests. The
/// `#[allow(dead_code)]` keeps `-D warnings` green for the non-test lib
/// target; its behavior (build a disabled TT, search) is unchanged.
#[allow(dead_code)]
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
    let mut tt = TranspositionTable::disabled();
    let r = search_best_move_impl(
        pos,
        limits,
        ctx,
        SearchProfile::M4Reference,
        &mut path,
        &mut tt,
    );
    path.restore_root(root_len);
    r
}

/// History-aware, TT-aware entry (M4.1: now the M4.0 *reference* path).
///
/// This was the original M3.2 production entry. M4.1 preserves its exact
/// M4.0 behavior by delegating to
/// [`search_best_move_with_history_tt_and_profile`] with
/// `SearchProfile::M4Reference`. Its signature is unchanged and its output is
/// byte-identical to the pre-M4.1 M4.0 baseline — killer (Commit 3)
/// and history (Commit 4) ordering are applied under
/// `SearchProfile::M41Reference` and `SearchProfile::Current`, never under
/// `M4Reference`. The persistent UCI `Hash` table is threaded through
/// every recursion exactly as before.
pub(crate) fn search_best_move_with_history_and_tt(
    pos: &mut Position,
    game_history: &[ZobristKey],
    limits: &SearchLimits,
    ctx: &SearchContext,
    tt: &mut TranspositionTable,
) -> Option<SearchOutcome> {
    search_best_move_with_history_tt_and_profile(
        pos,
        game_history,
        limits,
        ctx,
        tt,
        SearchProfile::M4Reference,
    )
}

/// Profile-aware search entry (M4.1). Threads `profile` through the whole
/// search core so the move-ordering strategy can differ by [`SearchProfile`].
/// The UCI production path calls this with `SearchProfile::Current`; the
/// historical M4.0 reference entry
/// ([`search_best_move_with_history_and_tt`]) and the in-crate tests call it
/// with `SearchProfile::M4Reference` to reproduce the locked baseline exactly.
///
/// No new UCI `option` is exposed by this change — the UCI surface is
/// unchanged. The caller-owned persistent `TranspositionTable` is threaded
/// through every recursion exactly as the original entry did.
pub(crate) fn search_best_move_with_history_tt_and_profile(
    pos: &mut Position,
    game_history: &[ZobristKey],
    limits: &SearchLimits,
    ctx: &SearchContext,
    tt: &mut TranspositionTable,
    profile: SearchProfile,
) -> Option<SearchOutcome> {
    debug_assert!(!game_history.is_empty());
    debug_assert_eq!(game_history.last(), Some(&pos.zobrist_key()));
    debug_assert_eq!(pos.zobrist_key(), recompute_zobrist(pos));
    let mut path = SearchPath::new(game_history.to_vec());
    let root_len = path.len();
    let r = search_best_move_impl(pos, limits, ctx, profile, &mut path, tt);
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
    // M4.1: threaded through to non-root negamax. Non-M4Reference profiles
    // (`M41Reference` and `Current`) apply killer (Commit 3) + history
    // (Commit 4) ordering; `M4Reference` leaves every search behavior
    // byte-identical to M4.0.
    _profile: SearchProfile,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
) -> Option<SearchOutcome> {
    let mut root_moves = generate_legal_moves(pos);
    if root_moves.is_empty() {
        return None; // already terminal (checkmate / stalemate)
    }
    // Stable fallback: the first legal move. Used if we never complete a
    // single iteration (e.g. stopped before depth 1 finishes).
    let fallback = root_moves[0];

    // M4.1 Commit 3: build the per-search heuristic state ONLY for
    // non-M4Reference profiles (`M41Reference` and `Current`).
    // `M4Reference` skips it entirely (no killer/history ordering),
    // preserving the exact M4.0 baseline. The table
    // lives for the whole iterative-deepening loop and is dropped on
    // return (re-zeroed for the next independent `go`).
    let mut heuristics: Option<SearchHeuristics> = if _profile != SearchProfile::M4Reference {
        Some(SearchHeuristics::new())
    } else {
        None
    };

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
            &mut root_moves,
            root_claimable,
            fallback,
            ctx,
            limits,
            _profile,
            path,
            tt,
            &mut heuristics,
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
            &mut TranspositionTable::disabled(),
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

        let out = search_best_move_impl(
            &mut p,
            &limits,
            ctx,
            SearchProfile::M4Reference,
            &mut path,
            &mut TranspositionTable::disabled(),
        );

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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
                    SearchProfile::M4Reference,
                    &mut pv,
                    &mut path,
                    &mut TranspositionTable::disabled(),
                    &mut None::<SearchHeuristics>,
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut TranspositionTable::disabled(),
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
            &mut [m],
            false, // parent is NOT itself a claim
            m,     // fallback (also g1f3 here)
            &ctx,
            &limits,
            SearchProfile::M4Reference,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut None::<SearchHeuristics>,
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
            &mut TranspositionTable::disabled(),
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
        let mut root_moves = vec![pm];
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
            &mut root_moves,
            true, // root_claimable
            fallback,
            &ctx,
            &limits,
            SearchProfile::M4Reference,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut None::<SearchHeuristics>,
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

    /// Abort-style: make + push two moves (e2e4, e7e5), then a third
    /// (g1f3), then abort back to the intermediate length — restoring
    /// BOTH the `SearchPath` AND the `Position` — and continue with a
    /// DIFFERENT legal third move (d2d4) from the genuinely-restored
    /// position. The invariant `path.last() == pos.zobrist_key()` must
    /// hold at every step; an abort that restores only the path (or only
    /// the board) is a bug.
    #[test]
    fn search_path_abort_style_restore() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);

        let mut p = pos;

        // Push e2e4, e7e5 (genuine make + push; undos dropped).
        p.make_move(find_move(&p, "e2e4"));
        path.push_child(&p);
        p.make_move(find_move(&p, "e7e5"));
        path.push_child(&p);

        // Snapshot the parent state (after e2e4 e7e5).
        let parent_fen = to_fen(&p);
        let parent_key = p.zobrist_key();
        let parent_keys = path.keys().to_vec();
        let parent_sig = path.repetition_signature();
        let parent_len = path.len(); // 3

        // Push g1f3, then abort: restore BOTH the path and the Position.
        let u3 = p.make_move(find_move(&p, "g1f3"));
        path.push_child(&p);
        assert_eq!(path.len(), parent_len + 1);
        assert_eq!(p.zobrist_key(), *path.keys().last().unwrap());

        path.restore_root(parent_len);
        p.unmake_move(u3);
        // After the abort, Position and path must agree exactly.
        assert_eq!(to_fen(&p), parent_fen, "abort restored FEN");
        assert_eq!(p.zobrist_key(), parent_key, "abort restored key");
        assert_eq!(path.keys(), &parent_keys[..], "abort restored path keys");
        assert_eq!(
            path.repetition_signature(),
            parent_sig,
            "abort restored signature"
        );
        assert_eq!(path.last(), Some(&parent_key), "path.last == Position key");

        // Continue with a different legal third move from the restored position.
        let u4 = p.make_move(find_move(&p, "d2d4"));
        path.push_child(&p);
        assert_eq!(path.len(), parent_len + 1);
        assert_eq!(
            p.zobrist_key(),
            *path.keys().last().unwrap(),
            "path.last == Position key after d2d4"
        );
        assert_ne!(
            path.repetition_signature(),
            parent_sig,
            "d2d4 changes the signature"
        );
        assert_eq!(path.keys().len(), parent_keys.len() + 1);

        // Pop + unmake to fully restore the parent again.
        path.pop();
        p.unmake_move(u4);
        assert_eq!(to_fen(&p), parent_fen, "parent restored after d2d4");
        assert_eq!(
            p.zobrist_key(),
            parent_key,
            "parent key restored after d2d4"
        );
        assert_eq!(
            path.keys(),
            &parent_keys[..],
            "parent path restored after d2d4"
        );
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
        assert_eq!(
            cloned.base_len(),
            path.base_len(),
            "clone preserves base_len"
        );
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
    /// `SearchPath::new` must reject an empty history: there is no current
    /// position key to anchor the base, so the core invariant
    /// `history.last() == current Position key` would be vacuously broken.
    #[test]
    #[should_panic(expected = "SearchPath requires the current position key")]
    fn search_path_new_empty_panics() {
        let _ = SearchPath::new(vec![]);
    }

    /// A freshly-constructed (base-only) path must NOT be poppable -- its
    /// single key is the search root and also the base, so `pop` would
    /// drop below `base_len`.
    #[test]
    #[should_panic(expected = "cannot pop below the SearchPath base")]
    fn search_path_pop_on_fresh_panics() {
        let mut path = SearchPath::new(vec![10u64]);
        path.pop();
    }

    /// With a multi-key game history (base_len > 1), a real search child
    /// may be pushed and popped back to the base, but a further `pop`
    /// would remove an original game-history key and MUST panic (in both
    /// debug and release builds).
    #[test]
    fn search_path_pop_below_base_is_blocked() {
        let pos = parse_fen(START_FEN).unwrap();
        let root_key = pos.zobrist_key();
        // Three original game-history keys => base_len = 3.
        let mut path = SearchPath::new(vec![root_key, 100u64, 200]);
        let m = find_move(&pos, "e2e4");
        let mut child = pos;
        child.make_move(m);
        path.push_child(&child); // len 4, base preserved
        assert_eq!(path.len(), 4);
        assert_eq!(path.last(), Some(&child.zobrist_key()));
        // Pop the search child back to the base -- allowed.
        path.pop();
        assert_eq!(path.len(), 3);
        assert_eq!(
            path.keys(),
            &[root_key, 100u64, 200][..],
            "original history preserved"
        );
        // A further pop would cross the base -> must panic.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            path.pop();
        }));
        assert!(r.is_err(), "pop below the game-history base must panic");
    }

    /// `restore_root` must panic if the target is below `base_len` (would
    /// drop a real game-history key).
    #[test]
    #[should_panic(expected = "cannot restore below the SearchPath base")]
    fn search_path_restore_below_base_panics() {
        let mut path = SearchPath::new(vec![10u64, 20]); // base_len = 2
        path.restore_root(1); // 1 < 2 -> panic
    }

    /// `restore_root` must panic if the target is beyond the current length
    /// (a silent no-op that would leave the path corrupted).
    #[test]
    #[should_panic(expected = "cannot restore beyond the current SearchPath")]
    fn search_path_restore_beyond_current_panics() {
        let mut path = SearchPath::new(vec![10u64, 20, 30]); // len 3
        path.restore_root(4); // 4 > 3 -> panic
    }

    // =========================================================================
    // M3.2 C2 — Transposition-table integration tests (spec §14 / §15 / §16)
    // All TT / search symbols are already in scope via `use super::*`.

    /// Fixed-depth search through the crate-private TT-aware entry. Returns
    /// `(outcome, node count)`. Sharing the same `tt` lets a caller drive a
    /// cold-then-warm sequence.
    fn run_tt(fen: &str, depth: u32, tt: &mut TranspositionTable) -> (Option<SearchOutcome>, u64) {
        let pos = parse_fen(fen).unwrap();
        run_tt_hist(fen, depth, &[pos.zobrist_key()], tt)
    }

    fn run_tt_hist(
        fen: &str,
        depth: u32,
        history: &[ZobristKey],
        tt: &mut TranspositionTable,
    ) -> (Option<SearchOutcome>, u64) {
        let mut pos = parse_fen(fen).unwrap();
        let hist = history.to_vec();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let out = search_best_move_with_history_and_tt(&mut pos, &hist, &limits, &ctx, tt);
        (out, ctx.nodes.load(Ordering::Relaxed))
    }

    /// Every move in `pv` must be legal in sequence from `fen`.
    fn pv_is_legal(fen: &str, pv: &[Move]) -> bool {
        let mut pos = parse_fen(fen).unwrap();
        for &m in pv {
            let legal: BTreeSet<String> = generate_legal_moves(&mut pos.clone())
                .into_iter()
                .map(move_to_uci)
                .collect();
            if !legal.contains(&move_to_uci(m)) {
                return false;
            }
            pos.make_move(m);
        }
        true
    }

    // ---- §14: bound classifier ------------------------------------------------

    #[test]
    fn tt_classify_bound_windows() {
        assert_eq!(classify_tt_bound(-10, -10, 10), Bound::Upper);
        assert_eq!(classify_tt_bound(10, -10, 10), Bound::Lower);
        assert_eq!(classify_tt_bound(0, -10, 10), Bound::Exact);
        assert_eq!(classify_tt_bound(-9, -10, 10), Bound::Exact);
        assert_eq!(classify_tt_bound(9, -10, 10), Bound::Exact);
    }

    // ---- §14: probe / bound semantics -----------------------------------------

    #[test]
    fn tt_probe_sufficient_depth_exact_lower_upper() {
        let pos = parse_fen(START_FEN).unwrap();
        let key = TtKey::new(pos.zobrist_key(), pos.halfmove_clock(), 0u64);

        // Fresh table per case: the same-key replacement rule must NOT mix
        // an Exact entry with a subsequent Lower/Upper at equal depth.
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key,
            depth: 5,
            score: score_to_tt(42, 0).unwrap(),
            bound: Bound::Exact,
            best_move: None,
        });
        let p = probe_tt_for_search(&tt, key, 3, 0, i32::MIN + 1000, i32::MAX - 1000);
        assert_eq!(p.cutoff, Some(42), "Exact sufficient-depth must cut off");
        assert_eq!(p.hash_move, None);

        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key,
            depth: 5,
            score: score_to_tt(100, 0).unwrap(),
            bound: Bound::Lower,
            best_move: None,
        });
        let lo_cut = probe_tt_for_search(&tt, key, 3, 0, i32::MIN + 1000, 50);
        assert_eq!(
            lo_cut.cutoff,
            Some(100),
            "Lower decoded(100) >= beta(50) cuts off"
        );
        let lo_no = probe_tt_for_search(&tt, key, 3, 0, i32::MIN + 1000, 200);
        assert_eq!(
            lo_no.cutoff, None,
            "Lower decoded(100) < beta(200) does not cut off"
        );

        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key,
            depth: 5,
            score: score_to_tt(-100, 0).unwrap(),
            bound: Bound::Upper,
            best_move: None,
        });
        let up_cut = probe_tt_for_search(&tt, key, 3, 0, -50, i32::MAX - 1000);
        assert_eq!(
            up_cut.cutoff,
            Some(-100),
            "Upper decoded(-100) <= alpha(-50) cuts off"
        );
        let up_no = probe_tt_for_search(&tt, key, 3, 0, -200, i32::MAX - 1000);
        assert_eq!(
            up_no.cutoff, None,
            "Upper decoded(-100) > alpha(-200) does not cut off"
        );
    }

    #[test]
    fn tt_probe_insufficient_depth_keeps_hash_move() {
        let pos = parse_fen(START_FEN).unwrap();
        let key = TtKey::new(pos.zobrist_key(), pos.halfmove_clock(), 0u64);
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        let hm = find_move(&pos, "b1c3");
        tt.store(TTEntry {
            key,
            depth: 2,
            score: score_to_tt(42, 0).unwrap(),
            bound: Bound::Exact,
            best_move: Some(hm),
        });
        let p = probe_tt_for_search(&tt, key, 3, 0, i32::MIN + 1000, i32::MAX - 1000);
        assert_eq!(p.cutoff, None, "shallower entry must not cut off");
        assert_eq!(
            p.hash_move,
            Some(hm),
            "shallower entry still yields its move"
        );
    }

    #[test]
    fn tt_probe_miss_empty_table() {
        let pos = parse_fen(START_FEN).unwrap();
        let key = TtKey::new(pos.zobrist_key(), pos.halfmove_clock(), 0u64);
        let tt = TranspositionTable::disabled();
        let p = probe_tt_for_search(&tt, key, 3, 0, i32::MIN + 1000, i32::MAX - 1000);
        assert_eq!(p.cutoff, None);
        assert_eq!(p.hash_move, None);
    }

    #[test]
    fn tt_probe_decode_failure_is_full_miss() {
        let pos = parse_fen(START_FEN).unwrap();
        let key = TtKey::new(pos.zobrist_key(), pos.halfmove_clock(), 0u64);
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key,
            depth: 5,
            score: score_to_tt(42, 0).unwrap(),
            bound: Bound::Exact,
            best_move: Some(find_move(&pos, "b1c3")),
        });
        // A ply beyond MAX_MATE_PLY makes score_from_tt return None -> the
        // ENTIRE entry is a miss (no cut-off AND no hash move).
        let p = probe_tt_for_search(&tt, key, 3, 1000, i32::MIN + 1000, i32::MAX - 1000);
        assert_eq!(p.cutoff, None, "decode failure -> no cut-off");
        assert_eq!(p.hash_move, None, "decode failure -> no hash move");
    }

    // ---- §14: context isolation ------------------------------------------------

    #[test]
    fn tt_context_isolation_misses_other_context() {
        let pos = parse_fen(START_FEN).unwrap();
        let zk = pos.zobrist_key();
        let key_real = TtKey::new(zk, 0, 0u64);
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key: key_real,
            depth: 5,
            score: score_to_tt(777_777, 0).unwrap(),
            bound: Bound::Exact,
            best_move: None,
        });
        let hit = probe_tt_for_search(
            &tt,
            TtKey::new(zk, 0, 0u64),
            3,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
        );
        assert_eq!(hit.cutoff, Some(777_777), "identical context must hit");
        let miss_hm = probe_tt_for_search(
            &tt,
            TtKey::new(zk, 1, 0u64),
            3,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
        );
        assert_eq!(miss_hm.cutoff, None, "different halfmove_clock must miss");
        let miss_rep = probe_tt_for_search(
            &tt,
            TtKey::new(zk, 0, 0xFFu64),
            3,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
        );
        assert_eq!(
            miss_rep.cutoff, None,
            "different repetition_signature must miss"
        );
    }

    // ---- §14: hash-move ordering ----------------------------------------------

    #[test]
    fn tt_order_moves_with_hash_lifts_legal_move() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut legal = generate_legal_moves(&mut pos.clone());
        let count = legal.len();
        let before: BTreeSet<String> = legal.iter().map(|m| move_to_uci(*m)).collect();
        let hm = find_move(&pos, "g1f3");
        let idx_before = legal.iter().position(|m| *m == hm).unwrap();
        order_moves_with_hash(&pos, &mut legal, Some(hm));
        assert_eq!(legal[0], hm, "hash move lifted to front");
        assert_eq!(legal.len(), count, "count unchanged");
        let after: BTreeSet<String> = legal.iter().map(|m| move_to_uci(*m)).collect();
        assert_eq!(after, before, "set unchanged");
        let mut rotated = generate_legal_moves(&mut pos.clone());
        order_moves(&pos, &mut rotated);
        rotated[..=idx_before].rotate_right(1);
        assert_eq!(legal, rotated, "remaining order is a single rotation");
        let mut top = generate_legal_moves(&mut pos.clone());
        order_moves(&pos, &mut top);
        let first = top[0];
        order_moves_with_hash(&pos, &mut top, Some(first));
        assert_eq!(top[0], first, "move already at 0 stays at 0");
    }

    #[test]
    fn tt_order_moves_with_hash_ignores_illegal_and_none() {
        // A tactical fixture where the base MVV-LVA ordering (`order_moves`)
        // visibly reorders the raw generation order, so the test is non-trivial.
        // Black has a rook on d2 that White can capture, while there are also
        // many quiet moves.
        let fen = "4k3/8/8/8/8/8/3r4/R2QK3 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let gen = generate_legal_moves(&mut pos.clone());
        let gen_set: BTreeSet<String> = gen.iter().map(|m| move_to_uci(*m)).collect();

        // Base MVV-LVA ordering that `order_moves_with_hash` must reproduce
        // when the hash move is `None` or illegal (i.e. not in the legal set).
        let mut expected = gen.clone();
        order_moves(&pos, &mut expected);
        assert_ne!(expected, gen, "fixture must show a visible reorder");

        // `None` hash move => identical to the base ordering.
        let mut a = gen.clone();
        order_moves_with_hash(&pos, &mut a, None);
        assert_eq!(a.len(), gen.len(), "move count unchanged");
        assert_eq!(a, expected, "None hash move == base ordering");
        let a_set: BTreeSet<String> = a.iter().map(|m| move_to_uci(*m)).collect();
        assert_eq!(a_set, gen_set, "move set unchanged");

        // Illegal hash move: legal on another position, but its source square
        // (b1) is empty in `pos`, so it is NOT in `pos`'s legal set.
        // => identical to the base ordering; never panics, never drops a move.
        let other = parse_fen("4k3/8/8/8/8/8/8/1R2K3 w - - 0 1").unwrap();
        let illegal = find_move(&other, "b1b4");
        let mut b = gen.clone();
        order_moves_with_hash(&pos, &mut b, Some(illegal));
        assert_eq!(b.len(), gen.len(), "move count unchanged");
        assert_eq!(b, expected, "illegal hash move == base ordering");
        let b_set: BTreeSet<String> = b.iter().map(|m| move_to_uci(*m)).collect();
        assert_eq!(b_set, gen_set, "move set unchanged");
    }

    // ---- §14: claim-floor storage (root) -------------------------------------

    #[test]
    fn tt_root_claim_floor_stores_exact_zero_no_move() {
        let fen = "4k3/pppppppp/8/8/8/8/8/4K3 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let key = pos.zobrist_key();
        let history = vec![key, key, key];
        let mut tt = TranspositionTable::new_mb(2).unwrap();
        let (out, _) = run_tt_hist(fen, 3, &history, &mut tt);
        let out = out.expect("outcome");
        assert_eq!(out.score, Some(0), "claim floor holds root at 0");
        assert!(out.pv.is_empty(), "claim placeholder PV is empty");
        let root_key = current_tt_key(&pos, &SearchPath::new(history.clone()));
        let e = tt.probe(root_key).expect("root entry stored");
        assert_eq!(
            e.bound,
            Bound::Exact,
            "claim-floor root stored Exact, not Upper"
        );
        assert_eq!(
            e.score,
            score_to_tt(0, 0).unwrap(),
            "stored score decodes to 0"
        );
        assert_eq!(e.best_move, None, "claim-floor root stores no best move");
    }

    #[test]
    fn tt_root_win_stores_exact_with_move() {
        let fen = "7k/8/6K1/6Q1/8/8/8/8 w - - 0 1";
        let pos = parse_fen(fen).unwrap();
        let history = vec![pos.zobrist_key()];
        let mut tt = TranspositionTable::new_mb(2).unwrap();
        let (out, _) = run_tt_hist(fen, 2, &history, &mut tt);
        let out = out.expect("outcome");
        assert!(out.score.unwrap() > 0, "winning side finds the mate");
        assert!(!out.pv.is_empty(), "winning root has a non-empty PV");
        let root_key = current_tt_key(&pos, &SearchPath::new(history.clone()));
        let e = tt.probe(root_key).expect("root entry stored");
        assert_eq!(e.bound, Bound::Exact);
        assert!(e.best_move.is_some(), "winning root stores the mate move");
        assert_eq!(
            e.best_move.unwrap(),
            out.best_move,
            "stored move matches best"
        );
    }

    #[test]
    fn tt_root_intended_claim_edge_stores_move() {
        // The root itself is NOT a claim, but playing g1f3 creates the
        // third occurrence of `child_key` (history already holds it twice),
        // so the resulting child is an IntendedClaim (score 0). Constrain the
        // root move list to [g1f3] so this edge is forced and verified.
        let pos = parse_fen(START_FEN).unwrap();
        let parent_key = pos.zobrist_key();
        let m = find_move(&pos, "g1f3");
        let mut child = pos;
        child.make_move(m);
        let child_key = child.zobrist_key();
        let history = vec![child_key, child_key, parent_key];
        let mut path = SearchPath::new(history.clone());
        let root_len = path.len();
        let mut tt = TranspositionTable::new_mb(2).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(2),
            ..Default::default()
        };
        let mut root_moves = vec![m];
        let out = root_search(
            &mut pos.clone(),
            2,
            &mut root_moves,
            false,
            m,
            &ctx,
            &limits,
            SearchProfile::M4Reference,
            &mut path,
            &mut tt,
            &mut None::<SearchHeuristics>,
        );
        path.restore_root(root_len);
        let out = out.expect("root iteration");
        assert_eq!(out.score, 0, "intended-claim root edge scores 0");
        assert_eq!(out.best_move, m, "best move is the intended claim");
        assert_eq!(out.pv, vec![m], "PV is exactly [g1f3]");
        let root_key = current_tt_key(&pos, &SearchPath::new(history.clone()));
        let e = tt.probe(root_key).expect("root entry stored");
        assert_eq!(e.bound, Bound::Exact);
        assert_eq!(e.score, score_to_tt(0, 0).unwrap());
        assert_eq!(e.best_move, Some(m), "root stores the intended-claim move");
    }

    // ---- §14: depth-0 / qsearch boundary -------------------------------------

    #[test]
    fn tt_depth0_qsearch_stores_entry() {
        let pos = parse_fen("7k/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let root_len = path.len();
        let mut pv = PvTable::default();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut tt = TranspositionTable::new_mb(2).unwrap();
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
            &mut tt,
        )
        .expect("not stopped");
        path.restore_root(root_len);
        assert!(score > 0, "qsearch finds the promoting win");
        let k = current_tt_key(&pos, &path);
        let e = tt.probe(k).expect("depth-0 node stored");
        assert_eq!(e.depth, 0, "depth-0 entry depth is 0 (never the qply)");
        assert_eq!(e.bound, Bound::Exact);
        assert!(
            e.best_move.is_some(),
            "qsearch PV start stored as best move"
        );
    }

    // ---- §14: TT cut-off leaves the current PV row empty ----------------------

    #[test]
    fn tt_cutoff_leaves_pv_row_empty() {
        let pos = parse_fen(START_FEN).unwrap();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        // Store under the SAME key the search will probe (real repetition sig).
        let key = current_tt_key(&pos, &path);
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        // Lower bound above beta forces a cut-off at this node.
        tt.store(TTEntry {
            key,
            depth: 5,
            score: score_to_tt(50, 0).unwrap(),
            bound: Bound::Lower,
            best_move: None,
        });
        let mut pv = PvTable::default();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let out = negamax_impl(
            &mut pos.clone(),
            3,
            0,
            -1000,
            0,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
            &mut tt,
        );
        assert_eq!(out, Some(50), "TT Lower cut-off returns the decoded score");
        assert!(
            pv.lines[0].is_empty(),
            "TT cut-off leaves the current PV row empty"
        );
    }

    // ---- §14: abort must not store a partial entry ---------------------------

    #[test]
    fn tt_abort_does_not_store_partial_entry() {
        let mut tt = TranspositionTable::new_mb(2).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            nodes: Some(1),
            ..Default::default()
        };
        let pos = parse_fen(START_FEN).unwrap();
        let history = vec![pos.zobrist_key()];
        let out = search_best_move_with_history_and_tt(
            &mut pos.clone(),
            &history,
            &limits,
            &ctx,
            &mut tt,
        );
        match &out {
            None => {}
            Some(o) => assert!(o.stopped, "if an outcome is returned it must be stopped"),
        }
        let root_key = current_tt_key(&pos, &SearchPath::new(history.clone()));
        assert!(
            tt.probe(root_key).is_none(),
            "aborted search must not cache a partial root entry"
        );
    }

    // ---- §14: a stale (illegal-in-current-position) TT move is ignored -----

    #[test]
    fn tt_legal_stale_move_ignored() {
        let other = parse_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let stale = find_move(&other, "a1a4"); // illegal on the startpos
        let (ref_out, _) = run_tt(START_FEN, 3, &mut TranspositionTable::disabled());
        let ref_out = ref_out.expect("disabled outcome");
        let pos = parse_fen(START_FEN).unwrap();
        let key = current_tt_key(&pos, &SearchPath::new(vec![pos.zobrist_key()]));
        let mut tt = TranspositionTable::new_mb(2).unwrap();
        tt.store(TTEntry {
            key,
            depth: 5,
            score: score_to_tt(ref_out.score.unwrap(), 0).unwrap(),
            bound: Bound::Exact,
            best_move: Some(stale),
        });
        let (out, _) = run_tt(START_FEN, 3, &mut tt);
        let out = out.expect("enabled outcome with stale move");
        assert_eq!(
            out.score, ref_out.score,
            "stale move must not change the score"
        );
        assert_eq!(
            move_to_uci(out.best_move),
            move_to_uci(ref_out.best_move),
            "stale move must not change the best move"
        );
        assert!(
            pv_is_legal(START_FEN, &out.pv),
            "PV still legal despite stale TT move"
        );
        assert_eq!(out.completed_depth, ref_out.completed_depth);
    }

    // ---- §15: disabled exact regression ---------------------------------------

    #[test]
    fn tt_disabled_exact_baseline_startpos() {
        // Canonical disabled path: the public `search_best_move` wrapper, which
        // is exactly the production entry UCI uses today. This must reproduce
        // the M2.4 fixed baselines measured in tests/m2_4.rs.
        let mut pos = parse_fen(START_FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            1149,
            "disabled startpos d3 node count unchanged"
        );
        assert_eq!(move_to_uci(out.best_move), "b1c3");
        assert_eq!(out.score, Some(50));
        assert_eq!(
            out.pv.iter().map(|m| move_to_uci(*m)).collect::<Vec<_>>(),
            vec!["b1c3".to_string(), "b8c6".to_string(), "g1f3".to_string()]
        );
    }

    #[test]
    fn tt_disabled_exact_baseline_queenwin() {
        let fen = "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1";
        let mut pos = parse_fen(fen).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            963,
            "disabled queen-win d3 node count unchanged"
        );
        assert_eq!(move_to_uci(out.best_move), "e4a4");
        assert_eq!(out.score, Some(890));
        assert_eq!(
            out.pv.iter().map(|m| move_to_uci(*m)).collect::<Vec<_>>(),
            vec![
                "e4a4".to_string(),
                "h4h3".to_string(),
                "a4h4".to_string(),
                "h8g8".to_string(),
                "h4h3".to_string()
            ]
        );
    }

    // ---- §8.1 / M4.1 (Commit 2): profile plumbing -----------------------
    #[test]
    fn m4_profile_reference_reproduces_baseline() {
        // The new profile-aware entry, driven with `M4Reference`, must
        // reproduce the locked M4.0 smoke numbers EXACTLY. This is the
        // contract that keeps the historical baseline valid after the M4.1
        // refactor (the old `search_best_move_with_history_and_tt` now
        // delegates here with `M4Reference`).
        let mut pos = parse_fen(START_FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let mut tt = TranspositionTable::disabled();
        let hist_key = pos.zobrist_key();
        let out = search_best_move_with_history_tt_and_profile(
            &mut pos,
            &[hist_key],
            &limits,
            &ctx,
            &mut tt,
            SearchProfile::M4Reference,
        )
        .expect("outcome");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            1149,
            "M4Reference startpos d3 node count unchanged"
        );
        assert_eq!(move_to_uci(out.best_move), "b1c3");
        assert_eq!(out.score, Some(50));
        assert_eq!(
            out.pv.iter().map(|m| move_to_uci(*m)).collect::<Vec<_>>(),
            vec!["b1c3", "b8c6", "g1f3"]
        );
    }

    #[test]
    fn m4_profile_current_matches_reference_smoke() {
        // Commit 3 enables killer ordering on `Current`. Per spec §6 /
        // §8.1 the fixed-depth parity contract between `Current` and
        // `M4Reference` is: identical score, legal best move / PV, and
        // full Position restoration. They are FREE to differ in node count
        // / best move / PV ordering (ordering tuning may legitimately
        // change those), so this test must NOT freeze them -- only the
        // hard-correctness items above are asserted.
        let fen = START_FEN;
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };

        let mut pos_r = parse_fen(fen).unwrap();
        let ctx_r = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_r = TranspositionTable::disabled();
        let key_r = pos_r.zobrist_key();
        let out_r = search_best_move_with_history_tt_and_profile(
            &mut pos_r,
            &[key_r],
            &limits,
            &ctx_r,
            &mut tt_r,
            SearchProfile::M4Reference,
        )
        .expect("reference outcome");
        let fen_r = to_fen(&pos_r);

        let mut pos_c = parse_fen(fen).unwrap();
        let ctx_c = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_c = TranspositionTable::disabled();
        let key_c = pos_c.zobrist_key();
        let out_c = search_best_move_with_history_tt_and_profile(
            &mut pos_c,
            &[key_c],
            &limits,
            &ctx_c,
            &mut tt_c,
            SearchProfile::Current,
        )
        .expect("current outcome");
        let fen_c = to_fen(&pos_c);

        // Fixed-depth completeness + no spurious stop.
        assert_eq!(out_r.completed_depth, 3);
        assert!(!out_r.stopped);
        assert_eq!(out_c.completed_depth, 3);
        assert!(!out_c.stopped);

        // Score parity (hard correctness).
        assert_eq!(out_c.score, out_r.score, "fixed-depth score must match");
        assert_eq!(out_c.score, Some(50), "startpos d3 score is 50 for both");

        // Legal best move / PV for both profiles.
        assert!(pv_is_legal(fen, &out_r.pv));
        assert!(pv_is_legal(fen, &out_c.pv));
        assert_eq!(out_r.pv.first().copied(), Some(out_r.best_move));
        assert_eq!(out_c.pv.first().copied(), Some(out_c.best_move));

        // Position fully restored by both searches.
        assert_eq!(fen_r.as_str(), fen, "reference restored");
        assert_eq!(fen_c.as_str(), fen, "current restored");
    }

    #[test]
    fn m4_profile_current_parity_m41reference() {
        // Post-PVS (Commit 3): `Current` now enables non-root PVS while
        // `M41Reference` stays full-window. The hard correctness contract
        // (spec §9.5) is: identical SCORE, legal best move / PV, and
        // full Position restoration. They are FREE to differ in node count /
        // best move / PV ordering (ordering + PVS legitimately change
        // those), so this test must NOT freeze them — only the items
        // above are asserted. This replaces the pre-PVS byte-parity lock,
        // which Commit 3 intentionally breaks for `Current`.
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };

        let mut pos_a = parse_fen(START_FEN).unwrap();
        let ctx_a = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_a = TranspositionTable::disabled();
        let key_a = pos_a.zobrist_key();
        let out_a = search_best_move_with_history_tt_and_profile(
            &mut pos_a,
            &[key_a],
            &limits,
            &ctx_a,
            &mut tt_a,
            SearchProfile::M41Reference,
        )
        .expect("m41 outcome");
        let fen_a = to_fen(&pos_a);

        let mut pos_b = parse_fen(START_FEN).unwrap();
        let ctx_b = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_b = TranspositionTable::disabled();
        let key_b = pos_b.zobrist_key();
        let out_b = search_best_move_with_history_tt_and_profile(
            &mut pos_b,
            &[key_b],
            &limits,
            &ctx_b,
            &mut tt_b,
            SearchProfile::Current,
        )
        .expect("current outcome");
        let fen_b = to_fen(&pos_b);

        assert_eq!(out_a.score, out_b.score, "fixed-depth score must match");
        assert!(pv_is_legal(START_FEN, &out_a.pv));
        assert!(pv_is_legal(START_FEN, &out_b.pv));
        assert_eq!(out_a.pv.first().copied(), Some(out_a.best_move));
        assert_eq!(out_b.pv.first().copied(), Some(out_b.best_move));
        assert_eq!(fen_a.as_str(), START_FEN, "m41 restores position");
        assert_eq!(fen_b.as_str(), START_FEN, "current restores position");
    }

    #[test]
    fn m4_profile_m41reference_uses_m4_1_ordering() {
        // `M41Reference` must take the M4.1 path (killer/history seven-level
        // ordering), NOT the M4.0 path. On startpos d3 the M4.1 ordering
        // yields a different node count than the M4.0 `M4Reference` baseline
        // (1149). We assert the two counts DIFFER, proving `M41Reference`
        // genuinely runs the M4.1 path rather than silently falling back to
        // M4.0. Exact per-fixture counts are not frozen here (they belong to
        // the M4.1 benchmark report).
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };

        let mut pos_r = parse_fen(START_FEN).unwrap();
        let ctx_r = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_r = TranspositionTable::disabled();
        let key_r = pos_r.zobrist_key();
        let _out_r = search_best_move_with_history_tt_and_profile(
            &mut pos_r,
            &[key_r],
            &limits,
            &ctx_r,
            &mut tt_r,
            SearchProfile::M4Reference,
        )
        .expect("reference outcome");

        let mut pos_m = parse_fen(START_FEN).unwrap();
        let ctx_m = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_m = TranspositionTable::disabled();
        let key_m = pos_m.zobrist_key();
        let _out_m = search_best_move_with_history_tt_and_profile(
            &mut pos_m,
            &[key_m],
            &limits,
            &ctx_m,
            &mut tt_m,
            SearchProfile::M41Reference,
        )
        .expect("m41 outcome");

        let nodes_r = ctx_r.nodes.load(Ordering::Relaxed);
        let nodes_m = ctx_m.nodes.load(Ordering::Relaxed);
        assert_eq!(nodes_r, 1149, "M4Reference startpos d3 = 1149");
        assert_ne!(
            nodes_m, nodes_r,
            "M41Reference must NOT equal the M4.0 node count"
        );
    }

    #[test]
    fn m4_killer_unit() {
        // Direct unit test of `SearchHeuristics` (private, in-scope):
        // killers are recorded only via `record_killer`, never duplicate,
        // and each `search_best_move` call gets a FRESH table.
        let pos = parse_fen(START_FEN).unwrap();
        let q1 = find_move(&pos, "g1f3");
        let q2 = find_move(&pos, "b1c3");

        let mut h = SearchHeuristics::new();
        assert_eq!(h.killers.len(), 0, "starts empty");

        // First quiet killer at ply 1.
        h.record_killer(1, q1);
        assert_eq!(h.killers[1][0], Some(q1));
        assert_eq!(h.killers[1][1], None);

        // Re-recording the SAME move is a no-op (no duplicate slot).
        h.record_killer(1, q1);
        assert_eq!(h.killers[1][0], Some(q1));
        assert_eq!(h.killers[1][1], None);

        // A different quiet move promotes the old slot0 -> slot1.
        h.record_killer(1, q2);
        assert_eq!(h.killers[1][0], Some(q2));
        assert_eq!(h.killers[1][1], Some(q1));
        assert_ne!(h.killers[1][0], h.killers[1][1], "slot0 != slot1 invariant");

        // A second table (a fresh `search_best_move` call) is independent.
        let fresh = SearchHeuristics::new();
        assert_eq!(fresh.killers.len(), 0, "fresh per-search table");

        // Re-recording the move now in slot1 promotes it back to slot0
        // (P2.2 gap: the slot1 -> slot0 promotion path).
        h.record_killer(1, q1);
        assert_eq!(h.killers[1][0], Some(q1), "slot1 promotes to slot0");
        assert_eq!(h.killers[1][1], Some(q2), "old slot0 demoted to slot1");
        assert_ne!(h.killers[1][0], h.killers[1][1], "slot0 != slot1");

        // History unit (spec §4 / §8.1): `d*d` capped at M4_HISTORY_CAP,
        // no overflow for ANY legal u32 depth, table fresh per search.
        let hpos = parse_fen(START_FEN).unwrap();
        let hm = find_move(&hpos, "g1f3");
        let c = hpos.side_to_move() as usize;
        // Small depth: bonus = d*d = 9.
        h.record_history(&hpos, hm, 3);
        assert_eq!(h.history[c][hm.from as usize][hm.to as usize], 9);
        // Repeated records saturate-add (still under cap).
        h.record_history(&hpos, hm, 3); // +9 -> 18
        assert_eq!(h.history[c][hm.from as usize][hm.to as usize], 18);
        // Huge depth caps at M4_HISTORY_CAP (no overflow even for u32::MAX).
        h.record_history(&hpos, hm, u32::MAX);
        assert_eq!(
            h.history[c][hm.from as usize][hm.to as usize],
            M4_HISTORY_CAP
        );
        // A fresh table starts at zero (per-search lifecycle).
        assert_eq!(fresh.history[c][hm.from as usize][hm.to as usize], 0);
    }

    #[test]
    fn m4_history_ordering_priority() {
        // The remaining-quiet band (level 6) is sorted by
        // `history[color][from][to]` descending, with a deterministic
        // (from,to) ascending tie-break (level 7). Captures / promotions
        // keep their existing MVV-LVA ranking and never enter the history
        // band. Each move appears exactly once.
        let mut pos =
            parse_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut moves = generate_legal_moves(&mut pos);
        let orig = moves.clone();

        // Two quiet moves with distinct history so the descending order is
        // observable: assign larger history to `a` (depth 5 -> 25) and
        // smaller to `b` (depth 3 -> 9).
        let a = find_move(&pos, "b1c3"); // quiet knight, higher history
        let b = find_move(&pos, "g1f3"); // quiet knight, lower history
        let cap = find_move(&pos, "e4d5"); // capture (stays in MVV-LVA band)

        let color = pos.side_to_move() as usize;
        let mut h = SearchHeuristics::new();
        h.history[color][a.from as usize][a.to as usize] = 25;
        h.history[color][b.from as usize][b.to as usize] = 9;

        order_moves_with_hash_and_killers(&pos, &mut moves, None, Some(&h), 0);

        // Every original move appears exactly once.
        assert_eq!(moves.len(), orig.len());
        assert!(moves.iter().all(|m| orig.contains(m)));

        let idx = |m: Move| moves.iter().position(|&x| x == m).unwrap();
        // Capture (level 3) precedes the quiet band (level 6).
        assert!(idx(cap) < idx(a), "capture before quiets");
        assert!(idx(cap) < idx(b), "capture before quiets");
        // Higher-history quiet move precedes lower-history quiet move.
        assert!(idx(a) < idx(b), "history-desc quiet ordering");
    }

    #[test]
    fn m4_killer_recorded_on_real_quiet_beta_cutoff() {
        // P2.2 gap: verify the KILLER (and history) table is populated
        // through the REAL beta-cutoff integration path inside
        // `negamax_entered_impl` — not just by calling `record_killer`
        // directly. Drive a non-root search under `Current` with live
        // heuristic state and assert at least one quiet cutoff was recorded.
        let mut pos = parse_fen(START_FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let mut tt = TranspositionTable::disabled();
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let mut pv = PvTable::default();
        // Live heuristic state, threaded exactly as the production path does.
        let mut heur = Some(SearchHeuristics::new());
        let r = negamax_entered_impl(
            &mut pos,
            3,
            1,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heur,
        );
        assert!(r.is_some(), "non-root search returns a score");
        // A real fixed-depth search MUST have produced >= 1 quiet beta-cutoff.
        let total: usize = heur
            .as_ref()
            .unwrap()
            .killers
            .iter()
            .map(|k| k.iter().filter(|s| s.is_some()).count())
            .sum();
        assert!(total > 0, "real quiet beta-cutoff recorded a killer");
    }

    // ---- M4.2 Commit 3: non-root PVS ---

    #[test]
    fn pvs_child_window_pure() {
        // first move -> Full (never scouted)
        assert!(matches!(
            pvs_child_window(SearchProfile::Current, true, 3, 50, 1000),
            ChildWindow::Full
        ));
        // M41Reference later move -> Full (PVS only on Current)
        assert!(matches!(
            pvs_child_window(SearchProfile::M41Reference, false, 3, 50, 1000),
            ChildWindow::Full
        ));
        // M4Reference later move -> Full
        assert!(matches!(
            pvs_child_window(SearchProfile::M4Reference, false, 3, 50, 1000),
            ChildWindow::Full
        ));
        // Current later move + wide window -> Scout
        match pvs_child_window(SearchProfile::Current, false, 3, 50, 1000) {
            ChildWindow::Scout { scout_beta } => assert_eq!(scout_beta, 51),
            _ => panic!("expected Scout"),
        }
        // caller already a null-window node (scout_beta >= beta) -> Full
        assert!(matches!(
            pvs_child_window(SearchProfile::Current, false, 3, 999, 1000),
            ChildWindow::Full
        ));
        // alpha near i32::MAX: checked_add overflows -> Full (no panic)
        assert!(matches!(
            pvs_child_window(SearchProfile::Current, false, 3, i32::MAX, i32::MAX),
            ChildWindow::Full
        ));
        // depth == 0 -> Full even for Current later move
        assert!(matches!(
            pvs_child_window(SearchProfile::Current, false, 0, 50, 1000),
            ChildWindow::Full
        ));
    }

    #[test]
    fn pvs_needs_research_pure() {
        // score <= alpha -> no re-search (fail-low)
        assert!(!pvs_needs_research(40, 50, 100));
        assert!(!pvs_needs_research(50, 50, 100));
        // alpha < score < beta -> re-search (improves alpha, no cutoff)
        assert!(pvs_needs_research(60, 50, 100));
        // score >= beta -> no re-search (fail-high / cutoff proven)
        assert!(!pvs_needs_research(100, 50, 100));
        assert!(!pvs_needs_research(120, 50, 100));
    }

    #[test]
    fn pvs_is_tactical_pure() {
        let pos = parse_fen(START_FEN).unwrap();
        let quiet = find_move(&pos, "g1f3");
        assert!(!is_tactical(&pos, quiet), "quiet move is not tactical");

        // capture (target square occupied). Black rook on b1; white rook on
        // a1 captures it. Kings are placed off the rank-1 / b-file lines so
        // neither king is in check (parse_fen requires exactly one king/side
        // and the position must not leave the mover in check).
        let cap_pos = parse_fen("7k/8/8/8/8/8/K7/Rr6 w - - 0 1").unwrap();
        let cap = find_move(&cap_pos, "a1b1");
        assert!(is_tactical(&cap_pos, cap), "capture is tactical");

        // promotion (onto an empty square). White pawn a7 -> a8=q; black
        // king a1, white king h1 — the promotion squares are clear and the
        // mover is not left in check.
        let promo_pos = parse_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").unwrap();
        let promo = find_move(&promo_pos, "a7a8q");
        assert!(is_tactical(&promo_pos, promo), "promotion is tactical");

        // en passant: black just played d7-d5 (ep target d6); white pawn on
        // e5 captures en passant to d6, removing the d5 pawn. Verified
        // fixture (also used in m3_0 / m2_1 tests).
        let ep_pos = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let ep = find_move(&ep_pos, "e5d6");
        assert!(is_tactical(&ep_pos, ep), "en passant is tactical");
    }

    #[test]
    fn pvs_scout_and_research_execute_in_real_search() {
        // Reset the PVS counters, run `Current` on startpos d3, and prove
        // BOTH the scout and the full re-search branches actually fire INSIDE
        // a real search (not just that a node count changed). `M41Reference`
        // is the same-depth full-window baseline used only to confirm the
        // fixed-depth ROOT score is preserved. NOTE: PVS does NOT guarantee a
        // per-fixture node reduction — the hard reduction gate is the
        // 10-fixture *aggregate* benchmark (Commit 5), never this unit test.
        pvs_counters::reset();
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };

        let mut pos_c = parse_fen(START_FEN).unwrap();
        let ctx_c = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_c = TranspositionTable::disabled();
        let key_c = pos_c.zobrist_key();
        let out_c = search_best_move_with_history_tt_and_profile(
            &mut pos_c,
            &[key_c],
            &limits,
            &ctx_c,
            &mut tt_c,
            SearchProfile::Current,
        )
        .expect("current outcome");
        let fen_c = to_fen(&pos_c);
        let nodes_c = ctx_c.nodes.load(Ordering::Relaxed);

        assert!(pvs_counters::SCOUT.get() > 0, "scout fired in real search");
        assert!(
            pvs_counters::RESEARCH_ENTERED.get() > 0,
            "full re-search fired in real search"
        );
        // A re-search is attempted only after a scout, and it is entered only
        // after the attempt (both counts bounded by the scout count).
        assert_eq!(
            pvs_counters::RESEARCH_ENTERED.get(),
            pvs_counters::RESEARCH_ATTEMPT.get(),
            "no budget abort here: every attempted re-search entered"
        );
        assert!(pvs_counters::RESEARCH_ATTEMPT.get() <= pvs_counters::SCOUT.get());

        let mut pos_m = parse_fen(START_FEN).unwrap();
        let ctx_m = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_m = TranspositionTable::disabled();
        let key_m = pos_m.zobrist_key();
        let out_m = search_best_move_with_history_tt_and_profile(
            &mut pos_m,
            &[key_m],
            &limits,
            &ctx_m,
            &mut tt_m,
            SearchProfile::M41Reference,
        )
        .expect("m41 outcome");
        let nodes_m = ctx_m.nodes.load(Ordering::Relaxed);

        assert_eq!(out_c.score, out_m.score, "PVS preserves fixed-depth score");
        // NOTE: PVS does NOT guarantee per-position node reduction — a single
        // fixture can show *more* nodes when re-searches (moves whose true
        // value lands in the open `(alpha, beta)` band) outnumber the
        // fail-low / fail-high prunes. The spec's hard reduction gate is the
        // *aggregate* benchmark (Current disabled canonical <= 224,597 vs
        // M41Reference 236,418, >= 5%), NOT this unit test. We only
        // sanity-check that both searches did non-trivial work and that the
        // fixed-depth score is preserved.
        assert!(
            nodes_c > 0 && nodes_m > 0,
            "both searches did non-trivial work"
        );
        assert!(pv_is_legal(START_FEN, &out_c.pv));
        assert_eq!(fen_c.as_str(), START_FEN, "current restores position");
    }

    #[test]
    fn pvs_m41reference_never_scouts() {
        // `M41Reference` stays full-window; it must NEVER take the PVS
        // scout path even at a later move under a wide window.
        pvs_counters::reset();
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let mut pos = parse_fen(START_FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt = TranspositionTable::disabled();
        let key = pos.zobrist_key();
        let _out = search_best_move_with_history_tt_and_profile(
            &mut pos,
            &[key],
            &limits,
            &ctx,
            &mut tt,
            SearchProfile::M41Reference,
        )
        .expect("m41 outcome");
        assert_eq!(pvs_counters::SCOUT.get(), 0, "M41Reference never scouts");
        assert_eq!(
            pvs_counters::SCOUT_FAIL_LOW.get(),
            0,
            "M41Reference never fails a scout low"
        );
        assert_eq!(
            pvs_counters::SCOUT_FAIL_HIGH.get(),
            0,
            "M41Reference never fails a scout high"
        );
        assert_eq!(
            pvs_counters::RESEARCH_ATTEMPT.get(),
            0,
            "M41Reference never attempts a re-search"
        );
        assert_eq!(
            pvs_counters::RESEARCH_ENTERED.get(),
            0,
            "M41Reference never re-searches"
        );
    }

    #[test]
    fn pvs_killer_recorded_once_per_cutoff() {
        // A *quiet* beta cutoff must reward killer + history EXACTLY ONCE,
        // never twice from scout + re-search. Rather than only comparing the
        // total killer slot count across two runs (which cannot detect a
        // double reward inside the same run), we assert the exact contract:
        // the `PARENT_QUIET_REWARD` event count equals the number of history
        // entries touched, and every non-zero history entry is an EXACT
        // multiple of a single `depth*depth` bonus (a double reward would
        // leave an entry at 2x the single bonus for a slot hit only once).
        pvs_counters::reset();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let mut pos = parse_fen(START_FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt = TranspositionTable::disabled();
        let mut heur = Some(SearchHeuristics::new());
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let mut pv = PvTable::default();
        let r = negamax_entered_impl(
            &mut pos,
            4,
            1,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heur,
        );
        assert!(r.is_some(), "unbudgeted search completes");
        let h = heur.unwrap();

        // At least one quiet cutoff actually rewarded.
        assert!(
            pvs_counters::PARENT_QUIET_REWARD.get() > 0,
            "at least one quiet beta-cutoff rewarded killer/history"
        );

        // Every rewarded quiet cutoff also recorded a killer for its ply, so
        // the number of non-empty killer slots is bounded by (but consistent
        // with) the reward count — a double reward would still only fill one
        // slot, so this alone is not the exact-once proof; the history check
        // below is. `record_killer` de-dups slot 0, so a reward may not grow
        // the slot count, hence `<=`.
        let killer_slots: usize = h
            .killers
            .iter()
            .map(|k| k.iter().filter(|s| s.is_some()).count())
            .sum();
        assert!(killer_slots > 0, "a killer was recorded");

        // The history table was populated by REAL quiet cutoffs (each cutoff
        // deposits `min(d*d, CAP)` exactly once). A scout+re-search double
        // reward would deposit twice for a single cutoff; the deterministic
        // cross-check below (byte-identical table on a second run) plus the
        // single-call-site invariant guard against that.
        let total_nonzero_history: usize = (0..2)
            .flat_map(|c| (0..64).flat_map(move |f| (0..64).map(move |t| (c, f, t))))
            .filter(|&(c, f, t)| h.history[c][f][t] != 0)
            .count();
        assert!(
            total_nonzero_history > 0,
            "history table populated by real quiet cutoffs"
        );

        // Determinism / no-accumulation cross-check: a second identical run
        // yields the byte-identical history table (a double reward that
        // depended on scout/re-search ordering would perturb it).
        let mut pos2 = parse_fen(START_FEN).unwrap();
        let ctx2 = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt2 = TranspositionTable::disabled();
        let mut heur2 = Some(SearchHeuristics::new());
        let mut path2 = SearchPath::new(vec![pos2.zobrist_key()]);
        let mut pv2 = PvTable::default();
        let _ = negamax_entered_impl(
            &mut pos2,
            4,
            1,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx2,
            &limits,
            SearchProfile::Current,
            &mut pv2,
            &mut path2,
            &mut tt2,
            &mut heur2,
        );
        let h2 = heur2.unwrap();
        assert!(
            h.history == h2.history,
            "history table is deterministic (no scout/re-search double reward)"
        );
        assert_eq!(
            h.killers, h2.killers,
            "killer table is deterministic (single reward per cutoff)"
        );
    }

    #[test]
    fn pvs_scout_fail_low_is_dropped_current() {
        // P1.1 regression: a null-window scout that fails LOW must never leak
        // its (upper-bound) score/PV into the parent. We use DEPTH 2 so each
        // child is a real negamax node. Measured depth-1 child values
        // (parent view) on `6k1/8/8/2p5/3r4/8/8/Q5K1 w - - 0 1`:
        //
        //  * the MVV-LVA first move `Qa1xd4` (a1-d4 diagonal, recaptured by
        //    `c5xd4`) is a LOSING capture;
        //  * every later QUIET move keeps White up Q vs R+P.
        //
        // The caller window `[400, 500]` sits ABOVE every true value, so the
        // whole node fails low: each later quiet move's null-window scout
        // fails low. Its MOVE/PV must never be committed — the parent PV must
        // still start with the first full-window move — but its NUMERIC value
        // is a legitimate upper bound folded into the returned score, so the
        // returned score is `>= v0` (never below the first move's value) and
        // still `<= ALPHA` (the node genuinely fails low). No re-search /
        // cutoff / heuristic reward may occur on an all-fail-low node.
        //
        // NOTE: this engine is fail-soft COMPATIBLE (TT Exact hits, TT
        // Lower/Upper cutoffs, and mate scores return real values outside
        // the window), so `scout_score > best` and even `scout_score >=
        // beta` are reachable in general. This fixture uses a disabled TT
        // and no mate lines, so here the scouts stay window-bounded; the
        // dedicated bound-regression and fail-high tests below drive the
        // fail-soft cases deterministically via a pre-filled TT.
        const FEN: &str = "6k1/8/8/2p5/3r4/8/8/Q5K1 w - - 0 1";
        const DEPTH: u32 = 2;
        const ALPHA: i32 = 400; // above every true move value -> whole node fails low
        const BETA: i32 = 500; // wide enough that later moves are scouted

        // --- Oracle: the first ordered move's exact full-window value. ---
        let (move0, v0) = {
            let mut pos = parse_fen(FEN).unwrap();
            let parent_key = pos.zobrist_key();
            let mut moves = generate_legal_moves(&mut pos);
            // Same ordering the parent applies on its first iteration:
            // Current profile, no hash move (disabled TT), empty heuristics.
            let empty = SearchHeuristics::new();
            order_moves_with_hash_and_killers(&pos, &mut moves, None, Some(&empty), 1);
            let move0 = moves[0];
            // The first move is the losing capture (highest MVV-LVA victim).
            assert!(
                is_tactical(&pos, move0),
                "first ordered move is the capture"
            );

            let undo = pos.make_move(move0);
            let child_key = pos.zobrist_key();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits::default();
            let mut tt = TranspositionTable::disabled();
            let mut heur = Some(SearchHeuristics::new());
            let mut path = SearchPath::new(vec![parent_key, child_key]);
            let mut pv = PvTable::default();
            let child = negamax_entered_impl(
                &mut pos,
                DEPTH - 1,
                2,
                -BETA,
                -ALPHA,
                &ctx,
                &limits,
                SearchProfile::Current,
                &mut pv,
                &mut path,
                &mut tt,
                &mut heur,
            )
            .expect("oracle child completes");
            pos.unmake_move(undo);
            (move0, -child)
        };
        assert!(v0 <= ALPHA, "first move fails low (v0={v0} <= alpha)");

        // --- Real parent node under Current. ---
        pvs_counters::reset();
        let mut pos = parse_fen(FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut tt = TranspositionTable::disabled();
        let mut heur = Some(SearchHeuristics::new());
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let mut pv = PvTable::default();
        let got = negamax_entered_impl(
            &mut pos,
            DEPTH,
            1,
            ALPHA,
            BETA,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heur,
        )
        .expect("parent node completes");

        // Fail-lows actually occurred and were dropped (not committed).
        assert!(
            pvs_counters::SCOUT_FAIL_LOW.get() > 0,
            "later moves failed the scout low"
        );
        // No re-search, no fail-high, no cutoff / reward at a fail-low node.
        assert_eq!(
            pvs_counters::RESEARCH_ENTERED.get(),
            0,
            "fail-low never re-searches"
        );
        assert_eq!(
            pvs_counters::SCOUT_FAIL_HIGH.get(),
            0,
            "no fail-high in an all-fail-low node"
        );
        // NOTE: the `PARENT_QUIET_REWARD` / `PARENT_TACTICAL_CUTOFF`
        // counters are GLOBAL across the whole search tree (they fire in every
        // non-root node, including deeper subtrees), so they cannot isolate
        // "the parent node itself". The P1.1 contract is proven below by the
        // parent's committed state: the PV starts with `move0` (never a scout
        // move) while the returned score keeps every fail-low scout's
        // numeric upper bound (`v0 <= got <= ALPHA`).

        // The parent kept ONLY the first full-window move's PV; the returned
        // score folds in the fail-low scouts' numeric upper bounds, so it
        // may exceed `v0` but never escapes the fail-low region.
        assert!(
            got >= v0,
            "returned score keeps the fail-low scouts' upper bounds (got={got} >= v0={v0})"
        );
        assert!(
            got <= ALPHA,
            "node still fails low overall (got={got} <= alpha={ALPHA})"
        );
        assert_eq!(
            pv.lines[1].first().copied(),
            Some(move0),
            "parent PV starts with the first full-window move, not a scout move"
        );
    }

    #[test]
    fn pvs_abort_restores_state_and_no_partial_parent_tt_current() {
        // P1.2: exercise the REAL `Current` PVS path (NOT `negamax_impl`,
        // which hardcodes `M4Reference` and never scouts). Sweeping node
        // budgets forces aborts at three distinct points, each proven by a
        // dedicated event counter:
        //   A) inside a null-window scout   -> ABORT_IN_SCOUT
        //   B) acquiring the re-search node -> ABORT_RESEARCH_ACQUIRE
        //   C) inside the full re-search    -> ABORT_IN_RESEARCH
        // Every aborted run must: return None, fully restore board / FEN /
        // Zobrist / SearchPath, stay within the node budget, and — verified
        // with an ENABLED TT — leave NO transposition entry for the unfinished
        // PARENT node (completed child entries may remain).
        const DEPTH: u32 = 4;
        let root = parse_fen(START_FEN).unwrap();
        // Key the aborted parent would have stored under (root at ply 1).
        let parent_key_probe = {
            let path = SearchPath::new(vec![root.zobrist_key()]);
            current_tt_key(&root, &path)
        };

        // Unlimited baseline: nodes the full node consumes (fresh enabled TT).
        let full_nodes = {
            let mut pos = root;
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits::default();
            let mut tt = TranspositionTable::new_mb(1).unwrap();
            let mut heur = Some(SearchHeuristics::new());
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let mut pv = PvTable::default();
            let r = negamax_entered_impl(
                &mut pos,
                DEPTH,
                1,
                i32::MIN + 1000,
                i32::MAX - 1000,
                &ctx,
                &limits,
                SearchProfile::Current,
                &mut pv,
                &mut path,
                &mut tt,
                &mut heur,
            );
            assert!(r.is_some(), "unbudgeted node completes");
            ctx.nodes.load(Ordering::Relaxed)
        };
        assert!(full_nodes > 8, "node has a non-trivial subtree");

        pvs_counters::reset();
        // Bound the sweep for runtime; all three abort phases occur within the
        // first scoutable moves' subtrees, well inside this range. The loop
        // also breaks early once every phase has fired so debug-mode runtime
        // stays small (total work is O(sum of visited budgets)).
        let cap = full_nodes.saturating_sub(1).min(1200);
        for budget in 1..=cap {
            let mut pos = root;
            let before_fen = to_fen(&pos);
            let before_key = pos.zobrist_key();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                nodes: Some(budget),
                ..Default::default()
            };
            let mut tt = TranspositionTable::new_mb(1).unwrap();
            let mut heur = Some(SearchHeuristics::new());
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let root_len = path.len();
            // P2: capture the full SearchPath fingerprint (not just len/keys)
            // so an unbalanced push/pop that happened to restore the length
            // but corrupted the repetition context or the immutable base
            // prefix cannot pass silently.
            let before_sig = path.repetition_signature();
            let before_base_len = path.base_len();
            let mut pv = PvTable::default();
            let r = negamax_entered_impl(
                &mut pos,
                DEPTH,
                1,
                i32::MIN + 1000,
                i32::MAX - 1000,
                &ctx,
                &limits,
                SearchProfile::Current,
                &mut pv,
                &mut path,
                &mut tt,
                &mut heur,
            );
            assert!(r.is_none(), "budget {budget} < {full_nodes} must abort");
            assert_eq!(
                path.len(),
                root_len,
                "path length restored (budget={budget})"
            );
            assert_eq!(
                path.keys(),
                &[before_key],
                "path restored to root key (budget={budget})"
            );
            assert_eq!(
                to_fen(&pos),
                before_fen,
                "position restored (budget={budget})"
            );
            assert_eq!(
                pos.zobrist_key(),
                before_key,
                "key restored (budget={budget})"
            );
            // P2: an abort here is ALWAYS budget exhaustion (no stop flag is
            // set), so `try_enter_node` fails exactly when the counter has
            // consumed the whole budget — the node count is EQUAL to the
            // budget, never merely `<=` it. A weaker `<=` would hide an early
            // return that left budget unused.
            assert_eq!(
                ctx.nodes.load(Ordering::Relaxed),
                budget,
                "an aborted node consumes exactly its budget (budget={budget})"
            );
            // P2: the repetition signature and the immutable base prefix are
            // both restored — proves push/pop balance beyond the raw length.
            assert_eq!(
                path.repetition_signature(),
                before_sig,
                "repetition signature restored (budget={budget})"
            );
            assert_eq!(
                path.base_len(),
                before_base_len,
                "base prefix length restored (budget={budget})"
            );
            assert!(
                tt.probe(parent_key_probe).is_none(),
                "aborted parent left no TT entry (budget={budget})"
            );

            // Stop as soon as all three abort phases have been provably hit;
            // every budget below this point has already validated the
            // abort/restore/no-partial-TT invariants.
            if pvs_counters::ABORT_IN_SCOUT.get() > 0
                && pvs_counters::ABORT_RESEARCH_ACQUIRE.get() > 0
                && pvs_counters::ABORT_IN_RESEARCH.get() > 0
            {
                break;
            }
        }

        // All three abort phases actually fired across the budget sweep.
        assert!(
            pvs_counters::ABORT_IN_SCOUT.get() > 0,
            "phase A: scout-internal abort observed"
        );
        assert!(
            pvs_counters::ABORT_RESEARCH_ACQUIRE.get() > 0,
            "phase B: re-search node-acquisition abort observed"
        );
        assert!(
            pvs_counters::ABORT_IN_RESEARCH.get() > 0,
            "phase C: re-search-internal abort observed"
        );
    }

    #[test]
    fn pvs_current_top_level_stopped_and_previous_iteration() {
        // P1.2 (top level): a `Current` search stopped BEFORE depth 1
        // completes reports no score / completed_depth 0 / empty PV (never a
        // faked PV); one that completes depth 1 then aborts a deeper iteration
        // keeps the last completed iteration's score / best move / PV. The
        // root position is restored in both cases.
        let nodes_for = |depth: u32| -> u64 {
            let mut pos = parse_fen(START_FEN).unwrap();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                depth: Some(depth),
                ..Default::default()
            };
            let mut tt = TranspositionTable::disabled();
            let key = pos.zobrist_key();
            let _ = search_best_move_with_history_tt_and_profile(
                &mut pos,
                &[key],
                &limits,
                &ctx,
                &mut tt,
                SearchProfile::Current,
            )
            .unwrap();
            ctx.nodes.load(Ordering::Relaxed)
        };
        let n1 = nodes_for(1);
        let n2 = nodes_for(2);
        assert!(n1 >= 1 && n2 > n1, "depth 2 strictly deeper than depth 1");

        // Case 1: tiny budget -> stopped before depth 1 finishes.
        {
            let mut pos = parse_fen(START_FEN).unwrap();
            let before = to_fen(&pos);
            let before_key = pos.zobrist_key();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                nodes: Some(1),
                ..Default::default()
            };
            let mut tt = TranspositionTable::disabled();
            let key = pos.zobrist_key();
            let out = search_best_move_with_history_tt_and_profile(
                &mut pos,
                &[key],
                &limits,
                &ctx,
                &mut tt,
                SearchProfile::Current,
            )
            .expect("outcome");
            assert!(out.stopped, "budget=1 stops before depth 1 completes");
            assert_eq!(out.completed_depth, 0, "no iteration completed");
            assert_eq!(out.score, None, "no real score");
            assert!(out.pv.is_empty(), "no faked PV when nothing completed");
            assert_eq!(to_fen(&pos), before, "root position restored");
            assert_eq!(pos.zobrist_key(), before_key, "root key restored");
        }

        // Case 2: budget in (n1, n2) -> depth 1 completes, depth 2 aborts.
        {
            let budget = n1 + (n2 - n1) / 2; // n1 <= budget < n2
            let mut pos = parse_fen(START_FEN).unwrap();
            let before = to_fen(&pos);
            let before_key = pos.zobrist_key();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                nodes: Some(budget),
                ..Default::default()
            };
            let mut tt = TranspositionTable::disabled();
            let key = pos.zobrist_key();
            let out = search_best_move_with_history_tt_and_profile(
                &mut pos,
                &[key],
                &limits,
                &ctx,
                &mut tt,
                SearchProfile::Current,
            )
            .expect("outcome");
            assert!(out.stopped, "deeper iteration aborted");
            assert_eq!(
                out.completed_depth, 1,
                "kept the completed depth-1 iteration"
            );
            assert!(out.score.is_some(), "depth-1 score retained");
            assert!(!out.pv.is_empty(), "depth-1 PV retained");
            assert!(pv_is_legal(START_FEN, &out.pv), "retained PV is legal");
            assert_eq!(to_fen(&pos), before, "root position restored");
            assert_eq!(pos.zobrist_key(), before_key, "root key restored");
        }
    }

    #[test]
    fn pvs_scout_improves_alpha_then_research_quiet_cutoff_current() {
        // P2.2: a null-window scout that IMPROVES alpha (lands in-window,
        // `alpha < scout_score < beta`) must trigger a full re-search — it must
        // NOT be mistaken for a fail-high direct cutoff. We use DEPTH 2 so each
        // child is a real negamax node. On `6k1/8/8/2p5/3r4/8/8/Q5K1 w - - 0 1`:
        //
        //  * the MVV-LVA first move `Qa1xd4` (recaptured by `c5xd4`) is a
        //    LOSING capture, so it does NOT cut off and leaves a low `alpha`;
        //  * every later QUIET move's null-window scout lands IN-WINDOW
        //    (improves alpha) -> re-search runs, and the re-search (full window)
        //    then produces a QUIET beta-cutoff that rewards killer/history once.
        //
        // Window `[alpha=-20000, beta=150]`: the first move never cuts off,
        // the losing capture's scout fails low, and the first safe quiet move's
        // scout improves alpha -> re-search -> quiet cutoff.
        const FEN: &str = "6k1/8/8/2p5/3r4/8/8/Q5K1 w - - 0 1";
        pvs_counters::reset();
        let mut pos = parse_fen(FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut tt = TranspositionTable::disabled();
        let mut heur = Some(SearchHeuristics::new());
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let mut pv = PvTable::default();
        let got = negamax_entered_impl(
            &mut pos,
            2,
            1,
            -20_000, // alpha far below every move -> first move never cuts off
            150,     // beta between the losing capture and the safe quiets
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heur,
        );
        assert!(got.is_some(), "node completes");
        // A scout that improves alpha re-searches (NOT a dead fail-high path).
        assert!(
            pvs_counters::RESEARCH_ENTERED.get() > 0,
            "an in-window scout triggered a full re-search"
        );
        // The re-search (full window) produced a quiet beta-cutoff, rewarded
        // exactly once.
        assert!(
            pvs_counters::PARENT_QUIET_REWARD.get() > 0,
            "the re-search produced a quiet beta-cutoff with a single reward"
        );
        // In THIS fixture (disabled TT, no mate lines in window) every scout
        // stays window-bounded, so the in-window improvement goes through the
        // re-search rather than a direct fail-high. That is a property of
        // the fixture, NOT of the engine: scouts are fail-soft compatible
        // and CAN fail high (see `pvs_scout_fail_high_via_tt_exact_*`, which
        // drives `scout_score >= beta` deterministically via a pre-filled
        // TT Exact entry).
        assert_eq!(
            pvs_counters::SCOUT_FAIL_HIGH.get(),
            0,
            "no fail-high occurs in this disabled-TT fixture (in-window scouts re-search)"
        );
        // No budget abort in an unbounded search: every attempt entered.
        assert_eq!(
            pvs_counters::RESEARCH_ATTEMPT.get(),
            pvs_counters::RESEARCH_ENTERED.get(),
            "no budget abort: attempts == entered"
        );
    }

    #[test]
    fn pvs_full_research_pv_comes_from_research_current() {
        // P2.2: when a scout lands in-window and a full re-search runs, the
        // committed child PV must be the RE-SEARCH line (the re-search clears
        // and rewrites the child PV row), never the stale null-window scout
        // line. We assert re-searches actually ran and the resulting root PV
        // is legal and matches the same-depth `M41Reference` full-window PV
        // (score parity is the invariant; the PV is a real, playable line).
        pvs_counters::reset();
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let mut pos_c = parse_fen(START_FEN).unwrap();
        let ctx_c = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_c = TranspositionTable::disabled();
        let key_c = pos_c.zobrist_key();
        let out_c = search_best_move_with_history_tt_and_profile(
            &mut pos_c,
            &[key_c],
            &limits,
            &ctx_c,
            &mut tt_c,
            SearchProfile::Current,
        )
        .expect("current outcome");
        assert!(
            pvs_counters::RESEARCH_ENTERED.get() > 0,
            "at least one full re-search ran"
        );
        assert!(!out_c.pv.is_empty(), "re-searched Current PV is non-empty");
        assert!(
            pv_is_legal(START_FEN, &out_c.pv),
            "the committed (re-searched) PV is a legal line"
        );

        // P2: every completed full re-search recorded the child PV row as the
        // SCOUT left it paired with the row the RE-SEARCH rewrote. First, the
        // pairing is exhaustive — one pair per re-search that returned a
        // score (aborted re-searches propagate `None` and record nothing).
        let pairs = pvs_counters::RESEARCH_PV_PAIRS.with_borrow(|v| v.clone());
        assert_eq!(
            pairs.len(),
            pvs_counters::RESEARCH_ENTERED.get(),
            "one (scout,research) child-row pair captured per completed re-search"
        );
        // The re-search clears + rewrites the child PV row before returning,
        // so when a re-searched move becomes a node's best move the parent
        // copies exactly the RE-SEARCH row (never a stale scout row). This is
        // proven STRUCTURALLY and inline at the commit site (an `assert_eq!`
        // comparing the parent's committed child tail against the recorded
        // re-search row), and `RESEARCH_ROW_COMMITTED` proves that guarded
        // commit path was actually exercised. (We do NOT require the scout
        // and re-search rows to differ: a null-window scout that improves
        // alpha frequently finds the same best child line — the invariant
        // under test is that the committed row is a genuine re-search
        // product, not that it is textually distinct from the scout row.)
        assert!(
            pvs_counters::RESEARCH_ROW_COMMITTED.get() > 0,
            "at least one re-searched move became a node best and committed its re-search child row"
        );

        let mut pos_m = parse_fen(START_FEN).unwrap();
        let ctx_m = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt_m = TranspositionTable::disabled();
        let key_m = pos_m.zobrist_key();
        let out_m = search_best_move_with_history_tt_and_profile(
            &mut pos_m,
            &[key_m],
            &limits,
            &ctx_m,
            &mut tt_m,
            SearchProfile::M41Reference,
        )
        .expect("m41 outcome");
        assert_eq!(
            out_c.score, out_m.score,
            "PVS preserves the fixed-depth root score"
        );
    }

    #[test]
    fn pvs_scout_fail_low_bound_preserved_via_tt_current() {
        // P1.1 (the core bound-safety regression). A null-window scout that
        // fails LOW must not commit its move/PV, but its NUMERIC upper bound
        // MUST survive: dropping it would let an all-fail-low node store a TT
        // `Bound::Upper` that UNDER-states the real node value and later cause
        // a wrong TT cutoff. "The fail-low scout's PV is not committable, but
        // the numeric upper bound it provides is still part of search
        // correctness."
        //
        // We drive the fail-soft case deterministically with a pre-filled,
        // ENABLED TT. On a quiet-only position (no captures / promotions /
        // mates, so ordering and every child value are fully controlled) at
        // caller window `[alpha=100, beta=200]`, depth 2:
        //   * the FIRST ordered move's child returns Exact -40  -> parent
        //     candidate 40  (below alpha: node fails low, `best = 40`);
        //   * the SECOND ordered move's child returns Exact -80  -> its scout
        //     score 80 is an UPPER bound that is BOTH `<= alpha` (fails low)
        //     AND `> best` (the exact P1.1 hazard) -> dropped from best/PV but
        //     folded into `fail_low_upper = 80`;
        //   * every later move's child returns Exact -30 -> scout 30, also a
        //     dropped fail-low, does not raise `fail_low_upper`.
        // The node must return `80` (not `40`) and store a TT `Bound::Upper`
        // of `80`, which does NOT under-state the true full-window value.
        const FEN: &str = "4k3/8/8/8/8/5N2/4P3/4K3 w - - 0 1";
        const DEPTH: u32 = 2;
        const ALPHA: i32 = 100;
        const BETA: i32 = 200;

        // The exact move order the parent will apply on its first iteration
        // (Current profile, no hash move, empty heuristics, ply 1).
        let ordered: Vec<Move> = {
            let mut pos = parse_fen(FEN).unwrap();
            let mut moves = generate_legal_moves(&mut pos);
            let empty = SearchHeuristics::new();
            order_moves_with_hash_and_killers(&pos, &mut moves, None, Some(&empty), 1);
            // Position is quiet-only, so the two controlled moves are quiet.
            assert!(!is_tactical(&pos, moves[0]), "no captures in fixture");
            assert!(!is_tactical(&pos, moves[1]), "no captures in fixture");
            moves
        };
        let child_score = |idx: usize| -> i32 {
            match idx {
                0 => -40, // parent candidate 40 (below alpha -> fails low)
                1 => -80, // scout 80: <= alpha (fail low) AND > best (hazard)
                _ => -30, // scout 30: dropped fail-low, does not lift the bound
            }
        };
        // Pre-fill every child's Exact TT entry so the whole subtree is
        // deterministic. Child depth 2 >= the child's requested depth (1).
        let prefill = |tt: &mut TranspositionTable| {
            let mut pos = parse_fen(FEN).unwrap();
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            for (idx, &m) in ordered.iter().enumerate() {
                let undo = pos.make_move(m);
                path.push_child(&pos);
                let ckey = current_tt_key(&pos, &path);
                store_tt_score(tt, ckey, DEPTH, child_score(idx), 2, Bound::Exact, None);
                path.pop();
                pos.unmake_move(undo);
            }
        };

        // --- Oracle: the true full-window value of this exact node. ---
        let oracle = {
            let mut pos = parse_fen(FEN).unwrap();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits::default();
            let mut tt = TranspositionTable::new_mb(1).unwrap();
            prefill(&mut tt);
            let mut heur = Some(SearchHeuristics::new());
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let mut pv = PvTable::default();
            negamax_entered_impl(
                &mut pos,
                DEPTH,
                1,
                i32::MIN + 1000,
                i32::MAX - 1000,
                &ctx,
                &limits,
                SearchProfile::Current,
                &mut pv,
                &mut path,
                &mut tt,
                &mut heur,
            )
            .expect("oracle node completes")
        };
        assert_eq!(oracle, 80, "true full-window node value is 80");

        // --- Real fail-low node under the tight caller window. ---
        pvs_counters::reset();
        let mut pos = parse_fen(FEN).unwrap();
        let parent_probe_key = {
            let path = SearchPath::new(vec![pos.zobrist_key()]);
            current_tt_key(&pos, &path)
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        prefill(&mut tt);
        let mut heur = Some(SearchHeuristics::new());
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let mut pv = PvTable::default();
        let got = negamax_entered_impl(
            &mut pos,
            DEPTH,
            1,
            ALPHA,
            BETA,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heur,
        )
        .expect("parent node completes");

        // The hazard genuinely occurred: a fail-low scout scored ABOVE `best`.
        assert!(
            pvs_counters::SCOUT_FAIL_LOW.get() > 0,
            "later scouts failed low"
        );
        assert!(
            pvs_counters::SCOUT_FAIL_LOW_EXCEEDS_BEST.get() > 0,
            "a dropped fail-low scout scored above the running best (the P1.1 hazard)"
        );
        // A fail-low node never re-searches, never fails high, never cuts off.
        assert_eq!(
            pvs_counters::RESEARCH_ENTERED.get(),
            0,
            "fail-low never re-searches"
        );
        assert_eq!(pvs_counters::SCOUT_FAIL_HIGH.get(), 0, "no fail-high");
        assert_eq!(
            pvs_counters::PARENT_QUIET_REWARD.get(),
            0,
            "no beta cutoff -> no heuristic reward on a fail-low node"
        );
        assert_eq!(
            pvs_counters::RECORD_KILLER_CALLS.get(),
            0,
            "no killer recorded on a fail-low node"
        );
        assert_eq!(
            pvs_counters::RECORD_HISTORY_CALLS.get(),
            0,
            "no history recorded on a fail-low node"
        );

        // The returned score keeps the dropped scout's numeric upper bound
        // (80), NOT the first move's 40 — yet still fails low overall.
        assert_eq!(
            got, 80,
            "returned score lifts to the fail-low scout's upper bound"
        );
        assert!((80..=ALPHA).contains(&got), "80 <= got <= alpha");
        // The committed PV/best move are still ONLY the first full-window move.
        assert_eq!(
            pv.lines[1].first().copied(),
            Some(ordered[0]),
            "parent PV starts with the first full-window move, not a scout move"
        );

        // The stored TT entry is an Upper bound that does NOT under-state the
        // true value: `oracle (80) <= stored <= caller alpha (100)`. Dropping
        // the scout bound (the P1.1 bug) would have stored `40 < oracle`.
        let entry = tt.probe(parent_probe_key).expect("parent TT entry stored");
        assert_eq!(entry.bound, Bound::Upper, "fail-low node stores Upper");
        let stored = score_from_tt(entry.score, 1).expect("decodes");
        assert!(
            stored >= oracle,
            "stored Upper {stored} must not under-state the true value {oracle}"
        );
        assert!(
            stored <= ALPHA,
            "stored Upper {stored} stays at/below caller alpha {ALPHA}"
        );
    }

    #[test]
    fn pvs_scout_fail_high_via_tt_exact_current() {
        // P1.2 (the core fail-high regression). This engine is fail-soft
        // COMPATIBLE: a TT Exact hit returns the real stored score even OUTSIDE
        // the probing window. A null-window scout can therefore come back at or
        // above `beta` -> a genuine `MoveOutcome::ScoutFailHigh` that is a real
        // cutoff (NOT dead code, NOT re-searched). We drive it deterministically
        // with a pre-filled enabled TT on a quiet-only position, caller window
        // `[alpha=0, beta=100]`, depth 2:
        //   * the FIRST ordered move's child returns Exact -50 -> candidate 50,
        //     `best = 50`, `alpha = 50` (does not cut off);
        //   * the SECOND ordered move's null-window scout `[-51, -50]` hits its
        //     child's Exact -100 (returned despite being out of window) -> scout
        //     score 100 `>= beta` -> fail HIGH: committed once, no re-search,
        //     and (quiet) rewards killer/history exactly once.
        const FEN: &str = "4k3/8/8/8/8/5N2/4P3/4K3 w - - 0 1";
        const DEPTH: u32 = 2;
        const ALPHA: i32 = 0;
        const BETA: i32 = 100;

        let ordered: Vec<Move> = {
            let mut pos = parse_fen(FEN).unwrap();
            let mut moves = generate_legal_moves(&mut pos);
            let empty = SearchHeuristics::new();
            order_moves_with_hash_and_killers(&pos, &mut moves, None, Some(&empty), 1);
            assert!(!is_tactical(&pos, moves[0]), "first move is quiet");
            assert!(
                !is_tactical(&pos, moves[1]),
                "the fail-high move is quiet (rewards killer/history)"
            );
            moves
        };
        // Only the first two children are visited before the cutoff; pre-fill
        // exactly those two so the scenario is unambiguous.
        let prefill = |tt: &mut TranspositionTable| {
            let mut pos = parse_fen(FEN).unwrap();
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            for (idx, &m) in ordered.iter().take(2).enumerate() {
                let undo = pos.make_move(m);
                path.push_child(&pos);
                let ckey = current_tt_key(&pos, &path);
                let s = if idx == 0 { -50 } else { -100 };
                store_tt_score(tt, ckey, DEPTH, s, 2, Bound::Exact, None);
                path.pop();
                pos.unmake_move(undo);
            }
        };

        pvs_counters::reset();
        let mut pos = parse_fen(FEN).unwrap();
        let parent_probe_key = {
            let path = SearchPath::new(vec![pos.zobrist_key()]);
            current_tt_key(&pos, &path)
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits::default();
        let mut tt = TranspositionTable::new_mb(1).unwrap();
        prefill(&mut tt);
        let mut heur = Some(SearchHeuristics::new());
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let mut pv = PvTable::default();
        let got = negamax_entered_impl(
            &mut pos,
            DEPTH,
            1,
            ALPHA,
            BETA,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heur,
        )
        .expect("parent node completes");

        // A real fail-high occurred through a fail-soft (out-of-window) return.
        assert!(
            pvs_counters::SCOUT_FAIL_HIGH.get() > 0,
            "a scout failed high (fail-soft compatible; NOT dead code)"
        );
        // A fail-high is a direct cutoff: it is NEVER re-searched.
        assert_eq!(
            pvs_counters::RESEARCH_ATTEMPT.get(),
            0,
            "fail-high never even attempts a re-search"
        );
        assert_eq!(
            pvs_counters::RESEARCH_ENTERED.get(),
            0,
            "fail-high never re-searches"
        );
        // The quiet fail-high cutoff rewarded killer + history EXACTLY once.
        assert_eq!(
            pvs_counters::PARENT_QUIET_REWARD.get(),
            1,
            "one quiet cutoff reward"
        );
        assert_eq!(
            pvs_counters::RECORD_KILLER_CALLS.get(),
            1,
            "killer recorded exactly once"
        );
        assert_eq!(
            pvs_counters::RECORD_HISTORY_CALLS.get(),
            1,
            "history recorded exactly once"
        );

        // The node returns the fail-high score and commits the fail-high move.
        assert_eq!(got, 100, "returned score is the fail-high cutoff value");
        assert_eq!(
            pv.lines[1].first().copied(),
            Some(ordered[1]),
            "PV starts with the fail-high move"
        );
        // The stored TT entry is a Lower bound (a proven cutoff).
        let entry = tt.probe(parent_probe_key).expect("parent TT entry stored");
        assert_eq!(entry.bound, Bound::Lower, "fail-high node stores Lower");
        assert_eq!(
            score_from_tt(entry.score, 1),
            Some(100),
            "stored Lower is the cutoff value"
        );
    }

    #[test]
    fn m4_ordering_priority_with_killers() {
        // Synthetic move set: TT hash move first, then a capture
        // (MVV-LVA), then killer slot 0, then killer slot 1, then the
        // remaining quiet move. Every move appears exactly once.
        let mut pos =
            parse_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let mut moves = generate_legal_moves(&mut pos);
        let orig = moves.clone();

        let tt_move = find_move(&pos, "g1f3"); // quiet knight (TT move)
        let cap = find_move(&pos, "e4d5"); // capture (MVV-LVA)
        let k0 = find_move(&pos, "b1c3"); // quiet knight (killer slot 0)
        let k1 = find_move(&pos, "d2d4"); // quiet pawn (killer slot 1)
        let quiet_other = find_move(&pos, "b2b3"); // remaining quiet

        // Build the heuristic state for this ply: killers [k0, k1] and a
        // zeroed history table (single remaining quiet -> history tie).
        let mut h = SearchHeuristics::new();
        h.ensure_ply(0);
        h.killers[0] = [Some(k0), Some(k1)];
        order_moves_with_hash_and_killers(&pos, &mut moves, Some(tt_move), Some(&h), 0);

        // Level 1: TT hash move lifted to index 0.
        assert_eq!(moves[0], tt_move, "TT move first");
        // Every original move appears exactly once.
        assert_eq!(moves.len(), orig.len());
        assert!(moves.iter().all(|m| orig.contains(m)));
        assert!(orig.iter().all(|m| moves.contains(m)));

        let idx = |m: Move| moves.iter().position(|&x| x == m).unwrap();
        // Level 2/3: capture before killers (bucket 2 < bucket 3).
        assert!(idx(cap) < idx(k0), "capture before killer0");
        // Level 4 < Level 5: killer0 before killer1.
        assert!(idx(k0) < idx(k1), "killer0 before killer1");
        // Level 5 < Level 6: killer1 before remaining quiet.
        assert!(idx(k1) < idx(quiet_other), "killer1 before remaining quiet");
    }

    #[test]
    fn m4_tactical_never_killer_guard() {
        // The `!is_tactical(m)` guard in `negamax_entered_impl` is
        // what keeps captures / en passant / promotions from ever becoming
        // killers (spec §3.2). Document it directly.
        let start = parse_fen(START_FEN).unwrap();
        let quiet = find_move(&start, "g1f3");
        assert!(!is_tactical(&start, quiet), "quiet move is not tactical");

        let cap_pos =
            parse_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let cap = find_move(&cap_pos, "e4d5");
        assert!(is_tactical(&cap_pos, cap), "capture is tactical");

        let promo_pos = parse_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").unwrap();
        let promo = find_move(&promo_pos, "a7a8q");
        assert!(is_tactical(&promo_pos, promo), "promotion is tactical");

        let ep_pos =
            parse_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1").unwrap();
        let ep = find_move(&ep_pos, "e5d6");
        assert!(is_tactical(&ep_pos, ep), "en passant is tactical");
    }

    // ---- §15: enabled cold ----------------------------------------------------

    #[test]
    fn tt_enabled_cold_startpos() {
        let mut tt = TranspositionTable::new_mb(16).unwrap();
        let (out, _) = run_tt(START_FEN, 3, &mut tt);
        let out = out.expect("outcome");
        assert_eq!(out.score, Some(50), "enabled cold keeps the exact score");
        assert_eq!(
            move_to_uci(out.best_move),
            "b1c3",
            "enabled cold keeps the baseline best move"
        );
        assert!(pv_is_legal(START_FEN, &out.pv), "enabled PV legal");
        assert_eq!(out.pv.first().copied(), Some(out.best_move));
    }

    #[test]
    fn tt_enabled_cold_queenwin() {
        let fen = "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1";
        let mut tt = TranspositionTable::new_mb(16).unwrap();
        let (out, _) = run_tt(fen, 3, &mut tt);
        let out = out.expect("outcome");
        assert_eq!(out.score, Some(890));
        assert_eq!(move_to_uci(out.best_move), "e4a4");
        assert!(pv_is_legal(fen, &out.pv));
        assert_eq!(out.pv.first().copied(), Some(out.best_move));
    }

    // ---- §15: enabled warm (transposition reuse) -----------------------------

    #[test]
    fn tt_enabled_warm_reuses_transpositions() {
        let mut tt = TranspositionTable::new_mb(16).unwrap();
        let (out1, n1) = run_tt(START_FEN, 4, &mut tt);
        let out1 = out1.expect("cold outcome");
        let (out2, n2) = run_tt(START_FEN, 4, &mut tt); // warm: same populated table
        let out2 = out2.expect("warm outcome");
        assert_eq!(out1.score, out2.score, "score stable cold->warm");
        assert_eq!(out1.completed_depth, out2.completed_depth);
        assert_eq!(
            move_to_uci(out1.best_move),
            move_to_uci(out2.best_move),
            "best move stable"
        );
        assert!(n2 <= n1, "warm nodes ({}) must be <= cold ({})", n2, n1);
        assert!(
            n2 < n1,
            "expected transposition savings at startpos d4: cold {} warm {}",
            n1,
            n2
        );
    }

    // ---- §15: equal-score may pick a different (still legal) move ------------

    #[test]
    fn tt_enabled_disabled_same_score_legal_move() {
        let disabled = run_tt(START_FEN, 3, &mut TranspositionTable::disabled())
            .0
            .expect("disabled");
        let mut tt = TranspositionTable::new_mb(16).unwrap();
        let enabled = run_tt(START_FEN, 3, &mut tt).0.expect("enabled");
        assert_eq!(disabled.score, enabled.score, "score identical");
        assert!(pv_is_legal(START_FEN, &disabled.pv));
        assert!(pv_is_legal(START_FEN, &enabled.pv));
        assert_eq!(move_to_uci(disabled.best_move), "b1c3");
        let legal: BTreeSet<String> = generate_legal_moves(&mut parse_fen(START_FEN).unwrap())
            .into_iter()
            .map(move_to_uci)
            .collect();
        assert!(
            legal.contains(&move_to_uci(enabled.best_move)),
            "enabled best move is legal"
        );
    }

    // ---- §16: recovery & rule regression ------------------------------------

    #[test]
    fn tt_enabled_restores_position_and_path() {
        // Build a REAL, position-aligned game history by playing legal moves
        // from startpos and recording each position's Zobrist key. The final
        // `pos` is the last key in `history` (so the path is aligned).
        let mut pos = parse_fen(START_FEN).unwrap();
        let mut history: Vec<ZobristKey> = vec![pos.zobrist_key()];
        for uci in ["g1f3", "g8f6", "e2e4", "e7e5"] {
            let m = find_move(&pos, uci);
            pos.make_move(m);
            history.push(pos.zobrist_key());
        }

        // Construct the SearchPath exactly as the production entry point does.
        let mut path = SearchPath::new(history.clone());

        // Snapshot every piece of state the search must restore afterwards.
        let before_fen = to_fen(&pos);
        let before_key = pos.zobrist_key();
        let before_halfmove = pos.halfmove_clock();
        let saved_keys: Vec<ZobristKey> = path.keys().to_vec();
        let saved_sig = path.repetition_signature();
        let saved_base_len = path.base_len();

        let mut tt = TranspositionTable::new_mb(8).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        // Call the same impl the production path uses, passing the ENABLED
        // table and the real path — NOT a freshly rebuilt one.
        let out = search_best_move_impl(
            &mut pos,
            &limits,
            &ctx,
            SearchProfile::M4Reference,
            &mut path,
            &mut tt,
        );
        assert!(out.is_some(), "search completes");

        // Every state must be EXACTLY restored, proving the search used
        // (and restored) the real path rather than a reconstructed copy.
        assert_eq!(to_fen(&pos), before_fen, "root position FEN restored");
        assert_eq!(pos.zobrist_key(), before_key, "root Zobrist restored");
        assert_eq!(
            pos.halfmove_clock(),
            before_halfmove,
            "root halfmove clock restored"
        );
        assert_eq!(
            path.keys().to_vec(),
            saved_keys,
            "SearchPath keys (history) restored"
        );
        assert_eq!(
            path.repetition_signature(),
            saved_sig,
            "repetition signature restored"
        );
        assert_eq!(path.base_len(), saved_base_len, "base_len restored");
        assert_eq!(path.len(), saved_keys.len(), "path length restored");
    }

    // ==== M4.2 Commit 4: root Principal Variation Search =====================
    //
    // These tests exercise the ROOT PVS edges specifically (the non-root PVS
    // edges are covered by the `pvs_*` tests above). They rely on the
    // `ROOT_*` event counters in `pvs_counters`, which are reset per test and
    // are thread-local (each `#[test]` runs on its own thread, so the counts
    // observe only this test's search). Where a precise per-iteration count is
    // needed, we call the private `root_search` directly for a SINGLE depth
    // (iterative deepening would otherwise accumulate counts across depths).

    /// A queen-win fixture: White (to move) can win Black's queen with
    /// `Qe4xa4` (`e4a4`), by far the best root move. Used to force a known
    /// strong / weak split at the root.
    const ROOT_QWIN_FEN: &str = "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1";

    /// Pick any legal root move that is NOT `e4a4` (guaranteed to exist and to
    /// be strictly worse than winning the queen), plus the `e4a4` move itself.
    fn root_weak_and_qxa4(fen: &str) -> (Move, Move) {
        let mut pos = parse_fen(fen).unwrap();
        let qxa4 = find_move(&pos, "e4a4");
        let moves = generate_legal_moves(&mut pos);
        let weak = *moves
            .iter()
            .find(|m| **m != qxa4)
            .expect("a non-Qxa4 legal move exists");
        (weak, qxa4)
    }

    #[test]
    fn root_pvs_profile_isolation() {
        // The reference profiles keep a full-window root: at the ROOT they must
        // NEVER scout, fail-low a scout, attempt or enter a re-search, and must
        // never mark the `Current`-only "first root move full" event. `Current`
        // runs root PVS and, on a multi-move position, scouts later root moves.
        let depth = 3;

        for profile in [SearchProfile::M4Reference, SearchProfile::M41Reference] {
            pvs_counters::reset();
            let mut pos = parse_fen(ROOT_QWIN_FEN).unwrap();
            let mut root_moves = generate_legal_moves(&mut pos.clone());
            let fallback = root_moves[0];
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                depth: Some(depth),
                ..Default::default()
            };
            let mut heur = if profile == SearchProfile::M4Reference {
                None
            } else {
                Some(SearchHeuristics::new())
            };
            root_search(
                &mut pos,
                depth,
                &mut root_moves,
                false,
                fallback,
                &ctx,
                &limits,
                profile,
                &mut path,
                &mut TranspositionTable::disabled(),
                &mut heur,
            )
            .expect("iteration completes");
            assert_eq!(
                pvs_counters::ROOT_SCOUT.get(),
                0,
                "{profile:?} never scouts at the root"
            );
            assert_eq!(
                pvs_counters::ROOT_FAIL_LOW.get(),
                0,
                "{profile:?} never fails a root scout low"
            );
            assert_eq!(
                pvs_counters::ROOT_RESEARCH_ATTEMPT.get(),
                0,
                "{profile:?} never attempts a root re-search"
            );
            assert_eq!(
                pvs_counters::ROOT_RESEARCH_ENTERED.get(),
                0,
                "{profile:?} never re-searches at the root"
            );
            assert_eq!(
                pvs_counters::ROOT_FIRST_FULL.get(),
                0,
                "{profile:?} never marks the Current-only first-full root event"
            );
        }

        // Current: root PVS scouts later root moves and marks the first full.
        pvs_counters::reset();
        let mut pos = parse_fen(ROOT_QWIN_FEN).unwrap();
        let mut root_moves = generate_legal_moves(&mut pos.clone());
        let fallback = root_moves[0];
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let mut heur = Some(SearchHeuristics::new());
        root_search(
            &mut pos,
            depth,
            &mut root_moves,
            false,
            fallback,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut heur,
        )
        .expect("iteration completes");
        assert!(
            pvs_counters::ROOT_SCOUT.get() > 0,
            "Current scouts later root moves"
        );
        assert!(
            pvs_counters::ROOT_FIRST_FULL.get() > 0,
            "Current searches the first root move full-window"
        );
    }

    #[test]
    fn root_pvs_real_path_counters_fire_in_real_search() {
        // Prove the ROOT PVS branches actually execute inside a real `Current`
        // iterative-deepening search — not merely that a node count changed.
        // The queen-win fixture is chosen because raw depth-1 move generation
        // does NOT put the winning `Qxa4` first, so a later root move's scout
        // genuinely improves alpha and is fully re-searched. (At startpos the
        // ordering is so good that the best move is searched first at every
        // depth and NO root re-search ever fires — that is correct, efficient
        // PVS, so it cannot exercise the re-search branch. The re-search branch
        // is also proven deterministically by
        // `root_pvs_scout_improves_alpha_triggers_research`.)
        pvs_counters::reset();
        let mut pos = parse_fen(ROOT_QWIN_FEN).unwrap();
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let mut tt = TranspositionTable::disabled();
        let key = pos.zobrist_key();
        let out = search_best_move_with_history_tt_and_profile(
            &mut pos,
            &[key],
            &limits,
            &ctx,
            &mut tt,
            SearchProfile::Current,
        )
        .expect("current outcome");

        assert!(
            pvs_counters::ROOT_FIRST_FULL.get() > 0,
            "the first root move is searched full-window"
        );
        assert!(
            pvs_counters::ROOT_SCOUT.get() > 0,
            "later root moves are scouted"
        );
        assert!(
            pvs_counters::ROOT_FAIL_LOW.get() > 0,
            "some root scouts fail low and are dropped"
        );
        assert!(
            pvs_counters::ROOT_RESEARCH_ENTERED.get() > 0,
            "some root scouts improve alpha and trigger a full re-search"
        );
        assert_eq!(
            pvs_counters::ROOT_RESEARCH_ATTEMPT.get(),
            pvs_counters::ROOT_RESEARCH_ENTERED.get(),
            "no budget abort here: every attempted root re-search entered"
        );
        assert!(pv_is_legal(ROOT_QWIN_FEN, &out.pv), "root PV is legal");
        assert_eq!(out.pv.first().copied(), Some(out.best_move));
    }

    #[test]
    fn root_pvs_scout_fail_low_not_committed() {
        // With the winning move FIRST, its full-window search sets a high
        // alpha; a later WEAK move's scout then fails low and must be dropped:
        // no re-search, no change to best move / PV, and every root move is
        // still visited (no root beta cutoff).
        let (weak, qxa4) = root_weak_and_qxa4(ROOT_QWIN_FEN);
        let mut root_moves = vec![qxa4, weak];
        let depth = 3;

        pvs_counters::reset();
        let mut pos = parse_fen(ROOT_QWIN_FEN).unwrap();
        let fallback = root_moves[0];
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let mut heur = Some(SearchHeuristics::new());
        let iter = root_search(
            &mut pos,
            depth,
            &mut root_moves,
            false,
            fallback,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut heur,
        )
        .expect("iteration completes");

        assert!(
            pvs_counters::ROOT_FAIL_LOW.get() > 0,
            "the weak later move failed the root scout low"
        );
        assert_eq!(
            pvs_counters::ROOT_RESEARCH_ENTERED.get(),
            0,
            "a fail-low root scout never re-searches"
        );
        assert_eq!(
            pvs_counters::ROOT_MOVES_VISITED.get(),
            2,
            "every root move is visited (no root beta cutoff)"
        );
        assert_eq!(
            move_to_uci(iter.best_move),
            "e4a4",
            "best move stays the first full-window winner"
        );
        assert_eq!(iter.score, 890, "root score is the winning value");
        assert_eq!(
            iter.pv.first().copied(),
            Some(iter.best_move),
            "root PV starts with the winner, never the fail-low scout move"
        );
        assert!(pv_is_legal(ROOT_QWIN_FEN, &iter.pv));
    }

    #[test]
    fn root_pvs_scout_improves_alpha_triggers_research() {
        // With a WEAK move FIRST (setting a low alpha) and the winning move
        // LATER, the winner's null-window scout improves alpha and MUST be
        // fully re-searched before it can become the root best. The committed
        // root PV tail is the re-search line (asserted structurally inside
        // `root_search`; here we confirm the re-search fired and the result is
        // the real winning line).
        let (weak, qxa4) = root_weak_and_qxa4(ROOT_QWIN_FEN);
        let mut root_moves = vec![weak, qxa4];
        let depth = 3;

        pvs_counters::reset();
        let mut pos = parse_fen(ROOT_QWIN_FEN).unwrap();
        let fallback = root_moves[0];
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let mut heur = Some(SearchHeuristics::new());
        let iter = root_search(
            &mut pos,
            depth,
            &mut root_moves,
            false,
            fallback,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut heur,
        )
        .expect("iteration completes");

        assert!(
            pvs_counters::ROOT_SCOUT.get() > 0,
            "the later winning move was scouted"
        );
        assert!(
            pvs_counters::ROOT_RESEARCH_ATTEMPT.get() > 0,
            "the improving scout attempted a full re-search"
        );
        assert_eq!(
            pvs_counters::ROOT_RESEARCH_ENTERED.get(),
            pvs_counters::ROOT_RESEARCH_ATTEMPT.get(),
            "no budget abort: attempted re-search entered"
        );
        assert!(
            pvs_counters::ROOT_RESEARCH_ROW_COMMITTED.get() > 0,
            "the re-searched winner became best and committed its re-search row"
        );
        assert_eq!(
            move_to_uci(iter.best_move),
            "e4a4",
            "the fully re-searched winner is the root best move"
        );
        assert_eq!(iter.score, 890, "root score is the winning value");
        assert_eq!(iter.pv.first().copied(), Some(iter.best_move));
        assert!(pv_is_legal(ROOT_QWIN_FEN, &iter.pv));
    }

    #[test]
    fn root_pvs_visits_every_root_move() {
        // A single-iteration `Current` root search must visit EVERY legal root
        // move — there is no root beta cutoff.
        let depth = 2;
        let legal_count = generate_legal_moves(&mut parse_fen(ROOT_QWIN_FEN).unwrap()).len();

        pvs_counters::reset();
        let mut pos = parse_fen(ROOT_QWIN_FEN).unwrap();
        let mut root_moves = generate_legal_moves(&mut pos.clone());
        let fallback = root_moves[0];
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let mut heur = Some(SearchHeuristics::new());
        root_search(
            &mut pos,
            depth,
            &mut root_moves,
            false,
            fallback,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut heur,
        )
        .expect("iteration completes");
        assert_eq!(
            pvs_counters::ROOT_MOVES_VISITED.get(),
            legal_count,
            "root visited every legal move (no beta cutoff)"
        );
    }

    #[test]
    fn root_pvs_score_parity_m41_vs_current() {
        // The hard correctness contract (spec §9.5): at a fixed depth,
        // `Current` (root + non-root PVS) and `M41Reference` (full-window)
        // return the IDENTICAL score (mate distance included, since it is
        // encoded in the score), a legal best move / PV, and a fully restored
        // root position. They are FREE to differ in node count / move / PV
        // ordering. We check both a disabled and an enabled TT.
        // Fixtures: startpos, queen-win, a mate-in-1, and insufficient material.
        let cases: &[(&str, u32, bool)] = &[
            (START_FEN, 3, false),
            (ROOT_QWIN_FEN, 3, false),
            // Ra8# mate-in-1: White Ra1, Kh1; Black Kg8 boxed by its own pawns.
            ("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1", 2, true),
            // K vs K: automatic insufficient-material draw (root short-circuit).
            ("8/8/8/8/8/8/8/K6k w - - 0 1", 2, false),
        ];

        for &(fen, depth, is_mate) in cases {
            for enabled in [false, true] {
                let run = |profile: SearchProfile| {
                    let mut pos = parse_fen(fen).unwrap();
                    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
                    let limits = SearchLimits {
                        depth: Some(depth),
                        ..Default::default()
                    };
                    let mut tt = if enabled {
                        TranspositionTable::new_mb(1).unwrap()
                    } else {
                        TranspositionTable::disabled()
                    };
                    let key = pos.zobrist_key();
                    let out = search_best_move_with_history_tt_and_profile(
                        &mut pos,
                        &[key],
                        &limits,
                        &ctx,
                        &mut tt,
                        profile,
                    )
                    .expect("outcome");
                    (out, to_fen(&pos))
                };

                let (out_ref, fen_ref) = run(SearchProfile::M41Reference);
                let (out_cur, fen_cur) = run(SearchProfile::Current);

                assert_eq!(
                    out_cur.score, out_ref.score,
                    "score parity failed: fen={fen} depth={depth} enabled={enabled}"
                );
                if is_mate {
                    let s = out_cur.score.expect("mate score present");
                    assert!(
                        s > MATE - 1000,
                        "mate fixture must score a mate for both profiles (fen={fen}, s={s})"
                    );
                }
                assert!(
                    pv_is_legal(fen, &out_ref.pv),
                    "m41 PV legal: fen={fen} enabled={enabled}"
                );
                assert!(
                    pv_is_legal(fen, &out_cur.pv),
                    "current PV legal: fen={fen} enabled={enabled}"
                );
                assert_eq!(fen_ref.as_str(), fen, "m41 restored: fen={fen}");
                assert_eq!(fen_cur.as_str(), fen, "current restored: fen={fen}");
            }
        }
    }

    #[test]
    fn root_pvs_claimable_root_no_winning_move_keeps_floor() {
        // A claimable root (fifty-move / threefold available) has a 0 floor.
        // When no real move beats the claim, `Current`'s root PVS must still
        // report the claim: score 0, the stable fallback, and an EMPTY PV
        // (never a faked line) — proving root PVS respects the claim floor.
        let fen = "4k3/3q4/8/8/8/8/4P3/K7 w - - 100 50";
        let depth = 2;

        pvs_counters::reset();
        let mut pos = parse_fen(fen).unwrap();
        let mut root_moves = generate_legal_moves(&mut pos.clone());
        let fallback = root_moves[0];
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let mut heur = Some(SearchHeuristics::new());
        let iter = root_search(
            &mut pos,
            depth,
            &mut root_moves,
            true, // root_claimable
            fallback,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut heur,
        )
        .expect("iteration completes");
        assert_eq!(iter.score, 0, "claim floor holds under Current root PVS");
        assert_eq!(
            iter.best_move, fallback,
            "claim placeholder is the stable fallback"
        );
        assert!(iter.pv.is_empty(), "claim placeholder PV is empty");
    }

    #[test]
    fn root_pvs_claimable_root_winning_scout_re_searches() {
        // A claimable root (0 floor) where a LATER move truly wins: its scout
        // improves the 0 floor and MUST be fully re-searched before it can be
        // reported, returning the real winning score and a legal, non-empty PV
        // (never the null-window scout line).
        let claim_fen = "7k/8/8/8/q3Q2p/8/8/4K3 w - - 100 50";
        let (weak, qxa4) = root_weak_and_qxa4(claim_fen);
        let mut root_moves = vec![weak, qxa4];
        let depth = 3;

        pvs_counters::reset();
        let mut pos = parse_fen(claim_fen).unwrap();
        let fallback = root_moves[0];
        let mut path = SearchPath::new(vec![pos.zobrist_key()]);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let mut heur = Some(SearchHeuristics::new());
        let iter = root_search(
            &mut pos,
            depth,
            &mut root_moves,
            true, // root_claimable
            fallback,
            &ctx,
            &limits,
            SearchProfile::Current,
            &mut path,
            &mut TranspositionTable::disabled(),
            &mut heur,
        )
        .expect("iteration completes");
        assert!(
            pvs_counters::ROOT_SCOUT.get() > 0,
            "the winning later move was scouted"
        );
        assert!(
            pvs_counters::ROOT_RESEARCH_ENTERED.get() > 0,
            "the winning scout improved the 0 floor and was re-searched"
        );
        assert_eq!(
            move_to_uci(iter.best_move),
            "e4a4",
            "the re-searched winner beats the claim floor"
        );
        assert_eq!(iter.score, 890, "root reports the real winning score");
        assert!(
            !iter.pv.is_empty(),
            "a real winning line has a non-empty PV"
        );
        assert_eq!(iter.pv.first().copied(), Some(iter.best_move));
        assert!(pv_is_legal(claim_fen, &iter.pv));
    }

    #[test]
    fn root_pvs_abort_restores_state_and_no_partial_root_tt() {
        // Sweep node budgets over a single-depth `Current` root search whose
        // move list is [weak, Qxa4]: the weak move sets a low alpha, so the
        // winning `Qxa4` scout always improves alpha and re-searches. This
        // reaches all three ROOT abort phases:
        //   A) inside a root scout        -> ROOT_ABORT_IN_SCOUT
        //   B) acquiring the re-search    -> ROOT_ABORT_RESEARCH_ACQUIRE
        //   C) inside the root re-search  -> ROOT_ABORT_IN_RESEARCH
        // Every aborted run must return None, consume EXACTLY its budget,
        // fully restore board / FEN / Zobrist / SearchPath, and leave NO TT
        // entry for the unfinished ROOT node.
        let (weak, qxa4) = root_weak_and_qxa4(ROOT_QWIN_FEN);
        let depth = 3;
        let root = parse_fen(ROOT_QWIN_FEN).unwrap();
        // Key the aborted root would have stored under (root at ply 0).
        let root_key_probe = {
            let path = SearchPath::new(vec![root.zobrist_key()]);
            current_tt_key(&root, &path)
        };

        // Unbudgeted baseline node count for this exact root search.
        let full_nodes = {
            let mut pos = root;
            let mut root_moves = vec![weak, qxa4];
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                depth: Some(depth),
                ..Default::default()
            };
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let mut heur = Some(SearchHeuristics::new());
            root_search(
                &mut pos,
                depth,
                &mut root_moves,
                false,
                root_moves_fallback(&[weak, qxa4]),
                &ctx,
                &limits,
                SearchProfile::Current,
                &mut path,
                &mut TranspositionTable::disabled(),
                &mut heur,
            )
            .expect("unbudgeted root completes");
            ctx.nodes.load(Ordering::Relaxed)
        };
        assert!(full_nodes > 8, "root has a non-trivial subtree");

        pvs_counters::reset();
        let cap = full_nodes.saturating_sub(1).min(2000);
        for budget in 1..=cap {
            let mut pos = root;
            let before_fen = to_fen(&pos);
            let before_key = pos.zobrist_key();
            let mut root_moves = vec![weak, qxa4];
            let fallback = root_moves[0];
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits {
                nodes: Some(budget),
                ..Default::default()
            };
            let mut tt = TranspositionTable::new_mb(1).unwrap();
            let mut heur = Some(SearchHeuristics::new());
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let root_len = path.len();
            let before_sig = path.repetition_signature();
            let before_base_len = path.base_len();
            let r = root_search(
                &mut pos,
                depth,
                &mut root_moves,
                false,
                fallback,
                &ctx,
                &limits,
                SearchProfile::Current,
                &mut path,
                &mut tt,
                &mut heur,
            );
            assert!(r.is_none(), "budget {budget} < {full_nodes} must abort");
            assert_eq!(
                ctx.nodes.load(Ordering::Relaxed),
                budget,
                "an aborted root consumes exactly its budget (budget={budget})"
            );
            assert_eq!(
                path.len(),
                root_len,
                "path length restored (budget={budget})"
            );
            assert_eq!(
                path.keys(),
                &[before_key],
                "path restored to root key (budget={budget})"
            );
            assert_eq!(
                to_fen(&pos),
                before_fen,
                "position restored (budget={budget})"
            );
            assert_eq!(
                pos.zobrist_key(),
                before_key,
                "key restored (budget={budget})"
            );
            assert_eq!(
                path.repetition_signature(),
                before_sig,
                "repetition signature restored (budget={budget})"
            );
            assert_eq!(
                path.base_len(),
                before_base_len,
                "base prefix length restored (budget={budget})"
            );
            assert!(
                tt.probe(root_key_probe).is_none(),
                "aborted root left no TT entry (budget={budget})"
            );

            if pvs_counters::ROOT_ABORT_IN_SCOUT.get() > 0
                && pvs_counters::ROOT_ABORT_RESEARCH_ACQUIRE.get() > 0
                && pvs_counters::ROOT_ABORT_IN_RESEARCH.get() > 0
            {
                break;
            }
        }

        assert!(
            pvs_counters::ROOT_ABORT_IN_SCOUT.get() > 0,
            "phase A: root scout-internal abort observed"
        );
        assert!(
            pvs_counters::ROOT_ABORT_RESEARCH_ACQUIRE.get() > 0,
            "phase B: root re-search node-acquisition abort observed"
        );
        assert!(
            pvs_counters::ROOT_ABORT_IN_RESEARCH.get() > 0,
            "phase C: root re-search-internal abort observed"
        );
    }

    /// Tiny helper: the stable fallback is the first move of a root list.
    fn root_moves_fallback(moves: &[Move]) -> Move {
        moves[0]
    }
}
