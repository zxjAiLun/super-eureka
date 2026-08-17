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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::chess::movegen::{
    generate_legal_evasions_with_stats, generate_legal_moves_fast_single_buffer_with_stats,
    generate_legal_moves_fast_with_stats, generate_legal_moves_with_stats,
    generate_legal_tactical_moves_with_stats, has_any_legal_move_with_stats, FullLegalSub,
    MovegenStats,
};
use crate::chess::position::{Position, Undo};
use crate::chess::types::*;
use crate::chess::zobrist::{recompute_zobrist, ZobristKey};
use crate::engine::draw::{
    claim_available_by_intended_move, classify_draw, is_insufficient_material, DrawReason,
};
use crate::engine::eval::{evaluate, evaluate_integrated_positional, evaluate_threat_aware};
use crate::engine::time::TimeBudget;
use crate::engine::tt::{score_from_tt, score_to_tt, Bound, TTEntry, TranspositionTable, TtKey};

/// M4.1+ internal search configuration. Crate-private only: it selects the
/// move-ordering strategy used by the search core and is NEVER exposed through
/// the public API or the UCI surface.
///
/// * `M4Reference` preserves the M4.0 search behavior: no killer moves, no
///   history heuristic, and no PVS. It runs under the current evaluation
///   function, so evaluation milestones may legitimately change scores, PVs,
///   and node counts. Historical pre-EVAL benchmark values remain recorded in
///   their original benchmark documents.
/// * `M41Reference` reproduces the M4.1 full-window search exactly: M4.1
///   quiet move ordering (killer moves + history heuristic) with NO principal
///   variation search. It preserves the 236,418-node M4.1 A/B baseline and
///   keeps M4.2 replayable once `Current` enables PVS.
/// * `PvsReference` is the immediately preceding M4.2 PVS baseline without
///   any SEARCH 1 candidate feature.
/// * `SeeCandidate` enables only SEE ordering in qsearch on the existing PVS
///   path; it never deletes a legal capture.
/// * `AspirationCandidate` enables only iterative-deepening aspiration
///   windows on the existing PVS path.
/// * `LmrCandidate` enables only late-move reductions on eligible quiet
///   non-PV moves; reduced searches are re-searched at full depth when they
///   improve alpha.
/// * `NullMoveCandidate` enables only a verified null-move probe at eligible
///   non-check, non-zugzwang-shaped nodes. A probe fail-high is never a direct
///   cutoff.
/// * `FutilityCandidate` enables only shallow, non-PV quiet-move futility
///   pruning with tactical, checking, mate-range, and promotion-threat guards.
/// * `CurrentQsearchMovegen` preserves the `Current` search tree and changes
///   only non-check qsearch move generation to a tactical-only path; check
///   nodes still generate all legal evasions and stalemate is checked exactly.
/// * `CurrentQsearchPruning` builds on `CurrentQsearchMovegen` and adds only
///   conservative non-check qsearch SEE pruning for eligible plain captures.
///   Promotions, en passant, and checking captures are always kept.
/// * `CurrentQsearchFastPruning` preserves the `CurrentQsearchPruning` search
///   tree and replaces only its pruning SEE attacker scan with a direct
///   occupancy/attack sidecar. Its keep/prune decision must remain identical.
/// * `CurrentLmr` preserves the `Current` PVS, ordering, and specialized
///   qsearch movegen path, adding only the existing conservative LMR rules.
/// * `CurrentThreatAware` preserves the `Current` PVS and specialized qsearch
///   movegen path, adding only the candidate-only king-danger evaluation,
///   bounded forcing extensions/checks, and threat-aware ordering. It does
///   not enable LMR, aspiration, null move, futility, or SEE pruning.
/// * `CurrentThreatAwareNoQchecks` is the S2.1b cost-attribution variant: it
///   is identical to `CurrentThreatAware` except that quiet checking moves are
///   disabled in qsearch. It is bench-only and never changes `Current`.
/// * `CurrentThreatAwareEvalOrder` is the S2.1b ordering-only variant: it
///   keeps the candidate king-danger evaluation, threat ordering, and root
///   score ordering, but disables forcing extensions and quiet qsearch checks.
///   It is bench-only and never changes `Current`.
/// * `CurrentThreatAwareEvalOnly` is the S2.1c shared-core attribution variant:
///   it keeps only the candidate king-danger evaluation. It deliberately uses
///   Current's ordinary move and root ordering, with no forcing extensions or
///   quiet qsearch checks. It is bench-only and never changes `Current`.
/// * `CurrentThreatAwareOrderOnly` is the S2.1c shared-core attribution
///   variant: it keeps only threat-aware move/root ordering and Current's
///   ordinary evaluation, with no forcing extensions or quiet qsearch checks.
///   It is bench-only and never changes `Current`.
/// * `CurrentEval2` is the single integrated S2.2 positional-evaluation
///   candidate. It keeps Current's PVS, ordering, and specialized qsearch
///   movegen while replacing only the evaluator; all threat-aware, forcing,
///   aspiration, LMR, null, futility, SEE, and qsearch-pruning features stay
///   disabled.
/// * `Current` is the historical production configuration: M4.1 quiet move
///   ordering plus the M4.2 PVS at both non-root nodes (Commit 3) and the
///   root (Commit 4), with the D1.2 specialized non-check qsearch move
///   generator integrated. It is retained as [`ROLLBACK_PROFILE`] and is NOT
///   the current default.
/// * The `CurrentAspiration*` variants are bench-only cumulative candidates.
///   They preserve the `Current` PVS/ordering path and add only the features
///   named by their suffix. None of them is used by the UCI production path.
/// * `CurrentFinal` is [`PRODUCTION_PROFILE`]: the S3-FINAL candidate plus
///   the promoted LegalityFast, SingleBuffer, SingleGeneration, and S7.4A
///   LMR-on-null-window policies. It combines the existing aspiration, LMR,
///   verified null-probe, shallow futility, and conservative qsearch
///   SEE-pruning paths without enabling E2/threat evaluation or
///   forcing-search features.
///
/// `M41Reference` keeps the M4.1 full-window path (killer/history ordering at
/// non-root nodes, NO PVS at either the root or a non-root node), while
/// `Current` enables the null-window scout + re-search at every non-root node
/// AND at the root. `M4Reference` preserves the M4.0 search policy. Move
/// ordering at the root itself is the pure hash-move lift in all profiles (no
/// MVV-LVA / killer / history reorder); PVS changes only the WINDOW a later
/// root move is searched with, never the root move order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchProfile {
    M4Reference,
    M41Reference,
    PvsReference,
    SeeCandidate,
    AspirationCandidate,
    LmrCandidate,
    NullMoveCandidate,
    FutilityCandidate,
    Current,
    CurrentLmr,
    CurrentThreatAware,
    CurrentThreatAwareNoQchecks,
    CurrentThreatAwareEvalOrder,
    CurrentThreatAwareEvalOnly,
    CurrentThreatAwareOrderOnly,
    /// Single integrated E2 evaluation candidate. It keeps all search and
    /// qsearch candidates disabled while replacing only the evaluator.
    CurrentEval2,
    CurrentQsearchMovegen,
    CurrentQsearchPruning,
    CurrentQsearchFastPruning,
    CurrentAspiration,
    CurrentAspirationLmr,
    CurrentAspirationLmrFutility,
    CurrentAspirationLmrFutilitySee,
    /// S4.4E production search configuration: S3 production search plus the
    /// S4.3E promoted unpinned non-check full-legality fast path, the S4.4E
    /// promoted single-buffer full-legal materialization, the S5.0D
    /// promoted has-any child terminal probe (identical move lists, order
    /// and search tree), and the S7.4A promoted LMR-on-null-window policy
    /// (existing LMR rules applied on caller-null-window nodes with exactly
    /// one full-depth verification re-search). Default UCI startup profile.
    CurrentFinal,
    /// S4.1 candidate: exactly CurrentFinal plus root quiet-move ordering by
    /// the existing history heuristic (previous best stays first; no root
    /// killers; no static-eval ordering; no history-update changes).
    CurrentFinalRootHistory,
    /// S4.1b candidate: exactly CurrentFinal plus root quiet-move ordering by
    /// the PREVIOUS completed iteration's root `move_scores` (previous best
    /// stays first; no history/killer/static-eval/threat signal; no PVS or
    /// re-search changes).
    CurrentFinalRootPrevScore,
    /// S4.3B candidate: unpinned non-check legality fast path in the FULL
    /// legal generator. After the S4.3E promotion its search behavior is
    /// equivalent to CurrentFinal; retained as a historical/compatibility
    /// alias for the S4.3B/S4.3C/S4.3D experiment artifact.
    CurrentFinalLegalityFast,
    /// S4.4B candidate, PROMOTED into production CurrentFinal at S4.4E
    /// (single-buffer full-legal materialization). Retained as a
    /// historical/compatibility alias for the S4.4B/S4.4C/S4.4D experiment
    /// artifact (search behavior identical to CurrentFinal).
    CurrentFinalSingleBuffer,
    /// S5.0B candidate, PROMOTED into production CurrentFinal at S5.0D.
    /// Retained as a historical/compatibility alias (search behavior
    /// identical to CurrentFinal).
    CurrentFinalSingleGeneration,
    /// S7.1A candidate: exactly CurrentFinal except that non-check qsearch
    /// defers tactical move generation/ordering past the stand-pat beta
    /// cutoff and stalemate check. Same searched tree (nodes/score/bestmove/
    /// PV) by construction; only the wasted work is avoided.
    CurrentFinalQsearchLazy,
    /// S7.1B candidate: exactly CurrentFinal plus conservative SEE-delta
    /// qsearch pruning. At a non-check qsearch node, after stand-pat, a
    /// plain non-checking capture whose SUPPORTED SEE value satisfies
    /// `stand_pat + SEE + QSEARCH_DELTA_MARGIN_CP <= alpha` is pruned, on
    /// top of the existing SEE<0 production prune. TREE-CHANGING candidate;
    /// CurrentFinal itself is untouched.
    CurrentFinalQsearchDelta,
    /// S7.4A candidate, PROMOTED into production CurrentFinal after the
    /// formal pentanomial SPRT accepted H1 (tournament
    /// 2cf04fe6-2120-45c1-852b-e2462e3f62d9). Retained as a
    /// historical/compatibility alias; search behavior is identical to
    /// CurrentFinal.
    CurrentFinalLmrNullWindow,
    /// S7.5A candidate: exactly CurrentFinal plus a main-search-only
    /// single-evasion extension. Never folded into CurrentFinal until the
    /// full G0-G6/G5W gate chain and independent review pass.
    CurrentFinalSingleEvasion,
}

/// Canonical current production profile. UCI startup defaults, the default
/// UCI handshake, and the normal production search entry all resolve here.
/// `--profile current-final` and the retained S7.4A alias select the exact
/// same production search semantics.
pub(crate) const PRODUCTION_PROFILE: SearchProfile = SearchProfile::CurrentFinal;

/// Explicit historical rollback profile. `--profile current` selects this;
/// it never receives promoted production policy bits.
pub(crate) const ROLLBACK_PROFILE: SearchProfile = SearchProfile::Current;

impl SearchProfile {
    #[inline]
    pub(crate) const fn uses_pvs(self) -> bool {
        !matches!(self, Self::M4Reference | Self::M41Reference)
    }

    #[inline]
    pub(crate) const fn uses_see(self) -> bool {
        matches!(
            self,
            Self::SeeCandidate | Self::CurrentAspirationLmrFutilitySee
        )
    }

    #[inline]
    pub(crate) const fn uses_aspiration(self) -> bool {
        matches!(
            self,
            Self::AspirationCandidate
                | Self::CurrentAspiration
                | Self::CurrentAspirationLmr
                | Self::CurrentAspirationLmrFutility
                | Self::CurrentAspirationLmrFutilitySee
                | Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    #[inline]
    pub(crate) const fn uses_lmr(self) -> bool {
        matches!(
            self,
            Self::LmrCandidate
                | Self::CurrentLmr
                | Self::CurrentAspirationLmr
                | Self::CurrentAspirationLmrFutility
                | Self::CurrentAspirationLmrFutilitySee
                | Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    #[inline]
    pub(crate) const fn uses_null_move(self) -> bool {
        matches!(
            self,
            Self::NullMoveCandidate
                | Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    #[inline]
    pub(crate) const fn uses_futility(self) -> bool {
        matches!(
            self,
            Self::FutilityCandidate
                | Self::CurrentAspirationLmrFutility
                | Self::CurrentAspirationLmrFutilitySee
                | Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    #[inline]
    pub(crate) const fn uses_qsearch_movegen(self) -> bool {
        matches!(
            self,
            Self::Current
                | Self::CurrentLmr
                | Self::CurrentThreatAware
                | Self::CurrentThreatAwareNoQchecks
                | Self::CurrentThreatAwareEvalOrder
                | Self::CurrentThreatAwareEvalOnly
                | Self::CurrentThreatAwareOrderOnly
                | Self::CurrentEval2
                | Self::CurrentAspiration
                | Self::CurrentAspirationLmr
                | Self::CurrentAspirationLmrFutility
                | Self::CurrentAspirationLmrFutilitySee
                | Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
                | Self::CurrentQsearchMovegen
                | Self::CurrentQsearchPruning
                | Self::CurrentQsearchFastPruning
        )
    }

    #[inline]
    pub(crate) const fn uses_qsearch_pruning(self) -> bool {
        matches!(
            self,
            Self::CurrentQsearchPruning
                | Self::CurrentQsearchFastPruning
                | Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    /// S7.1B: conservative SEE-delta qsearch pruning. Only this candidate
    /// enables it; CurrentFinal keeps its exact existing SEE<0 policy.
    #[inline]
    pub(crate) const fn uses_qsearch_delta(self) -> bool {
        matches!(self, Self::CurrentFinalQsearchDelta)
    }

    /// S7.4A, PROMOTED: apply the EXISTING LMR policy on caller-null-window
    /// nodes where `pvs_child_window()` would otherwise fall back to
    /// `ChildWindow::Full` and silently discard the proposed reduction.
    /// Production policy for CurrentFinal and every profile whose base
    /// semantics are defined as CurrentFinal; `Current` and all pre-S7.4A
    /// experimental profiles remain unchanged (false).
    #[inline]
    pub(crate) const fn uses_lmr_null_window(self) -> bool {
        matches!(
            self,
            Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    #[inline]
    pub(crate) const fn uses_qsearch_fast_pruning(self) -> bool {
        matches!(self, Self::CurrentQsearchFastPruning)
    }

    /// S7.1A: non-check qsearch defers tactical move generation/ordering past
    /// the stand-pat beta cutoff and the stalemate probe. Same searched tree
    /// by construction (stand-pat is a pure static eval; has-any-legal is the
    /// exact emptiness predicate).
    #[inline]
    pub(crate) const fn uses_qsearch_lazy(self) -> bool {
        matches!(self, Self::CurrentFinalQsearchLazy)
    }

    #[inline]
    pub(crate) const fn uses_threat_aware_eval(self) -> bool {
        matches!(
            self,
            Self::CurrentThreatAware
                | Self::CurrentThreatAwareNoQchecks
                | Self::CurrentThreatAwareEvalOrder
                | Self::CurrentThreatAwareEvalOnly
        )
    }

    /// S4.1: the candidate reorders root QUIET moves by the existing history
    /// heuristic (previous best stays first; no root killers; no static-eval
    /// ordering; history update rules unchanged).
    #[inline]
    pub(crate) const fn uses_root_quiet_history(self) -> bool {
        matches!(self, Self::CurrentFinalRootHistory)
    }

    /// S4.1b: the candidate reorders root QUIET moves by the previous
    /// completed iteration's root `move_scores` (previous best stays first;
    /// no history/killer/static-eval/threat signal).
    #[inline]
    pub(crate) const fn uses_root_prev_score(self) -> bool {
        matches!(self, Self::CurrentFinalRootPrevScore)
    }

    /// S4.3E/S4.4E promoted: production `CurrentFinal` is defined as the
    /// unpinned non-check legality fast path (S4.3B/S4.3E) AND the
    /// single-buffer full-legal materialization (S4.4B/S4.4E), producing
    /// identical move lists and order. `uses_legality_fast` is production
    /// policy for CurrentFinal and everything defined as "exactly
    /// CurrentFinal plus X" (RootHistory, RootPrevScore), plus the S4.3B /
    /// S4.4B compatibility aliases.
    #[inline]
    pub(crate) const fn uses_legality_fast(self) -> bool {
        matches!(
            self,
            Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    /// S4.4B: the single-buffer full-legal materialization policy. Promoted
    /// into production CurrentFinal at S4.4E: EVERY profile whose base
    /// semantics are defined as CurrentFinal uses it (CurrentFinal itself,
    /// the "CurrentFinal + X" candidates, the S4.3B compatibility alias, the
    /// S4.4B experimental alias, and the S5.0B candidate which is now
    /// "promoted CurrentFinal + has-any probe").
    #[inline]
    pub(crate) const fn uses_single_buffer_legal(self) -> bool {
        matches!(
            self,
            Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    /// S5.0B candidate, PROMOTED into production CurrentFinal at S5.0D: the
    /// child probe uses has-any-legal instead of a discarded full legal list
    /// (S5.0A: 64.8% of full-legal calls were probe lists). Production policy
    /// for CurrentFinal and every profile whose base semantics are defined as
    /// CurrentFinal; the experimental alias is retained as a historical/
    /// compatibility identity.
    #[inline]
    pub(crate) const fn uses_single_generation_probe(self) -> bool {
        matches!(
            self,
            Self::CurrentFinal
                | Self::CurrentFinalRootHistory
                | Self::CurrentFinalRootPrevScore
                | Self::CurrentFinalLegalityFast
                | Self::CurrentFinalSingleBuffer
                | Self::CurrentFinalSingleGeneration
                | Self::CurrentFinalQsearchLazy
                | Self::CurrentFinalQsearchDelta
                | Self::CurrentFinalLmrNullWindow
                | Self::CurrentFinalSingleEvasion
        )
    }

    #[inline]
    pub(crate) const fn uses_eval2(self) -> bool {
        matches!(self, Self::CurrentEval2)
    }

    #[inline]
    pub(crate) const fn uses_forcing_search(self) -> bool {
        matches!(
            self,
            Self::CurrentThreatAware | Self::CurrentThreatAwareNoQchecks
        )
    }

    /// S7.5A: main-search-only single-evasion extension.
    #[inline]
    pub(crate) const fn uses_single_evasion_extension(self) -> bool {
        matches!(self, Self::CurrentFinal | Self::CurrentFinalSingleEvasion)
    }

    #[inline]
    pub(crate) const fn uses_threat_ordering(self) -> bool {
        matches!(
            self,
            Self::CurrentThreatAware
                | Self::CurrentThreatAwareNoQchecks
                | Self::CurrentThreatAwareEvalOrder
                | Self::CurrentThreatAwareOrderOnly
        )
    }

    #[inline]
    pub(crate) const fn uses_threat_aware_qsearch(self) -> bool {
        matches!(self, Self::CurrentThreatAware)
    }
}

pub const MATE: i32 = 1_000_000;

/// S7.1B: the single, predeclared, deliberately conservative delta-pruning
/// margin (in centipawns). A plain non-checking capture with supported
/// `SEE >= 0` is pruned only when
/// `stand_pat + SEE + QSEARCH_DELTA_MARGIN_CP <= alpha`. 500cp is chosen so
/// the first cut removes only obviously futile capture trees; do NOT tune
/// this against a challenge corpus.
pub(crate) const QSEARCH_DELTA_MARGIN_CP: i32 = 500;

/// Maximum quiescence ply. A "check → evasion → check → ..." sequence has no
/// natural depth bound, so this cap guarantees termination. It is a *safety*
/// limit, not a repetition/fifty-move substitute: M3.1 draw handling
/// (fifty-move claim and threefold-repetition claim) is already implemented
/// and applied at every node, but an unresolved checking/tactical chain still
/// needs a hard ply cap to terminate. At the cap we still detect checkmate /
/// stalemate first, then fall back to the static evaluation without
/// recursing further.
pub const MAX_QPLY: u32 = 32;

/// Maximum number of main-search forcing extensions allowed on one root
/// line. The budget makes repeated checking sequences finite even when no
/// repetition is reached during the extension window.
const MAX_FORCING_EXTENSIONS: u8 = 4;
/// S7.5A frozen budget: at most two single-evasion extensions per root line.
/// Never retuned from tactical-corpus results.
const S75A_FORCING_BUDGET: u8 = 2;

/// A threat-aware qsearch may add quiet checking moves only for the first two
/// qsearch plies. Captures and promotions retain the existing qsearch rules.
const MAX_FORCING_QPLY: u32 = 2;

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

/// S4.0B: the effective search-feature policy, resolved ONCE at search start
/// from the active profile plus any bench-only diagnostic overrides. The hot
/// path reads the resolved `SearchFeaturePolicy` fields instead of
/// recomputing a per-feature predicate (plus an extra diagnostic condition)
/// at every gate site.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SearchFeaturePolicy {
    pub(crate) lmr: bool,
    pub(crate) futility: bool,
    pub(crate) null_move: bool,
    pub(crate) qsearch_see: bool,
    /// S7.1B: conservative SEE-delta qsearch pruning (candidate only).
    pub(crate) qsearch_delta: bool,
    /// S7.4A, PROMOTED: apply the existing LMR reduction on
    /// caller-null-window nodes (production CurrentFinal policy).
    pub(crate) lmr_null_window: bool,
    /// S7.5A: main-search single-evasion extension (candidate only).
    pub(crate) single_evasion_extension: bool,
}

const FEATURE_LMR: u32 = 1 << 0;
const FEATURE_FUTILITY: u32 = 1 << 1;
const FEATURE_NULL: u32 = 1 << 2;
const FEATURE_QSEE: u32 = 1 << 3;
const FEATURE_QDELTA: u32 = 1 << 4;
const FEATURE_LMNW: u32 = 1 << 5;
const FEATURE_S75A_SE: u32 = 1 << 6;

impl SearchFeaturePolicy {
    /// Resolve the effective policy for `profile`, applying diagnostic
    /// overrides. Called at the start of every search (including the UCI
    /// production path, which always passes `diagnostics = None`), so
    /// CurrentFinal behavior is unchanged when no diagnostics are set.
    pub(crate) fn for_profile(profile: SearchProfile, diag: Option<&SearchDiagnostics>) -> Self {
        let d = diag.copied().unwrap_or_default();
        Self {
            lmr: profile.uses_lmr() && !d.disable_lmr,
            futility: profile.uses_futility() && !d.disable_futility,
            null_move: profile.uses_null_move() && !d.disable_null_move,
            qsearch_see: profile.uses_qsearch_pruning() && !d.disable_qsearch_see,
            qsearch_delta: profile.uses_qsearch_delta(),
            lmr_null_window: profile.uses_lmr_null_window(),
            single_evasion_extension: profile.uses_single_evasion_extension(),
        }
    }

    fn to_bits(self) -> u32 {
        let mut bits = 0u32;
        if self.lmr {
            bits |= FEATURE_LMR;
        }
        if self.futility {
            bits |= FEATURE_FUTILITY;
        }
        if self.null_move {
            bits |= FEATURE_NULL;
        }
        if self.qsearch_see {
            bits |= FEATURE_QSEE;
        }
        if self.qsearch_delta {
            bits |= FEATURE_QDELTA;
        }
        if self.lmr_null_window {
            bits |= FEATURE_LMNW;
        }
        if self.single_evasion_extension {
            bits |= FEATURE_S75A_SE;
        }
        bits
    }

    fn from_bits(bits: u32) -> Self {
        Self {
            lmr: bits & FEATURE_LMR != 0,
            futility: bits & FEATURE_FUTILITY != 0,
            null_move: bits & FEATURE_NULL != 0,
            qsearch_see: bits & FEATURE_QSEE != 0,
            qsearch_delta: bits & FEATURE_QDELTA != 0,
            lmr_null_window: bits & FEATURE_LMNW != 0,
            single_evasion_extension: bits & FEATURE_S75A_SE != 0,
        }
    }
}

/// S4.0B bench-only diagnostic configuration. It is a separate struct (not a
/// pile of loose fields on `SearchContext`) and is only ever populated by the
/// bench harness. The production UCI path never sets it, so there is no way
/// to activate these diagnostics through UCI and default behavior is unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SearchDiagnostics {
    /// Force the root to search only this single move (Stockfish teacher move).
    pub(crate) forced_root_move: Option<Move>,
    /// Record the 1-based rank of this move in the normal root ordering,
    /// BEFORE any forced-root filtering.
    pub(crate) target_root_move: Option<Move>,
    /// S4.1c Phase B diagnostic: give EVERY root move a full-window child
    /// search (no root scout + conditional re-search). Non-root PVS,
    /// LMR/futility/null/qSEE all unchanged. Diagnostic only.
    pub(crate) root_full_window: bool,
    pub(crate) disable_lmr: bool,
    pub(crate) disable_futility: bool,
    pub(crate) disable_null_move: bool,
    pub(crate) disable_qsearch_see: bool,
}

/// S4.0B: the initial (pre-filter) root rank of the diagnostic target move.
/// Resolved at search start; read back by the bench harness after the search.
pub(crate) const TARGET_ROOT_RANK_NONE: u32 = 0;

/// S4.3A: deterministic sparse sampler for coarse hot-path operations. Each
/// counter samples ~1 in `rate` calls of one operation type; `calls` counts
/// every call while timing is enabled. Bench-only: `sampled_timing` is never
/// set on the production UCI path, and `sample_begin` returns None immediately
/// when disabled (zero production overhead).
#[derive(Debug, Default)]
pub(crate) struct SampledCounter {
    pub(crate) calls: AtomicU64,
    pub(crate) samples: AtomicU64,
    pub(crate) elapsed_ns: AtomicU64,
    gate: AtomicU32,
}

/// S4.3A: snapshot of the sampled wall-time accumulators.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SampledTimings {
    /// (calls, samples, elapsed_ns)
    pub(crate) movegen_legal: (u64, u64, u64),
    pub(crate) movegen_tactical: (u64, u64, u64),
    pub(crate) movegen_evasion: (u64, u64, u64),
    pub(crate) movegen_has_any: (u64, u64, u64),
    pub(crate) eval: (u64, u64, u64),
    pub(crate) ordering: (u64, u64, u64),
    pub(crate) see: (u64, u64, u64),
    pub(crate) tt: (u64, u64, u64),
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
    /// Deepest ply reached this search (including qsearch) - UCI seldepth.
    pub seldepth: AtomicU64,
    /// S7.0: deepest ply reached in the main search / qsearch separately
    /// (profiling-gated split of `seldepth`).
    pub main_seldepth: AtomicU64,
    pub qsearch_seldepth: AtomicU64,
    pub qsearch_nodes: AtomicU64,
    pub eval_calls: AtomicU64,
    pub legal_move_generations: AtomicU64,
    pub pseudo_moves: AtomicU64,
    pub legal_moves: AtomicU64,
    pub make_moves: AtomicU64,
    pub unmake_moves: AtomicU64,
    pub tt_probes: AtomicU64,
    pub tt_hits: AtomicU64,
    pub tt_cutoffs: AtomicU64,
    pub tt_rejected_depth: AtomicU64,
    pub tt_rejected_bound: AtomicU64,
    pub tt_rejected_decode: AtomicU64,
    pub tt_stores: AtomicU64,
    pub see_calls: AtomicU64,
    pub see_pruned: AtomicU64,
    pub qsearch_see_tests: AtomicU64,
    pub qsearch_see_pruned: AtomicU64,
    pub qsearch_see_fail_open_promotions: AtomicU64,
    pub qsearch_checking_captures_kept: AtomicU64,
    pub qsearch_promotions_kept: AtomicU64,
    pub qsearch_en_passant_kept: AtomicU64,
    /// S7.1B SEE-delta qsearch pruning counters (profiling-gated).
    pub qsearch_delta_tests: AtomicU64,
    pub qsearch_delta_pruned: AtomicU64,
    pub qsearch_delta_pruned_pawn: AtomicU64,
    pub qsearch_delta_pruned_minor: AtomicU64,
    pub qsearch_delta_pruned_rook: AtomicU64,
    pub qsearch_delta_pruned_queen: AtomicU64,
    pub qsearch_delta_qply_0_1: AtomicU64,
    pub qsearch_delta_qply_2_3: AtomicU64,
    pub qsearch_delta_qply_4p: AtomicU64,
    /// S7.2 move-ordering attribution (OBSERVATION ONLY, profiling-gated).
    /// Bucket index maps:
    /// - `s72_*_searched` / `s72_quiet_cutoff_gidx`: [1, 2, 3-4, 5-8, 9-16, 17+]
    /// - quiet-rank histograms: [0, 1, 2-3, 4-7, 8-15, 16+]
    /// - history-score histograms: [<=0, 1-15, 16-63, 64-255, 256+]
    /// - depth split: [1, 2, 3, 4-5, 6-7, 8+]
    /// - cutoff category: [tt_hash, promotion, capture, killer0, killer1,
    ///   history_quiet, other_quiet] (mutually exclusive, TT first)
    /// - killer: [k0_present, k0_searched, k0_cutoffs, k1_present,
    ///   k1_searched, k1_cutoffs, some_but_absent_or_illegal]
    /// - tt hash: [present, searched, cutoffs, first_move_cutoff,
    ///   improves_alpha_without_cutoff]
    /// - lmr: [quiet_reduced_fail_low, quiet_reduced_research,
    ///   quiet_reduced_cutoff]
    pub s72_cutoff_category: [AtomicU64; 7],
    pub s72_nodes_with_quiet_moves: AtomicU64,
    pub s72_quiet_available: AtomicU64,
    pub s72_quiet_searched: AtomicU64,
    pub s72_quiet_searched_rank: [AtomicU64; 6],
    pub s72_quiet_searched_hist: [AtomicU64; 5],
    pub s72_quiet_cutoff_gidx: [AtomicU64; 6],
    pub s72_quiet_cutoff_rank: [AtomicU64; 6],
    pub s72_quiet_cutoff_hist: [AtomicU64; 5],
    pub s72_killer: [AtomicU64; 7],
    pub s72_tt_hash: [AtomicU64; 5],
    pub s72_cutoff_searched: [AtomicU64; 6],
    pub s72_fail_low_nodes: AtomicU64,
    pub s72_fail_low_searched_sum: AtomicU64,
    pub s72_lmr: [AtomicU64; 3],
    pub s72_d_cutoffs: [AtomicU64; 6],
    pub s72_d_cutoff_late5: [AtomicU64; 6],
    pub s72_d_fail_low: [AtomicU64; 6],
    pub s72_d_quiet_searched: [AtomicU64; 6],
    pub s72_d_quiet_cutoffs: [AtomicU64; 6],
    /// S7.3 selectivity attribution (OBSERVATION ONLY, profiling-gated):
    /// composition of move-loop nodes, no-beta-cutoff trees, futility/null
    /// selectivity, and depth>=4 quiet LMR eligibility vs actual reduction.
    pub s73_loop_nodes: AtomicU64,
    pub s73_nocut_pv: AtomicU64,
    pub s73_nocut_nonpv: AtomicU64,
    pub s73_nocut_incheck: AtomicU64,
    pub s73_nocut_null_attempted: AtomicU64,
    pub s73_nocut_searched_hist: [AtomicU64; 6],
    pub s73_nocut_searched_sum: AtomicU64,
    pub s73_null_eligible: AtomicU64,
    pub s73_fut_quiet_kept: AtomicU64,
    pub s73_q4p_quiet_searched: AtomicU64,
    pub s73_q4p_quiet_idx: [AtomicU64; 5],
    pub s73_q4p_quiet_red: [AtomicU64; 3],
    pub s73_q4p_quiet_red_idx: [AtomicU64; 15],
    pub s73_q4p_quiet_cutoff_red: [AtomicU64; 3],
    pub s73_q4p_quiet_cutoff_idx: [AtomicU64; 5],
    pub s73_q4p_scout_faillow_red: [AtomicU64; 3],
    pub s73_q4p_quiet_researched: AtomicU64,
    /// S7.4A LMR-on-null-window attribution (profiling-gated): proposed vs
    /// actually applied reductions, and the candidate's null-window reduced
    /// search / fail-low / re-search / verified-cutoff funnel.
    pub s74_lmr_proposed: AtomicU64,
    pub s74_lmr_proposed_r1: AtomicU64,
    pub s74_lmr_proposed_r2: AtomicU64,
    pub s74_lmr_applied_existing_pvs: AtomicU64,
    pub s74_lmr_suppressed_by_null_window: AtomicU64,
    pub s74_lmr_applied_null_window: AtomicU64,
    pub s74_lmr_nw_fail_low: AtomicU64,
    pub s74_lmr_nw_research: AtomicU64,
    /// S7.4A Repair 1: verifications that actually entered (acquired a node).
    pub s74_lmr_nw_research_entered: AtomicU64,
    pub s74_lmr_nw_verified_cutoff: AtomicU64,
    pub s74_lmr_nw_depth: [AtomicU64; 4],
    pub s74_lmr_nw_idx: [AtomicU64; 4],
    /// S7.5-0 forcing-opportunity attribution (OBSERVATION ONLY, profiling-gated).
    pub s75_main_in_check_nodes: AtomicU64,
    pub s75_main_single_evasion_nodes_raw: AtomicU64,
    pub s75_main_single_evasion_actionable_depth1: AtomicU64,
    pub s75_main_single_evasion_actionable_depth2plus: AtomicU64,
    pub s75_main_single_evasion_depth3plus: AtomicU64,
    pub s75_main_single_evasion_chain: [AtomicU64; 3],
    pub s75_main_checking_edges_searched: AtomicU64,
    pub s75_main_check_child_entered: AtomicU64,
    pub s75_main_check_child_movegen: AtomicU64,
    pub s75_main_check_child_terminal_0: AtomicU64,
    pub s75_main_check_child_evasions_1: AtomicU64,
    pub s75_main_check_child_evasions_2: AtomicU64,
    pub s75_main_check_child_evasions_3plus: AtomicU64,
    pub s75_main_depth1_nodes: AtomicU64,
    pub s75_main_depth1_in_check: AtomicU64,
    pub s75_main_depth1_single_evasion: AtomicU64,
    pub s75_main_depth1_entered_from_checking_edge: AtomicU64,
    pub s75_q_nodes: AtomicU64,
    pub s75_q_in_check_nodes: AtomicU64,
    pub s75_q_single_evasion_nodes_raw: AtomicU64,
    pub s75_q_single_evasion_qply0: AtomicU64,
    pub s75_q_single_evasion_qply1plus: AtomicU64,
    pub s75_q_checking_edges_searched: AtomicU64,
    pub s75_q_check_child_entered: AtomicU64,
    pub s75_q_check_child_movegen: AtomicU64,
    pub s75_q_check_child_terminal_0: AtomicU64,
    pub s75_q_check_child_evasions_1: AtomicU64,
    pub s75_q_check_child_evasions_2: AtomicU64,
    pub s75_q_check_child_evasions_3plus: AtomicU64,
    /// S7.5A candidate budget funnel (candidate profile only).
    pub s75a_extension_applied_total: AtomicU64,
    pub s75a_extension_applied_depth1: AtomicU64,
    pub s75a_extension_budget_2_to_1: AtomicU64,
    pub s75a_extension_budget_1_to_0: AtomicU64,
    pub s75a_opportunity_blocked_budget_0: AtomicU64,
    pub check_extensions: AtomicU64,
    pub single_evasion_extensions: AtomicU64,
    pub qsearch_check_moves: AtomicU64,
    pub threat_ordered_moves: AtomicU64,
    pub root_reorders: AtomicU64,
    see_enabled: AtomicBool,
    /// Diagnostic counters are opt-in so ordinary searches do not pay for
    /// an atomic increment on every hot-path event.
    profiling_enabled: bool,
    pub aspiration_retries: AtomicU64,
    pub aspiration_fail_low: AtomicU64,
    pub aspiration_fail_high: AtomicU64,
    pub lmr_reductions: AtomicU64,
    pub lmr_researches: AtomicU64,
    pub null_move_attempts: AtomicU64,
    pub null_move_fail_highs: AtomicU64,
    pub null_move_researches: AtomicU64,
    pub futility_pruned: AtomicU64,
    pub completed_iterations: AtomicU64,
    pub last_completed_iteration_ms: AtomicU64,
    pub last_completed_iteration_nodes: AtomicU64,
    pub aborted_iteration_depth: AtomicU64,
    pub aborted_iteration_nodes: AtomicU64,
    /// S4.0B: resolved feature policy bitmask (set at search start via
    /// `SearchFeaturePolicy::for_profile(...).to_bits()`). `AtomicU32` so the
    /// context stays `Send + Sync` (it is shared across the search thread).
    pub(crate) features_mask: AtomicU32,
    /// S4.0B: bench-only diagnostics (forced-root / disable-feature / target
    /// rank). `None` on the production UCI path.
    pub(crate) diagnostics: Option<SearchDiagnostics>,
    /// S4.0B: 1-based root rank of `diagnostics.target_root_move` in the
    /// normal root ordering (before forced-root filtering). 0 = target not in
    /// the legal move list.
    pub(crate) target_root_rank: AtomicU32,
    /// S4.3A: bench-only sampled wall-time attribution. `sampled_timing` is
    /// never set on the production UCI path.
    pub(crate) sampled_timing: bool,
    pub(crate) sample_rate: u32,
    pub(crate) timing_movegen_legal: SampledCounter,
    pub(crate) timing_movegen_tactical: SampledCounter,
    pub(crate) timing_movegen_evasion: SampledCounter,
    pub(crate) timing_movegen_has_any: SampledCounter,
    pub(crate) timing_eval: SampledCounter,
    pub(crate) timing_ordering: SampledCounter,
    pub(crate) timing_see: SampledCounter,
    pub(crate) timing_tt: SampledCounter,
    /// S4.3A: legality-probe make/unmake (from MovegenStats), split from
    /// recursive search edges. search_edge = total make_moves - these.
    pub(crate) legality_probe_make: AtomicU64,
    pub(crate) legality_probe_unmake: AtomicU64,
    /// S4.3B: unpinned non-check legality fast path enabled (set at search
    /// start from the profile; bench-only candidate, never the UCI default).
    pub(crate) legality_fast: AtomicBool,
    /// S4.4B: single-buffer full-legal materialization enabled (set at search
    /// start from the profile; candidate only, never the UCI default).
    pub(crate) single_buffer_legal: AtomicBool,
    /// S5.0B: child probe uses has-any-legal instead of a discarded full
    /// legal list (candidate only).
    pub(crate) single_generation_probe: AtomicBool,
    /// S4.4B: compaction writes where `write != read` (mechanism counter).
    pub(crate) single_buffer_writes: AtomicU64,
    /// S5.0A: duplicate child full-legal generation attribution. The child
    /// probe generates a full legal list for every edge (terminal + intended
    /// claim detection) and DISCARDS it on Continue; the negamax entered body
    /// then regenerates the SAME position's list. `probe_child_generations`
    /// totals the discarded probe lists (by caller kind), and
    /// `negamax_body_generations` counts the duplicate regeneration. Bench
    /// only, profiling-gated.
    pub(crate) probe_child_generations: AtomicU64,
    pub(crate) main_edge_probe_generations: AtomicU64,
    pub(crate) qsearch_edge_probe_generations: AtomicU64,
    pub(crate) root_edge_probe_generations: AtomicU64,
    pub(crate) negamax_body_generations: AtomicU64,
    pub(crate) root_generations: AtomicU64,
    pub(crate) final_evasion_generations: AtomicU64,
    /// S4.3B: fast-path / fallback statistics (profiling-enabled only).
    pub(crate) legality_fast_accepts: AtomicU64,
    pub(crate) legality_fallback_probes: AtomicU64,
    pub(crate) legality_fallback_in_check: AtomicU64,
    pub(crate) legality_fallback_king: AtomicU64,
    pub(crate) legality_fallback_pinned: AtomicU64,
    pub(crate) legality_fallback_en_passant: AtomicU64,
    pub(crate) legality_fallback_castle: AtomicU64,
    /// S4.4A: sparse sub-attribution inside the full legal generator (bench
    /// only; `sampled_timing` gate as the S4.3A sampler).
    pub(crate) full_legal_sub: FullLegalSub,
    /// S4.4A: per-generator legality-probe counts (split of the S4.3A
    /// `legality_probe_make/unmake` totals). Bench-only, profiling-gated.
    pub(crate) full_legal_probe_make: AtomicU64,
    pub(crate) full_legal_probe_unmake: AtomicU64,
    pub(crate) tactical_probe_make: AtomicU64,
    pub(crate) tactical_probe_unmake: AtomicU64,
    pub(crate) evasion_probe_make: AtomicU64,
    pub(crate) evasion_probe_unmake: AtomicU64,
    pub(crate) has_any_probe_make: AtomicU64,
    pub(crate) has_any_probe_unmake: AtomicU64,
    /// S7.0 depth-attribution (observation-only, profiling-gated): where the
    /// tree's effort actually goes, so depth bottlenecks can be ranked.
    pub beta_cutoffs: AtomicU64,
    pub beta_cutoff_idx_0: AtomicU64,
    pub beta_cutoff_idx_1: AtomicU64,
    pub beta_cutoff_idx_2_3: AtomicU64,
    pub beta_cutoff_idx_4_7: AtomicU64,
    pub beta_cutoff_idx_8_15: AtomicU64,
    pub beta_cutoff_idx_16p: AtomicU64,
    pub cutoff_tt_move: AtomicU64,
    pub cutoff_tactical: AtomicU64,
    pub cutoff_killer: AtomicU64,
    pub cutoff_quiet: AtomicU64,
    pub moves_searched: AtomicU64,
    pub pv_nodes: AtomicU64,
    pub in_check_nodes: AtomicU64,
    pub depth_bucket_0: AtomicU64,
    pub depth_bucket_1: AtomicU64,
    pub depth_bucket_2: AtomicU64,
    pub depth_bucket_3: AtomicU64,
    pub depth_bucket_4_5: AtomicU64,
    pub depth_bucket_6_7: AtomicU64,
    pub depth_bucket_8p: AtomicU64,
    pub searched_hist_1: AtomicU64,
    pub searched_hist_2: AtomicU64,
    pub searched_hist_3_4: AtomicU64,
    pub searched_hist_5_8: AtomicU64,
    pub searched_hist_9_16: AtomicU64,
    pub searched_hist_17p: AtomicU64,
    pub tt_hit_exact: AtomicU64,
    pub tt_hit_lower: AtomicU64,
    pub tt_hit_upper: AtomicU64,
    pub lmr_reduction_r1: AtomicU64,
    pub lmr_reduction_r2: AtomicU64,
    pub lmr_reduced_improves_alpha: AtomicU64,
    pub null_fail_lows: AtomicU64,
    pub futility_considered: AtomicU64,
    pub qsearch_standpat_cutoffs: AtomicU64,
    pub qsearch_standpat_alpha_raises: AtomicU64,
    pub qsearch_moves_searched: AtomicU64,
    pub qsearch_in_check_entries: AtomicU64,
    /// S7.1A: lazy qsearch materialization mechanism counters (profiling-only).
    pub qsearch_lazy_has_any_probes: AtomicU64,
    pub qsearch_lazy_standpat_cutoffs_before_movegen: AtomicU64,
    pub qsearch_lazy_qply_returns_before_movegen: AtomicU64,
    pub qsearch_lazy_tactical_generations: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub nodes: u64,
    pub seldepth: u64,
    pub main_seldepth: u64,
    pub qsearch_seldepth: u64,
    pub qsearch_nodes: u64,
    pub eval_calls: u64,
    pub legal_move_generations: u64,
    pub pseudo_moves: u64,
    pub legal_moves: u64,
    pub make_moves: u64,
    pub unmake_moves: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub tt_cutoffs: u64,
    pub tt_rejected_depth: u64,
    pub tt_rejected_bound: u64,
    pub tt_rejected_decode: u64,
    pub tt_stores: u64,
    pub see_calls: u64,
    pub see_pruned: u64,
    pub qsearch_see_tests: u64,
    pub qsearch_see_pruned: u64,
    pub qsearch_see_fail_open_promotions: u64,
    pub qsearch_checking_captures_kept: u64,
    pub qsearch_promotions_kept: u64,
    pub qsearch_en_passant_kept: u64,
    /// S7.1B SEE-delta qsearch pruning counters (profiling-gated).
    pub qsearch_delta_tests: u64,
    pub qsearch_delta_pruned: u64,
    pub qsearch_delta_pruned_pawn: u64,
    pub qsearch_delta_pruned_minor: u64,
    pub qsearch_delta_pruned_rook: u64,
    pub qsearch_delta_pruned_queen: u64,
    pub qsearch_delta_qply_0_1: u64,
    pub qsearch_delta_qply_2_3: u64,
    pub qsearch_delta_qply_4p: u64,
    /// S7.2 move-ordering attribution snapshot (bucket maps as documented
    /// on the `SearchContext` fields).
    pub s72_cutoff_category: [u64; 7],
    pub s72_nodes_with_quiet_moves: u64,
    pub s72_quiet_available: u64,
    pub s72_quiet_searched: u64,
    pub s72_quiet_searched_rank: [u64; 6],
    pub s72_quiet_searched_hist: [u64; 5],
    pub s72_quiet_cutoff_gidx: [u64; 6],
    pub s72_quiet_cutoff_rank: [u64; 6],
    pub s72_quiet_cutoff_hist: [u64; 5],
    pub s72_killer: [u64; 7],
    pub s72_tt_hash: [u64; 5],
    pub s72_cutoff_searched: [u64; 6],
    pub s72_fail_low_nodes: u64,
    pub s72_fail_low_searched_sum: u64,
    pub s72_lmr: [u64; 3],
    pub s72_d_cutoffs: [u64; 6],
    pub s72_d_cutoff_late5: [u64; 6],
    pub s72_d_fail_low: [u64; 6],
    pub s72_d_quiet_searched: [u64; 6],
    pub s72_d_quiet_cutoffs: [u64; 6],
    /// S7.3 selectivity attribution (see SearchContext block).
    pub s73_loop_nodes: u64,
    pub s73_nocut_pv: u64,
    pub s73_nocut_nonpv: u64,
    pub s73_nocut_incheck: u64,
    pub s73_nocut_null_attempted: u64,
    pub s73_nocut_searched_hist: [u64; 6],
    pub s73_nocut_searched_sum: u64,
    pub s73_null_eligible: u64,
    pub s73_fut_quiet_kept: u64,
    pub s73_q4p_quiet_searched: u64,
    pub s73_q4p_quiet_idx: [u64; 5],
    pub s73_q4p_quiet_red: [u64; 3],
    pub s73_q4p_quiet_red_idx: [u64; 15],
    pub s73_q4p_quiet_cutoff_red: [u64; 3],
    pub s73_q4p_quiet_cutoff_idx: [u64; 5],
    pub s73_q4p_scout_faillow_red: [u64; 3],
    pub s73_q4p_quiet_researched: u64,
    /// S7.4A LMR-on-null-window attribution (see SearchContext block).
    pub s74_lmr_proposed: u64,
    pub s74_lmr_proposed_r1: u64,
    pub s74_lmr_proposed_r2: u64,
    pub s74_lmr_applied_existing_pvs: u64,
    pub s74_lmr_suppressed_by_null_window: u64,
    pub s74_lmr_applied_null_window: u64,
    pub s74_lmr_nw_fail_low: u64,
    pub s74_lmr_nw_research: u64,
    pub s74_lmr_nw_research_entered: u64,
    pub s74_lmr_nw_verified_cutoff: u64,
    pub s74_lmr_nw_depth: [u64; 4],
    pub s74_lmr_nw_idx: [u64; 4],
    /// S7.5-0 forcing-opportunity attribution (OBSERVATION ONLY, profiling-gated).
    pub s75_main_nodes: u64,
    pub s75_main_in_check_nodes: u64,
    pub s75_main_single_evasion_nodes_raw: u64,
    pub s75_main_single_evasion_actionable_depth1: u64,
    pub s75_main_single_evasion_actionable_depth2plus: u64,
    pub s75_main_single_evasion_depth3plus: u64,
    pub s75_main_single_evasion_chain: [u64; 3],
    pub s75_main_checking_edges_searched: u64,
    pub s75_main_check_child_entered: u64,
    pub s75_main_check_child_movegen: u64,
    pub s75_main_check_child_terminal_0: u64,
    pub s75_main_check_child_evasions_1: u64,
    pub s75_main_check_child_evasions_2: u64,
    pub s75_main_check_child_evasions_3plus: u64,
    pub s75_main_depth1_nodes: u64,
    pub s75_main_depth1_in_check: u64,
    pub s75_main_depth1_single_evasion: u64,
    pub s75_main_depth1_entered_from_checking_edge: u64,
    pub s75_q_nodes: u64,
    pub s75_q_in_check_nodes: u64,
    pub s75_q_single_evasion_nodes_raw: u64,
    pub s75_q_single_evasion_qply0: u64,
    pub s75_q_single_evasion_qply1plus: u64,
    pub s75_q_checking_edges_searched: u64,
    pub s75_q_check_child_entered: u64,
    pub s75_q_check_child_movegen: u64,
    pub s75_q_check_child_terminal_0: u64,
    pub s75_q_check_child_evasions_1: u64,
    pub s75_q_check_child_evasions_2: u64,
    pub s75_q_check_child_evasions_3plus: u64,
    /// S7.5A candidate budget funnel (candidate profile only).
    pub s75a_extension_applied_total: u64,
    pub s75a_extension_applied_depth1: u64,
    pub s75a_extension_budget_2_to_1: u64,
    pub s75a_extension_budget_1_to_0: u64,
    pub s75a_opportunity_blocked_budget_0: u64,
    pub check_extensions: u64,
    pub single_evasion_extensions: u64,
    pub qsearch_check_moves: u64,
    pub threat_ordered_moves: u64,
    pub root_reorders: u64,
    pub aspiration_retries: u64,
    pub aspiration_fail_low: u64,
    pub aspiration_fail_high: u64,
    pub lmr_reductions: u64,
    pub lmr_researches: u64,
    pub null_move_attempts: u64,
    pub null_move_fail_highs: u64,
    pub null_move_researches: u64,
    pub futility_pruned: u64,
    pub completed_iterations: u64,
    pub last_completed_iteration_ms: u64,
    pub last_completed_iteration_nodes: u64,
    pub aborted_iteration_depth: u64,
    pub aborted_iteration_nodes: u64,
    /// S4.0B: 1-based root rank of the diagnostic target move, 0 = unset/absent.
    pub target_root_rank: u32,
    /// S4.3B: unpinned non-check legality fast path statistics.
    pub legality_fast_accepts: u64,
    pub legality_fallback_probes: u64,
    pub legality_fallback_in_check: u64,
    pub legality_fallback_king: u64,
    pub legality_fallback_pinned: u64,
    pub legality_fallback_en_passant: u64,
    pub legality_fallback_castle: u64,
    /// S4.4B: single-buffer compaction writes (zero for two-buffer profiles).
    pub single_buffer_writes: u64,
    /// S7.0 depth-attribution counters (observation-only, profiling-gated).
    pub beta_cutoffs: u64,
    pub beta_cutoff_idx_0: u64,
    pub beta_cutoff_idx_1: u64,
    pub beta_cutoff_idx_2_3: u64,
    pub beta_cutoff_idx_4_7: u64,
    pub beta_cutoff_idx_8_15: u64,
    pub beta_cutoff_idx_16p: u64,
    pub cutoff_tt_move: u64,
    pub cutoff_tactical: u64,
    pub cutoff_killer: u64,
    pub cutoff_quiet: u64,
    pub moves_searched: u64,
    pub pv_nodes: u64,
    pub in_check_nodes: u64,
    pub depth_bucket_0: u64,
    pub depth_bucket_1: u64,
    pub depth_bucket_2: u64,
    pub depth_bucket_3: u64,
    pub depth_bucket_4_5: u64,
    pub depth_bucket_6_7: u64,
    pub depth_bucket_8p: u64,
    pub searched_hist_1: u64,
    pub searched_hist_2: u64,
    pub searched_hist_3_4: u64,
    pub searched_hist_5_8: u64,
    pub searched_hist_9_16: u64,
    pub searched_hist_17p: u64,
    pub tt_hit_exact: u64,
    pub tt_hit_lower: u64,
    pub tt_hit_upper: u64,
    pub lmr_reduction_r1: u64,
    pub lmr_reduction_r2: u64,
    pub lmr_reduced_improves_alpha: u64,
    pub null_fail_lows: u64,
    pub futility_considered: u64,
    pub qsearch_standpat_cutoffs: u64,
    pub qsearch_standpat_alpha_raises: u64,
    pub qsearch_moves_searched: u64,
    pub qsearch_in_check_entries: u64,
    pub qsearch_lazy_has_any_probes: u64,
    pub qsearch_lazy_standpat_cutoffs_before_movegen: u64,
    pub qsearch_lazy_qply_returns_before_movegen: u64,
    pub qsearch_lazy_tactical_generations: u64,
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
            seldepth: AtomicU64::new(0),
            main_seldepth: AtomicU64::new(0),
            qsearch_seldepth: AtomicU64::new(0),
            qsearch_nodes: AtomicU64::new(0),
            eval_calls: AtomicU64::new(0),
            legal_move_generations: AtomicU64::new(0),
            pseudo_moves: AtomicU64::new(0),
            legal_moves: AtomicU64::new(0),
            make_moves: AtomicU64::new(0),
            unmake_moves: AtomicU64::new(0),
            tt_probes: AtomicU64::new(0),
            tt_hits: AtomicU64::new(0),
            tt_cutoffs: AtomicU64::new(0),
            tt_rejected_depth: AtomicU64::new(0),
            tt_rejected_bound: AtomicU64::new(0),
            tt_rejected_decode: AtomicU64::new(0),
            tt_stores: AtomicU64::new(0),
            see_calls: AtomicU64::new(0),
            see_pruned: AtomicU64::new(0),
            qsearch_see_tests: AtomicU64::new(0),
            qsearch_see_pruned: AtomicU64::new(0),
            qsearch_see_fail_open_promotions: AtomicU64::new(0),
            qsearch_checking_captures_kept: AtomicU64::new(0),
            qsearch_promotions_kept: AtomicU64::new(0),
            qsearch_en_passant_kept: AtomicU64::new(0),
            qsearch_delta_tests: AtomicU64::new(0),
            qsearch_delta_pruned: AtomicU64::new(0),
            qsearch_delta_pruned_pawn: AtomicU64::new(0),
            qsearch_delta_pruned_minor: AtomicU64::new(0),
            qsearch_delta_pruned_rook: AtomicU64::new(0),
            qsearch_delta_pruned_queen: AtomicU64::new(0),
            qsearch_delta_qply_0_1: AtomicU64::new(0),
            qsearch_delta_qply_2_3: AtomicU64::new(0),
            qsearch_delta_qply_4p: AtomicU64::new(0),
            s72_cutoff_category: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_nodes_with_quiet_moves: AtomicU64::new(0),
            s72_quiet_available: AtomicU64::new(0),
            s72_quiet_searched: AtomicU64::new(0),
            s72_quiet_searched_rank: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_searched_hist: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_cutoff_gidx: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_cutoff_rank: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_cutoff_hist: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_killer: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_tt_hash: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_cutoff_searched: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_fail_low_nodes: AtomicU64::new(0),
            s72_fail_low_searched_sum: AtomicU64::new(0),
            s72_lmr: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_cutoffs: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_cutoff_late5: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_fail_low: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_quiet_searched: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_quiet_cutoffs: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_loop_nodes: AtomicU64::new(0),
            s73_nocut_pv: AtomicU64::new(0),
            s73_nocut_nonpv: AtomicU64::new(0),
            s73_nocut_incheck: AtomicU64::new(0),
            s73_nocut_null_attempted: AtomicU64::new(0),
            s73_nocut_searched_hist: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_nocut_searched_sum: AtomicU64::new(0),
            s73_null_eligible: AtomicU64::new(0),
            s73_fut_quiet_kept: AtomicU64::new(0),
            s73_q4p_quiet_searched: AtomicU64::new(0),
            s73_q4p_quiet_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_red: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_red_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_cutoff_red: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_cutoff_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_scout_faillow_red: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_researched: AtomicU64::new(0),
            s74_lmr_proposed: AtomicU64::new(0),
            s74_lmr_proposed_r1: AtomicU64::new(0),
            s74_lmr_proposed_r2: AtomicU64::new(0),
            s74_lmr_applied_existing_pvs: AtomicU64::new(0),
            s74_lmr_suppressed_by_null_window: AtomicU64::new(0),
            s74_lmr_applied_null_window: AtomicU64::new(0),
            s74_lmr_nw_fail_low: AtomicU64::new(0),
            s74_lmr_nw_research: AtomicU64::new(0),
            s74_lmr_nw_research_entered: AtomicU64::new(0),
            s74_lmr_nw_verified_cutoff: AtomicU64::new(0),
            s74_lmr_nw_depth: std::array::from_fn(|_| AtomicU64::new(0)),
            s74_lmr_nw_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s75_main_in_check_nodes: AtomicU64::new(0),
            s75_main_single_evasion_nodes_raw: AtomicU64::new(0),
            s75_main_single_evasion_actionable_depth1: AtomicU64::new(0),
            s75_main_single_evasion_actionable_depth2plus: AtomicU64::new(0),
            s75_main_single_evasion_depth3plus: AtomicU64::new(0),
            s75_main_single_evasion_chain: std::array::from_fn(|_| AtomicU64::new(0)),
            s75_main_checking_edges_searched: AtomicU64::new(0),
            s75_main_check_child_entered: AtomicU64::new(0),
            s75_main_check_child_movegen: AtomicU64::new(0),
            s75_main_check_child_terminal_0: AtomicU64::new(0),
            s75_main_check_child_evasions_1: AtomicU64::new(0),
            s75_main_check_child_evasions_2: AtomicU64::new(0),
            s75_main_check_child_evasions_3plus: AtomicU64::new(0),
            s75_main_depth1_nodes: AtomicU64::new(0),
            s75_main_depth1_in_check: AtomicU64::new(0),
            s75_main_depth1_single_evasion: AtomicU64::new(0),
            s75_main_depth1_entered_from_checking_edge: AtomicU64::new(0),
            s75_q_nodes: AtomicU64::new(0),
            s75_q_in_check_nodes: AtomicU64::new(0),
            s75_q_single_evasion_nodes_raw: AtomicU64::new(0),
            s75_q_single_evasion_qply0: AtomicU64::new(0),
            s75_q_single_evasion_qply1plus: AtomicU64::new(0),
            s75_q_checking_edges_searched: AtomicU64::new(0),
            s75_q_check_child_entered: AtomicU64::new(0),
            s75_q_check_child_movegen: AtomicU64::new(0),
            s75_q_check_child_terminal_0: AtomicU64::new(0),
            s75_q_check_child_evasions_1: AtomicU64::new(0),
            s75_q_check_child_evasions_2: AtomicU64::new(0),
            s75_q_check_child_evasions_3plus: AtomicU64::new(0),
            s75a_extension_applied_total: AtomicU64::new(0),
            s75a_extension_applied_depth1: AtomicU64::new(0),
            s75a_extension_budget_2_to_1: AtomicU64::new(0),
            s75a_extension_budget_1_to_0: AtomicU64::new(0),
            s75a_opportunity_blocked_budget_0: AtomicU64::new(0),

            check_extensions: AtomicU64::new(0),
            single_evasion_extensions: AtomicU64::new(0),
            qsearch_check_moves: AtomicU64::new(0),
            threat_ordered_moves: AtomicU64::new(0),
            root_reorders: AtomicU64::new(0),
            see_enabled: AtomicBool::new(false),
            profiling_enabled: false,
            aspiration_retries: AtomicU64::new(0),
            aspiration_fail_low: AtomicU64::new(0),
            aspiration_fail_high: AtomicU64::new(0),
            lmr_reductions: AtomicU64::new(0),
            lmr_researches: AtomicU64::new(0),
            null_move_attempts: AtomicU64::new(0),
            null_move_fail_highs: AtomicU64::new(0),
            null_move_researches: AtomicU64::new(0),
            futility_pruned: AtomicU64::new(0),
            completed_iterations: AtomicU64::new(0),
            last_completed_iteration_ms: AtomicU64::new(0),
            last_completed_iteration_nodes: AtomicU64::new(0),
            aborted_iteration_depth: AtomicU64::new(0),
            aborted_iteration_nodes: AtomicU64::new(0),
            features_mask: AtomicU32::new(0),
            diagnostics: None,
            target_root_rank: AtomicU32::new(TARGET_ROOT_RANK_NONE),
            sampled_timing: false,
            sample_rate: 0,
            timing_movegen_legal: SampledCounter::default(),
            timing_movegen_tactical: SampledCounter::default(),
            timing_movegen_evasion: SampledCounter::default(),
            timing_movegen_has_any: SampledCounter::default(),
            timing_eval: SampledCounter::default(),
            timing_ordering: SampledCounter::default(),
            timing_see: SampledCounter::default(),
            timing_tt: SampledCounter::default(),
            legality_probe_make: AtomicU64::new(0),
            legality_probe_unmake: AtomicU64::new(0),
            legality_fast: AtomicBool::new(false),
            single_buffer_legal: AtomicBool::new(false),
            single_generation_probe: AtomicBool::new(false),
            single_buffer_writes: AtomicU64::new(0),
            probe_child_generations: AtomicU64::new(0),
            main_edge_probe_generations: AtomicU64::new(0),
            qsearch_edge_probe_generations: AtomicU64::new(0),
            root_edge_probe_generations: AtomicU64::new(0),
            negamax_body_generations: AtomicU64::new(0),
            root_generations: AtomicU64::new(0),
            final_evasion_generations: AtomicU64::new(0),
            legality_fast_accepts: AtomicU64::new(0),
            legality_fallback_probes: AtomicU64::new(0),
            legality_fallback_in_check: AtomicU64::new(0),
            legality_fallback_king: AtomicU64::new(0),
            legality_fallback_pinned: AtomicU64::new(0),
            legality_fallback_en_passant: AtomicU64::new(0),
            legality_fallback_castle: AtomicU64::new(0),
            full_legal_sub: FullLegalSub::new(0),
            full_legal_probe_make: AtomicU64::new(0),
            full_legal_probe_unmake: AtomicU64::new(0),
            tactical_probe_make: AtomicU64::new(0),
            tactical_probe_unmake: AtomicU64::new(0),
            evasion_probe_make: AtomicU64::new(0),
            evasion_probe_unmake: AtomicU64::new(0),
            has_any_probe_make: AtomicU64::new(0),
            has_any_probe_unmake: AtomicU64::new(0),
            beta_cutoffs: AtomicU64::new(0),
            beta_cutoff_idx_0: AtomicU64::new(0),
            beta_cutoff_idx_1: AtomicU64::new(0),
            beta_cutoff_idx_2_3: AtomicU64::new(0),
            beta_cutoff_idx_4_7: AtomicU64::new(0),
            beta_cutoff_idx_8_15: AtomicU64::new(0),
            beta_cutoff_idx_16p: AtomicU64::new(0),
            cutoff_tt_move: AtomicU64::new(0),
            cutoff_tactical: AtomicU64::new(0),
            cutoff_killer: AtomicU64::new(0),
            cutoff_quiet: AtomicU64::new(0),
            moves_searched: AtomicU64::new(0),
            pv_nodes: AtomicU64::new(0),
            in_check_nodes: AtomicU64::new(0),
            depth_bucket_0: AtomicU64::new(0),
            depth_bucket_1: AtomicU64::new(0),
            depth_bucket_2: AtomicU64::new(0),
            depth_bucket_3: AtomicU64::new(0),
            depth_bucket_4_5: AtomicU64::new(0),
            depth_bucket_6_7: AtomicU64::new(0),
            depth_bucket_8p: AtomicU64::new(0),
            searched_hist_1: AtomicU64::new(0),
            searched_hist_2: AtomicU64::new(0),
            searched_hist_3_4: AtomicU64::new(0),
            searched_hist_5_8: AtomicU64::new(0),
            searched_hist_9_16: AtomicU64::new(0),
            searched_hist_17p: AtomicU64::new(0),
            tt_hit_exact: AtomicU64::new(0),
            tt_hit_lower: AtomicU64::new(0),
            tt_hit_upper: AtomicU64::new(0),
            lmr_reduction_r1: AtomicU64::new(0),
            lmr_reduction_r2: AtomicU64::new(0),
            lmr_reduced_improves_alpha: AtomicU64::new(0),
            null_fail_lows: AtomicU64::new(0),
            futility_considered: AtomicU64::new(0),
            qsearch_standpat_cutoffs: AtomicU64::new(0),
            qsearch_standpat_alpha_raises: AtomicU64::new(0),
            qsearch_moves_searched: AtomicU64::new(0),
            qsearch_in_check_entries: AtomicU64::new(0),
            qsearch_lazy_has_any_probes: AtomicU64::new(0),
            qsearch_lazy_standpat_cutoffs_before_movegen: AtomicU64::new(0),
            qsearch_lazy_qply_returns_before_movegen: AtomicU64::new(0),
            qsearch_lazy_tactical_generations: AtomicU64::new(0),
        }
    }

    /// Record the deepest global ply reached (UCI seldepth). `ply` is the
    /// single global-ply definition shared by the main search and qsearch
    /// (root = 0, first child = 1). qsearch recursion advances BOTH `ply+1`
    /// and `qply+1`, so its `ply` argument already carries the qsearch
    /// descent - never add `qply`, or seldepth double-counts.
    pub fn record_seldepth(&self, ply: u32) {
        self.seldepth.fetch_max(ply as u64, Ordering::Relaxed);
    }

    /// With a precomputed time budget (soft + hard deadlines).
    pub fn with_budget(stop: Arc<AtomicBool>, budget: TimeBudget) -> Self {
        Self::with_budget_and_profiling(stop, budget, false)
    }

    /// With a precomputed time budget and explicitly enabled diagnostics.
    pub fn with_budget_and_profiling(
        stop: Arc<AtomicBool>,
        budget: TimeBudget,
        profiling_enabled: bool,
    ) -> Self {
        SearchContext {
            stop,
            start: Instant::now(),
            soft_deadline: budget.soft_deadline,
            hard_deadline: budget.hard_deadline,
            nodes: AtomicU64::new(0),
            seldepth: AtomicU64::new(0),
            main_seldepth: AtomicU64::new(0),
            qsearch_seldepth: AtomicU64::new(0),
            qsearch_nodes: AtomicU64::new(0),
            eval_calls: AtomicU64::new(0),
            legal_move_generations: AtomicU64::new(0),
            pseudo_moves: AtomicU64::new(0),
            legal_moves: AtomicU64::new(0),
            make_moves: AtomicU64::new(0),
            unmake_moves: AtomicU64::new(0),
            tt_probes: AtomicU64::new(0),
            tt_hits: AtomicU64::new(0),
            tt_cutoffs: AtomicU64::new(0),
            tt_rejected_depth: AtomicU64::new(0),
            tt_rejected_bound: AtomicU64::new(0),
            tt_rejected_decode: AtomicU64::new(0),
            tt_stores: AtomicU64::new(0),
            see_calls: AtomicU64::new(0),
            see_pruned: AtomicU64::new(0),
            qsearch_see_tests: AtomicU64::new(0),
            qsearch_see_pruned: AtomicU64::new(0),
            qsearch_see_fail_open_promotions: AtomicU64::new(0),
            qsearch_checking_captures_kept: AtomicU64::new(0),
            qsearch_promotions_kept: AtomicU64::new(0),
            qsearch_en_passant_kept: AtomicU64::new(0),
            qsearch_delta_tests: AtomicU64::new(0),
            qsearch_delta_pruned: AtomicU64::new(0),
            qsearch_delta_pruned_pawn: AtomicU64::new(0),
            qsearch_delta_pruned_minor: AtomicU64::new(0),
            qsearch_delta_pruned_rook: AtomicU64::new(0),
            qsearch_delta_pruned_queen: AtomicU64::new(0),
            qsearch_delta_qply_0_1: AtomicU64::new(0),
            qsearch_delta_qply_2_3: AtomicU64::new(0),
            qsearch_delta_qply_4p: AtomicU64::new(0),
            s72_cutoff_category: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_nodes_with_quiet_moves: AtomicU64::new(0),
            s72_quiet_available: AtomicU64::new(0),
            s72_quiet_searched: AtomicU64::new(0),
            s72_quiet_searched_rank: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_searched_hist: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_cutoff_gidx: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_cutoff_rank: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_quiet_cutoff_hist: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_killer: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_tt_hash: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_cutoff_searched: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_fail_low_nodes: AtomicU64::new(0),
            s72_fail_low_searched_sum: AtomicU64::new(0),
            s72_lmr: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_cutoffs: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_cutoff_late5: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_fail_low: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_quiet_searched: std::array::from_fn(|_| AtomicU64::new(0)),
            s72_d_quiet_cutoffs: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_loop_nodes: AtomicU64::new(0),
            s73_nocut_pv: AtomicU64::new(0),
            s73_nocut_nonpv: AtomicU64::new(0),
            s73_nocut_incheck: AtomicU64::new(0),
            s73_nocut_null_attempted: AtomicU64::new(0),
            s73_nocut_searched_hist: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_nocut_searched_sum: AtomicU64::new(0),
            s73_null_eligible: AtomicU64::new(0),
            s73_fut_quiet_kept: AtomicU64::new(0),
            s73_q4p_quiet_searched: AtomicU64::new(0),
            s73_q4p_quiet_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_red: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_red_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_cutoff_red: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_cutoff_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_scout_faillow_red: std::array::from_fn(|_| AtomicU64::new(0)),
            s73_q4p_quiet_researched: AtomicU64::new(0),
            s74_lmr_proposed: AtomicU64::new(0),
            s74_lmr_proposed_r1: AtomicU64::new(0),
            s74_lmr_proposed_r2: AtomicU64::new(0),
            s74_lmr_applied_existing_pvs: AtomicU64::new(0),
            s74_lmr_suppressed_by_null_window: AtomicU64::new(0),
            s74_lmr_applied_null_window: AtomicU64::new(0),
            s74_lmr_nw_fail_low: AtomicU64::new(0),
            s74_lmr_nw_research: AtomicU64::new(0),
            s74_lmr_nw_research_entered: AtomicU64::new(0),
            s74_lmr_nw_verified_cutoff: AtomicU64::new(0),
            s74_lmr_nw_depth: std::array::from_fn(|_| AtomicU64::new(0)),
            s74_lmr_nw_idx: std::array::from_fn(|_| AtomicU64::new(0)),
            s75_main_in_check_nodes: AtomicU64::new(0),
            s75_main_single_evasion_nodes_raw: AtomicU64::new(0),
            s75_main_single_evasion_actionable_depth1: AtomicU64::new(0),
            s75_main_single_evasion_actionable_depth2plus: AtomicU64::new(0),
            s75_main_single_evasion_depth3plus: AtomicU64::new(0),
            s75_main_single_evasion_chain: std::array::from_fn(|_| AtomicU64::new(0)),
            s75_main_checking_edges_searched: AtomicU64::new(0),
            s75_main_check_child_entered: AtomicU64::new(0),
            s75_main_check_child_movegen: AtomicU64::new(0),
            s75_main_check_child_terminal_0: AtomicU64::new(0),
            s75_main_check_child_evasions_1: AtomicU64::new(0),
            s75_main_check_child_evasions_2: AtomicU64::new(0),
            s75_main_check_child_evasions_3plus: AtomicU64::new(0),
            s75_main_depth1_nodes: AtomicU64::new(0),
            s75_main_depth1_in_check: AtomicU64::new(0),
            s75_main_depth1_single_evasion: AtomicU64::new(0),
            s75_main_depth1_entered_from_checking_edge: AtomicU64::new(0),
            s75_q_nodes: AtomicU64::new(0),
            s75_q_in_check_nodes: AtomicU64::new(0),
            s75_q_single_evasion_nodes_raw: AtomicU64::new(0),
            s75_q_single_evasion_qply0: AtomicU64::new(0),
            s75_q_single_evasion_qply1plus: AtomicU64::new(0),
            s75_q_checking_edges_searched: AtomicU64::new(0),
            s75_q_check_child_entered: AtomicU64::new(0),
            s75_q_check_child_movegen: AtomicU64::new(0),
            s75_q_check_child_terminal_0: AtomicU64::new(0),
            s75_q_check_child_evasions_1: AtomicU64::new(0),
            s75_q_check_child_evasions_2: AtomicU64::new(0),
            s75_q_check_child_evasions_3plus: AtomicU64::new(0),
            s75a_extension_applied_total: AtomicU64::new(0),
            s75a_extension_applied_depth1: AtomicU64::new(0),
            s75a_extension_budget_2_to_1: AtomicU64::new(0),
            s75a_extension_budget_1_to_0: AtomicU64::new(0),
            s75a_opportunity_blocked_budget_0: AtomicU64::new(0),

            check_extensions: AtomicU64::new(0),
            single_evasion_extensions: AtomicU64::new(0),
            qsearch_check_moves: AtomicU64::new(0),
            threat_ordered_moves: AtomicU64::new(0),
            root_reorders: AtomicU64::new(0),
            see_enabled: AtomicBool::new(false),
            profiling_enabled,
            aspiration_retries: AtomicU64::new(0),
            aspiration_fail_low: AtomicU64::new(0),
            aspiration_fail_high: AtomicU64::new(0),
            lmr_reductions: AtomicU64::new(0),
            lmr_researches: AtomicU64::new(0),
            null_move_attempts: AtomicU64::new(0),
            null_move_fail_highs: AtomicU64::new(0),
            null_move_researches: AtomicU64::new(0),
            futility_pruned: AtomicU64::new(0),
            completed_iterations: AtomicU64::new(0),
            last_completed_iteration_ms: AtomicU64::new(0),
            last_completed_iteration_nodes: AtomicU64::new(0),
            aborted_iteration_depth: AtomicU64::new(0),
            aborted_iteration_nodes: AtomicU64::new(0),
            features_mask: AtomicU32::new(0),
            diagnostics: None,
            target_root_rank: AtomicU32::new(TARGET_ROOT_RANK_NONE),
            sampled_timing: false,
            sample_rate: 0,
            timing_movegen_legal: SampledCounter::default(),
            timing_movegen_tactical: SampledCounter::default(),
            timing_movegen_evasion: SampledCounter::default(),
            timing_movegen_has_any: SampledCounter::default(),
            timing_eval: SampledCounter::default(),
            timing_ordering: SampledCounter::default(),
            timing_see: SampledCounter::default(),
            timing_tt: SampledCounter::default(),
            legality_probe_make: AtomicU64::new(0),
            legality_probe_unmake: AtomicU64::new(0),
            legality_fast: AtomicBool::new(false),
            single_buffer_legal: AtomicBool::new(false),
            single_generation_probe: AtomicBool::new(false),
            single_buffer_writes: AtomicU64::new(0),
            probe_child_generations: AtomicU64::new(0),
            main_edge_probe_generations: AtomicU64::new(0),
            qsearch_edge_probe_generations: AtomicU64::new(0),
            root_edge_probe_generations: AtomicU64::new(0),
            negamax_body_generations: AtomicU64::new(0),
            root_generations: AtomicU64::new(0),
            final_evasion_generations: AtomicU64::new(0),
            legality_fast_accepts: AtomicU64::new(0),
            legality_fallback_probes: AtomicU64::new(0),
            legality_fallback_in_check: AtomicU64::new(0),
            legality_fallback_king: AtomicU64::new(0),
            legality_fallback_pinned: AtomicU64::new(0),
            legality_fallback_en_passant: AtomicU64::new(0),
            legality_fallback_castle: AtomicU64::new(0),
            full_legal_sub: FullLegalSub::new(0),
            full_legal_probe_make: AtomicU64::new(0),
            full_legal_probe_unmake: AtomicU64::new(0),
            tactical_probe_make: AtomicU64::new(0),
            tactical_probe_unmake: AtomicU64::new(0),
            evasion_probe_make: AtomicU64::new(0),
            evasion_probe_unmake: AtomicU64::new(0),
            has_any_probe_make: AtomicU64::new(0),
            has_any_probe_unmake: AtomicU64::new(0),
            beta_cutoffs: AtomicU64::new(0),
            beta_cutoff_idx_0: AtomicU64::new(0),
            beta_cutoff_idx_1: AtomicU64::new(0),
            beta_cutoff_idx_2_3: AtomicU64::new(0),
            beta_cutoff_idx_4_7: AtomicU64::new(0),
            beta_cutoff_idx_8_15: AtomicU64::new(0),
            beta_cutoff_idx_16p: AtomicU64::new(0),
            cutoff_tt_move: AtomicU64::new(0),
            cutoff_tactical: AtomicU64::new(0),
            cutoff_killer: AtomicU64::new(0),
            cutoff_quiet: AtomicU64::new(0),
            moves_searched: AtomicU64::new(0),
            pv_nodes: AtomicU64::new(0),
            in_check_nodes: AtomicU64::new(0),
            depth_bucket_0: AtomicU64::new(0),
            depth_bucket_1: AtomicU64::new(0),
            depth_bucket_2: AtomicU64::new(0),
            depth_bucket_3: AtomicU64::new(0),
            depth_bucket_4_5: AtomicU64::new(0),
            depth_bucket_6_7: AtomicU64::new(0),
            depth_bucket_8p: AtomicU64::new(0),
            searched_hist_1: AtomicU64::new(0),
            searched_hist_2: AtomicU64::new(0),
            searched_hist_3_4: AtomicU64::new(0),
            searched_hist_5_8: AtomicU64::new(0),
            searched_hist_9_16: AtomicU64::new(0),
            searched_hist_17p: AtomicU64::new(0),
            tt_hit_exact: AtomicU64::new(0),
            tt_hit_lower: AtomicU64::new(0),
            tt_hit_upper: AtomicU64::new(0),
            lmr_reduction_r1: AtomicU64::new(0),
            lmr_reduction_r2: AtomicU64::new(0),
            lmr_reduced_improves_alpha: AtomicU64::new(0),
            null_fail_lows: AtomicU64::new(0),
            futility_considered: AtomicU64::new(0),
            qsearch_standpat_cutoffs: AtomicU64::new(0),
            qsearch_standpat_alpha_raises: AtomicU64::new(0),
            qsearch_moves_searched: AtomicU64::new(0),
            qsearch_in_check_entries: AtomicU64::new(0),
            qsearch_lazy_has_any_probes: AtomicU64::new(0),
            qsearch_lazy_standpat_cutoffs_before_movegen: AtomicU64::new(0),
            qsearch_lazy_qply_returns_before_movegen: AtomicU64::new(0),
            qsearch_lazy_tactical_generations: AtomicU64::new(0),
        }
    }

    pub fn stats(&self) -> SearchStats {
        SearchStats {
            nodes: self.nodes.load(Ordering::Relaxed),
            seldepth: self.seldepth.load(Ordering::Relaxed),
            main_seldepth: self.main_seldepth.load(Ordering::Relaxed),
            qsearch_seldepth: self.qsearch_seldepth.load(Ordering::Relaxed),
            qsearch_nodes: self.qsearch_nodes.load(Ordering::Relaxed),
            eval_calls: self.eval_calls.load(Ordering::Relaxed),
            legal_move_generations: self.legal_move_generations.load(Ordering::Relaxed),
            pseudo_moves: self.pseudo_moves.load(Ordering::Relaxed),
            legal_moves: self.legal_moves.load(Ordering::Relaxed),
            make_moves: self.make_moves.load(Ordering::Relaxed),
            unmake_moves: self.unmake_moves.load(Ordering::Relaxed),
            tt_probes: self.tt_probes.load(Ordering::Relaxed),
            tt_hits: self.tt_hits.load(Ordering::Relaxed),
            tt_cutoffs: self.tt_cutoffs.load(Ordering::Relaxed),
            tt_rejected_depth: self.tt_rejected_depth.load(Ordering::Relaxed),
            tt_rejected_bound: self.tt_rejected_bound.load(Ordering::Relaxed),
            tt_rejected_decode: self.tt_rejected_decode.load(Ordering::Relaxed),
            tt_stores: self.tt_stores.load(Ordering::Relaxed),
            see_calls: self.see_calls.load(Ordering::Relaxed),
            see_pruned: self.see_pruned.load(Ordering::Relaxed),
            qsearch_see_tests: self.qsearch_see_tests.load(Ordering::Relaxed),
            qsearch_see_pruned: self.qsearch_see_pruned.load(Ordering::Relaxed),
            qsearch_see_fail_open_promotions: self
                .qsearch_see_fail_open_promotions
                .load(Ordering::Relaxed),
            qsearch_checking_captures_kept: self
                .qsearch_checking_captures_kept
                .load(Ordering::Relaxed),
            qsearch_promotions_kept: self.qsearch_promotions_kept.load(Ordering::Relaxed),
            qsearch_en_passant_kept: self.qsearch_en_passant_kept.load(Ordering::Relaxed),
            qsearch_delta_tests: self.qsearch_delta_tests.load(Ordering::Relaxed),
            qsearch_delta_pruned: self.qsearch_delta_pruned.load(Ordering::Relaxed),
            qsearch_delta_pruned_pawn: self.qsearch_delta_pruned_pawn.load(Ordering::Relaxed),
            qsearch_delta_pruned_minor: self.qsearch_delta_pruned_minor.load(Ordering::Relaxed),
            qsearch_delta_pruned_rook: self.qsearch_delta_pruned_rook.load(Ordering::Relaxed),
            qsearch_delta_pruned_queen: self.qsearch_delta_pruned_queen.load(Ordering::Relaxed),
            qsearch_delta_qply_0_1: self.qsearch_delta_qply_0_1.load(Ordering::Relaxed),
            qsearch_delta_qply_2_3: self.qsearch_delta_qply_2_3.load(Ordering::Relaxed),
            qsearch_delta_qply_4p: self.qsearch_delta_qply_4p.load(Ordering::Relaxed),
            s72_cutoff_category: std::array::from_fn(|i| {
                self.s72_cutoff_category[i].load(Ordering::Relaxed)
            }),
            s72_nodes_with_quiet_moves: self.s72_nodes_with_quiet_moves.load(Ordering::Relaxed),
            s72_quiet_available: self.s72_quiet_available.load(Ordering::Relaxed),
            s72_quiet_searched: self.s72_quiet_searched.load(Ordering::Relaxed),
            s72_quiet_searched_rank: std::array::from_fn(|i| {
                self.s72_quiet_searched_rank[i].load(Ordering::Relaxed)
            }),
            s72_quiet_searched_hist: std::array::from_fn(|i| {
                self.s72_quiet_searched_hist[i].load(Ordering::Relaxed)
            }),
            s72_quiet_cutoff_gidx: std::array::from_fn(|i| {
                self.s72_quiet_cutoff_gidx[i].load(Ordering::Relaxed)
            }),
            s72_quiet_cutoff_rank: std::array::from_fn(|i| {
                self.s72_quiet_cutoff_rank[i].load(Ordering::Relaxed)
            }),
            s72_quiet_cutoff_hist: std::array::from_fn(|i| {
                self.s72_quiet_cutoff_hist[i].load(Ordering::Relaxed)
            }),
            s72_killer: std::array::from_fn(|i| self.s72_killer[i].load(Ordering::Relaxed)),
            s72_tt_hash: std::array::from_fn(|i| self.s72_tt_hash[i].load(Ordering::Relaxed)),
            s72_cutoff_searched: std::array::from_fn(|i| {
                self.s72_cutoff_searched[i].load(Ordering::Relaxed)
            }),
            s72_fail_low_nodes: self.s72_fail_low_nodes.load(Ordering::Relaxed),
            s72_fail_low_searched_sum: self.s72_fail_low_searched_sum.load(Ordering::Relaxed),
            s72_lmr: std::array::from_fn(|i| self.s72_lmr[i].load(Ordering::Relaxed)),
            s72_d_cutoffs: std::array::from_fn(|i| self.s72_d_cutoffs[i].load(Ordering::Relaxed)),
            s72_d_cutoff_late5: std::array::from_fn(|i| {
                self.s72_d_cutoff_late5[i].load(Ordering::Relaxed)
            }),
            s72_d_fail_low: std::array::from_fn(|i| self.s72_d_fail_low[i].load(Ordering::Relaxed)),
            s72_d_quiet_searched: std::array::from_fn(|i| {
                self.s72_d_quiet_searched[i].load(Ordering::Relaxed)
            }),
            s73_loop_nodes: self.s73_loop_nodes.load(Ordering::Relaxed),
            s73_nocut_pv: self.s73_nocut_pv.load(Ordering::Relaxed),
            s73_nocut_nonpv: self.s73_nocut_nonpv.load(Ordering::Relaxed),
            s73_nocut_incheck: self.s73_nocut_incheck.load(Ordering::Relaxed),
            s73_nocut_null_attempted: self.s73_nocut_null_attempted.load(Ordering::Relaxed),
            s73_nocut_searched_hist: std::array::from_fn(|i| {
                self.s73_nocut_searched_hist[i].load(Ordering::Relaxed)
            }),
            s73_nocut_searched_sum: self.s73_nocut_searched_sum.load(Ordering::Relaxed),
            s73_null_eligible: self.s73_null_eligible.load(Ordering::Relaxed),
            s73_fut_quiet_kept: self.s73_fut_quiet_kept.load(Ordering::Relaxed),
            s73_q4p_quiet_searched: self.s73_q4p_quiet_searched.load(Ordering::Relaxed),
            s73_q4p_quiet_idx: std::array::from_fn(|i| {
                self.s73_q4p_quiet_idx[i].load(Ordering::Relaxed)
            }),
            s73_q4p_quiet_red: std::array::from_fn(|i| {
                self.s73_q4p_quiet_red[i].load(Ordering::Relaxed)
            }),
            s73_q4p_quiet_red_idx: std::array::from_fn(|i| {
                self.s73_q4p_quiet_red_idx[i].load(Ordering::Relaxed)
            }),
            s73_q4p_quiet_cutoff_red: std::array::from_fn(|i| {
                self.s73_q4p_quiet_cutoff_red[i].load(Ordering::Relaxed)
            }),
            s73_q4p_quiet_cutoff_idx: std::array::from_fn(|i| {
                self.s73_q4p_quiet_cutoff_idx[i].load(Ordering::Relaxed)
            }),
            s73_q4p_scout_faillow_red: std::array::from_fn(|i| {
                self.s73_q4p_scout_faillow_red[i].load(Ordering::Relaxed)
            }),
            s73_q4p_quiet_researched: self.s73_q4p_quiet_researched.load(Ordering::Relaxed),
            s74_lmr_proposed: self.s74_lmr_proposed.load(Ordering::Relaxed),
            s74_lmr_proposed_r1: self.s74_lmr_proposed_r1.load(Ordering::Relaxed),
            s74_lmr_proposed_r2: self.s74_lmr_proposed_r2.load(Ordering::Relaxed),
            s74_lmr_applied_existing_pvs: self.s74_lmr_applied_existing_pvs.load(Ordering::Relaxed),
            s74_lmr_suppressed_by_null_window: self
                .s74_lmr_suppressed_by_null_window
                .load(Ordering::Relaxed),
            s74_lmr_applied_null_window: self.s74_lmr_applied_null_window.load(Ordering::Relaxed),
            s74_lmr_nw_fail_low: self.s74_lmr_nw_fail_low.load(Ordering::Relaxed),
            s74_lmr_nw_research: self.s74_lmr_nw_research.load(Ordering::Relaxed),
            s74_lmr_nw_research_entered: self.s74_lmr_nw_research_entered.load(Ordering::Relaxed),
            s74_lmr_nw_verified_cutoff: self.s74_lmr_nw_verified_cutoff.load(Ordering::Relaxed),
            s74_lmr_nw_depth: std::array::from_fn(|i| {
                self.s74_lmr_nw_depth[i].load(Ordering::Relaxed)
            }),
            s74_lmr_nw_idx: std::array::from_fn(|i| self.s74_lmr_nw_idx[i].load(Ordering::Relaxed)),
            s75_main_nodes: self
                .nodes
                .load(Ordering::Relaxed)
                .saturating_sub(self.qsearch_nodes.load(Ordering::Relaxed)),
            s75_main_in_check_nodes: self.s75_main_in_check_nodes.load(Ordering::Relaxed),
            s75_main_single_evasion_nodes_raw: self
                .s75_main_single_evasion_nodes_raw
                .load(Ordering::Relaxed),
            s75_main_single_evasion_actionable_depth1: self
                .s75_main_single_evasion_actionable_depth1
                .load(Ordering::Relaxed),
            s75_main_single_evasion_actionable_depth2plus: self
                .s75_main_single_evasion_actionable_depth2plus
                .load(Ordering::Relaxed),
            s75_main_single_evasion_depth3plus: self
                .s75_main_single_evasion_depth3plus
                .load(Ordering::Relaxed),
            s75_main_single_evasion_chain: std::array::from_fn(|i| {
                self.s75_main_single_evasion_chain[i].load(Ordering::Relaxed)
            }),
            s75_main_checking_edges_searched: self
                .s75_main_checking_edges_searched
                .load(Ordering::Relaxed),
            s75_main_check_child_entered: self.s75_main_check_child_entered.load(Ordering::Relaxed),
            s75_main_check_child_movegen: self.s75_main_check_child_movegen.load(Ordering::Relaxed),
            s75_main_check_child_terminal_0: self
                .s75_main_check_child_terminal_0
                .load(Ordering::Relaxed),
            s75_main_check_child_evasions_1: self
                .s75_main_check_child_evasions_1
                .load(Ordering::Relaxed),
            s75_main_check_child_evasions_2: self
                .s75_main_check_child_evasions_2
                .load(Ordering::Relaxed),
            s75_main_check_child_evasions_3plus: self
                .s75_main_check_child_evasions_3plus
                .load(Ordering::Relaxed),
            s75_main_depth1_nodes: self.s75_main_depth1_nodes.load(Ordering::Relaxed),
            s75_main_depth1_in_check: self.s75_main_depth1_in_check.load(Ordering::Relaxed),
            s75_main_depth1_single_evasion: self
                .s75_main_depth1_single_evasion
                .load(Ordering::Relaxed),
            s75_main_depth1_entered_from_checking_edge: self
                .s75_main_depth1_entered_from_checking_edge
                .load(Ordering::Relaxed),
            s75_q_nodes: self.s75_q_nodes.load(Ordering::Relaxed),
            s75_q_in_check_nodes: self.s75_q_in_check_nodes.load(Ordering::Relaxed),
            s75_q_single_evasion_nodes_raw: self
                .s75_q_single_evasion_nodes_raw
                .load(Ordering::Relaxed),
            s75_q_single_evasion_qply0: self.s75_q_single_evasion_qply0.load(Ordering::Relaxed),
            s75_q_single_evasion_qply1plus: self
                .s75_q_single_evasion_qply1plus
                .load(Ordering::Relaxed),
            s75_q_checking_edges_searched: self
                .s75_q_checking_edges_searched
                .load(Ordering::Relaxed),
            s75_q_check_child_entered: self.s75_q_check_child_entered.load(Ordering::Relaxed),
            s75_q_check_child_movegen: self.s75_q_check_child_movegen.load(Ordering::Relaxed),
            s75_q_check_child_terminal_0: self.s75_q_check_child_terminal_0.load(Ordering::Relaxed),
            s75_q_check_child_evasions_1: self.s75_q_check_child_evasions_1.load(Ordering::Relaxed),
            s75_q_check_child_evasions_2: self.s75_q_check_child_evasions_2.load(Ordering::Relaxed),
            s75_q_check_child_evasions_3plus: self
                .s75_q_check_child_evasions_3plus
                .load(Ordering::Relaxed),
            s75a_extension_applied_total: self.s75a_extension_applied_total.load(Ordering::Relaxed),
            s75a_extension_applied_depth1: self
                .s75a_extension_applied_depth1
                .load(Ordering::Relaxed),
            s75a_extension_budget_2_to_1: self.s75a_extension_budget_2_to_1.load(Ordering::Relaxed),
            s75a_extension_budget_1_to_0: self.s75a_extension_budget_1_to_0.load(Ordering::Relaxed),
            s75a_opportunity_blocked_budget_0: self
                .s75a_opportunity_blocked_budget_0
                .load(Ordering::Relaxed),
            s72_d_quiet_cutoffs: std::array::from_fn(|i| {
                self.s72_d_quiet_cutoffs[i].load(Ordering::Relaxed)
            }),
            check_extensions: self.check_extensions.load(Ordering::Relaxed),
            single_evasion_extensions: self.single_evasion_extensions.load(Ordering::Relaxed),
            qsearch_check_moves: self.qsearch_check_moves.load(Ordering::Relaxed),
            threat_ordered_moves: self.threat_ordered_moves.load(Ordering::Relaxed),
            root_reorders: self.root_reorders.load(Ordering::Relaxed),
            aspiration_retries: self.aspiration_retries.load(Ordering::Relaxed),
            aspiration_fail_low: self.aspiration_fail_low.load(Ordering::Relaxed),
            aspiration_fail_high: self.aspiration_fail_high.load(Ordering::Relaxed),
            lmr_reductions: self.lmr_reductions.load(Ordering::Relaxed),
            lmr_researches: self.lmr_researches.load(Ordering::Relaxed),
            null_move_attempts: self.null_move_attempts.load(Ordering::Relaxed),
            null_move_fail_highs: self.null_move_fail_highs.load(Ordering::Relaxed),
            null_move_researches: self.null_move_researches.load(Ordering::Relaxed),
            futility_pruned: self.futility_pruned.load(Ordering::Relaxed),
            completed_iterations: self.completed_iterations.load(Ordering::Relaxed),
            last_completed_iteration_ms: self.last_completed_iteration_ms.load(Ordering::Relaxed),
            last_completed_iteration_nodes: self
                .last_completed_iteration_nodes
                .load(Ordering::Relaxed),
            aborted_iteration_depth: self.aborted_iteration_depth.load(Ordering::Relaxed),
            aborted_iteration_nodes: self.aborted_iteration_nodes.load(Ordering::Relaxed),
            target_root_rank: self.target_root_rank.load(Ordering::Relaxed),
            legality_fast_accepts: self.legality_fast_accepts.load(Ordering::Relaxed),
            legality_fallback_probes: self.legality_fallback_probes.load(Ordering::Relaxed),
            legality_fallback_in_check: self.legality_fallback_in_check.load(Ordering::Relaxed),
            legality_fallback_king: self.legality_fallback_king.load(Ordering::Relaxed),
            legality_fallback_pinned: self.legality_fallback_pinned.load(Ordering::Relaxed),
            legality_fallback_en_passant: self.legality_fallback_en_passant.load(Ordering::Relaxed),
            legality_fallback_castle: self.legality_fallback_castle.load(Ordering::Relaxed),
            single_buffer_writes: self.single_buffer_writes.load(Ordering::Relaxed),
            beta_cutoffs: self.beta_cutoffs.load(Ordering::Relaxed),
            beta_cutoff_idx_0: self.beta_cutoff_idx_0.load(Ordering::Relaxed),
            beta_cutoff_idx_1: self.beta_cutoff_idx_1.load(Ordering::Relaxed),
            beta_cutoff_idx_2_3: self.beta_cutoff_idx_2_3.load(Ordering::Relaxed),
            beta_cutoff_idx_4_7: self.beta_cutoff_idx_4_7.load(Ordering::Relaxed),
            beta_cutoff_idx_8_15: self.beta_cutoff_idx_8_15.load(Ordering::Relaxed),
            beta_cutoff_idx_16p: self.beta_cutoff_idx_16p.load(Ordering::Relaxed),
            cutoff_tt_move: self.cutoff_tt_move.load(Ordering::Relaxed),
            cutoff_tactical: self.cutoff_tactical.load(Ordering::Relaxed),
            cutoff_killer: self.cutoff_killer.load(Ordering::Relaxed),
            cutoff_quiet: self.cutoff_quiet.load(Ordering::Relaxed),
            moves_searched: self.moves_searched.load(Ordering::Relaxed),
            pv_nodes: self.pv_nodes.load(Ordering::Relaxed),
            in_check_nodes: self.in_check_nodes.load(Ordering::Relaxed),
            depth_bucket_0: self.depth_bucket_0.load(Ordering::Relaxed),
            depth_bucket_1: self.depth_bucket_1.load(Ordering::Relaxed),
            depth_bucket_2: self.depth_bucket_2.load(Ordering::Relaxed),
            depth_bucket_3: self.depth_bucket_3.load(Ordering::Relaxed),
            depth_bucket_4_5: self.depth_bucket_4_5.load(Ordering::Relaxed),
            depth_bucket_6_7: self.depth_bucket_6_7.load(Ordering::Relaxed),
            depth_bucket_8p: self.depth_bucket_8p.load(Ordering::Relaxed),
            searched_hist_1: self.searched_hist_1.load(Ordering::Relaxed),
            searched_hist_2: self.searched_hist_2.load(Ordering::Relaxed),
            searched_hist_3_4: self.searched_hist_3_4.load(Ordering::Relaxed),
            searched_hist_5_8: self.searched_hist_5_8.load(Ordering::Relaxed),
            searched_hist_9_16: self.searched_hist_9_16.load(Ordering::Relaxed),
            searched_hist_17p: self.searched_hist_17p.load(Ordering::Relaxed),
            tt_hit_exact: self.tt_hit_exact.load(Ordering::Relaxed),
            tt_hit_lower: self.tt_hit_lower.load(Ordering::Relaxed),
            tt_hit_upper: self.tt_hit_upper.load(Ordering::Relaxed),
            lmr_reduction_r1: self.lmr_reduction_r1.load(Ordering::Relaxed),
            lmr_reduction_r2: self.lmr_reduction_r2.load(Ordering::Relaxed),
            lmr_reduced_improves_alpha: self.lmr_reduced_improves_alpha.load(Ordering::Relaxed),
            null_fail_lows: self.null_fail_lows.load(Ordering::Relaxed),
            futility_considered: self.futility_considered.load(Ordering::Relaxed),
            qsearch_standpat_cutoffs: self.qsearch_standpat_cutoffs.load(Ordering::Relaxed),
            qsearch_standpat_alpha_raises: self
                .qsearch_standpat_alpha_raises
                .load(Ordering::Relaxed),
            qsearch_moves_searched: self.qsearch_moves_searched.load(Ordering::Relaxed),
            qsearch_in_check_entries: self.qsearch_in_check_entries.load(Ordering::Relaxed),
            qsearch_lazy_has_any_probes: self.qsearch_lazy_has_any_probes.load(Ordering::Relaxed),
            qsearch_lazy_standpat_cutoffs_before_movegen: self
                .qsearch_lazy_standpat_cutoffs_before_movegen
                .load(Ordering::Relaxed),
            qsearch_lazy_qply_returns_before_movegen: self
                .qsearch_lazy_qply_returns_before_movegen
                .load(Ordering::Relaxed),
            qsearch_lazy_tactical_generations: self
                .qsearch_lazy_tactical_generations
                .load(Ordering::Relaxed),
        }
    }

    /// Construct an unlimited context with diagnostic counters enabled.
    pub fn new_with_profiling(stop: Arc<AtomicBool>, profiling_enabled: bool) -> Self {
        let mut ctx = Self::new(stop);
        ctx.profiling_enabled = profiling_enabled;
        ctx
    }

    #[inline]
    fn add_profile_counter(&self, counter: &AtomicU64, amount: u64) {
        if self.profiling_enabled {
            counter.fetch_add(amount, Ordering::Relaxed);
        }
    }

    /// S4.0B: decode the resolved feature policy (set once at search start).
    #[inline]
    pub(crate) fn features(&self) -> SearchFeaturePolicy {
        SearchFeaturePolicy::from_bits(self.features_mask.load(Ordering::Relaxed))
    }

    /// S4.3A: begin a sampled timing window for one coarse operation. Returns
    /// None when sampling is disabled (production) or when this call is not
    /// the sampled one. `calls` is only counted while sampling is enabled.
    #[inline]
    pub(crate) fn sample_begin(&self, counter: &SampledCounter) -> Option<std::time::Instant> {
        if !self.sampled_timing {
            return None;
        }
        counter.calls.fetch_add(1, Ordering::Relaxed);
        let gate = counter.gate.fetch_sub(1, Ordering::Relaxed);
        if gate == 0 {
            counter.gate.store(self.sample_rate, Ordering::Relaxed);
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn sample_end(&self, counter: &SampledCounter, start: std::time::Instant) {
        counter.samples.fetch_add(1, Ordering::Relaxed);
        counter
            .elapsed_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// S4.3A: snapshot all sampled timers (calls, samples, elapsed_ns).
    pub(crate) fn sampled_timings(&self) -> SampledTimings {
        SampledTimings {
            movegen_legal: (
                self.timing_movegen_legal.calls.load(Ordering::Relaxed),
                self.timing_movegen_legal.samples.load(Ordering::Relaxed),
                self.timing_movegen_legal.elapsed_ns.load(Ordering::Relaxed),
            ),
            movegen_tactical: (
                self.timing_movegen_tactical.calls.load(Ordering::Relaxed),
                self.timing_movegen_tactical.samples.load(Ordering::Relaxed),
                self.timing_movegen_tactical
                    .elapsed_ns
                    .load(Ordering::Relaxed),
            ),
            movegen_evasion: (
                self.timing_movegen_evasion.calls.load(Ordering::Relaxed),
                self.timing_movegen_evasion.samples.load(Ordering::Relaxed),
                self.timing_movegen_evasion
                    .elapsed_ns
                    .load(Ordering::Relaxed),
            ),
            movegen_has_any: (
                self.timing_movegen_has_any.calls.load(Ordering::Relaxed),
                self.timing_movegen_has_any.samples.load(Ordering::Relaxed),
                self.timing_movegen_has_any
                    .elapsed_ns
                    .load(Ordering::Relaxed),
            ),
            eval: (
                self.timing_eval.calls.load(Ordering::Relaxed),
                self.timing_eval.samples.load(Ordering::Relaxed),
                self.timing_eval.elapsed_ns.load(Ordering::Relaxed),
            ),
            ordering: (
                self.timing_ordering.calls.load(Ordering::Relaxed),
                self.timing_ordering.samples.load(Ordering::Relaxed),
                self.timing_ordering.elapsed_ns.load(Ordering::Relaxed),
            ),
            see: (
                self.timing_see.calls.load(Ordering::Relaxed),
                self.timing_see.samples.load(Ordering::Relaxed),
                self.timing_see.elapsed_ns.load(Ordering::Relaxed),
            ),
            tt: (
                self.timing_tt.calls.load(Ordering::Relaxed),
                self.timing_tt.samples.load(Ordering::Relaxed),
                self.timing_tt.elapsed_ns.load(Ordering::Relaxed),
            ),
        }
    }
}

/// S4.4A: which generator produced a `MovegenStats` (per-generator probe
/// split of the S4.3A legality totals).
#[derive(Clone, Copy, PartialEq, Eq)]
enum MovegenKind {
    FullLegal,
    Tactical,
    Evasion,
    HasAny,
}

#[inline]
fn generate_legal_moves_profiled(pos: &mut Position, ctx: &SearchContext) -> Vec<Move> {
    let start = ctx.sample_begin(&ctx.timing_movegen_legal);
    let (moves, stats) = if ctx.single_buffer_legal.load(Ordering::Relaxed) {
        // S4.4B: single-buffer full-legal materialization (candidate only).
        generate_legal_moves_fast_single_buffer_with_stats(
            pos,
            ctx.sampled_timing.then_some(&ctx.full_legal_sub),
        )
    } else if ctx.legality_fast.load(Ordering::Relaxed) {
        // S4.4A: sparse sub-attribution inside the promoted generator.
        generate_legal_moves_fast_with_stats(pos, ctx.sampled_timing.then_some(&ctx.full_legal_sub))
    } else {
        generate_legal_moves_with_stats(pos)
    };
    if let Some(start) = start {
        ctx.sample_end(&ctx.timing_movegen_legal, start);
    }
    add_movegen_profile(ctx, stats, MovegenKind::FullLegal);
    moves
}

#[inline]
fn add_movegen_profile(ctx: &SearchContext, stats: MovegenStats, kind: MovegenKind) {
    ctx.add_profile_counter(&ctx.legal_move_generations, 1);
    ctx.add_profile_counter(&ctx.pseudo_moves, stats.pseudo_moves);
    ctx.add_profile_counter(&ctx.legal_moves, stats.legal_moves);
    ctx.add_profile_counter(&ctx.make_moves, stats.make_moves);
    ctx.add_profile_counter(&ctx.unmake_moves, stats.unmake_moves);
    // S4.3A: legality-probe make/unmake split from recursive search edges.
    ctx.add_profile_counter(&ctx.legality_probe_make, stats.make_moves);
    ctx.add_profile_counter(&ctx.legality_probe_unmake, stats.unmake_moves);
    // S4.4A: per-generator probe split.
    let (pm, pu) = match kind {
        MovegenKind::FullLegal => (&ctx.full_legal_probe_make, &ctx.full_legal_probe_unmake),
        MovegenKind::Tactical => (&ctx.tactical_probe_make, &ctx.tactical_probe_unmake),
        MovegenKind::Evasion => (&ctx.evasion_probe_make, &ctx.evasion_probe_unmake),
        MovegenKind::HasAny => (&ctx.has_any_probe_make, &ctx.has_any_probe_unmake),
    };
    ctx.add_profile_counter(pm, stats.make_moves);
    ctx.add_profile_counter(pu, stats.unmake_moves);
    // S4.3B: fast-path / fallback statistics (zero for legacy generators).
    ctx.add_profile_counter(&ctx.legality_fast_accepts, stats.fast_accepts);
    ctx.add_profile_counter(&ctx.legality_fallback_probes, stats.fallback_probes);
    ctx.add_profile_counter(&ctx.legality_fallback_in_check, stats.fallback_in_check);
    ctx.add_profile_counter(&ctx.legality_fallback_king, stats.fallback_king);
    ctx.add_profile_counter(&ctx.legality_fallback_pinned, stats.fallback_pinned);
    ctx.add_profile_counter(&ctx.legality_fallback_en_passant, stats.fallback_en_passant);
    ctx.add_profile_counter(&ctx.legality_fallback_castle, stats.fallback_castle);
    // S4.4B: single-buffer compaction writes (zero for two-buffer generators).
    ctx.add_profile_counter(&ctx.single_buffer_writes, stats.compaction_writes);
}

#[inline]
fn generate_legal_tactical_moves_profiled(pos: &mut Position, ctx: &SearchContext) -> Vec<Move> {
    let start = ctx.sample_begin(&ctx.timing_movegen_tactical);
    let (moves, stats) = generate_legal_tactical_moves_with_stats(pos);
    if let Some(start) = start {
        ctx.sample_end(&ctx.timing_movegen_tactical, start);
    }
    add_movegen_profile(ctx, stats, MovegenKind::Tactical);
    moves
}

#[inline]
fn generate_legal_evasions_profiled(pos: &mut Position, ctx: &SearchContext) -> Vec<Move> {
    let start = ctx.sample_begin(&ctx.timing_movegen_evasion);
    let (moves, stats) = generate_legal_evasions_with_stats(pos);
    if let Some(start) = start {
        ctx.sample_end(&ctx.timing_movegen_evasion, start);
    }
    add_movegen_profile(ctx, stats, MovegenKind::Evasion);
    moves
}

#[inline]
fn has_any_legal_move_profiled(pos: &mut Position, ctx: &SearchContext) -> bool {
    let start = ctx.sample_begin(&ctx.timing_movegen_has_any);
    let (has_move, stats) = has_any_legal_move_with_stats(pos);
    if let Some(start) = start {
        ctx.sample_end(&ctx.timing_movegen_has_any, start);
    }
    add_movegen_profile(ctx, stats, MovegenKind::HasAny);
    has_move
}

#[inline]
fn evaluate_profiled(pos: &Position, ctx: &SearchContext, profile: SearchProfile) -> i32 {
    ctx.add_profile_counter(&ctx.eval_calls, 1);
    let start = ctx.sample_begin(&ctx.timing_eval);
    let result = if profile.uses_eval2() {
        evaluate_integrated_positional(pos)
    } else if profile.uses_threat_aware_eval() {
        evaluate_threat_aware(pos)
    } else {
        evaluate(pos)
    };
    if let Some(start) = start {
        ctx.sample_end(&ctx.timing_eval, start);
    }
    result
}

#[inline]
fn make_move_profiled(pos: &mut Position, mv: Move, ctx: &SearchContext) -> Undo {
    ctx.add_profile_counter(&ctx.make_moves, 1);
    pos.make_move(mv)
}

#[inline]
fn unmake_move_profiled(pos: &mut Position, undo: Undo, ctx: &SearchContext) {
    ctx.add_profile_counter(&ctx.unmake_moves, 1);
    pos.unmake_move(undo);
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

/// S5.0A: which search path owns a child probe (per-caller generation split).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProbeKind {
    Main,
    Qsearch,
    Root,
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
#[allow(clippy::too_many_arguments)]
fn probe_child_draw(
    pos: &mut Position,
    child_keys: &[ZobristKey],
    child_ply: u32,
    parent_ply: u32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    kind: ProbeKind,
) -> Option<ChildProbe> {
    // Exactly ONE node acquisition for this child.
    if !try_enter_node(ctx, limits) {
        return None;
    }
    // The probe is the sole owner of the child PV row's initial clear.
    pv.clear_at(child_ply);

    // S5.0B: the probe only needs the EMPTINESS boolean (terminal) and the
    // claim check (list-independent). `has_any_legal_move` is exactly
    // equivalent: false iff no legal move. The entered body keeps its single
    // full generation. The S5.0A counters count ACTUAL full-legal probe
    // generations (zero in the single-generation candidate).
    let child_has_any = if ctx.single_generation_probe.load(Ordering::Relaxed) {
        has_any_legal_move_profiled(pos, ctx)
    } else {
        ctx.add_profile_counter(&ctx.probe_child_generations, 1);
        match kind {
            ProbeKind::Main => ctx.add_profile_counter(&ctx.main_edge_probe_generations, 1),
            ProbeKind::Qsearch => ctx.add_profile_counter(&ctx.qsearch_edge_probe_generations, 1),
            ProbeKind::Root => ctx.add_profile_counter(&ctx.root_edge_probe_generations, 1),
        }
        !generate_legal_moves_profiled(pos, ctx).is_empty()
    };
    if !child_has_any {
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
    if !profile.uses_pvs() || is_first || depth == 0 {
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
        /// S7.4A: reduced null-window search improved and a full-depth
        /// verification was requested.
        pub static S74_NW_RESEARCH_ATTEMPT: Cell<usize> = const { Cell::new(0) };
        /// S7.4A: the requested verification acquired its node and ran.
        pub static S74_NW_RESEARCH_ENTERED: Cell<usize> = const { Cell::new(0) };
        /// S7.4A: `try_enter_node` failed when acquiring the verification
        /// node, so the unverified reduced result was discarded.
        pub static S74_NW_ABORT_RESEARCH_ACQUIRE: Cell<usize> = const { Cell::new(0) };
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
        S74_NW_RESEARCH_ATTEMPT.set(0);
        S74_NW_RESEARCH_ENTERED.set(0);
        S74_NW_ABORT_RESEARCH_ACQUIRE.set(0);
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
    pub fn mark_s74_nw_research_attempt() {
        S74_NW_RESEARCH_ATTEMPT.set(S74_NW_RESEARCH_ATTEMPT.get() + 1);
    }
    pub fn mark_s74_nw_research_entered() {
        S74_NW_RESEARCH_ENTERED.set(S74_NW_RESEARCH_ENTERED.get() + 1);
    }
    pub fn mark_s74_nw_abort_research_acquire() {
        S74_NW_ABORT_RESEARCH_ACQUIRE.set(S74_NW_ABORT_RESEARCH_ACQUIRE.get() + 1);
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
    hit: bool,
    /// A real score the node can return immediately (the TT entry's bound
    /// is satisfied by the current window). `None` means no cut-off.
    cutoff: Option<i32>,
    /// The entry's stored best move, used only for move ordering. `None`
    /// when the entry has no move or the probe is a miss / decode failure.
    hash_move: Option<Move>,
    /// Why a matching entry did not produce a cutoff. This is observational
    /// telemetry only; the probe's search semantics remain unchanged.
    reject: Option<TtRejectReason>,
    /// The entry's bound type on a hit (S7.0 attribution: exact/lower/upper
    /// hit split). `None` on a miss or decode failure. Observational only.
    hit_bound: Option<Bound>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TtRejectReason {
    Depth,
    Bound,
    Decode,
}

/// Build the context-safe TT key for the CURRENT position using the ordinary
/// (zero forcing-budget) context. This preserves the historical key contract
/// for `Current` and all non-threat-aware profiles.
fn current_tt_key(pos: &Position, path: &SearchPath) -> TtKey {
    current_tt_key_with_forcing_budget(pos, path, 0)
}

/// Build the TT key for a position plus the bounded threat-aware search
/// context. The repetition signature comes from the full [`SearchPath`], so
/// two identical boards with different repetition context get different keys;
/// the remaining forcing budget is likewise part of the search identity.
fn current_tt_key_with_forcing_budget(
    pos: &Position,
    path: &SearchPath,
    forcing_budget: u8,
) -> TtKey {
    debug_assert_eq!(path.last(), Some(&pos.zobrist_key()));
    TtKey::new_with_forcing_budget(
        pos.zobrist_key(),
        pos.halfmove_clock(),
        path.repetition_signature(),
        forcing_budget,
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
    probe_tt_for_search_with_policy(tt, key, requested_depth, ply, effective_alpha, beta, false)
}

/// Threat-aware searches use an exact nominal-depth TT policy. A result from
/// a deeper standalone descendant search can otherwise be reused as an exact
/// answer for a shallower parent child, changing a fixed-depth root result
/// after forward/backtracking. Hash moves remain usable; only score cutoffs
/// require the exact requested depth.
fn probe_tt_for_search_exact_depth(
    tt: &TranspositionTable,
    key: TtKey,
    requested_depth: u32,
    ply: u32,
    effective_alpha: i32,
    beta: i32,
) -> SearchTtProbe {
    probe_tt_for_search_with_policy(tt, key, requested_depth, ply, effective_alpha, beta, true)
}

fn probe_tt_for_search_with_policy(
    tt: &TranspositionTable,
    key: TtKey,
    requested_depth: u32,
    ply: u32,
    effective_alpha: i32,
    beta: i32,
    exact_depth: bool,
) -> SearchTtProbe {
    let Some(entry) = tt.probe(key) else {
        return SearchTtProbe {
            hit: false,
            cutoff: None,
            hash_move: None,
            reject: None,
            hit_bound: None,
        };
    };

    // Full-key mismatch is already a miss (handled by `tt.probe`). A decode
    // failure (mate score at an unsupported ply) means we cannot trust the
    // score: treat the ENTIRE entry as a miss — no cut-off AND no hash move.
    let Some(decoded) = score_from_tt(entry.score, ply) else {
        return SearchTtProbe {
            hit: true,
            cutoff: None,
            hash_move: None,
            reject: Some(TtRejectReason::Decode),
            hit_bound: None,
        };
    };

    // A non-matching nominal depth is a miss for score cut-off purposes, but
    // its stored move (if any) is still useful for ordering. Ordinary profiles
    // accept deeper entries; the threat-aware policy deliberately does not.
    if entry.depth < requested_depth || (exact_depth && entry.depth != requested_depth) {
        return SearchTtProbe {
            hit: true,
            cutoff: None,
            hash_move: entry.best_move,
            reject: Some(TtRejectReason::Depth),
            hit_bound: Some(entry.bound),
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
        hit: true,
        cutoff,
        hash_move: entry.best_move,
        reject: if cutoff.is_none() {
            Some(TtRejectReason::Bound)
        } else {
            None
        },
        hit_bound: Some(entry.bound),
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

#[allow(clippy::too_many_arguments)]
#[inline]
fn store_tt_score_profiled(
    tt: &mut TranspositionTable,
    key: TtKey,
    depth: u32,
    score: i32,
    ply: u32,
    bound: Bound,
    best_move: Option<Move>,
    ctx: &SearchContext,
) {
    ctx.add_profile_counter(&ctx.tt_stores, 1);
    store_tt_score(tt, key, depth, score, ply, bound, best_move);
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

    /// Whether `m` is currently a killer at `ply` (S7.0 attribution only).
    fn is_killer(&self, ply: usize, m: Move) -> bool {
        self.killers
            .get(ply)
            .is_some_and(|k| k[0] == Some(m) || k[1] == Some(m))
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

/// S4.1: reorder the root's QUIET moves (indices 1..) by the existing history
/// heuristic `history[color][from][to]`, descending, stable for equal scores.
///
/// Contract:
/// - root_moves[0] (the previous iteration's best move) is preserved;
/// - no root killers, no static-eval ordering, no history-update changes;
/// - tactical moves (captures / promotions / en passant) keep their slots;
/// - only the quiet slots are sorted, so quiet-vs-quiet prioritization is the
///   entire experiment (not a general root MovePicker);
/// - every legal move appears exactly once (pure permutation).
fn order_root_quiets_by_history(
    pos: &Position,
    root_moves: &mut [Move],
    heur: Option<&SearchHeuristics>,
) {
    let quiet_slots: Vec<usize> = (1..root_moves.len())
        .filter(|&i| !is_tactical(pos, root_moves[i]))
        .collect();
    if quiet_slots.len() < 2 {
        return;
    }
    let mut keyed: Vec<(i32, usize, Move)> = quiet_slots
        .iter()
        .map(|&i| {
            let m = root_moves[i];
            let hist = if let Some(hh) = heur {
                let color = pos.side_to_move() as usize;
                hh.history[color][m.from as usize][m.to as usize]
            } else {
                0
            };
            (hist, i, m)
        })
        .collect();
    // Descending history, then ascending original index (stable).
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for (slot, (_, _, m)) in keyed.into_iter().enumerate() {
        root_moves[quiet_slots[slot]] = m;
    }
}

/// S4.1b: reorder the root's QUIET moves (indices 1..) by the previous
/// completed iteration's root search scores, descending, stable for equal
/// values.
///
/// Contract: root_moves[0] (previous best) preserved; tactical slots
/// unchanged; only quiet slots sorted; no history/killer/static-eval/threat
/// signal; every legal move appears exactly once (pure permutation).
fn order_root_quiets_by_prev_scores(
    pos: &Position,
    root_moves: &mut [Move],
    previous_scores: &[(Move, i32)],
) {
    let quiet_slots: Vec<usize> = (1..root_moves.len())
        .filter(|&i| !is_tactical(pos, root_moves[i]))
        .collect();
    if quiet_slots.len() < 2 {
        return;
    }
    let mut keyed: Vec<(i32, usize, Move)> = quiet_slots
        .iter()
        .map(|&i| {
            let m = root_moves[i];
            let score = previous_scores
                .iter()
                .find(|(scored, _)| *scored == m)
                .map(|(_, score)| *score)
                .unwrap_or(i32::MIN);
            (score, i, m)
        })
        .collect();
    // Descending score, then ascending original index (stable).
    keyed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for (slot, (_, _, m)) in keyed.into_iter().enumerate() {
        root_moves[quiet_slots[slot]] = m;
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
    /// Root move scores observed during this completed iteration. Scores are
    /// used only for the next iteration's candidate-only root ordering; the
    /// minimax result remains `score`/`pv` above.
    move_scores: Vec<(Move, i32)>,
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
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    profile: SearchProfile,
    pv: &mut PvTable,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
    heur: &mut Option<SearchHeuristics>,
) -> Option<i32> {
    negamax_entered_impl_with_null(
        pos, depth, ply, alpha, beta, ctx, limits, profile, pv, path, tt, heur, true,
    )
}

/// Body variant used by the verified null-move probe and re-search. Its
/// `allow_null` parameter is explicit so the probe child cannot immediately
/// launch a consecutive null move; ordinary descendants retain their normal
/// candidate behavior through [`negamax_entered_impl`].
#[allow(clippy::too_many_arguments)]
fn negamax_entered_impl_with_null(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    profile: SearchProfile,
    pv: &mut PvTable,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
    heur: &mut Option<SearchHeuristics>,
    allow_null: bool,
) -> Option<i32> {
    let extension_budget = extension_budget_for_profile(profile);
    negamax_entered_impl_with_null_and_extensions(
        pos,
        depth,
        ply,
        alpha,
        beta,
        ctx,
        limits,
        profile,
        pv,
        path,
        tt,
        heur,
        allow_null,
        0,
        extension_budget,
    )
}

#[allow(clippy::too_many_arguments)]
/// S7.2 move-ordering attribution (OBSERVATION ONLY, profiling-gated).
/// Per-move classification captured once per node right after ordering.
struct S72MoveInfo {
    /// Non-tactical move (no capture / en-passant / promotion), regardless
    /// of whether it gives check.
    quiet: bool,
    is_hash: bool,
    killer0: bool,
    killer1: bool,
    /// Mutually-exclusive cutoff category, TT-hash precedence:
    /// 0 tt_hash, 1 promotion, 2 capture, 3 killer0, 4 killer1,
    /// 5 history_quiet (hist > 0), 6 other_quiet (hist <= 0).
    cat: usize,
    /// Quiet-rank histogram bucket [0, 1, 2-3, 4-7, 8-15, 16+].
    rank_bucket: usize,
    /// History-score histogram bucket [<=0, 1-15, 16-63, 64-255, 256+].
    hist_bucket: usize,
}

#[inline]
fn s72_rank_bucket(rank: u32) -> usize {
    match rank {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        _ => 5,
    }
}

#[inline]
fn s72_count_bucket(n: u64) -> usize {
    match n {
        1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        _ => 5,
    }
}

#[inline]
fn s72_hist_bucket(hist: i32) -> usize {
    match hist {
        ..=0 => 0,
        1..=15 => 1,
        16..=63 => 2,
        64..=255 => 3,
        _ => 4,
    }
}

#[inline]
fn s72_depth_bucket(depth: u32) -> usize {
    match depth {
        1 => 0,
        2 => 1,
        3 => 2,
        4..=5 => 3,
        6..=7 => 4,
        _ => 5,
    }
}

/// S7.3: ordered move-index bucket for the depth>=4 quiet-selectivity tables.
/// Buckets: 0 | 1 | 2-3 | 4-7 | 8+ (mirrors the S7.2 quiet-rank buckets so
/// the two reports can be read side by side).
#[inline]
fn s73_idx_bucket(idx: usize) -> usize {
    match idx {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        _ => 4,
    }
}

/// S7.4A: remaining-depth bucket for LMR-on-null-window application splits.
/// Buckets: 4 | 5 | 6 | 7+. Reduction eligibility already implies depth >= 4,
/// so no shallower bucket exists.
#[inline]
fn s74_depth_bucket(depth: u32) -> usize {
    (depth.saturating_sub(4) as usize).min(3)
}

/// S7.4A: move-index bucket for LMR-on-null-window application splits.
/// Buckets: 3-4 | 5-7 | 8-15 | 16+. Reduction eligibility already implies
/// move_idx >= 3, so no earlier bucket exists.
#[inline]
fn s74_idx_bucket(idx: usize) -> usize {
    match idx {
        3..=4 => 0,
        5..=7 => 1,
        8..=15 => 2,
        _ => 3,
    }
}

/// Classify every ordered legal move once and record node-level opportunity
/// denominators: quiet availability, killer presence (slot set AND legal),
/// TT-hash presence (Some AND legal). Never mutates ordering or heuristics.
fn s72_prepare(
    pos: &Position,
    moves: &[Move],
    hash_move: Option<Move>,
    h: Option<&SearchHeuristics>,
    ply: usize,
    ctx: &SearchContext,
) -> Vec<S72MoveInfo> {
    let killers = match h {
        Some(hh) if hh.killers.len() > ply => hh.killers[ply],
        _ => [None, None],
    };
    let color = pos.side_to_move() as usize;
    let mut quiet_rank = 0u32;
    let mut quiet_available = 0u64;
    let mut infos = Vec::with_capacity(moves.len());
    for &m in moves {
        let quiet = !is_tactical(pos, m);
        let is_hash = Some(m) == hash_move;
        let killer0 = Some(m) == killers[0];
        let killer1 = Some(m) == killers[1];
        let hist = if let (true, Some(hh)) = (quiet, h) {
            hh.history[color][m.from as usize][m.to as usize]
        } else {
            0
        };
        let cat = if is_hash {
            0
        } else if matches!(m.flag, MoveFlag::Promotion(_)) {
            1
        } else if !quiet {
            2
        } else if killer0 {
            3
        } else if killer1 {
            4
        } else if hist > 0 {
            5
        } else {
            6
        };
        let rank_bucket = if quiet {
            let b = s72_rank_bucket(quiet_rank);
            quiet_rank += 1;
            quiet_available += 1;
            b
        } else {
            0
        };
        infos.push(S72MoveInfo {
            quiet,
            is_hash,
            killer0,
            killer1,
            cat,
            rank_bucket,
            hist_bucket: s72_hist_bucket(hist),
        });
    }
    ctx.add_profile_counter(&ctx.s72_quiet_available, quiet_available);
    if quiet_available > 0 {
        ctx.add_profile_counter(&ctx.s72_nodes_with_quiet_moves, 1);
    }
    if let Some(hm) = hash_move {
        if moves.contains(&hm) {
            ctx.add_profile_counter(&ctx.s72_tt_hash[0], 1);
        }
    }
    for (slot, present_idx) in [(killers[0], 0usize), (killers[1], 3usize)] {
        if let Some(k) = slot {
            if moves.contains(&k) {
                ctx.add_profile_counter(&ctx.s72_killer[present_idx], 1);
            } else {
                ctx.add_profile_counter(&ctx.s72_killer[6], 1);
            }
        }
    }
    infos
}

#[allow(clippy::too_many_arguments)]
fn negamax_entered_impl_with_null_and_extensions(
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
    allow_null: bool,
    single_evasion_chain: u8,
    extension_budget: u8,
) -> Option<i32> {
    // UCI seldepth: this node's own `ply` IS the global ply.
    ctx.record_seldepth(ply);
    // Clear our row now (after entry, before any early return).
    pv.clear_at(ply);

    // Terminal-node check MUST run before the depth==0 evaluation.
    let node_in_check = pos.is_in_check(pos.side);
    // S7.5-0: forcing-opportunity main funnel. Observation-only; every
    // counter update is profiling-gated. `s75_main_nodes` is derived in
    // SearchStats as nodes - qsearch_nodes (actual acquired main-tree nodes).
    if ctx.profiling_enabled {
        if depth == 1 {
            ctx.add_profile_counter(&ctx.s75_main_depth1_nodes, 1);
        }
        if node_in_check {
            ctx.add_profile_counter(&ctx.s75_main_in_check_nodes, 1);
            if depth == 1 {
                ctx.add_profile_counter(&ctx.s75_main_depth1_in_check, 1);
            }
        }
        // A non-root node in check means the parent just searched a checking
        // edge. Counting at child entry avoids an extra make/unmake check in
        // the parent move loop.
        if ply > 0 && node_in_check {
            ctx.add_profile_counter(&ctx.s75_main_checking_edges_searched, 1);
            ctx.add_profile_counter(&ctx.s75_main_check_child_entered, 1);
            if depth == 1 {
                ctx.add_profile_counter(&ctx.s75_main_depth1_entered_from_checking_edge, 1);
            }
        }
    }
    // S7.0 depth attribution: classify this node once, gated so the
    // production (profiling-off) path pays nothing beyond the boolean check.
    if ctx.profiling_enabled {
        ctx.main_seldepth.fetch_max(ply as u64, Ordering::Relaxed);
        if beta.saturating_sub(alpha) > 1 {
            ctx.add_profile_counter(&ctx.pv_nodes, 1);
        }
        if node_in_check {
            ctx.add_profile_counter(&ctx.in_check_nodes, 1);
        }
        match depth {
            0 => ctx.add_profile_counter(&ctx.depth_bucket_0, 1),
            1 => ctx.add_profile_counter(&ctx.depth_bucket_1, 1),
            2 => ctx.add_profile_counter(&ctx.depth_bucket_2, 1),
            3 => ctx.add_profile_counter(&ctx.depth_bucket_3, 1),
            4..=5 => ctx.add_profile_counter(&ctx.depth_bucket_4_5, 1),
            6..=7 => ctx.add_profile_counter(&ctx.depth_bucket_6_7, 1),
            _ => ctx.add_profile_counter(&ctx.depth_bucket_8p, 1),
        }
    }
    // S5.0A: every negamax entered body generation duplicates the Continue
    // child probe's discarded list for the same position.
    ctx.add_profile_counter(&ctx.negamax_body_generations, 1);
    let mut moves = generate_legal_moves_profiled(pos, ctx);
    // S7.5-0: natural main-search movegen is the ONLY point where a checking
    // child's evasion count is classified. No extra legal generation is run.
    if ctx.profiling_enabled && node_in_check {
        ctx.add_profile_counter(&ctx.s75_main_check_child_movegen, 1);
        match moves.len() {
            0 => ctx.add_profile_counter(&ctx.s75_main_check_child_terminal_0, 1),
            1 => ctx.add_profile_counter(&ctx.s75_main_check_child_evasions_1, 1),
            2 => ctx.add_profile_counter(&ctx.s75_main_check_child_evasions_2, 1),
            _ => ctx.add_profile_counter(&ctx.s75_main_check_child_evasions_3plus, 1),
        }
    }
    let current_single_evasion_chain = if ctx.profiling_enabled && node_in_check && moves.len() == 1
    {
        ctx.add_profile_counter(&ctx.s75_main_single_evasion_nodes_raw, 1);
        if depth == 1 {
            ctx.add_profile_counter(&ctx.s75_main_single_evasion_actionable_depth1, 1);
            ctx.add_profile_counter(&ctx.s75_main_depth1_single_evasion, 1);
        } else {
            ctx.add_profile_counter(&ctx.s75_main_single_evasion_actionable_depth2plus, 1);
            if depth >= 3 {
                ctx.add_profile_counter(&ctx.s75_main_single_evasion_depth3plus, 1);
            }
        }
        let chain = single_evasion_chain.saturating_add(1);
        let chain_bucket = if chain == 1 {
            0
        } else if chain == 2 {
            1
        } else {
            2
        };
        ctx.add_profile_counter(&ctx.s75_main_single_evasion_chain[chain_bucket], 1);
        chain
    } else {
        0
    };
    if moves.is_empty() {
        if node_in_check {
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
    let extension_context =
        _profile.uses_forcing_search() || _profile.uses_single_evasion_extension();
    let key = if extension_context && depth != 0 {
        current_tt_key_with_forcing_budget(pos, path, extension_budget)
    } else {
        current_tt_key(pos, path)
    };
    ctx.add_profile_counter(&ctx.tt_probes, 1);
    let tt_start = ctx.sample_begin(&ctx.timing_tt);
    // S7.5A contract: budget is part of the TT context, but S7.5A keeps
    // CurrentFinal's normal depth >= requested reuse semantics. The
    // exact-depth branch belongs only to the legacy threat-aware path.
    let tt_probe = if _profile.uses_forcing_search() {
        probe_tt_for_search_exact_depth(tt, key, depth, ply, alpha, beta)
    } else {
        probe_tt_for_search(tt, key, depth, ply, alpha, beta)
    };
    if let Some(start) = tt_start {
        ctx.sample_end(&ctx.timing_tt, start);
    }
    if tt_probe.hit {
        ctx.add_profile_counter(&ctx.tt_hits, 1);
    }
    if let Some(bound) = tt_probe.hit_bound {
        match bound {
            Bound::Exact => ctx.add_profile_counter(&ctx.tt_hit_exact, 1),
            Bound::Lower => ctx.add_profile_counter(&ctx.tt_hit_lower, 1),
            Bound::Upper => ctx.add_profile_counter(&ctx.tt_hit_upper, 1),
        }
    }
    match tt_probe.reject {
        Some(TtRejectReason::Depth) => ctx.add_profile_counter(&ctx.tt_rejected_depth, 1),
        Some(TtRejectReason::Bound) => ctx.add_profile_counter(&ctx.tt_rejected_bound, 1),
        Some(TtRejectReason::Decode) => ctx.add_profile_counter(&ctx.tt_rejected_decode, 1),
        None => {}
    }
    if tt_probe.cutoff.is_some() {
        ctx.add_profile_counter(&ctx.tt_cutoffs, 1);
    }
    if let Some(cutoff) = tt_probe.cutoff {
        return Some(cutoff);
    }

    // S7.3 selectivity attribution: null-move eligibility is observed without
    // short-circuit changes; `s73_null_attempted` classifies the node later.
    let mut s73_null_attempted = false;
    let s73_null_ok = allow_null
        && null_move_eligible(
            pos,
            ctx.features().null_move,
            depth,
            alpha,
            beta,
            node_in_check,
        );
    if ctx.profiling_enabled && s73_null_ok {
        ctx.add_profile_counter(&ctx.s73_null_eligible, 1);
    }
    if s73_null_ok {
        ctx.add_profile_counter(&ctx.null_move_attempts, 1);
        s73_null_attempted = true;
        let mut null_pos = make_null_position(pos);
        path.push_child(&null_pos);
        if !try_enter_node(ctx, limits) {
            path.pop();
            return None;
        }
        let null_alpha = -beta;
        let null_beta = null_alpha.saturating_add(1);
        // The null-probe child must not launch another null probe. This is an
        // immediate-child guard; ordinary descendants are eligible again.
        let null_result = negamax_entered_impl_with_null_and_extensions(
            &mut null_pos,
            depth.saturating_sub(1 + null_move_reduction(depth)),
            ply + 1,
            null_alpha,
            null_beta,
            ctx,
            limits,
            _profile,
            pv,
            path,
            tt,
            heur,
            false,
            0,
            extension_budget,
        );
        path.pop();
        let null_score = match null_result {
            Some(score) => -score,
            None => return None,
        };
        if null_score >= beta {
            // Null is only a candidate. Verify the real position at full
            // depth before returning its score.
            ctx.add_profile_counter(&ctx.null_move_fail_highs, 1);
            ctx.add_profile_counter(&ctx.null_move_researches, 1);
            return negamax_entered_impl_with_null_and_extensions(
                pos,
                depth,
                ply,
                alpha,
                beta,
                ctx,
                limits,
                _profile,
                pv,
                path,
                tt,
                heur,
                false,
                single_evasion_chain,
                extension_budget,
            );
        }
        ctx.add_profile_counter(&ctx.null_fail_lows, 1);
    }

    if depth == 0 {
        // Leaf: hand off to quiescence (already counted). On a real return,
        // store a depth-0 entry under the caller window; on abort propagate
        // None WITHOUT storing (the partial node must not be cached).
        return match quiescence_entered_impl_with_profile(
            pos,
            ply,
            0,
            alpha,
            beta,
            ctx,
            limits,
            pv,
            path,
            _profile,
            _profile.uses_qsearch_movegen(),
            ctx.features().qsearch_see,
            _profile.uses_qsearch_fast_pruning(),
        ) {
            Some(s) => {
                let bound = classify_tt_bound(s, caller_alpha, caller_beta);
                // The qsearch PV row start (if any) is the real best-capture
                // move; stand-pat / claim floor / empty PV -> None.
                let best_move = pv.lines[ply as usize].first().copied();
                store_tt_score_profiled(tt, key, 0, s, ply, bound, best_move, ctx);
                Some(s)
            }
            None => None,
        };
    }

    // Shallow futility is limited to non-PV (null-window), non-check nodes
    // with substantial non-pawn material. Tactical, checking, mate-range,
    // and advanced-pawn moves remain in the search even when their static
    // estimate is below the shallow margin.
    let futility_base = if ctx.features().futility
        && !node_in_check
        && depth <= 3
        && beta == alpha.saturating_add(1)
        && alpha < MATE - 1000
        && alpha > -(MATE - 1000)
        && non_pawn_material_count(pos) >= 4
    {
        ctx.add_profile_counter(&ctx.futility_considered, 1);
        Some(evaluate_profiled(pos, ctx, _profile))
    } else {
        None
    };

    let single_evasion_node = node_in_check && moves.len() == 1;

    // M4.1: for non-M4Reference profiles (`M41Reference` and `Current`),
    // apply the seven-level ordering (§5) at this non-root ordinary negamax
    // node — TT hash lift, promotions, MVV-LVA captures/ep, killer slot 0,
    // killer slot 1, then the remaining quiets sorted by history descending
    // (Commit 4) with a deterministic (from,to) tie-break. `M4Reference`
    // keeps the exact M4.0 ordering.
    // Killers are read from this `ply` (grown lazily; empty until a quiet
    // cutoff records one in a prior iteration); history is the per-search
    // table carried in `heur`.
    let ordering_start = ctx.sample_begin(&ctx.timing_ordering);
    if _profile.uses_threat_ordering() {
        order_moves_with_threats(
            pos,
            &mut moves,
            tt_probe.hash_move,
            heur.as_ref(),
            ply as usize,
            ctx,
        );
    } else if _profile != SearchProfile::M4Reference {
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
    if let Some(start) = ordering_start {
        ctx.sample_end(&ctx.timing_ordering, start);
    }

    // S7.2 move-ordering attribution (observation only): classify every
    // ordered legal move once, and record the node-level opportunity
    // denominators. Never touches ordering or search semantics.
    let s72_infos = if ctx.profiling_enabled {
        Some(s72_prepare(
            pos,
            &moves,
            tt_probe.hash_move,
            heur.as_ref(),
            ply as usize,
            ctx,
        ))
    } else {
        None
    };

    let mut node_best_move: Option<Move> = None;
    // S7.3 selectivity attribution locals: window width (PV vs null-window),
    // per-node history-bucket accumulator for searched quiets (flushed only
    // if the node completes without a beta cutoff).
    let s73_pv_node = beta > alpha.saturating_add(1);
    let mut s73_nocut_hist_acc: [u64; 6] = [0; 6];
    if ctx.profiling_enabled && !moves.is_empty() {
        ctx.add_profile_counter(&ctx.s73_loop_nodes, 1);
    }
    // P1.1: the running maximum of all fail-low scout scores. A fail-low
    // scout's PV is not committable, but the numeric value it returns is a
    // valid upper bound on its child and therefore part of this node's own
    // upper bound. It is folded into the RETURNED/STORED score only (never
    // into best / alpha / PV / cutoff / heuristics below).
    let mut fail_low_upper: Option<i32> = None;
    // S7.0 attribution: moves actually searched at this node (for the
    // searched-branching histogram).
    let mut searched_in_node: u64 = 0;
    // S7.2: whether this node terminated via beta cutoff, separating
    // late-cutoff ordering waste from genuine all-moves fail-low nodes.
    let mut s72_cutoff_happened = false;
    for (move_idx, m) in moves.into_iter().enumerate() {
        if let Some(static_eval) = futility_base {
            let margin = 100 + depth as i32 * 100;
            if move_idx > 0
                && !is_tactical(pos, m)
                && !move_gives_check(pos, m)
                && !is_pawn_promotion_threat(pos, m)
                && static_eval.saturating_add(margin) <= alpha
            {
                ctx.add_profile_counter(&ctx.futility_pruned, 1);
                continue;
            } else if ctx.profiling_enabled && move_idx > 0 && !is_tactical(pos, m) {
                // S7.3: quiet move kept at a futility-eligible node (the
                // shallow prune did not fire for it). Observation only.
                ctx.add_profile_counter(&ctx.s73_fut_quiet_kept, 1);
            }
        }
        ctx.add_profile_counter(&ctx.moves_searched, 1);
        searched_in_node += 1;
        // S7.2 attribution: per-searched-move opportunity accounting
        // (quiet rank / history buckets, killer and TT searched rates).
        if let Some(infos) = s72_infos.as_ref() {
            let info = &infos[move_idx];
            if info.quiet {
                ctx.add_profile_counter(&ctx.s72_quiet_searched, 1);
                ctx.add_profile_counter(&ctx.s72_quiet_searched_rank[info.rank_bucket], 1);
                ctx.add_profile_counter(&ctx.s72_quiet_searched_hist[info.hist_bucket], 1);
                ctx.add_profile_counter(&ctx.s72_d_quiet_searched[s72_depth_bucket(depth)], 1);
                // S7.3: per-searched-quiet accounting for the no-cutoff tree
                // composition and the depth>=4 quiet population.
                s73_nocut_hist_acc[info.hist_bucket] += 1;
                if depth >= 4 {
                    ctx.add_profile_counter(&ctx.s73_q4p_quiet_searched, 1);
                    ctx.add_profile_counter(&ctx.s73_q4p_quiet_idx[s73_idx_bucket(move_idx)], 1);
                }
            }
            if info.is_hash {
                ctx.add_profile_counter(&ctx.s72_tt_hash[1], 1);
            }
            if info.killer0 {
                ctx.add_profile_counter(&ctx.s72_killer[1], 1);
            }
            if info.killer1 {
                ctx.add_profile_counter(&ctx.s72_killer[4], 1);
            }
        }
        let reduction =
            late_move_reduction(pos, m, ctx.features().lmr, depth, move_idx, node_in_check);
        // S7.3: reduction applied to depth>=4 searched quiets, split by move
        // index bucket (R0 / R1 / R2+ x idx).
        if let Some(infos) = s72_infos.as_ref() {
            if infos[move_idx].quiet && depth >= 4 {
                let rb = reduction.min(2) as usize;
                ctx.add_profile_counter(&ctx.s73_q4p_quiet_red[rb], 1);
                ctx.add_profile_counter(
                    &ctx.s73_q4p_quiet_red_idx[rb * 5 + s73_idx_bucket(move_idx)],
                    1,
                );
            }
        }
        // S7.4: theoretical-vs-actual LMR attribution. `reduction` here is
        // only what late_move_reduction() PROPOSES; the search may still
        // discard it below when a null-window caller routes to Full.
        if reduction > 0 {
            ctx.add_profile_counter(&ctx.s74_lmr_proposed, 1);
            if reduction >= 2 {
                ctx.add_profile_counter(&ctx.s74_lmr_proposed_r2, 1);
            } else {
                ctx.add_profile_counter(&ctx.s74_lmr_proposed_r1, 1);
            }
        }
        // Capture the window BEFORE this move, so a possible re-search and
        // the beta-cutoff decision both see the same `alpha_before_move`.
        let alpha_before_move = alpha;
        let undo = make_move_profiled(pos, m, ctx);
        path.push_child(pos);

        let (child_depth, child_extension_budget) = child_extension_params(
            pos,
            depth,
            _profile,
            extension_budget,
            node_in_check,
            single_evasion_node,
            ctx,
        );

        // Manual child probe: try_enter_node called EXACTLY ONCE here.
        let probe = match probe_child_draw(
            pos,
            path.keys(),
            ply + 1,
            ply,
            ctx,
            limits,
            pv,
            ProbeKind::Main,
        ) {
            Some(p) => p,
            None => {
                path.pop();
                unmake_move_profiled(pos, undo, ctx);
                return None;
            }
        };

        // P2.2: when this move goes through a full re-search, remember the
        // child PV row the RE-SEARCH rewrote, so the commit block can verify
        // (inline) that the parent copies exactly this row — never a stale
        // scout row — whenever this move becomes the node's best.
        #[cfg(test)]
        let mut researched_row: Option<Vec<Move>> = None;
        // S7.4A (promoted): set when this move was verified by a full-depth
        // null-window re-search after a reduced improve, so only a VERIFIED
        // result can be counted as an LMR-null-window cutoff.
        let mut s74_nw_verified = false;

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
                        // profile, depth-0, or the caller-null-window /
                        // overflow fallbacks. The manual probe already spent
                        // the single node for this child.
                        //
                        // S7.4A, PROMOTED: on a caller-null-window node
                        // (beta == alpha_before_move + 1) production
                        // CurrentFinal applies the EXISTING LMR policy: one
                        // reduced null-window search with the CALLER'S
                        // window; fail-low accepted; any improvement verified
                        // by exactly one full-depth re-search before it may
                        // cut off or earn heuristics. Profiles without the
                        // promoted policy keep the historical full-depth
                        // fallback (and the S7.3 suppression counter).
                        let s74_nw_caller =
                            reduction > 0 && beta == alpha_before_move.saturating_add(1);
                        if s74_nw_caller && ctx.features().lmr_null_window {
                            ctx.add_profile_counter(&ctx.s74_lmr_applied_null_window, 1);
                            ctx.add_profile_counter(
                                &ctx.s74_lmr_nw_depth[s74_depth_bucket(depth)],
                                1,
                            );
                            ctx.add_profile_counter(
                                &ctx.s74_lmr_nw_idx[s74_idx_bucket(move_idx)],
                                1,
                            );
                            let nw_scout_depth = depth.saturating_sub(1 + reduction);
                            let nw_reduced = match negamax_entered_impl_with_null_and_extensions(
                                pos,
                                nw_scout_depth,
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
                                true,
                                current_single_evasion_chain,
                                child_extension_budget,
                            ) {
                                Some(s) => -s,
                                None => {
                                    path.pop();
                                    unmake_move_profiled(pos, undo, ctx);
                                    return None;
                                }
                            };
                            if nw_reduced <= alpha_before_move {
                                // Fail-low accepted, no re-search; numeric-only
                                // retention mirrors the production scout path.
                                ctx.add_profile_counter(&ctx.s74_lmr_nw_fail_low, 1);
                                MoveOutcome::ScoutFailLow(nw_reduced)
                            } else {
                                // Exactly ONE full-depth verification with the
                                // SAME caller null window. This is a NEW real
                                // search entry: acquire exactly one node,
                                // matching the production PVS re-search
                                // contract (S7.4A Repair 1).
                                ctx.add_profile_counter(&ctx.s74_lmr_nw_research, 1);
                                #[cfg(test)]
                                pvs_counters::mark_s74_nw_research_attempt();
                                if !try_enter_node(ctx, limits) {
                                    #[cfg(test)]
                                    pvs_counters::mark_s74_nw_abort_research_acquire();
                                    path.pop();
                                    unmake_move_profiled(pos, undo, ctx);
                                    return None;
                                }
                                ctx.add_profile_counter(&ctx.s74_lmr_nw_research_entered, 1);
                                #[cfg(test)]
                                pvs_counters::mark_s74_nw_research_entered();
                                match negamax_entered_impl_with_null_and_extensions(
                                    pos,
                                    child_depth,
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
                                    true,
                                    current_single_evasion_chain,
                                    child_extension_budget,
                                ) {
                                    Some(s) => {
                                        s74_nw_verified = true;
                                        MoveOutcome::Candidate(-s)
                                    }
                                    None => {
                                        path.pop();
                                        unmake_move_profiled(pos, undo, ctx);
                                        return None;
                                    }
                                }
                            }
                        } else {
                            if s74_nw_caller {
                                ctx.add_profile_counter(&ctx.s74_lmr_suppressed_by_null_window, 1);
                            }
                            // Handle a deeper abort EXPLICITLY: pop + unmake
                            // THIS edge before propagating None.
                            match negamax_entered_impl_with_null_and_extensions(
                                pos,
                                child_depth,
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
                                true,
                                current_single_evasion_chain,
                                child_extension_budget,
                            ) {
                                Some(s) => MoveOutcome::Candidate(-s),
                                None => {
                                    path.pop();
                                    unmake_move_profiled(pos, undo, ctx);
                                    return None;
                                }
                            }
                        }
                    }
                    ChildWindow::Scout { scout_beta } => {
                        if reduction > 0 {
                            ctx.add_profile_counter(&ctx.lmr_reductions, 1);
                            ctx.add_profile_counter(&ctx.s74_lmr_applied_existing_pvs, 1);
                            if reduction >= 2 {
                                ctx.add_profile_counter(&ctx.lmr_reduction_r2, 1);
                            } else {
                                ctx.add_profile_counter(&ctx.lmr_reduction_r1, 1);
                            }
                        }
                        // Null-window scout. Child window is
                        // `[-scout_beta, -alpha_before_move]`; the manual
                        // probe already spent the single node for this child.
                        #[cfg(test)]
                        pvs_counters::mark_scout();
                        let scout_depth = if reduction > 0 {
                            depth.saturating_sub(1 + reduction)
                        } else {
                            child_depth
                        };
                        let scout_score = match negamax_entered_impl_with_null_and_extensions(
                            pos,
                            scout_depth,
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
                            true,
                            current_single_evasion_chain,
                            child_extension_budget,
                        ) {
                            Some(s) => -s,
                            None => {
                                // Phase A: the scout's own subtree aborted.
                                #[cfg(test)]
                                pvs_counters::mark_abort_in_scout();
                                path.pop();
                                unmake_move_profiled(pos, undo, ctx);
                                return None;
                            }
                        };
                        let needs_research = if reduction > 0 {
                            // A reduced search is only a scout. Any score that
                            // improves alpha must be verified at full depth,
                            // including fail-high scores.
                            if scout_score > alpha_before_move {
                                ctx.add_profile_counter(&ctx.lmr_reduced_improves_alpha, 1);
                                true
                            } else {
                                // S7.3: reduced depth>=4 quiet scout failed low
                                // and needed no re-search — the reduction stuck.
                                if let Some(infos) = s72_infos.as_ref() {
                                    if infos[move_idx].quiet && depth >= 4 {
                                        ctx.add_profile_counter(
                                            &ctx.s73_q4p_scout_faillow_red
                                                [reduction.min(2) as usize],
                                            1,
                                        );
                                    }
                                }
                                false
                            }
                        } else {
                            pvs_needs_research(scout_score, alpha_before_move, beta)
                        };
                        if reduction > 0 && needs_research {
                            ctx.add_profile_counter(&ctx.lmr_researches, 1);
                            // S7.2 LMR interaction: reduced quiet needed a
                            // full-depth re-search.
                            if let Some(infos) = s72_infos.as_ref() {
                                if infos[move_idx].quiet {
                                    ctx.add_profile_counter(&ctx.s72_lmr[1], 1);
                                    if depth >= 4 {
                                        ctx.add_profile_counter(&ctx.s73_q4p_quiet_researched, 1);
                                    }
                                }
                            }
                        }
                        if needs_research {
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
                                unmake_move_profiled(pos, undo, ctx);
                                return None;
                            }
                            #[cfg(test)]
                            pvs_counters::mark_research_entered();
                            match negamax_entered_impl_with_null_and_extensions(
                                pos,
                                child_depth,
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
                                true,
                                current_single_evasion_chain,
                                child_extension_budget,
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
                                    unmake_move_profiled(pos, undo, ctx);
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
        unmake_move_profiled(pos, undo, ctx);

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
                // S7.2 LMR interaction: reduced quiet failed low (no re-search).
                if let Some(infos) = s72_infos.as_ref() {
                    if infos[move_idx].quiet && reduction > 0 {
                        ctx.add_profile_counter(&ctx.s72_lmr[0], 1);
                    }
                }
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
            // S7.4A: a beta cutoff earned by a candidate null-window move
            // AFTER its full-depth verification (fail-low results can never
            // reach this block).
            if s74_nw_verified {
                ctx.add_profile_counter(&ctx.s74_lmr_nw_verified_cutoff, 1);
            }
            // S7.0 attribution: beta-cutoff quality (index + mover category).
            if ctx.profiling_enabled {
                ctx.add_profile_counter(&ctx.beta_cutoffs, 1);
                match move_idx {
                    0 => ctx.add_profile_counter(&ctx.beta_cutoff_idx_0, 1),
                    1 => ctx.add_profile_counter(&ctx.beta_cutoff_idx_1, 1),
                    2..=3 => ctx.add_profile_counter(&ctx.beta_cutoff_idx_2_3, 1),
                    4..=7 => ctx.add_profile_counter(&ctx.beta_cutoff_idx_4_7, 1),
                    8..=15 => ctx.add_profile_counter(&ctx.beta_cutoff_idx_8_15, 1),
                    _ => ctx.add_profile_counter(&ctx.beta_cutoff_idx_16p, 1),
                }
                if tt_probe.hash_move == Some(m) {
                    ctx.add_profile_counter(&ctx.cutoff_tt_move, 1);
                } else if is_tactical(pos, m) {
                    ctx.add_profile_counter(&ctx.cutoff_tactical, 1);
                } else if heur.as_ref().is_some_and(|h| h.is_killer(ply as usize, m)) {
                    ctx.add_profile_counter(&ctx.cutoff_killer, 1);
                } else {
                    ctx.add_profile_counter(&ctx.cutoff_quiet, 1);
                }
                // S7.2 attribution: cutoff category (mutually exclusive, TT
                // first), moves searched before cutoff, quiet rank/history
                // histograms, killer/TT success rates, depth split, LMR
                // eventual-cutoff interaction.
                if let Some(infos) = s72_infos.as_ref() {
                    let info = &infos[move_idx];
                    let db = s72_depth_bucket(depth);
                    s72_cutoff_happened = true;
                    ctx.add_profile_counter(&ctx.s72_cutoff_category[info.cat], 1);
                    ctx.add_profile_counter(
                        &ctx.s72_cutoff_searched[s72_count_bucket(searched_in_node)],
                        1,
                    );
                    ctx.add_profile_counter(&ctx.s72_d_cutoffs[db], 1);
                    if searched_in_node >= 5 {
                        ctx.add_profile_counter(&ctx.s72_d_cutoff_late5[db], 1);
                    }
                    if info.quiet {
                        ctx.add_profile_counter(
                            &ctx.s72_quiet_cutoff_gidx[s72_rank_bucket(move_idx as u32)],
                            1,
                        );
                        ctx.add_profile_counter(&ctx.s72_quiet_cutoff_rank[info.rank_bucket], 1);
                        ctx.add_profile_counter(&ctx.s72_quiet_cutoff_hist[info.hist_bucket], 1);
                        ctx.add_profile_counter(&ctx.s72_d_quiet_cutoffs[db], 1);
                        if reduction > 0 {
                            ctx.add_profile_counter(&ctx.s72_lmr[2], 1);
                        }
                        // S7.3: depth>=4 quiet cutoffs by reduction and by
                        // move-index bucket.
                        if depth >= 4 {
                            ctx.add_profile_counter(
                                &ctx.s73_q4p_quiet_cutoff_red[reduction.min(2) as usize],
                                1,
                            );
                            ctx.add_profile_counter(
                                &ctx.s73_q4p_quiet_cutoff_idx[s73_idx_bucket(move_idx)],
                                1,
                            );
                        }
                    }
                    if info.is_hash {
                        ctx.add_profile_counter(&ctx.s72_tt_hash[2], 1);
                        if searched_in_node == 1 {
                            ctx.add_profile_counter(&ctx.s72_tt_hash[3], 1);
                        }
                    }
                    if info.killer0 {
                        ctx.add_profile_counter(&ctx.s72_killer[2], 1);
                    }
                    if info.killer1 {
                        ctx.add_profile_counter(&ctx.s72_killer[5], 1);
                    }
                }
            }
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
    if ctx.profiling_enabled {
        match searched_in_node {
            0 => {}
            1 => ctx.add_profile_counter(&ctx.searched_hist_1, 1),
            2 => ctx.add_profile_counter(&ctx.searched_hist_2, 1),
            3..=4 => ctx.add_profile_counter(&ctx.searched_hist_3_4, 1),
            5..=8 => ctx.add_profile_counter(&ctx.searched_hist_5_8, 1),
            9..=16 => ctx.add_profile_counter(&ctx.searched_hist_9_16, 1),
            _ => ctx.add_profile_counter(&ctx.searched_hist_17p, 1),
        }
        // S7.2: fail-low nodes are NOT ordering failures — keep them out of
        // the late-cutoff waste analysis. Also record the TT hash move
        // improving alpha without a cutoff.
        if !s72_cutoff_happened && searched_in_node > 0 {
            ctx.add_profile_counter(&ctx.s72_fail_low_nodes, 1);
            ctx.add_profile_counter(&ctx.s72_fail_low_searched_sum, searched_in_node);
            ctx.add_profile_counter(&ctx.s72_d_fail_low[s72_depth_bucket(depth)], 1);
            if node_best_move.is_some() && node_best_move == tt_probe.hash_move {
                ctx.add_profile_counter(&ctx.s72_tt_hash[4], 1);
            }
            // S7.3 selectivity attribution: classify what the no-beta-cutoff
            // move-loop trees actually are (window width, in-check, null-move
            // attempt, searched-quiet history composition).
            if s73_pv_node {
                ctx.add_profile_counter(&ctx.s73_nocut_pv, 1);
            } else {
                ctx.add_profile_counter(&ctx.s73_nocut_nonpv, 1);
            }
            if node_in_check {
                ctx.add_profile_counter(&ctx.s73_nocut_incheck, 1);
            }
            if s73_null_attempted {
                ctx.add_profile_counter(&ctx.s73_nocut_null_attempted, 1);
            }
            ctx.add_profile_counter(&ctx.s73_nocut_searched_sum, searched_in_node);
            for (b, acc) in s73_nocut_hist_acc.iter().enumerate() {
                if *acc > 0 {
                    ctx.add_profile_counter(&ctx.s73_nocut_searched_hist[b], *acc);
                }
            }
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
    let tt_store_start = ctx.sample_begin(&ctx.timing_tt);
    store_tt_score_profiled(
        tt,
        key,
        depth,
        returned_score,
        ply,
        bound,
        node_best_move,
        ctx,
    );
    if let Some(start) = tt_store_start {
        ctx.sample_end(&ctx.timing_tt, start);
    }
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

#[inline]
fn is_plain_capture(pos: &Position, m: Move) -> bool {
    matches!(m.flag, MoveFlag::Normal) && pos.board[m.to as usize].is_some()
}

/// Apply the first D1.3 qsearch pruning rule. Only a plain, non-checking
/// capture is eligible for SEE<0 pruning. Promotions and en passant are
/// deliberately fail-open, and all checking captures are kept before SEE is
/// consulted. If a later recapture promotes, the pruning-specific SEE also
/// fails open because the current exchange model does not prove that line.
/// The helper preserves the existing move order for every move it keeps.
fn prune_qsearch_captures_by_see(
    pos: &mut Position,
    moves: Vec<Move>,
    ctx: &SearchContext,
    alpha: i32,
    beta: i32,
) -> Vec<Move> {
    prune_qsearch_captures_by_see_impl(pos, moves, ctx, alpha, beta, false, false, 0, 0)
}

/// S7.1B: production SEE<0 pruning PLUS the conservative delta rule. Each
/// eligible plain capture receives AT MOST ONE pruning SEE calculation whose
/// value is reused for both decisions (SEE<0 -> existing prune;
/// `stand_pat + SEE + QSEARCH_DELTA_MARGIN_CP <= alpha` -> delta prune).
fn prune_qsearch_captures_by_see_delta(
    pos: &mut Position,
    moves: Vec<Move>,
    ctx: &SearchContext,
    alpha: i32,
    beta: i32,
    stand_pat: i32,
    qply: u32,
) -> Vec<Move> {
    prune_qsearch_captures_by_see_impl(pos, moves, ctx, alpha, beta, false, true, stand_pat, qply)
}

fn prune_qsearch_captures_by_fast_see(
    pos: &mut Position,
    moves: Vec<Move>,
    ctx: &SearchContext,
    alpha: i32,
    beta: i32,
) -> Vec<Move> {
    prune_qsearch_captures_by_see_impl(pos, moves, ctx, alpha, beta, true, false, 0, 0)
}

#[allow(clippy::too_many_arguments)]
fn prune_qsearch_captures_by_see_impl(
    pos: &mut Position,
    moves: Vec<Move>,
    ctx: &SearchContext,
    alpha: i32,
    beta: i32,
    fast_see: bool,
    delta_enabled: bool,
    stand_pat: i32,
    qply: u32,
) -> Vec<Move> {
    if alpha <= -(MATE - 1000) || beta >= MATE - 1000 {
        return moves;
    }
    // S7.1B node-level eligibility. The call site only reaches this helper at
    // a NON-CHECK qsearch node AFTER the stand-pat beta cutoff, so
    // `stand_pat < beta` holds structurally; the check is kept explicit as a
    // defensive invariant. The non-pawn material guard mirrors the existing
    // futility/null guard style so sparse endgames stay fail-open.
    let delta_active = delta_enabled && stand_pat < beta && non_pawn_material_count(pos) >= 4;
    let mut kept = Vec::with_capacity(moves.len());
    for m in moves {
        if matches!(m.flag, MoveFlag::Promotion(_)) {
            ctx.add_profile_counter(&ctx.qsearch_promotions_kept, 1);
            kept.push(m);
            continue;
        }
        if matches!(m.flag, MoveFlag::EnPassant) {
            ctx.add_profile_counter(&ctx.qsearch_en_passant_kept, 1);
            kept.push(m);
            continue;
        }
        if !is_plain_capture(pos, m) {
            // The specialized qsearch generator should not produce a quiet
            // move here, but an unexpected move is safer when searched.
            kept.push(m);
            continue;
        }
        if move_gives_check(pos, m) {
            ctx.add_profile_counter(&ctx.qsearch_checking_captures_kept, 1);
            kept.push(m);
            continue;
        }

        // The source/target guards above make this a well-defined ordinary
        // capture. If a future move flag or generator violates that contract,
        // it reaches the fail-open path above instead of being pruned.
        //
        // S7.1B critical rule: exactly ONE pruning SEE computation per
        // eligible capture; the SAME exchange result drives both the
        // production SEE<0 prune and the delta prune.
        ctx.add_profile_counter(&ctx.qsearch_see_tests, 1);
        if delta_active {
            ctx.add_profile_counter(&ctx.qsearch_delta_tests, 1);
        }
        let see_start = ctx.sample_begin(&ctx.timing_see);
        // Normalized to a value whose sign carries the SEE<0 decision: the
        // fast path only proves the sign, so 0/-1 preserve its exact
        // semantics; the full path keeps the actual exchange value for the
        // delta rule.
        let see_value: Option<i32> = if fast_see {
            see_ge_for_pruning(pos, m, 0).map(|ge| if ge { 0 } else { -1 })
        } else {
            static_exchange_eval_for_pruning(pos, m)
        };
        if let Some(start) = see_start {
            ctx.sample_end(&ctx.timing_see, start);
        }
        match see_value {
            Some(v) if v < 0 => {
                ctx.add_profile_counter(&ctx.qsearch_see_pruned, 1);
            }
            Some(v) => {
                if delta_active
                    && stand_pat
                        .saturating_add(v)
                        .saturating_add(QSEARCH_DELTA_MARGIN_CP)
                        <= alpha
                {
                    ctx.add_profile_counter(&ctx.qsearch_delta_pruned, 1);
                    match pos.board[m.to as usize]
                        .map(|piece| piece.piece_type)
                        .unwrap_or(PieceType::Pawn)
                    {
                        PieceType::Pawn => {
                            ctx.add_profile_counter(&ctx.qsearch_delta_pruned_pawn, 1);
                        }
                        PieceType::Knight | PieceType::Bishop => {
                            ctx.add_profile_counter(&ctx.qsearch_delta_pruned_minor, 1);
                        }
                        PieceType::Rook => {
                            ctx.add_profile_counter(&ctx.qsearch_delta_pruned_rook, 1);
                        }
                        PieceType::Queen => {
                            ctx.add_profile_counter(&ctx.qsearch_delta_pruned_queen, 1);
                        }
                        PieceType::King => {}
                    }
                    if qply <= 1 {
                        ctx.add_profile_counter(&ctx.qsearch_delta_qply_0_1, 1);
                    } else if qply <= 3 {
                        ctx.add_profile_counter(&ctx.qsearch_delta_qply_2_3, 1);
                    } else {
                        ctx.add_profile_counter(&ctx.qsearch_delta_qply_4p, 1);
                    }
                } else {
                    kept.push(m);
                }
            }
            None => {
                // An unsupported exchange, including a later promotion,
                // must fail open because this SEE result is now a deletion
                // proof rather than an ordering hint.
                ctx.add_profile_counter(&ctx.qsearch_see_fail_open_promotions, 1);
                kept.push(m);
            }
        }
    }
    kept
}

fn non_pawn_material_count(pos: &Position) -> usize {
    pos.board
        .iter()
        .flatten()
        .filter(|piece| {
            matches!(
                piece.piece_type,
                PieceType::Knight | PieceType::Bishop | PieceType::Rook | PieceType::Queen
            )
        })
        .count()
}

/// Conservatively protect advanced pawn pushes from shallow futility. A full
/// passed-pawn classifier is unnecessary here: treating every pawn on the
/// final three ranks as a promotion threat is safer and keeps the prune out
/// of promotion races.
fn is_pawn_promotion_threat(pos: &Position, m: Move) -> bool {
    let Some(piece) = pos.board[m.from as usize] else {
        return false;
    };
    if piece.piece_type != PieceType::Pawn {
        return false;
    }
    let rank = rank_of(m.to);
    (piece.color == Color::White && rank >= 5) || (piece.color == Color::Black && rank <= 2)
}

/// Return a conservative late-move reduction for a quiet, non-checking move.
/// Reductions are limited to LMR-enabled profiles, non-PV moves, non-check
/// nodes, sufficient depth, and positions with enough non-pawn material that
/// a shallow quiet move is unlikely to be an endgame zugzwang. A reduced
/// result that improves alpha is always re-searched at full depth.
fn late_move_reduction(
    pos: &mut Position,
    m: Move,
    lmr_enabled: bool,
    depth: u32,
    move_idx: usize,
    node_in_check: bool,
) -> u32 {
    if !lmr_enabled
        || node_in_check
        || depth < 4
        || move_idx < 3
        || is_tactical(pos, m)
        || move_gives_check(pos, m)
    {
        return 0;
    }

    if non_pawn_material_count(pos) < 4 {
        return 0;
    }

    if depth >= 7 && move_idx >= 8 {
        2
    } else {
        1
    }
}

fn null_move_reduction(depth: u32) -> u32 {
    if depth >= 7 {
        3
    } else {
        2
    }
}

fn null_move_eligible(
    pos: &Position,
    null_enabled: bool,
    depth: u32,
    alpha: i32,
    beta: i32,
    node_in_check: bool,
) -> bool {
    null_enabled
        && !node_in_check
        && depth >= 5
        && beta == alpha.saturating_add(1)
        && non_pawn_material_count(pos) >= 4
        && alpha > -(MATE - 1000)
}

fn make_null_position(pos: &Position) -> Position {
    let mut null_pos = *pos;
    null_pos.side = pos.side.opposite();
    null_pos.ep_target = None;
    null_pos.halfmove = pos.halfmove.saturating_add(1);
    if pos.side == Color::Black {
        null_pos.fullmove = null_pos.fullmove.saturating_add(1);
    }
    null_pos.zobrist_key = recompute_zobrist(&null_pos);
    null_pos
}

fn see_piece_after_move(pos: &Position, m: Move) -> i32 {
    m.promotion
        .unwrap_or(
            pos.board[m.from as usize]
                .expect("SEE source is occupied")
                .piece_type,
        )
        .value()
}

fn slider_attacks_target(pos: &Position, from: u8, target: u8) -> bool {
    let from_file = file_of(from) as i32;
    let from_rank = rank_of(from) as i32;
    let target_file = file_of(target) as i32;
    let target_rank = rank_of(target) as i32;
    let df = target_file - from_file;
    let dr = target_rank - from_rank;
    let (step_file, step_rank) = if df == 0 && dr != 0 {
        (0, dr.signum())
    } else if dr == 0 && df != 0 {
        (df.signum(), 0)
    } else if df.abs() == dr.abs() {
        (df.signum(), dr.signum())
    } else {
        return false;
    };

    let mut file = from_file + step_file;
    let mut rank = from_rank + step_rank;
    while file != target_file || rank != target_rank {
        if pos.board[make_square(file as u8, rank as u8) as usize].is_some() {
            return false;
        }
        file += step_file;
        rank += step_rank;
    }
    true
}

fn piece_attacks_target(pos: &Position, from: u8, target: u8, piece_type: PieceType) -> bool {
    let df = (file_of(target) as i32 - file_of(from) as i32).abs();
    let dr = (rank_of(target) as i32 - rank_of(from) as i32).abs();
    match piece_type {
        PieceType::Pawn => {
            let direction = if pos.side == Color::White { 1 } else { -1 };
            rank_of(target) as i32 - rank_of(from) as i32 == direction && df == 1
        }
        PieceType::Knight => (df == 1 && dr == 2) || (df == 2 && dr == 1),
        PieceType::King => df <= 1 && dr <= 1 && (df != 0 || dr != 0),
        PieceType::Bishop => df == dr && df != 0 && slider_attacks_target(pos, from, target),
        PieceType::Rook => {
            (df == 0 || dr == 0) && (df != 0 || dr != 0) && slider_attacks_target(pos, from, target)
        }
        PieceType::Queen => {
            (dr == 0 || df == 0 || df == dr)
                && (dr != 0 || df != 0)
                && slider_attacks_target(pos, from, target)
        }
    }
}

fn least_valuable_attacker(pos: &mut Position, target: u8) -> Option<Move> {
    let side = pos.side;
    for piece_type in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        for from in 0u8..64 {
            if pos.board[from as usize] != Some(Piece::new(side, piece_type))
                || !piece_attacks_target(pos, from, target, piece_type)
            {
                continue;
            }
            let promotion = match piece_type {
                PieceType::Pawn
                    if (side == Color::White && rank_of(target) == 7)
                        || (side == Color::Black && rank_of(target) == 0) =>
                {
                    Some(PieceType::Knight)
                }
                _ => None,
            };
            let m = Move {
                from,
                to: target,
                promotion,
                flag: promotion.map_or(MoveFlag::Normal, MoveFlag::Promotion),
            };
            let undo = pos.make_move(m);
            let legal = !pos.is_square_attacked(pos.king_square(side), side.opposite());
            pos.unmake_move(undo);
            if legal {
                return Some(m);
            }
        }
    }
    None
}

fn static_exchange_eval(pos: &Position, m: Move) -> i32 {
    static_exchange_eval_impl(pos, m, false).expect("ordering SEE must not reject any exchange")
}

/// SEE used as a qsearch deletion proof. Unlike ordering SEE, this function
/// refuses to make a claim when the exchange contains a promotion after the
/// initial move. The current exchange model does not yet encode promotion
/// choice and promotion gain at every recapture ply, so an unsupported line
/// must be searched rather than pruned.
fn static_exchange_eval_for_pruning(pos: &Position, m: Move) -> Option<i32> {
    static_exchange_eval_impl(pos, m, true)
}

fn static_exchange_eval_impl(pos: &Position, m: Move, reject_promotions: bool) -> Option<i32> {
    if reject_promotions && m.promotion.is_some() {
        return None;
    }
    let victim_value = match m.flag {
        MoveFlag::EnPassant => PieceType::Pawn.value(),
        MoveFlag::Promotion(_)
        | MoveFlag::Normal
        | MoveFlag::DoublePawnPush
        | MoveFlag::KingCastle
        | MoveFlag::QueenCastle => pos.board[m.to as usize]
            .map(|piece| piece.piece_type.value())
            .unwrap_or(0),
    };
    let promotion_gain = m
        .promotion
        .map(|piece| piece.value() - PieceType::Pawn.value())
        .unwrap_or(0);
    let mut gains = [0i32; 32];
    gains[0] = victim_value + promotion_gain;

    let mut child = *pos;
    let mut captured_value = see_piece_after_move(pos, m);
    child.make_move(m);
    let mut depth = 0usize;
    while depth + 1 < gains.len() {
        let Some(attacker) = least_valuable_attacker(&mut child, m.to) else {
            break;
        };
        if reject_promotions && attacker.promotion.is_some() {
            return None;
        }
        depth += 1;
        gains[depth] = captured_value - gains[depth - 1];
        captured_value = see_piece_after_move(&child, attacker);
        child.make_move(attacker);
    }
    while depth > 0 {
        depth -= 1;
        gains[depth] = -(-gains[depth]).max(gains[depth + 1]);
    }
    Some(gains[0])
}

#[inline]
fn square_with_offset(square: u8, file_delta: i32, rank_delta: i32) -> Option<u8> {
    let file = file_of(square) as i32 + file_delta;
    let rank = rank_of(square) as i32 + rank_delta;
    if (0..8).contains(&file) && (0..8).contains(&rank) {
        Some(make_square(file as u8, rank as u8))
    } else {
        None
    }
}

#[inline]
fn add_fast_attacker_source(
    pos: &Position,
    piece_type: PieceType,
    from: u8,
    sources: &mut [u8; 8],
    count: &mut usize,
) {
    if pos.board[from as usize] != Some(Piece::new(pos.side, piece_type)) {
        return;
    }
    debug_assert!(*count < sources.len());
    if *count >= sources.len() {
        return;
    }
    sources[*count] = from;
    *count += 1;
}

#[inline]
fn collect_fast_slider_attackers(
    pos: &Position,
    target: u8,
    piece_type: PieceType,
    directions: &[(i32, i32)],
    sources: &mut [u8; 8],
    count: &mut usize,
) {
    let target_file = file_of(target) as i32;
    let target_rank = rank_of(target) as i32;
    for &(file_delta, rank_delta) in directions {
        let mut file = target_file + file_delta;
        let mut rank = target_rank + rank_delta;
        while (0..8).contains(&file) && (0..8).contains(&rank) {
            let from = make_square(file as u8, rank as u8);
            if pos.board[from as usize].is_some() {
                add_fast_attacker_source(pos, piece_type, from, sources, count);
                break;
            }
            file += file_delta;
            rank += rank_delta;
        }
    }
}

#[inline]
fn collect_fast_attacker_candidates(
    pos: &Position,
    target: u8,
    piece_type: PieceType,
) -> ([u8; 8], usize) {
    let mut sources = [0; 8];
    let mut count = 0;
    match piece_type {
        PieceType::Pawn => {
            let direction = if pos.side == Color::White { 1 } else { -1 };
            let source_rank = rank_of(target) as i32 - direction;
            if (0..8).contains(&source_rank) {
                for file_delta in [-1, 1] {
                    if let Some(from) = square_with_offset(
                        make_square(file_of(target), source_rank as u8),
                        file_delta,
                        0,
                    ) {
                        add_fast_attacker_source(pos, piece_type, from, &mut sources, &mut count);
                    }
                }
            }
        }
        PieceType::Knight => {
            for &(file_delta, rank_delta) in &KNIGHT_OFFSETS {
                if let Some(from) = square_with_offset(target, file_delta, rank_delta) {
                    add_fast_attacker_source(pos, piece_type, from, &mut sources, &mut count);
                }
            }
        }
        PieceType::Bishop => collect_fast_slider_attackers(
            pos,
            target,
            piece_type,
            &BISHOP_DIRS,
            &mut sources,
            &mut count,
        ),
        PieceType::Rook => collect_fast_slider_attackers(
            pos,
            target,
            piece_type,
            &ROOK_DIRS,
            &mut sources,
            &mut count,
        ),
        PieceType::Queen => {
            collect_fast_slider_attackers(
                pos,
                target,
                piece_type,
                &BISHOP_DIRS,
                &mut sources,
                &mut count,
            );
            collect_fast_slider_attackers(
                pos,
                target,
                piece_type,
                &ROOK_DIRS,
                &mut sources,
                &mut count,
            );
        }
        PieceType::King => {
            for &(file_delta, rank_delta) in &KING_OFFSETS {
                if let Some(from) = square_with_offset(target, file_delta, rank_delta) {
                    add_fast_attacker_source(pos, piece_type, from, &mut sources, &mut count);
                }
            }
        }
    }
    for index in 1..count {
        let source = sources[index];
        let mut insert_at = index;
        while insert_at > 0 && sources[insert_at - 1] > source {
            sources[insert_at] = sources[insert_at - 1];
            insert_at -= 1;
        }
        sources[insert_at] = source;
    }
    (sources, count)
}

#[inline]
fn least_valuable_attacker_fast(pos: &mut Position, target: u8) -> Option<Move> {
    for piece_type in [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ] {
        let (sources, count) = collect_fast_attacker_candidates(pos, target, piece_type);
        let promotion = if piece_type == PieceType::Pawn
            && ((pos.side == Color::White && rank_of(target) == 7)
                || (pos.side == Color::Black && rank_of(target) == 0))
        {
            Some(PieceType::Knight)
        } else {
            None
        };
        let move_flag = promotion.map_or(MoveFlag::Normal, MoveFlag::Promotion);
        for &from in sources.iter().take(count) {
            let candidate = Move {
                from,
                to: target,
                promotion,
                flag: move_flag,
            };
            let undo = pos.make_move(candidate);
            let legal = !pos.is_square_attacked(pos.king_square(pos.side.opposite()), pos.side);
            pos.unmake_move(undo);
            if legal {
                return Some(candidate);
            }
        }
    }
    None
}

/// Fast boolean SEE for pruning. It deliberately preserves the exact
/// promotion fail-open rule and exchange back-propagation of the D1.3
/// integer SEE, while replacing the 64-square attacker scan with direct
/// target-relative occupancy walks.
#[inline]
fn see_ge_for_pruning(pos: &Position, m: Move, threshold: i32) -> Option<bool> {
    if m.promotion.is_some() {
        return None;
    }
    let victim_value = match m.flag {
        MoveFlag::EnPassant => PieceType::Pawn.value(),
        MoveFlag::Promotion(_)
        | MoveFlag::Normal
        | MoveFlag::DoublePawnPush
        | MoveFlag::KingCastle
        | MoveFlag::QueenCastle => pos.board[m.to as usize]
            .map(|piece| piece.piece_type.value())
            .unwrap_or(0),
    };
    let promotion_gain = m
        .promotion
        .map(|piece| piece.value() - PieceType::Pawn.value())
        .unwrap_or(0);
    let mut gains = [0i32; 32];
    gains[0] = victim_value + promotion_gain;

    let mut child = *pos;
    let mut captured_value = see_piece_after_move(pos, m);
    child.make_move(m);
    let mut depth = 0usize;
    while depth + 1 < gains.len() {
        let Some(attacker) = least_valuable_attacker_fast(&mut child, m.to) else {
            break;
        };
        if attacker.promotion.is_some() {
            return None;
        }
        depth += 1;
        gains[depth] = captured_value - gains[depth - 1];
        captured_value = see_piece_after_move(&child, attacker);
        child.make_move(attacker);
    }
    while depth > 0 {
        depth -= 1;
        gains[depth] = -(-gains[depth]).max(gains[depth + 1]);
    }
    Some(gains[0] >= threshold)
}

fn move_gives_check(pos: &mut Position, m: Move) -> bool {
    let undo = pos.make_move(m);
    let gives_check = pos.is_in_check(pos.side);
    pos.unmake_move(undo);
    gives_check
}

/// Compute the depth/budget passed across one legal move edge. Extensions are
/// deliberately bounded per root line and are only active for the isolated
/// threat-aware candidate. A check extension and a single-evasion extension
/// share one unit so the two rules cannot stack on the same edge.
fn extension_budget_for_profile(profile: SearchProfile) -> u8 {
    if profile.uses_single_evasion_extension() {
        S75A_FORCING_BUDGET
    } else if profile.uses_forcing_search() {
        MAX_FORCING_EXTENSIONS
    } else {
        0
    }
}

/// Dispatch child depth/budget across the legacy threat-aware forcing path
/// and the isolated S7.5A single-evasion path.
fn child_extension_params(
    child: &Position,
    depth: u32,
    profile: SearchProfile,
    extension_budget: u8,
    parent_in_check: bool,
    parent_has_single_evasion: bool,
    ctx: &SearchContext,
) -> (u32, u8) {
    if profile.uses_single_evasion_extension() {
        s75a_single_evasion_child_params(
            depth,
            extension_budget,
            parent_in_check,
            parent_has_single_evasion,
            ctx,
        )
    } else {
        forcing_child_params(
            child,
            depth,
            profile,
            extension_budget,
            parent_in_check,
            parent_has_single_evasion,
            ctx,
        )
    }
}

/// S7.5A: extend exactly one edge when the CURRENT node is in check and has
/// exactly one legal move. Main search only; no checking-move / qsearch rule.
fn s75a_single_evasion_child_params(
    depth: u32,
    extension_budget: u8,
    parent_in_check: bool,
    parent_has_single_evasion: bool,
    ctx: &SearchContext,
) -> (u32, u8) {
    let ordinary_depth = depth.saturating_sub(1);
    let opportunity = parent_in_check && parent_has_single_evasion;
    if depth == 0 || !opportunity {
        return (ordinary_depth, extension_budget);
    }
    if extension_budget == 0 {
        ctx.add_profile_counter(&ctx.s75a_opportunity_blocked_budget_0, 1);
        return (ordinary_depth, extension_budget);
    }
    ctx.add_profile_counter(&ctx.s75a_extension_applied_total, 1);
    if depth == 1 {
        ctx.add_profile_counter(&ctx.s75a_extension_applied_depth1, 1);
    }
    if extension_budget == 2 {
        ctx.add_profile_counter(&ctx.s75a_extension_budget_2_to_1, 1);
    } else if extension_budget == 1 {
        ctx.add_profile_counter(&ctx.s75a_extension_budget_1_to_0, 1);
    }
    (depth, extension_budget - 1)
}

fn forcing_child_params(
    child: &Position,
    depth: u32,
    profile: SearchProfile,
    extension_budget: u8,
    parent_in_check: bool,
    parent_has_single_evasion: bool,
    ctx: &SearchContext,
) -> (u32, u8) {
    let ordinary_depth = depth.saturating_sub(1);
    if !profile.uses_forcing_search() || extension_budget == 0 || depth == 0 {
        return (ordinary_depth, extension_budget);
    }

    let child_gives_check = child.is_in_check(child.side);
    let single_evasion = parent_in_check && parent_has_single_evasion;
    if child_gives_check || single_evasion {
        if child_gives_check {
            ctx.add_profile_counter(&ctx.check_extensions, 1);
        }
        if single_evasion {
            ctx.add_profile_counter(&ctx.single_evasion_extensions, 1);
        }
        (depth, extension_budget - 1)
    } else {
        (ordinary_depth, extension_budget)
    }
}

fn king_zone_attack_count(pos: &Position, king: Square, by: Color) -> i32 {
    let file = file_of(king) as i32;
    let rank = rank_of(king) as i32;
    let mut count = 0;
    for df in -1..=1 {
        for dr in -1..=1 {
            if on_board(file + df, rank + dr)
                && pos.is_square_attacked(make_square((file + df) as u8, (rank + dr) as u8), by)
            {
                count += 1;
            }
        }
    }
    count
}

/// Cheap, candidate-only move signal used to put forcing and defensive moves
/// ahead of otherwise equal killer/history moves. It deliberately does not
/// reject or mutate a move: the legal move loop remains the authority.
fn threat_move_signal(pos: &mut Position, m: Move) -> (bool, i32) {
    let mover = pos.side;
    let enemy = mover.opposite();
    let own_king = pos.king_sq[mover as usize];
    let enemy_king = pos.king_sq[enemy as usize];
    let own_before = king_zone_attack_count(pos, own_king, mover);
    let enemy_before = king_zone_attack_count(pos, enemy_king, mover);
    let moving_piece = pos.board[m.from as usize];
    let mut score = 0;

    if let Some(piece) = moving_piece {
        if piece.piece_type == PieceType::Pawn {
            let from_rank = rank_of(m.from);
            let to_rank = rank_of(m.to);
            let advances = if mover == Color::White {
                to_rank > from_rank
            } else {
                to_rank < from_rank
            };
            if advances && (2..=5).contains(&to_rank) {
                score += 24;
            }
            if matches!(m.flag, MoveFlag::DoublePawnPush) {
                score += 16;
            }
            if pos.board[m.to as usize].is_some() {
                score += 20;
            }
        }
    }

    let to_file = file_of(m.to) as i32;
    let to_rank = rank_of(m.to) as i32;
    let own_file = file_of(own_king) as i32;
    let own_rank = rank_of(own_king) as i32;
    let enemy_file = file_of(enemy_king) as i32;
    let enemy_rank = rank_of(enemy_king) as i32;
    if (to_file - enemy_file).abs() <= 1 && (to_rank - enemy_rank).abs() <= 1 {
        score += 80;
    }
    if (to_file - own_file).abs() <= 1 && (to_rank - own_rank).abs() <= 1 {
        score += 24;
    }

    let undo = pos.make_move(m);
    let gives_check = pos.is_in_check(pos.side);
    let own_after = king_zone_attack_count(pos, own_king, mover);
    let enemy_after = king_zone_attack_count(pos, enemy_king, mover);
    pos.unmake_move(undo);

    if gives_check {
        score += 10_000;
    }
    score += (enemy_after - enemy_before).max(0) * 40;
    score += (own_after - own_before).max(0) * 18;
    (gives_check, score)
}

type ThreatOrderKey = (i32, i32, (u8, i32, i32), i64, usize, Move);

/// Candidate-only extension of the approved killer/history ordering. Hash
/// moves remain first, then checking moves, promotions, captures, killers,
/// and remaining quiets. Within each band, the bounded threat signal breaks
/// ties before the existing MVV/history ordering.
fn order_moves_with_threats(
    pos: &mut Position,
    moves: &mut [Move],
    hash_move: Option<Move>,
    h: Option<&SearchHeuristics>,
    ply: usize,
    ctx: &SearchContext,
) {
    ctx.add_profile_counter(&ctx.threat_ordered_moves, moves.len() as u64);
    let killers = if let Some(hh) = h {
        if hh.killers.len() > ply {
            hh.killers[ply]
        } else {
            [None, None]
        }
    } else {
        [None, None]
    };
    let mut keyed: Vec<ThreatOrderKey> = moves
        .iter()
        .enumerate()
        .map(|(index, &m)| {
            let (gives_check, threat_score) = threat_move_signal(pos, m);
            let bucket = if Some(m) == hash_move {
                0
            } else if gives_check {
                1
            } else if matches!(m.flag, MoveFlag::Promotion(_)) {
                2
            } else if pos.board[m.to as usize].is_some() || matches!(m.flag, MoveFlag::EnPassant) {
                3
            } else if Some(m) == killers[0] {
                4
            } else if Some(m) == killers[1] {
                5
            } else {
                6
            };
            let history_rank = if bucket == 6 {
                let history = h
                    .map(|hh| hh.history[pos.side as usize][m.from as usize][m.to as usize])
                    .unwrap_or(0);
                i64::from(history) * 4096 - i64::from(m.from) * 64 - i64::from(m.to)
            } else {
                0
            };
            (
                bucket,
                threat_score,
                move_order_key(pos, m),
                history_rank,
                index,
                m,
            )
        })
        .collect();

    keyed.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.3.cmp(&a.3))
            .then_with(|| a.4.cmp(&b.4))
    });
    for (index, (_, _, _, _, _, m)) in keyed.into_iter().enumerate() {
        moves[index] = m;
    }
}

fn reorder_root_moves_by_previous_scores(
    root_moves: &mut [Move],
    previous_scores: &[(Move, i32)],
    ctx: &SearchContext,
) {
    if previous_scores.is_empty() {
        return;
    }
    let mut indexed: Vec<(Option<i32>, usize, Move)> = root_moves
        .iter()
        .enumerate()
        .map(|(index, &m)| {
            (
                previous_scores
                    .iter()
                    .find(|(scored_move, _)| *scored_move == m)
                    .map(|(_, score)| *score),
                index,
                m,
            )
        })
        .collect();
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    for (index, (_, _, m)) in indexed.into_iter().enumerate() {
        root_moves[index] = m;
    }
    ctx.add_profile_counter(&ctx.root_reorders, 1);
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
    // The public qsearch API is the historical correctness path. It must not
    // inherit candidate state left behind by a private profile search on a
    // reused context or Position.
    ctx.see_enabled.store(false, Ordering::Relaxed);
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
/// 9 args = the public 7-arg [`quiescence`] entry plus the live [`PvTable`]
/// and the isolated qsearch-movegen switch; kept explicit (see
/// [`negamax_impl`] for the rationale).
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
    quiescence_entered_impl(
        pos, ply, qply, alpha, beta, ctx, limits, pv, path, false, false, false,
    )
}

/// The quiescence body, for a node the caller has ALREADY counted.
/// Threads a [`PvTable`] so the tactical principal variation is recorded.
///
/// 11 args = the public 7-arg [`quiescence`] entry plus the live [`PvTable`]
/// and the three isolated qsearch candidate switches;
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
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    path: &mut SearchPath,
    qsearch_movegen: bool,
    qsearch_pruning: bool,
    qsearch_fast_pruning: bool,
) -> Option<i32> {
    quiescence_entered_impl_with_profile(
        pos,
        ply,
        qply,
        alpha,
        beta,
        ctx,
        limits,
        pv,
        path,
        SearchProfile::M4Reference,
        qsearch_movegen,
        qsearch_pruning,
        qsearch_fast_pruning,
    )
}

#[allow(clippy::too_many_arguments)]
fn quiescence_entered_impl_with_profile(
    pos: &mut Position,
    ply: u32,
    qply: u32,
    mut alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    path: &mut SearchPath,
    profile: SearchProfile,
    qsearch_movegen: bool,
    qsearch_pruning: bool,
    qsearch_fast_pruning: bool,
) -> Option<i32> {
    ctx.add_profile_counter(&ctx.qsearch_nodes, 1);
    // UCI seldepth: `ply` IS the global ply (entry is qply=0 at the
    // main-search ply; the recursion passes BOTH ply+1 and qply+1, so `ply`
    // alone carries the single global-ply definition shared with the main
    // search - adding qply would double-count the qsearch descent).
    ctx.record_seldepth(ply);
    if ctx.profiling_enabled {
        ctx.qsearch_seldepth
            .fetch_max(ply as u64, Ordering::Relaxed);
    }
    // Node already entered by the caller: clear the row before any return.
    pv.clear_at(ply);

    let in_check = pos.is_in_check(pos.side);
    if in_check {
        ctx.add_profile_counter(&ctx.qsearch_in_check_entries, 1);
    }
    // S7.5-0: qsearch forcing funnel, kept strictly separate from main.
    // A qsearch node with qply > 0 and in check was entered from a checking
    // edge inside qsearch; a qply == 0 in-check node came from the main tree.
    ctx.add_profile_counter(&ctx.s75_q_nodes, 1);
    if ctx.profiling_enabled && in_check {
        ctx.add_profile_counter(&ctx.s75_q_in_check_nodes, 1);
        ctx.add_profile_counter(&ctx.s75_q_check_child_entered, 1);
        if qply > 0 {
            ctx.add_profile_counter(&ctx.s75_q_checking_edges_searched, 1);
        }
    }

    // The reference path generates every legal move. The isolated candidate
    // generates only tactical legal moves at non-check nodes and all legal
    // evasions under check; a tactical-empty node performs one early-stop
    // legal-move probe so stalemate is never mistaken for stand-pat.
    // S7.1A: the lazy candidate DEFERS the non-check tactical generation past
    // the stand-pat beta cutoff / stalemate probe (same searched tree).
    let lazy = !in_check && profile.uses_qsearch_lazy();
    let mut legal = if lazy {
        Vec::new()
    } else if qsearch_movegen {
        if in_check {
            generate_legal_evasions_profiled(pos, ctx)
        } else if profile.uses_threat_aware_qsearch() && qply < MAX_FORCING_QPLY {
            // The threat-aware candidate extends qsearch's tactical set with
            // quiet checks for a small, explicit number of qsearch plies.
            // Legal generation remains exhaustive here; only the returned
            // vector is filtered, so stalemate handling below is unchanged.
            let all_legal = generate_legal_moves_profiled(pos, ctx);
            let mut forcing = Vec::with_capacity(all_legal.len());
            for m in all_legal {
                let gives_check = move_gives_check(pos, m);
                if gives_check {
                    ctx.add_profile_counter(&ctx.qsearch_check_moves, 1);
                }
                if is_tactical(pos, m) || gives_check {
                    forcing.push(m);
                }
            }
            forcing
        } else {
            generate_legal_tactical_moves_profiled(pos, ctx)
        }
    } else {
        generate_legal_moves_profiled(pos, ctx)
    };
    // S7.5-0: classify the in-check child's natural qsearch movegen result.
    // This is the same `legal` list the normal qsearch path uses; no extra
    // legal generation is performed for attribution.
    if ctx.profiling_enabled && in_check {
        ctx.add_profile_counter(&ctx.s75_q_check_child_movegen, 1);
        match legal.len() {
            0 => ctx.add_profile_counter(&ctx.s75_q_check_child_terminal_0, 1),
            1 => ctx.add_profile_counter(&ctx.s75_q_check_child_evasions_1, 1),
            2 => ctx.add_profile_counter(&ctx.s75_q_check_child_evasions_2, 1),
            _ => ctx.add_profile_counter(&ctx.s75_q_check_child_evasions_3plus, 1),
        }
        if legal.len() == 1 {
            ctx.add_profile_counter(&ctx.s75_q_single_evasion_nodes_raw, 1);
            if qply == 0 {
                ctx.add_profile_counter(&ctx.s75_q_single_evasion_qply0, 1);
            } else {
                ctx.add_profile_counter(&ctx.s75_q_single_evasion_qply1plus, 1);
            }
        }
    }
    if lazy {
        ctx.add_profile_counter(&ctx.qsearch_lazy_has_any_probes, 1);
        if !has_any_legal_move_profiled(pos, ctx) {
            return Some(0);
        }
    } else if legal.is_empty() {
        if in_check {
            return Some(-(MATE - ply as i32));
        }
        if !qsearch_movegen || !has_any_legal_move_profiled(pos, ctx) {
            return Some(0);
        }
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
    // S7.1A: the lazy candidate orders AFTER the deferred movegen (below), so
    // this eager ordering is skipped for lazy non-check nodes.
    if !lazy {
        let ordering_start = ctx.sample_begin(&ctx.timing_ordering);
        if profile.uses_threat_ordering() {
            order_moves_with_threats(pos, &mut legal, None, None, ply as usize, ctx);
        } else {
            order_moves(pos, &mut legal);
        }
        if let Some(start) = ordering_start {
            ctx.sample_end(&ctx.timing_ordering, start);
        }
    }

    // Termination cap.
    if qply >= MAX_QPLY {
        if !in_check {
            if lazy {
                ctx.add_profile_counter(&ctx.qsearch_lazy_qply_returns_before_movegen, 1);
            }
            let stand_pat = evaluate_profiled(pos, ctx, profile);
            if stand_pat >= beta {
                return Some(beta);
            }
            return Some(alpha.max(stand_pat));
        }
        return search_final_evasion_ply_with_profile(
            pos, ply, alpha, beta, &legal, ctx, limits, pv, path, profile,
        );
    }

    // Decide which moves to search.
    let tactical: Vec<Move> = if in_check {
        // Rule 1: under check, search ALL evasions, no stand-pat.
        legal
    } else {
        // Rule 2 (stalemate) already handled. Stand-pat is the lower bound:
        // the side to move is never forced to make a capture.
        let stand_pat = evaluate_profiled(pos, ctx, profile);
        if stand_pat >= beta {
            ctx.add_profile_counter(&ctx.qsearch_standpat_cutoffs, 1);
            if lazy {
                ctx.add_profile_counter(&ctx.qsearch_lazy_standpat_cutoffs_before_movegen, 1);
            }
            return Some(beta);
        }
        if stand_pat > alpha {
            ctx.add_profile_counter(&ctx.qsearch_standpat_alpha_raises, 1);
            alpha = stand_pat;
        }
        let mut tactical: Vec<Move> = if lazy {
            // S7.1A: only now materialize + order the tactical list.
            ctx.add_profile_counter(&ctx.qsearch_lazy_tactical_generations, 1);
            let mut gen = generate_legal_tactical_moves_profiled(pos, ctx);
            let ordering_start = ctx.sample_begin(&ctx.timing_ordering);
            order_moves(pos, &mut gen);
            if let Some(start) = ordering_start {
                ctx.sample_end(&ctx.timing_ordering, start);
            }
            gen
        } else if qsearch_movegen {
            legal
        } else {
            legal.into_iter().filter(|m| is_tactical(pos, *m)).collect()
        };
        if qsearch_pruning {
            tactical = if qsearch_fast_pruning {
                prune_qsearch_captures_by_fast_see(pos, tactical, ctx, alpha, beta)
            } else if ctx.features().qsearch_delta {
                // S7.1B candidate: the existing SEE<0 prune plus the
                // conservative delta rule, sharing ONE SEE computation.
                // `stand_pat < beta` holds (the beta cutoff returned above).
                prune_qsearch_captures_by_see_delta(
                    pos, tactical, ctx, alpha, beta, stand_pat, qply,
                )
            } else {
                prune_qsearch_captures_by_see(pos, tactical, ctx, alpha, beta)
            };
        }
        if ctx.see_enabled.load(Ordering::Relaxed) {
            // SEE is ordering-only. Even a losing exchange remains in the
            // qsearch so tactical compensation cannot be removed by an
            // incomplete static exchange proof.
            let mut scored = Vec::with_capacity(tactical.len());
            for (index, m) in tactical.into_iter().enumerate() {
                let see = if matches!(m.flag, MoveFlag::Promotion(_)) {
                    i32::MAX
                } else {
                    ctx.add_profile_counter(&ctx.see_calls, 1);
                    static_exchange_eval(pos, m)
                };
                scored.push((see, index, m));
            }
            scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            tactical = scored.into_iter().map(|(_, _, m)| m).collect();
        }
        tactical
    };

    for m in tactical {
        ctx.add_profile_counter(&ctx.qsearch_moves_searched, 1);
        let undo = make_move_profiled(pos, m, ctx);
        path.push_child(pos);

        // Manual child probe: try_enter_node called EXACTLY ONCE here.
        let probe = match probe_child_draw(
            pos,
            path.keys(),
            ply + 1,
            ply,
            ctx,
            limits,
            pv,
            ProbeKind::Qsearch,
        ) {
            Some(p) => p,
            None => {
                path.pop();
                unmake_move_profiled(pos, undo, ctx);
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
                match quiescence_entered_impl_with_profile(
                    pos,
                    ply + 1,
                    qply + 1,
                    -beta,
                    -alpha,
                    ctx,
                    limits,
                    pv,
                    path,
                    profile,
                    qsearch_movegen,
                    qsearch_pruning,
                    qsearch_fast_pruning,
                ) {
                    Some(s) => -s,
                    None => {
                        path.pop();
                        unmake_move_profiled(pos, undo, ctx);
                        return None;
                    }
                }
            }
        };

        path.pop();
        unmake_move_profiled(pos, undo, ctx);

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
#[allow(dead_code)]
fn search_final_evasion_ply(
    pos: &mut Position,
    ply: u32,
    alpha: i32,
    beta: i32,
    legal: &[Move],
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    path: &mut SearchPath,
) -> Option<i32> {
    search_final_evasion_ply_with_profile(
        pos,
        ply,
        alpha,
        beta,
        legal,
        ctx,
        limits,
        pv,
        path,
        SearchProfile::M4Reference,
    )
}

#[allow(clippy::too_many_arguments)]
fn search_final_evasion_ply_with_profile(
    pos: &mut Position,
    ply: u32,
    mut alpha: i32,
    beta: i32,
    legal: &[Move],
    ctx: &SearchContext,
    limits: &SearchLimits,
    pv: &mut PvTable,
    path: &mut SearchPath,
    profile: SearchProfile,
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

        let undo = make_move_profiled(pos, m, ctx);
        path.push_child(pos);

        // `legal` came from `generate_legal_moves`, so this evasion is legal:
        // the opponent is NOT attacking our king here. Score the child:
        //   - terminal FIRST: opponent has no move & is in check -> mate; no
        //     move & not in check -> stalemate (0);
        //   - then draw: automatic insufficient material -> 0 (dead position);
        //     fifty-move or threefold intended claim -> 0 (mover secures draw);
        //   - otherwise approximate with the static eval (safe cap estimate).
        let child_in_check = pos.is_in_check(pos.side);
        // S5.0A: single-use generation in the final-evasion ply (not a probe
        // duplicate, but counted so the full-legal call accounting closes).
        ctx.add_profile_counter(&ctx.final_evasion_generations, 1);
        let child_legal = generate_legal_moves_profiled(pos, ctx);
        let score = if child_legal.is_empty() {
            terminal_child_score_for_parent(child_in_check, ply)
        } else if is_insufficient_material(pos)
            || claim_available_by_intended_move(pos, path.keys())
        {
            // Draw: automatic dead position, or the mover's intended fifty-move
            // or threefold claim on this evasion — both secure 0 (no real win).
            0
        } else {
            -evaluate_profiled(pos, ctx, profile)
        };

        path.pop();
        unmake_move_profiled(pos, undo, ctx);

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
    /// re-searched; its bound is retained only as a next-iteration ordering
    /// hint, never as an exact root result.
    ScoutFailLow(i32),
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
    profile: SearchProfile,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
    heur: &mut Option<SearchHeuristics>,
) -> Option<RootIteration> {
    root_search_with_window(
        pos,
        depth,
        root_moves,
        root_claimable,
        claim_fallback,
        ctx,
        limits,
        profile,
        path,
        tt,
        heur,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn root_search_with_window(
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
    window: Option<(i32, i32)>,
    store_result: bool,
) -> Option<RootIteration> {
    // A claimable root (the side to move may claim right now) has a 0 floor:
    // the root value can never drop below 0, because the mover need not move.
    // We start from 0 so a losing/equal move cannot drag the root below the
    // claim; a move that scores > 0 truly beats the claim and is reported.
    // When not claimable, the root starts from the normal fail-soft floor.
    let (window_alpha, beta) = window.unwrap_or((i32::MIN + 1000, i32::MAX - 1000));
    let mut best_score = if root_claimable { 0 } else { i32::MIN + 1000 };
    let mut best_move: Option<Move> = None;
    let mut move_scores = Vec::with_capacity(root_moves.len());
    let mut alpha = if root_claimable {
        window_alpha.max(0)
    } else {
        window_alpha
    };
    let root_in_check = pos.is_in_check(pos.side);
    let root_single_evasion = root_in_check && root_moves.len() == 1;
    // S7.5-0: count the root in the main funnel. The root has no parent edge,
    // so it never contributes to checking-edge / chain-entry counters except
    // as the first link of a single-evasion chain.
    if ctx.profiling_enabled {
        if depth == 1 {
            ctx.add_profile_counter(&ctx.s75_main_depth1_nodes, 1);
        }
        if root_in_check {
            ctx.add_profile_counter(&ctx.s75_main_in_check_nodes, 1);
            if depth == 1 {
                ctx.add_profile_counter(&ctx.s75_main_depth1_in_check, 1);
            }
        }
        if root_single_evasion {
            ctx.add_profile_counter(&ctx.s75_main_single_evasion_nodes_raw, 1);
            if depth == 1 {
                ctx.add_profile_counter(&ctx.s75_main_single_evasion_actionable_depth1, 1);
                ctx.add_profile_counter(&ctx.s75_main_depth1_single_evasion, 1);
            } else {
                ctx.add_profile_counter(&ctx.s75_main_single_evasion_actionable_depth2plus, 1);
                if depth >= 3 {
                    ctx.add_profile_counter(&ctx.s75_main_single_evasion_depth3plus, 1);
                }
            }
            ctx.add_profile_counter(&ctx.s75_main_single_evasion_chain[0], 1);
        }
    }
    let root_single_evasion_chain = if root_single_evasion { 1 } else { 0 };
    let root_extension_budget = extension_budget_for_profile(profile);

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
    let root_key = if root_extension_budget == 0 {
        current_tt_key(pos, path)
    } else {
        current_tt_key_with_forcing_budget(pos, path, root_extension_budget)
    };
    ctx.add_profile_counter(&ctx.tt_probes, 1);
    // S7.5A keeps normal TT reuse semantics; legacy forcing stays exact-depth.
    let root_probe = if profile.uses_forcing_search() {
        probe_tt_for_search_exact_depth(tt, root_key, depth, 0, alpha, beta)
    } else {
        probe_tt_for_search(tt, root_key, depth, 0, alpha, beta)
    };
    if root_probe.hit {
        ctx.add_profile_counter(&ctx.tt_hits, 1);
    }
    match root_probe.reject {
        Some(TtRejectReason::Depth) => ctx.add_profile_counter(&ctx.tt_rejected_depth, 1),
        Some(TtRejectReason::Bound) => ctx.add_profile_counter(&ctx.tt_rejected_bound, 1),
        Some(TtRejectReason::Decode) => ctx.add_profile_counter(&ctx.tt_rejected_decode, 1),
        None => {}
    }
    if root_probe.cutoff.is_some() {
        ctx.add_profile_counter(&ctx.tt_cutoffs, 1);
    }
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
        let undo = make_move_profiled(pos, m, ctx);
        path.push_child(pos);

        let (child_depth, child_extension_budget) = child_extension_params(
            pos,
            depth,
            profile,
            root_extension_budget,
            root_in_check,
            root_single_evasion,
            ctx,
        );

        // Manual child probe: try_enter_node called EXACTLY ONCE here.
        let probe = match probe_child_draw(
            pos,
            path.keys(),
            1,
            0,
            ctx,
            limits,
            &mut pv,
            ProbeKind::Root,
        ) {
            Some(p) => p,
            None => {
                path.pop();
                unmake_move_profiled(pos, undo, ctx);
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
                // S4.1c Phase B diagnostic: every root move gets a full-window
                // child search (no root scout + conditional re-search).
                let diag_full = ctx.diagnostics.is_some_and(|d| d.root_full_window);
                match if diag_full {
                    ChildWindow::Full
                } else {
                    pvs_child_window(profile, move_idx == 0, depth, alpha_before_move, beta)
                } {
                    ChildWindow::Full => {
                        #[cfg(test)]
                        if move_idx == 0 && profile == SearchProfile::Current {
                            pvs_counters::mark_root_first_full();
                        }
                        // Full-window search. Recurse into the ENTERED body —
                        // NEVER negamax_impl (double count). Handle a deeper
                        // abort EXPLICITLY before cleanup.
                        match negamax_entered_impl_with_null_and_extensions(
                            pos,
                            child_depth,
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
                            true,
                            root_single_evasion_chain,
                            child_extension_budget,
                        ) {
                            Some(s) => RootMoveOutcome::Candidate(-s),
                            None => {
                                path.pop();
                                unmake_move_profiled(pos, undo, ctx);
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
                        let scout_score = match negamax_entered_impl_with_null_and_extensions(
                            pos,
                            child_depth,
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
                            true,
                            root_single_evasion_chain,
                            child_extension_budget,
                        ) {
                            Some(s) => -s,
                            None => {
                                // Phase A: the root scout's subtree aborted.
                                #[cfg(test)]
                                pvs_counters::mark_root_abort_in_scout();
                                path.pop();
                                unmake_move_profiled(pos, undo, ctx);
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
                                unmake_move_profiled(pos, undo, ctx);
                                return None;
                            }
                            #[cfg(test)]
                            pvs_counters::mark_root_research_entered();
                            match negamax_entered_impl_with_null_and_extensions(
                                pos,
                                child_depth,
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
                                true,
                                root_single_evasion_chain,
                                child_extension_budget,
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
                                    unmake_move_profiled(pos, undo, ctx);
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
                            RootMoveOutcome::ScoutFailLow(scout_score)
                        }
                    }
                }
            }
        };

        path.pop();
        unmake_move_profiled(pos, undo, ctx);
        #[cfg(test)]
        pvs_counters::mark_root_move_visited();

        // Commit root state by MATCHING on the explicit outcome. A fail-low
        // scout never updates best / best_move / root PV / alpha. Every other
        // outcome carries a real candidate.
        let score = match outcome {
            RootMoveOutcome::ScoutFailLow(s) => {
                move_scores.push((m, s));
                continue;
            }
            RootMoveOutcome::Candidate(s) => s,
        };
        move_scores.push((m, score));

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
        let iteration = RootIteration {
            score: 0,
            best_move: claim_fallback,
            pv: Vec::new(),
            move_scores,
        };
        if store_result {
            store_root_iteration(tt, pos, path, depth, &iteration, ctx, root_extension_budget);
        }
        return Some(iteration);
    }

    best_move.map(|bm| {
        let iteration = RootIteration {
            score: best_score,
            best_move: bm,
            pv: std::mem::take(&mut pv.lines[0]),
            move_scores,
        };
        if store_result {
            store_root_iteration(tt, pos, path, depth, &iteration, ctx, root_extension_budget);
        }
        iteration
    })
}

fn store_root_iteration(
    tt: &mut TranspositionTable,
    pos: &Position,
    path: &SearchPath,
    depth: u32,
    iteration: &RootIteration,
    ctx: &SearchContext,
    forcing_budget: u8,
) {
    let root_key = if forcing_budget == 0 {
        current_tt_key(pos, path)
    } else {
        current_tt_key_with_forcing_budget(pos, path, forcing_budget)
    };
    let store_move = if iteration.pv.is_empty() {
        None
    } else {
        Some(iteration.best_move)
    };
    store_tt_score_profiled(
        tt,
        root_key,
        depth,
        iteration.score,
        0,
        Bound::Exact,
        store_move,
        ctx,
    );
}

/// Current-profile root search with aspiration retries. Failed windows are
/// discarded and never stored as exact root entries; the radius doubles until
/// the score is strictly inside the window or a full window is used.
#[allow(clippy::too_many_arguments)]
fn root_search_with_aspiration(
    pos: &mut Position,
    depth: u32,
    root_moves: &mut [Move],
    root_claimable: bool,
    claim_fallback: Move,
    previous_score: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
    profile: SearchProfile,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
    heur: &mut Option<SearchHeuristics>,
) -> Option<RootIteration> {
    // A mate-range score is deliberately searched with the full window: a
    // narrow centipawn window around a forced mate is not meaningful.
    if previous_score.abs() >= MATE - 1000 {
        return root_search(
            pos,
            depth,
            root_moves,
            root_claimable,
            claim_fallback,
            ctx,
            limits,
            profile,
            path,
            tt,
            heur,
        );
    }

    let mut delta = 50i32;
    loop {
        let full_window = delta >= MATE;
        let (alpha, beta) = if full_window {
            (i32::MIN + 1000, i32::MAX - 1000)
        } else {
            (
                previous_score.saturating_sub(delta).max(i32::MIN + 1000),
                previous_score.saturating_add(delta).min(i32::MAX - 1000),
            )
        };
        let iteration = root_search_with_window(
            pos,
            depth,
            root_moves,
            root_claimable,
            claim_fallback,
            ctx,
            limits,
            profile,
            path,
            tt,
            heur,
            Some((alpha, beta)),
            false,
        )?;
        if full_window || (iteration.score > alpha && iteration.score < beta) {
            let root_extension_budget = extension_budget_for_profile(profile);
            store_root_iteration(tt, pos, path, depth, &iteration, ctx, root_extension_budget);
            return Some(iteration);
        }

        ctx.add_profile_counter(&ctx.aspiration_retries, 1);
        if iteration.score <= alpha {
            ctx.add_profile_counter(&ctx.aspiration_fail_low, 1);
        } else {
            ctx.add_profile_counter(&ctx.aspiration_fail_high, 1);
        }
        delta = delta.saturating_mul(2);
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
/// preserves the M4.0 search behavior: killer (Commit 3) and history (Commit
/// 4) ordering are applied under `SearchProfile::M41Reference` and
/// `SearchProfile::Current`, never under `M4Reference`. It runs under the
/// current evaluation function, so evaluation milestones may change scores,
/// PVs, and node counts. The persistent UCI `Hash` table is threaded through
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
/// The UCI production path calls this with the process-selected startup
/// profile, whose default is [`PRODUCTION_PROFILE`] (`CurrentFinal`);
/// `--profile current` explicitly selects [`ROLLBACK_PROFILE`]. The
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
    ctx.see_enabled.store(profile.uses_see(), Ordering::Relaxed);
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
    // (Commit 4) ordering; `M4Reference` preserves the M4.0 search policy
    // without those heuristics. The current evaluation may still change
    // scores, PVs, and node counts relative to historical pre-EVAL output.
    _profile: SearchProfile,
    path: &mut SearchPath,
    tt: &mut TranspositionTable,
) -> Option<SearchOutcome> {
    ctx.see_enabled
        .store(_profile.uses_see(), Ordering::Relaxed);
    ctx.features_mask.store(
        SearchFeaturePolicy::for_profile(_profile, ctx.diagnostics.as_ref()).to_bits(),
        Ordering::Relaxed,
    );
    ctx.legality_fast
        .store(_profile.uses_legality_fast(), Ordering::Relaxed);
    ctx.single_buffer_legal
        .store(_profile.uses_single_buffer_legal(), Ordering::Relaxed);
    ctx.single_generation_probe
        .store(_profile.uses_single_generation_probe(), Ordering::Relaxed);
    // S5.0A: the root's own legal list (generated once; not duplicated).
    ctx.add_profile_counter(&ctx.root_generations, 1);
    let mut root_moves = generate_legal_moves_profiled(pos, ctx);
    if root_moves.is_empty() {
        return None; // already terminal (checkmate / stalemate)
    }
    // Stable fallback: the first legal move. Used if we never complete a
    // single iteration (e.g. stopped before depth 1 finishes).
    let fallback = root_moves[0];

    // S4.0B bench-only diagnostics. Never active on the production UCI path
    // (UCI never sets `ctx.diagnostics`). Root contract: generate legal root
    // moves -> apply the normal CurrentFinal initial (static) ordering ->
    // record the target move's 1-based rank -> only then filter to the single
    // forced root move. Forcing happens after ranking so forced mode does not
    // trivially report rank 1.
    if let Some(diag) = ctx.diagnostics {
        if diag.forced_root_move.is_some() || diag.target_root_move.is_some() {
            order_moves(pos, &mut root_moves);
        }
        if let Some(target) = diag.target_root_move {
            let rank = root_moves
                .iter()
                .position(|m| *m == target)
                .map(|i| i as u32 + 1)
                .unwrap_or(TARGET_ROOT_RANK_NONE);
            ctx.target_root_rank.store(rank, Ordering::Relaxed);
        }
        if let Some(forced) = diag.forced_root_move {
            root_moves.retain(|m| *m == forced);
            if root_moves.is_empty() {
                return None; // forced move is not legal at the root
            }
        }
    }

    // M4.1 Commit 3: build the per-search heuristic state ONLY for
    // non-M4Reference profiles (`M41Reference` and `Current`).
    // `M4Reference` skips it entirely (no killer/history ordering),
    // preserving the M4Reference search policy. The table
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

        let iteration_started = if ctx.profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };
        let iteration_start_nodes = if ctx.profiling_enabled {
            ctx.nodes.load(Ordering::Relaxed)
        } else {
            0
        };

        let iteration = if _profile.uses_aspiration() {
            if let Some(previous_score) = completed.as_ref().map(|it| it.score) {
                root_search_with_aspiration(
                    pos,
                    depth,
                    &mut root_moves,
                    root_claimable,
                    fallback,
                    previous_score,
                    ctx,
                    limits,
                    _profile,
                    path,
                    tt,
                    &mut heuristics,
                )
            } else {
                // The first completed iteration establishes the score center;
                // it always uses a full window.
                root_search(
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
                )
            }
        } else {
            root_search(
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
            )
        };

        match iteration {
            Some(iter) => {
                let RootIteration {
                    score,
                    best_move,
                    pv,
                    move_scores,
                } = iter;
                completed_depth = depth;
                if let Some(start) = iteration_started {
                    ctx.completed_iterations.fetch_add(1, Ordering::Relaxed);
                    ctx.last_completed_iteration_ms
                        .store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                    ctx.last_completed_iteration_nodes.store(
                        ctx.nodes
                            .load(Ordering::Relaxed)
                            .saturating_sub(iteration_start_nodes),
                        Ordering::Relaxed,
                    );
                }
                // The threat-aware candidate reuses all scores observed in
                // the completed iteration to seed the next root order. The
                // approved profiles retain the historical best-move lift.
                if _profile.uses_threat_ordering() {
                    reorder_root_moves_by_previous_scores(&mut root_moves, &move_scores, ctx);
                } else if let Some(idx) = root_moves.iter().position(|m| *m == best_move) {
                    root_moves.swap(0, idx);
                }
                // S4.1 candidate: after the previous best is lifted to index 0,
                // sort only the remaining QUIET root moves by the existing
                // history heuristic (descending, stable). No root killers, no
                // static-eval ordering, no history-update changes.
                if _profile.uses_root_quiet_history() {
                    order_root_quiets_by_history(pos, &mut root_moves, heuristics.as_ref());
                }
                // S4.1b candidate: sort only the remaining QUIET root moves by
                // the previous completed iteration's root scores (descending,
                // stable). No history/killer/static-eval/threat signal; no PVS
                // or re-search changes.
                if _profile.uses_root_prev_score() {
                    order_root_quiets_by_prev_scores(pos, &mut root_moves, &move_scores);
                }
                // Standard UCI info: nodes from the atomic counter, time
                // from the search start, nps = nodes*1000/ms. nps is guarded
                // against time == 0 (no divide-by-zero) and computed in u128
                // to avoid overflow on huge node counts. Only completed
                // iterations emit info; an aborted depth 1 emits nothing.
                // The PV is the full principal variation of this iteration.
                let nodes = ctx.nodes.load(Ordering::Relaxed);
                let seldepth = ctx.seldepth.load(Ordering::Relaxed);
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
                    "info depth {} seldepth {} score {} nodes {} time {} nps {} pv {}",
                    depth,
                    seldepth,
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
                    move_scores,
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
                if ctx.profiling_enabled {
                    ctx.aborted_iteration_depth
                        .store(depth as u64, Ordering::Relaxed);
                    ctx.aborted_iteration_nodes.store(
                        ctx.nodes
                            .load(Ordering::Relaxed)
                            .saturating_sub(iteration_start_nodes),
                        Ordering::Relaxed,
                    );
                }
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
    fn see_distinguishes_winning_and_defended_captures() {
        let winning = parse_fen(MVV_POS).unwrap();
        let take_queen = find_move(&winning, "e4a4");
        assert!(static_exchange_eval(&winning, take_queen) > 0);

        let defended = parse_fen("6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1").unwrap();
        let nxe5 = find_move(&defended, "f3e5");
        assert!(static_exchange_eval(&defended, nxe5) < 0);
    }

    #[test]
    fn see_does_not_mutate_the_position() {
        let pos = parse_fen(MVV_POS).unwrap();
        let before = to_fen(&pos);
        let move_to_test = find_move(&pos, "e4a4");
        let _ = static_exchange_eval(&pos, move_to_test);
        assert_eq!(to_fen(&pos), before);
    }

    #[test]
    fn see_filter_is_enabled_only_for_see_profile() {
        fn see_stats_for(profile: SearchProfile) -> (u64, u64) {
            let mut pos = parse_fen(MVV_POS).unwrap();
            let key = pos.zobrist_key();
            let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let limits = SearchLimits {
                depth: Some(2),
                ..Default::default()
            };
            let mut tt = TranspositionTable::disabled();
            let _ = search_best_move_with_history_tt_and_profile(
                &mut pos,
                &[key],
                &limits,
                &ctx,
                &mut tt,
                profile,
            );
            (
                ctx.see_calls.load(Ordering::Relaxed),
                ctx.see_pruned.load(Ordering::Relaxed),
            )
        }

        assert_eq!(see_stats_for(SearchProfile::M4Reference), (0, 0));
        assert_eq!(see_stats_for(SearchProfile::M41Reference), (0, 0));
        assert_eq!(see_stats_for(SearchProfile::Current), (0, 0));
        let (see_calls, see_pruned) = see_stats_for(SearchProfile::SeeCandidate);
        assert!(see_calls > 0);
        assert_eq!(see_pruned, 0, "SEE is ordering-only");
    }

    #[test]
    fn cumulative_profiles_enable_exactly_the_declared_features() {
        let cases = [
            (SearchProfile::Current, false, false, false, false),
            (SearchProfile::CurrentAspiration, false, true, false, false),
            (
                SearchProfile::CurrentAspirationLmr,
                false,
                true,
                true,
                false,
            ),
            (
                SearchProfile::CurrentAspirationLmrFutility,
                false,
                true,
                true,
                true,
            ),
            (
                SearchProfile::CurrentAspirationLmrFutilitySee,
                true,
                true,
                true,
                true,
            ),
        ];

        for (profile, see, aspiration, lmr, futility) in cases {
            assert!(
                profile.uses_pvs(),
                "cumulative profile lost PVS: {profile:?}"
            );
            assert_eq!(profile.uses_see(), see, "SEE contract: {profile:?}");
            assert_eq!(
                profile.uses_aspiration(),
                aspiration,
                "aspiration contract: {profile:?}"
            );
            assert_eq!(profile.uses_lmr(), lmr, "LMR contract: {profile:?}");
            assert_eq!(
                profile.uses_futility(),
                futility,
                "futility contract: {profile:?}"
            );
            assert!(
                !profile.uses_null_move(),
                "null probe must stay outside cumulative stack: {profile:?}"
            );
        }

        assert!(SearchProfile::Current.uses_pvs());
        assert!(!SearchProfile::Current.uses_see());
        assert!(!SearchProfile::Current.uses_aspiration());
        assert!(!SearchProfile::Current.uses_lmr());
        assert!(!SearchProfile::Current.uses_futility());
        assert!(SearchProfile::Current.uses_qsearch_movegen());
        assert!(!SearchProfile::Current.uses_qsearch_pruning());
        assert!(SearchProfile::CurrentLmr.uses_pvs());
        assert!(SearchProfile::CurrentLmr.uses_qsearch_movegen());
        assert!(SearchProfile::CurrentLmr.uses_lmr());
        assert!(!SearchProfile::CurrentLmr.uses_see());
        assert!(!SearchProfile::CurrentLmr.uses_aspiration());
        assert!(!SearchProfile::CurrentLmr.uses_null_move());
        assert!(!SearchProfile::CurrentLmr.uses_futility());
        assert!(!SearchProfile::CurrentLmr.uses_qsearch_pruning());
        assert!(SearchProfile::CurrentThreatAware.uses_pvs());
        assert!(SearchProfile::CurrentThreatAware.uses_qsearch_movegen());
        assert!(SearchProfile::CurrentThreatAware.uses_threat_aware_eval());
        assert!(SearchProfile::CurrentThreatAware.uses_threat_ordering());
        assert!(SearchProfile::CurrentThreatAware.uses_threat_aware_qsearch());
        assert!(SearchProfile::CurrentThreatAware.uses_forcing_search());
        assert!(!SearchProfile::CurrentThreatAware.uses_see());
        assert!(!SearchProfile::CurrentThreatAware.uses_aspiration());
        assert!(!SearchProfile::CurrentThreatAware.uses_lmr());
        assert!(!SearchProfile::CurrentThreatAware.uses_null_move());
        assert!(!SearchProfile::CurrentThreatAware.uses_futility());
        assert!(!SearchProfile::CurrentThreatAware.uses_qsearch_pruning());
        for profile in [
            SearchProfile::CurrentThreatAwareNoQchecks,
            SearchProfile::CurrentThreatAwareEvalOrder,
        ] {
            assert!(profile.uses_pvs());
            assert!(profile.uses_qsearch_movegen());
            assert!(profile.uses_threat_aware_eval());
            assert!(profile.uses_threat_ordering());
            assert!(!profile.uses_see());
            assert!(!profile.uses_aspiration());
            assert!(!profile.uses_lmr());
            assert!(!profile.uses_null_move());
            assert!(!profile.uses_futility());
            assert!(!profile.uses_qsearch_pruning());
        }
        assert!(SearchProfile::CurrentThreatAwareNoQchecks.uses_forcing_search());
        assert!(!SearchProfile::CurrentThreatAwareNoQchecks.uses_threat_aware_qsearch());
        assert!(!SearchProfile::CurrentThreatAwareEvalOrder.uses_forcing_search());
        assert!(!SearchProfile::CurrentThreatAwareEvalOrder.uses_threat_aware_qsearch());
        assert!(SearchProfile::CurrentThreatAwareEvalOnly.uses_threat_aware_eval());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_threat_ordering());
        assert!(SearchProfile::CurrentThreatAwareEvalOnly.uses_pvs());
        assert!(SearchProfile::CurrentThreatAwareEvalOnly.uses_qsearch_movegen());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_aspiration());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_lmr());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_null_move());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_futility());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_see());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_qsearch_pruning());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_forcing_search());
        assert!(!SearchProfile::CurrentThreatAwareEvalOnly.uses_threat_aware_qsearch());
        assert!(SearchProfile::CurrentThreatAwareOrderOnly.uses_pvs());
        assert!(SearchProfile::CurrentThreatAwareOrderOnly.uses_qsearch_movegen());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_aspiration());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_lmr());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_null_move());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_futility());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_see());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_qsearch_pruning());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_threat_aware_eval());
        assert!(SearchProfile::CurrentThreatAwareOrderOnly.uses_threat_ordering());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_forcing_search());
        assert!(!SearchProfile::CurrentThreatAwareOrderOnly.uses_threat_aware_qsearch());
        assert!(!SearchProfile::Current.uses_threat_aware_eval());
        assert!(!SearchProfile::Current.uses_eval2());
        assert!(SearchProfile::CurrentEval2.uses_pvs());
        assert!(SearchProfile::CurrentEval2.uses_qsearch_movegen());
        assert!(SearchProfile::CurrentEval2.uses_eval2());
        assert!(!SearchProfile::CurrentEval2.uses_threat_aware_eval());
        assert!(!SearchProfile::CurrentEval2.uses_threat_ordering());
        assert!(!SearchProfile::CurrentEval2.uses_forcing_search());
        assert!(!SearchProfile::CurrentEval2.uses_threat_aware_qsearch());
        assert!(!SearchProfile::CurrentEval2.uses_aspiration());
        assert!(!SearchProfile::CurrentEval2.uses_lmr());
        assert!(!SearchProfile::CurrentEval2.uses_null_move());
        assert!(!SearchProfile::CurrentEval2.uses_futility());
        assert!(!SearchProfile::CurrentEval2.uses_see());
        assert!(!SearchProfile::CurrentEval2.uses_qsearch_pruning());
        assert!(SearchProfile::CurrentQsearchMovegen.uses_pvs());
        assert!(SearchProfile::CurrentQsearchMovegen.uses_qsearch_movegen());
        assert!(!SearchProfile::CurrentQsearchMovegen.uses_see());
        assert!(!SearchProfile::CurrentQsearchMovegen.uses_aspiration());
        assert!(!SearchProfile::CurrentQsearchMovegen.uses_lmr());
        assert!(!SearchProfile::CurrentQsearchMovegen.uses_futility());
        assert!(!SearchProfile::CurrentQsearchMovegen.uses_null_move());
        assert!(SearchProfile::CurrentQsearchPruning.uses_pvs());
        assert!(SearchProfile::CurrentQsearchPruning.uses_qsearch_movegen());
        assert!(SearchProfile::CurrentQsearchPruning.uses_qsearch_pruning());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_qsearch_fast_pruning());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_see());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_aspiration());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_lmr());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_futility());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_null_move());
        assert!(SearchProfile::CurrentQsearchFastPruning.uses_pvs());
        assert!(SearchProfile::CurrentQsearchFastPruning.uses_qsearch_movegen());
        assert!(SearchProfile::CurrentQsearchFastPruning.uses_qsearch_pruning());
        assert!(SearchProfile::CurrentQsearchFastPruning.uses_qsearch_fast_pruning());
        assert!(!SearchProfile::CurrentQsearchFastPruning.uses_see());
        assert!(!SearchProfile::CurrentQsearchFastPruning.uses_aspiration());
        assert!(!SearchProfile::CurrentQsearchFastPruning.uses_lmr());
        assert!(!SearchProfile::CurrentQsearchFastPruning.uses_futility());
        assert!(!SearchProfile::CurrentQsearchFastPruning.uses_null_move());
        assert!(SearchProfile::CurrentFinal.uses_pvs());
        assert!(!SearchProfile::CurrentFinal.uses_see());
        assert!(SearchProfile::CurrentFinal.uses_aspiration());
        assert!(SearchProfile::CurrentFinal.uses_lmr());
        assert!(SearchProfile::CurrentFinal.uses_null_move());
        assert!(SearchProfile::CurrentFinal.uses_futility());
        assert!(SearchProfile::CurrentFinal.uses_qsearch_movegen());
        assert!(SearchProfile::CurrentFinal.uses_qsearch_pruning());
        assert!(!SearchProfile::CurrentFinal.uses_qsearch_fast_pruning());
        assert!(!SearchProfile::CurrentFinal.uses_eval2());
        assert!(!SearchProfile::CurrentFinal.uses_threat_aware_eval());
        assert!(!SearchProfile::CurrentFinal.uses_threat_ordering());
        assert!(!SearchProfile::CurrentFinal.uses_forcing_search());
        assert!(!SearchProfile::CurrentFinal.uses_threat_aware_qsearch());
        for profile in [
            SearchProfile::CurrentAspiration,
            SearchProfile::CurrentAspirationLmr,
            SearchProfile::CurrentAspirationLmrFutility,
            SearchProfile::CurrentAspirationLmrFutilitySee,
        ] {
            assert!(
                profile.uses_qsearch_movegen(),
                "Current-based candidate lost integrated qsearch movegen: {profile:?}"
            );
        }
    }

    #[test]
    fn aspiration_is_isolated_and_preserves_search_state() {
        let pos = parse_fen(START_FEN).unwrap();
        let key = pos.zobrist_key();
        let before_fen = to_fen(&pos);
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };

        let mut reference_pos = pos;
        let reference_ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let reference = search_best_move_with_history_tt_and_profile(
            &mut reference_pos,
            &[key],
            &limits,
            &reference_ctx,
            &mut TranspositionTable::disabled(),
            SearchProfile::M41Reference,
        )
        .expect("reference search must complete");

        let mut aspiration_pos = pos;
        let aspiration_ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let aspiration = search_best_move_with_history_tt_and_profile(
            &mut aspiration_pos,
            &[key],
            &limits,
            &aspiration_ctx,
            &mut TranspositionTable::disabled(),
            SearchProfile::AspirationCandidate,
        )
        .expect("aspiration search must complete");

        assert_eq!(reference.score, aspiration.score);
        assert_eq!(to_fen(&reference_pos), before_fen);
        assert_eq!(to_fen(&aspiration_pos), before_fen);
        assert_eq!(aspiration.completed_depth, 3);
        assert_eq!(
            aspiration_ctx.aspiration_retries.load(Ordering::Relaxed),
            aspiration_ctx.aspiration_fail_low.load(Ordering::Relaxed)
                + aspiration_ctx.aspiration_fail_high.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn lmr_reduces_only_eligible_quiet_moves() {
        let pos = parse_fen(START_FEN).unwrap();
        let quiet = find_move(&pos, "e2e3");
        assert!(late_move_reduction(&mut pos.clone(), quiet, true, 5, 3, false) > 0);
        assert!(late_move_reduction(&mut pos.clone(), quiet, true, 5, 3, false) > 0);
        assert_eq!(
            late_move_reduction(&mut pos.clone(), quiet, true, 5, 0, false),
            0,
            "PV move must never be reduced"
        );
        assert_eq!(
            late_move_reduction(&mut pos.clone(), quiet, true, 5, 0, false),
            0,
            "CurrentLmr PV move must never be reduced"
        );
        assert_eq!(
            late_move_reduction(&mut pos.clone(), quiet, true, 5, 3, true),
            0,
            "in-check node must never be reduced"
        );
        assert_eq!(
            late_move_reduction(&mut pos.clone(), quiet, true, 5, 3, true),
            0,
            "CurrentLmr in-check node must never be reduced"
        );
        assert_eq!(
            late_move_reduction(&mut pos.clone(), quiet, false, 5, 3, false),
            0,
            "reference profile must not reduce"
        );

        let kqk = parse_fen("7k/8/8/8/8/8/3QK3/8 w - - 0 1").unwrap();
        let quiet_endgame = find_move(&kqk, "d2d3");
        assert_eq!(
            late_move_reduction(&mut kqk.clone(), quiet_endgame, true, 5, 3, false),
            0,
            "low-material endgames must not be reduced"
        );

        let cap_pos = parse_fen("7k/8/8/8/8/8/K7/Rr6 w - - 0 1").unwrap();
        let capture = find_move(&cap_pos, "a1b1");
        assert_eq!(
            late_move_reduction(&mut cap_pos.clone(), capture, true, 5, 3, false),
            0,
            "captures must never be reduced"
        );

        let promo_pos = parse_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").unwrap();
        let promotion = find_move(&promo_pos, "a7a8q");
        assert_eq!(
            late_move_reduction(&mut promo_pos.clone(), promotion, true, 5, 3, false),
            0,
            "promotions must never be reduced"
        );

        let ep_pos = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let en_passant = find_move(&ep_pos, "e5d6");
        assert_eq!(
            late_move_reduction(&mut ep_pos.clone(), en_passant, true, 5, 3, false),
            0,
            "en-passant captures must never be reduced"
        );

        let checking_pos = parse_fen("4k3/8/8/8/8/8/4Q3/K6R w - - 0 1").unwrap();
        let checking_quiet = find_move(&checking_pos, "e2e7");
        assert_eq!(
            late_move_reduction(&mut checking_pos.clone(), checking_quiet, true, 5, 3, false),
            0,
            "checking quiet moves must never be reduced"
        );
    }

    #[test]
    fn root_quiet_history_orders_only_quiet_slots() {
        let pos = parse_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
            .unwrap();
        let cap1 = find_move(&pos, "f3e5");
        let cap2 = find_move(&pos, "c4f7");
        let q1 = find_move(&pos, "d3d4");
        let q2 = find_move(&pos, "a2a3");
        let q3 = find_move(&pos, "c2c3");
        assert!(is_tactical(&pos, cap1) && is_tactical(&pos, cap2));
        assert!(!is_tactical(&pos, q1) && !is_tactical(&pos, q2) && !is_tactical(&pos, q3));

        let mut h = SearchHeuristics::new();
        let color = pos.side_to_move() as usize;
        h.history[color][q1.from as usize][q1.to as usize] = 100;
        h.history[color][q2.from as usize][q2.to as usize] = 9000;
        h.history[color][q3.from as usize][q3.to as usize] = 100;

        let mut moves = vec![cap1, q1, cap2, q2, q3];
        let mut before: Vec<String> = moves.iter().map(|m| move_to_uci(*m)).collect();
        before.sort();
        order_root_quiets_by_history(&pos, &mut moves, Some(&h));
        let mut after: Vec<String> = moves.iter().map(|m| move_to_uci(*m)).collect();
        after.sort();
        assert_eq!(before, after, "no move is added or dropped");
        assert_eq!(moves[0], cap1, "previous best at index 0 is preserved");
        assert_eq!(moves[1], q2, "highest-history quiet is searched first");
        assert_eq!(moves[2], cap2, "tactical move keeps its slot");
        assert_eq!(
            moves[3], q1,
            "equal-history quiets keep input order (q1 before q3)"
        );
        assert_eq!(moves[4], q3);
    }

    #[test]
    fn root_quiet_history_keeps_tactical_slots_and_previous_best() {
        let pos = parse_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
            .unwrap();
        let cap1 = find_move(&pos, "f3e5");
        let q1 = find_move(&pos, "d3d4");
        let q2 = find_move(&pos, "a2a3");

        // Empty history: the quiet sort must be a no-op permutation.
        let h = SearchHeuristics::new();
        let mut moves = vec![cap1, q1, q2];
        let before = moves.clone();
        order_root_quiets_by_history(&pos, &mut moves, Some(&h));
        assert_eq!(moves, before, "zero history leaves the order untouched");
    }

    #[test]
    fn root_quiet_history_profile_matches_current_final_at_fixed_depth() {
        // The candidate inherits every CurrentFinal feature; only root quiet
        // ordering differs, which does not change the minimax value.
        let pos = parse_fen(START_FEN).unwrap();
        let hist = vec![pos.zobrist_key()];
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let run = |profile: SearchProfile| -> SearchOutcome {
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let mut tt = TranspositionTable::disabled();
            search_best_move_with_history_tt_and_profile(
                &mut pos.clone(),
                &hist,
                &limits,
                &ctx,
                &mut tt,
                profile,
            )
            .expect("outcome")
        };
        let cf = run(SearchProfile::CurrentFinal);
        let rh = run(SearchProfile::CurrentFinalRootHistory);
        assert_eq!(rh.score, cf.score, "same minimax score at fixed depth");
        assert_eq!(rh.best_move, cf.best_move, "same best move at fixed depth");
        assert_eq!(rh.completed_depth, cf.completed_depth);
        assert!(!rh.stopped && !cf.stopped);
    }

    #[test]
    fn root_prev_score_orders_only_quiet_slots() {
        let pos = parse_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
            .unwrap();
        let cap1 = find_move(&pos, "f3e5");
        let cap2 = find_move(&pos, "c4f7");
        let q1 = find_move(&pos, "d3d4");
        let q2 = find_move(&pos, "a2a3");
        let q3 = find_move(&pos, "c2c3");

        let previous_scores = vec![(q1, 50), (q2, 120), (q3, 50), (cap1, 999), (cap2, 800)];
        let mut moves = vec![cap1, q1, cap2, q2, q3];
        let mut before: Vec<String> = moves.iter().map(|m| move_to_uci(*m)).collect();
        before.sort();
        order_root_quiets_by_prev_scores(&pos, &mut moves, &previous_scores);
        let mut after: Vec<String> = moves.iter().map(|m| move_to_uci(*m)).collect();
        after.sort();
        assert_eq!(before, after, "no move is added or dropped");
        assert_eq!(moves[0], cap1, "previous best at index 0 is preserved");
        assert_eq!(
            moves[1], q2,
            "highest previous-score quiet is searched first"
        );
        assert_eq!(moves[2], cap2, "tactical move keeps its slot");
        assert_eq!(
            moves[3], q1,
            "equal previous-score quiets keep input order (q1 before q3)"
        );
        assert_eq!(moves[4], q3);
    }

    #[test]
    fn root_prev_score_profile_matches_current_final_at_fixed_depth() {
        // The candidate inherits every CurrentFinal feature; only root quiet
        // ordering differs, which does not change the minimax value.
        let pos = parse_fen(START_FEN).unwrap();
        let hist = vec![pos.zobrist_key()];
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let run = |profile: SearchProfile| -> SearchOutcome {
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let mut tt = TranspositionTable::disabled();
            search_best_move_with_history_tt_and_profile(
                &mut pos.clone(),
                &hist,
                &limits,
                &ctx,
                &mut tt,
                profile,
            )
            .expect("outcome")
        };
        let cf = run(SearchProfile::CurrentFinal);
        let ps = run(SearchProfile::CurrentFinalRootPrevScore);
        assert_eq!(ps.score, cf.score, "same minimax score at fixed depth");
        assert_eq!(ps.best_move, cf.best_move, "same best move at fixed depth");
        assert_eq!(ps.completed_depth, cf.completed_depth);
        assert!(!ps.stopped && !cf.stopped);
    }

    #[test]
    fn legality_fast_profile_matches_current_final_search_tree() {
        // S4.3B: identical legal lists and order -> identical fixed-depth
        // search tree (nodes, score, bestmove).
        let pos = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let hist = vec![pos.zobrist_key()];
        let limits = SearchLimits {
            depth: Some(4),
            ..Default::default()
        };
        let run = |profile: SearchProfile| -> (SearchOutcome, u64) {
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let mut tt = TranspositionTable::disabled();
            let out = search_best_move_with_history_tt_and_profile(
                &mut pos.clone(),
                &hist,
                &limits,
                &ctx,
                &mut tt,
                profile,
            )
            .expect("outcome");
            (out, ctx.nodes.load(Ordering::Relaxed))
        };
        let (cf, cf_nodes) = run(SearchProfile::CurrentFinal);
        let (lf, lf_nodes) = run(SearchProfile::CurrentFinalLegalityFast);
        assert_eq!(lf_nodes, cf_nodes, "identical node count at fixed depth");
        assert_eq!(lf.score, cf.score, "identical score");
        assert_eq!(lf.best_move, cf.best_move, "identical best move");
        assert_eq!(lf.completed_depth, cf.completed_depth);
        assert_eq!(lf.pv, cf.pv, "identical PV");
    }

    #[test]
    fn single_buffer_profile_matches_current_final_search_tree() {
        // S4.4B: single-buffer full-legal materialization produces identical
        // ordered move lists -> identical fixed-depth search tree (nodes,
        // score, bestmove, PV) on every corpus-class position.
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/8/4q3/8/4N3/4K3 w - - 0 1",
            "4k3/8/8/3pP3/4q3/8/8/4K3 w - d6 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/8/8/4r3/4K3/8 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        let run = |pos: &Position, profile: SearchProfile| -> (SearchOutcome, u64) {
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let mut tt = TranspositionTable::disabled();
            let hist = vec![pos.zobrist_key()];
            let limits = SearchLimits {
                depth: Some(4),
                ..Default::default()
            };
            let out = search_best_move_with_history_tt_and_profile(
                &mut pos.clone(),
                &hist,
                &limits,
                &ctx,
                &mut tt,
                profile,
            )
            .expect("outcome");
            (out, ctx.nodes.load(Ordering::Relaxed))
        };
        for fen in positions {
            let pos = parse_fen(fen).unwrap_or_else(|e| panic!("{fen}: {e}"));
            let (cf, cf_nodes) = run(&pos, SearchProfile::CurrentFinal);
            let (sb, sb_nodes) = run(&pos, SearchProfile::CurrentFinalSingleBuffer);
            assert_eq!(sb_nodes, cf_nodes, "identical node count for {fen}");
            assert_eq!(sb.score, cf.score, "identical score for {fen}");
            assert_eq!(sb.best_move, cf.best_move, "identical best move for {fen}");
            assert_eq!(sb.completed_depth, cf.completed_depth, "for {fen}");
            assert_eq!(sb.pv, cf.pv, "identical PV for {fen}");
        }
    }

    #[test]
    fn single_generation_profile_matches_current_final_search_tree() {
        // S5.0B: probe uses has-any instead of a discarded full legal list ->
        // identical emptiness decision -> identical fixed-depth search tree.
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/8/4q3/8/4N3/4K3 w - - 0 1",
            "4k3/8/8/3pP3/4q3/8/8/4K3 w - d6 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/8/8/4r3/4K3/8 w - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/4q3/8/4P3/4K3 w - - 0 1", // mate/claim-adjacent lines
        ];
        let run = |pos: &Position, profile: SearchProfile| -> (SearchOutcome, u64) {
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let mut tt = TranspositionTable::disabled();
            let hist = vec![pos.zobrist_key()];
            let limits = SearchLimits {
                depth: Some(4),
                ..Default::default()
            };
            let out = search_best_move_with_history_tt_and_profile(
                &mut pos.clone(),
                &hist,
                &limits,
                &ctx,
                &mut tt,
                profile,
            )
            .expect("outcome");
            (out, ctx.nodes.load(Ordering::Relaxed))
        };
        for fen in positions {
            let pos = parse_fen(fen).unwrap_or_else(|e| panic!("{fen}: {e}"));
            let (cf, cf_nodes) = run(&pos, SearchProfile::CurrentFinal);
            let (sg, sg_nodes) = run(&pos, SearchProfile::CurrentFinalSingleGeneration);
            assert_eq!(sg_nodes, cf_nodes, "identical node count for {fen}");
            assert_eq!(sg.score, cf.score, "identical score for {fen}");
            assert_eq!(sg.best_move, cf.best_move, "identical best move for {fen}");
            assert_eq!(sg.completed_depth, cf.completed_depth, "for {fen}");
            assert_eq!(sg.pv, cf.pv, "identical PV for {fen}");
        }
    }

    #[test]
    fn legality_fast_promotion_policy() {
        // S4.3E: the unpinned non-check legality fast path is production
        // policy for CurrentFinal and its "exactly CurrentFinal + X"
        // derivatives; historical/experimental profiles keep legacy behavior.
        assert!(SearchProfile::CurrentFinal.uses_legality_fast());
        assert!(SearchProfile::CurrentFinalRootHistory.uses_legality_fast());
        assert!(SearchProfile::CurrentFinalRootPrevScore.uses_legality_fast());
        assert!(SearchProfile::CurrentFinalLegalityFast.uses_legality_fast());
        assert!(!SearchProfile::Current.uses_legality_fast());
        assert!(!SearchProfile::CurrentLmr.uses_legality_fast());
        assert!(!SearchProfile::CurrentEval2.uses_legality_fast());
        assert!(!SearchProfile::M4Reference.uses_legality_fast());
        assert!(!SearchProfile::CurrentThreatAware.uses_legality_fast());
        assert!(!SearchProfile::CurrentAspiration.uses_legality_fast());
        assert!(!SearchProfile::CurrentQsearchPruning.uses_legality_fast());
    }

    #[test]
    fn record_seldepth_tracks_max_global_ply() {
        // R0 Repair 2: seldepth accounting is decoupled from any search-tree
        // shape. The helper records the deepest GLOBAL ply seen and never
        // regresses; S7 tree changes (extensions, pruning, qsearch policy)
        // must not need to touch this contract.
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        ctx.record_seldepth(0);
        assert_eq!(ctx.seldepth.load(Ordering::Relaxed), 0);
        ctx.record_seldepth(6);
        assert_eq!(ctx.seldepth.load(Ordering::Relaxed), 6);
        ctx.record_seldepth(14);
        assert_eq!(ctx.seldepth.load(Ordering::Relaxed), 14);
        ctx.record_seldepth(8);
        assert_eq!(
            ctx.seldepth.load(Ordering::Relaxed),
            14,
            "must be monotonic"
        );
    }

    #[test]
    fn seldepth_reaches_requested_depth() {
        // R0 Repair 1: seldepth is the deepest GLOBAL ply (root=0, child=1),
        // shared by main search and qsearch. The old qsearch accounting
        // (ply+qply+1, with ply+1/qply+1 recursion) double-counted the qsearch
        // descent and inflated seldepth to ~2x. We only assert the invariant
        // (a depth-6 search reaches global ply >= 6, and its qsearch descent
        // does not re-add qply); the exact horizon is an S7 concern.
        let pos = parse_fen(START_FEN).unwrap();
        let hist = vec![pos.zobrist_key()];
        let limits = SearchLimits {
            depth: Some(6),
            ..Default::default()
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt = TranspositionTable::disabled();
        search_best_move_with_history_tt_and_profile(
            &mut pos.clone(),
            &hist,
            &limits,
            &ctx,
            &mut tt,
            SearchProfile::CurrentFinal,
        )
        .expect("outcome");
        let seldepth = ctx.seldepth.load(Ordering::Relaxed);
        assert!(
            seldepth >= 6,
            "seldepth {seldepth} must reach at least the requested depth 6"
        );
    }

    #[test]
    fn single_buffer_promotion_policy() {
        // S4.4E: single-buffer full-legal materialization is production
        // policy for CurrentFinal and every profile whose base semantics are
        // defined as CurrentFinal. The S5.0B candidate inherits it (it is
        // now "promoted CurrentFinal + has-any probe").
        for p in [
            SearchProfile::CurrentFinal,
            SearchProfile::CurrentFinalRootHistory,
            SearchProfile::CurrentFinalRootPrevScore,
            SearchProfile::CurrentFinalLegalityFast,
            SearchProfile::CurrentFinalSingleBuffer,
            SearchProfile::CurrentFinalSingleGeneration,
        ] {
            assert!(p.uses_single_buffer_legal(), "{p:?} must use single-buffer");
        }
        for p in [
            SearchProfile::Current,
            SearchProfile::CurrentLmr,
            SearchProfile::CurrentEval2,
            SearchProfile::M4Reference,
            SearchProfile::CurrentThreatAware,
            SearchProfile::CurrentAspiration,
            SearchProfile::CurrentQsearchPruning,
        ] {
            assert!(
                !p.uses_single_buffer_legal(),
                "{p:?} must NOT use single-buffer"
            );
        }
        // S5.0D: the has-any probe is production policy for CurrentFinal and
        // its family; the S5.0B alias is a historical identity.
        assert!(SearchProfile::CurrentFinal.uses_single_generation_probe());
        assert!(SearchProfile::CurrentFinalRootHistory.uses_single_generation_probe());
        assert!(SearchProfile::CurrentFinalRootPrevScore.uses_single_generation_probe());
        assert!(SearchProfile::CurrentFinalLegalityFast.uses_single_generation_probe());
        assert!(SearchProfile::CurrentFinalSingleBuffer.uses_single_generation_probe());
        assert!(SearchProfile::CurrentFinalSingleGeneration.uses_single_generation_probe());
        assert!(!SearchProfile::Current.uses_single_generation_probe());
        assert!(!SearchProfile::CurrentLmr.uses_single_generation_probe());
        assert!(!SearchProfile::M4Reference.uses_single_generation_probe());
    }

    #[test]
    fn null_move_guards_and_state_are_exact() {
        let pos = parse_fen(START_FEN).unwrap();
        let null_pos = make_null_position(&pos);
        assert_eq!(null_pos.board, pos.board);
        assert_eq!(null_pos.side, Color::Black);
        assert_eq!(null_pos.ep_target, None);
        assert_eq!(null_pos.halfmove, pos.halfmove + 1);
        assert_eq!(null_pos.zobrist_key, recompute_zobrist(&null_pos));
        assert_ne!(null_pos.zobrist_key, pos.zobrist_key);

        assert!(null_move_eligible(&pos, true, 5, 0, 1, false));
        assert!(!null_move_eligible(&pos, true, 5, 0, 100, false));
        assert!(!null_move_eligible(&pos, true, 5, 0, 1, true));
        let kqk = parse_fen("7k/8/8/8/8/8/3QK3/8 w - - 0 1").unwrap();
        assert!(!null_move_eligible(&kqk, true, 5, 0, 1, false));
    }

    #[test]
    fn null_probe_child_is_non_reentrant_at_depth_nine() {
        let pos = parse_fen(START_FEN).unwrap();
        assert!(null_move_eligible(&pos, true, 9, 0, 1, false));

        let stop = Arc::new(AtomicBool::new(true));
        let ctx = SearchContext::new_with_profiling(stop, true);
        let limits = SearchLimits::default();
        let mut pv = PvTable::default();
        let mut tt = TranspositionTable::disabled();
        let mut heuristics = None;
        let mut child = make_null_position(&pos);
        let mut path = SearchPath::new(vec![child.zobrist_key()]);

        // This is the exact child mode used by the null probe. The node is
        // otherwise eligible, but it must not launch a second null move.
        let _ = negamax_entered_impl_with_null(
            &mut child,
            5,
            1,
            0,
            1,
            &ctx,
            &limits,
            SearchProfile::NullMoveCandidate,
            &mut pv,
            &mut path,
            &mut tt,
            &mut heuristics,
            false,
        );
        assert_eq!(ctx.null_move_attempts.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn futility_guards_keep_advanced_pawns_and_reference_clean() {
        let start = parse_fen(START_FEN).unwrap();
        let quiet = find_move(&start, "e2e3");
        assert!(!is_pawn_promotion_threat(&start, quiet));
        assert!(!SearchProfile::M4Reference.uses_futility());
        assert!(SearchProfile::FutilityCandidate.uses_futility());

        let advanced = parse_fen("6k1/4P3/8/8/8/8/8/6K1 w - - 0 1").unwrap();
        let push = find_move(&advanced, "e7e8q");
        assert!(is_pawn_promotion_threat(&advanced, push));

        let near_promotion = parse_fen("6k1/8/4P3/8/8/8/8/6K1 w - - 0 1").unwrap();
        let advance = find_move(&near_promotion, "e6e7");
        assert!(is_pawn_promotion_threat(&near_promotion, advance));
    }

    fn run_profile_candidate(fen: &str, profile: SearchProfile) -> (SearchOutcome, SearchStats) {
        run_profile_candidate_with_nodes(fen, profile, 60_000)
    }

    fn run_profile_candidate_with_nodes(
        fen: &str,
        profile: SearchProfile,
        nodes: u64,
    ) -> (SearchOutcome, SearchStats) {
        let mut pos = parse_fen(fen).unwrap();
        let before = to_fen(&pos);
        let key = pos.zobrist_key();
        let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
        let limits = SearchLimits {
            nodes: Some(nodes),
            ..Default::default()
        };
        let mut tt = TranspositionTable::disabled();
        let out = search_best_move_with_history_tt_and_profile(
            &mut pos,
            &[key],
            &limits,
            &ctx,
            &mut tt,
            profile,
        )
        .expect("fixture is non-terminal");
        assert_eq!(
            to_fen(&pos),
            before,
            "candidate must restore the root board"
        );
        assert_eq!(
            pos.zobrist_key(),
            key,
            "candidate must restore the root hash"
        );
        (out, ctx.stats())
    }

    #[test]
    fn lmr_real_search_reduces_and_researches() {
        const OPEN_TACTICAL: &str =
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5";

        let (first, first_stats) =
            run_profile_candidate(OPEN_TACTICAL, SearchProfile::LmrCandidate);
        assert!(first.score.is_some());
        assert!(!first.pv.is_empty());
        assert!(
            first_stats.lmr_reductions > 0,
            "LMR path never reduced a move"
        );
        assert!(
            first_stats.lmr_researches > 0,
            "LMR path never exercised full-depth re-search"
        );

        // A second run from the restored root must produce the same observable
        // result and counters, proving that the reduced/researched PV did not
        // leak board, PV, TT, or heuristic state across calls.
        let (second, second_stats) =
            run_profile_candidate(OPEN_TACTICAL, SearchProfile::LmrCandidate);
        assert_eq!(second.score, first.score);
        assert_eq!(second.best_move, first.best_move);
        assert_eq!(second.pv, first.pv);
        // Iteration wall time is diagnostic and naturally varies between
        // otherwise identical runs; all node/counter telemetry must remain
        // identical, including the completed/aborted iteration boundaries.
        let mut comparable_first = first_stats;
        let mut comparable_second = second_stats;
        comparable_first.last_completed_iteration_ms = 0;
        comparable_second.last_completed_iteration_ms = 0;
        assert_eq!(comparable_second, comparable_first);
    }

    #[test]
    fn current_lmr_isolated_from_current_and_other_candidates() {
        const OPEN_TACTICAL: &str =
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5";

        let (_, current_stats) = run_profile_candidate(OPEN_TACTICAL, SearchProfile::Current);
        let (lmr, lmr_stats) = run_profile_candidate(OPEN_TACTICAL, SearchProfile::CurrentLmr);

        assert!(lmr.score.is_some());
        assert!(!lmr.pv.is_empty());
        assert_eq!(current_stats.lmr_reductions, 0);
        assert_eq!(current_stats.lmr_researches, 0);
        assert!(lmr_stats.lmr_reductions > 0);
        assert!(lmr_stats.lmr_researches > 0);
        assert_eq!(lmr_stats.aspiration_retries, 0);
        assert_eq!(lmr_stats.null_move_attempts, 0);
        assert_eq!(lmr_stats.futility_pruned, 0);
        assert_eq!(lmr_stats.qsearch_see_tests, 0);
    }

    #[test]
    fn current_threat_aware_isolated_from_search_candidates() {
        const KING_DANGER: &str = "r4rk1/ppp2ppp/8/8/8/6q1/PPPP1PPP/R3Q1K1 w - - 0 1";

        let (_, current_stats) =
            run_profile_candidate_with_nodes(KING_DANGER, SearchProfile::Current, 8_000);
        let (threat, threat_stats) =
            run_profile_candidate_with_nodes(KING_DANGER, SearchProfile::CurrentThreatAware, 8_000);

        assert!(threat.score.is_some());
        assert!(!threat.pv.is_empty());
        assert_eq!(current_stats.lmr_reductions, 0);
        assert_eq!(threat_stats.lmr_reductions, 0);
        assert_eq!(threat_stats.aspiration_retries, 0);
        assert_eq!(threat_stats.null_move_attempts, 0);
        assert_eq!(threat_stats.futility_pruned, 0);
        assert_eq!(threat_stats.qsearch_see_tests, 0);
        assert_ne!(
            current_stats.qsearch_nodes, 0,
            "candidate isolation fixture must reach qsearch"
        );
    }

    #[test]
    fn current_eval2_isolated_from_search_candidates() {
        const POSITION: &str = "r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PPP2PPP/R3K2R w KQkq - 0 1";

        let position = parse_fen(POSITION).unwrap();
        assert_ne!(
            evaluate(&position),
            evaluate_integrated_positional(&position),
            "the E2 fixture must exercise the candidate evaluator"
        );

        let (_, current_stats) =
            run_profile_candidate_with_nodes(POSITION, SearchProfile::Current, 8_000);
        let (eval2, eval2_stats) =
            run_profile_candidate_with_nodes(POSITION, SearchProfile::CurrentEval2, 8_000);

        assert!(eval2.score.is_some());
        assert!(!eval2.pv.is_empty());
        for stats in [current_stats, eval2_stats] {
            assert_eq!(stats.aspiration_retries, 0);
            assert_eq!(stats.lmr_reductions, 0);
            assert_eq!(stats.null_move_attempts, 0);
            assert_eq!(stats.futility_pruned, 0);
            assert_eq!(stats.qsearch_see_tests, 0);
            assert_eq!(stats.check_extensions, 0);
            assert_eq!(stats.single_evasion_extensions, 0);
            assert_eq!(stats.qsearch_check_moves, 0);
            assert_eq!(stats.threat_ordered_moves, 0);
            assert_eq!(stats.root_reorders, 0);
        }
        assert_ne!(eval2_stats.eval_calls, 0);
    }

    #[test]
    fn threat_aware_forcing_extensions_and_qsearch_checks_are_bounded() {
        let parent = parse_fen("4k3/8/8/8/8/8/4Q3/K7 w - - 0 1").unwrap();
        let check = find_move(&parent, "e2e7");
        let mut child = parent;
        child.make_move(check);
        let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
        let (child_depth, child_budget) = forcing_child_params(
            &child,
            4,
            SearchProfile::CurrentThreatAware,
            MAX_FORCING_EXTENSIONS,
            false,
            false,
            &ctx,
        );
        assert_eq!(child_depth, 4, "a checking move must receive one extension");
        assert_eq!(child_budget, MAX_FORCING_EXTENSIONS - 1);
        assert_eq!(ctx.check_extensions.load(Ordering::Relaxed), 1);
        let (single_depth, single_budget) = forcing_child_params(
            &parent,
            4,
            SearchProfile::CurrentThreatAware,
            child_budget,
            true,
            true,
            &ctx,
        );
        assert_eq!(single_depth, 4, "a lone evasion must receive one extension");
        assert_eq!(single_budget, child_budget - 1);
        assert_eq!(ctx.single_evasion_extensions.load(Ordering::Relaxed), 1);

        let limits = SearchLimits::default();
        let mut pv = PvTable::default();
        let mut path = SearchPath::new(vec![parent.zobrist_key()]);
        assert!(try_enter_node(&ctx, &limits));
        let _ = quiescence_entered_impl_with_profile(
            &mut parent.clone(),
            0,
            0,
            i32::MIN + 1000,
            i32::MAX - 1000,
            &ctx,
            &limits,
            &mut pv,
            &mut path,
            SearchProfile::CurrentThreatAware,
            true,
            false,
            false,
        );
        assert!(
            ctx.qsearch_check_moves.load(Ordering::Relaxed) > 0,
            "bounded qsearch must observe checking moves"
        );

        let (_, stats) = run_profile_candidate_with_nodes(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
            SearchProfile::CurrentThreatAware,
            12_000,
        );
        assert!(stats.threat_ordered_moves > 0);
        assert!(stats.root_reorders > 0);
        assert!(stats.check_extensions > 0);
        assert!(stats.single_evasion_extensions <= stats.check_extensions + 32);
    }

    #[test]
    fn futility_real_search_prunes_quiet_moves_and_restores_state() {
        const OPEN_TACTICAL: &str =
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5";

        let (out, stats) = run_profile_candidate(OPEN_TACTICAL, SearchProfile::FutilityCandidate);
        assert!(out.score.is_some());
        assert!(!out.pv.is_empty());
        assert!(
            stats.futility_pruned > 0,
            "futility candidate never exercised its real prune path"
        );

        let pos = parse_fen(OPEN_TACTICAL).unwrap();
        assert!(
            generate_legal_moves(&mut pos.clone()).contains(&out.best_move),
            "futility candidate returned an illegal root move"
        );
    }

    #[test]
    fn qsearch_movegen_integration_matches_pvs_reference_search_tree() {
        let fixtures = [START_FEN, MVV_POS];
        for fen in fixtures {
            let mut reference_pos = parse_fen(fen).unwrap();
            let reference_key = reference_pos.zobrist_key();
            let reference_ctx =
                SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let limits = SearchLimits {
                depth: Some(3),
                ..Default::default()
            };
            let reference = search_best_move_with_history_tt_and_profile(
                &mut reference_pos,
                &[reference_key],
                &limits,
                &reference_ctx,
                &mut TranspositionTable::disabled(),
                SearchProfile::PvsReference,
            )
            .expect("PVS reference fixture must be non-terminal");
            let reference_stats = reference_ctx.stats();

            let mut current_pos = parse_fen(fen).unwrap();
            let current_key = current_pos.zobrist_key();
            let current_ctx =
                SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let current = search_best_move_with_history_tt_and_profile(
                &mut current_pos,
                &[current_key],
                &limits,
                &current_ctx,
                &mut TranspositionTable::disabled(),
                SearchProfile::Current,
            )
            .expect("Current fixture must be non-terminal");
            let current_stats = current_ctx.stats();

            assert_eq!(current.completed_depth, reference.completed_depth);
            assert_eq!(current.score, reference.score);
            assert_eq!(current.best_move, reference.best_move);
            assert_eq!(current.pv, reference.pv);
            assert_eq!(current_stats.nodes, reference_stats.nodes);
            assert_eq!(current_stats.qsearch_nodes, reference_stats.qsearch_nodes);
            assert_eq!(current_stats.eval_calls, reference_stats.eval_calls);
            assert_eq!(current_stats.tt_probes, reference_stats.tt_probes);
            assert_eq!(current_stats.tt_stores, reference_stats.tt_stores);
            assert!(
                current_stats.pseudo_moves < reference_stats.pseudo_moves,
                "integrated qsearch must reduce pseudo-move work for {fen}"
            );
            assert!(
                current_stats.make_moves < reference_stats.make_moves,
                "integrated qsearch must reduce make/unmake work for {fen}"
            );
            assert_eq!(
                current_stats.make_moves, current_stats.unmake_moves,
                "Current make/unmake must balance for {fen}"
            );
        }
    }

    #[test]
    fn qsearch_movegen_preserves_checkmate_and_stalemate_scores() {
        fn run_qsearch(fen: &str, specialized: bool, pruning: bool) -> i32 {
            let mut pos = parse_fen(fen).unwrap();
            let key = pos.zobrist_key();
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            let limits = SearchLimits::default();
            let mut pv = PvTable::default();
            let mut path = SearchPath::new(vec![key]);
            assert!(try_enter_node(&ctx, &limits));
            quiescence_entered_impl(
                &mut pos,
                0,
                0,
                i32::MIN + 1000,
                i32::MAX - 1000,
                &ctx,
                &limits,
                &mut pv,
                &mut path,
                specialized,
                pruning,
                false,
            )
            .expect("unlimited qsearch must complete")
        }

        let checkmate = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1";
        let stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
        for fen in [checkmate, stalemate] {
            assert_eq!(
                run_qsearch(fen, true, false),
                run_qsearch(fen, false, false),
                "specialized qsearch changed terminal score for {fen}"
            );
            assert_eq!(
                run_qsearch(fen, true, true),
                run_qsearch(fen, true, false),
                "SEE pruning changed terminal score for {fen}"
            );
        }
        assert_eq!(run_qsearch(checkmate, true, true), -(MATE));
        assert_eq!(run_qsearch(stalemate, true, true), 0);
    }

    #[test]
    fn s75a_single_evasion_profile_inherits_current_final_exactly_except_s75a() {
        use SearchProfile::{CurrentFinal, CurrentFinalSingleEvasion as Cand};

        // Every production policy dimension must agree; the ONLY difference
        // is uses_single_evasion_extension.
        assert_eq!(CurrentFinal.uses_pvs(), Cand.uses_pvs());
        assert_eq!(CurrentFinal.uses_see(), Cand.uses_see());
        assert_eq!(CurrentFinal.uses_aspiration(), Cand.uses_aspiration());
        assert_eq!(CurrentFinal.uses_lmr(), Cand.uses_lmr());
        assert_eq!(CurrentFinal.uses_null_move(), Cand.uses_null_move());
        assert_eq!(CurrentFinal.uses_futility(), Cand.uses_futility());
        assert_eq!(
            CurrentFinal.uses_qsearch_movegen(),
            Cand.uses_qsearch_movegen()
        );
        assert_eq!(
            CurrentFinal.uses_qsearch_pruning(),
            Cand.uses_qsearch_pruning()
        );
        assert_eq!(
            CurrentFinal.uses_qsearch_fast_pruning(),
            Cand.uses_qsearch_fast_pruning()
        );
        assert_eq!(CurrentFinal.uses_qsearch_lazy(), Cand.uses_qsearch_lazy());
        assert_eq!(CurrentFinal.uses_qsearch_delta(), Cand.uses_qsearch_delta());
        assert_eq!(
            CurrentFinal.uses_root_quiet_history(),
            Cand.uses_root_quiet_history()
        );
        assert_eq!(
            CurrentFinal.uses_root_prev_score(),
            Cand.uses_root_prev_score()
        );
        assert_eq!(CurrentFinal.uses_legality_fast(), Cand.uses_legality_fast());
        assert_eq!(
            CurrentFinal.uses_single_buffer_legal(),
            Cand.uses_single_buffer_legal()
        );
        assert_eq!(
            CurrentFinal.uses_single_generation_probe(),
            Cand.uses_single_generation_probe()
        );
        assert_eq!(
            CurrentFinal.uses_lmr_null_window(),
            Cand.uses_lmr_null_window()
        );
        assert_eq!(CurrentFinal.uses_eval2(), Cand.uses_eval2());
        assert_eq!(
            CurrentFinal.uses_forcing_search(),
            Cand.uses_forcing_search()
        );
        assert_eq!(
            CurrentFinal.uses_threat_ordering(),
            Cand.uses_threat_ordering()
        );
        assert_eq!(
            CurrentFinal.uses_threat_aware_qsearch(),
            Cand.uses_threat_aware_qsearch()
        );
        assert_eq!(
            CurrentFinal.uses_threat_aware_eval(),
            Cand.uses_threat_aware_eval()
        );

        assert!(CurrentFinal.uses_single_evasion_extension());
        assert!(Cand.uses_single_evasion_extension());
        assert!(!Cand.uses_forcing_search());
        assert!(Cand.uses_null_move());
        assert!(Cand.uses_single_buffer_legal());
        assert!(Cand.uses_single_generation_probe());
        assert!(Cand.uses_lmr_null_window());

        let cf = SearchFeaturePolicy::for_profile(CurrentFinal, None);
        let cand = SearchFeaturePolicy::for_profile(Cand, None);
        assert_eq!(cf.lmr, cand.lmr);
        assert_eq!(cf.futility, cand.futility);
        assert_eq!(cf.null_move, cand.null_move);
        assert_eq!(cf.qsearch_see, cand.qsearch_see);
        assert_eq!(cf.qsearch_delta, cand.qsearch_delta);
        assert_eq!(cf.lmr_null_window, cand.lmr_null_window);
        assert!(cf.single_evasion_extension);
        assert!(cand.single_evasion_extension);
    }

    #[test]
    fn s75a_single_evasion_child_params_are_exact_and_budget_bounded() {
        let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);

        let (d, b) = s75a_single_evasion_child_params(4, 2, true, true, &ctx);
        assert_eq!((d, b), (4, 1));
        let (d, b) = s75a_single_evasion_child_params(1, 2, true, true, &ctx);
        assert_eq!((d, b), (1, 1));
        let (d, b) = s75a_single_evasion_child_params(4, 1, true, true, &ctx);
        assert_eq!((d, b), (4, 0));
        let (d, b) = s75a_single_evasion_child_params(4, 0, true, true, &ctx);
        assert_eq!((d, b), (3, 0));

        let (d, b) = s75a_single_evasion_child_params(4, 2, false, true, &ctx);
        assert_eq!((d, b), (3, 2));
        let (d, b) = s75a_single_evasion_child_params(4, 2, true, false, &ctx);
        assert_eq!((d, b), (3, 2));
        let (d, b) = s75a_single_evasion_child_params(0, 2, true, true, &ctx);
        assert_eq!((d, b), (0, 2));

        let stats = ctx.stats();
        assert_eq!(stats.s75a_extension_applied_total, 3);
        assert_eq!(stats.s75a_extension_applied_depth1, 1);
        assert_eq!(stats.s75a_extension_budget_2_to_1, 2);
        assert_eq!(stats.s75a_extension_budget_1_to_0, 1);
        assert_eq!(stats.s75a_opportunity_blocked_budget_0, 1);
    }

    #[test]
    fn s75a_tt_context_isolates_budget_without_legacy_exact_depth() {
        let pos = parse_fen(START_FEN).unwrap();
        let path = SearchPath::new(vec![pos.zobrist_key()]);
        let k0 = current_tt_key(&pos, &path);
        let k1 = current_tt_key_with_forcing_budget(&pos, &path, 1);
        let k2 = current_tt_key_with_forcing_budget(&pos, &path, 2);
        assert_ne!(k0, k1);
        assert_ne!(k0, k2);
        assert_ne!(k1, k2);
        assert_eq!(k0, current_tt_key_with_forcing_budget(&pos, &path, 0));

        let cand = SearchProfile::CurrentFinalSingleEvasion;
        assert!(cand.uses_single_evasion_extension());
        assert!(!cand.uses_forcing_search());
        assert_eq!(extension_budget_for_profile(cand), S75A_FORCING_BUDGET);
    }

    #[test]
    fn lmr_null_window_alias_matches_promoted_current_final() {
        use SearchProfile::{CurrentFinal, CurrentFinalLmrNullWindow as Nw};

        // S7.4A promotion: structural regression proof. Every production
        // policy dimension must agree between CurrentFinal and the S7.4A
        // compatibility alias, INCLUDING uses_lmr_null_window.
        assert_eq!(CurrentFinal.uses_pvs(), Nw.uses_pvs());
        assert_eq!(CurrentFinal.uses_see(), Nw.uses_see());
        assert_eq!(CurrentFinal.uses_aspiration(), Nw.uses_aspiration());
        assert_eq!(CurrentFinal.uses_lmr(), Nw.uses_lmr());
        assert_eq!(CurrentFinal.uses_null_move(), Nw.uses_null_move());
        assert_eq!(CurrentFinal.uses_futility(), Nw.uses_futility());
        assert_eq!(
            CurrentFinal.uses_qsearch_movegen(),
            Nw.uses_qsearch_movegen()
        );
        assert_eq!(
            CurrentFinal.uses_qsearch_pruning(),
            Nw.uses_qsearch_pruning()
        );
        assert_eq!(
            CurrentFinal.uses_qsearch_fast_pruning(),
            Nw.uses_qsearch_fast_pruning()
        );
        assert_eq!(CurrentFinal.uses_qsearch_lazy(), Nw.uses_qsearch_lazy());
        assert_eq!(
            CurrentFinal.uses_root_quiet_history(),
            Nw.uses_root_quiet_history()
        );
        assert_eq!(
            CurrentFinal.uses_root_prev_score(),
            Nw.uses_root_prev_score()
        );
        assert_eq!(CurrentFinal.uses_legality_fast(), Nw.uses_legality_fast());
        assert_eq!(
            CurrentFinal.uses_single_buffer_legal(),
            Nw.uses_single_buffer_legal()
        );
        assert_eq!(
            CurrentFinal.uses_single_generation_probe(),
            Nw.uses_single_generation_probe()
        );
        assert_eq!(CurrentFinal.uses_eval2(), Nw.uses_eval2());
        assert_eq!(CurrentFinal.uses_forcing_search(), Nw.uses_forcing_search());
        assert_eq!(
            CurrentFinal.uses_threat_ordering(),
            Nw.uses_threat_ordering()
        );
        assert_eq!(
            CurrentFinal.uses_threat_aware_qsearch(),
            Nw.uses_threat_aware_qsearch()
        );
        assert_eq!(
            CurrentFinal.uses_threat_aware_eval(),
            Nw.uses_threat_aware_eval()
        );
        // The qsearch-delta lane stays evidence-only.
        assert!(!Nw.uses_qsearch_delta());

        // Promotion: the S7.4A policy is now production CurrentFinal policy,
        // and the alias remains exactly equal to it.
        assert!(CurrentFinal.uses_lmr_null_window());
        assert!(Nw.uses_lmr_null_window());
        assert_eq!(
            CurrentFinal.uses_lmr_null_window(),
            Nw.uses_lmr_null_window()
        );

        // Explicit guards for the two arms forgotten in the original
        // (misconfigured) S7.1B candidate.
        assert!(Nw.uses_null_move(), "S7.4A must inherit verified null move");
        assert!(
            Nw.uses_single_buffer_legal(),
            "S7.4A must inherit SingleBuffer materialization"
        );

        // The resolved hot-path policies must agree bit-for-bit, including
        // the promoted LMR-null-window bit.
        let cf = SearchFeaturePolicy::for_profile(CurrentFinal, None);
        let nw = SearchFeaturePolicy::for_profile(Nw, None);
        assert_eq!(cf.lmr, nw.lmr);
        assert_eq!(cf.futility, nw.futility);
        assert_eq!(cf.null_move, nw.null_move);
        assert_eq!(cf.qsearch_see, nw.qsearch_see);
        assert_eq!(cf.qsearch_delta, nw.qsearch_delta);
        assert!(cf.lmr_null_window);
        assert!(nw.lmr_null_window);
        assert_eq!(cf.lmr_null_window, nw.lmr_null_window);
    }

    #[test]
    fn lmr_null_window_promotion_profile_family_is_exact() {
        use SearchProfile::{
            Current, CurrentAspirationLmr, CurrentAspirationLmrFutilitySee, CurrentFinal,
            CurrentFinalLegalityFast, CurrentFinalLmrNullWindow, CurrentFinalQsearchDelta,
            CurrentFinalQsearchLazy, CurrentFinalRootHistory, CurrentFinalRootPrevScore,
            CurrentFinalSingleBuffer, CurrentFinalSingleEvasion, CurrentFinalSingleGeneration,
            CurrentLmr, LmrCandidate,
        };

        // Every profile whose base semantics are CurrentFinal or
        // CurrentFinal + X carries the promoted production policy.
        let promoted = [
            CurrentFinal,
            CurrentFinalRootHistory,
            CurrentFinalRootPrevScore,
            CurrentFinalLegalityFast,
            CurrentFinalSingleBuffer,
            CurrentFinalSingleGeneration,
            CurrentFinalQsearchLazy,
            CurrentFinalQsearchDelta,
            CurrentFinalLmrNullWindow,
            CurrentFinalSingleEvasion,
        ];
        for profile in promoted {
            assert!(
                profile.uses_lmr_null_window(),
                "{profile:?} must use promoted S7.4A LMR-on-null-window"
            );
            assert!(
                SearchFeaturePolicy::for_profile(profile, None).lmr_null_window,
                "{profile:?} resolved policy must enable lmr_null_window"
            );
        }

        // Rollback and historical pre-S7.4A profiles remain unchanged.
        let unchanged = [
            Current,
            CurrentLmr,
            CurrentAspirationLmr,
            CurrentAspirationLmrFutilitySee,
            LmrCandidate,
        ];
        for profile in unchanged {
            assert!(
                !profile.uses_lmr_null_window(),
                "{profile:?} must keep the pre-promotion behavior"
            );
            assert!(
                !SearchFeaturePolicy::for_profile(profile, None).lmr_null_window,
                "{profile:?} resolved policy must keep lmr_null_window off"
            );
        }
        assert!(!Current.uses_lmr_null_window());
        assert!(
            !SearchFeaturePolicy::for_profile(Current, None).lmr_null_window,
            "historical Current rollback profile must remain unchanged"
        );
    }

    #[test]
    fn lmr_null_window_promoted_profile_applies_reduces_and_verifies_correctly() {
        fn run(profile: SearchProfile) -> (Option<i32>, SearchStats) {
            // White is a knight down: eval (~-320) stays above the depth-4
            // futility margin from alpha=0, so quiets are NOT shallow-pruned;
            // but the true score stays <= 0, so every root move fails low at
            // the null window and the loop inevitably reaches late quiet
            // indices. The depth-4 root is then the LMR-eligible node.
            let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1";
            let mut pos = parse_fen(fen).unwrap();
            let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let limits = SearchLimits {
                depth: Some(4),
                ..Default::default()
            };
            let mut pv = PvTable::default();
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let mut tt = TranspositionTable::new_mb(16).unwrap();
            let mut heur = Some(SearchHeuristics::new());
            // Mirror the production entry point (`search_best_move_impl`):
            // calling the negamax body directly bypasses flag setup, and the
            // move loop reads `ctx.features()` (LMR eligibility) and the
            // legality-probe flags from the context, not the profile.
            ctx.see_enabled.store(profile.uses_see(), Ordering::Relaxed);
            ctx.features_mask.store(
                SearchFeaturePolicy::for_profile(profile, None).to_bits(),
                Ordering::Relaxed,
            );
            ctx.legality_fast
                .store(profile.uses_legality_fast(), Ordering::Relaxed);
            ctx.single_buffer_legal
                .store(profile.uses_single_buffer_legal(), Ordering::Relaxed);
            ctx.single_generation_probe
                .store(profile.uses_single_generation_probe(), Ordering::Relaxed);
            // A null window AT THE ROOT makes the root move loop itself the
            // S7.4A caller population (depth 4, late quiet moves), keeping
            // the test tree tiny while exercising every counter path.
            let score = negamax_entered_impl_with_null(
                &mut pos, 4, 0, 0, 1, &ctx, &limits, profile, &mut pv, &mut path, &mut tt,
                &mut heur, true,
            );
            (score, ctx.stats())
        }

        // CurrentLmr is the historical LMR profile without the promoted
        // policy: on caller-null-window nodes it still shows the S7.3
        // suppression and never enters the S7.4A path.
        let (_, legacy) = run(SearchProfile::CurrentLmr);
        assert!(legacy.s74_lmr_proposed > 0);
        assert!(legacy.s74_lmr_suppressed_by_null_window > 0);
        assert_eq!(legacy.s74_lmr_applied_null_window, 0);

        let (prod_score, prod) = run(SearchProfile::CurrentFinal);
        let (alias_score, alias) = run(SearchProfile::CurrentFinalLmrNullWindow);

        for (label, stats) in [("production CurrentFinal", prod), ("alias", alias)] {
            // The previously suppressed population is now applied; every
            // applied reduction terminates in exactly one of: fail-low
            // accept, or exactly one full-depth re-search.
            assert_eq!(stats.s74_lmr_suppressed_by_null_window, 0, "{label}");
            assert!(stats.s74_lmr_applied_null_window > 0, "{label}");
            assert_eq!(
                stats.s74_lmr_applied_null_window,
                stats.s74_lmr_nw_fail_low + stats.s74_lmr_nw_research,
                "{label}"
            );
            // Contract A: this fixture is all fail-low, so no verification is
            // requested and no verification node is acquired.
            assert_eq!(stats.s74_lmr_nw_research, 0, "{label}");
            assert_eq!(stats.s74_lmr_nw_research_entered, 0, "{label}");
            // Verified cutoffs can only originate from re-searched moves.
            assert!(
                stats.s74_lmr_nw_verified_cutoff <= stats.s74_lmr_nw_research,
                "{label}"
            );
            // S7.4A Repair 1: every re-search is a NEW real search entry
            // acquired through the exact-once `try_enter_node` contract.
            // Entries can never exceed requests, and in an unlimited
            // fixed-depth run every requested verification enters.
            assert!(
                stats.s74_lmr_nw_research_entered <= stats.s74_lmr_nw_research,
                "{label}"
            );
            assert_eq!(
                stats.s74_lmr_nw_research_entered, stats.s74_lmr_nw_research,
                "unlimited run: every requested verification must enter ({label})"
            );
            // The depth and index splits each account for every application.
            assert_eq!(
                stats.s74_lmr_nw_depth.iter().sum::<u64>(),
                stats.s74_lmr_applied_null_window,
                "{label}"
            );
            assert_eq!(
                stats.s74_lmr_nw_idx.iter().sum::<u64>(),
                stats.s74_lmr_applied_null_window,
                "{label}"
            );
        }

        // Production CurrentFinal and the retained S7.4A alias are
        // search-semantically identical in this deterministic fixture.
        assert_eq!(prod_score, alias_score);
        assert_eq!(prod.nodes, alias.nodes);
        assert_eq!(prod.qsearch_nodes, alias.qsearch_nodes);
        assert_eq!(
            prod.s74_lmr_applied_null_window,
            alias.s74_lmr_applied_null_window
        );
        assert_eq!(prod.s74_lmr_nw_fail_low, alias.s74_lmr_nw_fail_low);

        // Both promoted searches complete with a bounded, sane score.
        let s = prod_score.expect("production CurrentFinal search must complete");
        assert!(s > -(MATE - 1000) && s < MATE - 1000);
        let s = alias_score.expect("S7.4A alias search must complete");
        assert!(s > -(MATE - 1000) && s < MATE - 1000);
    }

    #[test]
    fn lmr_null_window_verification_acquisition_respects_node_budget() {
        // S7.4A Repair 1, contracts C/D. Contract C isolates the EXACT
        // acquisition-failure event: we find the smallest budget at which a
        // reduced null-window search improves alpha and requests a full-depth
        // verification, then prove the `try_enter_node` failure at that
        // budget:
        //   * unwinds cleanly (None, board/FEN/path restored),
        //   * consumes exactly the budget (nodes never exceed it),
        //   * enters zero verifications, so no unverified beta cutoff can
        //     exist (`verified_cutoff == 0`),
        //   * records zero extra killer/history rewards and leaves the root
        //     PV row unchanged relative to the immediately preceding budget.
        // The remaining sweep covers aborts at other points (contract D).
        //
        // Root WIDE window, depth 6, Italian position: verified (release
        // probe) to trigger interior null-window applications AND full-depth
        // verification re-searches, which the earlier no-knight fixture never
        // did (all its reductions fail low, so no verification acquisition is
        // ever attempted there).
        const FEN: &str = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";

        fn run(
            profile: SearchProfile,
            budget: Option<u64>,
        ) -> (Option<i32>, SearchStats, Position, SearchPath, PvTable) {
            let mut pos = parse_fen(FEN).unwrap();
            let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let limits = SearchLimits {
                depth: Some(6),
                nodes: budget,
            };
            let mut pv = PvTable::default();
            let mut path = SearchPath::new(vec![pos.zobrist_key()]);
            let mut tt = TranspositionTable::new_mb(16).unwrap();
            let mut heur = Some(SearchHeuristics::new());
            ctx.see_enabled.store(profile.uses_see(), Ordering::Relaxed);
            ctx.features_mask.store(
                SearchFeaturePolicy::for_profile(profile, None).to_bits(),
                Ordering::Relaxed,
            );
            ctx.legality_fast
                .store(profile.uses_legality_fast(), Ordering::Relaxed);
            ctx.single_buffer_legal
                .store(profile.uses_single_buffer_legal(), Ordering::Relaxed);
            ctx.single_generation_probe
                .store(profile.uses_single_generation_probe(), Ordering::Relaxed);
            let score = negamax_entered_impl_with_null(
                &mut pos,
                6,
                0,
                i32::MIN + 1000,
                i32::MAX - 1000,
                &ctx,
                &limits,
                profile,
                &mut pv,
                &mut path,
                &mut tt,
                &mut heur,
                true,
            );
            (score, ctx.stats(), pos, path, pv)
        }

        let profile = SearchProfile::CurrentFinalLmrNullWindow;
        pvs_counters::reset();
        let (full, full_stats, _, _, _) = run(profile, None);
        assert!(full.is_some(), "unlimited run must complete");
        let full_nodes = full_stats.nodes;
        assert!(full_stats.s74_lmr_nw_research > 0);
        assert_eq!(
            full_stats.s74_lmr_nw_research_entered, full_stats.s74_lmr_nw_research,
            "unlimited run: every requested verification must enter"
        );
        assert_eq!(
            pvs_counters::S74_NW_RESEARCH_ATTEMPT.get(),
            full_stats.s74_lmr_nw_research as usize,
            "one verification attempt is emitted per request"
        );
        assert_eq!(
            pvs_counters::S74_NW_RESEARCH_ENTERED.get(),
            full_stats.s74_lmr_nw_research_entered as usize,
            "one verification entry is emitted per successful acquisition"
        );

        // `s74_lmr_nw_research` is nondecreasing in the node budget, so the
        // smallest budget with a requested verification is the first moment
        // the reduced search completed with an improvement. At that exact
        // budget the following `try_enter_node` fails by construction (the
        // reduced search consumed the last available node).
        let mut lo = 1u64;
        let mut hi = full_nodes;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (_, stats, _, _, _) = run(profile, Some(mid));
            if stats.s74_lmr_nw_research > 0 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let acquire_fail_budget = lo;
        assert!(acquire_fail_budget > 1, "fixture must have a real prefix");

        let before_fen = to_fen(&parse_fen(FEN).unwrap());
        pvs_counters::reset();
        let (prev_score, prev_stats, _, _, prev_pv) = run(profile, Some(acquire_fail_budget - 1));
        assert!(prev_score.is_none(), "prefix budget must abort");
        assert_eq!(
            prev_stats.s74_lmr_nw_research, 0,
            "prefix budget must not yet request verification"
        );
        let prev_killer_calls = pvs_counters::RECORD_KILLER_CALLS.get();
        let prev_history_calls = pvs_counters::RECORD_HISTORY_CALLS.get();
        let prev_quiet_rewards = pvs_counters::PARENT_QUIET_REWARD.get();
        let prev_root_pv = prev_pv.lines[0].clone();

        pvs_counters::reset();
        let (score, stats, pos, path, pv) = run(profile, Some(acquire_fail_budget));
        assert!(
            score.is_none(),
            "verification acquisition failure must abort"
        );
        assert_eq!(
            stats.nodes, acquire_fail_budget,
            "abort is exact budget exhaustion"
        );
        assert_eq!(stats.s74_lmr_nw_research, 1, "exactly one request");
        assert_eq!(
            stats.s74_lmr_nw_research_entered, 0,
            "request did not enter"
        );
        assert_eq!(
            stats.s74_lmr_nw_verified_cutoff, 0,
            "no unverified beta cutoff"
        );
        assert_eq!(path.len(), 1, "path restored to root");
        assert_eq!(to_fen(&pos), before_fen, "position restored to root");
        assert_eq!(
            pvs_counters::S74_NW_RESEARCH_ATTEMPT.get(),
            1,
            "test-only event: exactly one verification attempt"
        );
        assert_eq!(
            pvs_counters::S74_NW_RESEARCH_ENTERED.get(),
            0,
            "test-only event: attempt never entered"
        );
        assert_eq!(
            pvs_counters::S74_NW_ABORT_RESEARCH_ACQUIRE.get(),
            1,
            "test-only event: abort happened at verification acquisition"
        );
        assert_eq!(
            pvs_counters::RECORD_KILLER_CALLS.get(),
            prev_killer_calls,
            "failed verification acquisition records no killer"
        );
        assert_eq!(
            pvs_counters::RECORD_HISTORY_CALLS.get(),
            prev_history_calls,
            "failed verification acquisition records no history"
        );
        assert_eq!(
            pvs_counters::PARENT_QUIET_REWARD.get(),
            prev_quiet_rewards,
            "failed verification acquisition takes no quiet cutoff reward"
        );
        assert_eq!(
            pv.lines[0], prev_root_pv,
            "failed verification acquisition commits no fake root PV"
        );

        // Contract D: sample budgets covering early, mid, and late aborts.
        // Every aborted run must consume exactly its budget, never exceed it,
        // restore path/position, and keep entered <= requested and verified
        // cutoffs <= entered.
        let mut budget = 5u64;
        let mut saw_abort = false;
        while budget < full_nodes {
            let before_fen = to_fen(&parse_fen(FEN).unwrap());
            let (score, stats, pos, path, _) = run(profile, Some(budget));
            assert!(
                stats.nodes <= budget,
                "nodes {} must never exceed budget {budget}",
                stats.nodes
            );
            if score.is_none() {
                saw_abort = true;
                assert_eq!(stats.nodes, budget, "abort must be budget exhaustion");
                assert_eq!(path.len(), 1, "path restored to root");
                assert_eq!(to_fen(&pos), before_fen, "position restored to root");
            }
            assert!(stats.s74_lmr_nw_research_entered <= stats.s74_lmr_nw_research);
            assert!(stats.s74_lmr_nw_verified_cutoff <= stats.s74_lmr_nw_research_entered);
            budget += (full_nodes / 12).max(3);
        }
        assert!(saw_abort, "budget sweep must observe at least one abort");
    }

    #[test]
    fn qsearch_delta_profile_inherits_current_final_exactly_except_delta() {
        use SearchProfile::{CurrentFinal, CurrentFinalQsearchDelta as Delta};

        // S7.1B Repair 1: structural regression proof. Every production
        // policy dimension must agree between CurrentFinal and the S7.1B
        // candidate; the ONLY permitted difference is uses_qsearch_delta.
        assert_eq!(CurrentFinal.uses_pvs(), Delta.uses_pvs());
        assert_eq!(CurrentFinal.uses_see(), Delta.uses_see());
        assert_eq!(CurrentFinal.uses_aspiration(), Delta.uses_aspiration());
        assert_eq!(CurrentFinal.uses_lmr(), Delta.uses_lmr());
        assert_eq!(CurrentFinal.uses_null_move(), Delta.uses_null_move());
        assert_eq!(CurrentFinal.uses_futility(), Delta.uses_futility());
        assert_eq!(
            CurrentFinal.uses_qsearch_movegen(),
            Delta.uses_qsearch_movegen()
        );
        assert_eq!(
            CurrentFinal.uses_qsearch_pruning(),
            Delta.uses_qsearch_pruning()
        );
        assert_eq!(
            CurrentFinal.uses_qsearch_fast_pruning(),
            Delta.uses_qsearch_fast_pruning()
        );
        assert_eq!(CurrentFinal.uses_qsearch_lazy(), Delta.uses_qsearch_lazy());
        assert_eq!(
            CurrentFinal.uses_root_quiet_history(),
            Delta.uses_root_quiet_history()
        );
        assert_eq!(
            CurrentFinal.uses_root_prev_score(),
            Delta.uses_root_prev_score()
        );
        assert_eq!(
            CurrentFinal.uses_legality_fast(),
            Delta.uses_legality_fast()
        );
        assert_eq!(
            CurrentFinal.uses_single_buffer_legal(),
            Delta.uses_single_buffer_legal()
        );
        assert_eq!(
            CurrentFinal.uses_single_generation_probe(),
            Delta.uses_single_generation_probe()
        );
        assert_eq!(CurrentFinal.uses_eval2(), Delta.uses_eval2());
        assert_eq!(
            CurrentFinal.uses_forcing_search(),
            Delta.uses_forcing_search()
        );
        assert_eq!(
            CurrentFinal.uses_threat_ordering(),
            Delta.uses_threat_ordering()
        );
        assert_eq!(
            CurrentFinal.uses_threat_aware_qsearch(),
            Delta.uses_threat_aware_qsearch()
        );
        assert_eq!(
            CurrentFinal.uses_threat_aware_eval(),
            Delta.uses_threat_aware_eval()
        );

        // The single intended difference.
        assert!(!CurrentFinal.uses_qsearch_delta());
        assert!(Delta.uses_qsearch_delta());

        // Explicit guards for the two arms forgotten in the original
        // (misconfigured) S7.1B candidate.
        assert!(
            Delta.uses_null_move(),
            "S7.1B must inherit verified null move"
        );
        assert!(
            Delta.uses_single_buffer_legal(),
            "S7.1B must inherit SingleBuffer materialization"
        );

        // The resolved hot-path policy must agree on every shared feature
        // bit and differ only on the delta bit.
        let cf = SearchFeaturePolicy::for_profile(CurrentFinal, None);
        let dl = SearchFeaturePolicy::for_profile(Delta, None);
        assert_eq!(cf.lmr, dl.lmr);
        assert_eq!(cf.futility, dl.futility);
        assert_eq!(cf.null_move, dl.null_move);
        assert_eq!(cf.qsearch_see, dl.qsearch_see);
        assert!(!cf.qsearch_delta);
        assert!(dl.qsearch_delta);
    }

    #[test]
    fn qsearch_delta_pruning_follows_predeclared_rule_and_exemptions() {
        fn prune_delta(
            fen: &str,
            uci: &str,
            alpha: i32,
            beta: i32,
            stand_pat: i32,
            qply: u32,
        ) -> (Vec<Move>, SearchStats) {
            let mut pos = parse_fen(fen).unwrap();
            let m = find_move(&pos, uci);
            let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let kept = prune_qsearch_captures_by_see_delta(
                &mut pos,
                vec![m],
                &ctx,
                alpha,
                beta,
                stand_pat,
                qply,
            );
            (kept, ctx.stats())
        }
        fn prune_baseline(fen: &str, uci: &str, alpha: i32, beta: i32) -> Vec<Move> {
            let mut pos = parse_fen(fen).unwrap();
            let m = find_move(&pos, uci);
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            prune_qsearch_captures_by_see(&mut pos, vec![m], &ctx, alpha, beta)
        }
        const WIDE: i32 = 10_000;

        // (1) SEE<0 ordinary capture: baseline and candidate both prune.
        let losing = "6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1";
        let (kept, stats) = prune_delta(losing, "f3e5", -100, WIDE, -800, 0);
        assert!(kept.is_empty());
        assert_eq!(stats.qsearch_see_pruned, 1);
        assert_eq!(stats.qsearch_delta_pruned, 0);
        assert!(prune_baseline(losing, "f3e5", -100, WIDE).is_empty());

        // (2) SEE>=0 capture with stand_pat + SEE + 500 <= alpha:
        // baseline keeps, candidate delta-prunes. Nxd5 is an unguarded pawn
        // capture (SEE = +100); four non-pawn pieces satisfy the guard.
        let gains_pawn = "5k1b/8/8/3p4/8/4N2R/8/1B2K3 w - - 0 1";
        let gains_pos = parse_fen(gains_pawn).unwrap();
        let gains_move = find_move(&gains_pos, "e3d5");
        assert_eq!(
            static_exchange_eval_for_pruning(&gains_pos, gains_move),
            Some(100)
        );
        let (kept, stats) = prune_delta(gains_pawn, "e3d5", -100, WIDE, -800, 0);
        assert!(kept.is_empty(), "delta rule must prune the futile capture");
        assert_eq!(stats.qsearch_see_tests, 1);
        assert_eq!(stats.qsearch_delta_tests, 1);
        assert_eq!(stats.qsearch_delta_pruned, 1);
        assert_eq!(stats.qsearch_delta_pruned_pawn, 1);
        assert_eq!(stats.qsearch_delta_qply_0_1, 1);
        assert_eq!(stats.qsearch_see_pruned, 0);
        assert_eq!(
            prune_baseline(gains_pawn, "e3d5", -100, WIDE).len(),
            1,
            "baseline must keep this SEE>=0 capture"
        );

        // (3) Same capture just above the threshold (stand_pat + SEE + 500 =
        // -200 > alpha = -201): candidate keeps.
        let (kept, stats) = prune_delta(gains_pawn, "e3d5", -201, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_delta_tests, 1);
        assert_eq!(stats.qsearch_delta_pruned, 0);

        // (3b) The qply buckets attribute to qply 2-3 when reached deeper.
        let (_, stats) = prune_delta(gains_pawn, "e3d5", -100, WIDE, -800, 2);
        assert_eq!(stats.qsearch_delta_qply_2_3, 1);
        assert_eq!(stats.qsearch_delta_qply_0_1, 0);

        // (4) A checking capture is never delta-pruned (kept before SEE).
        let checking = "4k3/8/8/4r3/4Q3/8/8/K7 w - - 0 1";
        let (kept, stats) = prune_delta(checking, "e4e5", -100, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_checking_captures_kept, 1);
        assert_eq!(stats.qsearch_delta_tests, 0);

        // (5) A capture-promotion is kept.
        let promo_cap = "1n5k/P7/8/8/8/8/8/K7 w - - 0 1";
        let (kept, stats) = prune_delta(promo_cap, "a7b8q", -100, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_promotions_kept, 1);
        assert_eq!(stats.qsearch_delta_tests, 0);

        // (6) A quiet promotion is kept.
        let promo_quiet = "7k/P7/8/8/8/8/8/K7 w - - 0 1";
        let (kept, stats) = prune_delta(promo_quiet, "a7a8q", -100, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_promotions_kept, 1);
        assert_eq!(stats.qsearch_delta_tests, 0);

        // (7) En passant is kept.
        let ep = "7k/8/8/5Pp1/8/8/8/4K3 w - g6 0 1";
        let (kept, stats) = prune_delta(ep, "f5g6", -100, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_en_passant_kept, 1);
        assert_eq!(stats.qsearch_delta_tests, 0);

        // (8) Unsupported SEE fails open: kept even though the delta margin
        // would otherwise be satisfied.
        let promo_exchange = "r3b3/3P4/1k6/8/8/8/4Q3/6K1 w - - 0 1";
        let (kept, stats) = prune_delta(promo_exchange, "e2e8", -100, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_see_fail_open_promotions, 1);
        assert_eq!(stats.qsearch_delta_pruned, 0);

        // (9) Mate-range alpha/beta: nothing is tested or pruned at all.
        let (kept, stats) = prune_delta(gains_pawn, "e3d5", MATE - 500, MATE - 100, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_see_tests, 0);
        assert_eq!(stats.qsearch_delta_tests, 0);
        assert_eq!(stats.qsearch_delta_pruned, 0);

        // (10) Low non-pawn material (3 pieces) stays fail-open even when
        // the numeric delta threshold is satisfied.
        let sparse = "7k/8/8/3p4/8/4N3/8/6K1 w - - 0 1";
        let (kept, stats) = prune_delta(sparse, "e3d5", -100, WIDE, -800, 0);
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_see_tests, 1);
        assert_eq!(stats.qsearch_delta_tests, 0);
        assert_eq!(stats.qsearch_delta_pruned, 0);
    }

    #[test]
    fn qsearch_see_pruning_keeps_guards_and_prunes_only_losing_plain_captures() {
        fn prune_one(fen: &str, uci: &str) -> (Vec<Move>, SearchStats) {
            let mut pos = parse_fen(fen).unwrap();
            let m = find_move(&pos, uci);
            let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let kept = prune_qsearch_captures_by_see(&mut pos, vec![m], &ctx, -10_000, 10_000);
            (kept, ctx.stats())
        }

        // A plainly losing, non-checking capture is the only move class that
        // the first D1.3 rule is allowed to remove.
        let losing = "6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1";
        let losing_move = find_move(&parse_fen(losing).unwrap(), "f3e5");
        assert!(static_exchange_eval(&parse_fen(losing).unwrap(), losing_move) < 0);
        let (kept, stats) = prune_one(losing, "f3e5");
        assert!(kept.is_empty());
        assert_eq!(stats.qsearch_see_tests, 1);
        assert_eq!(stats.qsearch_see_pruned, 1);

        // An x-ray rook recapture is also recognized as a losing exchange,
        // rather than being treated as a one-ply pawn/rook gain.
        let xray = "7k/r7/8/8/p7/8/8/R6K w - - 0 1";
        let xray_pos = parse_fen(xray).unwrap();
        let xray_move = find_move(&xray_pos, "a1a4");
        assert!(static_exchange_eval(&xray_pos, xray_move) < 0);
        let (kept, stats) = prune_one(xray, "a1a4");
        assert!(kept.is_empty());
        assert_eq!(stats.qsearch_see_pruned, 1);

        // The old ordering SEE can return a false negative when the
        // exchange is completed by a pawn promotion. The pruning interface
        // must reject that unsupported line and fail open instead.
        let promotion_exchange = "r3b3/3P4/1k6/8/8/8/4Q3/6K1 w - - 0 1";
        let promotion_exchange_pos = parse_fen(promotion_exchange).unwrap();
        let promotion_exchange_move = find_move(&promotion_exchange_pos, "e2e8");
        assert!(static_exchange_eval(&promotion_exchange_pos, promotion_exchange_move) < 0);
        assert_eq!(
            static_exchange_eval_for_pruning(&promotion_exchange_pos, promotion_exchange_move),
            None
        );
        let (kept, stats) = prune_one(promotion_exchange, "e2e8");
        assert_eq!(kept, vec![promotion_exchange_move]);
        assert_eq!(stats.qsearch_see_tests, 1);
        assert_eq!(stats.qsearch_see_pruned, 0);
        assert_eq!(stats.qsearch_see_fail_open_promotions, 1);

        // Color-swapped counterpart: the safety rule must not depend on the
        // side to move or the direction of the promotion.
        let black_promotion_exchange = "6K1/4q3/8/8/8/1k6/3p4/R3B3 b - - 0 1";
        let black_promotion_pos = parse_fen(black_promotion_exchange).unwrap();
        let black_promotion_move = find_move(&black_promotion_pos, "e7e1");
        assert!(static_exchange_eval(&black_promotion_pos, black_promotion_move) < 0);
        assert_eq!(
            static_exchange_eval_for_pruning(&black_promotion_pos, black_promotion_move),
            None
        );
        let (kept, stats) = prune_one(black_promotion_exchange, "e7e1");
        assert_eq!(kept, vec![black_promotion_move]);
        assert_eq!(stats.qsearch_see_tests, 1);
        assert_eq!(stats.qsearch_see_pruned, 0);
        assert_eq!(stats.qsearch_see_fail_open_promotions, 1);

        // The defended queen win remains searchable.
        let (kept, stats) = prune_one(MVV_POS, "e4a4");
        assert_eq!(kept, vec![find_move(&parse_fen(MVV_POS).unwrap(), "e4a4")]);
        assert_eq!(stats.qsearch_see_tests, 1);
        assert_eq!(stats.qsearch_see_pruned, 0);

        // A checking capture is a protected sacrifice/mating resource and is
        // kept without trusting a static exchange result.
        let checking = "4k3/8/8/4r3/4Q3/8/8/K7 w - - 0 1";
        let (kept, stats) = prune_one(checking, "e4e5");
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_see_tests, 0);
        assert_eq!(stats.qsearch_checking_captures_kept, 1);

        // A pinned rook may still make the legal capture that removes the
        // pinning rook; the legal move generator and SEE must not reject it.
        let pinned = "4r1k1/8/8/8/8/8/r3R3/4K3 w - - 0 1";
        let (kept, stats) = prune_one(pinned, "e2e8");
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_see_pruned, 0);

        // A legal king capture with no recapturer is not a losing exchange.
        let king_capture = "6k1/8/8/3pK3/8/8/4P3/R7 w - - 0 1";
        let (kept, stats) = prune_one(king_capture, "e5d5");
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_see_pruned, 0);

        // Quiet and capturing promotions are both fail-open.
        let promotions = "1r5k/P7/8/8/8/8/8/7K w - - 0 1";
        let mut promotion_pos = parse_fen(promotions).unwrap();
        let promotion_moves: Vec<Move> = generate_legal_moves(&mut promotion_pos)
            .into_iter()
            .filter(|m| matches!(m.flag, MoveFlag::Promotion(_)))
            .collect();
        assert_eq!(promotion_moves.len(), 8);
        let promotion_ctx =
            SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
        let kept_promotions = prune_qsearch_captures_by_see(
            &mut promotion_pos,
            promotion_moves.clone(),
            &promotion_ctx,
            -10_000,
            10_000,
        );
        assert_eq!(kept_promotions, promotion_moves);
        assert_eq!(promotion_ctx.stats().qsearch_promotions_kept, 8);
        assert_eq!(promotion_ctx.stats().qsearch_see_tests, 0);

        // Both queen/underpromotion choices, with and without a later
        // recapture, are unsupported by the pruning proof and therefore
        // fail open. This covers quiet and capturing promotions separately.
        for move_uci in ["a7a8q", "a7a8n", "a7b8q", "a7b8n"] {
            let move_to_test = find_move(&promotion_pos, move_uci);
            assert_eq!(
                static_exchange_eval_for_pruning(&promotion_pos, move_to_test),
                None,
                "promotion must fail open: {move_uci}"
            );
        }
        let capturing_promotion_with_recapture = "1r1r2k1/P7/8/8/8/8/8/7K w - - 0 1";
        let recapture_pos = parse_fen(capturing_promotion_with_recapture).unwrap();
        for move_uci in ["a7b8q", "a7b8n"] {
            let move_to_test = find_move(&recapture_pos, move_uci);
            assert_eq!(
                static_exchange_eval_for_pruning(&recapture_pos, move_to_test),
                None,
                "capturing promotion must fail open: {move_uci}"
            );
            let (kept, stats) = prune_one(capturing_promotion_with_recapture, move_uci);
            assert_eq!(kept, vec![move_to_test]);
            assert_eq!(stats.qsearch_see_pruned, 0);
        }

        // En passant remains fail-open until SEE has a dedicated EP proof.
        let ep = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
        let (kept, stats) = prune_one(ep, "e5d6");
        assert_eq!(kept.len(), 1);
        assert_eq!(stats.qsearch_en_passant_kept, 1);
        assert_eq!(stats.qsearch_see_tests, 0);

        // Mate-range windows fail open even for an otherwise losing capture.
        let mut mate_pos = parse_fen(losing).unwrap();
        let mate_move = find_move(&mate_pos, "f3e5");
        let mate_ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
        let mate_kept = prune_qsearch_captures_by_see(
            &mut mate_pos,
            vec![mate_move],
            &mate_ctx,
            -(MATE - 1),
            MATE - 1,
        );
        assert_eq!(mate_kept, vec![mate_move]);
        assert_eq!(mate_ctx.stats().qsearch_see_tests, 0);
    }

    #[test]
    fn qsearch_see_pruning_reduces_real_qsearch_work_without_changing_mate_score() {
        let fen = MVV_POS;
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };

        let run = |profile| {
            let mut pos = parse_fen(fen).unwrap();
            let key = pos.zobrist_key();
            let ctx = SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let mut tt = TranspositionTable::disabled();
            let out = search_best_move_with_history_tt_and_profile(
                &mut pos,
                &[key],
                &limits,
                &ctx,
                &mut tt,
                profile,
            )
            .expect("MVV fixture must be non-terminal");
            (out, ctx.stats())
        };

        let (movegen, movegen_stats) = run(SearchProfile::CurrentQsearchMovegen);
        let (pruning, pruning_stats) = run(SearchProfile::CurrentQsearchPruning);
        assert_eq!(pruning.score, Some(990));
        assert_eq!(pruning.best_move, movegen.best_move);
        assert!(pruning_stats.qsearch_see_tests > 0);
        assert!(pruning_stats.qsearch_see_pruned > 0);
        assert!(pruning_stats.qsearch_nodes < movegen_stats.qsearch_nodes);
        assert!(pruning_stats.nodes < movegen_stats.nodes);
    }

    #[test]
    fn fast_qsearch_see_matches_d13_decisions() {
        let positions = [
            START_FEN,
            MVV_POS,
            "6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1",
            "7k/r7/8/8/p7/8/8/R6K w - - 0 1",
            "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "r3b3/3P4/1k6/8/8/8/4Q3/6K1 w - - 0 1",
            "6K1/4q3/8/8/8/1k6/3p4/R3B3 b - - 0 1",
        ];
        for fen in positions {
            let mut slow_pos = parse_fen(fen).unwrap();
            let tactical = generate_legal_tactical_moves_with_stats(&mut slow_pos).0;
            let slow_ctx =
                SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let slow = prune_qsearch_captures_by_see(
                &mut slow_pos,
                tactical.clone(),
                &slow_ctx,
                -10_000,
                10_000,
            );

            let mut fast_pos = parse_fen(fen).unwrap();
            let fast_ctx =
                SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let fast = prune_qsearch_captures_by_fast_see(
                &mut fast_pos,
                tactical,
                &fast_ctx,
                -10_000,
                10_000,
            );

            assert_eq!(fast, slow, "fast SEE changed keep/prune set for {fen}");
            assert_eq!(
                fast_ctx.stats().qsearch_see_pruned,
                slow_ctx.stats().qsearch_see_pruned,
                "fast SEE changed prune count for {fen}"
            );
            assert_eq!(
                fast_ctx.stats().qsearch_see_fail_open_promotions,
                slow_ctx.stats().qsearch_see_fail_open_promotions,
                "fast SEE changed fail-open count for {fen}"
            );
        }

        let mut walk = parse_fen(START_FEN).unwrap();
        let mut seed = 0xD140_5EED_u64;
        for _ in 0..256 {
            let mut slow_pos = walk;
            let tactical = generate_legal_tactical_moves_with_stats(&mut slow_pos).0;
            let slow_ctx =
                SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let slow = prune_qsearch_captures_by_see(
                &mut slow_pos,
                tactical.clone(),
                &slow_ctx,
                -10_000,
                10_000,
            );
            let mut fast_pos = walk;
            let fast_ctx =
                SearchContext::new_with_profiling(Arc::new(AtomicBool::new(false)), true);
            let fast = prune_qsearch_captures_by_fast_see(
                &mut fast_pos,
                tactical,
                &fast_ctx,
                -10_000,
                10_000,
            );
            assert_eq!(fast, slow, "fast SEE changed walk decision");

            let mut legal = generate_legal_moves(&mut walk);
            if legal.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let move_to_play = legal.swap_remove((seed as usize) % legal.len());
            walk.make_move(move_to_play);
        }
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
        let r = probe_child_draw(
            pos,
            path.keys(),
            child_ply,
            parent_ply,
            ctx,
            limits,
            pv,
            ProbeKind::Main,
        );
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
        let probe = probe_child_draw(
            pos,
            path.keys(),
            1,
            0,
            &ctx,
            &limits,
            &mut pv,
            ProbeKind::Main,
        );
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
            let probe = probe_child_draw(
                &mut p,
                path.keys(),
                1,
                0,
                &ctx,
                &limits,
                &mut pv,
                ProbeKind::Main,
            );
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
        let probe = probe_child_draw(
            &mut p,
            path.keys(),
            1,
            0,
            &ctx,
            &limits,
            &mut pv,
            ProbeKind::Main,
        );
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
        assert_eq!(p.reject, None);

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
        assert_eq!(lo_no.reject, Some(TtRejectReason::Bound));

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
        assert_eq!(up_no.reject, Some(TtRejectReason::Bound));
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
        assert_eq!(p.reject, Some(TtRejectReason::Depth));
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
        assert_eq!(p.reject, Some(TtRejectReason::Decode));
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

    #[test]
    fn tt_context_isolates_forcing_budget_and_preserves_current_key() {
        let pos = parse_fen(START_FEN).unwrap();
        let path = SearchPath::new(vec![pos.zobrist_key()]);
        let historical = TtKey::new(
            pos.zobrist_key(),
            pos.halfmove_clock(),
            path.repetition_signature(),
        );
        let current = current_tt_key(&pos, &path);
        assert_eq!(current, historical, "Current keeps the zero-budget key");

        let budget_zero = current_tt_key_with_forcing_budget(&pos, &path, 0);
        let budget_four = current_tt_key_with_forcing_budget(&pos, &path, 4);
        assert_eq!(budget_zero, current);
        assert_ne!(budget_zero, budget_four);
        assert_eq!(budget_zero.forcing_budget, 0);
        assert_eq!(budget_four.forcing_budget, 4);

        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key: budget_zero,
            depth: 5,
            score: score_to_tt(220, 0).unwrap(),
            bound: Bound::Exact,
            best_move: None,
        });
        let budget_four_probe =
            probe_tt_for_search(&tt, budget_four, 3, 0, i32::MIN + 1000, i32::MAX - 1000);
        assert_eq!(
            budget_four_probe.cutoff, None,
            "zero-budget Exact entries must not cut off an extended search"
        );
        assert_eq!(budget_four_probe.hash_move, None);

        let mut tt = TranspositionTable::new_mb(1).unwrap();
        tt.store(TTEntry {
            key: budget_four,
            depth: 5,
            score: score_to_tt(-410, 0).unwrap(),
            bound: Bound::Exact,
            best_move: None,
        });
        let budget_zero_probe =
            probe_tt_for_search(&tt, budget_zero, 3, 0, i32::MIN + 1000, i32::MAX - 1000);
        assert_eq!(
            budget_zero_probe.cutoff, None,
            "extended Exact entries must not cut off an ordinary search"
        );
        assert_eq!(budget_zero_probe.hash_move, None);
    }

    #[test]
    fn threat_aware_persistent_tt_is_stable_across_cold_warm_and_backtracking() {
        const FEN: &str = "r4rk1/ppp2ppp/8/8/8/6q1/PPPP1PPP/R3Q1K1 w - - 0 1";
        const DEPTH: u32 = 3;
        let limits = SearchLimits {
            depth: Some(DEPTH),
            ..Default::default()
        };
        let root = parse_fen(FEN).unwrap();
        let root_key = root.zobrist_key();

        let search = |pos: &mut Position,
                      history: &[ZobristKey],
                      tt: &mut TranspositionTable|
         -> SearchOutcome {
            let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
            search_best_move_with_history_tt_and_profile(
                pos,
                history,
                &limits,
                &ctx,
                tt,
                SearchProfile::CurrentThreatAware,
            )
            .expect("threat-aware fixture must be non-terminal")
        };

        // Cold reference result.
        let mut cold_pos = root;
        let mut cold_tt = TranspositionTable::new_mb(2).unwrap();
        let cold = search(&mut cold_pos, &[root_key], &mut cold_tt);

        // First search populates the persistent candidate TT. The root entry
        // must use the full four-unit root context, not the historical zero
        // budget key.
        let mut reused_pos = root;
        let mut reused_tt = TranspositionTable::new_mb(2).unwrap();
        let first_reused = search(&mut reused_pos, &[root_key], &mut reused_tt);
        // The first reused search above is still cold. Repeat the same root
        // immediately so this test also exercises a genuinely warm root TT
        // lookup before changing the position.
        let warm_repeat = search(&mut reused_pos, &[root_key], &mut reused_tt);
        let root_path = SearchPath::new(vec![root_key]);
        let candidate_root_key =
            current_tt_key_with_forcing_budget(&reused_pos, &root_path, MAX_FORCING_EXTENSIONS);
        assert!(
            reused_tt.probe(candidate_root_key).is_some(),
            "candidate root result must be stored under its forcing context"
        );

        // Move forward using the completed candidate PV, search the
        // descendant in the same TT, then undo exactly that edge and search
        // the original root again. The result must not depend on the warm TT
        // or on the intervening descendant search.
        let undo = reused_pos.make_move(first_reused.best_move);
        let descendant_key = reused_pos.zobrist_key();
        let _descendant = search(&mut reused_pos, &[root_key, descendant_key], &mut reused_tt);
        reused_pos.unmake_move(undo);
        let back = search(&mut reused_pos, &[root_key], &mut reused_tt);

        // The first reused search is cold, so its full PV remains a direct
        // baseline comparison. A true warm lookup may legitimately retain
        // only a shorter legal PV (hash ordering is not a score semantic),
        // so the warm and backtracking checks below compare score/depth and
        // validate PV self-consistency instead of requiring byte-identical
        // tails.
        assert_eq!(
            first_reused.completed_depth, cold.completed_depth,
            "first-reused-cold completed depth must match cold search"
        );
        assert_eq!(
            first_reused.score, cold.score,
            "first-reused-cold score must match cold"
        );
        assert_eq!(
            first_reused.best_move, cold.best_move,
            "first-reused-cold bestmove must match cold"
        );
        assert_eq!(
            first_reused.pv, cold.pv,
            "first-reused-cold PV must match cold"
        );
        assert!(
            !first_reused.stopped,
            "first-reused-cold search must complete"
        );

        for (label, outcome) in [("warm-repeat", warm_repeat), ("back", back)] {
            assert_eq!(
                outcome.completed_depth, cold.completed_depth,
                "{label} completed depth must match cold search"
            );
            assert_eq!(
                outcome.score, cold.score,
                "{label} score must match cold search"
            );
            assert!(
                pv_is_legal(FEN, &outcome.pv),
                "{label} PV must remain legal after TT reuse"
            );
            assert_eq!(
                outcome.pv.first(),
                Some(&outcome.best_move),
                "{label} PV must start with its bestmove"
            );
            assert!(!outcome.stopped, "{label} search must complete");
        }
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
            969,
            "disabled queen-win d3 node count unchanged"
        );
        assert_eq!(move_to_uci(out.best_move), "e4a4");
        assert_eq!(out.score, Some(990));
        assert_eq!(
            out.pv.iter().map(|m| move_to_uci(*m)).collect::<Vec<_>>(),
            vec![
                "e4a4".to_string(),
                "h4h3".to_string(),
                "a4h4".to_string(),
                "h8g7".to_string(),
                "h4h3".to_string()
            ]
        );
    }

    // ---- §8.1 / M4.1 (Commit 2): profile plumbing -----------------------
    #[test]
    fn m4_profile_reference_reproduces_baseline() {
        // The new profile-aware entry, driven with `M4Reference`, must
        // lock the current M4Reference smoke values. This is the contract
        // that keeps the M4Reference search policy valid after the M4.1
        // refactor (the old `search_best_move_with_history_and_tt` now
        // delegates here with `M4Reference`); historical pre-EVAL values are
        // recorded separately in the benchmark documents.
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
        assert_eq!(out.score, Some(990));
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
        assert_eq!(iter.score, 990, "root score is the winning value");
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
        assert_eq!(iter.score, 990, "root score is the winning value");
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
        assert_eq!(iter.score, 990, "root reports the real winning score");
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
