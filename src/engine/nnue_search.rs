//! S10-C2B — search-local NNUE accumulator stack.
//!
//! Delivers accumulator frames to the search along the SAME edge contract
//! as `SearchPath`: one child push per real search make, one pop per
//! unmake. Frames are plain copies (256 x i32 = 1 KiB) — correctness
//! first; undo-delta/arena storage is C3 optimization territory.
//!
//! The stack is deliberately NOT stored in `Position`/`Undo`; it is
//! threaded through the search like `SearchPath` and truncated by a
//! `restore_root` safety net at every public entry, mirroring the
//! existing abort-unwind contract.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::chess::position::Position;
use crate::chess::types::Move;

use crate::engine::nnue_v2q_runtime::{
    NnueMoveDelta, NnueV2Accumulator, NnueV2QuantizedModel,
};

/// Which accumulator delivery mechanism a search uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnueSearchMode {
    /// Full refresh at every static eval (C2B reference implementation).
    FullRefresh,
    /// Search-local incremental accumulator stack.
    Incremental,
}

/// S10-C3-0: OPTIONAL diagnostics bundle. `None` on normal performance
/// runs — the hot paths then contain ZERO atomic operations. Enabled
/// explicitly (`--nnue-stack-telemetry` / `--nnue-audit`) for evidence
/// runs. The Arc keeps counters readable after the search consumed the
/// state.
#[derive(Debug, Default)]
pub struct NnueDiagnostics {
    // stack telemetry
    pub pushes: AtomicU64,
    pub pops: AtomicU64,
    pub null_pushes: AtomicU64,
    pub full_refreshes: AtomicU64,
    pub delta_updates: AtomicU64,
    pub max_depth: AtomicU64,
    // deep audit counters
    pub audit_eval_calls: AtomicU64,
    pub audit_lane_mismatches: AtomicU64,
    pub audit_raw_mismatches: AtomicU64,
}

/// Plain-copy snapshot of the telemetry counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    pub pushes: u64,
    pub pops: u64,
    pub null_pushes: u64,
    pub full_refreshes: u64,
    pub delta_updates: u64,
    pub max_depth: u64,
}

/// Plain-copy snapshot of the deep-audit counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuditSnapshot {
    pub eval_calls: u64,
    pub lane_mismatches: u64,
    pub raw_mismatches: u64,
}

/// Search-local NNUE state: the frozen model plus the accumulator stack.
pub struct NnueSearchState {
    pub model: Arc<NnueV2QuantizedModel>,
    pub mode: NnueSearchMode,
    frames: Vec<NnueV2Accumulator>,
    /// S10-C3-0: `None` on normal performance runs (zero atomic ops on the
    /// hot paths). `Some` only when explicitly requested.
    pub diagnostics: Option<Arc<NnueDiagnostics>>,
    /// S10-C3-0: audit decision FROZEN at state construction — the eval
    /// path pays one plain bool check, never an atomic load.
    audit_enabled: bool,
}

impl NnueSearchState {
    /// Initialize at the root: load-free (model provided), full-refresh
    /// the root accumulator as frame 0. No diagnostics (performance shape).
    pub fn new(
        model: Arc<NnueV2QuantizedModel>,
        mode: NnueSearchMode,
        root: &Position,
    ) -> Self {
        Self::with_options(model, mode, root, false, false)
    }

    /// Full constructor. `telemetry` enables stack counters;
    /// `audit` enables the per-eval deep comparison (implies telemetry
    /// being meaningful; both freeze at construction — C3-0 hygiene).
    pub fn with_options(
        model: Arc<NnueV2QuantizedModel>,
        mode: NnueSearchMode,
        root: &Position,
        telemetry: bool,
        audit: bool,
    ) -> Self {
        assert!(
            !audit || matches!(mode, NnueSearchMode::Incremental),
            "deep audit requires the incremental profile"
        );
        let root_acc = model.full_accumulator(root);
        NnueSearchState {
            model,
            mode,
            frames: vec![root_acc],
            diagnostics: if telemetry || audit {
                Some(Arc::new(NnueDiagnostics::default()))
            } else {
                None
            },
            audit_enabled: audit,
        }
    }

    /// True when the deep audit is active (frozen at construction).
    pub fn audit_enabled(&self) -> bool {
        self.audit_enabled
    }

    /// Snapshot of the telemetry counters (zeros when diagnostics are off).
    pub fn telemetry_snapshot(&self) -> TelemetrySnapshot {
        match self.diagnostics.as_ref() {
            Some(d) => TelemetrySnapshot {
                pushes: d.pushes.load(Ordering::Relaxed),
                pops: d.pops.load(Ordering::Relaxed),
                null_pushes: d.null_pushes.load(Ordering::Relaxed),
                full_refreshes: d.full_refreshes.load(Ordering::Relaxed),
                delta_updates: d.delta_updates.load(Ordering::Relaxed),
                max_depth: d.max_depth.load(Ordering::Relaxed),
            },
            None => TelemetrySnapshot::default(),
        }
    }

    /// Snapshot of the deep-audit counters (zeros when audit is off).
    pub fn audit_snapshot(&self) -> AuditSnapshot {
        match self.diagnostics.as_ref() {
            Some(d) => AuditSnapshot {
                eval_calls: d.audit_eval_calls.load(Ordering::Relaxed),
                lane_mismatches: d.audit_lane_mismatches.load(Ordering::Relaxed),
                raw_mismatches: d.audit_raw_mismatches.load(Ordering::Relaxed),
            },
            None => AuditSnapshot::default(),
        }
    }

    /// Audited integer evaluation: when the audit is enabled, compare the
    /// stack top against a fresh full refresh (256 lanes + raw) and count
    /// mismatches, then return the FRESH score (a detected corruption must
    /// not change the search tree — the mismatch counters fail the gate).
    pub fn evaluate_cp_i32_audited(&self, pos: &Position) -> i32 {
        if !self.audit_enabled {
            return self.evaluate_cp_i32(pos);
        }
        let diag = self
            .diagnostics
            .as_ref()
            .expect("audit implies diagnostics");
        diag.audit_eval_calls.fetch_add(1, Ordering::Relaxed);
        let fresh = self.model.full_accumulator(pos);
        let top = self.top();
        if top.white != fresh.white {
            let n = top
                .white
                .iter()
                .zip(fresh.white.iter())
                .filter(|(a, b)| a != b)
                .count() as u64;
            diag.audit_lane_mismatches.fetch_add(n, Ordering::Relaxed);
        }
        if top.black != fresh.black {
            let n = top
                .black
                .iter()
                .zip(fresh.black.iter())
                .filter(|(a, b)| a != b)
                .count() as u64;
            diag.audit_lane_mismatches.fetch_add(n, Ordering::Relaxed);
        }
        let inc_raw = self.model.evaluate_raw_from_accumulator(pos, top);
        let fresh_raw = self.model.evaluate_raw(pos);
        if inc_raw != fresh_raw {
            diag.audit_raw_mismatches.fetch_add(1, Ordering::Relaxed);
        }
        NnueV2QuantizedModel::cp_i32_from_raw(fresh_raw)
    }

    /// Current top frame (the accumulator of the current node).
    #[inline]
    pub fn top(&self) -> &NnueV2Accumulator {
        self.frames.last().expect("nnue stack never empty")
    }

    /// Current stack depth (root = 1).
    #[inline]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// True when this state maintains the incremental accumulator stack.
    /// FullRefresh states carry the model for evaluation only: their
    /// push/pop/delta methods are no-ops so the full-refresh arm never
    /// pays incremental maintenance cost (C3 benchmark purity).
    #[inline]
    pub fn is_incremental(&self) -> bool {
        matches!(self.mode, NnueSearchMode::Incremental)
    }

    /// Push the child frame for a real search move edge.
    ///
    /// `delta` MUST have been prepared against the PARENT position
    /// (BEFORE the move); `child` is the position AFTER `make_move`.
    /// Order contract: prepare delta BEFORE make; push AFTER make.
    pub fn push_child(
        &mut self,
        delta: &NnueMoveDelta,
        child: &Position,
    ) {
        if !self.is_incremental() {
            return; // FullRefresh: no stack maintenance.
        }
        let mut child_acc = *self.top();
        let stats =
            self.model.update_accumulator_for_move(&mut child_acc, delta, child);
        self.frames.push(child_acc);
        if let Some(diag) = self.diagnostics.as_ref() {
            diag.delta_updates
                .fetch_add(stats.delta_updates as u64, Ordering::Relaxed);
            diag.full_refreshes
                .fetch_add(stats.full_refreshes as u64, Ordering::Relaxed);
            diag.pushes.fetch_add(1, Ordering::Relaxed);
            diag.max_depth
                .fetch_max(self.frames.len() as u64, Ordering::Relaxed);
        }
    }

    /// Push the null-move child: the board, pieces, and kings are
    /// unchanged, so the child accumulator is bit-identical to the
    /// parent's — push a plain copy. The dense forward swaps STM/NSTM
    /// via the flipped side-to-move at eval time.
    pub fn push_null_child(&mut self) {
        if !self.is_incremental() {
            return; // FullRefresh: no stack maintenance.
        }
        let child_acc = *self.top();
        self.frames.push(child_acc);
        if let Some(diag) = self.diagnostics.as_ref() {
            diag.null_pushes.fetch_add(1, Ordering::Relaxed);
            diag.pushes.fetch_add(1, Ordering::Relaxed);
            diag.max_depth
                .fetch_max(self.frames.len() as u64, Ordering::Relaxed);
        }
    }

    /// Pop the current child frame (edge unwind).
    pub fn pop(&mut self) {
        if !self.is_incremental() {
            return; // FullRefresh: no stack maintenance.
        }
        debug_assert!(
            self.frames.len() > 1,
            "nnue stack pop below root frame"
        );
        if self.frames.len() > 1 {
            self.frames.pop();
        }
        if let Some(diag) = self.diagnostics.as_ref() {
            diag.pops.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Abort-unwind safety net: truncate back to the root frame. Called
    /// at every public search entry after the recursion, mirroring
    /// `SearchPath::restore_root`.
    pub fn restore_root(&mut self) {
        self.frames.truncate(1);
    }

    /// Test-only mutable access to the frames (audit tamper tests).
    #[cfg(test)]
    pub(crate) fn frames_mut(&mut self) -> &mut Vec<NnueV2Accumulator> {
        &mut self.frames
    }

    /// Integer NNUE evaluation of `pos` under the configured mode.
    /// - FullRefresh: full accumulator + dense (reference path).
    /// - Incremental: dense from the current top frame (the position at
    ///   the current stack top MUST be `pos`).
    pub fn evaluate_cp_i32(&self, pos: &Position) -> i32 {
        match self.mode {
            NnueSearchMode::FullRefresh => self.model.evaluate_cp_i32(pos),
            NnueSearchMode::Incremental => self
                .model
                .evaluate_cp_i32_from_accumulator(pos, self.top()),
        }
    }

    /// Prepare a move delta against the parent position (call BEFORE
    /// `pos.make_move`).
    pub fn prepare_delta(&self, parent: &Position, mv: &Move) -> NnueMoveDelta {
        self.model.prepare_move_delta(parent, mv)
    }
}

/// Audit result for the deep incremental audit (lane/raw comparison of
/// the stack top against a fresh full refresh).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NnueAuditCounters {
    pub eval_calls: u64,
    pub lane_mismatches: u64,
    pub raw_mismatches: u64,
}

impl NnueAuditCounters {
    pub fn passed(&self) -> bool {
        self.lane_mismatches == 0 && self.raw_mismatches == 0
    }
}

/// Audit one evaluation: compare the incremental stack top against a
/// fresh full refresh of `pos` (all 256 lanes + raw integer output).
/// Returns the audit-updated evaluation (always computed from the FRESH
/// path so audit mode cannot mask an integration bug in the returned
/// score — the mismatch counters expose the stack).
pub fn audit_incremental_eval(
    state: &NnueSearchState,
    pos: &Position,
    counters: &mut NnueAuditCounters,
) -> i32 {
    counters.eval_calls += 1;
    let fresh = state.model.full_accumulator(pos);
    let top = state.top();
    if top.white != fresh.white {
        counters.lane_mismatches += top
            .white
            .iter()
            .zip(fresh.white.iter())
            .filter(|(a, b)| a != b)
            .count() as u64;
    }
    if top.black != fresh.black {
        counters.lane_mismatches += top
            .black
            .iter()
            .zip(fresh.black.iter())
            .filter(|(a, b)| a != b)
            .count() as u64;
    }
    let inc_raw = state
        .model
        .evaluate_raw_from_accumulator(pos, top);
    let fresh_raw = state.model.evaluate_raw(pos);
    if inc_raw != fresh_raw {
        counters.raw_mismatches += 1;
    }
    // Score from the FRESH path (audit must not mask bugs).
    NnueV2QuantizedModel::cp_i32_from_raw(fresh_raw)
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::chess::fen::parse_fen;

    use super::*;

    /// Minimal synthetic model (deterministic weights; see the runtime's
    /// synthetic_artifact_bytes for the byte-level format).
    fn synthetic_model()
        -> Arc<NnueV2QuantizedModel> {
        // Reuse the runtime's test-only synthetic builder via a public-ish
        // path: build from the same generator the runtime tests use.
        // For module isolation we rebuild a small valid artifact here.
        let bytes = crate::engine::nnue_v2q_runtime::synthetic_artifact_bytes_for_tests(
            crate::chess::types::START_FEN);
        Arc::new(NnueV2QuantizedModel::from_bytes(&bytes).unwrap())
    }

    #[test]
    fn stack_push_pop_balance_and_restore_root() {
        use crate::chess::movegen::generate_legal_moves;
        let model = synthetic_model();
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state = NnueSearchState::with_options(
            model, NnueSearchMode::Incremental, &pos, true, false);
        let root_acc = *state.top();
        assert_eq!(state.depth(), 1);
        let moves = generate_legal_moves(&mut pos.clone());
        assert!(!moves.is_empty());
        for m in moves {
            let delta = state.prepare_delta(&pos, &m);
            let undo = pos.make_move(m);
            state.push_child(&delta, &pos);
            assert_eq!(state.depth(), 2);
            // pop restores the root frame exactly
            state.pop();
            pos.unmake_move(undo);
            assert_eq!(state.depth(), 1);
            assert_eq!(*state.top(), root_acc);
        }
        let snap = state.telemetry_snapshot();
        assert!(snap.pushes > 0);
        assert_eq!(snap.pushes, snap.pops);
    }

    /// S10-C3-0 anti-contamination: a state built WITHOUT diagnostics has
    /// none (no handle), and its snapshots stay zero after real pushes.
    #[test]
    fn diagnostics_off_means_no_counters() {
        use crate::chess::movegen::generate_legal_moves;
        let model = synthetic_model();
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state =
            NnueSearchState::new(model, NnueSearchMode::Incremental, &pos);
        assert!(state.diagnostics.is_none());
        assert!(!state.audit_enabled());
        let moves = generate_legal_moves(&mut pos.clone());
        let m = moves[0];
        let delta = state.prepare_delta(&pos, &m);
        let undo = pos.make_move(m);
        state.push_child(&delta, &pos);
        state.pop();
        pos.unmake_move(undo);
        // No diagnostics -> snapshots stay zero (and no handle exists).
        assert_eq!(state.telemetry_snapshot(), TelemetrySnapshot::default());
        assert_eq!(state.audit_snapshot(), AuditSnapshot::default());
    }

    /// S10-C3-0 counter contract: telemetry on -> pushes/pops > 0.
    #[test]
    fn telemetry_on_counts_moves() {
        use crate::chess::movegen::generate_legal_moves;
        let model = synthetic_model();
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state = NnueSearchState::with_options(
            model, NnueSearchMode::Incremental, &pos, true, false);
        let m = generate_legal_moves(&mut pos.clone())[0];
        let delta = state.prepare_delta(&pos, &m);
        let undo = pos.make_move(m);
        state.push_child(&delta, &pos);
        pos.unmake_move(undo);
        let snap = state.telemetry_snapshot();
        assert!(snap.pushes > 0, "telemetry must count pushes");
    }

    /// S10-C3-0 counter contract: audit on -> eval_calls > 0.
    #[test]
    fn audit_on_counts_evals() {
        let model = synthetic_model();
        let pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let state = NnueSearchState::with_options(
            model, NnueSearchMode::Incremental, &pos, false, true);
        assert!(state.audit_enabled());
        let _ = state.evaluate_cp_i32_audited(&pos);
        let snap = state.audit_snapshot();
        assert!(snap.eval_calls > 0, "audit must count evals");
    }

    #[test]
    fn null_child_push_is_bit_identical() {
        let model = synthetic_model();
        let pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state = NnueSearchState::with_options(
            model, NnueSearchMode::Incremental, &pos, true, false);
        let parent = *state.top();
        state.push_null_child();
        assert_eq!(*state.top(), parent, "null child frame == parent frame");
        assert_eq!(state.telemetry_snapshot().null_pushes, 1);
        state.pop();
        assert_eq!(state.depth(), 1);
    }

    #[test]
    fn incremental_eval_matches_full_refresh_in_stack() {
        use crate::chess::movegen::generate_legal_moves;
        let model = synthetic_model();
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state =
            NnueSearchState::new(model, NnueSearchMode::Incremental, &pos);
        let moves = generate_legal_moves(&mut pos.clone());
        for m in moves {
            let delta = state.prepare_delta(&pos, &m);
            let undo = pos.make_move(m);
            state.push_child(&delta, &pos);
            let fresh = state.model.full_accumulator(&pos);
            assert_eq!(*state.top(), fresh, "stack frame == fresh for {m:?}");
            let inc_raw = state
                .model
                .evaluate_raw_from_accumulator(&pos, state.top());
            assert_eq!(inc_raw, state.model.evaluate_raw(&pos));
            state.pop();
            pos.unmake_move(undo);
        }
    }

    #[test]
    fn audit_counters_detect_tampered_stack() {
        let model = synthetic_model();
        let pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state = NnueSearchState::with_options(
            model, NnueSearchMode::Incremental, &pos, true, true);
        // Corrupt the top frame's first lane deliberately.
        state.frames_mut()[0].white[0] = state.top().white[0].wrapping_add(1);
        let _ = state.evaluate_cp_i32_audited(&pos);
        let snap = state.audit_snapshot();
        assert_eq!(snap.eval_calls, 1);
        assert!(snap.lane_mismatches > 0, "audit must detect corruption");
    }
}
