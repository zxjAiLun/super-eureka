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

/// Telemetry counters (profiling-gated increments keep production clean).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NnueStackTelemetry {
    pub pushes: u64,
    pub pops: u64,
    pub null_pushes: u64,
    pub full_refreshes: u64,
    pub delta_updates: u64,
    pub max_depth: usize,
}

/// Search-local NNUE state: the frozen model plus the accumulator stack.
pub struct NnueSearchState {
    pub model: Arc<NnueV2QuantizedModel>,
    pub mode: NnueSearchMode,
    frames: Vec<NnueV2Accumulator>,
    pub telemetry: NnueStackTelemetry,
}

impl NnueSearchState {
    /// Initialize at the root: load-free (model provided), full-refresh
    /// the root accumulator as frame 0.
    pub fn new(
        model: Arc<NnueV2QuantizedModel>,
        mode: NnueSearchMode,
        root: &Position,
    ) -> Self {
        let root_acc = model.full_accumulator(root);
        NnueSearchState {
            model,
            mode,
            frames: vec![root_acc],
            telemetry: NnueStackTelemetry {
                max_depth: 1,
                ..Default::default()
            },
        }
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
        let mut child_acc = *self.top();
        let stats =
            self.model.update_accumulator_for_move(&mut child_acc, delta, child);
        self.telemetry.delta_updates += stats.delta_updates as u64;
        self.telemetry.full_refreshes += stats.full_refreshes as u64;
        self.frames.push(child_acc);
        self.telemetry.pushes += 1;
        if self.frames.len() > self.telemetry.max_depth {
            self.telemetry.max_depth = self.frames.len();
        }
    }

    /// Push the null-move child: the board, pieces, and kings are
    /// unchanged, so the child accumulator is bit-identical to the
    /// parent's — push a plain copy. The dense forward swaps STM/NSTM
    /// via the flipped side-to-move at eval time.
    pub fn push_null_child(&mut self) {
        let child_acc = *self.top();
        self.frames.push(child_acc);
        self.telemetry.null_pushes += 1;
        self.telemetry.pushes += 1;
        if self.frames.len() > self.telemetry.max_depth {
            self.telemetry.max_depth = self.frames.len();
        }
    }

    /// Pop the current child frame (edge unwind).
    pub fn pop(&mut self) {
        debug_assert!(
            self.frames.len() > 1,
            "nnue stack pop below root frame"
        );
        if self.frames.len() > 1 {
            self.frames.pop();
        }
        self.telemetry.pops += 1;
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
        let mut state =
            NnueSearchState::new(model, NnueSearchMode::Incremental, &pos);
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
        assert_eq!(state.telemetry.pushes, state.telemetry.pops);
    }

    #[test]
    fn null_child_push_is_bit_identical() {
        let model = synthetic_model();
        let pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut state =
            NnueSearchState::new(model, NnueSearchMode::Incremental, &pos);
        let parent = *state.top();
        state.push_null_child();
        assert_eq!(*state.top(), parent, "null child frame == parent frame");
        assert_eq!(state.telemetry.null_pushes, 1);
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
        let mut state =
            NnueSearchState::new(model, NnueSearchMode::Incremental, &pos);
        // Corrupt the top frame's first lane deliberately.
        state.frames_mut()[0].white[0] = state.top().white[0].wrapping_add(1);
        let mut counters = NnueAuditCounters::default();
        let _ = audit_incremental_eval(&state, &pos, &mut counters);
        assert_eq!(counters.eval_calls, 1);
        assert!(counters.lane_mismatches > 0, "audit must detect corruption");
        assert!(!counters.passed());
    }
}
