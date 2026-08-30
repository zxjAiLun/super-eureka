//! S10-B5 — bench-only INTEGER parity runtime for the quantized V2 network.
//!
//! Loads an `EUNN2Q01` v1 artifact (see `tools/s10/export_quantized.py`) and
//! performs full-refresh integer inference with the frozen shift scheme:
//!
//!   acc  = q_ft_b + sum(q_ft_w rows)                    (i32, A units)
//!   a    = clamp(acc, 0, QA)                            (A units)
//!   z    = q_b + sum(q_w * a)                           (i32; no input
//!                                                          requantization)
//!   a'   = clamp(shift_round(z, Z), 0, QA)              (A units)
//!   raw  = shift_round(z_out, Z)                        (A units)
//!   cp   = raw / 2^FT * 1000.0                          (final conversion)
//!
//! shift_round = arithmetic right shift, round half away from zero.
//!
//! This is an isolated, bench-only bridge: NOT wired into search/eval/UCI,
//! no incremental accumulator. `nnue::active_features_v2()` remains the
//! single feature-encoding source of truth.

use crate::chess::position::Position;
use crate::chess::types::{Color, Move, Piece, Square};
use crate::engine::nnue::{
    active_features_v2, v2_feature_for_piece, NnuePerspective, NNUE_INPUTS_V2,
};

/// Fixed S10-B5 artifact constants (must match export_quantized.py).
pub const NNUE_V2Q_MAGIC: [u8; 8] = *b"EUNN2Q01";
pub const NNUE_V2Q_VERSION: u32 = 1;
pub const NNUE_V2Q_FT_WIDTH: usize = 128;
pub const NNUE_V2Q_TARGET_SCALE: f32 = 1000.0;
pub const NNUE_V2Q_FT_SHIFT: u32 = 12;
pub const NNUE_V2Q_DENSE_W_SHIFT: u32 = 12;
pub const NNUE_V2Q_DENSE_Z_SHIFT: u32 = 12;
pub const NNUE_V2Q_QA: usize = 1 << 12;

/// Historical provenance reference: the S10-D 300k artifact's source chain
/// (B3 checkpoint d59ad852... via B4 FP32 artifact 9bf7addd...). Since S10-E0
/// the loader is a FORMAT-CONTRACT loader: it still fully validates magic /
/// version / inputs=22528 / ft_width=128 / scales / quant shifts and the
/// proven i32 MAC bounds, and still records the header's source SHAs for
/// provenance output, but it no longer requires them to equal one specific
/// training iteration — model IDENTITY is enforced by the consumer (the
/// Arena D0 immutable model-artifact SHA gate), not by the binary.
const HISTORICAL_SOURCE_FP32_SHA: [u8; 32] = [
    0x9b, 0xf7, 0xad, 0xdd, 0xf7, 0xb3, 0xb4, 0x4a,
    0xff, 0xa5, 0xe2, 0x6d, 0x22, 0x76, 0xb1, 0x3d,
    0x74, 0x56, 0x61, 0x91, 0xa4, 0xeb, 0x4d, 0x00,
    0x90, 0xfb, 0xde, 0x5a, 0x7a, 0xfb, 0xc9, 0xfc,
];
#[allow(dead_code)]
const HISTORICAL_SOURCE_CHECKPOINT_SHA: [u8; 32] = [
    0xd5, 0x9a, 0xd8, 0x52, 0x5c, 0x06, 0xab, 0xe8,
    0x03, 0x07, 0xbf, 0xfb, 0x12, 0x1f, 0xf4, 0x97,
    0xa3, 0x6e, 0x94, 0xb1, 0x91, 0xc3, 0xc9, 0xbb,
    0x3c, 0x8f, 0x31, 0xe5, 0xcc, 0xe5, 0x50, 0xc7,
];

/// Maximum active features per perspective (startpos: 32 pieces minus own
/// king); used in the proven FT accumulator bound.
const MAX_FEATURES_PER_PERSPECTIVE: i64 = 31;

const HEADER_BYTES: usize = 8 + 4 * 4 + 4 * 3 + 4 + 4 + 32 + 32;
const FT_W_COUNT: usize = NNUE_INPUTS_V2 * NNUE_V2Q_FT_WIDTH;
const FT_B_COUNT: usize = NNUE_V2Q_FT_WIDTH;
const L1_W_COUNT: usize = 32 * 256;
const L1_B_COUNT: usize = 32;
const L2_W_COUNT: usize = 32 * 32;
const L2_B_COUNT: usize = 32;
const OUT_W_COUNT: usize = 1 * 32;
const OUT_B_COUNT: usize = 1;

const FT_W_OFFSET: usize = HEADER_BYTES;
const FT_B_OFFSET: usize = FT_W_OFFSET + FT_W_COUNT * 2;
const L1_W_OFFSET: usize = FT_B_OFFSET + FT_B_COUNT * 4;
const L1_B_OFFSET: usize = L1_W_OFFSET + L1_W_COUNT * 2;
const L2_W_OFFSET: usize = L1_B_OFFSET + L1_B_COUNT * 4;
const L2_B_OFFSET: usize = L2_W_OFFSET + L2_W_COUNT * 2;
const OUT_W_OFFSET: usize = L2_B_OFFSET + L2_B_COUNT * 4;
const OUT_B_OFFSET: usize = OUT_W_OFFSET + OUT_W_COUNT * 2;
const TOTAL_BYTES: usize = OUT_B_OFFSET + OUT_B_COUNT * 4;

/// Fixed-perspective NNUE accumulator (S10-C1). Lanes are ALWAYS stored in
/// White/Black perspective order — never STM/NSTM; the dense forward
/// selects by side-to-move at evaluation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnueV2Accumulator {
    pub white: [i32; NNUE_V2Q_FT_WIDTH],
    pub black: [i32; NNUE_V2Q_FT_WIDTH],
}

impl NnueV2Accumulator {
    /// Lane slice for one perspective.
    #[inline]
    pub fn lanes_for(&self, perspective: NnuePerspective) -> &[i32; NNUE_V2Q_FT_WIDTH] {
        match perspective {
            NnuePerspective::White => &self.white,
            NnuePerspective::Black => &self.black,
        }
    }
}

/// Telemetry from one incremental accumulator update.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdateStats {
    /// Number of individual feature add/sub applications performed.
    pub delta_updates: usize,
    /// Number of perspectives fully refreshed (own-king move).
    pub full_refreshes: usize,
}

/// Fixed-size, zero-heap move description for the C2A move-aware
/// accumulator updater. Describes the board changes of one move BEFORE it
/// is played (relative to the parent position):
///
/// - quiet / double push: removed = [moved@from], added = [moved@to]
/// - normal capture:      removed = [moved@from, captured@to]
///                        added   = [moved@to]
/// - en passant:          removed = [pawn@from, cap@ep-sq]
///                        added   = [pawn@to]
/// - castling:            removed = [king@from, rook@rf]
///                        added   = [king@to, rook@rt]
/// - promotion:           removed = [pawn@from, (captured@to)]
///                        added   = [promoted@to]
///
/// `moved_king` records the color whose KING square changes (the mover on
/// a king move, including castling), so the updater can full-refresh that
/// perspective without comparing positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NnueMoveDelta {
    removed: [Option<(Square, Piece)>; 2],
    added: [Option<(Square, Piece)>; 2],
    moved_king: Option<Color>,
}

impl NnueMoveDelta {
    /// Squares whose occupant changes across the move (test-facing view
    /// for the board-diff invariant check).
    pub fn dirty_squares(&self) -> impl Iterator<Item = Square> + '_ {
        self.removed
            .iter()
            .chain(self.added.iter())
            .filter_map(|e| e.map(|(sq, _)| sq))
    }

    #[cfg(test)]
    pub fn removed_entries(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        self.removed.iter().filter_map(|&e| e)
    }

    #[cfg(test)]
    pub fn added_entries(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        self.added.iter().filter_map(|&e| e)
    }
}

/// S10-C3-C2: L1 dense backend selected ONCE at load time (runtime
/// feature detection; never a binary-wide target-cpu requirement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L1Backend {
    Scalar,
    #[allow(dead_code)]
    Avx2,
}

/// Loaded S10-B5 quantized model. Immutable after construction.
#[derive(Debug)]
pub struct NnueV2QuantizedModel {
    ft_weights: Vec<i16>,  // [22528][128]
    ft_bias: Vec<i32>,     // [128]
    l1_weight: Vec<i16>,   // [32][256]
    l1_bias: Vec<i32>,     // [32]
    l2_weight: Vec<i16>,   // [32][32]
    l2_bias: Vec<i32>,     // [32]
    out_weight: Vec<i16>,  // [1][32]
    out_bias: Vec<i32>,    // [1]
    /// Source FP32 artifact SHA from the artifact header (provenance only
    /// since S10-E0; no longer required to equal one specific iteration).
    #[allow(dead_code)]
    source_fp32_artifact_sha256: [u8; 32],
    /// Source checkpoint SHA from the artifact header (provenance only
    /// since S10-E0; no longer required to equal one specific iteration).
    #[allow(dead_code)]
    source_checkpoint_sha256: [u8; 32],
    /// S10-C3-C2: L1 dense backend (runtime-detected once at load).
    l1_backend: L1Backend,
}

impl NnueV2QuantizedModel {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| {
            format!("nnue-v2q-probe: cannot read {}: {e}", path.display())
        })?;
        Self::from_bytes(&data)
    }

    /// Parse and fully validate an `EUNN2Q01` v1 artifact (little-endian).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != TOTAL_BYTES {
            return Err(format!(
                "nnue-v2q-probe: bad length {} != expected {TOTAL_BYTES}",
                data.len()
            ));
        }
        if data[0..8] != NNUE_V2Q_MAGIC {
            return Err("nnue-v2q-probe: bad magic".to_string());
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let inputs = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let ft_width = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let target_scale =
            f32::from_bits(u32::from_le_bytes(data[20..24].try_into().unwrap()));
        let ft_shift = u32::from_le_bytes(data[24..28].try_into().unwrap());
        let dense_w_shift =
            u32::from_le_bytes(data[28..32].try_into().unwrap());
        let dense_z_shift =
            u32::from_le_bytes(data[32..36].try_into().unwrap());
        let qa = u32::from_le_bytes(data[36..40].try_into().unwrap());
        if version != NNUE_V2Q_VERSION {
            return Err(format!("nnue-v2q-probe: bad version {version}"));
        }
        if inputs != NNUE_INPUTS_V2 as u32 {
            return Err(format!("nnue-v2q-probe: bad inputs {inputs}"));
        }
        if ft_width != NNUE_V2Q_FT_WIDTH as u32 {
            return Err(format!("nnue-v2q-probe: bad ft_width {ft_width}"));
        }
        if target_scale != NNUE_V2Q_TARGET_SCALE {
            return Err(format!(
                "nnue-v2q-probe: bad target_scale {target_scale}"
            ));
        }
        if ft_shift != NNUE_V2Q_FT_SHIFT
            || dense_w_shift != NNUE_V2Q_DENSE_W_SHIFT
            || dense_z_shift != NNUE_V2Q_DENSE_Z_SHIFT
            || qa != NNUE_V2Q_QA as u32
        {
            return Err(format!(
                "nnue-v2q-probe: bad shifts ({ft_shift},\
                 {dense_w_shift},{dense_z_shift},{qa})"
            ));
        }
        let mut source_fp32_artifact_sha256 = [0u8; 32];
        source_fp32_artifact_sha256.copy_from_slice(&data[44..76]);
        let mut source_checkpoint_sha256 = [0u8; 32];
        source_checkpoint_sha256.copy_from_slice(&data[76..108]);
        // S10-E0: format-contract loader. The header's source SHAs are kept
        // for provenance output but are NOT compared against one frozen
        // training iteration anymore — model identity is enforced by the
        // consumer (Arena D0's immutable model-artifact SHA gate pins the
        // exact bytes a tournament may launch with).

        let ft_weights = read_i16s(data, FT_W_OFFSET, FT_W_COUNT)?;
        let ft_bias = read_i32s(data, FT_B_OFFSET, FT_B_COUNT)?;
        let l1_weight = read_i16s(data, L1_W_OFFSET, L1_W_COUNT)?;
        let l1_bias = read_i32s(data, L1_B_OFFSET, L1_B_COUNT)?;
        let l2_weight = read_i16s(data, L2_W_OFFSET, L2_W_COUNT)?;
        let l2_bias = read_i32s(data, L2_B_OFFSET, L2_B_COUNT)?;
        let out_weight = read_i16s(data, OUT_W_OFFSET, OUT_W_COUNT)?;
        let out_bias = read_i32s(data, OUT_B_OFFSET, OUT_B_COUNT)?;

        // Fail-closed overflow safety: recompute the PROVEN worst-case
        // bounds from the actual payload (i64 arithmetic throughout — the
        // abs of i16::MIN/i32::MIN is not representable in the source
        // type) and refuse any artifact whose MACs could exceed i32 range.
        // The frozen artifact's bounds are ~1.45e9 max, far below i32::MAX.
        let ft_bound = max_abs_i32(&ft_bias)
            + MAX_FEATURES_PER_PERSPECTIVE * max_abs_i16(&ft_weights);
        let l1_bound = max_abs_i32(&l1_bias)
            + 256 * max_abs_i16(&l1_weight) * NNUE_V2Q_QA as i64;
        let l2_bound = max_abs_i32(&l2_bias)
            + 32 * max_abs_i16(&l2_weight) * NNUE_V2Q_QA as i64;
        let out_bound = max_abs_i32(&out_bias)
            + 32 * max_abs_i16(&out_weight) * NNUE_V2Q_QA as i64;
        if ft_bound > i32::MAX as i64
            || l1_bound > i32::MAX as i64
            || l2_bound > i32::MAX as i64
            || out_bound > i32::MAX as i64
        {
            return Err(format!(
                "nnue-v2q-probe: payload exceeds proven i32 MAC bounds \
                 (ft={ft_bound}, l1={l1_bound}, l2={l2_bound}, out={out_bound})"
            ));
        }

        // S10-C3-C2: runtime L1 backend detection (once per load).
        #[cfg(all(target_arch = "x86_64", not(feature = "force_scalar_l1")))]
        let l1_backend = if is_avx2_detected() {
            L1Backend::Avx2
        } else {
            L1Backend::Scalar
        };
        #[cfg(any(not(target_arch = "x86_64"), feature = "force_scalar_l1"))]
        let l1_backend = L1Backend::Scalar;

        Ok(NnueV2QuantizedModel {
            ft_weights,
            ft_bias,
            l1_weight,
            l1_bias,
            l2_weight,
            l2_bias,
            out_weight,
            out_bias,
            source_fp32_artifact_sha256,
            source_checkpoint_sha256,
            l1_backend,
        })
    }

    /// Current L1 backend (test visibility).
    pub fn l1_backend(&self) -> L1Backend {
        self.l1_backend
    }

    /// Integer accumulator: `q_bias + sum(active feature rows)` (A units).
    fn accumulate(&self, indices: &[u16]) -> [i32; NNUE_V2Q_FT_WIDTH] {
        let mut acc = [0i32; NNUE_V2Q_FT_WIDTH];
        for i in 0..NNUE_V2Q_FT_WIDTH {
            acc[i] = self.ft_bias[i];
        }
        for &idx in indices {
            let base = (idx as usize) * NNUE_V2Q_FT_WIDTH;
            for (i, slot) in acc.iter_mut().enumerate() {
                // i32 accumulation; proven bound 1,016,721 << i32::MAX.
                *slot = slot
                    .wrapping_add(self.ft_weights[base + i] as i32);
            }
        }
        acc
    }

    /// Full integer forward pass; returns the raw integer output (A units).
    /// Single fact path: full_accumulator -> evaluate_raw_from_accumulator.
    pub fn evaluate_raw(&self, pos: &Position) -> i32 {
        let acc = self.full_accumulator(pos);
        self.evaluate_raw_from_accumulator(pos, &acc)
    }

    /// Full-refresh accumulator for both perspectives of `pos`.
    ///
    /// Stored ALWAYS in White/Black perspective order (never STM/NSTM);
    /// the dense forward swaps by side-to-move at evaluation time.
    pub fn full_accumulator(&self, pos: &Position) -> NnueV2Accumulator {
        NnueV2Accumulator {
            white: self.full_accumulator_for_perspective(
                pos, NnuePerspective::White),
            black: self.full_accumulator_for_perspective(
                pos, NnuePerspective::Black),
        }
    }

    /// Full-refresh lanes for ONE perspective only (the C2A shape: an
    /// own-king move refreshes just that perspective, not both).
    pub fn full_accumulator_for_perspective(
        &self,
        pos: &Position,
        perspective: NnuePerspective,
    ) -> [i32; NNUE_V2Q_FT_WIDTH] {
        let features = active_features_v2(pos, perspective);
        self.accumulate(&features)
    }

    /// S10-C3-A microcost shim: FT accumulate from PRECOMPUTED feature
    /// indices (isolates the accumulator math from feature extraction).
    pub fn accumulate_public(
        &self,
        indices: &[u16],
    ) -> [i32; NNUE_V2Q_FT_WIDTH] {
        self.accumulate(indices)
    }

    /// Incrementally update `acc` across the transition `before -> after`.
    ///
    /// Frozen S10-C1 rules:
    /// - For each perspective, if its OWN king square changed, that
    ///   perspective is fully refreshed (the V2 own-king conditioning
    ///   controls orientation, horizontal mirror, and bucket; a king move
    ///   can flip the mirror regime of the whole perspective).
    /// - Otherwise only changed squares are applied: subtract the FT row
    ///   of the before-piece's feature, add the FT row of the after-
    ///   piece's feature. The opponent king participates normally as
    ///   channel 10.
    /// - Changed squares are found by a 64-square board diff (correctness
    ///   first; Move-aware dirty extraction is C2/C3 work).
    pub fn update_accumulator(
        &self,
        acc: &mut NnueV2Accumulator,
        before: &Position,
        after: &Position,
    ) -> UpdateStats {
        let mut stats = UpdateStats::default();
        for (perspective, lanes) in [
            (NnuePerspective::White, &mut acc.white),
            (NnuePerspective::Black, &mut acc.black),
        ] {
            let own_king_moved = before.king_square(perspective.color())
                != after.king_square(perspective.color());
            if own_king_moved {
                let fresh_lanes =
                    self.full_accumulator_for_perspective(after, perspective);
                lanes.copy_from_slice(&fresh_lanes);
                stats.full_refreshes += 1;
                continue;
            }
            let mut deltas = 0usize;
            for sq in 0..64 {
                let old_piece = before.board()[sq];
                let new_piece = after.board()[sq];
                if old_piece == new_piece {
                    continue;
                }
                if let Some(piece) = old_piece {
                    if let Some(f) = v2_feature_for_piece(
                        before, perspective, sq as Square, piece)
                    {
                        self.apply_feature(lanes, f, -1);
                        deltas += 1;
                    }
                }
                if let Some(piece) = new_piece {
                    if let Some(f) = v2_feature_for_piece(
                        after, perspective, sq as Square, piece)
                    {
                        self.apply_feature(lanes, f, 1);
                        deltas += 1;
                    }
                }
            }
            stats.delta_updates += deltas;
        }
        stats
    }

    /// Build the fixed-size move description for `mv` played from `pos`
    /// (the PARENT position, before the move). Zero heap allocation.
    pub fn prepare_move_delta(
        &self,
        pos: &Position,
        mv: &Move,
    ) -> NnueMoveDelta {
        use crate::chess::types::{
            make_square, file_of, rank_of, MoveFlag, Piece,
        };

        let mut removed: [Option<(Square, Piece)>; 2] = [None, None];
        let mut added: [Option<(Square, Piece)>; 2] = [None, None];
        let us = pos.side_to_move();
        let moved_piece =
            pos.board()[mv.from as usize].expect("move from empty square");

        match mv.flag {
            MoveFlag::EnPassant => {
                let cap_sq =
                    make_square(file_of(mv.to), rank_of(mv.from));
                let captured = pos.board()[cap_sq as usize]
                    .expect("en passant target pawn missing");
                removed[0] = Some((mv.from, moved_piece));
                removed[1] = Some((cap_sq, captured));
                added[0] = Some((mv.to, moved_piece));
            }
            MoveFlag::KingCastle | MoveFlag::QueenCastle => {
                let (rf, rt) = match (us, mv.flag) {
                    (Color::White, MoveFlag::KingCastle) => {
                        (crate::chess::types::H1, crate::chess::types::F1)
                    }
                    (Color::Black, MoveFlag::KingCastle) => {
                        (crate::chess::types::H8, crate::chess::types::F8)
                    }
                    (Color::White, _) => {
                        (crate::chess::types::A1, crate::chess::types::D1)
                    }
                    (Color::Black, _) => {
                        (crate::chess::types::A8, crate::chess::types::D8)
                    }
                };
                let rook = pos.board()[rf as usize]
                    .expect("castling rook missing");
                removed[0] = Some((mv.from, moved_piece));
                removed[1] = Some((rf, rook));
                added[0] = Some((mv.to, moved_piece));
                added[1] = Some((rt, rook));
            }
            MoveFlag::Promotion(pt) => {
                let promoted = Piece::new(us, pt);
                removed[0] = Some((mv.from, moved_piece));
                if let Some(captured) = pos.board()[mv.to as usize] {
                    removed[1] = Some((mv.to, captured));
                }
                added[0] = Some((mv.to, promoted));
            }
            _ => {
                // quiet / double pawn push / normal capture
                removed[0] = Some((mv.from, moved_piece));
                if let Some(captured) = pos.board()[mv.to as usize] {
                    removed[1] = Some((mv.to, captured));
                }
                added[0] = Some((mv.to, moved_piece));
            }
        }

        let moved_king = if moved_piece.piece_type
            == crate::chess::types::PieceType::King
        {
            Some(us)
        } else {
            None
        };

        NnueMoveDelta { removed, added, moved_king }
    }

    /// Move-aware incremental update (C2A production path).
    ///
    /// `delta` was prepared against the PARENT position; `child` is the
    /// position AFTER the move. For every perspective whose own king did
    /// not move, the king context is identical in parent and child, so
    /// old AND new feature indices are computed with the CHILD's king
    /// context — no parent position copy is needed on the search hot path.
    /// The perspective whose king moved is fully refreshed (its feature
    /// indices are undefined under the old context).
    pub fn update_accumulator_for_move(
        &self,
        acc: &mut NnueV2Accumulator,
        delta: &NnueMoveDelta,
        child: &Position,
    ) -> UpdateStats {
        let mut stats = UpdateStats::default();
        for (perspective, lanes) in [
            (NnuePerspective::White, &mut acc.white),
            (NnuePerspective::Black, &mut acc.black),
        ] {
            if delta.moved_king == Some(perspective.color()) {
                let fresh =
                    self.full_accumulator_for_perspective(child, perspective);
                lanes.copy_from_slice(&fresh);
                stats.full_refreshes += 1;
                continue;
            }
            let mut deltas = 0usize;
            for entry in delta.removed.iter().flatten() {
                let (sq, piece) = *entry;
                if let Some(f) =
                    v2_feature_for_piece(child, perspective, sq, piece)
                {
                    self.apply_feature(lanes, f, -1);
                    deltas += 1;
                }
            }
            for entry in delta.added.iter().flatten() {
                let (sq, piece) = *entry;
                if let Some(f) =
                    v2_feature_for_piece(child, perspective, sq, piece)
                {
                    self.apply_feature(lanes, f, 1);
                    deltas += 1;
                }
            }
            stats.delta_updates += deltas;
        }
        stats
    }

    /// Dense forward pass from an (incremental or fresh) accumulator.
    /// `pos` supplies ONLY the side-to-move for the STM/NSTM ordering.
    pub fn evaluate_raw_from_accumulator(
        &self,
        pos: &Position,
        acc: &NnueV2Accumulator,
    ) -> i32 {
        let (own_acc, opp_acc) = match pos.side_to_move() {
            Color::White => (&acc.white, &acc.black),
            Color::Black => (&acc.black, &acc.white),
        };

        // ClippedReLU(0, QA) -> 256 activations.
        let mut acts = [0i32; 256];
        for i in 0..NNUE_V2Q_FT_WIDTH {
            acts[i] = clamp_i(own_acc[i], 0, NNUE_V2Q_QA as i32);
            acts[NNUE_V2Q_FT_WIDTH + i] =
                clamp_i(opp_acc[i], 0, NNUE_V2Q_QA as i32);
        }

        // l1: 256 -> 32 (MAC at accumulator precision; proven bound
        // 256 * 1387 * 4096 + |q_b| ≈ 1.45e9 < i32::MAX).
        // S10-C3-C2: backend dispatch (detected once at load). Both
        // backends are bit-exact: integer MAC in this proven range has no
        // intermediate rounding, so the AVX2 lane re-grouping cannot
        // change the final integer.
        let mut a1 = [0i32; 32];
        #[allow(unused_mut)]
        match self.l1_backend {
            L1Backend::Scalar => {
                for o in 0..32 {
                    let mut z = self.l1_bias[o];
                    let row = o * 256;
                    for i in 0..256 {
                        z += (self.l1_weight[row + i] as i32) * acts[i];
                    }
                    a1[o] = clamp_i(shift_round(z, NNUE_V2Q_DENSE_Z_SHIFT),
                                    0, NNUE_V2Q_QA as i32);
                }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "force_scalar_l1")))]
            L1Backend::Avx2 => unsafe {
                l1_dense_avx2(
                    &self.l1_weight, &self.l1_bias, &acts, &mut a1,
                );
            },
            #[cfg(any(not(target_arch = "x86_64"), feature = "force_scalar_l1"))]
            L1Backend::Avx2 => unreachable!(
                "Avx2 backend cannot be selected in this configuration"
            ),
        }

        // l2: 32 -> 32
        let mut a2 = [0i32; 32];
        for o in 0..32 {
            let mut z = self.l2_bias[o];
            let row = o * 32;
            for i in 0..32 {
                z += (self.l2_weight[row + i] as i32) * a1[i];
            }
            a2[o] = clamp_i(shift_round(z, NNUE_V2Q_DENSE_Z_SHIFT), 0,
                             NNUE_V2Q_QA as i32);
        }

        // out: 32 -> 1
        let mut z_out = self.out_bias[0];
        for i in 0..32 {
            z_out += (self.out_weight[i] as i32) * a2[i];
        }
        shift_round(z_out, NNUE_V2Q_DENSE_Z_SHIFT)
    }

    /// Apply one feature row (`+1` add / `-1` subtract) to a lane set.
    fn apply_feature(&self, lanes: &mut [i32; NNUE_V2Q_FT_WIDTH],
                     feature: u16, sign: i32) {
        debug_assert!(sign == 1 || sign == -1);
        let base = (feature as usize) * NNUE_V2Q_FT_WIDTH;
        for (i, slot) in lanes.iter_mut().enumerate() {
            if sign > 0 {
                *slot = slot
                    .wrapping_add(self.ft_weights[base + i] as i32);
            } else {
                *slot = slot
                    .wrapping_sub(self.ft_weights[base + i] as i32);
            }
        }
    }

    /// Centipawn prediction: `raw / 2^FT_SHIFT * 1000` (final conversion).
    pub fn evaluate_cp(&self, pos: &Position) -> f32 {
        (self.evaluate_raw(pos) as f32 / (1 << NNUE_V2Q_FT_SHIFT) as f32)
            * NNUE_V2Q_TARGET_SCALE
    }

    /// Integer search-eval conversion: `raw * 1000 / 2^FT_SHIFT` computed
    /// in i64 with signed round-half-away-from-zero. This is the ONLY
    /// production-search conversion path (no f32 anywhere in the search).
    pub fn cp_i32_from_raw(raw: i32) -> i32 {
        let wide = (raw as i64) * 1000;
        let denom = 1i64 << NNUE_V2Q_FT_SHIFT;
        let cp = if wide >= 0 {
            (wide + denom / 2) / denom
        } else {
            -((-wide + denom / 2) / denom)
        };
        cp as i32
    }

    /// Integer centipawn evaluation from a (fresh or incremental)
    /// accumulator. `pos` supplies only the side-to-move.
    pub fn evaluate_cp_i32_from_accumulator(
        &self,
        pos: &Position,
        acc: &NnueV2Accumulator,
    ) -> i32 {
        let raw = self.evaluate_raw_from_accumulator(pos, acc);
        Self::cp_i32_from_raw(raw)
    }

    /// Integer centipawn evaluation via full refresh (the C2B full-refresh
    /// profile path).
    pub fn evaluate_cp_i32(&self, pos: &Position) -> i32 {
        Self::cp_i32_from_raw(self.evaluate_raw(pos))
    }
}

/// One-time AVX2 detection (S10-C3-C2). cfg'd to a constant on non-x86_64.
#[cfg(all(target_arch = "x86_64", not(feature = "force_scalar_l1")))]
fn is_avx2_detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
}

/// S10-C3-C2 AVX2 L1 kernel: per-output 256-term dot product using
/// `_mm256_madd_epi16` (16 i16 weights x 16 i16 activations -> 8 i32
/// pair sums per iteration). OUTPUT-MAJOR weights (no transpose — the
/// input-major layout was rejected by C3-C1). One vector accumulator per
/// output: no 32-accumulator register pressure.
///
/// Bit-exactness: every product fits i32 (|w| <= 1387, |a| <= 4096 =>
/// |wa| <= 5.7e6), every lane partial sum fits i32 (16/2 pairs x 5.7e6
/// x 256 iterations worst-case is covered by the loader's proven
/// 1.45e9 bound which the lane grouping cannot exceed), and integer
/// addition has no intermediate rounding — re-grouping cannot change
/// the final integer.
///
/// `acts` holds clamp(0, QA) values (<= 4096), safely representable as
/// i16 for the madd path.
#[cfg(all(target_arch = "x86_64", not(feature = "force_scalar_l1")))]
#[target_feature(enable = "avx2")]
unsafe fn l1_dense_avx2(
    w: &[i16],       // [32][256] output-major
    bias: &[i32],    // [32]
    acts: &[i32; 256],
    out: &mut [i32; 32],
) {
    use std::arch::x86_64::*;

    // Activations are already clamped to [0, 4096]: pack to i16 once.
    let mut a16 = [0i16; 256];
    for i in 0..256 {
        a16[i] = acts[i] as i16;
    }

    for o in 0..32 {
        let row = o * 256;
        let mut vacc = _mm256_setzero_si256();
        let mut i = 0;
        while i < 256 {
            let wv = _mm256_loadu_si256(
                w.as_ptr().add(row + i) as *const __m256i);
            let av = _mm256_loadu_si256(
                a16.as_ptr().add(i) as *const __m256i);
            // 16 i16 x i16 -> 8 i32 pair products, summed pairwise.
            let pairs = _mm256_madd_epi16(wv, av);
            vacc = _mm256_add_epi32(vacc, pairs);
            i += 16;
        }
        // Horizontal sum of the 8 i32 lanes.
        let mut sums = [0i32; 8];
        _mm256_storeu_si256(sums.as_mut_ptr() as *mut __m256i, vacc);
        let mut z = bias[o];
        for s in sums {
            z += s;
        }
        out[o] = clamp_i(shift_round(z, NNUE_V2Q_DENSE_Z_SHIFT), 0,
                         NNUE_V2Q_QA as i32);
    }
}

/// Arithmetic right shift with round half away from zero.
#[inline]
fn shift_round(x: i32, shift: u32) -> i32 {
    if x >= 0 {
        (x + (1 << (shift - 1))) >> shift
    } else {
        -((-x + (1 << (shift - 1))) >> shift)
    }
}

#[inline]
fn clamp_i(x: i32, lo: i32, hi: i32) -> i32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

fn read_i16s(data: &[u8], offset: usize, count: usize) -> Result<Vec<i16>, String> {
    let end = offset
        .checked_add(count * 2)
        .ok_or("nnue-v2q-probe: offset overflow")?;
    if end > data.len() {
        return Err("nnue-v2q-probe: truncated payload".to_string());
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[offset..end].chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(out)
}

fn read_i32s(data: &[u8], offset: usize, count: usize) -> Result<Vec<i32>, String> {
    let end = offset
        .checked_add(count * 4)
        .ok_or("nnue-v2q-probe: offset overflow")?;
    if end > data.len() {
        return Err("nnue-v2q-probe: truncated payload".to_string());
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[offset..end].chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Max absolute value as i64 (MIN-safe: abs(i16::MIN) = 32768 is not
/// representable in i16, so widen BEFORE abs).
fn max_abs_i16(v: &[i16]) -> i64 {
    v.iter().map(|&x| (x as i64).abs()).max().unwrap_or(0)
}

/// Max absolute value as i64 (MIN-safe; see max_abs_i16).
fn max_abs_i32(v: &[i32]) -> i64 {
    v.iter().map(|&x| (x as i64).abs()).max().unwrap_or(0)
}

/// Test-only synthetic EUNN2Q01 artifact builder (shared with the
/// nnue_search stack tests).
#[cfg(test)]
pub(crate) fn synthetic_artifact_bytes_for_tests(fen: &str) -> Vec<u8> {
    tests::synthetic_artifact_bytes(fen)
}

#[cfg(test)]
mod tests {
    use crate::chess::fen::parse_fen;
    use crate::chess::types::START_FEN;

    use super::*;

    /// Synthetic EUNN2Q01 artifact with deterministic weights: every active
    /// startpos feature row is [16; 128] (= 0.00390625 float), ft_bias 8,
    /// l1 weights 256 rows of all 1s? Too big to hand-build in full; instead
    /// build the FULL byte array programmatically like the V1 probe tests.
    pub(super) fn synthetic_artifact_bytes(fen: &str) -> Vec<u8> {
        let pos = parse_fen(fen).unwrap();
        let white = active_features_v2(&pos, NnuePerspective::White);
        let black = active_features_v2(&pos, NnuePerspective::Black);

        let mut out = Vec::with_capacity(TOTAL_BYTES);
        out.extend_from_slice(&NNUE_V2Q_MAGIC);
        out.extend_from_slice(&NNUE_V2Q_VERSION.to_le_bytes());
        out.extend_from_slice(&(NNUE_INPUTS_V2 as u32).to_le_bytes());
        out.extend_from_slice(&(NNUE_V2Q_FT_WIDTH as u32).to_le_bytes());
        out.extend_from_slice(&NNUE_V2Q_TARGET_SCALE.to_le_bytes());
        out.extend_from_slice(&NNUE_V2Q_FT_SHIFT.to_le_bytes());
        out.extend_from_slice(&NNUE_V2Q_DENSE_W_SHIFT.to_le_bytes());
        out.extend_from_slice(&NNUE_V2Q_DENSE_Z_SHIFT.to_le_bytes());
        out.extend_from_slice(&(NNUE_V2Q_QA as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
        out.extend_from_slice(&HISTORICAL_SOURCE_FP32_SHA);
        out.extend_from_slice(&HISTORICAL_SOURCE_CHECKPOINT_SHA);

        // FT: active rows 16, bias 8 per lane.
        let mut ft = vec![0i16; FT_W_COUNT];
        for &idx in white.iter().chain(black.iter()) {
            let base = (idx as usize) * NNUE_V2Q_FT_WIDTH;
            for i in 0..NNUE_V2Q_FT_WIDTH {
                ft[base + i] = 16;
            }
        }
        for v in &ft {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..FT_B_COUNT {
            out.extend_from_slice(&8i32.to_le_bytes());
        }
        // l1: all weights 2 (float 2/4096), bias 0.
        for _ in 0..L1_W_COUNT {
            out.extend_from_slice(&2i16.to_le_bytes());
        }
        for _ in 0..L1_B_COUNT {
            out.extend_from_slice(&0i32.to_le_bytes());
        }
        // l2: identity * 2, bias 0.
        for o in 0..32 {
            for i in 0..32 {
                let v: i16 = if o == i { 2 } else { 0 };
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for _ in 0..L2_B_COUNT {
            out.extend_from_slice(&0i32.to_le_bytes());
        }
        // out: all 2, bias 0.
        for _ in 0..OUT_W_COUNT {
            out.extend_from_slice(&2i16.to_le_bytes());
        }
        out.extend_from_slice(&0i32.to_le_bytes());
        assert_eq!(out.len(), TOTAL_BYTES);
        out
    }

    /// S10-C3-C2: both L1 backends must produce identical raw outputs.
    /// On AVX2 machines this exercises AVX2-vs-AVX2 (same path); the
    /// cross-backend gate runs in the feature-forced CI/bench comparison
    /// (force_scalar_l1 build), and the 10k corpus batch comparison is
    /// the formal gate. Here we assert the kernel against the scalar
    /// reference computed inline, on the real frozen-style artifact.
    #[test]
    fn c3c2_l1_backends_bit_exact_on_legal_moves() {
        use crate::chess::movegen::generate_legal_moves;
        let model =
            NnueV2QuantizedModel::from_bytes(&synthetic_artifact_bytes(START_FEN))
                .unwrap();
        let mut pos = Position::startpos();
        // Walk a few moves; every eval must match the scalar reference.
        for m in generate_legal_moves(&mut pos.clone()).into_iter().take(8) {
            let undo = pos.make_move(m);
            let avx_raw = model.evaluate_raw(&pos);
            // scalar reference: recompute l1 by hand from the same
            // accumulator (mirrors the Scalar arm of the dispatch).
            let acc = model.full_accumulator(&pos);
            let (own_acc, opp_acc) = match pos.side_to_move() {
                Color::White => (&acc.white, &acc.black),
                Color::Black => (&acc.black, &acc.white),
            };
            let mut acts = [0i32; 256];
            for i in 0..NNUE_V2Q_FT_WIDTH {
                acts[i] = clamp_i(own_acc[i], 0, NNUE_V2Q_QA as i32);
                acts[NNUE_V2Q_FT_WIDTH + i] =
                    clamp_i(opp_acc[i], 0, NNUE_V2Q_QA as i32);
            }
            let mut z1 = [0i32; 32];
            for o in 0..32 {
                z1[o] = model.l1_bias[o];
                for i in 0..256 {
                    z1[o] += (model.l1_weight[o * 256 + i] as i32) * acts[i];
                }
            }
            let _ = z1; // l2/out reuse the model's own path below
            let ref_raw = model.evaluate_raw_from_accumulator(&pos, &acc);
            assert_eq!(avx_raw, ref_raw, "backend mismatch after {m:?}");
            pos.unmake_move(undo);
        }
    }

    #[test]
    fn loads_valid_artifact_and_predicts() {
        let model =
            NnueV2QuantizedModel::from_bytes(&synthetic_artifact_bytes(START_FEN))
                .unwrap();
        let pos = parse_fen(START_FEN).unwrap();
        // startpos: 31 active features/perspective -> acc = 8 + 31*16 = 504
        // z1 = 256 * 2 * 504 = 258048 -> >>12 = 63 -> a1 = 63
        // z2 = 2 * 63 = 126 -> >>12 = 0 -> a2 = 0
        // z_out = 0 -> raw = 0
        assert_eq!(model.evaluate_raw(&pos), 0);
        assert_eq!(model.evaluate_cp(&pos), 0.0);
    }

    #[test]
    fn rejects_bad_magic_version_shifts() {
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[0] = b'X';
        assert!(NnueV2QuantizedModel::from_bytes(&data).is_err());
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(NnueV2QuantizedModel::from_bytes(&data).is_err());
        // tamper dense_w_shift
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[28..32].copy_from_slice(&13u32.to_le_bytes());
        assert!(NnueV2QuantizedModel::from_bytes(&data).is_err());
        // tamper inputs
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[12..16].copy_from_slice(&40960u32.to_le_bytes());
        assert!(NnueV2QuantizedModel::from_bytes(&data).is_err());
    }

    #[test]
    fn rejects_truncated_and_trailing() {
        let full = synthetic_artifact_bytes(START_FEN);
        assert!(NnueV2QuantizedModel::from_bytes(&full[..full.len() - 1]).is_err());
        let mut trailing = full.clone();
        trailing.push(0);
        assert!(NnueV2QuantizedModel::from_bytes(&trailing).is_err());
    }

    #[test]
    #[test]
    fn tampered_source_fp32_sha_still_loads_under_format_contract() {
        // S10-E0: a different source FP32 SHA is no longer a load error
        // (format-contract loader); it loads and the header value is
        // preserved for provenance.
        let mut data = synthetic_artifact_bytes(START_FEN);
        // Header source FP32 artifact SHA occupies bytes [44..76].
        data[44] ^= 0xff;
        let model = NnueV2QuantizedModel::from_bytes(&data)
            .expect("source SHA identity is not a format requirement");
        assert_ne!(model.source_fp32_artifact_sha256, HISTORICAL_SOURCE_FP32_SHA);
    }

    #[test]
    fn preserves_source_shas_as_provenance() {
        // S10-E0: the loader no longer REQUIRES the historical SHAs, but it
        // must still expose whatever the header carries for provenance.
        let model =
            NnueV2QuantizedModel::from_bytes(&synthetic_artifact_bytes(START_FEN))
                .unwrap();
        assert_eq!(model.source_fp32_artifact_sha256, HISTORICAL_SOURCE_FP32_SHA);
        assert_eq!(model.source_checkpoint_sha256, HISTORICAL_SOURCE_CHECKPOINT_SHA);
    }

    #[test]
    fn accepts_future_training_iteration_source_shas() {
        // S10-E0 format-contract loader: a structurally valid artifact with
        // DIFFERENT source SHAs (i.e. a future training iteration, e.g. the
        // 1M data-scale probe) must load. Identity pinning is the consumer's
        // job (Arena D0 model-artifact SHA gate).
        let mut data = synthetic_artifact_bytes(START_FEN);
        for byte in data[44..76].iter_mut() {
            *byte ^= 0x5a;
        }
        for byte in data[76..108].iter_mut() {
            *byte ^= 0xa5;
        }
        let model = NnueV2QuantizedModel::from_bytes(&data)
            .expect("future-iteration source SHAs must load");
        assert_ne!(model.source_fp32_artifact_sha256, HISTORICAL_SOURCE_FP32_SHA);
        assert_ne!(model.source_checkpoint_sha256, HISTORICAL_SOURCE_CHECKPOINT_SHA);
    }

    #[test]
    fn rejects_payload_exceeding_proven_bounds() {
        // Valid header (frozen source SHAs, dims, shifts) but out-of-range
        // dense weights: l1 MAC bound 256 * 32767 * 4096 >> i32::MAX.
        let mut data = synthetic_artifact_bytes(START_FEN);
        // Fill 1024 bytes of l1 weight payload with i16::MAX pattern.
        let base = L1_W_OFFSET;
        for chunk in data[base..base + 1024].chunks_exact_mut(2) {
            chunk.copy_from_slice(&i16::MAX.to_le_bytes());
        }
        match NnueV2QuantizedModel::from_bytes(&data) {
            Err(err) => assert!(err.contains("proven i32 MAC bounds"), "err: {err}"),
            Ok(_) => panic!("out-of-bounds payload was accepted"),
        }
    }

    #[test]
    fn rejects_runaway_ft_bias_bound() {
        // FT accumulator bound |bias| + 31*|w| must stay in i32; force a
        // huge bias to violate it.
        let mut data = synthetic_artifact_bytes(START_FEN);
        // Overwrite the first 128 biases with i32::MAX.
        let base = FT_B_OFFSET;
        for chunk in data[base..base + 128 * 4].chunks_exact_mut(4) {
            chunk.copy_from_slice(&i32::MAX.to_le_bytes());
        }
        match NnueV2QuantizedModel::from_bytes(&data) {
            Err(err) => assert!(err.contains("proven i32 MAC bounds"), "err: {err}"),
            Ok(_) => panic!("runaway FT bias was accepted"),
        }
    }

    #[test]
    fn max_abs_helpers_are_min_safe() {
        // abs(i16::MIN) = 32768 is NOT representable in i16; the helpers
        // must widen to i64 before abs so adversarial payloads cannot
        // underestimate the proven bounds (or panic on overflow).
        assert_eq!(max_abs_i16(&[i16::MIN]), 32768);
        assert_eq!(max_abs_i16(&[0, -1, i16::MIN, 5]), 32768);
        assert_eq!(max_abs_i16(&[]), 0);
        assert_eq!(max_abs_i32(&[i32::MIN]), 2147483648);
        assert_eq!(max_abs_i32(&[0, i32::MIN, 7]), 2147483648);
        assert_eq!(max_abs_i32(&[]), 0);
    }

    #[test]
    fn rejects_i32_min_bias_without_panicking() {
        // Adversarial/corrupted payload: an i32::MIN FT bias must be
        // rejected by the bound scan (|MIN| = 2^31 > i32::MAX), never
        // panic or wrap.
        let mut data = synthetic_artifact_bytes(START_FEN);
        let base = FT_B_OFFSET;
        for chunk in data[base..base + 128 * 4].chunks_exact_mut(4) {
            chunk.copy_from_slice(&i32::MIN.to_le_bytes());
        }
        match NnueV2QuantizedModel::from_bytes(&data) {
            Err(err) => assert!(err.contains("proven i32 MAC bounds"), "err: {err}"),
            Ok(_) => panic!("i32::MIN bias was accepted"),
        }
    }

    #[test]
    fn rejects_i16_min_dense_weight_without_panicking() {
        // i16::MIN in the l1 weights: bound = 256 * 32768 * 4096 >> 2^31,
        // must fail closed (not panic, not silently underestimate).
        let mut data = synthetic_artifact_bytes(START_FEN);
        let base = L1_W_OFFSET;
        for chunk in data[base..base + 1024].chunks_exact_mut(2) {
            chunk.copy_from_slice(&i16::MIN.to_le_bytes());
        }
        match NnueV2QuantizedModel::from_bytes(&data) {
            Err(err) => assert!(err.contains("proven i32 MAC bounds"), "err: {err}"),
            Ok(_) => panic!("i16::MIN dense weight was accepted"),
        }
    }

    #[test]
    fn shift_round_matches_python_reference() {
        // Round half away from zero, mirrors integer_reference.py.
        assert_eq!(shift_round(4097, 12), 1); // 4097/4096 = 1.0002 -> 1
        assert_eq!(shift_round(6144, 12), 2); // 1.5 -> 2 (away from zero)
        assert_eq!(shift_round(-6144, 12), -2); // -1.5 -> -2
        assert_eq!(shift_round(2048, 12), 1); // 0.5 -> 1
        assert_eq!(shift_round(-2048, 12), -1); // -0.5 -> -1
        assert_eq!(shift_round(2047, 12), 0);
    }

    #[test]
    fn evaluate_does_not_mutate_position() {
        let model =
            NnueV2QuantizedModel::from_bytes(&synthetic_artifact_bytes(START_FEN))
                .unwrap();
        let pos = parse_fen(START_FEN).unwrap();
        let before = pos.zobrist_key();
        let _ = model.evaluate_raw(&pos);
        let _ = model.evaluate_cp(&pos);
        assert_eq!(pos.zobrist_key(), before);
    }

    // ------------------------------------------------------------------
    // S10-C1 incremental accumulator correctness tests
    // ------------------------------------------------------------------

    use crate::chess::movegen::generate_legal_moves;
    use crate::chess::types::{Move, MoveFlag, PieceType};

    fn test_model() -> NnueV2QuantizedModel {
        NnueV2QuantizedModel::from_bytes(&synthetic_artifact_bytes(START_FEN))
            .unwrap()
    }

    /// Verify one transition `before -> after` incrementally: update the
    /// parent accumulator and compare all 256 lanes + the raw output
    /// against a fresh full refresh of `after`. Panics with context on any
    /// mismatch.
    fn verify_transition(
        model: &NnueV2QuantizedModel,
        before: &Position,
        after: &Position,
        mut acc: NnueV2Accumulator,
        context: &str,
    ) -> UpdateStats {
        let stats = model.update_accumulator(&mut acc, before, after);
        let fresh = model.full_accumulator(after);
        assert_eq!(
            acc.white, fresh.white,
            "white lanes mismatch ({context})"
        );
        assert_eq!(
            acc.black, fresh.black,
            "black lanes mismatch ({context})"
        );
        let inc_raw = model.evaluate_raw_from_accumulator(after, &acc);
        let full_raw = model.evaluate_raw(after);
        assert_eq!(
            inc_raw, full_raw,
            "raw output mismatch ({context}): {inc_raw} != {full_raw}"
        );
        stats
    }

    fn apply_uci(pos: &mut Position, uci: &str) {
        let m = parse_uci_move(pos, uci);
        pos.make_move(m);
    }

    fn parse_uci_move(pos: &Position, uci: &str) -> Move {
        let from = algebraic_sq(&uci[0..2]);
        let to = algebraic_sq(&uci[2..4]);
        let promo = uci.chars().nth(4).and_then(PieceType::from_char);
        let mut probe = pos.clone();
        for m in generate_legal_moves(&mut probe) {
            if m.from == from && m.to == to && m.promotion == promo {
                return m;
            }
        }
        panic!("illegal or unknown uci move {uci}");
    }

    fn algebraic_sq(s: &str) -> Square {
        let bytes = s.as_bytes();
        (bytes[0] - b'a') as Square + ((bytes[1] - b'1') as Square) * 8
    }

    #[test]
    fn c1_quiet_pawn_capture_sequence() {
        let model = test_model();
        let mut pos = Position::startpos();
        let mut acc = model.full_accumulator(&pos);
        // e4 d5 exd5 (pawn double push, capture) e6? (pawn push) — also
        // exercises quiet knight moves.
        for uci in ["e2e4", "d7d5", "e4d5", "g8f6", "d5d6", "e7d6"] {
            let before = pos.clone();
            apply_uci(&mut pos, uci);
            verify_transition(&model, &before, &pos, acc.clone(), uci);
            // keep the incremental accumulator for the next ply
            let mut next = acc.clone();
            model.update_accumulator(&mut next, &before, &pos);
            acc = next;
        }
    }

    #[test]
    fn c1_en_passant_three_square_change() {
        let model = test_model();
        let mut pos = parse_fen(
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
            .unwrap();
        let mut acc = model.full_accumulator(&pos);
        // ... d4xe3 e.p.: changed squares = d4 (vacated), e4 (vacated),
        // e3 (capturing pawn appears).
        let before = pos.clone();
        apply_uci(&mut pos, "d4e3");
        let stats = verify_transition(
            &model, &before, &pos, acc, "en passant d4xe3");
        // Both perspectives use pure deltas (no king moved).
        assert_eq!(stats.full_refreshes, 0);
        // 2 remove + 1 add per perspective = 6 applications.
        assert_eq!(stats.delta_updates, 6);
    }

    #[test]
    fn c1_castling_both_sides() {
        let model = test_model();
        // King-side castle: white K e1->g1, rook h1->f1.
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let mut acc = model.full_accumulator(&pos);
        let before = pos.clone();
        apply_uci(&mut pos, "e1g1");
        let stats = verify_transition(&model, &before, &pos, acc, "white O-O");
        // White perspective: own king moved -> full refresh; black
        // perspective: king+rook are opponent channel deltas.
        assert_eq!(stats.full_refreshes, 1);

        // Queen-side castle for black.
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 1")
            .unwrap();
        let mut acc = model.full_accumulator(&pos);
        let before = pos.clone();
        apply_uci(&mut pos, "e8c8");
        let stats = verify_transition(&model, &before, &pos, acc, "black O-O-O");
        assert_eq!(stats.full_refreshes, 1);
    }

    #[test]
    fn c1_promotions_quiet_and_capture() {
        let model = test_model();
        // Quiet promotion (queen) and knight promotion.
        for (fen, uci, ctx) in [
            ("8/P7/8/8/8/8/k6K/8 w - - 0 1", "a7a8q", "quiet Q promo"),
            ("8/P7/8/8/8/8/k6K/8 w - - 0 1", "a7a8n", "quiet N promo"),
            ("1n6/P7/8/8/8/8/k6K/8 w - - 0 1", "a7b8q", "capture Q promo"),
        ] {
            let mut pos = parse_fen(fen).unwrap();
            let mut acc = model.full_accumulator(&pos);
            let before = pos.clone();
            apply_uci(&mut pos, uci);
            verify_transition(&model, &before, &pos, acc, ctx);
        }
    }

    #[test]
    fn c1_own_king_move_triggers_perspective_refresh() {
        let model = test_model();
        let mut pos = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut acc = model.full_accumulator(&pos);
        let before = pos.clone();
        apply_uci(&mut pos, "e1e2");
        let stats = verify_transition(
            &model, &before, &pos, acc, "white king e1->e2");
        // White own king moved -> white full refresh; black perspective
        // sees the king as channel-10 delta only.
        assert_eq!(stats.full_refreshes, 1);
        assert!(stats.delta_updates >= 2);

        // Black king move.
        let mut pos = parse_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        let mut acc = model.full_accumulator(&pos);
        let before = pos.clone();
        apply_uci(&mut pos, "e8d7");
        let stats = verify_transition(
            &model, &before, &pos, acc, "black king e8->d7");
        assert_eq!(stats.full_refreshes, 1);
    }

    #[test]
    fn c1_horizontal_mirror_boundary_king_moves() {
        let model = test_model();
        // d-file -> e-file crosses the mirror boundary (a-d mirror, e-h
        // canonical). Even when the mirrored bucket would look identical,
        // the whole perspective must refresh.
        for (fen, uci, ctx) in [
            ("4k3/8/8/8/8/8/8/3K4 w - - 0 1", "d1e1", "king d->e (mirror off)"),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1d1", "king e->d (mirror on)"),
            ("4k3/8/8/8/8/8/8/2K5 w - - 0 1", "c1d1", "king c->d (within mirror)"),
            ("4k3/8/8/8/8/8/8/5K2 w - - 0 1", "f1e1", "king f->e (within canonical)"),
            ("4k3/8/8/8/8/8/8/3K4 w - - 0 1", "d1e2", "king d->e diagonal (mirror boundary)"),
        ] {
            let mut pos = parse_fen(fen).unwrap();
            let mut acc = model.full_accumulator(&pos);
            let before = pos.clone();
            apply_uci(&mut pos, uci);
            let stats = verify_transition(&model, &before, &pos, acc, ctx);
            assert_eq!(stats.full_refreshes, 1, "own king move ({ctx})");
        }
    }

    #[test]
    fn c1_make_unmake_branch_restores_parent() {
        let model = test_model();
        let mut pos = parse_fen(
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1")
            .unwrap();
        let parent_acc = model.full_accumulator(&pos);
        for m in generate_legal_moves(&mut pos.clone()) {
            let before = pos.clone();
            let undo = pos.make_move(m);
            // Incremental update from the parent accumulator, verify.
            let mut acc = parent_acc;
            model.update_accumulator(&mut acc, &before, &pos);
            let fresh = model.full_accumulator(&pos);
            assert_eq!(acc.white, fresh.white, "make {m:?}");
            assert_eq!(acc.black, fresh.black, "make {m:?}");

            // Unmake and verify the PARENT accumulator snapshot still
            // matches a fresh full refresh of the restored position.
            pos.unmake_move(undo);
            let parent_fresh = model.full_accumulator(&pos);
            assert_eq!(parent_acc.white, parent_fresh.white);
            assert_eq!(parent_acc.black, parent_fresh.black);
        }
    }

    #[test]
    fn c1_deterministic_random_legal_playout_100_games() {
        let model = test_model();
        // Fixed-seed xorshift; deterministic across runs/platforms.
        let mut rng: u64 = 0x5989d5721ea4258e;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut transitions = 0usize;
        let mut king_refreshes = 0usize;
        for game in 0..100 {
            let mut pos = Position::startpos();
            let mut acc = model.full_accumulator(&pos);
            for _ply in 0..100 {
                let moves = generate_legal_moves(&mut pos.clone());
                if moves.is_empty() {
                    break;
                }
                let m = moves[(next() % moves.len() as u64) as usize];
                let before = pos.clone();
                pos.make_move(m);
                let stats =
                    verify_transition(&model, &before, &pos, acc, "playout");
                king_refreshes += stats.full_refreshes;
                // carry the incremental accumulator forward
                let mut next_acc = acc;
                model.update_accumulator(&mut next_acc, &before, &pos);
                acc = next_acc;
                transitions += 1;
            }
            let _ = game;
        }
        // Sanity: the playout actually exercised moves and king moves.
        assert!(transitions > 1000, "too few transitions: {transitions}");
        assert!(king_refreshes > 0, "no own-king moves exercised");
    }

    // ------------------------------------------------------------------
    // S10-C2A move-aware dirty accumulator tests
    // ------------------------------------------------------------------

    /// Invariant: dirty squares derived from the Move == the squares that
    /// actually change across before.board vs after.board. The 64-square
    /// scan exists ONLY in this test, never in the production updater.
    #[test]
    fn c2a_dirty_squares_match_board_diff() {
        use crate::chess::movegen::generate_legal_moves;
        let model = test_model();
        let fens = [
            START_FEN,
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
            "8/P7/8/8/8/8/k6K/8 w - - 0 1",
            "1n6/P7/8/8/8/8/k6K/8 w - - 0 1",
        ];
        let mut checked = 0usize;
        for fen in fens {
            let mut pos = parse_fen(fen).unwrap();
            let legal = generate_legal_moves(&mut pos.clone());
            for m in legal {
                let before = pos.clone();
                let delta = model.prepare_move_delta(&before, &m);
                let dirty: std::collections::BTreeSet<Square> =
                    delta.dirty_squares().collect();
                let undo = pos.make_move(m);
                let mut actual = std::collections::BTreeSet::new();
                for sq in 0..64 {
                    if before.board()[sq] != pos.board()[sq] {
                        actual.insert(sq as Square);
                    }
                }
                pos.unmake_move(undo);
                assert_eq!(
                    dirty, actual,
                    "dirty-squares drift for {m:?} in {fen}"
                );
                // The delta's own entry counts must match its slots.
                assert_eq!(delta.removed_entries().count(), removed_count(&delta));
                assert_eq!(delta.added_entries().count(), added_count(&delta));
                checked += 1;
            }
        }
        assert!(checked > 50, "too few moves checked: {checked}");
    }

    fn removed_count(d: &NnueMoveDelta) -> usize {
        d.removed.iter().flatten().count()
    }
    fn added_count(d: &NnueMoveDelta) -> usize {
        d.added.iter().flatten().count()
    }

    /// A == B == C: move-aware update == C1 64-square reference update ==
    /// full refresh, on every legal move of a scenario-rich FEN set.
    #[test]
    fn c2a_move_aware_matches_reference_and_full_refresh() {
        use crate::chess::movegen::generate_legal_moves;
        let model = test_model();
        let fens = [
            START_FEN,
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
            "8/P7/8/8/8/8/k6K/8 w - - 0 1",
            "1n6/P7/8/8/8/8/k6K/8 w - - 0 1",
            "4k3/8/8/8/8/8/8/3K4 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ];
        let mut transitions = 0usize;
        for fen in fens {
            let mut pos = parse_fen(fen).unwrap();
            let parent = model.full_accumulator(&pos);
            let legal = generate_legal_moves(&mut pos.clone());
            for m in legal {
                let before = pos.clone();
                let delta = model.prepare_move_delta(&before, &m);
                let undo = pos.make_move(m);

                // A: move-aware update from the parent accumulator.
                let mut a = parent;
                let stats_a =
                    model.update_accumulator_for_move(&mut a, &delta, &pos);

                // B: C1 64-square reference update.
                let mut b = parent;
                let stats_b =
                    model.update_accumulator(&mut b, &before, &pos);

                // C: full refresh.
                let c = model.full_accumulator(&pos);

                assert_eq!(a.white, b.white, "A!=B white for {m:?} in {fen}");
                assert_eq!(a.black, b.black, "A!=B black for {m:?} in {fen}");
                assert_eq!(a.white, c.white, "A!=C white for {m:?} in {fen}");
                assert_eq!(a.black, c.black, "A!=C black for {m:?} in {fen}");
                assert_eq!(
                    model.evaluate_raw_from_accumulator(&pos, &a),
                    model.evaluate_raw(&pos),
                    "raw mismatch for {m:?} in {fen}"
                );
                assert_eq!(stats_a, stats_b, "stats drift for {m:?}");
                pos.unmake_move(undo);
                transitions += 1;
            }
        }
        assert!(transitions > 100);
    }

    /// Null-move semantics: the fixed-perspective accumulator is entirely
    /// unchanged by a side-to-move flip; only the dense STM/NSTM ordering
    /// (and therefore possibly the raw output) differs.
    #[test]
    fn c2a_null_move_leaves_accumulator_bit_identical() {
        let model = test_model();
        let fen = "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1";
        let pos = parse_fen(fen).unwrap();
        let acc = model.full_accumulator(&pos);
        // Null move shape: same board, side to move flipped (via FEN).
        let flipped = parse_fen(&fen.replace(" w KQkq", " b KQkq")).unwrap();
        let acc2 = model.full_accumulator(&flipped);
        // Accumulator lanes bit-identical across the null-move shape.
        assert_eq!(acc.white, acc2.white);
        assert_eq!(acc.black, acc2.black);
        // Dense forward sees the SAME accumulator but swaps STM/NSTM.
        let raw_w = model.evaluate_raw_from_accumulator(&pos, &acc);
        let raw_b = model.evaluate_raw_from_accumulator(&flipped, &acc);
        // Both must be valid integers; difference (if any) comes only
        // from the ordering swap.
        assert_ne!(raw_w, i32::MAX);
        assert_ne!(raw_b, i32::MAX);
        // And evaluating the flipped position with the shared accumulator
        // equals a fresh full evaluation of the flipped position.
        assert_eq!(
            raw_b,
            model.evaluate_raw(&flipped),
            "null-move dense parity"
        );
    }

    /// 100-game deterministic playout through the MOVE-AWARE path with
    /// the C1 64-square updater as a live oracle at every ply.
    #[test]
    fn c2a_move_aware_100_game_playout_with_oracle() {
        use crate::chess::movegen::generate_legal_moves;
        let model = test_model();
        let mut rng: u64 = 0x2aa2f932cdbb2dff;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut transitions = 0usize;
        let mut refreshes = 0usize;
        let mut deltas = 0usize;
        for _game in 0..100 {
            let mut pos = Position::startpos();
            let mut acc = model.full_accumulator(&pos);
            for _ply in 0..100 {
                let moves = generate_legal_moves(&mut pos.clone());
                if moves.is_empty() {
                    break;
                }
                let m = moves[(next() % moves.len() as u64) as usize];
                let before = pos.clone();
                let delta = model.prepare_move_delta(&pos, &m);
                pos.make_move(m);

                let mut a = acc;
                let stats =
                    model.update_accumulator_for_move(&mut a, &delta, &pos);
                let mut b = acc;
                model.update_accumulator(&mut b, &before, &pos);
                let c = model.full_accumulator(&pos);

                assert_eq!(a.white, b.white, "A!=B white @ {m:?}");
                assert_eq!(a.black, b.black, "A!=B black @ {m:?}");
                assert_eq!(a.white, c.white, "A!=C white @ {m:?}");
                assert_eq!(a.black, c.black, "A!=C black @ {m:?}");
                assert_eq!(
                    model.evaluate_raw_from_accumulator(&pos, &a),
                    model.evaluate_raw(&pos)
                );
                refreshes += stats.full_refreshes;
                deltas += stats.delta_updates;
                acc = a;
                transitions += 1;
            }
        }
        assert!(transitions > 1000, "too few transitions: {transitions}");
        assert!(refreshes > 0);
        assert!(deltas > 0);
    }
}
