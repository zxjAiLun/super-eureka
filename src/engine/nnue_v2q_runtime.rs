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
use crate::chess::types::Color;
use crate::engine::nnue::{active_features_v2, NnuePerspective, NNUE_INPUTS_V2};

/// Fixed S10-B5 artifact constants (must match export_quantized.py).
pub const NNUE_V2Q_MAGIC: [u8; 8] = *b"EUNN2Q01";
pub const NNUE_V2Q_VERSION: u32 = 1;
pub const NNUE_V2Q_FT_WIDTH: usize = 128;
pub const NNUE_V2Q_TARGET_SCALE: f32 = 1000.0;
pub const NNUE_V2Q_FT_SHIFT: u32 = 12;
pub const NNUE_V2Q_DENSE_W_SHIFT: u32 = 12;
pub const NNUE_V2Q_DENSE_Z_SHIFT: u32 = 12;
pub const NNUE_V2Q_QA: usize = 1 << 12;

/// Frozen provenance: the EUNN2Q01 artifact must have been exported from
/// the B3 production checkpoint (d59ad852...) via the B4 FP32 artifact
/// (9bf7addd...). The loader rejects anything else.
const EXPECTED_SOURCE_FP32_SHA: [u8; 32] = [
    0x9b, 0xf7, 0xad, 0xdd, 0xf7, 0xb3, 0xb4, 0x4a,
    0xff, 0xa5, 0xe2, 0x6d, 0x22, 0x76, 0xb1, 0x3d,
    0x74, 0x56, 0x61, 0x91, 0xa4, 0xeb, 0x4d, 0x00,
    0x90, 0xfb, 0xde, 0x5a, 0x7a, 0xfb, 0xc9, 0xfc,
];
const EXPECTED_SOURCE_CHECKPOINT_SHA: [u8; 32] = [
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

/// Loaded S10-B5 quantized model. Immutable after construction.
pub struct NnueV2QuantizedModel {
    ft_weights: Vec<i16>,  // [22528][128]
    ft_bias: Vec<i32>,     // [128]
    l1_weight: Vec<i16>,   // [32][256]
    l1_bias: Vec<i32>,     // [32]
    l2_weight: Vec<i16>,   // [32][32]
    l2_bias: Vec<i32>,     // [32]
    out_weight: Vec<i16>,  // [1][32]
    out_bias: Vec<i32>,    // [1]
    /// Verified-equal to EXPECTED_SOURCE_FP32_SHA at load time.
    #[allow(dead_code)]
    source_fp32_artifact_sha256: [u8; 32],
    /// Verified-equal to EXPECTED_SOURCE_CHECKPOINT_SHA at load time.
    #[allow(dead_code)]
    source_checkpoint_sha256: [u8; 32],
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
        // Fail-closed provenance: the artifact must have been exported from
        // the frozen B3/B4 sources.
        if source_fp32_artifact_sha256 != EXPECTED_SOURCE_FP32_SHA {
            return Err(
                "nnue-v2q-probe: source FP32 artifact SHA mismatch".to_string()
            );
        }
        if source_checkpoint_sha256 != EXPECTED_SOURCE_CHECKPOINT_SHA {
            return Err(
                "nnue-v2q-probe: source checkpoint SHA mismatch".to_string()
            );
        }

        let ft_weights = read_i16s(data, FT_W_OFFSET, FT_W_COUNT)?;
        let ft_bias = read_i32s(data, FT_B_OFFSET, FT_B_COUNT)?;
        let l1_weight = read_i16s(data, L1_W_OFFSET, L1_W_COUNT)?;
        let l1_bias = read_i32s(data, L1_B_OFFSET, L1_B_COUNT)?;
        let l2_weight = read_i16s(data, L2_W_OFFSET, L2_W_COUNT)?;
        let l2_bias = read_i32s(data, L2_B_OFFSET, L2_B_COUNT)?;
        let out_weight = read_i16s(data, OUT_W_OFFSET, OUT_W_COUNT)?;
        let out_bias = read_i32s(data, OUT_B_OFFSET, OUT_B_COUNT)?;

        // Fail-closed overflow safety: recompute the PROVEN worst-case
        // bounds from the actual payload (i64 arithmetic) and refuse any
        // artifact whose MACs could exceed i32 range. The frozen artifact's
        // bounds are ~1.45e9 max, far below i32::MAX.
        let ft_bound = max_abs_i32(&ft_bias) as i64
            + MAX_FEATURES_PER_PERSPECTIVE as i64 * max_abs_i16(&ft_weights) as i64;
        let l1_bound = max_abs_i32(&l1_bias) as i64
            + 256 * max_abs_i16(&l1_weight) as i64 * NNUE_V2Q_QA as i64;
        let l2_bound = max_abs_i32(&l2_bias) as i64
            + 32 * max_abs_i16(&l2_weight) as i64 * NNUE_V2Q_QA as i64;
        let out_bound = max_abs_i32(&out_bias) as i64
            + 32 * max_abs_i16(&out_weight) as i64 * NNUE_V2Q_QA as i64;
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
        })
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
    pub fn evaluate_raw(&self, pos: &Position) -> i32 {
        let white = active_features_v2(pos, NnuePerspective::White);
        let black = active_features_v2(pos, NnuePerspective::Black);
        let (own, opp) = match pos.side_to_move() {
            Color::White => (white, black),
            Color::Black => (black, white),
        };
        let own_acc = self.accumulate(&own);
        let opp_acc = self.accumulate(&opp);

        // ClippedReLU(0, QA) -> 256 activations, then dense input requant.
        let mut acts = [0i32; 256];
        for i in 0..NNUE_V2Q_FT_WIDTH {
            acts[i] = clamp_i(own_acc[i], 0, NNUE_V2Q_QA as i32);
            acts[NNUE_V2Q_FT_WIDTH + i] =
                clamp_i(opp_acc[i], 0, NNUE_V2Q_QA as i32);
        }

        // l1: 256 -> 32 (MAC at accumulator precision; proven bound
        // 256 * 1387 * 4096 + |q_b| ≈ 1.45e9 < i32::MAX)
        let mut a1 = [0i32; 32];
        for o in 0..32 {
            let mut z = self.l1_bias[o];
            let row = o * 256;
            for i in 0..256 {
                z += (self.l1_weight[row + i] as i32) * acts[i];
            }
            a1[o] = clamp_i(shift_round(z, NNUE_V2Q_DENSE_Z_SHIFT), 0,
                             NNUE_V2Q_QA as i32);
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

    /// Centipawn prediction: `raw / 2^FT_SHIFT * 1000` (final conversion).
    pub fn evaluate_cp(&self, pos: &Position) -> f32 {
        (self.evaluate_raw(pos) as f32 / (1 << NNUE_V2Q_FT_SHIFT) as f32)
            * NNUE_V2Q_TARGET_SCALE
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

fn max_abs_i16(v: &[i16]) -> i16 {
    v.iter().fold(0i16, |m, &x| m.max(x.abs()))
}

fn max_abs_i32(v: &[i32]) -> i32 {
    v.iter().fold(0i32, |m, &x| m.max(x.abs()))
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
    fn synthetic_artifact_bytes(fen: &str) -> Vec<u8> {
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
        out.extend_from_slice(&EXPECTED_SOURCE_FP32_SHA);
        out.extend_from_slice(&EXPECTED_SOURCE_CHECKPOINT_SHA);

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
    fn rejects_tampered_source_fp32_sha() {
        let mut data = synthetic_artifact_bytes(START_FEN);
        // Header source FP32 artifact SHA occupies bytes [44..76].
        data[44] ^= 0xff;
        match NnueV2QuantizedModel::from_bytes(&data) {
            Err(err) => assert!(err.contains("source FP32 artifact SHA")),
            Ok(_) => panic!("tampered source FP32 SHA was accepted"),
        }
    }

    #[test]
    fn rejects_tampered_source_checkpoint_sha() {
        let mut data = synthetic_artifact_bytes(START_FEN);
        // Header source checkpoint SHA occupies bytes [76..108].
        data[76] ^= 0xff;
        match NnueV2QuantizedModel::from_bytes(&data) {
            Err(err) => assert!(err.contains("source checkpoint SHA")),
            Ok(_) => panic!("tampered source checkpoint SHA was accepted"),
        }
    }

    #[test]
    fn accepts_exact_frozen_source_identities() {
        // The synthetic builder writes the frozen SHAs; loading must succeed
        // and expose them for inspection.
        let model =
            NnueV2QuantizedModel::from_bytes(&synthetic_artifact_bytes(START_FEN))
                .unwrap();
        assert_eq!(model.source_fp32_artifact_sha256, EXPECTED_SOURCE_FP32_SHA);
        assert_eq!(model.source_checkpoint_sha256, EXPECTED_SOURCE_CHECKPOINT_SHA);
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
}
