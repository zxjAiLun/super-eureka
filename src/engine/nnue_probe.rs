//! S6-N2 — bench-only NNUE probe runtime (float32 full-refresh inference).
//!
//! This is an isolated, bench-only bridge for the S6-N2 probe artifact
//! (`EUNN1F32` v1). It is NOT a production model format, NOT wired into
//! search/eval/UCI, has no fallback, no accumulator, no quantization, no
//! SIMD, and no global model caching.
//!
//! The feature encoding is NOT reimplemented here: `nnue::active_features()`
//! in [`crate::engine::nnue`] remains the single source of truth.

use crate::chess::position::Position;
use crate::chess::types::Color;
use crate::engine::nnue::{active_features, NnuePerspective, NNUE_INPUTS};

/// Fixed S6-N2 probe artifact constants.
pub const NNUE_PROBE_MAGIC: [u8; 8] = *b"EUNN1F32";
pub const NNUE_PROBE_VERSION: u32 = 1;
pub const NNUE_PROBE_WIDTH: usize = 32;
pub const NNUE_PROBE_TARGET_SCALE: f32 = 1000.0;

const HEADER_BYTES: usize = 8 + 4 + 4 + 4 + 4 + 32;
const FEATURES_COUNT: usize = NNUE_INPUTS * NNUE_PROBE_WIDTH;
const HEADER_AND_FEATURES: usize = HEADER_BYTES + FEATURES_COUNT * 4;
const ACC_BIAS_BYTES: usize = NNUE_PROBE_WIDTH * 4;
const HEAD_WEIGHT_BYTES: usize = NNUE_PROBE_WIDTH * 2 * 4;
const HEAD_BIAS_BYTES: usize = 4;
const TOTAL_BYTES: usize =
    HEADER_AND_FEATURES + ACC_BIAS_BYTES + HEAD_WEIGHT_BYTES + HEAD_BIAS_BYTES;

/// Loaded S6-N2 probe model. Immutable after construction; every inference
/// call performs a full-refresh from `active_features()`.
pub struct NnueProbeModelV1 {
    /// Shared feature table, input-major `[40960][32]`.
    features: Vec<f32>,
    /// Shared 32-dim accumulator bias.
    acc_bias: Vec<f32>,
    /// Head weights `[64]` in own-then-opponent order.
    head_weight: Vec<f32>,
    head_bias: f32,
    target_scale: f32,
    #[allow(dead_code)]
    checkpoint_sha256: [u8; 32],
}

impl NnueProbeModelV1 {
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("nnue-probe: cannot read {}: {e}", path.display()))?;
        Self::from_bytes(&data)
    }

    /// Parse and fully validate an `EUNN1F32` v1 artifact (little-endian).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() != TOTAL_BYTES {
            return Err(format!(
                "nnue-probe: bad length {} != expected {TOTAL_BYTES}",
                data.len()
            ));
        }
        if data[0..8] != NNUE_PROBE_MAGIC {
            return Err("nnue-probe: bad magic".to_string());
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let inputs = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let width = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let target_scale = f32::from_bits(u32::from_le_bytes(data[20..24].try_into().unwrap()));
        if version != NNUE_PROBE_VERSION {
            return Err(format!("nnue-probe: bad version {version}"));
        }
        if inputs != NNUE_INPUTS as u32 {
            return Err(format!("nnue-probe: bad inputs {inputs}"));
        }
        if width != NNUE_PROBE_WIDTH as u32 {
            return Err(format!("nnue-probe: bad width {width}"));
        }
        if target_scale != NNUE_PROBE_TARGET_SCALE {
            return Err(format!("nnue-probe: bad target_scale {target_scale}"));
        }
        let mut checkpoint_sha256 = [0u8; 32];
        checkpoint_sha256.copy_from_slice(&data[24..56]);

        let mut features = read_f32s(data, HEADER_BYTES, FEATURES_COUNT)?;
        let acc_bias = read_f32s(data, HEADER_AND_FEATURES, NNUE_PROBE_WIDTH)?;
        let head_weight = read_f32s(
            data,
            HEADER_AND_FEATURES + ACC_BIAS_BYTES,
            NNUE_PROBE_WIDTH * 2,
        )?;
        let head_bias = read_f32s(
            data,
            HEADER_AND_FEATURES + ACC_BIAS_BYTES + HEAD_WEIGHT_BYTES,
            1,
        )?[0];
        // Finiteness covers every payload float (features, bias, head).
        if !features.iter().all(|x| x.is_finite())
            || !acc_bias.iter().all(|x| x.is_finite())
            || !head_weight.iter().all(|x| x.is_finite())
            || !head_bias.is_finite()
        {
            return Err("nnue-probe: non-finite payload value".to_string());
        }
        features.shrink_to_fit();
        Ok(NnueProbeModelV1 {
            features,
            acc_bias,
            head_weight,
            head_bias,
            target_scale,
            checkpoint_sha256,
        })
    }

    /// Accumulate one perspective: `acc_bias + sum(active feature rows)`.
    fn accumulate(&self, indices: &[u16]) -> [f32; NNUE_PROBE_WIDTH] {
        let mut acc = [0.0f32; NNUE_PROBE_WIDTH];
        acc.copy_from_slice(&self.acc_bias);
        for &idx in indices {
            let base = (idx as usize) * NNUE_PROBE_WIDTH;
            for (i, slot) in acc.iter_mut().enumerate() {
                *slot += self.features[base + i];
            }
        }
        acc
    }

    /// Full-refresh scaled prediction in `[-1, 1]`-ish units (never rounded).
    pub fn evaluate_scaled(&self, pos: &Position) -> f32 {
        let white = active_features(pos, NnuePerspective::White);
        let black = active_features(pos, NnuePerspective::Black);
        let (own, opp) = match pos.side_to_move() {
            Color::White => (white, black),
            Color::Black => (black, white),
        };
        let own_acc = self.accumulate(&own).map(|x| x.max(0.0));
        let opp_acc = self.accumulate(&opp).map(|x| x.max(0.0));
        let mut scaled = self.head_bias;
        for i in 0..NNUE_PROBE_WIDTH {
            scaled += self.head_weight[i] * own_acc[i];
            scaled += self.head_weight[NNUE_PROBE_WIDTH + i] * opp_acc[i];
        }
        scaled
    }

    /// Centipawn prediction: `scaled * target_scale`, no integer rounding.
    pub fn evaluate_cp(&self, pos: &Position) -> f32 {
        self.evaluate_scaled(pos) * self.target_scale
    }
}

/// Read `count` little-endian f32s starting at `offset` (no unsafe).
fn read_f32s(data: &[u8], offset: usize, count: usize) -> Result<Vec<f32>, String> {
    let end = offset
        .checked_add(count * 4)
        .ok_or("nnue-probe: offset overflow")?;
    if end > data.len() {
        return Err("nnue-probe: truncated payload".to_string());
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[offset..end].chunks_exact(4) {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use crate::chess::fen::parse_fen;
    use crate::chess::types::{Color, START_FEN};

    use super::*;

    /// Build a valid artifact where every active-startpos feature row is
    /// `[1.0; 32]`, acc_bias `[0.5; 32]`, head own=1.0 / opp=2.0, bias 0.5.
    fn synthetic_artifact_bytes(fen: &str) -> Vec<u8> {
        let pos = parse_fen(fen).unwrap();
        let white = active_features(&pos, NnuePerspective::White);
        let black = active_features(&pos, NnuePerspective::Black);
        let mut out = Vec::with_capacity(TOTAL_BYTES);
        out.extend_from_slice(&NNUE_PROBE_MAGIC);
        out.extend_from_slice(&NNUE_PROBE_VERSION.to_le_bytes());
        out.extend_from_slice(&(NNUE_INPUTS as u32).to_le_bytes());
        out.extend_from_slice(&(NNUE_PROBE_WIDTH as u32).to_le_bytes());
        out.extend_from_slice(&NNUE_PROBE_TARGET_SCALE.to_le_bytes());
        out.extend_from_slice(&[0xab; 32]);
        let mut features = vec![0.0f32; FEATURES_COUNT];
        for &idx in white.iter().chain(black.iter()) {
            let base = (idx as usize) * NNUE_PROBE_WIDTH;
            for i in 0..NNUE_PROBE_WIDTH {
                features[base + i] = 1.0;
            }
        }
        for v in features {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&0.5f32.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&2.0f32.to_le_bytes());
        }
        out.extend_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(out.len(), TOTAL_BYTES);
        out
    }

    #[test]
    fn loads_valid_artifact() {
        let model = NnueProbeModelV1::from_bytes(&synthetic_artifact_bytes(START_FEN)).unwrap();
        let pos = parse_fen(START_FEN).unwrap();
        // Each perspective lists ALL 30 non-king pieces (15 own + 15
        // opponent) with relative channels: acc = 0.5 + 30 * 1.0 = 30.5.
        let expected = 32.0 * 30.5 + 32.0 * 2.0 * 30.5 + 0.5;
        let scaled = model.evaluate_scaled(&pos);
        assert!(
            (scaled - expected).abs() < 1e-3,
            "scaled {scaled} != {expected}"
        );
        assert!((model.evaluate_cp(&pos) - expected * 1000.0).abs() < 1.0);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[0] = b'X';
        assert!(NnueProbeModelV1::from_bytes(&data).is_err());
    }

    #[test]
    fn rejects_bad_version_and_dimensions() {
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(NnueProbeModelV1::from_bytes(&data).is_err());
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[12..16].copy_from_slice(&123u32.to_le_bytes());
        assert!(NnueProbeModelV1::from_bytes(&data).is_err());
        let mut data = synthetic_artifact_bytes(START_FEN);
        data[16..20].copy_from_slice(&7u32.to_le_bytes());
        assert!(NnueProbeModelV1::from_bytes(&data).is_err());
    }

    #[test]
    fn rejects_truncated_and_trailing_bytes() {
        let full = synthetic_artifact_bytes(START_FEN);
        assert!(NnueProbeModelV1::from_bytes(&full[..full.len() - 1]).is_err());
        let mut trailing = full.clone();
        trailing.push(0);
        assert!(NnueProbeModelV1::from_bytes(&trailing).is_err());
    }

    #[test]
    fn rejects_nan_and_inf() {
        let mut data = synthetic_artifact_bytes(START_FEN);
        // Last payload byte belongs to head_bias; corrupt it to NaN.
        let n = data.len();
        data[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(NnueProbeModelV1::from_bytes(&data).is_err());
        let mut data = synthetic_artifact_bytes(START_FEN);
        // A feature-table float at the very start of the payload.
        let base = HEADER_BYTES;
        data[base..base + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(NnueProbeModelV1::from_bytes(&data).is_err());
    }

    #[test]
    fn accumulator_and_head_match_hand_computation() {
        let model = NnueProbeModelV1::from_bytes(&synthetic_artifact_bytes(START_FEN)).unwrap();
        let pos = parse_fen(START_FEN).unwrap();
        let white = active_features(&pos, NnuePerspective::White);
        let black = active_features(&pos, NnuePerspective::Black);
        assert_eq!(white.len(), 30);
        assert_eq!(black.len(), 30);
        let white_acc = model.accumulate(&white);
        let black_acc = model.accumulate(&black);
        for v in white_acc {
            assert!((v - 30.5).abs() < 1e-6, "white acc {v}");
        }
        for v in black_acc {
            assert!((v - 30.5).abs() < 1e-6, "black acc {v}");
        }
    }

    /// Artifact where `one_rows` feature rows are 1.0, `two_rows` rows are
    /// 2.0, acc_bias = 0, head own block = 1.0, opponent block = 10.0, bias 0.
    fn asymmetric_artifact_bytes(one_rows: &[u16], two_rows: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(TOTAL_BYTES);
        out.extend_from_slice(&NNUE_PROBE_MAGIC);
        out.extend_from_slice(&NNUE_PROBE_VERSION.to_le_bytes());
        out.extend_from_slice(&(NNUE_INPUTS as u32).to_le_bytes());
        out.extend_from_slice(&(NNUE_PROBE_WIDTH as u32).to_le_bytes());
        out.extend_from_slice(&NNUE_PROBE_TARGET_SCALE.to_le_bytes());
        out.extend_from_slice(&[0xcd; 32]);
        let mut features = vec![0.0f32; FEATURES_COUNT];
        for &idx in one_rows {
            let base = (idx as usize) * NNUE_PROBE_WIDTH;
            for i in 0..NNUE_PROBE_WIDTH {
                features[base + i] = 1.0;
            }
        }
        for &idx in two_rows {
            let base = (idx as usize) * NNUE_PROBE_WIDTH;
            for i in 0..NNUE_PROBE_WIDTH {
                features[base + i] = 2.0;
            }
        }
        for v in features {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&10.0f32.to_le_bytes());
        }
        out.extend_from_slice(&0.0f32.to_le_bytes());
        assert_eq!(out.len(), TOTAL_BYTES);
        out
    }

    #[test]
    fn stm_own_opponent_ordering() {
        // KQK: white list and black list are disjoint single indices, so the
        // two accumulators differ and the head's own/opponent blocks are
        // distinguishable.
        let kqk_fen = "7k/8/8/8/8/8/3QK3/8 w - - 0 1";
        let pos = parse_fen(kqk_fen).unwrap();
        let white = active_features(&pos, NnuePerspective::White);
        let black = active_features(&pos, NnuePerspective::Black);
        assert_eq!(white.len(), 1, "KQK white list: {white:?}");
        assert_eq!(black.len(), 1, "KQK black list: {black:?}");
        assert_ne!(white[0], black[0]);
        let model =
            NnueProbeModelV1::from_bytes(&asymmetric_artifact_bytes(&white, &black)).unwrap();
        // White to move: own=white acc [1.0;32] (weight 1.0), opp=black acc
        // [2.0;32] (weight 10.0) -> 32*1 + 32*10*2 = 672.
        let scaled_w = model.evaluate_scaled(&pos);
        assert!((scaled_w - 672.0).abs() < 1e-3, "w-stm scaled {scaled_w}");
        // Black to move (same board): own=black acc [2.0;32], opp=white acc
        // [1.0;32] -> 32*2 + 32*10*1 = 384.
        let black_fen = "7k/8/8/8/8/8/3QK3/8 b - - 0 1";
        let pos_b = parse_fen(black_fen).unwrap();
        assert_eq!(pos_b.side_to_move(), Color::Black);
        let scaled_b = model.evaluate_scaled(&pos_b);
        assert!((scaled_b - 384.0).abs() < 1e-3, "b-stm scaled {scaled_b}");
    }

    #[test]
    fn evaluate_does_not_mutate_position() {
        let model = NnueProbeModelV1::from_bytes(&synthetic_artifact_bytes(START_FEN)).unwrap();
        let pos = parse_fen(START_FEN).unwrap();
        let before = pos.zobrist_key();
        let _ = model.evaluate_scaled(&pos);
        let _ = model.evaluate_cp(&pos);
        assert_eq!(pos.zobrist_key(), before);
    }
}
