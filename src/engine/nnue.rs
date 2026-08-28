//! S6-N0 / S10-A — NNUE sparse input encodings:
//! - NnueFeatureSetV1 (S6 legacy control): 64 king buckets * 10 piece channels * 64 piece squares = 40,960.
//! - NnueFeatureSetV2 (S10 HalfKAv2_hm-inspired): 32 horizontal king buckets * 11 piece channels * 64 piece squares = 22,528.
//!
//! FROZEN CONTRACTS:
//! V1:
//! - two perspectives: White / Black, each conditioned on its own king square;
//! - king buckets: 64 (one per oriented king square);
//! - piece channels: 10 = own P/N/B/R/Q (0..=4) + opponent P/N/B/R/Q (5..=9);
//! - own and opp kings are NOT piece channels (each perspective conditions on its own king);
//! - feature index: `((oriented_king_square * 10 + relative_channel) * 64 + oriented_piece_square)`;
//! - `NNUE_INPUTS = 40_960`.
//!
//! V2 (S10-A):
//! - two perspectives: White / Black, each conditioned on its own king square with horizontal symmetry;
//! - if oriented king is on files a–d (`(sq & 7) < 4`), both the king and all piece squares are horizontally mirrored (`sq ^ 7`),
//!   mapping the king to files e–h (files 4..=7);
//! - king buckets: 32 = `(oriented_king_sq / 8) * 4 + ((oriented_king_sq % 8) - 4)`;
//! - piece channels: 11 = own P/N/B/R/Q (0..=4) + opponent P/N/B/R/Q (5..=9) + opponent King (10);
//! - own king remains the conditioning bucket only; opponent king is an active piece channel;
//! - feature index: `((king_bucket * 11 + channel) * 64 + oriented_piece_sq)`;
//! - `NNUE_INPUTS_V2 = 32 * 11 * 64 = 22_528`;
//! - startpos has exactly 31 active features per perspective.
//!
//! This module is NOT wired into `evaluate_profiled`, the search, or UCI.

use crate::chess::position::Position;
use crate::chess::types::{Color, Piece, PieceType, Square};

/// Total sparse input dimension for V1 (legacy default): 64 king buckets * 10 piece channels * 64 piece squares.
pub const NNUE_INPUTS: usize = 64 * 10 * 64;

/// Explicit V1 sparse input dimension alias: 40,960.
pub const NNUE_INPUTS_V1: usize = NNUE_INPUTS;

/// Sparse input dimension for V2: 32 king buckets * 11 piece channels * 64 piece squares = 22,528.
pub const NNUE_INPUTS_V2: usize = 32 * 11 * 64;

/// Supported NNUE feature set representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnueFeatureSet {
    V1,
    V2,
}

impl NnueFeatureSet {
    /// Sparse input dimension for this feature set.
    #[inline]
    pub const fn input_dim(self) -> usize {
        match self {
            NnueFeatureSet::V1 => NNUE_INPUTS_V1,
            NnueFeatureSet::V2 => NNUE_INPUTS_V2,
        }
    }

    /// Expected active features count for a standard start position.
    #[inline]
    pub const fn startpos_active_features(self) -> usize {
        match self {
            NnueFeatureSet::V1 => 30,
            NnueFeatureSet::V2 => 31,
        }
    }
}

/// One of the two king-conditioned accumulator perspectives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NnuePerspective {
    White,
    Black,
}

impl NnuePerspective {
    /// The board color this perspective accumulates for.
    #[inline]
    pub const fn color(self) -> Color {
        match self {
            NnuePerspective::White => Color::White,
            NnuePerspective::Black => Color::Black,
        }
    }

    /// Orient a square into this perspective's coordinate frame: White keeps
    /// the square, Black flips the rank (`sq ^ 56`).
    #[inline]
    pub const fn orient(self, sq: Square) -> Square {
        match self {
            NnuePerspective::White => sq,
            NnuePerspective::Black => sq ^ 56,
        }
    }
}

// ---------------------------------------------------------------------------
// V1 Feature Representation (Legacy Source-Compatible)
// ---------------------------------------------------------------------------

/// Map a piece to its relative channel for `perspective` under V1. Kings are excluded.
#[inline]
pub const fn relative_channel(perspective: NnuePerspective, piece: Piece) -> Option<u8> {
    let own = matches!(
        (perspective.color(), piece.color),
        (Color::White, Color::White) | (Color::Black, Color::Black)
    );
    let base = if own { 0 } else { 5 };
    let index = match piece.piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => return None,
    };
    Some(base + index)
}

/// Feature index for one active feature in V1.
/// Range: `0 .. 40_960` (fits `u16`).
#[inline]
pub const fn feature_index(
    oriented_king_square: Square,
    channel: u8,
    oriented_piece_square: Square,
) -> usize {
    (oriented_king_square as usize * 10 + channel as usize) * 64 + oriented_piece_square as usize
}

/// Active feature indices for one perspective of `pos` under V1 (sparse, unsorted).
/// Count = number of non-king pieces on the board (30 for the start position).
pub fn active_features(pos: &Position, perspective: NnuePerspective) -> Vec<u16> {
    let king_sq = perspective.orient(pos.king_square(perspective.color()));
    let mut out = Vec::with_capacity(30);
    for (sq, piece) in pos.board().iter().enumerate() {
        let Some(piece) = piece else { continue };
        let Some(channel) = relative_channel(perspective, *piece) else {
            continue;
        };
        let index = feature_index(king_sq, channel, perspective.orient(sq as Square));
        out.push(index as u16);
    }
    out
}

/// Explicit V1 alias for active_features.
#[inline]
pub fn active_features_v1(pos: &Position, perspective: NnuePerspective) -> Vec<u16> {
    active_features(pos, perspective)
}

// ---------------------------------------------------------------------------
// V2 Feature Representation (HalfKAv2_hm-inspired, 22,528 inputs)
// ---------------------------------------------------------------------------

/// V2 relative piece channels:
/// - 0..=4: Own P, N, B, R, Q
/// - 5..=9: Opp P, N, B, R, Q
/// - 10:    Opp King
///
/// (Own king returns None as it is the conditioning bucket).
#[inline]
pub const fn v2_relative_channel(perspective: NnuePerspective, piece: Piece) -> Option<u8> {
    let own = matches!(
        (perspective.color(), piece.color),
        (Color::White, Color::White) | (Color::Black, Color::Black)
    );
    if own {
        match piece.piece_type {
            PieceType::Pawn => Some(0),
            PieceType::Knight => Some(1),
            PieceType::Bishop => Some(2),
            PieceType::Rook => Some(3),
            PieceType::Queen => Some(4),
            PieceType::King => None,
        }
    } else {
        match piece.piece_type {
            PieceType::Pawn => Some(5),
            PieceType::Knight => Some(6),
            PieceType::Bishop => Some(7),
            PieceType::Rook => Some(8),
            PieceType::Queen => Some(9),
            PieceType::King => Some(10),
        }
    }
}

/// V2 king bucket index $\in [0, 31]$ for a king square that is already rank-oriented and mirrored to files e–h (4..=7).
#[inline]
pub const fn v2_king_bucket(mirrored_king_sq: Square) -> usize {
    let rank = (mirrored_king_sq / 8) as usize;
    let file = (mirrored_king_sq % 8) as usize;
    rank * 4 + (file - 4)
}

/// V2 feature index for one active feature.
/// Range: `0 .. 22_528` (fits `u16`).
#[inline]
pub const fn v2_feature_index(king_bucket: usize, channel: u8, mirrored_piece_sq: Square) -> usize {
    (king_bucket * 11 + channel as usize) * 64 + mirrored_piece_sq as usize
}

/// Shared V2 king context for one perspective of a position: oriented +
/// mirrored king square, mirror flag, and bucket. Used by both
/// `active_features_v2()` and `v2_feature_for_piece()` so the full-refresh
/// and incremental paths share ONE feature formula.
#[inline]
pub(crate) fn v2_king_context(
    pos: &Position,
    perspective: NnuePerspective,
) -> (usize, bool) {
    let raw_king_sq = perspective.orient(pos.king_square(perspective.color()));
    let mirror_file = (raw_king_sq & 7) < 4;
    let king_sq = if mirror_file {
        raw_king_sq ^ 7
    } else {
        raw_king_sq
    };
    (v2_king_bucket(king_sq), mirror_file)
}

/// V2 feature index for one specific piece on one specific square under
/// one perspective of `pos` (the single feature formula shared by
/// full-refresh extraction and incremental accumulator updates).
///
/// Returns None for the perspective's own king (conditioning bucket only).
#[inline]
pub(crate) fn v2_feature_for_piece(
    pos: &Position,
    perspective: NnuePerspective,
    square: Square,
    piece: Piece,
) -> Option<u16> {
    let (bucket, mirror_file) = v2_king_context(pos, perspective);
    let channel = v2_relative_channel(perspective, piece)?;
    let oriented_sq = perspective.orient(square);
    let final_sq = if mirror_file {
        oriented_sq ^ 7
    } else {
        oriented_sq
    };
    Some(v2_feature_index(bucket, channel, final_sq) as u16)
}

/// Active feature indices for one perspective of `pos` under V2 (sparse, unsorted).
/// Count = number of all pieces minus own king on the board (31 for start position).
pub fn active_features_v2(pos: &Position, perspective: NnuePerspective) -> Vec<u16> {
    let mut out = Vec::with_capacity(31);
    for (sq, piece) in pos.board().iter().enumerate() {
        let Some(piece) = piece else { continue };
        if let Some(index) =
            v2_feature_for_piece(pos, perspective, sq as Square, *piece)
        {
            out.push(index);
        }
    }
    out
}

/// Dispatcher helper to extract active features for any given feature set.
pub fn active_features_for(
    pos: &Position,
    perspective: NnuePerspective,
    feature_set: NnueFeatureSet,
) -> Vec<u16> {
    match feature_set {
        NnueFeatureSet::V1 => active_features_v1(pos, perspective),
        NnueFeatureSet::V2 => active_features_v2(pos, perspective),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::chess::fen::parse_fen;
    use crate::chess::types::{E1, G1, START_FEN};

    use super::*;

    fn features_v1(pos: &Position, perspective: NnuePerspective) -> BTreeSet<u16> {
        active_features_v1(pos, perspective).into_iter().collect()
    }

    fn features_v2(pos: &Position, perspective: NnuePerspective) -> BTreeSet<u16> {
        active_features_v2(pos, perspective).into_iter().collect()
    }

    /// S10-C1 no-drift: the shared per-piece formula must reproduce
    /// `active_features_v2` exactly (set equality, both perspectives)
    /// across representative positions incl. mirror regimes and
    /// asymmetric material.
    #[test]
    fn v2_feature_for_piece_matches_active_features_v2() {
        let fens = [
            START_FEN,
            "7k/8/8/8/8/8/3QK3/8 w - - 0 1",
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/8/8/8/8/3K4 w - - 0 1",             // kings on d/e files
            "3k4/8/8/8/8/8/8/4K3 w - - 0 1",             // mirror boundary
            "rnbqkbnr/pp1ppppp/8/8/8/8/PP1PPPPP/RNBQKBNR w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // ep-ish, pawns
            "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
            "r3k3/8/8/8/8/8/8/4K3 b q - 0 1",
        ];
        for fen in fens {
            let pos = parse_fen(fen).expect(fen);
            for perspective in [NnuePerspective::White, NnuePerspective::Black] {
                let reference = features_v2(&pos, perspective);
                let per_piece: BTreeSet<u16> = pos
                    .board()
                    .iter()
                    .enumerate()
                    .filter_map(|(sq, piece)| {
                        let piece = (*piece)?;
                        v2_feature_for_piece(
                            &pos, perspective, sq as Square, piece)
                    })
                    .collect();
                assert_eq!(
                    per_piece, reference,
                    "per-piece drift for {fen} / {perspective:?}"
                );
            }
        }
    }

    /// The per-piece formula must never emit the perspective's own king.
    #[test]
    fn v2_feature_for_piece_excludes_own_king() {
        let pos = parse_fen(START_FEN).unwrap();
        for perspective in [NnuePerspective::White, NnuePerspective::Black] {
            let king_sq = pos.king_square(perspective.color());
            let king = pos.board()[king_sq as usize].unwrap();
            assert!(
                v2_feature_for_piece(&pos, perspective, king_sq, king)
                    .is_none()
            );
            // Opponent king must be a channel-10 feature.
            let opp_sq = pos.king_square(perspective.color().opposite());
            let opp_king = pos.board()[opp_sq as usize].unwrap();
            assert!(
                v2_feature_for_piece(&pos, perspective, opp_sq, opp_king)
                    .is_some()
            );
        }
    }

    /// Vertical mirror (rank flip `sq ^ 56`) + color swap, expressed as a FEN.
    fn mirror_color_swap_fen(fen: &str) -> String {
        let mut parts = fen.split_whitespace();
        let board = parts.next().expect("fen board");
        let rest: Vec<&str> = parts.collect();
        let flipped: String = board
            .split('/')
            .rev()
            .map(|rank| {
                rank.chars()
                    .map(|c| match c {
                        'P' => 'p',
                        'N' => 'n',
                        'B' => 'b',
                        'R' => 'r',
                        'Q' => 'q',
                        'K' => 'k',
                        'p' => 'P',
                        'n' => 'N',
                        'b' => 'B',
                        'r' => 'R',
                        'q' => 'Q',
                        'k' => 'K',
                        digit => digit,
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("/");
        let mut out = flipped;
        for part in rest {
            out.push(' ');
            out.push_str(part);
        }
        out
    }

    /// Horizontal mirror (file flip `sq ^ 7`), expressed as a FEN.
    fn horizontal_mirror_fen(fen: &str) -> String {
        let mut parts = fen.split_whitespace();
        let board = parts.next().expect("fen board");
        let rest: Vec<&str> = parts.collect();
        let flipped: String = board
            .split('/')
            .map(|rank| {
                let mut expanded = Vec::new();
                for c in rank.chars() {
                    if let Some(d) = c.to_digit(10) {
                        expanded.extend(std::iter::repeat_n('1', d as usize));
                    } else {
                        expanded.push(c);
                    }
                }
                expanded.reverse();
                // re-compress empty squares
                let mut compressed = String::new();
                let mut empties = 0;
                for c in expanded {
                    if c == '1' {
                        empties += 1;
                    } else {
                        if empties > 0 {
                            compressed.push_str(&empties.to_string());
                            empties = 0;
                        }
                        compressed.push(c);
                    }
                }
                if empties > 0 {
                    compressed.push_str(&empties.to_string());
                }
                compressed
            })
            .collect::<Vec<_>>()
            .join("/");
        let mut out = flipped;
        for part in rest {
            out.push(' ');
            out.push_str(part);
        }
        out
    }

    #[test]
    fn startpos_has_30_active_features_v1_and_31_active_features_v2() {
        let pos = parse_fen(START_FEN).unwrap();
        for perspective in [NnuePerspective::White, NnuePerspective::Black] {
            let active1 = active_features_v1(&pos, perspective);
            assert_eq!(active1.len(), 30, "V1 {perspective:?} active count");
            for index in active1 {
                assert!(
                    (index as usize) < NNUE_INPUTS_V1,
                    "V1 {perspective:?} index {index} out of range"
                );
            }

            let active2 = active_features_v2(&pos, perspective);
            assert_eq!(active2.len(), 31, "V2 {perspective:?} active count");
            for index in active2 {
                assert!(
                    (index as usize) < NNUE_INPUTS_V2,
                    "V2 {perspective:?} index {index} out of range"
                );
            }
        }
    }

    #[test]
    fn all_v1_and_v2_indices_in_range_and_no_duplicates_across_fixtures() {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1",
            "4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1",
        ];
        for fen in fens {
            let pos = parse_fen(fen).unwrap();
            for perspective in [NnuePerspective::White, NnuePerspective::Black] {
                let set1 = features_v1(&pos, perspective);
                assert_eq!(set1.len(), active_features_v1(&pos, perspective).len());
                for index in &set1 {
                    assert!((*index as usize) < NNUE_INPUTS_V1);
                }

                let set2 = features_v2(&pos, perspective);
                assert_eq!(set2.len(), active_features_v2(&pos, perspective).len());
                for index in &set2 {
                    assert!((*index as usize) < NNUE_INPUTS_V2);
                }
            }
        }
    }

    #[test]
    fn startpos_v1_indices_match_hand_computation() {
        let pos = parse_fen(START_FEN).unwrap();
        let white = features_v1(&pos, NnuePerspective::White);
        let black = features_v1(&pos, NnuePerspective::Black);

        // White perspective: king e1 (bucket 4).
        //   a2 white pawn:  channel 0, square 8  -> (4*10+0)*64+8  = 2568
        //   g1 white knight: channel 1, square 6 -> (4*10+1)*64+6  = 2630
        assert!(white.contains(&2568), "white a2 pawn");
        assert!(white.contains(&2630), "white g1 knight");

        // Black perspective: king e8 (60) oriented -> bucket 4.
        //   d7 black pawn:  square 51 oriented -> 11 -> (4*10+0)*64+11 = 2571
        //   d1 white queen: square 3 oriented -> 59, channel 9 (opponent queen)
        //                   -> (4*10+9)*64+59  = 3195
        assert!(black.contains(&2571), "black d7 pawn");
        assert!(black.contains(&3195), "black perspective opponent queen");

        assert_eq!(E1, 4);
        assert_eq!(G1, 6);
    }

    #[test]
    fn startpos_v2_indices_match_hand_computation() {
        let pos = parse_fen(START_FEN).unwrap();
        let white = features_v2(&pos, NnuePerspective::White);
        let black = features_v2(&pos, NnuePerspective::Black);

        // White perspective: king e1 (sq 4: rank 0, file 4 >= 4, no mirror).
        //   king bucket: 0 * 4 + (4 - 4) = 0.
        //   a2 white pawn (channel 0, square 8):  (0 * 11 + 0) * 64 + 8 = 8.
        //   g1 white knight (channel 1, square 6): (0 * 11 + 1) * 64 + 6 = 70.
        //   e8 black king (channel 10, square 60): (0 * 11 + 10) * 64 + 60 = 700.
        assert!(white.contains(&8), "white a2 pawn");
        assert!(white.contains(&70), "white g1 knight");
        assert!(white.contains(&700), "white perspective opp king e8");

        // Black perspective: king e8 (sq 60) -> oriented sq = 60 ^ 56 = 4 (rank 0, file 4 >= 4, no mirror).
        //   king bucket: 0 * 4 + (4 - 4) = 0.
        //   d7 black pawn: sq 51 -> oriented sq = 51 ^ 56 = 11, channel 0: (0 * 11 + 0) * 64 + 11 = 11.
        //   d1 white queen: sq 3 -> oriented sq = 3 ^ 56 = 59, channel 9: (0 * 11 + 9) * 64 + 59 = 635.
        //   e1 white king: sq 4 -> oriented sq = 4 ^ 56 = 60, channel 10: (0 * 11 + 10) * 64 + 60 = 700.
        assert!(black.contains(&11), "black d7 pawn");
        assert!(black.contains(&635), "black perspective opp queen d1");
        assert!(black.contains(&700), "black perspective opp king e1");
    }

    #[test]
    fn v2_horizontal_mirror_preserves_features_identically() {
        let fens = [
            START_FEN,
            // king on d1 (file 3 < 4) vs king on e1 (file 4 >= 4)
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R2K3R w KQkq - 0 1",
            "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1",
        ];
        for fen in fens {
            let original = parse_fen(fen).unwrap();
            let mirrored = parse_fen(&horizontal_mirror_fen(fen)).unwrap();
            assert_eq!(
                features_v2(&original, NnuePerspective::White),
                features_v2(&mirrored, NnuePerspective::White),
                "original White == mirrored White V2 for {fen}"
            );
            assert_eq!(
                features_v2(&original, NnuePerspective::Black),
                features_v2(&mirrored, NnuePerspective::Black),
                "original Black == mirrored Black V2 for {fen}"
            );
        }
    }

    #[test]
    fn v1_and_v2_vertical_mirror_color_swap_preserves_indices() {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1",
        ];
        for fen in fens {
            let original = parse_fen(fen).unwrap();
            let mirrored = parse_fen(&mirror_color_swap_fen(fen)).unwrap();
            assert_eq!(
                features_v1(&original, NnuePerspective::White),
                features_v1(&mirrored, NnuePerspective::Black),
                "V1 original White == mirrored Black for {fen}"
            );
            assert_eq!(
                features_v1(&original, NnuePerspective::Black),
                features_v1(&mirrored, NnuePerspective::White),
                "V1 original Black == mirrored White for {fen}"
            );
            assert_eq!(
                features_v2(&original, NnuePerspective::White),
                features_v2(&mirrored, NnuePerspective::Black),
                "V2 original White == mirrored Black for {fen}"
            );
            assert_eq!(
                features_v2(&original, NnuePerspective::Black),
                features_v2(&mirrored, NnuePerspective::White),
                "V2 original Black == mirrored White for {fen}"
            );
        }
    }

    #[test]
    fn v1_and_v2_encoding_does_not_mutate_position() {
        let pos = parse_fen(START_FEN).unwrap();
        let before = pos.zobrist_key();
        let _ = active_features_v1(&pos, NnuePerspective::White);
        let _ = active_features_v1(&pos, NnuePerspective::Black);
        let _ = active_features_v2(&pos, NnuePerspective::White);
        let _ = active_features_v2(&pos, NnuePerspective::Black);
        assert_eq!(pos.zobrist_key(), before);
    }
}
