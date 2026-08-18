//! S6-N0 — NnueFeatureSetV1: king-conditioned sparse NNUE input encoding.
//!
//! FROZEN S6-N0 CONTRACT (no additional feature engineering):
//! - two perspectives: White / Black, each conditioned on its own king square;
//! - king buckets: 64 (one per oriented king square);
//! - piece channels: 10 = own P/N/B/R/Q (0..=4) + opponent P/N/B/R/Q (5..=9);
//! - kings are NOT ordinary piece features (each accumulator is conditioned on
//!   its own king, so both king locations enter the network via the buckets);
//! - feature index:
//!   `((oriented_king_square * 10 + relative_channel) * 64 + oriented_piece_square)`;
//! - `NNUE_INPUTS = 64 * 10 * 64 = 40_960`;
//! - side to move, tempo, castling, en passant, halfmove, phase, pawn
//!   structure, mobility, king safety, and the legacy 227-feature
//!   `FeatureSetV1` are deliberately NOT part of this first input schema.
//!
//! This module is NOT wired into `evaluate_profiled`, the search, or UCI. It is
//! a standalone input contract for the future NNUE runtime and trainer; S6-N0
//! only freezes the encoding and measures its cost.

use crate::chess::position::Position;
use crate::chess::types::{Color, Piece, PieceType, Square};

/// Total sparse input dimension: 64 king buckets * 10 piece channels * 64
/// piece squares.
pub const NNUE_INPUTS: usize = 64 * 10 * 64;

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

/// Map a piece to its relative channel for `perspective`. Kings are excluded
/// (they are represented by the king bucket, never as ordinary piece features).
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

/// Feature index for one active feature. All three arguments are already
/// oriented into the calling perspective's frame.
///
/// Range: `0 .. 40_960` (fits `u16`).
#[inline]
pub const fn feature_index(
    oriented_king_square: Square,
    channel: u8,
    oriented_piece_square: Square,
) -> usize {
    (oriented_king_square as usize * 10 + channel as usize) * 64 + oriented_piece_square as usize
}

/// Active feature indices for one perspective of `pos` (sparse, unsorted).
///
/// Count = number of non-king pieces on the board (30 for the start position).
/// Read-only: `pos` is never mutated.
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::chess::fen::parse_fen;
    use crate::chess::types::{E1, G1, START_FEN};

    use super::*;

    fn features(pos: &Position, perspective: NnuePerspective) -> BTreeSet<u16> {
        active_features(pos, perspective).into_iter().collect()
    }

    /// Vertical mirror (rank flip `sq ^ 56`) + color swap, expressed as a FEN.
    /// Side to move, castling, and en passant fields are carried over verbatim:
    /// the S6-N0 encoding ignores them, so the invariant only needs the board.
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

    #[test]
    fn startpos_has_30_active_features_per_perspective() {
        let pos = parse_fen(START_FEN).unwrap();
        for perspective in [NnuePerspective::White, NnuePerspective::Black] {
            let active = active_features(&pos, perspective);
            assert_eq!(active.len(), 30, "{perspective:?} active feature count");
            for index in active {
                assert!(
                    (index as usize) < NNUE_INPUTS,
                    "{perspective:?} index {index} out of range"
                );
            }
        }
    }

    #[test]
    fn all_indices_in_range_and_no_duplicates_across_fixtures() {
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
                let set = features(&pos, perspective);
                assert_eq!(set.len(), active_features(&pos, perspective).len());
                for index in &set {
                    assert!(
                        (*index as usize) < NNUE_INPUTS,
                        "{perspective:?} index {index} out of range for {fen}"
                    );
                }
            }
        }
    }

    #[test]
    fn startpos_indices_match_hand_computation() {
        let pos = parse_fen(START_FEN).unwrap();
        let white = features(&pos, NnuePerspective::White);
        let black = features(&pos, NnuePerspective::Black);

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
    fn vertical_mirror_color_swap_preserves_indices() {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1",
        ];
        for fen in fens {
            let original = parse_fen(fen).unwrap();
            let mirrored = parse_fen(&mirror_color_swap_fen(fen)).unwrap();
            assert_eq!(
                features(&original, NnuePerspective::White),
                features(&mirrored, NnuePerspective::Black),
                "original White == mirrored Black for {fen}"
            );
            assert_eq!(
                features(&original, NnuePerspective::Black),
                features(&mirrored, NnuePerspective::White),
                "original Black == mirrored White for {fen}"
            );
        }
    }

    #[test]
    fn encoding_does_not_mutate_position() {
        let pos = parse_fen(START_FEN).unwrap();
        let before = pos.zobrist_key();
        let _ = active_features(&pos, NnuePerspective::White);
        let _ = active_features(&pos, NnuePerspective::Black);
        assert_eq!(pos.zobrist_key(), before);
    }
}
