//! Deterministic, compile-time-derived Zobrist hashing for M3.0.
//!
//! Every key is derived from a single fixed seed via SplitMix64, so that
//! every machine, build (debug/release), and process run produces bit-identical
//! hashes. There is no runtime RNG and no `rand` crate — the table is
//! fully determined by the source below. See
//! `docs/specs/m3.0-game-state-zobrist-foundation.md` for the locked
//! constants and the startpos reference value `0x61A2_181F_8F2F_9B9C`.
//!
//! The Zobrist key identifies a *board position* for repetition purposes.
//! It therefore does NOT include the halfmove clock, the fullmove number,
//! any UCI history length, the search depth, the evaluation, or a cached
//! check flag. Two FENs that differ only in halfmove/fullmove must hash
//! identically (M3.2 must solve the rule-50 context on its own).

use crate::chess::position::Position;
use crate::chess::types::*;

/// A 64-bit Zobrist key.
pub type ZobristKey = u64;

// Locked constants — DO NOT change these to make an implementation "match".
// If `recompute_zobrist` ever disagrees with `0x61A2_181F_8F2F_9B9C`,
// fix the mapping/formula below, never the seed or these values.
const ZOBRIST_SEED: u64 = 0xD1B5_4A32_D192_ED03;
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 one-shot mix. `const fn` so the keys can be baked in.
const fn zobrist_value(slot: u64) -> u64 {
    let mut z = ZOBRIST_SEED.wrapping_add((slot + 1).wrapping_mul(SPLITMIX_GAMMA));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Slot assignment is locked (spec 4.2):
///   0..=767   piece-square keys (color * 6 + piece) * 64 + square
///   768       black-to-move key
///   769..=784 castling mask 0..=15
///   785..=792 en-passant file a..h
fn color_index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn piece_index(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
    }
}

/// Key for `side == Black` to move. White to move contributes nothing.
pub const SIDE_KEY: u64 = zobrist_value(768);

/// Key for a given castling-rights mask (0..=15).
#[inline]
pub fn castling_key(mask: usize) -> u64 {
    zobrist_value(769 + mask as u64)
}

/// Key for a given en-passant file (0..=7, where 0 = file a).
#[inline]
pub fn ep_file_key(file: u8) -> u64 {
    zobrist_value(785 + file as u64)
}

/// Key for `piece` sitting on `square`.
#[inline]
pub fn piece_square_key(piece: Piece, square: Square) -> u64 {
    let slot = ((color_index(piece.color) * 6 + piece_index(piece.piece_type)) * 64
        + (square as usize)) as u64;
    zobrist_value(slot)
}

/// Map `CastlingRights` to the 0..=15 mask used by `castling_key`.
pub fn castling_mask(c: &CastlingRights) -> usize {
    let mut m = 0usize;
    if c.white_king {
        m |= 1;
    }
    if c.white_queen {
        m |= 2;
    }
    if c.black_king {
        m |= 4;
    }
    if c.black_queen {
        m |= 8;
    }
    m
}

/// Reference recomputation: full 64-square scan. Used by FEN/startpos
/// initialization, tests, and `debug_assert` invariants.
///
/// MUST NOT be called on the hot path after every `make_move` — the search
/// updates the key incrementally instead.
pub fn recompute_zobrist(pos: &Position) -> ZobristKey {
    let mut key = 0u64;
    for sq in 0..64 {
        if let Some(piece) = pos.board()[sq] {
            key ^= piece_square_key(piece, sq as Square);
        }
    }
    if pos.side_to_move() == Color::Black {
        key ^= SIDE_KEY;
    }
    key ^= castling_key(castling_mask(&pos.castling_rights()));
    if let Some(file) = effective_ep_file(pos) {
        key ^= ep_file_key(file);
    }
    key
}

/// En-passant *file* (0..=7) only when the side to move has at least one
/// **legal** en-passant capture available; `None` otherwise.
///
/// The three "no EP" situations handled here:
///   1. `ep_target` is `None`;
///   2. `ep_target` square is already occupied (illegal FEN — also rejected
///      by `parse_fen`);
///   3. `ep_target` is on a rank a pawn of the side to move could not have
///      advanced to (so there is no adjacent own pawn);
///   4. no enemy pawn sits on the captured square (the square directly behind
///      the target on its file);
///   5. the only candidate capture would leave the mover's king in check
///      (horizontal pin / self-check) — detected by the throwaway clone.
pub fn effective_ep_file(pos: &Position) -> Option<u8> {
    let target = pos.ep_target()?;

    // Precondition 2: the en-passant target must be empty.
    if pos.board()[target as usize].is_some() {
        return None;
    }

    // Precondition 3: the target must lie on the rank a pawn of the side
    // to move would have just crossed (White -> rank 6 / index 5,
    // Black -> rank 3 / index 2).
    let expected_rank: u8 = match pos.side_to_move() {
        Color::White => 5,
        Color::Black => 2,
    };
    if rank_of(target) != expected_rank {
        return None;
    }

    let us = pos.side_to_move();
    let them = us.opposite();

    // The capturing pawn sits on the rank just behind `target` (from the
    // mover's perspective) and on a file adjacent to `target`'s file.
    // The enemy pawn to be removed sits on the same file as `target`,
    // one rank behind it.
    let pawn_rank: u8 = if us == Color::White {
        rank_of(target) - 1
    } else {
        rank_of(target) + 1
    };
    let cap_sq = make_square(file_of(target), pawn_rank);

    for df in [-1i32, 1i32] {
        let pf = file_of(target) as i32 + df;
        if !(0..=7).contains(&pf) {
            continue;
        }
        let from = make_square(pf as u8, pawn_rank);
        match pos.board()[from as usize] {
            Some(p) if p.color == us && p.piece_type == PieceType::Pawn => {}
            _ => continue,
        }
        // Precondition 4: an enemy pawn must actually be capturable.
        match pos.board()[cap_sq as usize] {
            Some(p) if p.color == them && p.piece_type == PieceType::Pawn => {}
            _ => continue,
        }

        // Precondition 5 (pin / self-check): simulate the capture on a
        // throwaway COPY. Its `zobrist_key` stays intentionally stale and is
        // never read; we only mutate its board and ask whether the mover's
        // king is attacked. We never call `make_move`,
        // `generate_legal_moves`, or `recompute_zobrist` on it.
        let mut probe = *pos;
        // Direct `pub(crate)` field mutation on a throwaway copy — no
        // mutable getter exists by design (spec 2.1).
        probe.board[from as usize] = None;
        probe.board[cap_sq as usize] = None;
        probe.board[target as usize] = Some(Piece::new(us, PieceType::Pawn));
        if !probe.is_square_attacked(probe.king_square(us), them) {
            return Some(file_of(target));
        }
    }

    None
}
