//! Evaluation.
//!
//! Returns the position's value from the side-to-move's perspective
//! (positive = good for the side to move). EVAL 1A keeps the existing
//! material and non-king PST values, but evaluates them through separate
//! middlegame/endgame score lanes and interpolates them by non-pawn phase.
//! The King now has distinct middlegame safety and endgame-centralisation
//! PSTs. This module deliberately does not add mobility, pawn structure,
//! king-safety features, or search pruning.
//!
use crate::chess::position::Position;
use crate::chess::types::{
    file_of, make_square, on_board, rank_of, Color, Piece, PieceType, BISHOP_DIRS, KING_OFFSETS,
    KNIGHT_OFFSETS, QUEEN_DIRS, ROOK_DIRS,
};

/// Piece-square tables, a1-first (index 0 = a1 ... 63 = h8), one row per
/// rank (rank 1 first). Values are from Tomasz Michniewski's "Simplified
/// Evaluation Function"; the public tables start at a8 (rank 8 first), so
/// they were reversed rank-wise to match this engine's `a1 = 0`. Black
/// pieces reuse the same tables via a vertical mirror (`sq ^ 56`, see
/// `pst_idx`).
///
/// These are locked by `docs/specs/m2.4-piece-square-evaluation.md` —
/// do NOT re-tune them here.
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
    // a1 .. h1  (rank 1)
         0,   0,   0,   0,   0,   0,   0,   0,
    // a2 .. h2  (rank 2)
         5,  10,  10, -20, -20,  10,  10,   5,
    // a3 .. h3  (rank 3)
         5,  -5, -10,   0,   0, -10,  -5,   5,
    // a4 .. h4  (rank 4)
         0,   0,   0,  20,  20,   0,   0,   0,
    // a5 .. h5  (rank 5)
         5,   5,  10,  25,  25,  10,   5,   5,
    // a6 .. h6  (rank 6)
        10,  10,  20,  30,  30,  20,  10,  10,
    // a7 .. h7  (rank 7)
        50,  50,  50,  50,  50,  50,  50,  50,
    // a8 .. h8  (rank 8)
         0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
    // a1 .. h1
        -50, -40, -30, -30, -30, -30, -40, -50,
    // a2 .. h2
        -40, -20,   0,   5,   5,   0, -20, -40,
    // a3 .. h3
        -30,   5,  10,  15,  15,  10,   5, -30,
    // a4 .. h4
        -30,   0,  15,  20,  20,  15,   0, -30,
    // a5 .. h5
        -30,   5,  15,  20,  20,  15,   5, -30,
    // a6 .. h6
        -30,   0,  10,  15,  15,  10,   0, -30,
    // a7 .. h7
        -40, -20,   0,   0,   0,   0, -20, -40,
    // a8 .. h8
        -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
    // a1 .. h1
        -20, -10, -10, -10, -10, -10, -10, -20,
    // a2 .. h2
        -10,   5,   0,   0,   0,   0,   5, -10,
    // a3 .. h3
        -10,  10,  10,  10,  10,  10,  10, -10,
    // a4 .. h4
        -10,   0,  10,  10,  10,  10,   0, -10,
    // a5 .. h5
        -10,   5,   5,  10,  10,   5,   5, -10,
    // a6 .. h6
        -10,   0,   5,  10,  10,   5,   0, -10,
    // a7 .. h7
        -10,   0,   0,   0,   0,   0,   0, -10,
    // a8 .. h8
        -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const ROOK_PST: [i32; 64] = [
    // a1 .. h1
          0,   0,   0,   5,   5,   0,   0,   0,
    // a2 .. h2
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a3 .. h3
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a4 .. h4
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a5 .. h5
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a6 .. h6
         -5,   0,   0,   0,   0,   0,   0,  -5,
    // a7 .. h7
          5,  10,  10,  10,  10,  10,  10,   5,
    // a8 .. h8
          0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const QUEEN_PST: [i32; 64] = [
    // a1 .. h1
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    // a2 .. h2
        -10,   0,   5,   0,   0,   0,   0, -10,
    // a3 .. h3
        -10,   5,   5,   5,   5,   5,   0, -10,
    // a4 .. h4
          0,   0,   5,   5,   5,   5,   0,  -5,
    // a5 .. h5
         -5,   0,   5,   5,   5,   5,   0,  -5,
    // a6 .. h6
        -10,   0,   5,   5,   5,   5,   0, -10,
    // a7 .. h7
        -10,   0,   0,   0,   0,   0,   0, -10,
    // a8 .. h8
        -20, -10, -10,  -5,  -5, -10, -10, -20,
];

/// Middlegame King PST, a1-first. Safe shelter squares near the home rank
/// are preferred while central exposure is penalised.
#[rustfmt::skip]
const KING_MG_PST: [i32; 64] = [
    // a1 .. h1
        20,  30,  10,   0,   0,  10,  30,  20,
    // a2 .. h2
        20,  20,   0,   0,   0,   0,  20,  20,
    // a3 .. h3
       -10, -20, -20, -20, -20, -20, -20, -10,
    // a4 .. h4
       -20, -30, -30, -40, -40, -30, -30, -20,
    // a5 .. h5
       -30, -40, -40, -50, -50, -40, -40, -30,
    // a6 .. h6
       -30, -40, -40, -50, -50, -40, -40, -30,
    // a7 .. h7
       -30, -40, -40, -50, -50, -40, -40, -30,
    // a8 .. h8
       -30, -40, -40, -50, -50, -40, -40, -30,
];

/// Endgame King PST, a1-first. Centralisation is rewarded symmetrically;
/// the table is intentionally independent from the middlegame shelter table.
#[rustfmt::skip]
const KING_EG_PST: [i32; 64] = [
    // a1 .. h1
       -50, -40, -30, -20, -20, -30, -40, -50,
    // a2 .. h2
       -30, -20, -10,   0,   0, -10, -20, -30,
    // a3 .. h3
       -30, -10,  20,  30,  30,  20, -10, -30,
    // a4 .. h4
       -30, -10,  30,  40,  40,  30, -10, -30,
    // a5 .. h5
       -30, -10,  30,  40,  40,  30, -10, -30,
    // a6 .. h6
       -30, -10,  20,  30,  30,  20, -10, -30,
    // a7 .. h7
       -30, -20, -10,   0,   0, -10, -20, -30,
    // a8 .. h8
       -50, -40, -30, -20, -20, -30, -40, -50,
];

const MAX_PHASE: i32 = 24;
const STALEMATE_BONUS: i32 = -900;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Score {
    mg: i32,
    eg: i32,
}

impl Score {
    const fn same(value: i32) -> Score {
        Score {
            mg: value,
            eg: value,
        }
    }
}

/// Map a piece type to its piece-square table. An exhaustive `match` (not
/// `table[piece_type as usize]`) so the compiler forces every new piece
/// kind to be handled explicitly instead of silently indexing the wrong row
/// if the `PieceType` enum order ever changes.
fn pst_score(pt: PieceType, idx: usize) -> Score {
    match pt {
        PieceType::Pawn => Score::same(PAWN_PST[idx]),
        PieceType::Knight => Score::same(KNIGHT_PST[idx]),
        PieceType::Bishop => Score::same(BISHOP_PST[idx]),
        PieceType::Rook => Score::same(ROOK_PST[idx]),
        PieceType::Queen => Score::same(QUEEN_PST[idx]),
        PieceType::King => Score {
            mg: KING_MG_PST[idx],
            eg: KING_EG_PST[idx],
        },
    }
}

fn material_score(pt: PieceType) -> Score {
    Score::same(pt.value())
}

fn phase_weight(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn | PieceType::King => 0,
        PieceType::Knight | PieceType::Bishop => 1,
        PieceType::Rook => 2,
        PieceType::Queen => 4,
    }
}

/// Non-pawn material phase: 24 is the normal starting position and 0 is a
/// pure king/pawn ending. Promotions are clamped so the interpolation range
/// remains stable even when a position contains extra heavy pieces.
pub(crate) fn game_phase(pos: &Position) -> i32 {
    let phase: i32 = pos
        .board
        .iter()
        .filter_map(|piece| *piece)
        .map(|piece| phase_weight(piece.piece_type))
        .sum();
    phase.min(MAX_PHASE)
}

#[inline]
fn interpolate(mg: i32, eg: i32, phase: i32) -> i32 {
    ((mg as i64 * phase as i64 + eg as i64 * (MAX_PHASE - phase) as i64) / MAX_PHASE as i64) as i32
}

fn finish_evaluation(pos: &Position, mg: i32, eg: i32, phase: i32) -> i32 {
    let base = interpolate(mg, eg, phase);
    let Some(mop_up) = exact_mop_up(pos) else {
        return base;
    };
    let bonus = exact_mop_up_bonus(pos, mop_up);

    if pos.side == mop_up.strong {
        base + bonus
    } else if bonus == STALEMATE_BONUS {
        0
    } else {
        base - bonus
    }
}

#[derive(Clone, Copy, Debug)]
struct ExactMopUp {
    strong: Color,
    piece: PieceType,
    strong_king: u8,
    weak_king: u8,
    strong_piece: u8,
}

/// Recognise only the exact KQK and KRK material configurations. The exact
/// match is intentional: a mop-up term must never replace the normal
/// evaluation in a position with pawns or another attacker still on the
/// board.
fn exact_mop_up(pos: &Position) -> Option<ExactMopUp> {
    let mut kings = [None; 2];
    let mut extra = None;

    for (sq, piece) in pos.board.iter().enumerate() {
        let Some(piece) = piece else { continue };
        match piece.piece_type {
            PieceType::King if kings[piece.color as usize].is_none() => {
                kings[piece.color as usize] = Some(sq as u8);
            }
            PieceType::Queen | PieceType::Rook => {
                if extra.is_some() {
                    // The dedicated term is only for the exact KQK/KRK
                    // signature. A second heavy piece must fall back to the
                    // ordinary tapered evaluation rather than being ignored.
                    return None;
                }
                extra = Some((piece.color, piece.piece_type, sq as u8));
            }
            _ => return None,
        }
    }

    let (strong, piece, strong_piece) = extra?;
    let strong_king = kings[strong as usize]?;
    let weak = strong.opposite();
    let weak_king = kings[weak as usize]?;

    Some(ExactMopUp {
        strong,
        piece,
        strong_king,
        weak_king,
        strong_piece,
    })
}

#[inline]
fn chebyshev_distance(a: u8, b: u8) -> i32 {
    let file_delta = (file_of(a) as i32 - file_of(b) as i32).abs();
    let rank_delta = (rank_of(a) as i32 - rank_of(b) as i32).abs();
    file_delta.max(rank_delta)
}

#[inline]
fn distance_to_edge(sq: u8) -> i32 {
    let file = file_of(sq) as i32;
    let rank = rank_of(sq) as i32;
    file.min(7 - file).min(rank).min(7 - rank)
}

/// Count legal king escapes for the weak side without changing the caller's
/// position. The temporary board is necessary when the king moves off a
/// rook/queen ray or captures the sole strong piece: attack status must be
/// tested in the resulting board, not the current one.
fn weak_king_mobility(pos: &Position, weak: Color, weak_king: u8) -> i32 {
    let mut mobility = 0;
    let from_file = file_of(weak_king) as i32;
    let from_rank = rank_of(weak_king) as i32;

    for (df, dr) in KING_OFFSETS {
        let file = from_file + df;
        let rank = from_rank + dr;
        if !(0..8).contains(&file) || !(0..8).contains(&rank) {
            continue;
        }
        let to = make_square(file as u8, rank as u8);
        if to == pos.king_sq[weak.opposite() as usize] {
            // A king may not capture the opposing king. In the temporary
            // board below the destination would otherwise overwrite the
            // strong king and make the attack test lose that constraint.
            continue;
        }
        if pos.board[to as usize].is_some_and(|piece| piece.color == weak) {
            continue;
        }

        let mut next = *pos;
        next.board[weak_king as usize] = None;
        next.board[to as usize] = Some(Piece::new(weak, PieceType::King));
        next.king_sq[weak as usize] = to;
        if !next.is_square_attacked(to, weak.opposite()) {
            mobility += 1;
        }
    }

    mobility
}

/// Return a bounded, side-independent mop-up bonus for exact KQK/KRK. The
/// material score remains the primary signal. The bonus rewards the standard
/// winning plan in order: bring the attacking king closer, drive the enemy
/// king toward the edge, and reduce its legal activity. An undefended rook or
/// queen next to the enemy king is discounted so the evaluator does not
/// encourage sacrificing the only winning material.
fn exact_mop_up_bonus(pos: &Position, mop_up: ExactMopUp) -> i32 {
    let weak = mop_up.strong.opposite();
    let mobility = weak_king_mobility(pos, weak, mop_up.weak_king);

    // A side with no legal move and no check is already stalemated. Static
    // evaluation must not turn that terminal draw into a large winning score.
    if mobility == 0 && !pos.is_in_check(weak) && pos.side == weak {
        return STALEMATE_BONUS;
    }

    let proximity = (7 - chebyshev_distance(mop_up.strong_king, mop_up.weak_king)).max(0);
    let edge = (3 - distance_to_edge(mop_up.weak_king)).max(0);
    let confinement = (8 - mobility).max(0);
    let mut bonus = proximity * 20 + edge * 18 + confinement * 16;

    // The only possible attacker of the strong piece in these exact material
    // positions is the weak king. Preserve the winning material if that king
    // can capture it immediately and the strong king does not defend it.
    let piece_attacked = chebyshev_distance(mop_up.weak_king, mop_up.strong_piece) <= 1;
    let piece_defended = chebyshev_distance(mop_up.strong_king, mop_up.strong_piece) <= 1;
    if piece_attacked && !piece_defended {
        bonus -= match mop_up.piece {
            PieceType::Queen => 300,
            PieceType::Rook => 260,
            _ => unreachable!("exact mop-up only accepts queen or rook"),
        };
    }

    bonus.clamp(-300, 400)
}

/// Square index into a piece-square table. White uses the square directly.
/// Black mirrors it vertically: `sq ^ 56` flips the rank bits (0b111000)
/// while leaving the file unchanged, so the same table serves both colors.
#[inline]
fn pst_idx(sq: usize, color: Color) -> usize {
    if color == Color::Black {
        sq ^ 56
    } else {
        sq
    }
}

/// Static evaluation from the side-to-move's perspective: material plus
/// piece-square-table bonuses. Read-only — it never mutates `pos`.
pub fn evaluate(pos: &Position) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    for sq in 0..64usize {
        if let Some(p) = pos.board[sq] {
            let sign = if p.color == pos.side { 1 } else { -1 };
            let material = material_score(p.piece_type);
            let positional = pst_score(p.piece_type, pst_idx(sq, p.color));
            mg += sign * (material.mg + positional.mg);
            eg += sign * (material.eg + positional.eg);
        }
    }
    finish_evaluation(pos, mg, eg, game_phase(pos))
}

/// Return whether a concrete piece attacks a target square in the current
/// position. This is intentionally local to the threat-aware candidate: the
/// rules module remains the single source of truth for legal move generation,
/// while evaluation only needs a read-only attack map.
fn piece_attacks_square(pos: &Position, from: u8, piece: Piece, target: u8) -> bool {
    let from_file = file_of(from) as i32;
    let from_rank = rank_of(from) as i32;
    let target_file = file_of(target) as i32;
    let target_rank = rank_of(target) as i32;
    let df = target_file - from_file;
    let dr = target_rank - from_rank;

    match piece.piece_type {
        PieceType::Pawn => {
            let direction = if piece.color == Color::White { 1 } else { -1 };
            dr == direction && df.abs() == 1
        }
        PieceType::Knight => KNIGHT_OFFSETS
            .iter()
            .any(|&(offset_file, offset_rank)| df == offset_file && dr == offset_rank),
        PieceType::King => df.abs() <= 1 && dr.abs() <= 1 && (df != 0 || dr != 0),
        PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
            let directions: &[(i32, i32)] = match piece.piece_type {
                PieceType::Bishop => &BISHOP_DIRS,
                PieceType::Rook => &ROOK_DIRS,
                PieceType::Queen => &QUEEN_DIRS,
                _ => unreachable!(),
            };
            let Some(&(step_file, step_rank)) = directions
                .iter()
                .find(|&&(sf, sr)| (df != 0 || dr != 0) && df.signum() == sf && dr.signum() == sr)
            else {
                return false;
            };
            let mut file = from_file + step_file;
            let mut rank = from_rank + step_rank;
            while on_board(file, rank) && (file != target_file || rank != target_rank) {
                if pos.board[make_square(file as u8, rank as u8) as usize].is_some() {
                    return false;
                }
                file += step_file;
                rank += step_rank;
            }
            file == target_file && rank == target_rank
        }
    }
}

fn king_zone(king: u8) -> Vec<u8> {
    let file = file_of(king) as i32;
    let rank = rank_of(king) as i32;
    let mut zone = Vec::with_capacity(9);
    for df in -1..=1 {
        for dr in -1..=1 {
            if on_board(file + df, rank + dr) {
                zone.push(make_square((file + df) as u8, (rank + dr) as u8));
            }
        }
    }
    zone
}

fn attack_unit_value(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::Pawn => 1,
        PieceType::Knight | PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 1,
    }
}

/// Estimate the danger around one king. The score is deliberately bounded and
/// asymmetric only through the board position: callers subtract the two
/// sides' values. It is a middlegame threat signal, not a replacement for
/// legal checkmate detection.
fn king_danger(pos: &Position, victim: Color) -> i32 {
    let king = pos.king_sq[victim as usize];
    let enemy = victim.opposite();
    let zone = king_zone(king);

    let mut attack_units = 0;
    for (sq, piece) in pos.board.iter().enumerate().filter_map(|(sq, piece)| {
        piece
            .filter(|piece| piece.color == enemy)
            .map(|piece| (sq, piece))
    }) {
        if zone
            .iter()
            .any(|&target| piece_attacks_square(pos, sq as u8, piece, target))
        {
            attack_units += attack_unit_value(piece.piece_type);
        }
    }

    let attacked_zone = zone
        .iter()
        .filter(|&&target| pos.is_square_attacked(target, enemy))
        .count() as i32;
    let defended_zone = zone
        .iter()
        .filter(|&&target| pos.is_square_attacked(target, victim))
        .count() as i32;

    let king_file = file_of(king) as i32;
    let king_rank = rank_of(king) as i32;
    let forward = if victim == Color::White { 1 } else { -1 };
    let shield_rank = king_rank + forward;
    let mut missing_shield = 0;
    let mut storm_pawns = 0;
    if (0..8).contains(&shield_rank) {
        for file in (king_file - 1)..=(king_file + 1) {
            if !(0..8).contains(&file) {
                continue;
            }
            let square = make_square(file as u8, shield_rank as u8);
            match pos.board[square as usize] {
                Some(piece) if piece.color == victim && piece.piece_type == PieceType::Pawn => {}
                Some(piece) if piece.color == enemy && piece.piece_type == PieceType::Pawn => {
                    missing_shield += 1;
                    storm_pawns += 1;
                }
                _ => missing_shield += 1,
            }
        }
    }

    let mut open_nearby_files = 0;
    for file in (king_file - 1)..=(king_file + 1) {
        if !(0..8).contains(&file) {
            continue;
        }
        let own_pawn = (0..8).any(|rank| {
            pos.board[make_square(file as u8, rank) as usize]
                .is_some_and(|piece| piece.color == victim && piece.piece_type == PieceType::Pawn)
        });
        let any_pawn = (0..8).any(|rank| {
            pos.board[make_square(file as u8, rank) as usize]
                .is_some_and(|piece| piece.piece_type == PieceType::Pawn)
        });
        if !own_pawn {
            open_nearby_files += 1;
            if !any_pawn {
                open_nearby_files += 1;
            }
        }
    }

    let escape_squares = zone
        .iter()
        .filter(|&&target| target != king)
        .filter(|&&target| {
            !pos.board[target as usize].is_some_and(|piece| piece.color == victim)
                && target != pos.king_sq[enemy as usize]
                && !pos.is_square_attacked(target, enemy)
        })
        .count() as i32;
    let cramped_escapes = (3 - escape_squares).max(0);
    let check_bonus = if pos.is_in_check(victim) { 120 } else { 0 };

    (attack_units * attack_units * 10
        + attacked_zone * 8
        + missing_shield * 22
        + storm_pawns * 12
        + open_nearby_files * 14
        + cramped_escapes * 18
        + check_bonus
        - defended_zone * 3)
        .clamp(0, 700)
}

/// Candidate-only evaluation that adds a bounded king-danger signal to the
/// approved tapered material/PST evaluator. The result remains from the
/// side-to-move perspective and never mutates `Position`.
pub fn evaluate_threat_aware(pos: &Position) -> i32 {
    let base = evaluate(pos);
    let own_danger = king_danger(pos, pos.side);
    let enemy_danger = king_danger(pos, pos.side.opposite());
    let danger_delta = enemy_danger - own_danger;
    let phase = game_phase(pos);
    let tapered_threat = interpolate(danger_delta, danger_delta / 4, phase);
    base.saturating_add(tapered_threat)
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate, evaluate_threat_aware, exact_mop_up, game_phase, interpolate, KING_EG_PST,
        KING_MG_PST, MAX_PHASE,
    };
    use crate::chess::fen::parse_fen;
    use crate::chess::fen::to_fen;

    #[test]
    fn phase_counts_non_pawn_material_only() {
        let start = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let king_pawn = parse_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").unwrap();
        let queen_ending = parse_fen("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1").unwrap();

        assert_eq!(game_phase(&start), MAX_PHASE);
        assert_eq!(game_phase(&king_pawn), 0);
        assert_eq!(game_phase(&queen_ending), 4);
    }

    #[test]
    fn interpolation_has_locked_phase_extremes() {
        assert_eq!(interpolate(100, 200, MAX_PHASE), 100);
        assert_eq!(interpolate(100, 200, 0), 200);
        assert_eq!(interpolate(100, 200, MAX_PHASE / 2), 150);
    }

    #[test]
    fn king_tables_are_distinct() {
        assert_ne!(KING_MG_PST[4], KING_EG_PST[4]);
        assert_ne!(KING_MG_PST[28], KING_EG_PST[28]);
    }

    #[test]
    fn exact_mopup_requires_one_and_only_one_heavy_piece() {
        let exact = parse_fen("7k/8/5K2/8/3Q4/8/8/8 w - - 0 1").unwrap();
        let two_queens = parse_fen("7k/8/5K2/8/3Q4/8/1Q6/8 w - - 0 1").unwrap();

        assert!(exact_mop_up(&exact).is_some());
        assert!(exact_mop_up(&two_queens).is_none());
    }

    #[test]
    fn threat_aware_eval_is_side_to_move_symmetric() {
        let white = parse_fen("r4rk1/ppp2p2/8/8/8/6q1/PPPP1P2/R3R1K1 w - - 0 1").unwrap();
        let black = parse_fen("r4rk1/ppp2p2/8/8/8/6q1/PPPP1P2/R3R1K1 b - - 0 1").unwrap();
        assert_eq!(
            evaluate_threat_aware(&white),
            -evaluate_threat_aware(&black)
        );
    }

    #[test]
    fn threat_aware_eval_penalises_exposed_king_without_mutation() {
        let safe = parse_fen("r4rk1/ppp2ppp/8/8/8/8/PPPP1PPP/R3R1K1 w - - 0 1").unwrap();
        let exposed = parse_fen("r4rk1/ppp2p2/8/8/8/6q1/PPPP1P2/R3R1K1 w - - 0 1").unwrap();
        let before = to_fen(&exposed);
        let base = evaluate(&exposed);
        let threat = evaluate_threat_aware(&exposed);
        assert_eq!(to_fen(&exposed), before);
        assert!(
            threat < base,
            "exposed king should be penalised: base={base}, threat={threat}"
        );
        assert!(
            evaluate_threat_aware(&safe) > threat,
            "safe king should score above the exposed equivalent"
        );
    }

    #[test]
    fn threat_awareness_fixture_manifest_is_parseable_and_covers_required_categories() {
        use std::collections::HashMap;

        let mut cases = HashMap::new();
        for line in include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/threat-awareness.epd"
        ))
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        {
            let fields: Vec<&str> = line.split('|').collect();
            assert_eq!(fields.len(), 4, "fixture must have four fields: {line}");
            let position = parse_fen(fields[2]).expect("threat fixture FEN must parse");
            assert!(
                cases
                    .insert(fields[0].to_string(), (fields[1].to_string(), position))
                    .is_none(),
                "fixture ids must be unique: {}",
                fields[0]
            );
        }

        assert_eq!(cases.len(), 10);
        for required in ["king-danger", "king-defense", "development", "pawn-break"] {
            assert!(
                cases.values().any(|(category, _)| category == required),
                "missing threat-awareness category {required}"
            );
        }

        let safe = &cases["king-danger-safe"].1;
        let exposed = &cases["king-danger-exposed"].1;
        assert!(
            evaluate_threat_aware(safe) > evaluate_threat_aware(exposed),
            "paired king-danger fixture must prefer the shielded king"
        );
        let exposed_black = &cases["king-danger-exposed-black"].1;
        assert_eq!(
            evaluate_threat_aware(exposed),
            -evaluate_threat_aware(exposed_black),
            "side-to-move mirror must preserve evaluation symmetry"
        );
    }
}
