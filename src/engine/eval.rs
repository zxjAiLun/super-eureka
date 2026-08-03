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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Score {
    pub(crate) mg: i32,
    pub(crate) eg: i32,
}

impl Score {
    const fn same(value: i32) -> Score {
        Score {
            mg: value,
            eg: value,
        }
    }
}

/// All score lanes produced while walking one position. The later E2
/// candidate fills the positional lanes; the production evaluator currently
/// leaves them at zero so this refactor is behavior-preserving.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EvalTerms {
    pub(crate) material_pst: Score,
    pub(crate) pawn_structure: Score,
    pub(crate) mobility: Score,
    pub(crate) piece_activity: Score,
    pub(crate) rook_activity: Score,
    pub(crate) development_space: Score,
    pub(crate) king_safety: Score,
}

/// Fixed-storage facts collected from a position in one board walk.
///
/// The attack maps are intentionally lazy in this first framework commit:
/// `Current` does not pay to construct them, while the E2 candidate can build
/// them once and share them across its terms. No field uses a heap-allocated
/// collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EvalContext {
    pub(crate) phase: i32,
    pub(crate) piece_counts: [[u8; 6]; 2],
    pub(crate) piece_squares: [[u64; 6]; 2],
    pub(crate) pawn_files: [[u8; 8]; 2],
    pub(crate) pawn_attacks: [u64; 2],
    pub(crate) occupancy: u64,
    pub(crate) king_squares: [u8; 2],
    /// Pseudo-attacks for the piece currently occupying each square. The
    /// fixed array lets all E2 terms share one attack calculation.
    pub(crate) piece_attacks: [u64; 64],
    pub(crate) attack_maps: [u64; 2],
    pub(crate) attack_maps_ready: bool,
    pub(crate) terms: EvalTerms,
}

impl EvalContext {
    fn from_position(pos: &Position) -> Self {
        let mut context = Self {
            phase: 0,
            piece_counts: [[0; 6]; 2],
            piece_squares: [[0; 6]; 2],
            pawn_files: [[0; 8]; 2],
            pawn_attacks: [0; 2],
            occupancy: 0,
            king_squares: pos.king_sq,
            piece_attacks: [0; 64],
            attack_maps: [0; 2],
            attack_maps_ready: false,
            terms: EvalTerms::default(),
        };

        for (sq, piece) in pos.board.iter().enumerate() {
            let Some(piece) = piece else { continue };
            let color = piece.color as usize;
            let piece_type = piece_type_index(piece.piece_type);
            let square_bit = 1_u64 << sq;
            context.occupancy |= square_bit;
            context.piece_counts[color][piece_type] =
                context.piece_counts[color][piece_type].saturating_add(1);
            context.piece_squares[color][piece_type] |= square_bit;
            context.phase += phase_weight(piece.piece_type);

            if piece.piece_type == PieceType::Pawn {
                context.pawn_files[color][file_of(sq as u8) as usize] =
                    context.pawn_files[color][file_of(sq as u8) as usize].saturating_add(1);
                add_pawn_attacks(&mut context.pawn_attacks[color], sq as u8, piece.color);
            }

            let sign = if piece.color == pos.side { 1 } else { -1 };
            let material = material_score(piece.piece_type);
            let positional = pst_score(piece.piece_type, pst_idx(sq, piece.color));
            context.terms.material_pst.mg += sign * (material.mg + positional.mg);
            context.terms.material_pst.eg += sign * (material.eg + positional.eg);
        }
        context.phase = context.phase.min(MAX_PHASE);
        context
    }

    /// Build one shared pseudo-attack map per side for the integrated E2
    /// candidate. This is deliberately lazy: the approved evaluator never
    /// asks for it, while all E2 terms reuse the same fixed-storage maps.
    fn ensure_attack_maps(&mut self, pos: &Position) {
        if self.attack_maps_ready {
            return;
        }

        self.piece_attacks = [0; 64];
        self.attack_maps = [0; 2];
        for (sq, piece) in pos.board.iter().enumerate() {
            let Some(piece) = piece else { continue };
            let attacks = piece_attack_mask(pos, sq as u8, *piece);
            self.piece_attacks[sq] = attacks;
            self.attack_maps[piece.color as usize] |= attacks;
        }
        self.attack_maps_ready = true;
    }
}

#[inline]
fn piece_type_index(piece_type: PieceType) -> usize {
    match piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 1,
        PieceType::Bishop => 2,
        PieceType::Rook => 3,
        PieceType::Queen => 4,
        PieceType::King => 5,
    }
}

#[inline]
fn add_pawn_attacks(attacks: &mut u64, square: u8, color: Color) {
    let file = file_of(square);
    let rank = rank_of(square);
    match color {
        Color::White => {
            if file > 0 && rank < 7 {
                *attacks |= 1_u64 << (square + 7);
            }
            if file < 7 && rank < 7 {
                *attacks |= 1_u64 << (square + 9);
            }
        }
        Color::Black => {
            if file > 0 && rank > 0 {
                *attacks |= 1_u64 << (square - 9);
            }
            if file < 7 && rank > 0 {
                *attacks |= 1_u64 << (square - 7);
            }
        }
    }
}

#[inline]
fn piece_attack_mask(pos: &Position, from: u8, piece: Piece) -> u64 {
    let file = file_of(from) as i32;
    let rank = rank_of(from) as i32;
    let mut attacks = 0_u64;

    match piece.piece_type {
        PieceType::Pawn => add_pawn_attacks(&mut attacks, from, piece.color),
        PieceType::Knight => {
            for &(df, dr) in &KNIGHT_OFFSETS {
                if on_board(file + df, rank + dr) {
                    attacks |= 1_u64 << make_square((file + df) as u8, (rank + dr) as u8);
                }
            }
        }
        PieceType::King => {
            for &(df, dr) in &KING_OFFSETS {
                if on_board(file + df, rank + dr) {
                    attacks |= 1_u64 << make_square((file + df) as u8, (rank + dr) as u8);
                }
            }
        }
        PieceType::Bishop | PieceType::Rook | PieceType::Queen => {
            let directions: &[(i32, i32)] = match piece.piece_type {
                PieceType::Bishop => &BISHOP_DIRS,
                PieceType::Rook => &ROOK_DIRS,
                PieceType::Queen => &QUEEN_DIRS,
                _ => unreachable!(),
            };
            for &(df, dr) in directions {
                let mut next_file = file + df;
                let mut next_rank = rank + dr;
                while on_board(next_file, next_rank) {
                    let target = make_square(next_file as u8, next_rank as u8);
                    attacks |= 1_u64 << target;
                    if pos.board[target as usize].is_some() {
                        break;
                    }
                    next_file += df;
                    next_rank += dr;
                }
            }
        }
    }

    attacks
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
    EvalContext::from_position(pos).phase
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
    evaluate_breakdown(pos).final_score
}

/// Behavior-preserving base evaluation breakdown. It is crate-visible so the
/// E2 candidate and bench/tests can inspect terms without exposing a GUI or
/// UCI option.
pub(crate) fn evaluate_breakdown(pos: &Position) -> EvalBreakdown {
    let context = EvalContext::from_position(pos);
    let base = context.terms.material_pst;
    let final_score = finish_evaluation(pos, base.mg, base.eg, context.phase);
    EvalBreakdown {
        terms: context.terms,
        phase: context.phase,
        total_mg: base.mg,
        total_eg: base.eg,
        final_score,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EvalBreakdown {
    pub(crate) terms: EvalTerms,
    pub(crate) phase: i32,
    pub(crate) total_mg: i32,
    pub(crate) total_eg: i32,
    pub(crate) final_score: i32,
}

#[inline]
fn white_difference(white: Score, black: Score) -> Score {
    Score {
        mg: white.mg - black.mg,
        eg: white.eg - black.eg,
    }
}

#[inline]
fn from_white_perspective(score: Score, side: Color) -> Score {
    if side == Color::White {
        score
    } else {
        Score {
            mg: -score.mg,
            eg: -score.eg,
        }
    }
}

#[inline]
fn occupied_by(context: &EvalContext, color: Color) -> u64 {
    context.piece_squares[color as usize]
        .iter()
        .fold(0_u64, |occupied, &squares| occupied | squares)
}

/// Return the mobility area used by the positional terms for one piece.
///
/// Minor pieces do not receive mobility credit for squares immediately
/// controlled by an enemy pawn: those squares are tactical destinations that
/// are likely to be challenged or exchanged, rather than stable activity.
/// Keeping this rule in one helper prevents the base mobility and cramped
/// minor terms from silently using different areas.
fn mobility_mask_for_piece(context: &EvalContext, square: usize, piece: Piece) -> u64 {
    let own_occupied = occupied_by(context, piece.color);
    let mut moves = context.piece_attacks[square] & !own_occupied;
    if matches!(piece.piece_type, PieceType::Knight | PieceType::Bishop) {
        moves &= !context.pawn_attacks[piece.color.opposite() as usize];
    }
    moves
}

#[inline]
fn bit(square: u8) -> u64 {
    1_u64 << square
}

fn pawn_is_passed(context: &EvalContext, square: u8, color: Color) -> bool {
    let enemy_pawns =
        context.piece_squares[color.opposite() as usize][piece_type_index(PieceType::Pawn)];
    let file = file_of(square) as i32;
    let rank = rank_of(square) as i32;
    let mut pawns = enemy_pawns;

    while pawns != 0 {
        let enemy_square = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let enemy_file = file_of(enemy_square) as i32;
        let enemy_rank = rank_of(enemy_square) as i32;
        if (enemy_file - file).abs() <= 1
            && ((color == Color::White && enemy_rank > rank)
                || (color == Color::Black && enemy_rank < rank))
        {
            return false;
        }
    }
    true
}

fn pawn_is_connected(context: &EvalContext, square: u8, color: Color) -> bool {
    let own_pawns = context.piece_squares[color as usize][piece_type_index(PieceType::Pawn)];
    let file = file_of(square) as i32;
    let rank = rank_of(square) as i32;
    let mut pawns = own_pawns & !bit(square);

    while pawns != 0 {
        let other = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        if (file_of(other) as i32 - file).abs() == 1 && (rank_of(other) as i32 - rank).abs() <= 1 {
            return true;
        }
    }
    false
}

fn pawn_structure_for_color(context: &EvalContext, color: Color) -> Score {
    let side = color as usize;
    let enemy = color.opposite();
    let pawn_index = piece_type_index(PieceType::Pawn);
    let mut mg = 0;
    let mut eg = 0;

    let mut islands = 0;
    let mut previous_file = false;
    for file in 0..8 {
        let count = context.pawn_files[side][file];
        let present = count != 0;
        if present && !previous_file {
            islands += 1;
        }
        previous_file = present;

        if count > 1 {
            let extra = i32::from(count - 1);
            mg -= 10 * extra;
            eg -= 8 * extra;
        }
        if present
            && (file == 0 || context.pawn_files[side][file - 1] == 0)
            && (file == 7 || context.pawn_files[side][file + 1] == 0)
        {
            mg -= 10;
            eg -= 12;
        }
    }
    mg -= (islands - 2).max(0) * 3;
    eg -= (islands - 2).max(0) * 4;

    let own_pawns = context.piece_squares[side][pawn_index];
    let enemy_pawns = context.piece_squares[enemy as usize][pawn_index];
    let mut pawns = own_pawns;
    while pawns != 0 {
        let square = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let rank = rank_of(square) as i32;
        let advanced = if color == Color::White {
            rank - 1
        } else {
            6 - rank
        };
        let advanced = advanced.clamp(0, 5);

        if pawn_is_passed(context, square, color) {
            mg += 12 + advanced * 3;
            eg += 22 + advanced * 7;
            if context.pawn_attacks[side] & bit(square) != 0 {
                mg += 6;
                eg += 10;
            }
            if pawn_is_connected(context, square, color) {
                mg += 4;
                eg += 7;
            }
        }
    }

    // A small lever term rewards contact with opposing pawns without
    // pretending to solve the full pawn-break/search problem.
    mg += (context.pawn_attacks[side] & enemy_pawns).count_ones() as i32 * 3;

    // Space is deliberately small and symmetric: advanced pawns contribute
    // only when they occupy the opponent's half of the board.
    let mut space = 0;
    let mut pawns = own_pawns;
    while pawns != 0 {
        let square = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let rank = rank_of(square);
        if (color == Color::White && rank >= 4) || (color == Color::Black && rank <= 3) {
            space += 1;
        }
    }
    mg += space * 2;
    eg += space;

    Score { mg, eg }
}

fn undeveloped_minor_count(pos: &Position, color: Color) -> i32 {
    let home = if color == Color::White {
        [
            (make_square(1, 0), PieceType::Knight),
            (make_square(6, 0), PieceType::Knight),
            (make_square(2, 0), PieceType::Bishop),
            (make_square(5, 0), PieceType::Bishop),
        ]
    } else {
        [
            (make_square(1, 7), PieceType::Knight),
            (make_square(6, 7), PieceType::Knight),
            (make_square(2, 7), PieceType::Bishop),
            (make_square(5, 7), PieceType::Bishop),
        ]
    };
    home.iter()
        .filter(|&&(square, expected)| {
            pos.board[square as usize]
                .is_some_and(|piece| piece.color == color && piece.piece_type == expected)
        })
        .count() as i32
}

fn mobility_for_color(pos: &Position, context: &EvalContext, color: Color) -> i32 {
    let mut mobility = 0;
    for (square, piece) in pos.board.iter().enumerate() {
        let Some(piece) = piece else { continue };
        if piece.color != color || piece.piece_type == PieceType::Pawn {
            continue;
        }
        let moves = mobility_mask_for_piece(context, square, *piece);
        let weight = match piece.piece_type {
            PieceType::Knight => 4,
            PieceType::Bishop => 4,
            PieceType::Rook => 2,
            PieceType::Queen => 1,
            PieceType::King | PieceType::Pawn => 0,
        };
        mobility += moves.count_ones() as i32 * weight;
    }
    mobility
}

fn piece_activity_for_color(pos: &Position, context: &EvalContext, color: Color) -> Score {
    let enemy = color.opposite();
    let enemy_pawn_attacks = context.pawn_attacks[enemy as usize];
    let mut mg = 0;
    let mut eg = 0;
    let mut bishops = 0;

    for (square, piece) in pos.board.iter().enumerate() {
        let Some(piece) = piece else { continue };
        if piece.color != color {
            continue;
        }
        let square = square as u8;
        let mobility =
            mobility_mask_for_piece(context, square as usize, *piece).count_ones() as i32;
        match piece.piece_type {
            PieceType::Bishop => {
                bishops += 1;
            }
            PieceType::Knight => {
                let rank = rank_of(square);
                let in_opponent_half =
                    (color == Color::White && rank >= 4) || (color == Color::Black && rank <= 3);
                let is_outpost = in_opponent_half
                    && context.pawn_attacks[color as usize] & bit(square) != 0
                    && enemy_pawn_attacks & bit(square) == 0;
                if is_outpost {
                    mg += 16;
                    eg += 10;
                }
            }
            PieceType::Rook | PieceType::Queen => {}
            PieceType::King | PieceType::Pawn => {}
        }
        if matches!(piece.piece_type, PieceType::Knight | PieceType::Bishop) && mobility <= 2 {
            mg -= 6;
        }
    }

    if bishops >= 2 {
        mg += 24;
        eg += 32;
    }
    Score { mg, eg }
}

fn development_space_for_color(pos: &Position, context: &EvalContext, color: Color) -> Score {
    let side = color as usize;
    let undeveloped = undeveloped_minor_count(pos, color);

    let center = bit(make_square(3, 3))
        | bit(make_square(4, 3))
        | bit(make_square(3, 4))
        | bit(make_square(4, 4));
    let extended_center = center
        | bit(make_square(2, 2))
        | bit(make_square(3, 2))
        | bit(make_square(4, 2))
        | bit(make_square(5, 2))
        | bit(make_square(2, 3))
        | bit(make_square(5, 3))
        | bit(make_square(2, 4))
        | bit(make_square(5, 4))
        | bit(make_square(2, 5))
        | bit(make_square(3, 5))
        | bit(make_square(4, 5))
        | bit(make_square(5, 5));
    let own_attacks = context.attack_maps[side];
    let enemy_attacks = context.attack_maps[color.opposite() as usize];
    let center_control = (own_attacks & center).count_ones() as i32;
    let safe_space =
        (context.pawn_attacks[side] & extended_center & !enemy_attacks).count_ones() as i32;

    let king = context.king_squares[side];
    let king_file = file_of(king);
    let king_rank = rank_of(king);
    let central_king = (2..=5).contains(&king_file) && (2..=5).contains(&king_rank);
    let mut open_nearby = 0;
    for file in (i32::from(king_file) - 1)..=(i32::from(king_file) + 1) {
        if !(0..8).contains(&file) || context.pawn_files[side][file as usize] == 0 {
            open_nearby += 1;
        }
    }
    let king_penalty = if context.phase >= 16 && central_king {
        open_nearby * 5
    } else {
        0
    };

    Score {
        mg: -undeveloped * 10 + center_control * 4 + safe_space * 2 - king_penalty,
        eg: center_control * 2 + safe_space - king_penalty / 2,
    }
}

fn line_clear(pos: &Position, from: u8, to: u8) -> bool {
    let from_file = file_of(from) as i32;
    let from_rank = rank_of(from) as i32;
    let to_file = file_of(to) as i32;
    let to_rank = rank_of(to) as i32;
    let df = (to_file - from_file).signum();
    let dr = (to_rank - from_rank).signum();
    let mut file = from_file + df;
    let mut rank = from_rank + dr;
    while file != to_file || rank != to_rank {
        if pos.board[make_square(file as u8, rank as u8) as usize].is_some() {
            return false;
        }
        file += df;
        rank += dr;
    }
    true
}

fn king_zone_mask(king: u8) -> u64 {
    let file = file_of(king) as i32;
    let rank = rank_of(king) as i32;
    let mut zone = 0_u64;
    for df in -1..=1 {
        for dr in -1..=1 {
            if on_board(file + df, rank + dr) {
                zone |= bit(make_square((file + df) as u8, (rank + dr) as u8));
            }
        }
    }
    zone
}

fn rook_activity_for_color(pos: &Position, context: &EvalContext, color: Color) -> Score {
    let side = color as usize;
    let enemy = color.opposite();
    let own_occupied = occupied_by(context, color);
    let mut mg = 0;
    let mut eg = 0;
    let mut rooks = context.piece_squares[side][piece_type_index(PieceType::Rook)];

    while rooks != 0 {
        let square = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        let file = file_of(square) as usize;
        let attacks = context.piece_attacks[square as usize];
        let mobility = (attacks & !own_occupied).count_ones() as i32;
        let own_pawns = context.pawn_files[side][file];
        let enemy_pawns = context.pawn_files[enemy as usize][file];
        if own_pawns == 0 {
            if enemy_pawns == 0 {
                mg += 14;
                eg += 10;
            } else {
                mg += 8;
                eg += 6;
            }
        }
        if mobility <= 3 {
            mg -= 10;
        }
        let rank = rank_of(square);
        if (color == Color::White && rank == 6) || (color == Color::Black && rank == 1) {
            mg += 16;
            eg += 12;
        }

        let enemy_king_zone = king_zone_mask(context.king_squares[enemy as usize]);
        let own_king_zone = king_zone_mask(context.king_squares[side]);
        if attacks & enemy_king_zone != 0 {
            mg += 6;
        }
        if attacks & own_king_zone != 0 {
            mg += 4;
        }

        let mut connected_rooks =
            context.piece_squares[side][piece_type_index(PieceType::Rook)] & !bit(square);
        while connected_rooks != 0 {
            let other_square = connected_rooks.trailing_zeros() as u8;
            connected_rooks &= connected_rooks - 1;
            if (file_of(other_square) == file_of(square)
                || rank_of(other_square) == rank_of(square))
                && line_clear(pos, square, other_square)
            {
                mg += 7;
                eg += 7;
                break;
            }
        }
    }
    Score { mg, eg }
}

fn king_danger_e2(context: &EvalContext, pos: &Position, victim: Color) -> i32 {
    let side = victim as usize;
    let enemy = victim.opposite();
    let king = context.king_squares[side];
    let zone = king_zone_mask(king);
    let mut attack_units = 0;
    let mut attackers = 0;
    for (square, piece) in pos.board.iter().enumerate() {
        let Some(piece) = piece else { continue };
        if piece.color == enemy && context.piece_attacks[square] & zone != 0 {
            attackers += 1;
            attack_units += match piece.piece_type {
                PieceType::Pawn | PieceType::King => 1,
                PieceType::Knight | PieceType::Bishop => 2,
                PieceType::Rook => 3,
                PieceType::Queen => 4,
            };
        }
    }

    let attacked_zone = (context.attack_maps[enemy as usize] & zone).count_ones() as i32;
    let defended_zone = (context.attack_maps[side] & zone).count_ones() as i32;
    let king_file = file_of(king) as i32;
    let king_rank = rank_of(king) as i32;
    let forward = if victim == Color::White { 1 } else { -1 };
    let shield_rank = king_rank + forward;
    let mut missing_shield = 0;
    let mut storm = 0;
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
                    storm += 1;
                }
                _ => missing_shield += 1,
            }
        }
    }

    let mut open_files = 0;
    for file in (king_file - 1)..=(king_file + 1) {
        if !(0..8).contains(&file) {
            continue;
        }
        if context.pawn_files[side][file as usize] == 0 {
            open_files += 1;
        }
    }

    let nonlinear = [0, 0, 2, 5, 9, 14, 20, 27][attackers.min(7) as usize];
    let raw = (attack_units * attack_units * 2
        + attacked_zone * 3
        + missing_shield * 7
        + storm * 3
        + open_files * 3
        + nonlinear
        - defended_zone * 2)
        .max(0);

    // Queens-on-board is a cheap phase gate. With no queen the signal remains
    // a small shield/activity hint rather than a large artificial king score.
    let queens = context.piece_counts[0][piece_type_index(PieceType::Queen)]
        + context.piece_counts[1][piece_type_index(PieceType::Queen)];
    let gate = match queens {
        0 => 1,
        1 => 2,
        _ => 4,
    };
    (raw * gate * context.phase / (MAX_PHASE * 4)).clamp(0, 160)
}

fn king_safety_score(context: &EvalContext, pos: &Position) -> Score {
    let white_danger = king_danger_e2(context, pos, Color::White);
    let black_danger = king_danger_e2(context, pos, Color::Black);
    let difference = black_danger - white_danger;
    Score {
        mg: difference,
        eg: difference / 4,
    }
}

fn integrated_positional_terms(pos: &Position, context: &mut EvalContext) -> EvalTerms {
    context.ensure_attack_maps(pos);

    let white_pawns = pawn_structure_for_color(context, Color::White);
    let black_pawns = pawn_structure_for_color(context, Color::Black);
    let white_mobility = mobility_for_color(pos, context, Color::White);
    let black_mobility = mobility_for_color(pos, context, Color::Black);
    let white_activity = piece_activity_for_color(pos, context, Color::White);
    let black_activity = piece_activity_for_color(pos, context, Color::Black);
    let white_development = development_space_for_color(pos, context, Color::White);
    let black_development = development_space_for_color(pos, context, Color::Black);
    let white_rooks = rook_activity_for_color(pos, context, Color::White);
    let black_rooks = rook_activity_for_color(pos, context, Color::Black);

    EvalTerms {
        material_pst: context.terms.material_pst,
        pawn_structure: white_difference(white_pawns, black_pawns),
        mobility: Score {
            mg: (white_mobility - black_mobility) * 2,
            eg: white_mobility - black_mobility,
        },
        piece_activity: white_difference(white_activity, black_activity),
        rook_activity: white_difference(white_rooks, black_rooks),
        development_space: white_difference(white_development, black_development),
        king_safety: king_safety_score(context, pos),
    }
}

#[inline]
fn total_terms(terms: EvalTerms) -> (i32, i32) {
    let mg = terms.material_pst.mg
        + terms.pawn_structure.mg
        + terms.mobility.mg
        + terms.piece_activity.mg
        + terms.rook_activity.mg
        + terms.development_space.mg
        + terms.king_safety.mg;
    let eg = terms.material_pst.eg
        + terms.pawn_structure.eg
        + terms.mobility.eg
        + terms.piece_activity.eg
        + terms.rook_activity.eg
        + terms.development_space.eg
        + terms.king_safety.eg;
    (mg, eg)
}

/// Full breakdown for the integrated E2 candidate. Every new term is
/// produced from one shared fixed-storage context and is enabled only for
/// `current-eval2`; the term lanes use the same side-to-move perspective as
/// the existing base lane.
pub(crate) fn evaluate_integrated_breakdown(pos: &Position) -> EvalBreakdown {
    let mut context = EvalContext::from_position(pos);
    let base = context.terms.material_pst;

    // Preserve the exact KQK/KRK path while the candidate is being evaluated;
    // no positional term is allowed to dilute the dedicated mop-up logic.
    if exact_mop_up(pos).is_some() {
        return EvalBreakdown {
            terms: context.terms,
            phase: context.phase,
            total_mg: base.mg,
            total_eg: base.eg,
            final_score: finish_evaluation(pos, base.mg, base.eg, context.phase),
        };
    }

    let white_terms = integrated_positional_terms(pos, &mut context);
    let side_terms = EvalTerms {
        material_pst: base,
        pawn_structure: from_white_perspective(white_terms.pawn_structure, pos.side),
        mobility: from_white_perspective(white_terms.mobility, pos.side),
        piece_activity: from_white_perspective(white_terms.piece_activity, pos.side),
        rook_activity: from_white_perspective(white_terms.rook_activity, pos.side),
        development_space: from_white_perspective(white_terms.development_space, pos.side),
        king_safety: from_white_perspective(white_terms.king_safety, pos.side),
    };
    context.terms = side_terms;

    let (total_mg, total_eg) = total_terms(context.terms);
    EvalBreakdown {
        terms: context.terms,
        phase: context.phase,
        total_mg,
        total_eg,
        final_score: finish_evaluation(pos, total_mg, total_eg, context.phase),
    }
}

/// Integrated E2 candidate evaluation. `Current` continues to call the
/// behavior-preserving base evaluator.
pub(crate) fn evaluate_integrated_positional(pos: &Position) -> i32 {
    evaluate_integrated_breakdown(pos).final_score
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
        evaluate, evaluate_breakdown, evaluate_integrated_breakdown,
        evaluate_integrated_positional, evaluate_threat_aware, exact_mop_up, game_phase,
        interpolate, mobility_for_color, mobility_mask_for_piece, pawn_is_connected,
        piece_activity_for_color, undeveloped_minor_count, EvalContext, Score, KING_EG_PST,
        KING_MG_PST, MAX_PHASE,
    };
    use crate::chess::fen::parse_fen;
    use crate::chess::fen::to_fen;
    use crate::chess::types::{make_square, Color};

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
    fn eval_context_preserves_base_evaluation_and_uses_fixed_storage() {
        let position =
            parse_fen("r1bq1rk1/ppp2ppp/2n1pn2/8/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1").unwrap();
        let context = EvalContext::from_position(&position);
        let breakdown = evaluate_breakdown(&position);

        assert_eq!(breakdown.final_score, evaluate(&position));
        assert_eq!(breakdown.phase, context.phase);
        assert_eq!(breakdown.total_mg, context.terms.material_pst.mg);
        assert_eq!(breakdown.total_eg, context.terms.material_pst.eg);
        assert_eq!(context.occupancy.count_ones(), 30);
        assert!(!context.attack_maps_ready);
        assert_eq!(context.terms.pawn_structure, Score::default());
        assert_eq!(context.terms.mobility, Score::default());
        assert!(std::mem::size_of::<EvalContext>() > 0);

        let mut attacked = context;
        attacked.ensure_attack_maps(&position);
        assert!(attacked.attack_maps_ready);
        assert_eq!(
            attacked
                .piece_attacks
                .iter()
                .filter(|&&attacks| attacks != 0)
                .count(),
            30,
            "one cached attack mask per occupied square"
        );
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
    fn integrated_eval_is_side_symmetric_and_preserves_exact_mopup() {
        let white = parse_fen("r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PPP2PPP/R3K2R w KQkq - 0 1").unwrap();
        let black = parse_fen("r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PPP2PPP/R3K2R b KQkq - 0 1").unwrap();
        let before = to_fen(&white);
        assert_eq!(
            evaluate_integrated_positional(&white),
            -evaluate_integrated_positional(&black)
        );
        assert_eq!(to_fen(&white), before);

        let kqk = parse_fen("7k/8/5K2/8/3Q4/8/8/8 w - - 0 1").unwrap();
        assert_eq!(
            evaluate_integrated_positional(&kqk),
            evaluate(&kqk),
            "E2 must leave the exact KQK mop-up path unchanged"
        );
    }

    #[test]
    fn connected_pawns_exclude_same_file_doubled_pawns() {
        let adjacent = parse_fen("4k3/8/8/3PP3/8/8/8/4K3 w - - 0 1").unwrap();
        let offset = parse_fen("4k3/8/8/3P4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let doubled = parse_fen("4k3/8/3P4/3P4/8/8/8/4K3 w - - 0 1").unwrap();

        assert!(pawn_is_connected(
            &EvalContext::from_position(&adjacent),
            make_square(3, 4),
            Color::White
        ));
        assert!(pawn_is_connected(
            &EvalContext::from_position(&offset),
            make_square(3, 4),
            Color::White
        ));
        assert!(!pawn_is_connected(
            &EvalContext::from_position(&doubled),
            make_square(3, 4),
            Color::White
        ));
    }

    #[test]
    fn development_counts_exact_minor_piece_identity() {
        let correct = parse_fen("4k3/8/8/8/8/8/8/1N1B2NK w - - 0 1").unwrap();
        let wrong_piece = parse_fen("4k3/8/8/8/8/8/8/1R1B2NK w - - 0 1").unwrap();

        assert_eq!(undeveloped_minor_count(&correct, Color::White), 2);
        assert_eq!(undeveloped_minor_count(&wrong_piece, Color::White), 1);
    }

    #[test]
    fn integrated_breakdown_exposes_positional_lanes() {
        let position =
            parse_fen("r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PPP2PPP/R3K2R w KQkq - 0 1").unwrap();
        let breakdown = evaluate_integrated_breakdown(&position);
        let terms = breakdown.terms;
        assert!(
            terms.pawn_structure != Score::default()
                || terms.mobility != Score::default()
                || terms.piece_activity != Score::default()
                || terms.rook_activity != Score::default()
                || terms.development_space != Score::default()
                || terms.king_safety != Score::default(),
            "integrated breakdown must expose at least one non-base lane"
        );
        assert_eq!(
            breakdown.final_score,
            evaluate_integrated_positional(&position)
        );
    }

    #[test]
    fn minor_mobility_area_excludes_enemy_pawn_attacks_and_shares_cramped_rule() {
        let open = parse_fen("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1").unwrap();
        let pawn_challenged = parse_fen("4k3/8/8/8/1p6/8/8/1N2K3 w - - 0 1").unwrap();

        let mut open_context = EvalContext::from_position(&open);
        open_context.ensure_attack_maps(&open);
        let open_piece = open.board[make_square(1, 0) as usize].unwrap();
        let open_safe =
            mobility_mask_for_piece(&open_context, make_square(1, 0) as usize, open_piece);

        let mut challenged_context = EvalContext::from_position(&pawn_challenged);
        challenged_context.ensure_attack_maps(&pawn_challenged);
        let challenged_piece = pawn_challenged.board[make_square(1, 0) as usize].unwrap();
        let challenged_safe = mobility_mask_for_piece(
            &challenged_context,
            make_square(1, 0) as usize,
            challenged_piece,
        );

        assert!(
            challenged_safe.count_ones() < open_safe.count_ones(),
            "enemy pawn attacks must reduce minor mobility area"
        );
        assert_eq!(
            mobility_for_color(&open, &open_context, Color::White),
            open_safe.count_ones() as i32 * 4
        );
        assert_eq!(
            mobility_for_color(&pawn_challenged, &challenged_context, Color::White),
            challenged_safe.count_ones() as i32 * 4
        );

        let open_activity = piece_activity_for_color(&open, &open_context, Color::White);
        let challenged_activity =
            piece_activity_for_color(&pawn_challenged, &challenged_context, Color::White);
        assert!(
            challenged_activity.mg < open_activity.mg,
            "cramped minor activity must use the same safe mobility count"
        );
    }

    #[test]
    fn integrated_positional_terms_have_directional_relationships() {
        let passed = parse_fen("4k3/8/8/4P3/3p4/8/3P4/4K3 w - - 0 1").unwrap();
        let blocked = parse_fen("4k3/3p4/8/4P3/8/8/3P4/4K3 w - - 0 1").unwrap();
        let passed_pawn_score = evaluate_integrated_breakdown(&passed)
            .terms
            .pawn_structure
            .mg;
        let blocked_pawn_score = evaluate_integrated_breakdown(&blocked)
            .terms
            .pawn_structure
            .mg;
        assert!(
            passed_pawn_score > blocked_pawn_score,
            "a passed pawn should score above a pawn stopped by an enemy pawn: {passed_pawn_score} > {blocked_pawn_score}"
        );

        let protected = parse_fen("4k3/p7/8/3pP3/3P4/8/8/4K3 w - - 0 1").unwrap();
        let unsupported = parse_fen("4k3/p7/8/3pP3/8/8/P7/4K3 w - - 0 1").unwrap();
        assert!(
            evaluate_integrated_breakdown(&protected)
                .terms
                .pawn_structure
                .mg
                > evaluate_integrated_breakdown(&unsupported)
                    .terms
                    .pawn_structure
                    .mg,
            "a protected passer should score above an unsupported passer"
        );

        let open_file = parse_fen("r5k1/8/8/8/8/8/P7/4R1K1 w - - 0 1").unwrap();
        let blocked_file = parse_fen("r5k1/8/8/8/8/8/4P3/4R1K1 w - - 0 1").unwrap();
        assert!(
            evaluate_integrated_breakdown(&open_file)
                .terms
                .rook_activity
                .mg
                > evaluate_integrated_breakdown(&blocked_file)
                    .terms
                    .rook_activity
                    .mg,
            "an open-file rook should score above a rook blocked by its pawn"
        );

        let connected_rooks = parse_fen("r5k1/8/8/8/8/8/8/R3R1K1 w - - 0 1").unwrap();
        let separated_rooks = parse_fen("r5k1/8/8/8/8/8/R7/4R1K1 w - - 0 1").unwrap();
        assert!(
            evaluate_integrated_breakdown(&connected_rooks)
                .terms
                .rook_activity
                .mg
                > evaluate_integrated_breakdown(&separated_rooks)
                    .terms
                    .rook_activity
                    .mg,
            "connected rooks should score above separated rooks"
        );

        let bishop_pair = parse_fen("6k1/8/8/8/8/8/8/2B1B1K1 w - - 0 1").unwrap();
        let single_bishop = parse_fen("6k1/8/8/8/8/8/8/2B3K1 w - - 0 1").unwrap();
        assert!(
            evaluate_integrated_breakdown(&bishop_pair)
                .terms
                .piece_activity
                .mg
                > evaluate_integrated_breakdown(&single_bishop)
                    .terms
                    .piece_activity
                    .mg,
            "the bishop pair should score above a single bishop"
        );

        let supported_outpost = parse_fen("6k1/8/8/4N3/3P4/8/8/6K1 w - - 0 1").unwrap();
        let challenged_outpost = parse_fen("6k1/8/3p4/4N3/3P4/8/8/6K1 w - - 0 1").unwrap();
        let supported_outpost_score = evaluate_integrated_breakdown(&supported_outpost)
            .terms
            .piece_activity
            .mg;
        let challenged_outpost_score = evaluate_integrated_breakdown(&challenged_outpost)
            .terms
            .piece_activity
            .mg;
        assert!(
            supported_outpost_score > challenged_outpost_score,
            "a supported outpost should score above a pawn-challenged square: {supported_outpost_score} > {challenged_outpost_score}"
        );

        let shielded_king = parse_fen("r5k1/8/8/8/8/6q1/5PPP/R5K1 w - - 0 1").unwrap();
        let exposed_king = parse_fen("r5k1/8/8/8/8/6q1/8/R5K1 w - - 0 1").unwrap();
        let shielded_terms = evaluate_integrated_breakdown(&shielded_king)
            .terms
            .king_safety
            .mg;
        let exposed_terms = evaluate_integrated_breakdown(&exposed_king)
            .terms
            .king_safety
            .mg;
        assert!(
            shielded_terms > exposed_terms,
            "a shielded king should score above an exposed king: {shielded_terms} > {exposed_terms}"
        );

        let no_queen = parse_fen("r5k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1").unwrap();
        let with_queen = parse_fen("rq4k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1").unwrap();
        let no_queen_term = evaluate_integrated_breakdown(&no_queen)
            .terms
            .king_safety
            .mg
            .abs();
        let with_queen_term = evaluate_integrated_breakdown(&with_queen)
            .terms
            .king_safety
            .mg
            .abs();
        assert!(
            with_queen_term > no_queen_term,
            "queen-on-board king safety should use a larger gate"
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
