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
use crate::chess::types::{file_of, make_square, rank_of, Color, Piece, PieceType, KING_OFFSETS};

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

/// Cached base tapered-evaluation lanes. The values are always White minus
/// Black; the side-to-move perspective is applied only when evaluating.
/// Exact KQK/KRK mop-up is deliberately not cached in this structure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BaseEvalCache {
    pub(crate) white_minus_black_mg: i32,
    pub(crate) white_minus_black_eg: i32,
    pub(crate) phase: i32,
}

#[inline]
fn signed_piece_score(piece: Piece, sq: usize) -> BaseEvalCache {
    let sign = if piece.color == Color::White { 1 } else { -1 };
    let material = material_score(piece.piece_type);
    let positional = pst_score(piece.piece_type, pst_idx(sq, piece.color));
    BaseEvalCache {
        white_minus_black_mg: sign * (material.mg + positional.mg),
        white_minus_black_eg: sign * (material.eg + positional.eg),
        phase: phase_weight(piece.piece_type),
    }
}

/// Return the incremental contribution of one piece on one square.
pub(crate) fn base_eval_piece_delta(piece: Piece, sq: usize) -> BaseEvalCache {
    signed_piece_score(piece, sq)
}

/// Recompute only the cached base tapered lanes from the board. FEN parsing
/// uses this once; the search candidate then maintains the result through
/// make/unmake rather than calling this on every leaf.
pub(crate) fn recompute_base_eval(pos: &Position) -> BaseEvalCache {
    let mut cache = BaseEvalCache::default();
    for (sq, piece) in pos.board.iter().enumerate() {
        if let Some(piece) = piece {
            let delta = signed_piece_score(*piece, sq);
            cache.white_minus_black_mg += delta.white_minus_black_mg;
            cache.white_minus_black_eg += delta.white_minus_black_eg;
            cache.phase += delta.phase;
        }
    }
    cache
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

/// Evaluate the same full-board function while computing MG, EG, and phase
/// in one 64-square pass. This is a bench-only candidate: it deliberately
/// keeps the existing `Position`, make/unmake, and exact KQK/KRK paths
/// unchanged.
pub(crate) fn evaluate_one_pass(pos: &Position) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    let mut phase = 0;
    for sq in 0..64usize {
        if let Some(p) = pos.board[sq] {
            let sign = if p.color == pos.side { 1 } else { -1 };
            let material = material_score(p.piece_type);
            let positional = pst_score(p.piece_type, pst_idx(sq, p.color));
            mg += sign * (material.mg + positional.mg);
            eg += sign * (material.eg + positional.eg);
            phase += phase_weight(p.piece_type);
        }
    }
    finish_evaluation(pos, mg, eg, phase.min(MAX_PHASE))
}

/// Evaluate the same function as [`evaluate`] using the Position-maintained
/// base tapered lanes. Exact KQK/KRK mop-up intentionally remains the
/// existing board-scan path, so D1.6 changes only the base evaluation cost.
pub(crate) fn evaluate_incremental(pos: &Position) -> i32 {
    let cache = pos
        .eval_cache
        .expect("incremental evaluation requires an active cache");
    let perspective = if pos.side == Color::White { 1 } else { -1 };
    finish_evaluation(
        pos,
        perspective * cache.white_minus_black_mg,
        perspective * cache.white_minus_black_eg,
        cache.phase.min(MAX_PHASE),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate, evaluate_incremental, evaluate_one_pass, exact_mop_up, game_phase, interpolate,
        KING_EG_PST, KING_MG_PST, MAX_PHASE,
    };
    use crate::chess::fen::{parse_fen, to_fen};
    use crate::chess::movegen::generate_legal_moves;
    use crate::chess::types::{move_to_uci, START_FEN};

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
    fn one_pass_evaluation_matches_two_pass_evaluation() {
        let fens = [
            START_FEN,
            "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/R7/4K3 b - - 0 1",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        ];

        for fen in fens {
            let pos = parse_fen(fen).expect("one-pass fixture must parse");
            let before = to_fen(&pos);
            assert_eq!(evaluate_one_pass(&pos), evaluate(&pos), "{fen}");
            assert_eq!(
                to_fen(&pos),
                before,
                "one-pass evaluation must be read-only"
            );
        }

        let mut walk = parse_fen(START_FEN).unwrap();
        let mut seed = 0xD17_0A55_u64;
        for _ in 0..512 {
            assert_eq!(evaluate_one_pass(&walk), evaluate(&walk));
            let legal = generate_legal_moves(&mut walk);
            assert!(!legal.is_empty(), "deterministic walk ended early");
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mv = legal[(seed as usize) % legal.len()];
            let undo = walk.make_move(mv);
            assert_eq!(evaluate_one_pass(&walk), evaluate(&walk));
            walk.unmake_move(undo);
            let _ = walk.make_move(mv);
        }
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
    fn incremental_base_matches_full_for_special_moves_and_unmake() {
        let cases = [
            (START_FEN, "e2e4"),
            (
                "rnbqkbnr/pppp1ppp/8/3pP3/8/8/PPPP2PP/RNBQKBNR w KQkq d6 0 3",
                "e5d6",
            ),
            ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", "a2a3"),
            ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", "a2a4"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
            ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1q"),
            ("4k3/8/8/8/8/8/1p6/R3K3 b - - 0 1", "b2a1q"),
        ];

        for (fen, uci) in cases {
            let mut pos = parse_fen(fen).expect("test FEN must parse");
            let previous_cache = pos.begin_incremental_eval();
            let before = evaluate(&pos);
            let mv = generate_legal_moves(&mut pos.clone())
                .into_iter()
                .find(|mv| move_to_uci(*mv) == uci)
                .unwrap_or_else(|| panic!("missing legal move {uci} in {fen}"));
            let undo = pos.make_move(mv);
            assert_eq!(evaluate_incremental(&pos), evaluate(&pos), "after {uci}");
            pos.unmake_move(undo);
            assert_eq!(evaluate_incremental(&pos), before, "after unmake {uci}");
            assert_eq!(pos.eval_cache, Some(super::recompute_base_eval(&pos)));
            pos.end_incremental_eval(previous_cache);
            assert_eq!(pos.eval_cache, previous_cache);
        }
    }

    #[test]
    fn ordinary_position_moves_do_not_maintain_candidate_cache() {
        let mut pos = parse_fen(START_FEN).unwrap();
        assert!(pos.eval_cache.is_none());
        let mv = generate_legal_moves(&mut pos.clone())
            .into_iter()
            .next()
            .expect("startpos has a legal move");
        let undo = pos.make_move(mv);
        assert!(pos.eval_cache.is_none());
        pos.unmake_move(undo);
        assert!(pos.eval_cache.is_none());
    }

    #[test]
    fn incremental_base_matches_full_on_deterministic_legal_walk() {
        let mut pos = parse_fen(START_FEN).unwrap();
        let previous_cache = pos.begin_incremental_eval();
        let mut seed = 0x0D16_0EA1_u64;
        let mut plies = 0;
        for _ in 0..512 {
            assert_eq!(evaluate_incremental(&pos), evaluate(&pos));
            let legal = generate_legal_moves(&mut pos);
            if legal.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mv = legal[(seed as usize) % legal.len()];
            let undo = pos.make_move(mv);
            assert_eq!(evaluate_incremental(&pos), evaluate(&pos));
            pos.unmake_move(undo);
            assert_eq!(evaluate_incremental(&pos), evaluate(&pos));
            let _ = pos.make_move(mv);
            plies += 1;
        }
        assert_eq!(plies, 512);
        pos.end_incremental_eval(previous_cache);
        assert_eq!(pos.eval_cache, previous_cache);
    }
}
