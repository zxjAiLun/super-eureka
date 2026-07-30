//! Move generation (Phase 1).
//!
//! Step 1: generate *pseudo-legal* moves (they ignore whether our own king
//! ends up in check). Step 2: make each move, test if our king is attacked,
//! unmake it — only keep the ones that leave the king safe.

use crate::chess::position::Position;
use crate::chess::types::*;

/// Fill `moves` with every pseudo-legal move for the side to move.
pub fn generate_pseudo_moves(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side;
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color != us {
                continue;
            }
            match p.piece_type {
                PieceType::Pawn => gen_pawn(pos, sq, moves),
                PieceType::Knight => gen_step(pos, sq, moves, &KNIGHT_OFFSETS),
                PieceType::King => {
                    gen_step(pos, sq, moves, &KING_OFFSETS);
                    gen_castling(pos, moves);
                }
                PieceType::Bishop => gen_slider(pos, sq, moves, &BISHOP_DIRS),
                PieceType::Rook => gen_slider(pos, sq, moves, &ROOK_DIRS),
                PieceType::Queen => gen_slider(pos, sq, moves, &QUEEN_DIRS),
            }
        }
    }
}

/// Generate only strictly legal moves.
pub fn generate_legal_moves(pos: &mut Position) -> Vec<Move> {
    generate_legal_moves_with_stats(pos).0
}

/// Search profiling counters for one legal-move generation. The movegen
/// implementation is shared with the public helper above; this variant only
/// exposes counts that are already known while doing the same work, so
/// profiling never generates the move list twice.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MovegenStats {
    pub(crate) pseudo_moves: u64,
    pub(crate) legal_moves: u64,
    pub(crate) make_moves: u64,
    pub(crate) unmake_moves: u64,
}

pub(crate) fn generate_legal_moves_with_stats(pos: &mut Position) -> (Vec<Move>, MovegenStats) {
    let mut pseudo = Vec::new();
    generate_pseudo_moves(pos, &mut pseudo);
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let mut legal = Vec::new();
    for m in pseudo {
        let undo = pos.make_move(m);
        if !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite()) {
            legal.push(m);
        }
        pos.unmake_move(undo);
    }
    let legal_count = legal.len() as u64;
    (
        legal,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: legal_count,
            make_moves: pseudo_count,
            unmake_moves: pseudo_count,
        },
    )
}

/// Generate pseudo-legal moves in the exact relative order of
/// [`generate_pseudo_moves`], but retain only captures, en passant, and all
/// promotions. Quiet non-promotion moves are never materialized.
pub(crate) fn generate_pseudo_tactical_moves(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side;
    for sq in 0u8..64u8 {
        if let Some(p) = pos.board[sq as usize] {
            if p.color != us {
                continue;
            }
            match p.piece_type {
                PieceType::Pawn => gen_pawn_tactical(pos, sq, moves),
                PieceType::Knight | PieceType::King => {
                    gen_step_tactical(
                        pos,
                        sq,
                        moves,
                        if p.piece_type == PieceType::Knight {
                            &KNIGHT_OFFSETS
                        } else {
                            &KING_OFFSETS
                        },
                    );
                }
                PieceType::Bishop => gen_slider_tactical(pos, sq, moves, &BISHOP_DIRS),
                PieceType::Rook => gen_slider_tactical(pos, sq, moves, &ROOK_DIRS),
                PieceType::Queen => gen_slider_tactical(pos, sq, moves, &QUEEN_DIRS),
            }
        }
    }
}

/// Generate legal captures, en passant, and promotions without first
/// materializing quiet non-promotion moves. The order is the same as filtering
/// `generate_legal_moves_with_stats` for tactical moves.
pub(crate) fn generate_legal_tactical_moves_with_stats(
    pos: &mut Position,
) -> (Vec<Move>, MovegenStats) {
    let mut pseudo = Vec::new();
    generate_pseudo_tactical_moves(pos, &mut pseudo);
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let mut legal = Vec::new();
    for m in pseudo {
        let undo = pos.make_move(m);
        if !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite()) {
            legal.push(m);
        }
        pos.unmake_move(undo);
    }
    let legal_count = legal.len() as u64;
    (
        legal,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: legal_count,
            make_moves: pseudo_count,
            unmake_moves: pseudo_count,
        },
    )
}

/// Check whether any legal move exists, stopping after the first legal move.
/// This is used only when a non-check qsearch node has no tactical move: it
/// distinguishes a quiet position from stalemate without building a full
/// legal move list.
pub(crate) fn has_any_legal_move_with_stats(pos: &mut Position) -> (bool, MovegenStats) {
    let mut pseudo = Vec::new();
    generate_pseudo_moves(pos, &mut pseudo);
    let pseudo_count = pseudo.len() as u64;
    let us = pos.side;
    let mut checked = 0u64;
    for m in pseudo {
        checked += 1;
        let undo = pos.make_move(m);
        let legal = !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite());
        pos.unmake_move(undo);
        if legal {
            return (
                true,
                MovegenStats {
                    pseudo_moves: pseudo_count,
                    legal_moves: 1,
                    make_moves: checked,
                    unmake_moves: checked,
                },
            );
        }
    }
    (
        false,
        MovegenStats {
            pseudo_moves: pseudo_count,
            legal_moves: 0,
            make_moves: checked,
            unmake_moves: checked,
        },
    )
}

/// The first qsearch evasion implementation deliberately reuses the complete
/// legal generator. Its separate name makes the check-node contract explicit
/// and leaves room for checker/pin-specific generation later.
pub(crate) fn generate_legal_evasions_with_stats(pos: &mut Position) -> (Vec<Move>, MovegenStats) {
    generate_legal_moves_with_stats(pos)
}

fn push_move(
    moves: &mut Vec<Move>,
    from: Square,
    to: Square,
    flag: MoveFlag,
    promo: Option<PieceType>,
) {
    moves.push(Move {
        from,
        to,
        promotion: promo,
        flag,
    });
}

fn gen_step(pos: &Position, sq: Square, moves: &mut Vec<Move>, offsets: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in offsets {
        let nf = f + df;
        let nr = r + dr;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        match pos.board[to as usize] {
            None => push_move(moves, sq, to, MoveFlag::Normal, None),
            Some(p) if p.color != us => push_move(moves, sq, to, MoveFlag::Normal, None),
            _ => {}
        }
    }
}

fn gen_step_tactical(pos: &Position, sq: Square, moves: &mut Vec<Move>, offsets: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in offsets {
        let nf = f + df;
        let nr = r + dr;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        if matches!(pos.board[to as usize], Some(p) if p.color != us) {
            push_move(moves, sq, to, MoveFlag::Normal, None);
        }
    }
}

fn gen_slider(pos: &Position, sq: Square, moves: &mut Vec<Move>, dirs: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in dirs {
        let mut nf = f + df;
        let mut nr = r + dr;
        while on_board(nf, nr) {
            let to = make_square(nf as u8, nr as u8);
            match pos.board[to as usize] {
                None => push_move(moves, sq, to, MoveFlag::Normal, None),
                Some(p) => {
                    if p.color != us {
                        push_move(moves, sq, to, MoveFlag::Normal, None);
                    }
                    break;
                }
            }
            nf += df;
            nr += dr;
        }
    }
}

fn gen_slider_tactical(pos: &Position, sq: Square, moves: &mut Vec<Move>, dirs: &[(i32, i32)]) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    for (df, dr) in dirs {
        let mut nf = f + df;
        let mut nr = r + dr;
        while on_board(nf, nr) {
            let to = make_square(nf as u8, nr as u8);
            if let Some(p) = pos.board[to as usize] {
                if p.color != us {
                    push_move(moves, sq, to, MoveFlag::Normal, None);
                }
                break;
            }
            nf += df;
            nr += dr;
        }
    }
}

fn add_promotions(moves: &mut Vec<Move>, from: Square, to: Square) {
    for pt in [
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
    ] {
        push_move(moves, from, to, MoveFlag::Promotion(pt), Some(pt));
    }
}

fn gen_pawn(pos: &Position, sq: Square, moves: &mut Vec<Move>) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let dir = if us == Color::White { 1 } else { -1 };
    let start_rank: i32 = if us == Color::White { 1 } else { 6 };
    let promo_rank: i32 = if us == Color::White { 7 } else { 0 };

    // Single push.
    let to1 = make_square(f as u8, (r + dir) as u8);
    if pos.board[to1 as usize].is_none() {
        if r + dir == promo_rank {
            add_promotions(moves, sq, to1);
        } else {
            push_move(moves, sq, to1, MoveFlag::Normal, None);
            // Double push (only from the starting rank, into an empty square).
            if r == start_rank {
                let to2 = make_square(f as u8, (r + 2 * dir) as u8);
                if pos.board[to2 as usize].is_none() {
                    push_move(moves, sq, to2, MoveFlag::DoublePawnPush, None);
                }
            }
        }
    }

    // Captures (including en passant), plus promotions.
    for df in [-1i32, 1i32] {
        let nf = f + df;
        let nr = r + dir;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        match pos.board[to as usize] {
            Some(p) if p.color != us => {
                if nr == promo_rank {
                    add_promotions(moves, sq, to);
                } else {
                    push_move(moves, sq, to, MoveFlag::Normal, None);
                }
            }
            _ => {
                if Some(to) == pos.ep_target {
                    push_move(moves, sq, to, MoveFlag::EnPassant, None);
                }
            }
        }
    }
}

fn gen_pawn_tactical(pos: &Position, sq: Square, moves: &mut Vec<Move>) {
    let us = pos.side;
    let f = file_of(sq) as i32;
    let r = rank_of(sq) as i32;
    let dir = if us == Color::White { 1 } else { -1 };
    let promo_rank: i32 = if us == Color::White { 7 } else { 0 };

    // A quiet promotion is tactical for qsearch and must retain the original
    // pawn generator's position before capture generation.
    let to1 = make_square(f as u8, (r + dir) as u8);
    if pos.board[to1 as usize].is_none() && r + dir == promo_rank {
        add_promotions(moves, sq, to1);
    }

    // Captures (including en passant), plus capturing promotions.
    for df in [-1i32, 1i32] {
        let nf = f + df;
        let nr = r + dir;
        if !on_board(nf, nr) {
            continue;
        }
        let to = make_square(nf as u8, nr as u8);
        match pos.board[to as usize] {
            Some(p) if p.color != us => {
                if nr == promo_rank {
                    add_promotions(moves, sq, to);
                } else {
                    push_move(moves, sq, to, MoveFlag::Normal, None);
                }
            }
            _ if Some(to) == pos.ep_target => {
                push_move(moves, sq, to, MoveFlag::EnPassant, None);
            }
            _ => {}
        }
    }
}

fn gen_castling(pos: &Position, moves: &mut Vec<Move>) {
    let us = pos.side;
    let enemy = us.opposite();
    if us == Color::White {
        let wk = Piece::new(Color::White, PieceType::King);
        let wr = Piece::new(Color::White, PieceType::Rook);
        if pos.castling.white_king
            && pos.board[E1 as usize] == Some(wk)
            && pos.board[H1 as usize] == Some(wr)
            && pos.board[F1 as usize].is_none()
            && pos.board[G1 as usize].is_none()
            && !pos.is_square_attacked(E1, enemy)
            && !pos.is_square_attacked(F1, enemy)
            && !pos.is_square_attacked(G1, enemy)
        {
            push_move(moves, E1, G1, MoveFlag::KingCastle, None);
        }
        if pos.castling.white_queen
            && pos.board[E1 as usize] == Some(wk)
            && pos.board[A1 as usize] == Some(wr)
            && pos.board[B1 as usize].is_none()
            && pos.board[C1 as usize].is_none()
            && pos.board[D1 as usize].is_none()
            && !pos.is_square_attacked(E1, enemy)
            && !pos.is_square_attacked(D1, enemy)
            && !pos.is_square_attacked(C1, enemy)
        {
            push_move(moves, E1, C1, MoveFlag::QueenCastle, None);
        }
    } else {
        let bk = Piece::new(Color::Black, PieceType::King);
        let br = Piece::new(Color::Black, PieceType::Rook);
        if pos.castling.black_king
            && pos.board[E8 as usize] == Some(bk)
            && pos.board[H8 as usize] == Some(br)
            && pos.board[F8 as usize].is_none()
            && pos.board[G8 as usize].is_none()
            && !pos.is_square_attacked(E8, enemy)
            && !pos.is_square_attacked(F8, enemy)
            && !pos.is_square_attacked(G8, enemy)
        {
            push_move(moves, E8, G8, MoveFlag::KingCastle, None);
        }
        if pos.castling.black_queen
            && pos.board[E8 as usize] == Some(bk)
            && pos.board[A8 as usize] == Some(br)
            && pos.board[B8 as usize].is_none()
            && pos.board[C8 as usize].is_none()
            && pos.board[D8 as usize].is_none()
            && !pos.is_square_attacked(E8, enemy)
            && !pos.is_square_attacked(D8, enemy)
            && !pos.is_square_attacked(C8, enemy)
        {
            push_move(moves, E8, C8, MoveFlag::QueenCastle, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::parse_fen;

    fn is_tactical(pos: &Position, m: Move) -> bool {
        m.promotion.is_some()
            || matches!(m.flag, MoveFlag::EnPassant)
            || pos.board[m.to as usize].is_some()
    }

    fn assert_tactical_matches_general(fen: &str) {
        let mut general_pos = parse_fen(fen).expect("fixture must parse");
        let expected: Vec<Move> = generate_legal_moves(&mut general_pos)
            .into_iter()
            .filter(|m| is_tactical(&general_pos, *m))
            .collect();
        let mut specialized_pos = parse_fen(fen).expect("fixture must parse");
        let (actual, _) = generate_legal_tactical_moves_with_stats(&mut specialized_pos);
        assert_eq!(actual, expected, "tactical move order differs for {fen}");
        assert_eq!(specialized_pos.zobrist_key, general_pos.zobrist_key);
    }

    fn assert_evasions_match_general(fen: &str) {
        let mut general_pos = parse_fen(fen).expect("fixture must parse");
        let expected = generate_legal_moves(&mut general_pos);
        let mut specialized_pos = parse_fen(fen).expect("fixture must parse");
        let (actual, _) = generate_legal_evasions_with_stats(&mut specialized_pos);
        assert_eq!(actual, expected, "evasion move order differs for {fen}");
        assert_eq!(specialized_pos.zobrist_key, general_pos.zobrist_key);
    }

    #[test]
    fn specialized_tactical_generation_covers_required_move_classes() {
        for fen in [
            // ordinary capture
            "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
            // en passant
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            // quiet promotion
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // capturing promotion
            "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            // pinned capture must be filtered by the legality check
            "4r1k1/8/8/8/8/8/r3R3/4K3 w - - 0 1",
        ] {
            assert_tactical_matches_general(fen);
        }
    }

    #[test]
    fn specialized_evasions_match_all_check_classes() {
        for fen in [
            // single check: capture the checking rook
            "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            // single check: a bishop can block the rook check
            "4r1k1/8/8/8/2B5/8/8/4K3 w - - 0 1",
            // single check: king moves
            "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1",
            // double check: only king evasions are legal
            "4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1",
            // checkmate
            "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
            // stalemate is not a check, but must remain an empty legal list
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        ] {
            assert_evasions_match_general(fen);
        }
    }

    #[test]
    fn tactical_generation_differential_on_deterministic_legal_walk() {
        let mut pos = Position::startpos();
        let mut state = 0x9e37_79b9_u64;
        for _ in 0..256 {
            let expected: Vec<Move> = {
                let mut probe = pos;
                generate_legal_moves(&mut probe)
                    .into_iter()
                    .filter(|m| is_tactical(&probe, *m))
                    .collect()
            };
            let (actual, _) = generate_legal_tactical_moves_with_stats(&mut pos);
            assert_eq!(
                actual, expected,
                "differential mismatch at key {:x}",
                pos.zobrist_key
            );

            if pos.is_in_check(pos.side) {
                let mut evasion_pos = pos;
                let expected_evasions = generate_legal_moves(&mut evasion_pos);
                let (actual_evasions, _) = generate_legal_evasions_with_stats(&mut pos);
                assert_eq!(actual_evasions, expected_evasions);
            }

            let mut next_probe = pos;
            let legal = generate_legal_moves(&mut next_probe);
            if legal.is_empty() {
                break;
            }
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            let next = legal[(state as usize) % legal.len()];
            pos.make_move(next);
        }
    }

    #[test]
    fn has_any_legal_move_distinguishes_quiet_position_and_stalemate() {
        let mut quiet = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut quiet_pseudo = Vec::new();
        generate_pseudo_moves(&quiet, &mut quiet_pseudo);
        let (has_quiet, quiet_stats) = has_any_legal_move_with_stats(&mut quiet);
        assert!(has_quiet);
        assert_eq!(quiet_stats.pseudo_moves, quiet_pseudo.len() as u64);
        assert!(quiet_stats.make_moves < quiet_stats.pseudo_moves);

        let mut stalemate = parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        let mut stalemate_pseudo = Vec::new();
        generate_pseudo_moves(&stalemate, &mut stalemate_pseudo);
        let (has_stalemate_move, stalemate_stats) = has_any_legal_move_with_stats(&mut stalemate);
        assert!(!has_stalemate_move);
        assert_eq!(stalemate_stats.pseudo_moves, stalemate_pseudo.len() as u64);
        assert_eq!(stalemate_stats.make_moves, stalemate_stats.pseudo_moves);
    }
}
