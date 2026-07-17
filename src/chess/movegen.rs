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
    let mut pseudo = Vec::new();
    generate_pseudo_moves(pos, &mut pseudo);
    let us = pos.side;
    let mut legal = Vec::new();
    for m in pseudo {
        let undo = pos.make_move(m);
        if !pos.is_square_attacked(pos.king_sq[us as usize], us.opposite()) {
            legal.push(m);
        }
        pos.unmake_move(undo);
    }
    legal
}

fn push_move(moves: &mut Vec<Move>, from: Square, to: Square, flag: MoveFlag, promo: Option<PieceType>) {
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
