//! Position: board state, make/unmake, attack detection, and Perft.
//!
//! This is the heart of Phase 1. `make_move`/`unmake_move` are written so that
//! an `Undo` record fully reverses a move. The rest of the engine
//! (search, UCI) never needs to know how the board is stored.
//!
//! M3.0 adds a derived `zobrist_key` field (crate-private, updated
//! incrementally by `make_move` and restored verbatim by `unmake_move`).
//! All `Position` fields are `pub(crate)`: external code may only read
//! them through the read-only getters below and may only mutate the board
//! through `make_move`/`unmake_move`/`parse_fen`.

use crate::chess::fen::parse_fen;
use crate::chess::movegen::generate_legal_moves;
use crate::chess::types::*;
use crate::chess::zobrist::{
    castling_key, castling_mask, effective_ep_file, ep_file_key, piece_square_key, ZobristKey,
    SIDE_KEY,
};

#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub(crate) board: [Option<Piece>; 64],
    pub(crate) side: Color,
    pub(crate) castling: CastlingRights,
    pub(crate) ep_target: Option<Square>,
    pub(crate) halfmove: u32,
    pub(crate) fullmove: u32,
    /// Tracked king squares so check detection is O(1)-ish lookups.
    pub(crate) king_sq: [Square; 2],
    /// Derived Zobrist key (crate-private; never written by callers).
    pub(crate) zobrist_key: ZobristKey,
}

/// Everything needed to reverse a move. Stored by value so `unmake_move`
/// needs no extra arguments.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    mv: Move,
    moved_piece: Piece,
    captured: Option<Piece>,
    castling: CastlingRights,
    ep_target: Option<Square>,
    halfmove: u32,
    fullmove: u32,
    king_sq: [Square; 2],
    /// `zobrist_key` captured before the move, so `unmake_move` can
    /// restore it verbatim (no reverse hash replay).
    previous_zobrist_key: ZobristKey,
}

impl Position {
    pub fn startpos() -> Position {
        parse_fen(START_FEN).expect("startpos FEN is valid")
    }

    // --- Read-only getters (M3.0 field encapsulation) ---
    pub fn board(&self) -> &[Option<Piece>; 64] {
        &self.board
    }
    pub fn side_to_move(&self) -> Color {
        self.side
    }
    pub fn castling_rights(&self) -> CastlingRights {
        self.castling
    }
    pub fn ep_target(&self) -> Option<Square> {
        self.ep_target
    }
    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove
    }
    pub fn fullmove_number(&self) -> u32 {
        self.fullmove
    }
    pub fn king_square(&self, color: Color) -> Square {
        self.king_sq[color as usize]
    }
    pub fn zobrist_key(&self) -> ZobristKey {
        self.zobrist_key
    }

    /// Apply `m` and return an `Undo` that reverses it.
    pub fn make_move(&mut self, m: Move) -> Undo {
        let us = self.side;
        let enemy = us.opposite();
        let moved_piece = self.board[m.from as usize].expect("make_move from empty square");

        // Resolve captured piece up front (en passant captures a different square).
        let (captured, ep_cap_sq) = match m.flag {
            MoveFlag::EnPassant => {
                let cap_sq = make_square(file_of(m.to), rank_of(m.from));
                (self.board[cap_sq as usize], Some(cap_sq))
            }
            _ => (self.board[m.to as usize], None),
        };

        let undo = Undo {
            mv: m,
            moved_piece,
            captured,
            castling: self.castling,
            ep_target: self.ep_target,
            halfmove: self.halfmove,
            fullmove: self.fullmove,
            king_sq: self.king_sq,
            previous_zobrist_key: self.zobrist_key,
        };

        // --- Zobrist: strip the OLD state (ep + castling) before the board moves. ---
        if let Some(file) = effective_ep_file(self) {
            self.zobrist_key ^= ep_file_key(file);
        }
        self.zobrist_key ^= castling_key(castling_mask(&self.castling));

        // --- board changes + piece-square XORs ---
        self.board[m.from as usize] = None;
        self.xor_piece(moved_piece, m.from);

        match m.flag {
            MoveFlag::EnPassant => {
                let cap_sq = ep_cap_sq.unwrap();
                self.board[cap_sq as usize] = None;
                self.xor_piece(captured.unwrap(), cap_sq);
                self.board[m.to as usize] = Some(moved_piece);
                self.xor_piece(moved_piece, m.to);
            }
            MoveFlag::KingCastle => {
                let (rf, rt) = if us == Color::White {
                    (H1, F1)
                } else {
                    (H8, F8)
                };
                let rook = self.board[rf as usize].expect("king castle rook missing");
                self.board[rf as usize] = None;
                self.xor_piece(rook, rf);
                self.board[rt as usize] = Some(rook);
                self.xor_piece(rook, rt);
                self.board[m.to as usize] = Some(moved_piece);
                self.xor_piece(moved_piece, m.to);
            }
            MoveFlag::QueenCastle => {
                let (rf, rt) = if us == Color::White {
                    (A1, D1)
                } else {
                    (A8, D8)
                };
                let rook = self.board[rf as usize].expect("queen castle rook missing");
                self.board[rf as usize] = None;
                self.xor_piece(rook, rf);
                self.board[rt as usize] = Some(rook);
                self.xor_piece(rook, rt);
                self.board[m.to as usize] = Some(moved_piece);
                self.xor_piece(moved_piece, m.to);
            }
            MoveFlag::Promotion(pt) => {
                if let Some(cap) = captured {
                    self.board[m.to as usize] = None;
                    self.xor_piece(cap, m.to);
                }
                let promoted = Piece::new(us, pt);
                self.board[m.to as usize] = Some(promoted);
                self.xor_piece(promoted, m.to);
            }
            _ => {
                if let Some(cap) = captured {
                    self.board[m.to as usize] = None;
                    self.xor_piece(cap, m.to);
                }
                self.board[m.to as usize] = Some(moved_piece);
                self.xor_piece(moved_piece, m.to);
            }
        }

        self.ep_target = if m.flag == MoveFlag::DoublePawnPush {
            Some(make_square(
                file_of(m.to),
                (rank_of(m.from) + rank_of(m.to)) / 2,
            ))
        } else {
            None
        };

        self.update_castling_rights(m, us);

        self.halfmove = if moved_piece.piece_type == PieceType::Pawn || captured.is_some() {
            0
        } else {
            self.halfmove + 1
        };
        if us == Color::Black {
            self.fullmove += 1;
        }
        if moved_piece.piece_type == PieceType::King {
            self.king_sq[us as usize] = m.to;
        }
        self.side = enemy;

        // --- Zobrist: add the NEW state (side + castling + ep). ---
        self.zobrist_key ^= SIDE_KEY;
        self.zobrist_key ^= castling_key(castling_mask(&self.castling));
        if let Some(file) = effective_ep_file(self) {
            self.zobrist_key ^= ep_file_key(file);
        }

        undo
    }

    /// Reverse a previously applied move.
    pub fn unmake_move(&mut self, undo: Undo) {
        let m = undo.mv;
        let us = self.side.opposite(); // side was flipped in make_move

        self.side = us;

        // Clear destination, put the moving piece back on its origin.
        self.board[m.to as usize] = None;
        self.board[m.from as usize] = Some(undo.moved_piece);

        match m.flag {
            MoveFlag::EnPassant => {
                let cap_sq = make_square(file_of(m.to), rank_of(m.from));
                self.board[cap_sq as usize] = undo.captured;
            }
            MoveFlag::KingCastle => {
                let (rf, rt) = if us == Color::White {
                    (H1, F1)
                } else {
                    (H8, F8)
                };
                let rook = self.board[rt as usize];
                self.board[rt as usize] = None;
                self.board[rf as usize] = rook;
            }
            MoveFlag::QueenCastle => {
                let (rf, rt) = if us == Color::White {
                    (A1, D1)
                } else {
                    (A8, D8)
                };
                let rook = self.board[rt as usize];
                self.board[rt as usize] = None;
                self.board[rf as usize] = rook;
            }
            MoveFlag::Promotion(_) | MoveFlag::Normal | MoveFlag::DoublePawnPush => {
                if let Some(cap) = undo.captured {
                    self.board[m.to as usize] = Some(cap);
                }
            }
        }

        self.castling = undo.castling;
        self.ep_target = undo.ep_target;
        self.halfmove = undo.halfmove;
        self.fullmove = undo.fullmove;
        self.king_sq = undo.king_sq;
        // Restore the derived key verbatim — no reverse hash replay, so the
        // make/unmake hash logic can never silently drift.
        self.zobrist_key = undo.previous_zobrist_key;
    }

    #[inline]
    fn xor_piece(&mut self, piece: Piece, sq: Square) {
        self.zobrist_key ^= piece_square_key(piece, sq);
    }

    fn update_castling_rights(&mut self, m: Move, _us: Color) {
        let from = m.from;
        let to = m.to;
        if from == E1 || to == E1 {
            self.castling.white_king = false;
            self.castling.white_queen = false;
        }
        if from == E8 || to == E8 {
            self.castling.black_king = false;
            self.castling.black_queen = false;
        }
        if from == H1 || to == H1 {
            self.castling.white_king = false;
        }
        if from == A1 || to == A1 {
            self.castling.white_queen = false;
        }
        if from == H8 || to == H8 {
            self.castling.black_king = false;
        }
        if from == A8 || to == A8 {
            self.castling.black_queen = false;
        }
    }

    /// Is `sq` attacked by any piece of color `by`?
    pub fn is_square_attacked(&self, sq: Square, by: Color) -> bool {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;

        // Pawns: a `by` pawn on (f±1, r - dir) attacks `sq`.
        let pawn_dir = if by == Color::White { -1 } else { 1 };
        for df in [-1i32, 1i32] {
            let pf = f + df;
            let pr = r + pawn_dir;
            if on_board(pf, pr) {
                if let Some(p) = self.board[make_square(pf as u8, pr as u8) as usize] {
                    if p.color == by && p.piece_type == PieceType::Pawn {
                        return true;
                    }
                }
            }
        }

        // Knights.
        for (df, dr) in KNIGHT_OFFSETS {
            let nf = f + df;
            let nr = r + dr;
            if on_board(nf, nr) {
                if let Some(p) = self.board[make_square(nf as u8, nr as u8) as usize] {
                    if p.color == by && p.piece_type == PieceType::Knight {
                        return true;
                    }
                }
            }
        }

        // King.
        for (df, dr) in KING_OFFSETS {
            let nf = f + df;
            let nr = r + dr;
            if on_board(nf, nr) {
                if let Some(p) = self.board[make_square(nf as u8, nr as u8) as usize] {
                    if p.color == by && p.piece_type == PieceType::King {
                        return true;
                    }
                }
            }
        }

        // Sliding pieces (bishop/queen on diagonals, rook/queen on ranks/files).
        for (df, dr) in BISHOP_DIRS {
            let mut nf = f + df;
            let mut nr = r + dr;
            while on_board(nf, nr) {
                let to = make_square(nf as u8, nr as u8);
                if let Some(p) = self.board[to as usize] {
                    if p.color == by
                        && (p.piece_type == PieceType::Bishop || p.piece_type == PieceType::Queen)
                    {
                        return true;
                    }
                    break;
                }
                nf += df;
                nr += dr;
            }
        }
        for (df, dr) in ROOK_DIRS {
            let mut nf = f + df;
            let mut nr = r + dr;
            while on_board(nf, nr) {
                let to = make_square(nf as u8, nr as u8);
                if let Some(p) = self.board[to as usize] {
                    if p.color == by
                        && (p.piece_type == PieceType::Rook || p.piece_type == PieceType::Queen)
                    {
                        return true;
                    }
                    break;
                }
                nf += df;
                nr += dr;
            }
        }

        false
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        self.is_square_attacked(self.king_sq[color as usize], color.opposite())
    }

    /// Number of legal moves (used for mate/stalemate detection).
    pub fn legal_move_count(&self) -> usize {
        let mut p = *self;
        generate_legal_moves(&mut p).len()
    }

    pub fn is_checkmate(&self) -> bool {
        self.is_in_check(self.side) && self.legal_move_count() == 0
    }

    pub fn is_stalemate(&self) -> bool {
        !self.is_in_check(self.side) && self.legal_move_count() == 0
    }

    /// Perft: count leaf nodes of strictly legal moves (Phase 2 debugging tool).
    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = generate_legal_moves(self);
        let mut nodes = 0u64;
        for m in moves {
            let undo = self.make_move(m);
            nodes += self.perft(depth - 1);
            self.unmake_move(undo);
        }
        nodes
    }

    /// Perft divided by root move — shows which root move is wrong when the
    /// total does not match the reference.
    pub fn perft_divide(&mut self, depth: u32) -> Vec<(String, u64)> {
        let moves = generate_legal_moves(self);
        let mut out = Vec::new();
        for m in moves {
            let undo = self.make_move(m);
            let n = self.perft(depth - 1);
            self.unmake_move(undo);
            out.push((move_to_uci(m), n));
        }
        out
    }
}
