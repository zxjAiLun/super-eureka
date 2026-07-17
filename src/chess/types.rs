//! Core chess types: colors, pieces, squares, moves.
//!
//! Design note (Phase 1): we deliberately use a plain `[Option<Piece>; 64]`
//! board and rich Rust enums instead of bitboards. This keeps the rules code
//! readable and correct first; bitboards can be swapped in later behind the same
//! `Position` API without touching the search or UCI layers.

/// Side to move / piece color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// The six piece kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceType {
    pub fn from_char(c: char) -> Option<PieceType> {
        match c.to_ascii_lowercase() {
            'p' => Some(PieceType::Pawn),
            'n' => Some(PieceType::Knight),
            'b' => Some(PieceType::Bishop),
            'r' => Some(PieceType::Rook),
            'q' => Some(PieceType::Queen),
            'k' => Some(PieceType::King),
            _ => None,
        }
    }

    pub fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }

    /// First version of the evaluation only uses material values (Phase 4 will
    /// extend this with piece-square tables, mobility, etc.).
    pub fn value(self) -> i32 {
        match self {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            PieceType::King => 20000,
        }
    }
}

/// A single piece on the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    pub fn new(color: Color, piece_type: PieceType) -> Piece {
        Piece { color, piece_type }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        let color = if c.is_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        PieceType::from_char(c).map(|pt| Piece::new(color, pt))
    }

    pub fn to_char(self) -> char {
        let c = self.piece_type.to_char();
        if self.color == Color::White {
            c.to_ascii_uppercase()
        } else {
            c
        }
    }
}

/// A square is just an index 0..=63.
/// Index = `rank * 8 + file`, where rank 0 = rank "1" (White's back rank),
/// file 0 = file "a". So a1 = 0, h1 = 7, a8 = 56, h8 = 63.
pub type Square = u8;

pub const A1: Square = 0;
pub const B1: Square = 1;
pub const C1: Square = 2;
pub const D1: Square = 3;
pub const E1: Square = 4;
pub const F1: Square = 5;
pub const G1: Square = 6;
pub const H1: Square = 7;
pub const A8: Square = 56;
pub const B8: Square = 57;
pub const C8: Square = 58;
pub const D8: Square = 59;
pub const E8: Square = 60;
pub const F8: Square = 61;
pub const G8: Square = 62;
pub const H8: Square = 63;

pub const START_FEN: &str =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[inline]
pub fn file_of(sq: Square) -> u8 {
    sq % 8
}

#[inline]
pub fn rank_of(sq: Square) -> u8 {
    sq / 8
}

#[inline]
pub fn make_square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

#[inline]
pub fn on_board(file: i32, rank: i32) -> bool {
    file >= 0 && file < 8 && rank >= 0 && rank < 8
}

pub fn square_name(sq: Square) -> String {
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{}{}", f, r)
}

pub fn parse_square(s: &str) -> Result<Square, String> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("invalid square '{}'", s));
    }
    let file = (bytes[0] as char).to_ascii_lowercase() as i32 - b'a' as i32;
    let rank = bytes[1] as i32 - b'1' as i32;
    if file < 0 || file > 7 || rank < 0 || rank > 7 {
        return Err(format!("invalid square '{}'", s));
    }
    Ok(make_square(file as u8, rank as u8))
}

/// How a move is special; used by `make_move`/`unmake_move`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveFlag {
    Normal,
    DoublePawnPush,
    EnPassant,
    KingCastle,
    QueenCastle,
    Promotion(PieceType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceType>,
    pub flag: MoveFlag,
}

/// Serialize a move to UCI notation, e.g. "e2e4" or "e7e8q".
pub fn move_to_uci(m: Move) -> String {
    let mut s = format!("{}{}", square_name(m.from), square_name(m.to));
    if let Some(pt) = m.promotion {
        s.push(pt.to_char().to_ascii_lowercase());
    }
    s
}

/// Castling availability for both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastlingRights {
    pub white_king: bool,
    pub white_queen: bool,
    pub black_king: bool,
    pub black_queen: bool,
}

/// Knight move offsets (file, rank).
pub const KNIGHT_OFFSETS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

/// King move offsets (one step in every direction).
pub const KING_OFFSETS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

pub const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
pub const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const QUEEN_DIRS: [(i32, i32); 8] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
];
