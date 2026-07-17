//! FEN parsing/output and ASCII board printing.

use crate::chess::types::*;
use crate::chess::position::Position;

pub fn parse_fen(fen: &str) -> Result<Position, String> {
    let mut parts = fen.trim().split_whitespace();
    let placement = parts.next().ok_or("missing piece placement")?;
    let side_str = parts.next().ok_or("missing side to move")?;
    let castling_str = parts.next().ok_or("missing castling rights")?;
    let ep_str = parts.next().ok_or("missing en passant target")?;
    let half_str = parts.next();
    let full_str = parts.next();

    let mut board = [None; 64];
    let mut king_sq = [0u8; 2];

    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 {
        return Err("piece placement must have 8 ranks".into());
    }
    for (i, rank_str) in ranks.iter().enumerate() {
        // i = 0 is rank 8 (the top of a FEN), which is our rank index 7.
        let our_rank = 7 - i as u8;
        let mut file = 0u8;
        for ch in rank_str.chars() {
            if let Some(d) = ch.to_digit(10) {
                file += d as u8;
            } else {
                let piece = Piece::from_char(ch)
                    .ok_or_else(|| format!("invalid piece character '{}'", ch))?;
                let sq = make_square(file, our_rank);
                board[sq as usize] = Some(piece);
                if piece.piece_type == PieceType::King {
                    king_sq[piece.color as usize] = sq;
                }
                file += 1;
            }
        }
    }

    let side = match side_str {
        "w" => Color::White,
        "b" => Color::Black,
        _ => return Err(format!("invalid side to move '{}'", side_str)),
    };

    let mut castling = CastlingRights {
        white_king: false,
        white_queen: false,
        black_king: false,
        black_queen: false,
    };
    if castling_str != "-" {
        for ch in castling_str.chars() {
            match ch {
                'K' => castling.white_king = true,
                'Q' => castling.white_queen = true,
                'k' => castling.black_king = true,
                'q' => castling.black_queen = true,
                _ => return Err(format!("invalid castling character '{}'", ch)),
            }
        }
    }

    let ep_target = if ep_str == "-" {
        None
    } else {
        Some(parse_square(ep_str)?)
    };

    let halfmove = half_str
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let fullmove = full_str
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);

    Ok(Position {
        board,
        side,
        castling,
        ep_target,
        halfmove,
        fullmove,
        king_sq,
    })
}

pub fn to_fen(pos: &Position) -> String {
    let mut s = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            let sq = make_square(file, rank);
            match pos.board[sq as usize] {
                Some(p) => {
                    if empty > 0 {
                        s.push_str(&empty.to_string());
                        empty = 0;
                    }
                    s.push(p.to_char());
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            s.push_str(&empty.to_string());
        }
        if rank > 0 {
            s.push('/');
        }
    }
    s.push(' ');
    s.push(match pos.side {
        Color::White => 'w',
        Color::Black => 'b',
    });
    s.push(' ');

    let mut c = String::new();
    if pos.castling.white_king {
        c.push('K');
    }
    if pos.castling.white_queen {
        c.push('Q');
    }
    if pos.castling.black_king {
        c.push('k');
    }
    if pos.castling.black_queen {
        c.push('q');
    }
    if c.is_empty() {
        c.push('-');
    }
    s.push_str(&c);
    s.push(' ');

    match pos.ep_target {
        Some(sq) => s.push_str(&square_name(sq)),
        None => s.push('-'),
    }
    s.push(' ');
    s.push_str(&pos.halfmove.to_string());
    s.push(' ');
    s.push_str(&pos.fullmove.to_string());
    s
}

/// Print the board to stdout as simple ASCII (Phase 1 sanity check).
pub fn print_ascii(pos: &Position) {
    for rank in (0..8).rev() {
        print!("{} ", rank + 1);
        for file in 0..8 {
            let sq = make_square(file, rank);
            match pos.board[sq as usize] {
                Some(p) => print!("{} ", p.to_char()),
                None => print!(". "),
            }
        }
        println!();
    }
    println!("  a b c d e f g h");
}
