//! FEN parsing/output and ASCII board printing.

use crate::chess::position::Position;
use crate::chess::types::*;
use crate::chess::zobrist::recompute_zobrist;

pub fn parse_fen(fen: &str) -> Result<Position, String> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().ok_or("missing piece placement")?;
    let side_str = parts.next().ok_or("missing side to move")?;
    let castling_str = parts.next().ok_or("missing castling rights")?;
    let ep_str = parts.next().ok_or("missing en passant target")?;
    let half_str = parts.next();
    let full_str = parts.next();
    // Reject any extra trailing fields so a malformed FEN cannot silently
    // be accepted as valid.
    if parts.next().is_some() {
        return Err("FEN has too many fields".into());
    }

    let mut board = [None; 64];
    let mut king_sq = [0u8; 2];
    let mut white_kings = 0u8;
    let mut black_kings = 0u8;

    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 {
        return Err("piece placement must have 8 ranks".into());
    }
    for (i, rank_str) in ranks.iter().enumerate() {
        // i = 0 is rank 8 (the top of a FEN), which is our rank index 7.
        let our_rank = 7 - i as u8;
        let mut file = 0u8;
        for ch in rank_str.chars() {
            // Run-length digits must be plain ASCII '1'..='8'. We use an
            // explicit ASCII range rather than `ch.to_digit(10)` so that
            // Unicode digits (e.g. '١' U+0661) are rejected instead of being
            // silently accepted as a legal FEN run-length.
            if ('1'..='8').contains(&ch) {
                let run = (ch as u8) - b'0';
                // `file` is always kept <= 8 at this point, so `8 - file`
                // can never underflow. Checking the remaining room *before*
                // adding catches any rank that would exceed 8 squares and
                // removes the u8 overflow that e.g. "8888...8" triggered in
                // debug builds (file += 8 would wrap past 255).
                if run > 8 - file {
                    return Err(format!(
                        "rank {} has more than 8 squares (digit '{}' overflows)",
                        i + 1,
                        ch
                    ));
                }
                file += run;
            } else {
                // A piece must fall within the 8 squares of the rank. If the
                // run-lengths already filled the rank (or overflowed it), placing
                // a piece here would compute an out-of-bounds square and panic on
                // `board[sq]`. Guard *before* the write.
                if file >= 8 {
                    return Err(format!("rank {} has more than 8 squares", i + 1));
                }
                let piece = Piece::from_char(ch)
                    .ok_or_else(|| format!("invalid piece character '{}'", ch))?;
                let sq = make_square(file, our_rank);
                board[sq as usize] = Some(piece);
                if piece.piece_type == PieceType::King {
                    if piece.color == Color::White {
                        white_kings += 1;
                        king_sq[0] = sq;
                    } else {
                        black_kings += 1;
                        king_sq[1] = sq;
                    }
                }
                file += 1;
            }
        }
        if file != 8 {
            return Err(format!(
                "rank {} must have exactly 8 squares, got {}",
                i + 1,
                file
            ));
        }
    }

    if white_kings != 1 || black_kings != 1 {
        return Err("position must contain exactly one king per side".into());
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
        let sq = parse_square(ep_str)?;
        // The en-passant target must lie on the rank a pawn would have
        // crossed: rank 6 (index 5) when White is to move, rank 3
        // (index 2) when Black is to move.
        if board[sq as usize].is_some() {
            return Err("en passant target square must be empty".into());
        }
        let expected_rank: u8 = if side == Color::White { 5 } else { 2 };
        if rank_of(sq) != expected_rank {
            return Err(format!(
                "en passant target '{}' is on an illegal rank for the side to move",
                ep_str
            ));
        }
        Some(sq)
    };

    let halfmove = match half_str {
        Some(s) => s
            .parse::<u32>()
            .map_err(|_| format!("invalid halfmove clock '{}'", s))?,
        None => 0,
    };
    let fullmove = match full_str {
        Some(s) => {
            let v: u32 = s
                .parse()
                .map_err(|_| format!("invalid fullmove number '{}'", s))?;
            if v < 1 {
                return Err("fullmove number must be >= 1".into());
            }
            v
        }
        None => 1,
    };

    let mut pos = Position {
        board,
        side,
        castling,
        ep_target,
        halfmove,
        fullmove,
        king_sq,
        zobrist_key: 0,
    };
    pos.zobrist_key = recompute_zobrist(&pos);
    Ok(pos)
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
