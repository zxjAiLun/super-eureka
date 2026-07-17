//! UCI protocol loop (Phase 6 entry point).
//!
//! Implements the minimum the roadmap lists: `uci`, `isready`, `ucinewgame`,
//! `position startpos|fen ... moves ...`, `go depth N`, `stop`, `quit`.
//! A `perft` debug command is also accepted so the engine can self-verify
//! from a GUI or the command line.

use std::io::{self, BufRead, Write};

use crate::chess::types::*;
use crate::chess::position::Position;
use crate::chess::movegen::generate_pseudo_moves;
use crate::chess::fen;
use crate::engine::search;

pub fn run() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut pos = Position::startpos();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name ChessEngineDemo");
                println!("id author Rust-learner");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                pos = Position::startpos();
            }
            "position" => {
                pos = parse_position(&tokens);
            }
            "go" => {
                let depth = parse_go_depth(&tokens).unwrap_or(4);
                match search::search_best_move(&mut pos, depth) {
                    Some((m, _)) => println!("bestmove {}", move_to_uci(m)),
                    None => println!("bestmove 0000"),
                }
            }
            "stop" => { /* single-threaded, no async interrupt yet (Phase 7) */ }
            "perft" => {
                let depth: u32 = tokens
                    .get(1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(4);
                let n = pos.perft(depth);
                println!("perft({}) = {}", depth, n);
            }
            "quit" | "exit" => break,
            _ => { /* ignore unknown commands */ }
        }

        let _ = stdout.lock().flush();
    }
}

fn parse_position(tokens: &[&str]) -> Position {
    let mut idx = 1;
    let mut pos = if tokens.get(1) == Some(&"startpos") {
        idx = 2;
        Position::startpos()
    } else if tokens.get(1) == Some(&"fen") {
        let mut i = 2;
        let mut fen_parts: Vec<&str> = Vec::new();
        while i < tokens.len() && tokens[i] != "moves" {
            fen_parts.push(tokens[i]);
            i += 1;
        }
        idx = i;
        let fen_str = fen_parts.join(" ");
        fen::parse_fen(&fen_str).unwrap_or_else(|_| Position::startpos())
    } else {
        Position::startpos()
    };

    if tokens.get(idx) == Some(&"moves") {
        let mut i = idx + 1;
        while i < tokens.len() {
            if let Some(m) = find_move(&mut pos, tokens[i]) {
                pos.make_move(m);
            }
            i += 1;
        }
    }
    pos
}

/// Match a UCI move string to a generated (pseudo-legal) move so that
/// en-passant, castling, and promotion flags are reconstructed correctly.
fn find_move(pos: &mut Position, uci: &str) -> Option<Move> {
    if uci.len() < 4 {
        return None;
    }
    let from = parse_square(&uci[0..2]).ok()?;
    let to = parse_square(&uci[2..4]).ok()?;
    let promo = if uci.len() >= 5 {
        uci.chars().nth(4).and_then(PieceType::from_char)
    } else {
        None
    };
    let mut moves = Vec::new();
    generate_pseudo_moves(pos, &mut moves);
    moves
        .into_iter()
        .find(|m| m.from == from && m.to == to && m.promotion == promo)
}

fn parse_go_depth(tokens: &[&str]) -> Option<u32> {
    let mut i = 1;
    while i < tokens.len() {
        if tokens[i] == "depth" {
            if let Some(d) = tokens.get(i + 1).and_then(|s| s.parse::<u32>().ok()) {
                return Some(d);
            }
        }
        i += 1;
    }
    None
}
