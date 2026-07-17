//! UCI protocol loop (Phase 6 entry point).
//!
//! Implements the minimum the roadmap lists: `uci`, `isready`, `ucinewgame`,
//! `position startpos|fen ... moves ...`, `go depth N`, `stop`, `quit`.
//! A `perft` debug command is also accepted so the engine can self-verify
//! from a GUI or the command line.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::chess::fen;
use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::engine::search;
use crate::engine::search::SearchLimits;

/// A search currently running on its own thread. `stop` is shared with the
/// thread's `SearchContext`, so flipping it aborts the search; `handle` lets
/// the main loop `join` the thread (and collect its `bestmove`) before it
/// starts a new search or mutates the position.
struct ActiveSearch {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

/// Stop any in-flight search and wait for its thread to finish.
///
/// The search thread prints its own `bestmove` (real or aborted) as it
/// unwinds, so we must `join` *before* touching `pos` or starting another
/// search — otherwise a stale `bestmove` from the old position could arrive
/// after the new one has already begun.
fn stop_and_join(active: &mut Option<ActiveSearch>) {
    if let Some(a) = active.take() {
        a.stop.store(true, Ordering::SeqCst);
        let _ = a.handle.join();
    }
}

/// Spawn the search on a dedicated thread. The thread owns its own clone of
/// the position and prints `bestmove` (with a final flush) when it finishes,
/// whether by completing or by being stopped.
fn spawn_search(pos: Position, limits: SearchLimits, stop: Arc<AtomicBool>) -> JoinHandle<()> {
    thread::spawn(move || {
        let ctx = search::SearchContext::new(stop.clone());
        let mut pos = pos;
        match search::search_best_move(&mut pos, &limits, &ctx) {
            Some(outcome) => println!("bestmove {}", move_to_uci(outcome.best_move)),
            None => println!("bestmove 0000"),
        }
        let _ = std::io::stdout().flush();
    })
}

pub fn run() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut pos = Position::startpos();
    // The active background search, if any. `None` while idle.
    let mut active: Option<ActiveSearch> = None;

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
            "isready" => {
                // Answer immediately, even while a search runs on its own
                // thread. We never block on the search here.
                println!("readyok");
            }
            "ucinewgame" => {
                // Stop any in-flight search before resetting the board so a
                // stale `bestmove` can't arrive for the old game.
                stop_and_join(&mut active);
                pos = Position::startpos();
            }
            "position" => {
                // Stop first, then mutate. The search thread holds its own
                // clone of `pos`, so this is race-free; we still stop first so
                // a half-applied position never races a running search's output.
                stop_and_join(&mut active);
                if let Err(e) = apply_position(&mut pos, &tokens) {
                    println!("info string {}", e);
                }
            }
            "go" => {
                // M1.2: search on its own thread. Always stop and join any
                // previous search first, so a finished/aborted old thread can
                // never print a `bestmove` for the wrong position.
                stop_and_join(&mut active);
                let limits = parse_go_limits(&tokens);
                let stop = Arc::new(AtomicBool::new(false));
                let handle = spawn_search(pos, limits, stop.clone());
                active = Some(ActiveSearch { stop, handle });
            }
            "stop" => {
                // Real stop: flip the flag and join. The thread prints
                // `bestmove` as it unwinds; we wait for that so the GUI
                // always receives a complete result.
                stop_and_join(&mut active);
            }
            "perft" => {
                let depth: u32 = tokens
                    .get(1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(4);
                let n = pos.perft(depth);
                println!("perft({}) = {}", depth, n);
            }
            "quit" | "exit" => {
                stop_and_join(&mut active);
                break;
            }
            _ => { /* ignore unknown commands */ }
        }

        let _ = stdout.lock().flush();
    }

    // stdin closed (EOF) without `quit`: don't leave a search thread
    // dangling.
    stop_and_join(&mut active);
}

/// Apply a `position` command to `pos` in place. On any error (bad FEN,
/// illegal history move, ...) the current position is left untouched and the
/// error is returned so the caller can report it. This replaces the old silent
/// `unwrap_or_else(startpos)` fallback that hid malformed input.
fn apply_position(pos: &mut Position, tokens: &[&str]) -> Result<(), String> {
    let idx;
    let mut new_pos = if tokens.get(1) == Some(&"startpos") {
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
        fen::parse_fen(&fen_str)?
    } else {
        return Err("position command needs 'startpos' or 'fen'".into());
    };

    if tokens.get(idx) == Some(&"moves") {
        let mut i = idx + 1;
        while i < tokens.len() {
            match find_move(&mut new_pos, tokens[i]) {
                Some(m) => {
                    new_pos.make_move(m);
                }
                None => return Err(format!("invalid move {}", tokens[i])),
            }
            i += 1;
        }
    }

    *pos = new_pos;
    Ok(())
}

/// Match a UCI move string to a *strictly legal* move so that en-passant,
/// castling, and promotion flags are reconstructed correctly. We use legal
/// (not pseudo-legal) generation: a malformed history must never be allowed
/// to leave the king in check or otherwise reach an illegal position.
pub fn find_move(pos: &mut Position, uci: &str) -> Option<Move> {
    let bytes = uci.as_bytes();
    // Reject anything that is not a clean 4- or 5-byte ASCII move. This
    // defends against (a) over-long strings like "e2e4garbage", (b) a junk
    // 5th byte being silently downgraded to "no promotion", and (c) UTF-8
    // input whose byte slice would otherwise land mid-character and panic.
    if !matches!(bytes.len(), 4 | 5) || !bytes.is_ascii() {
        return None;
    }
    let from = parse_square(std::str::from_utf8(&bytes[0..2]).unwrap()).ok()?;
    let to = parse_square(std::str::from_utf8(&bytes[2..4]).unwrap()).ok()?;
    let promo = if bytes.len() == 5 {
        // The promotion piece must be spelled out exactly; an unknown 5th byte
        // is rejected rather than tolerated.
        match bytes[4] {
            b'q' => Some(PieceType::Queen),
            b'r' => Some(PieceType::Rook),
            b'b' => Some(PieceType::Bishop),
            b'n' => Some(PieceType::Knight),
            _ => return None,
        }
    } else {
        None
    };
    let moves = generate_legal_moves(pos);
    moves
        .into_iter()
        .find(|m| m.from == from && m.to == to && m.promotion == promo)
}

/// Parse a `go` command into `SearchLimits`. M1.1 honours
/// `depth` and `nodes`; `movetime` / `wtime` / `btime` / `winc` /
/// `binc` / `infinite` are parsed by M1.3 time control. When no
/// `depth` is given we fall back to a fixed cap so a synchronous `go`
/// (no stop yet) can't search forever — true infinite time control
/// arrives in M1.3.
fn parse_go_limits(tokens: &[&str]) -> search::SearchLimits {
    let mut limits = search::SearchLimits::default();
    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "depth" => {
                if let Some(d) = tokens.get(i + 1).and_then(|s| s.parse::<u32>().ok()) {
                    limits.depth = Some(d);
                }
            }
            "nodes" => {
                if let Some(n) = tokens.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                    limits.nodes = Some(n);
                }
            }
            _ => {}
        }
        i += 1;
    }
    limits
}
