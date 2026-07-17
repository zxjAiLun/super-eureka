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
use std::time::{Duration, Instant};

use crate::chess::fen;
use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::engine::search;
use crate::engine::search::SearchLimits;
use crate::engine::time::{self, TimeBudget, TimeInput};

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
fn spawn_search(
    pos: Position,
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    budget: TimeBudget,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let ctx = search::SearchContext::with_budget(stop.clone(), budget);
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
                // M1.2/M1.3: search on its own thread. Always stop and join
                // any previous search first, so a finished/aborted old thread
                // can never print a `bestmove` for the wrong position. The
                // `go` params are split into search limits (depth/nodes/
                // infinite) and a time budget (soft/hard deadlines) for the
                // side to move.
                stop_and_join(&mut active);
                let params = parse_go_params(&tokens);
                let (limits, budget) = build_limits_and_budget(&params, pos.side);
                let stop = Arc::new(AtomicBool::new(false));
                let handle = spawn_search(pos, limits, stop.clone(), budget);
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

/// Raw `go` parameters exactly as they appear on the UCI line. This is
/// deliberately separate from `SearchLimits`: UCI string parsing and the
/// search core must not be coupled, and the side-to-move selection of
/// `wtime`/`btime` happens here, not in the search.
#[derive(Default)]
struct GoParams {
    depth: Option<u32>,
    nodes: Option<u64>,
    movetime: Option<Duration>,
    wtime: Option<Duration>,
    btime: Option<Duration>,
    winc: Option<Duration>,
    binc: Option<Duration>,
    movestogo: Option<u32>,
    infinite: bool,
}

/// Parse a `go` command into raw `GoParams`. Unknown keys are skipped (per
/// the UCI spec, engines must ignore tokens they don't understand).
fn parse_go_params(tokens: &[&str]) -> GoParams {
    let mut p = GoParams::default();
    let mut i = 1;
    // Helper: read tokens[i+1] as milliseconds. Returns None if absent or
    // not a valid integer.
    let read_ms = |tokens: &[&str], i: usize| -> Option<Duration> {
        tokens
            .get(i + 1)
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
    };
    while i < tokens.len() {
        match tokens[i] {
            "infinite" => {
                p.infinite = true;
                i += 1;
            }
            "depth" => {
                p.depth = tokens.get(i + 1).and_then(|s| s.parse::<u32>().ok());
                i += 2;
            }
            "nodes" => {
                p.nodes = tokens.get(i + 1).and_then(|s| s.parse::<u64>().ok());
                i += 2;
            }
            "movestogo" => {
                p.movestogo = tokens.get(i + 1).and_then(|s| s.parse::<u32>().ok());
                i += 2;
            }
            "movetime" => {
                p.movetime = read_ms(tokens, i);
                i += 2;
            }
            "wtime" => {
                p.wtime = read_ms(tokens, i);
                i += 2;
            }
            "btime" => {
                p.btime = read_ms(tokens, i);
                i += 2;
            }
            "winc" => {
                p.winc = read_ms(tokens, i);
                i += 2;
            }
            "binc" => {
                p.binc = read_ms(tokens, i);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    p
}

/// Turn raw `go` params for the side to move into search limits + a time
/// budget. Picks `wtime`/`winc` or `btime`/`binc` based on `side`. A search
/// is treated as `infinite` (iterate until `stop`) when `go infinite` is
/// given or when no depth, nodes, or time limit was supplied at all.
fn build_limits_and_budget(params: &GoParams, side: Color) -> (SearchLimits, TimeBudget) {
    let time_input = TimeInput {
        movetime: params.movetime,
        remaining: if side == Color::White {
            params.wtime
        } else {
            params.btime
        },
        increment: if side == Color::White {
            params.winc
        } else {
            params.binc
        },
        movestogo: params.movestogo,
    };
    let now = Instant::now();
    let budget = time::compute_budget(&time_input, now);
    let has_time = budget.hard_deadline.is_some();
    let infinite =
        params.infinite || (params.depth.is_none() && params.nodes.is_none() && !has_time);
    let limits = SearchLimits {
        depth: params.depth,
        nodes: params.nodes,
        infinite,
    };
    (limits, budget)
}
