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
use crate::chess::game::GameState;
use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::*;
use crate::engine::search;
use crate::engine::search::SearchLimits;
use crate::engine::time::{self, TimeBudget, TimeInput};

/// Largest UCI time value (in milliseconds) we accept. UCI times arrive as
/// raw `u64` strings; a corrupted or malicious value such as
/// `go movetime 18446744073709551615` would otherwise build a `Duration`
/// large enough to make `Instant + Duration` panic on some platforms. We
/// clamp far below any `Instant` representable range: ~49 days is more than
/// any real game could ever need.
const MAX_UCI_TIME_MS: u64 = u32::MAX as u64;

/// Parse a UCI time token (milliseconds) into a `Duration`, clamping to
/// `MAX_UCI_TIME_MS`. Returns `None` if the token is missing or not a
/// non-negative base-10 integer.
fn parse_ms(s: &str) -> Option<Duration> {
    let ms = s.parse::<u64>().ok()?;
    Some(Duration::from_millis(ms.min(MAX_UCI_TIME_MS)))
}

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
        match a.handle.join() {
            Ok(()) => {}
            Err(_) => {
                // The search thread panicked. A GUI would otherwise wait
                // forever for a `bestmove` that never comes and get no clue
                // why. Report it and emit a safe fallback move so the protocol
                // stays complete.
                println!("info string search thread panicked");
                println!("bestmove 0000");
                let _ = std::io::stdout().flush();
            }
        }
    }
}

/// Spawn the search on a dedicated thread. The thread owns its own clone
/// of the `GameState` (handed in by `go`) and prints `bestmove`
/// (with a final flush) when it finishes, whether by completing or by
/// being stopped. The live game in the main loop is never touched:
/// `into_search_parts` moves the history out of the clone.
fn spawn_search(
    game: GameState,
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    budget: TimeBudget,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let ctx = search::SearchContext::with_budget(stop.clone(), budget);
        let (mut pos, game_history) = game.into_search_parts();
        match search::search_best_move_with_history(&mut pos, &game_history, &limits, &ctx) {
            Some(outcome) => println!("bestmove {}", move_to_uci(outcome.best_move)),
            None => println!("bestmove 0000"),
        }
        let _ = std::io::stdout().flush();
    })
}

pub fn run() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    // The live game state: current `Position` plus the real, chronological
    // UCI history of Zobrist keys. The search runs on its own thread
    // and receives a *clone* of this (via `into_search_parts`), so the
    // live `gs` is never mutated by a search.
    let mut gs = GameState::startpos();
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
                gs = GameState::startpos();
            }
            "position" => {
                // Stop first, then mutate. The search thread holds its own
                // clone of the game, so this is race-free; we still stop
                // first so a half-applied position never races a running
                // search's output.
                stop_and_join(&mut active);
                if let Err(e) = apply_position(&mut gs, &tokens) {
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
                let (limits, budget) =
                    build_limits_and_budget(&params, gs.position().side_to_move());
                let stop = Arc::new(AtomicBool::new(false));
                // Hand the thread a *clone* of the live game; the search
                // splits it via `into_search_parts` and never touches `gs`.
                let handle = spawn_search(gs.clone(), limits, stop.clone(), budget);
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
                // Perft only touches the current position; it must not mutate
                // the real `GameState` or its `key_history`.
                let mut p = *gs.position();
                let n = p.perft(depth);
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

/// Apply a `position` command to `gs` in place. On any error (bad FEN,
/// illegal history move, ...) the current game is left untouched and the
/// error is returned so the caller can report it. This replaces the old
/// silent `unwrap_or_else(startpos)` fallback that hid malformed input.
///
/// The new game is built *fresh* from the FEN/startpos root and then
/// advanced with `push_known_legal_move`, so its `key_history` starts at
/// the root key and appends exactly one key per applied move. A new
/// `position` command therefore *replaces* the old history (it does not
/// append to it); an illegal move discards the whole temporary game and
/// leaves `gs` byte-for-byte unchanged.
fn apply_position(gs: &mut GameState, tokens: &[&str]) -> Result<(), String> {
    let idx;
    let mut new_gs = if tokens.get(1) == Some(&"startpos") {
        idx = 2;
        GameState::startpos()
    } else if tokens.get(1) == Some(&"fen") {
        let mut i = 2;
        let mut fen_parts: Vec<&str> = Vec::new();
        while i < tokens.len() && tokens[i] != "moves" {
            fen_parts.push(tokens[i]);
            i += 1;
        }
        idx = i;
        let fen_str = fen_parts.join(" ");
        let pos = fen::parse_fen(&fen_str)?;
        GameState::from_position(pos)
    } else {
        return Err("position command needs 'startpos' or 'fen'".into());
    };

    if tokens.get(idx) == Some(&"moves") {
        let mut i = idx + 1;
        while i < tokens.len() {
            match find_move(new_gs.position(), tokens[i]) {
                Some(m) => {
                    // Committed legal move: advances both position and history.
                    new_gs.push_known_legal_move(m);
                }
                None => return Err(format!("invalid move {}", tokens[i])),
            }
            i += 1;
        }
    }

    *gs = new_gs;
    Ok(())
}

/// Match a UCI move string to a *strictly legal* move so that en-passant,
/// castling, and promotion flags are reconstructed correctly. We use legal
/// (not pseudo-legal) generation: a malformed history must never be allowed
/// to leave the king in check or otherwise reach an illegal position.
///
/// Takes a read-only `Position` (never a `&mut Position`): the caller owns
/// the `GameState` and its history; `find_move` only needs the legal-move
/// list, which it generates on a local copy of the position.
pub fn find_move(pos: &Position, uci: &str) -> Option<Move> {
    let mut probe = *pos;
    let moves = generate_legal_moves(&mut probe);
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
        tokens.get(i + 1).and_then(|s| parse_ms(s))
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
/// budget. Picks `wtime`/`winc` or `btime`/`binc` based on `side`.
///
/// `go infinite` is the highest-priority directive: it searches until `stop`
/// and *ignores* any clock / movetime / nodes also present on the line (a GUI
/// may send `go infinite wtime 1000 btime 1000` for analysis mode). "Infinite"
/// is encoded as `SearchLimits { depth: None, nodes: None }` plus a
/// `TimeBudget` with no deadlines — there is no separate flag, so the search
/// core has a single source of truth for "keep deepening" (the absence of a
/// depth cap, a node cap, and a hard deadline). A bare `go` (no limits at
/// all) falls through to the same infinite behaviour via `compute_budget`.
fn build_limits_and_budget(params: &GoParams, side: Color) -> (SearchLimits, TimeBudget) {
    // Highest priority: `go infinite` overrides every other time parameter.
    if params.infinite {
        return (
            SearchLimits {
                depth: None,
                nodes: None,
            },
            TimeBudget {
                soft_deadline: None,
                hard_deadline: None,
            },
        );
    }
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
    let limits = SearchLimits {
        depth: params.depth,
        nodes: params.nodes,
    };
    (limits, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::fen::{parse_fen, to_fen};
    use crate::chess::game::GameState;
    use crate::chess::types::START_FEN;
    use crate::chess::zobrist::recompute_zobrist;

    #[test]
    fn huge_millis_is_clamped_not_panicked() {
        // P1: `go movetime 18446744073709551615` must not panic when the
        // deadline is built; the value is clamped to MAX_UCI_TIME_MS.
        let d = parse_ms("18446744073709551615").expect("must parse");
        assert!(
            d <= Duration::from_millis(MAX_UCI_TIME_MS),
            "u64::MAX ms must be clamped"
        );
        // Building a deadline from the clamped value must not panic either.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            time::compute_budget(
                &TimeInput {
                    movetime: Some(d),
                    ..Default::default()
                },
                Instant::now(),
            )
        }));
        assert!(res.is_ok(), "deadline build must not panic");
    }

    #[test]
    fn huge_clock_is_clamped() {
        let tokens: Vec<&str> = "go wtime 18446744073709551615 btime 18446744073709551615"
            .split_whitespace()
            .collect();
        let p = parse_go_params(&tokens);
        assert!(
            p.wtime.unwrap() <= Duration::from_millis(MAX_UCI_TIME_MS),
            "wtime must be clamped"
        );
        assert!(
            p.btime.unwrap() <= Duration::from_millis(MAX_UCI_TIME_MS),
            "btime must be clamped"
        );
    }

    // ===== §16.7 GameState / UCI history =====
    //
    // `apply_position` builds a *fresh* GameState from the FEN/startpos
    // root and advances it with `push_known_legal_move`, so the history
    // starts at the root key and appends exactly one key per move. A new
    // `position` command replaces the old history (never appends); an
    // illegal move leaves the live game untouched.

    #[test]
    fn position_fen_with_moves_appends_from_root() {
        let root_fen = "4r1k1/4p3/8/8/8/8/4P3/4K3 w - - 0 1";
        let mut gs = GameState::startpos();
        let cmd = format!("position fen {} moves e2e4 e7e5", root_fen);
        let tokens: Vec<&str> = cmd.split_whitespace().collect();
        apply_position(&mut gs, &tokens).expect("apply must succeed");
        // history: FEN root + 2 applied moves.
        assert_eq!(gs.key_history().len(), 3, "root + 2 moves");
        // history[0] is the FEN root key (not the startpos key).
        assert_eq!(
            gs.key_history()[0],
            parse_fen(root_fen).unwrap().zobrist_key(),
            "history[0] is the FEN root key"
        );
        // history.last is the current (post-moves) position's key.
        assert_eq!(
            gs.key_history().last().copied(),
            Some(gs.current_key()),
            "history last == current key"
        );
        // current key differs from the root (two moves applied).
        assert_ne!(
            gs.current_key(),
            gs.key_history()[0],
            "current differs from root"
        );
        // current key matches a fresh recomputation of the live position.
        assert_eq!(gs.current_key(), recompute_zobrist(gs.position()));
    }

    #[test]
    fn new_position_replaces_old_history() {
        // Build a game with some history, then issue a *different* `position`
        // command; the old history must be discarded, not appended.
        let mut gs = GameState::startpos();
        let t1: Vec<&str> = "position startpos moves e2e4 e7e5"
            .split_whitespace()
            .collect();
        apply_position(&mut gs, &t1).unwrap();
        assert_eq!(gs.key_history().len(), 3);

        let t2: Vec<&str> = "position fen 4r1k1/4p3/8/8/8/8/4P3/4K3 w - - 0 1 moves e2e4"
            .split_whitespace()
            .collect();
        apply_position(&mut gs, &t2).unwrap();
        // Fresh history: FEN root + the one applied move -> len 2.
        assert_eq!(
            gs.key_history().len(),
            2,
            "new position starts fresh history"
        );
        assert_eq!(
            gs.key_history()[0],
            parse_fen("4r1k1/4p3/8/8/8/8/4P3/4K3 w - - 0 1")
                .unwrap()
                .zobrist_key(),
            "new history[0] is the FEN root key"
        );
    }

    #[test]
    fn ucinewgame_restores_startpos_single_history() {
        // Mimic `ucinewgame`: reset to startpos and verify a single-element
        // history whose key equals the startpos key.
        let mut gs = GameState::startpos();
        let t: Vec<&str> = "position startpos moves e2e4 e7e5"
            .split_whitespace()
            .collect();
        apply_position(&mut gs, &t).unwrap();
        assert!(gs.key_history().len() >= 3);

        gs = GameState::startpos();
        assert_eq!(gs.key_history().len(), 1, "ucinewgame -> single history");
        assert_eq!(
            gs.key_history()[0],
            gs.position().zobrist_key(),
            "ucinewgame history == startpos key"
        );
    }

    #[test]
    fn illegal_uci_move_leaves_game_untouched() {
        let mut gs = GameState::startpos();
        let tokens: Vec<&str> = "position startpos moves e2e4 z9z9"
            .split_whitespace()
            .collect();
        let err = apply_position(&mut gs, &tokens);
        assert!(err.is_err(), "illegal move must error");
        // Game untouched: still startpos, single history, same key/FEN.
        assert_eq!(gs.key_history().len(), 1, "history unchanged");
        assert_eq!(
            gs.key_history()[0],
            gs.position().zobrist_key(),
            "key unchanged"
        );
        assert_eq!(
            to_fen(gs.position()),
            to_fen(&parse_fen(START_FEN).unwrap()),
            "FEN unchanged"
        );
    }
}
