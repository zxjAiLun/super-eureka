//! UCI protocol loop (Phase 6 entry point).
//!
//! Implements the minimum the roadmap lists: `uci`, `isready`, `ucinewgame`,
//! `position startpos|fen ... moves ...`, `go depth N`, `stop`, `quit`.
//! A `perft` debug command is also accepted so the engine can self-verify
//! from a GUI or the command line.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
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
use crate::engine::tt::TranspositionTable;

/// Largest UCI time value (in milliseconds) we accept. UCI times arrive as
/// raw `u64` strings; a corrupted or malicious value such as
/// `go movetime 18446744073709551615` would otherwise build a `Duration`
/// large enough to make `Instant + Duration` panic on some platforms. We
/// clamp far below any `Instant` representable range: ~49 days is more than
/// any real game could ever need.
const MAX_UCI_TIME_MS: u64 = u32::MAX as u64;

/// Default transposition-table size (megabytes) allocated at engine startup.
const DEFAULT_HASH_MB: usize = 16;
/// Smallest legal `Hash` value. A requested `0` is clamped up to this.
const MIN_HASH_MB: usize = 1;
/// Largest legal `Hash` value. Anything larger is clamped down to this.
const MAX_HASH_MB: usize = 1024;

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
///
/// The persistent transposition table is held for the *entire* search run:
/// the thread takes the only mutable guard up front (recovering from a
/// poisoned mutex if needed) and releases it only when the thread exits.
/// No per-node locking — `search_best_move_with_history_and_tt` reads and
/// writes the table through the single `&mut` guard.
fn spawn_search(
    game: GameState,
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    budget: TimeBudget,
    tt: Arc<Mutex<TranspositionTable>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let ctx = search::SearchContext::with_budget(stop.clone(), budget);
        let (mut pos, game_history) = game.into_search_parts();
        // Acquire the (only) mutable TT guard and keep it for the whole run.
        let mut guard = lock_tt_recover(&tt);
        match search::search_best_move_with_history_and_tt(
            &mut pos,
            &game_history,
            &limits,
            &ctx,
            &mut guard,
        ) {
            Some(outcome) => println!("bestmove {}", move_to_uci(outcome.best_move)),
            None => println!("bestmove 0000"),
        }
        let _ = std::io::stdout().flush();
        // `guard` drops here, releasing the TT mutex for the next command.
    })
}

/// Acquire the TT mutex, recovering from poison without panicking.
///
/// On a clean lock we return the guard as-is. If the mutex was poisoned
/// (a previous owner panicked while holding it), we take ownership of the
/// poisoned guard via `into_inner`, clear the table in place (preserving
/// the configured `size_mb` and capacity — no reallocation), emit a
/// diagnostic, and return the recovered guard. This is the single place
/// the engine ever touches a poisoned TT; `isready` must never call it.
fn lock_tt_recover(tt: &Mutex<TranspositionTable>) -> MutexGuard<'_, TranspositionTable> {
    match tt.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            guard.clear();
            let _ = writeln!(io::stdout(), "info string TT mutex poisoned; recovered");
            let _ = io::stdout().flush();
            guard
        }
    }
}

/// Write the `uci` handshake to `out`. The `startup_tt_failed` flag
/// appends a diagnostic (after the `Hash` option line, before `uciok`)
/// telling the GUI that the default table could not be allocated and TT is
/// disabled. This helper is pure with respect to any global state, so it can
/// be exercised against an in-memory buffer in tests. Callers that need the
/// "report the failure at most once" semantics should use
/// `write_pending_uci_handshake` instead.
fn write_uci_handshake<W: Write>(out: &mut W, startup_tt_failed: bool) -> io::Result<()> {
    writeln!(out, "id name ChessEngineDemo")?;
    writeln!(out, "id author Rust-learner")?;
    writeln!(
        out,
        "option name Hash type spin default {} min {} max {}",
        DEFAULT_HASH_MB, MIN_HASH_MB, MAX_HASH_MB
    )?;
    if startup_tt_failed {
        writeln!(
            out,
            "info string unable to allocate default Hash table; TT disabled"
        )?;
    }
    writeln!(out, "uciok")?;
    Ok(())
}

/// Consumable variant used by `run()`. It takes a mutable `startup_tt_notice_pending`
/// flag, emits the startup diagnostic at most ONCE (the first handshake after a
/// failed allocation), and clears the flag so later `uci` handshakes stay silent
/// even if the table never recovers. A successful `setoption Hash` also clears
/// the flag (see `run()`), so the notice never survives a recovered table.
fn write_pending_uci_handshake<W: Write>(
    out: &mut W,
    startup_tt_notice_pending: &mut bool,
) -> io::Result<()> {
    let report_failure = std::mem::take(startup_tt_notice_pending);
    write_uci_handshake(out, report_failure)
}

/// Write the `isready` reply. Deliberately takes **no** TT parameter: it
/// must never block on, lock, or wait for the search/table. The structural
/// guarantee that `isready` cannot touch the TT is that this helper has no
/// way to reach it.
fn write_readyok<W: Write>(out: &mut W) -> io::Result<()> {
    writeln!(out, "readyok")?;
    Ok(())
}

/// Result of parsing a `setoption` line for the `Hash` option.
#[derive(Debug, PartialEq, Eq)]
enum HashOptionCommand {
    /// Not a recognized `Hash` setoption (unknown option, or malformed).
    Unknown,
    /// Recognized `Hash` but the value is missing/invalid; the table must
    /// be left untouched.
    Invalid,
    /// Recognized `Hash` with a clamped megabyte value to resize to.
    Resize(usize),
}

/// Outcome of `handle_setoption`, surfaced to `run()` so it can clear the
/// one-shot startup-failure notice once the table is successfully (re)enabled.
/// Only `Resized` clears that pending notice; every other outcome leaves it
/// untouched (including a later resize *failure*, which already prints its own
/// immediate diagnostic and must not re-open a long-gone startup notice).
#[derive(Debug, PartialEq, Eq)]
enum SetoptionOutcome {
    Ignored,
    Invalid,
    Resized,
    ResizeFailed,
}

/// Parse a `setoption name Hash value N` line.
///
/// * The option `name` marker, the `value` marker, and the option name
///   (`Hash`) are all matched case-insensitively.
/// * The full option name is collected between `name` and `value`, so
///   multi-word names are handled; only a lowercased `hash` triggers.
/// * Unknown options yield `Unknown` (silently ignored by the handler).
/// * A recognized `Hash` without a value yields `Invalid`.
/// * The value must be non-empty and consist solely of ASCII digits; signed
///   values and embedded non-digits are `Invalid`. An overflowing all-digit
///   string is NOT `Invalid`: it is parsed as a `u64` (which saturates to
///   `Err` on overflow) and treated as a huge positive integer, then clamped
///   to `MAX_HASH_MB`. The parser never panics on an over-long digit string.
/// * The accepted magnitude is clamped: `0 -> 1`, `1..=1024` kept,
///   `>1024` clamped to `1024`.
fn parse_hash_setoption(tokens: &[&str]) -> HashOptionCommand {
    if tokens.first() != Some(&"setoption") {
        return HashOptionCommand::Unknown;
    }

    let mut name_start: Option<usize> = None;
    let mut name_end: Option<usize> = None;
    let mut value: Option<&str> = None;
    let mut i = 1;
    while i < tokens.len() {
        let tok = tokens[i];
        if tok.eq_ignore_ascii_case("name") {
            // Collect every token after `name` up to (but not including)
            // `value` as the option name.
            name_start = Some(i + 1);
            let mut j = i + 1;
            while j < tokens.len() && !tokens[j].eq_ignore_ascii_case("value") {
                j += 1;
            }
            name_end = Some(j);
            i = j;
        } else if tok.eq_ignore_ascii_case("value") {
            value = tokens.get(i + 1).copied();
            i = tokens.len();
        } else {
            i += 1;
        }
    }

    // Recover the option name (lowercased) between name..value.
    let opt_name = match (name_start, name_end) {
        (Some(s), Some(e)) if e > s => tokens[s..e]
            .iter()
            .map(|t| t.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" "),
        _ => return HashOptionCommand::Unknown,
    };
    if opt_name != "hash" {
        return HashOptionCommand::Unknown;
    }

    // `Hash` recognized. A missing value is invalid.
    let raw = match value {
        Some(v) => v,
        None => return HashOptionCommand::Invalid,
    };
    // Empty or any non-digit (signs, letters, etc.) is invalid.
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return HashOptionCommand::Invalid;
    }
    // `raw` is now guaranteed all-ASCII-digits. `u64` parsing returns
    // `Err` on overflow; an all-digit overflow is treated as a huge positive
    // value and clamped to `MAX_HASH_MB` below (never panics).
    let v: u64 = match raw.parse::<u64>() {
        Ok(v) => v,
        Err(_) => return HashOptionCommand::Resize(MAX_HASH_MB),
    };
    let clamped = if v == 0 {
        MIN_HASH_MB
    } else if v >= MAX_HASH_MB as u64 {
        MAX_HASH_MB
    } else {
        usize::try_from(v).unwrap_or(MAX_HASH_MB)
    };
    HashOptionCommand::Resize(clamped)
}

/// Handle a `setoption` line. The handler owns the lifecycle rules:
///
/// * `Unknown` — silently ignored; the table and any running search are
///   left untouched.
/// * `Invalid` (recognized `Hash` with a bad value) — the active search
///   is stopped/joined first (`position`/`go` discipline), the table is
///   left as-is, and a single diagnostic is emitted.
/// * `Resize(n)` — the active search is stopped/joined first, then the
///   table is resized behind `lock_tt_recover`. A successful `resize_mb`
///   already yields an empty table, so we do **not** clear again. On
///   failure the old size/capacity/entries are preserved and a diagnostic
///   is emitted.
fn handle_setoption(
    tokens: &[&str],
    active: &mut Option<ActiveSearch>,
    tt: &Arc<Mutex<TranspositionTable>>,
) -> SetoptionOutcome {
    match parse_hash_setoption(tokens) {
        HashOptionCommand::Unknown => {
            // Silenty ignore unrecognized options; never touch the table or
            // the running search.
            SetoptionOutcome::Ignored
        }
        HashOptionCommand::Invalid => {
            stop_and_join(active);
            println!("info string invalid Hash value");
            let _ = io::stdout().flush();
            SetoptionOutcome::Invalid
        }
        HashOptionCommand::Resize(n) => {
            stop_and_join(active);
            let mut guard = lock_tt_recover(tt);
            match guard.resize_mb(n) {
                Ok(()) => {
                    // Success: resize_mb already publishes an empty table.
                    // No extra output, no redundant clear.
                    SetoptionOutcome::Resized
                }
                Err(_) => {
                    println!("info string unable to resize Hash table");
                    let _ = io::stdout().flush();
                    SetoptionOutcome::ResizeFailed
                }
            }
        }
    }
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

    // Persistent transposition table, shared across every `go` and owned
    // independently of `gs` and `active`. It is created once here, survives
    // `position` (which never clears it), is cleared (capacity-preserving)
    // by `ucinewgame`, and is resized (and cleared) by `setoption Hash`.
    // Each `go` clones the `Arc` and hands the *same* underlying mutex to
    // its search thread; because every table-mutating command stops/joins
    // the active search first, at most one search thread holds the mutable
    // guard at any time. A startup allocation failure must not panic: we
    // fall back to a disabled table and remember the failure as a *pending
    // one-shot* notice. `uci` reports it at most once (the first handshake
    // after the failure), then clears the flag; a later successful
    // `setoption Hash` also clears it because the table is then live. This
    // prevents the "TT disabled" notice from repeating across multiple
    // `uci` lines or from lying after recovery.
    let (tt, mut startup_tt_notice_pending) = match TranspositionTable::new_mb(DEFAULT_HASH_MB) {
        Ok(t) => (Arc::new(Mutex::new(t)), false),
        Err(_) => (Arc::new(Mutex::new(TranspositionTable::disabled())), true),
    };

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
                // Handshake: reports the `Hash` option and (if startup
                // allocation failed and the notice is still pending) a one-shot
                // diagnostic before `uciok`. The consumable helper clears the
                // pending flag, so repeated `uci` lines never repeat it.
                let _ =
                    write_pending_uci_handshake(&mut io::stdout(), &mut startup_tt_notice_pending);
            }
            "isready" => {
                // Answer immediately, even while a search runs on its own
                // thread. We never block on, lock, or wait for the search
                // or the TT here.
                let _ = write_readyok(&mut io::stdout());
            }
            "ucinewgame" => {
                // Stop any in-flight search before resetting the board so a
                // stale `bestmove` can't arrive for the old game. Reset the
                // board, then clear the TT (preserving its MB/capacity).
                stop_and_join(&mut active);
                ucinewgame_reset(&mut gs, &tt);
            }
            "position" => {
                // Stop first, then mutate. The search thread holds its own
                // clone of the game, so this is race-free; we still stop
                // first so a half-applied position never races a running
                // search's output. The persistent TT is deliberately NOT
                // cleared or resized by `position`.
                stop_and_join(&mut active);
                if let Err(e) = apply_position(&mut gs, &tokens) {
                    println!("info string {}", e);
                }
            }
            "setoption" => {
                // Route through the lifecycle-aware handler. Unknown options
                // are ignored; `Hash` stops/joins any search, then resizes.
                // A successful (re)enable also clears any unsent startup
                // notice: the table is now live, so "TT disabled" would lie.
                let outcome = handle_setoption(&tokens, &mut active, &tt);
                if outcome == SetoptionOutcome::Resized {
                    startup_tt_notice_pending = false;
                }
            }
            "go" => {
                // M1.2/M1.3: search on its own thread. Always stop and join
                // any previous search first, so a finished/aborted old thread
                // can never print a `bestmove` for the wrong position. The
                // `go` params are split into search limits (depth/nodes/
                // infinite) and a time budget (soft/hard deadlines) for the
                // side to move. The persistent TT (cloned `Arc`) is reused
                // across `go` calls — never reallocated per search.
                stop_and_join(&mut active);
                let params = parse_go_params(&tokens);
                let (limits, budget) =
                    build_limits_and_budget(&params, gs.position().side_to_move());
                let stop = Arc::new(AtomicBool::new(false));
                // Hand the thread a *clone* of the live game and the shared
                // TT; the search splits `gs` via `into_search_parts` and
                // never touches the live `gs`. It holds the TT guard for the
                // whole run.
                let handle = spawn_search(gs.clone(), limits, stop.clone(), budget, tt.clone());
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

/// `ucinewgame` lifecycle: reset the live game to the start position and
/// clear the persistent TT (preserving its configured MB and capacity).
/// Called after `stop_and_join` so no search thread is mid-run. It does
/// not allocate a new table, and never resizes — only zeroes the slots.
fn ucinewgame_reset(gs: &mut GameState, tt: &Arc<Mutex<TranspositionTable>>) {
    *gs = GameState::startpos();
    let mut guard = lock_tt_recover(tt);
    guard.clear();
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
    // `TranspositionTable`, `SearchLimits`, `TimeBudget`, `Mutex`, `Arc`,
    // `AtomicBool`, `Ordering`, `thread`, `Duration`, `Write`, `io` are all
    // reachable via `super::*` (the parent module's own `use` imports). Only
    // the entry/key types live in `tt` and were not imported by the parent.
    use crate::engine::tt::{Bound, TTEntry, TtKey};

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

    // ===== M3.2 Phase-3: UCI `Hash` option & TT lifecycle =====

    #[test]
    fn handshake_reports_hash_option_and_order() {
        // The `uci` handshake must advertise the `Hash` spin option with
        // exact default/min/max, and it must appear before `uciok`.
        let mut buf: Vec<u8> = Vec::new();
        write_uci_handshake(&mut buf, false).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        assert!(lines.contains(&"id name ChessEngineDemo"));
        assert!(lines.contains(&"id author Rust-learner"));

        let opt_idx = lines
            .iter()
            .position(|l| l.starts_with("option name Hash"))
            .expect("Hash option present");
        let opt = lines[opt_idx];
        assert!(opt.contains("type spin"), "option is a spin: {}", opt);
        assert!(opt.contains("default 16"), "default 16: {}", opt);
        assert!(opt.contains("min 1"), "min 1: {}", opt);
        assert!(opt.contains("max 1024"), "max 1024: {}", opt);

        let uciok_idx = lines
            .iter()
            .position(|l| *l == "uciok")
            .expect("uciok present");
        assert!(opt_idx < uciok_idx, "Hash option before uciok");
        assert_eq!(lines[uciok_idx], "uciok");
    }

    #[test]
    fn handshake_reports_startup_failure_before_uciok() {
        // When the default table could not be allocated, the diagnostic is
        // emitted after the option line but before `uciok`.
        let mut buf: Vec<u8> = Vec::new();
        write_uci_handshake(&mut buf, true).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = text.lines().collect();

        let opt_idx = lines
            .iter()
            .position(|l| l.starts_with("option name Hash"))
            .expect("Hash option present");
        let diag_idx = lines
            .iter()
            .position(|l| l.contains("unable to allocate default Hash table"))
            .expect("startup-failure diagnostic present");
        let uciok_idx = lines
            .iter()
            .position(|l| *l == "uciok")
            .expect("uciok present");

        assert!(opt_idx < diag_idx, "diagnostic after option line");
        assert!(diag_idx < uciok_idx, "diagnostic before uciok");
    }

    #[test]
    fn parse_hash_setoption_matrix() {
        // Exercise the full parse/clamp matrix. Values are the *value* tokens
        // of a `setoption name Hash value <v>` line.
        let cases: Vec<(Vec<&str>, HashOptionCommand)> = vec![
            (
                vec!["setoption", "name", "Hash", "value", "1"],
                HashOptionCommand::Resize(1),
            ),
            (
                vec!["setoption", "name", "hash", "value", "1"],
                HashOptionCommand::Resize(1),
            ),
            (
                vec!["setoption", "name", "HASH", "value", "1"],
                HashOptionCommand::Resize(1),
            ),
            (
                vec!["setoption", "name", "Hash", "value", "16"],
                HashOptionCommand::Resize(16),
            ),
            // `0` clamps up to the minimum (1).
            (
                vec!["setoption", "name", "Hash", "value", "0"],
                HashOptionCommand::Resize(1),
            ),
            (
                vec!["setoption", "name", "Hash", "value", "1024"],
                HashOptionCommand::Resize(1024),
            ),
            // Over-max clamps down to 1024 (parser only, no real resize).
            (
                vec!["setoption", "name", "Hash", "value", "2048"],
                HashOptionCommand::Resize(1024),
            ),
            // A very long all-digit string overflows u64 -> huge -> 1024.
            (
                vec![
                    "setoption",
                    "name",
                    "Hash",
                    "value",
                    "999999999999999999999999999999",
                ],
                HashOptionCommand::Resize(1024),
            ),
            // Missing value -> Invalid (recognized Hash, bad value).
            (
                vec!["setoption", "name", "Hash", "value"],
                HashOptionCommand::Invalid,
            ),
            // Empty value token -> Invalid.
            (
                vec!["setoption", "name", "Hash", "value", ""],
                HashOptionCommand::Invalid,
            ),
            // Non-digit value -> Invalid.
            (
                vec!["setoption", "name", "Hash", "value", "abc"],
                HashOptionCommand::Invalid,
            ),
            // Signed value -> Invalid.
            (
                vec!["setoption", "name", "Hash", "value", "-1"],
                HashOptionCommand::Invalid,
            ),
            // Unknown option -> Unknown (silently ignored by handler).
            (
                vec!["setoption", "name", "Foo", "value", "1"],
                HashOptionCommand::Unknown,
            ),
            // Multi-word unknown option -> Unknown.
            (
                vec!["setoption", "name", "Foo", "Bar", "value", "1"],
                HashOptionCommand::Unknown,
            ),
            // Not a setoption line at all -> Unknown.
            (vec!["uci"], HashOptionCommand::Unknown),
        ];
        for (tokens, expected) in cases {
            assert_eq!(
                parse_hash_setoption(&tokens),
                expected,
                "tokens = {:?}",
                tokens
            );
        }
    }

    #[test]
    fn parse_hash_setoption_marker_case_insensitive() {
        // The `name`, `value`, and option-name markers are all matched
        // case-insensitively, so mixed-case setoption lines still parse.
        let cases: Vec<(Vec<&str>, HashOptionCommand)> = vec![
            (
                vec!["setoption", "NAME", "Hash", "VALUE", "16"],
                HashOptionCommand::Resize(16),
            ),
            (
                vec!["setoption", "Name", "HASH", "Value", "16"],
                HashOptionCommand::Resize(16),
            ),
            (
                vec!["setoption", "nAmE", "hash", "vAlUe", "16"],
                HashOptionCommand::Resize(16),
            ),
            // Lowercase baseline still works.
            (
                vec!["setoption", "name", "Hash", "value", "16"],
                HashOptionCommand::Resize(16),
            ),
        ];
        for (tokens, expected) in cases {
            assert_eq!(
                parse_hash_setoption(&tokens),
                expected,
                "tokens = {:?}",
                tokens
            );
        }
    }

    #[test]
    fn pending_uci_handshake_reports_once() {
        // The startup-failure notice is a one-shot: it appears on the first
        // handshake, then the pending flag is cleared so later handshakes
        // stay silent even though the table never recovered.
        let mut pending = true;
        let mut first: Vec<u8> = Vec::new();
        write_pending_uci_handshake(&mut first, &mut pending).unwrap();
        assert!(!pending, "pending cleared after first handshake");
        let first_text = String::from_utf8(first).unwrap();
        assert!(
            first_text.contains("unable to allocate default Hash table"),
            "first handshake reports startup failure"
        );

        let mut second: Vec<u8> = Vec::new();
        write_pending_uci_handshake(&mut second, &mut pending).unwrap();
        let second_text = String::from_utf8(second).unwrap();
        assert!(
            !second_text.contains("unable to allocate default Hash table"),
            "second handshake stays silent"
        );
    }

    #[test]
    fn resize_success_clears_pending_notice() {
        // Simulate the real startup path: default allocation failed, so the
        // persistent table fell back to `disabled()`. A successful
        // `setoption name Hash value 1` then (re)enables it and also clears
        // a still-pending startup notice, so a subsequent handshake must NOT
        // claim "TT disabled".
        let tt = Arc::new(Mutex::new(TranspositionTable::disabled()));
        let mut pending = true;
        let tokens: Vec<&str> = "setoption name Hash value 1".split_whitespace().collect();
        let mut active: Option<ActiveSearch> = None;
        let outcome = handle_setoption(&tokens, &mut active, &tt);
        assert_eq!(outcome, SetoptionOutcome::Resized, "resize succeeded");
        assert!(active.is_none(), "search stopped/joined before resize");
        assert_eq!(tt.lock().unwrap().size_mb(), 1, "table now 1 MB");
        assert!(
            tt.lock().unwrap().capacity_entries() > 0,
            "recovered table has entries"
        );
        // `run()` clears the pending notice on `Resized`.
        if outcome == SetoptionOutcome::Resized {
            pending = false;
        }
        assert!(!pending, "Resized clears the pending startup notice");

        let mut buf: Vec<u8> = Vec::new();
        write_pending_uci_handshake(&mut buf, &mut pending).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(
            !text.contains("unable to allocate default Hash table"),
            "recovered table must not report TT disabled"
        );
    }

    #[test]
    fn invalid_hash_keeps_pending_notice() {
        // A recognized-but-invalid Hash must NOT clear a pending startup
        // notice: the table is left untouched and the notice stays for a
        // future handshake. (Structurally identical to a later resize
        // *failure*: only `Resized` clears the flag, so `ResizeFailed`
        // also preserves it by construction.)
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(1).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(123u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        let mut pending = true;
        let tokens: Vec<&str> = "setoption name Hash value abc".split_whitespace().collect();
        let mut active: Option<ActiveSearch> = None;
        let outcome = handle_setoption(&tokens, &mut active, &tt);
        assert_eq!(outcome, SetoptionOutcome::Invalid, "invalid value");
        assert!(active.is_none(), "invalid Hash stops search");
        assert!(pending, "invalid Hash must not clear pending notice");
        // The pre-stored entry and size survive.
        let g = tt.lock().unwrap();
        assert!(
            g.probe(TtKey::new(123u64, 0, 0)).is_some(),
            "entry preserved on invalid Hash"
        );
        assert_eq!(g.size_mb(), 1, "size unchanged on invalid Hash");
        drop(g);

        // A later handshake must still carry the (still-pending) notice, and
        // then consume it.
        let mut buf: Vec<u8> = Vec::new();
        write_pending_uci_handshake(&mut buf, &mut pending).unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(
            text.contains("unable to allocate default Hash table"),
            "notice still pending after invalid Hash"
        );
        assert!(!pending, "notice consumed by the handshake");
    }

    #[test]
    fn resize_1mb_to_2mb_success_clears_table() {
        // A successful resize empties the table and publishes the new MB,
        // preserving a positive capacity. Uses only a safe small capacity.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(1).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(3u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        // No active search: stop_and_join is a no-op; resize proceeds.
        let mut active: Option<ActiveSearch> = None;
        let tokens: Vec<&str> = "setoption name Hash value 2".split_whitespace().collect();
        handle_setoption(&tokens, &mut active, &tt);

        let g = tt.lock().unwrap();
        assert_eq!(g.size_mb(), 2, "resized to 2 MB");
        assert!(g.capacity_entries() > 0, "capacity preserved");
        assert!(
            g.probe(TtKey::new(3u64, 0, 0)).is_none(),
            "old entry cleared after resize"
        );
    }

    #[test]
    fn invalid_hash_parser_does_not_resize_table() {
        // When the parser rejects the value, `handle_setoption` must not call
        // `resize_mb` at all; the original entry, size, and capacity stay.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(1).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(7u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        let mut active: Option<ActiveSearch> = None;
        let tokens: Vec<&str> = "setoption name Hash value abc".split_whitespace().collect();
        assert_eq!(
            parse_hash_setoption(&tokens),
            HashOptionCommand::Invalid,
            "parser rejects non-numeric value"
        );
        handle_setoption(&tokens, &mut active, &tt);

        let g = tt.lock().unwrap();
        assert!(
            g.probe(TtKey::new(7u64, 0, 0)).is_some(),
            "entry preserved on invalid value"
        );
        assert_eq!(g.size_mb(), 1, "size unchanged on invalid value");
    }

    #[test]
    fn resize_failure_preserves_table() {
        // A failing `resize_mb` (here `0`, which is an invalid size and does
        // NOT allocate) must leave size, capacity, and every entry intact.
        // The UCI handler relies on this for its failure branch; we exercise
        // the underlying guarantee directly (the clamped parser can only
        // reach [1,1024], so a real failure is driven here without a 1GB
        // allocation).
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(1).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(11u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        let old_cap = tt.lock().unwrap().capacity_entries();
        {
            let mut g = lock_tt_recover(&tt);
            let err = g.resize_mb(0);
            assert!(err.is_err(), "resize 0 fails");
        }
        let g = tt.lock().unwrap();
        assert_eq!(g.size_mb(), 1, "size preserved on failure");
        assert_eq!(
            g.capacity_entries(),
            old_cap,
            "capacity preserved on failure"
        );
        assert!(
            g.probe(TtKey::new(11u64, 0, 0)).is_some(),
            "entries preserved on failure"
        );
    }

    #[test]
    fn handle_setoption_resize_stops_active_search_first() {
        // `setoption name Hash value N` must stop/join any running search
        // before resizing. We build a synthetic search thread that only exits
        // once its stop flag is set; `handle_setoption` sets it via
        // `stop_and_join`, then resizes.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(1).unwrap()));
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = thread::spawn(move || {
            while !stop_thread.load(Ordering::SeqCst) {
                thread::yield_now();
            }
        });
        let mut active = Some(ActiveSearch { stop, handle });

        let tokens: Vec<&str> = "setoption name Hash value 2".split_whitespace().collect();
        handle_setoption(&tokens, &mut active, &tt);

        assert!(active.is_none(), "search stopped and joined before resize");
        assert_eq!(tt.lock().unwrap().size_mb(), 2, "resized to 2 MB");
    }

    #[test]
    fn handle_setoption_invalid_stops_search_first() {
        // Even a recognized-but-invalid Hash must stop/join the active search;
        // the table (entry + size) stays untouched.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(1).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(999u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = thread::spawn(move || {
            while !stop_thread.load(Ordering::SeqCst) {
                thread::yield_now();
            }
        });
        let mut active = Some(ActiveSearch { stop, handle });

        let tokens: Vec<&str> = "setoption name Hash value abc".split_whitespace().collect();
        handle_setoption(&tokens, &mut active, &tt);

        assert!(active.is_none(), "invalid Hash stops search");
        let g = tt.lock().unwrap();
        assert!(
            g.probe(TtKey::new(999u64, 0, 0)).is_some(),
            "entry preserved on invalid Hash"
        );
        assert_eq!(g.size_mb(), 1, "size unchanged on invalid Hash");
    }

    #[test]
    fn ucinewgame_clears_tt_preserves_capacity() {
        // `ucinewgame` clears the persistent TT but keeps its MB/capacity,
        // and resets the game to a single-element startpos history.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(2).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(77u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        let mut gs = GameState::startpos();
        let t: Vec<&str> = "position startpos moves e2e4 e7e5"
            .split_whitespace()
            .collect();
        apply_position(&mut gs, &t).unwrap();
        assert!(gs.key_history().len() >= 3);

        let cap = tt.lock().unwrap().capacity_entries();
        let sz = tt.lock().unwrap().size_mb();
        ucinewgame_reset(&mut gs, &tt);

        assert!(
            tt.lock().unwrap().probe(TtKey::new(77u64, 0, 0)).is_none(),
            "entry cleared by ucinewgame"
        );
        assert_eq!(tt.lock().unwrap().size_mb(), sz, "size_mb preserved");
        assert_eq!(
            tt.lock().unwrap().capacity_entries(),
            cap,
            "capacity preserved"
        );
        assert_eq!(gs.key_history().len(), 1, "ucinewgame -> single history");
        assert_eq!(gs.key_history()[0], gs.position().zobrist_key());
    }

    #[test]
    fn position_preserves_tt() {
        // `position` updates the game but must never clear or resize the TT;
        // a pre-stored entry, size, and capacity survive.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(2).unwrap()));
        {
            let mut g = tt.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(55u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
        }
        let mut gs = GameState::startpos();
        let cap_before = tt.lock().unwrap().capacity_entries();

        let tokens: Vec<&str> = "position startpos moves e2e4 e7e5"
            .split_whitespace()
            .collect();
        apply_position(&mut gs, &tokens).unwrap();

        // Game changed.
        assert_ne!(gs.current_key(), GameState::startpos().current_key());
        // TT untouched.
        let g = tt.lock().unwrap();
        assert!(
            g.probe(TtKey::new(55u64, 0, 0)).is_some(),
            "TT entry preserved across position"
        );
        assert_eq!(g.capacity_entries(), cap_before, "capacity unchanged");
        assert_eq!(g.size_mb(), 2);
    }

    #[test]
    fn tt_poison_recovery_clears_and_preserves_config() {
        // A thread that panics while holding the TT mutex poisons it. After
        // joining, `lock_tt_recover` must recover (clear + keep config) and
        // not panic; subsequent store/probe behave normally.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(2).unwrap()));
        let tt2 = tt.clone();
        let holder = thread::spawn(move || {
            let mut g = tt2.lock().unwrap();
            g.store(TTEntry {
                key: TtKey::new(123u64, 0, 0),
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
            panic!("simulated panic while holding TT");
        });
        let _ = holder.join(); // ignore the panic

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = lock_tt_recover(&tt);
            assert_eq!(g.size_mb(), 2, "size_mb preserved");
            assert!(g.capacity_entries() > 0, "capacity preserved");
            assert!(
                g.probe(TtKey::new(123u64, 0, 0)).is_none(),
                "entry cleared after poison recovery"
            );
            // Subsequent store/probe work.
            let k2 = TtKey::new(456u64, 0, 0);
            g.store(TTEntry {
                key: k2,
                depth: 1,
                score: 0,
                bound: Bound::Exact,
                best_move: None,
            });
            assert!(g.probe(k2).is_some(), "store/probe normal after recovery");
        }));
        assert!(res.is_ok(), "lock_tt_recover must not panic");
    }

    #[test]
    fn isready_writes_readyok_without_locking_tt() {
        // `isready` writes `readyok` immediately and must not touch the TT
        // mutex at all. We prove this structurally: another thread holds the
        // TT lock, yet `write_readyok` returns at once with exact output.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(2).unwrap()));
        let tt2 = tt.clone();
        let acquired = Arc::new(AtomicBool::new(false));
        let acquired2 = acquired.clone();
        let holder = thread::spawn(move || {
            let _g = tt2.lock().unwrap();
            acquired2.store(true, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(200));
        });
        // Wait until the holder owns the lock.
        while !acquired.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        let mut buf: Vec<u8> = Vec::new();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            write_readyok(&mut buf).unwrap();
        }));
        assert!(res.is_ok(), "write_readyok must not panic");
        assert_eq!(buf, b"readyok\n", "exact readyok output");
        let _ = holder.join();
    }

    #[test]
    fn search_thread_smoke_uses_shared_tt() {
        // One real (shallow) search on the shared TT thread: it joins
        // normally, prints a bestmove (not a panic), and the table Arc is
        // still lockable afterward with its configuration intact. Uses a safe
        // 2 MB table — never resizes to 1024 MB.
        let tt = Arc::new(Mutex::new(TranspositionTable::new_mb(2).unwrap()));
        let game = GameState::startpos();
        let limits = SearchLimits {
            depth: Some(2),
            nodes: None,
        };
        let stop = Arc::new(AtomicBool::new(false));
        let budget = TimeBudget {
            soft_deadline: None,
            hard_deadline: None,
        };
        let handle = spawn_search(game, limits, stop, budget, tt.clone());
        let _ = handle.join();

        // Table still lockable; config preserved.
        let g = tt.lock().unwrap();
        assert_eq!(g.size_mb(), 2, "config preserved after search");
        assert!(g.capacity_entries() > 0);
    }
}
