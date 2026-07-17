//! M1.3: time control — search-core tests for soft/hard deadlines and the
//! unified depth/nodes/time termination semantics, plus UCI integration
//! tests that drive the real binary through `go movetime` / `go infinite` /
//! `go wtime btime` and check the `info` fields.
//!
//! The first group drives `search_best_move` directly (no binary) with
//! explicit `TimeBudget`s so they are deterministic regardless of host
//! speed; the second group spawns the engine binary.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chess_engine_demo::chess::types::START_FEN;
use chess_engine_demo::chess::{generate_legal_moves, parse_fen, to_fen};
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits};
use chess_engine_demo::engine::TimeBudget;

fn ctx_with(stop: Arc<AtomicBool>, soft: Option<Instant>, hard: Option<Instant>) -> SearchContext {
    SearchContext::with_budget(
        stop,
        TimeBudget {
            soft_deadline: soft,
            hard_deadline: hard,
        },
    )
}

/// A hard deadline that is already in the past must abort before any node is
/// searched: zero nodes, legal fallback, no score, depth 0, stopped.
#[test]
fn hard_deadline_already_expired_stops_immediately() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let now = Instant::now();
    let hard = now.checked_sub(Duration::from_millis(1)).unwrap_or(now);
    let ctx = ctx_with(Arc::new(AtomicBool::new(false)), None, Some(hard));
    let limits = SearchLimits {
        depth: None,
        nodes: None,
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
    assert!(
        generate_legal_moves(&mut pos.clone()).contains(&out.best_move),
        "fallback must be legal"
    );
    assert_eq!(out.score, None, "no iteration completed -> score None");
    assert_eq!(out.completed_depth, 0, "no iteration completed");
    assert!(out.stopped, "must be stopped");
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        0,
        "must count zero nodes"
    );
    assert_eq!(to_fen(&pos), before, "position untouched");
}

/// A soft deadline only fires *between* iterations, so depth 1 still
/// completes even when soft is already in the past; depth 2 never starts.
#[test]
fn soft_deadline_keeps_depth1_and_skips_depth2() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let now = Instant::now();
    let soft = now.checked_sub(Duration::from_millis(1)).unwrap_or(now);
    let hard = now.checked_add(Duration::from_secs(10)).unwrap();
    let ctx = ctx_with(Arc::new(AtomicBool::new(false)), Some(soft), Some(hard));
    let limits = SearchLimits {
        depth: Some(10),
        nodes: None,
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
    assert_eq!(
        out.completed_depth, 1,
        "depth 1 must complete; soft must prevent depth 2"
    );
    assert!(out.score.is_some(), "depth 1 produced a real score");
    assert!(
        out.stopped,
        "soft deadline must stop before reaching the depth cap"
    );
}

/// With both depth and nodes, whichever limit hits first stops the search.
#[test]
fn depth_and_nodes_whichever_hits_first() {
    // depth limit hits first -> natural finish, not stopped.
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = ctx_with(Arc::new(AtomicBool::new(false)), None, None);
    let out = search_best_move(
        &mut pos,
        &SearchLimits {
            depth: Some(2),
            nodes: Some(1_000_000),
        },
        &ctx,
    )
    .expect("outcome");
    assert_eq!(out.completed_depth, 2);
    assert!(
        !out.stopped,
        "reaching the depth cap naturally is not a stop"
    );

    // nodes limit hits first -> stopped, did not reach the depth cap.
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = ctx_with(Arc::new(AtomicBool::new(false)), None, None);
    let out = search_best_move(
        &mut pos,
        &SearchLimits {
            depth: Some(100),
            nodes: Some(1),
        },
        &ctx,
    )
    .expect("outcome");
    assert!(out.stopped, "node budget must stop the search");
    assert!(
        out.completed_depth < 100,
        "must not reach the depth cap when nodes hit first"
    );
}

/// Regression: a nodes-only search must NOT cap at depth 4. With a budget
/// large enough for depth 5, it must reach depth >= 5.
#[test]
fn nodes_only_does_not_cap_at_depth_four() {
    // Probe: how many nodes does a full depth-1..5 search cost?
    let mut pos = parse_fen(START_FEN).unwrap();
    let probe_ctx = ctx_with(Arc::new(AtomicBool::new(false)), None, None);
    let _ = search_best_move(
        &mut pos,
        &SearchLimits {
            depth: Some(5),
            nodes: None,
        },
        &probe_ctx,
    )
    .expect("probe completes");
    let budget = probe_ctx.nodes.load(Ordering::Relaxed);
    assert!(budget > 0, "depth-5 probe must search some nodes");

    // Same budget, no depth cap: must complete depth 5 then stop on budget.
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = ctx_with(Arc::new(AtomicBool::new(false)), None, None);
    let out = search_best_move(
        &mut pos,
        &SearchLimits {
            depth: None,
            nodes: Some(budget),
        },
        &ctx,
    )
    .expect("outcome");
    assert!(
        out.completed_depth >= 5,
        "nodes-only search must reach depth 5+, got {} (no depth-4 cap)",
        out.completed_depth
    );
    assert!(out.stopped, "must stop when the node budget is exhausted");
}

// ---------------------------------------------------------------------------
// UCI integration tests: drive the real engine binary over the protocol.
// ---------------------------------------------------------------------------

/// `go movetime 100` must return a legal bestmove within a loose window:
/// not instantly (it should use roughly the movetime) and not hang.
#[test]
fn uci_go_movetime_returns_bestmove_in_window() {
    let (mut child, mut stdin, stdout) = common::spawn_engine();
    let rx = common::spawn_reader(stdout);
    common::send(&mut stdin, "position startpos");
    let start = Instant::now();
    common::send(&mut stdin, "go movetime 100");
    let bm = common::recv_until(&rx, "bestmove", Duration::from_secs(3));
    let elapsed = start.elapsed();
    common::send(&mut stdin, "quit");
    let _ = child.wait();
    let bm = bm.expect("movetime must produce a bestmove");
    assert!(bm.starts_with("bestmove "), "got {:?}", bm);
    // Loose window: not too fast (should use ~the movetime), not a hang.
    assert!(
        elapsed >= Duration::from_millis(20),
        "returned too fast: {:?}",
        elapsed
    );
    assert!(
        elapsed <= Duration::from_millis(1500),
        "took too long: {:?}",
        elapsed
    );
}

/// `go infinite` must NOT self-emit a bestmove; after `stop` it must.
#[test]
fn uci_go_infinite_waits_for_stop() {
    let (mut child, mut stdin, stdout) = common::spawn_engine();
    let rx = common::spawn_reader(stdout);
    common::send(&mut stdin, "position startpos");
    common::send(&mut stdin, "go infinite");
    // Let it search for a while on its own.
    std::thread::sleep(Duration::from_millis(300));
    // Peek for any bestmove: there must be none before `stop`.
    let leaked = common::recv_until(&rx, "bestmove", Duration::from_millis(80));
    assert!(
        leaked.is_none(),
        "go infinite must not self-emit bestmove before stop"
    );
    common::send(&mut stdin, "stop");
    let bm = common::recv_until(&rx, "bestmove", Duration::from_secs(3));
    common::send(&mut stdin, "quit");
    let _ = child.wait();
    let bm = bm.expect("stop must produce a bestmove");
    assert!(bm.starts_with("bestmove "), "got {:?}", bm);
}

/// `go wtime btime` must return well before the whole clock is spent.
#[test]
fn uci_go_wtime_btime_returns_before_clock_out() {
    let (mut child, mut stdin, stdout) = common::spawn_engine();
    let rx = common::spawn_reader(stdout);
    common::send(&mut stdin, "position startpos");
    let start = Instant::now();
    common::send(&mut stdin, "go wtime 1000 btime 1000");
    let bm = common::recv_until(&rx, "bestmove", Duration::from_secs(3));
    let elapsed = start.elapsed();
    common::send(&mut stdin, "quit");
    let _ = child.wait();
    let bm = bm.expect("wtime/btime must produce a bestmove");
    assert!(bm.starts_with("bestmove "), "got {:?}", bm);
    // White to move: allocation ~ (1000-20)/30 ~ 32ms. Must be far under 1s.
    assert!(
        elapsed <= Duration::from_millis(800),
        "took too long: {:?}",
        elapsed
    );
}

/// When Black is to move, the engine must allocate from `btime`, not
/// `wtime`. Give Black almost no time and White a huge clock: a correct
/// engine returns almost immediately; one that reads `wtime` would search
/// for seconds and time out this assertion.
#[test]
fn uci_black_to_move_uses_btime_not_wtime() {
    let (mut child, mut stdin, stdout) = common::spawn_engine();
    let rx = common::spawn_reader(stdout);
    // After 1.e4 it is Black to move.
    common::send(&mut stdin, "position startpos moves e2e4");
    let start = Instant::now();
    common::send(&mut stdin, "go wtime 60000 btime 60");
    let bm = common::recv_until(&rx, "bestmove", Duration::from_secs(3));
    let elapsed = start.elapsed();
    common::send(&mut stdin, "quit");
    let _ = child.wait();
    let bm = bm.expect("must produce a bestmove");
    assert!(bm.starts_with("bestmove "), "got {:?}", bm);
    // btime=60ms -> allocation ~1-2ms. Must return well under 60ms btime
    // (and nowhere near the 60000ms wtime a buggy engine would use).
    assert!(
        elapsed <= Duration::from_millis(500),
        "took too long (used wtime?): {:?}",
        elapsed
    );
}

/// `info` lines for completed iterations must include nodes / time / nps.
#[test]
fn uci_info_includes_nodes_time_nps() {
    let (mut child, mut stdin, stdout) = common::spawn_engine();
    let rx = common::spawn_reader(stdout);
    common::send(&mut stdin, "position startpos");
    common::send(&mut stdin, "go depth 2");
    let info = common::recv_until(&rx, "info", Duration::from_secs(3));
    // Drain to bestmove so the engine can exit cleanly.
    let bm = common::recv_until(&rx, "bestmove", Duration::from_secs(3));
    common::send(&mut stdin, "quit");
    let _ = child.wait();
    let info = info.expect("must emit at least one info line");
    assert!(info.contains("nodes "), "info must include nodes: {}", info);
    assert!(info.contains("time "), "info must include time: {}", info);
    assert!(info.contains("nps "), "info must include nps: {}", info);
    assert!(bm.is_some(), "must emit bestmove");
}

/// `go infinite` must override any clock / movetime also on the line
/// (e.g. `go infinite wtime 100 btime 100 movetime 50`): the engine must
/// keep searching until `stop`, not self-emit a bestmove after the 50ms
/// movetime or the 100ms clock would have elapsed.
#[test]
fn uci_go_infinite_overrides_clock_and_movetime() {
    let (mut child, mut stdin, stdout) = common::spawn_engine();
    let rx = common::spawn_reader(stdout);
    common::send(&mut stdin, "position startpos");
    common::send(&mut stdin, "go infinite wtime 100 btime 100 movetime 50");
    // Let it run well past the 50ms movetime / 100ms clock.
    std::thread::sleep(Duration::from_millis(300));
    // Peek for a bestmove: there must be none before `stop`.
    let leaked = common::recv_until(&rx, "bestmove", Duration::from_millis(50));
    assert!(
        leaked.is_none(),
        "go infinite must ignore clock/movetime and wait for stop"
    );
    common::send(&mut stdin, "stop");
    let bm = common::recv_until(&rx, "bestmove", Duration::from_secs(3));
    common::send(&mut stdin, "quit");
    let _ = child.wait();
    let bm = bm.expect("stop must produce a bestmove");
    assert!(bm.starts_with("bestmove "), "got {:?}", bm);
}
