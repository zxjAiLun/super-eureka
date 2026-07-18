//! M2.3: principal variation tracking + UCI `info ... pv`.
//!
//! The PV must NOT change any search decision or node count; a partially
//! searched (aborted) iteration's PV must never leak into the final result.
//!
//! Acceptance baselines measured on `c339b37` (pre-PV): startpos depth 3
//! is 610 nodes, bestmove b1c3, cp 0; queen-win depth 3 is 927 nodes,
//! bestmove e4a4, cp 900. Those exact node counts must hold AFTER PV
//! tracking, proving PV only records results and triggers no extra search.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chess_engine_demo::chess::{generate_legal_moves, move_to_uci, parse_fen, to_fen, Position};
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits, MATE};

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const ALPHA: i32 = i32::MIN + 1000;
const BETA: i32 = i32::MAX - 1000;

fn fresh_ctx() -> SearchContext {
    SearchContext::new(Arc::new(AtomicBool::new(false)))
}

fn stopped_ctx() -> SearchContext {
    SearchContext::new(Arc::new(AtomicBool::new(true)))
}

/// Test 1: after a completed fixed-depth search the PV is non-empty and its
/// first move is the reported best move.
#[test]
fn fixed_depth_pv_nonempty_and_starts_with_best() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
    assert!(
        !out.pv.is_empty(),
        "pv must be non-empty after a completed search"
    );
    assert_eq!(out.pv[0], out.best_move, "pv[0] is the best move");
}

/// Test 2: the PV replays legally from the root, and the root position is
/// left byte-for-byte intact. These are TWO independent invariants — the
/// board is not restored "because" the PV replays; they are checked apart.
#[test]
fn pv_replays_legally_from_root_and_restores_position() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");

    let mut replay = pos;
    for mv in &out.pv {
        let legal = generate_legal_moves(&mut replay);
        assert!(
            legal.contains(mv),
            "pv move {} must be legal at this point",
            move_to_uci(*mv)
        );
        replay.make_move(*mv);
    }
    // The root position itself must be untouched by the search.
    assert_eq!(
        to_fen(&pos),
        before,
        "search must not mutate the root position"
    );
}

/// Test 4: a mate-in-one PV starts with the mating move; the score is still
/// a mate (PV tracking must not perturb the tactical value).
#[test]
fn mate_in_one_pv_starts_with_mating_move() {
    let mut pos = parse_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(4),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
    assert_eq!(
        move_to_uci(out.pv[0]),
        "a1a8",
        "mate-in-one PV must start with Ra8"
    );
    assert!(
        out.score.expect("real score") > MATE - 1000,
        "score must still be a mate, got {:?}",
        out.score
    );
}

/// Test 5: a deterministic node budget aborts a *deeper* iteration while the
/// last fully completed iteration's PV is preserved (never the partial new one).
///
/// startpos uses 20 nodes for depth 1; a budget of 21 lets depth 2
/// acquire at most one more node, then abort.
#[test]
fn node_budget_aborts_deeper_iteration_preserves_completed_pv() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: None,
        nodes: Some(21),
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");

    assert_eq!(out.completed_depth, 1, "only depth 1 should complete");
    assert!(out.stopped, "must be stopped");
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        21,
        "exactly the node budget is consumed"
    );
    assert!(
        !out.pv.is_empty(),
        "pv from the completed depth-1 iteration must be present"
    );
    assert_eq!(out.pv[0], out.best_move, "pv[0] is the best move");
    assert!(out.score.is_some(), "depth 1 yields a real score");
}

/// Test 6: stopped before depth 1 finishes -> legal fallback, score None,
/// completed_depth 0, and an EMPTY pv (no fabricated principal variation).
#[test]
fn preset_stop_before_depth1_yields_empty_pv() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = stopped_ctx();
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");

    assert_eq!(out.completed_depth, 0, "no iteration completed");
    assert!(out.stopped);
    assert!(out.score.is_none(), "no real score before depth 1");
    assert!(out.pv.is_empty(), "pv must be empty");
    // The fallback is still a legal move.
    assert!(
        generate_legal_moves(&mut pos).contains(&out.best_move),
        "fallback must be a legal move"
    );
    assert_eq!(to_fen(&pos), START_FEN, "root position untouched");
}

/// Test 7: PV tracking must not change the search's node count. The exact
/// baselines (startpos=610, queen-win=927) were recorded on `c339b37`
/// BEFORE the PV work; any delta here means PV triggered extra search.
#[test]
fn fixed_depth_node_count_unchanged_by_pv_tracking() {
    // startpos depth 3.
    {
        let mut pos = parse_fen(START_FEN).unwrap();
        let ctx = fresh_ctx();
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            610,
            "startpos depth-3 node count must stay 610"
        );
        assert_eq!(move_to_uci(out.best_move), "b1c3");
        assert_eq!(out.score, Some(0));
    }
    // queen-win depth 3.
    {
        let mut pos = parse_fen("7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1").unwrap();
        let ctx = fresh_ctx();
        let limits = SearchLimits {
            depth: Some(3),
            ..Default::default()
        };
        let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            927,
            "queen-win depth-3 node count must stay 927"
        );
        assert_eq!(move_to_uci(out.best_move), "e4a4");
        assert_eq!(out.score, Some(900));
    }
}

/// Test 8: UCI integration. Every completed iteration emits `info ... pv`
/// with a non-empty PV, and the final `bestmove` equals the first move of
/// the last completed iteration's PV. Driven through the real binary so the
/// end-to-end protocol path (printed info lines) is exercised.
#[test]
fn uci_info_carries_full_pv_and_bestmove_matches() {
    // Cargo exposes the compiled binary as `CARGO_BIN_EXE_<name>`, where
    // `<name>` is the literal package/bin name (kept verbatim, not the
    // upper-snake form older docs describe).
    let bin = std::env::var("CARGO_BIN_EXE_chess-engine-demo")
        .map(std::path::PathBuf::from)
        .expect("CARGO_BIN_EXE_CHESS_ENGINE_DEMO must be set by cargo test");

    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::thread;

    let mut child = Command::new(&bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    let mut stdin = child.stdin.take().unwrap();
    // Keep stdin open so the search can finish, then quit.
    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go depth 3").unwrap();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(2));
        let _ = writeln!(stdin, "quit");
    });
    let output = child.wait_with_output().unwrap();
    let text = String::from_utf8(output.stdout).unwrap();

    // Final bestmove.
    let best_line = text
        .lines()
        .find(|l| l.starts_with("bestmove"))
        .expect("bestmove emitted");
    let best_move = best_line
        .split_whitespace()
        .nth(1)
        .expect("bestmove has a move token");

    // Each completed iteration's info carries a non-empty pv; the last one's
    // first move matches the final bestmove.
    for d in 1..=3 {
        let info = text
            .lines()
            .rfind(|l| l.starts_with(&format!("info depth {}", d)))
            .unwrap_or_else(|| panic!("info depth {} emitted", d));
        let pv = info.split(" pv ").nth(1).expect("info carries a pv field");
        assert!(
            !pv.trim().is_empty(),
            "depth {} info pv must be non-empty",
            d
        );
        if d == 3 {
            let first = pv.split_whitespace().next().expect("pv non-empty");
            assert_eq!(
                first, best_move,
                "bestmove must equal the first pv move of the last completed iteration"
            );
        }
    }
}

/// Test 9: an aborted (node-budgeted) `negamax` search leaves the board
/// byte-for-byte intact. The internal temp PV may retain partial content
/// after the unwind — that is allowed; what must hold is (a) the board is
/// restored, (b) `None` propagates, and (c) because the table is a
/// discarded local, nothing leaks into `SearchOutcome` or UCI output.
#[test]
fn negamax_abort_restores_position() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    // depth 1, budget 1: the entry node is spent, then the first child
    // cannot acquire a node and the search aborts.
    let limits = SearchLimits {
        depth: Some(1),
        nodes: Some(1),
    };
    let out = negamax_call(&mut pos, 1, 0, ALPHA, BETA, &ctx, &limits);
    assert!(out.is_none(), "exhausted node budget must abort");
    assert_eq!(
        to_fen(&pos),
        before,
        "position must be untouched after abort"
    );
}

/// Thin wrapper so test 9 reads like the public surface it exercises.
fn negamax_call(
    pos: &mut Position,
    depth: u32,
    ply: u32,
    alpha: i32,
    beta: i32,
    ctx: &SearchContext,
    limits: &SearchLimits,
) -> Option<i32> {
    chess_engine_demo::engine::search::negamax(pos, depth, ply, alpha, beta, ctx, limits)
}
