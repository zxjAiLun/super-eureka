//! M2.3: principal variation tracking + UCI `info ... pv`.
//!
//! The PV must NOT change any search decision or node count; a partially
//! searched (aborted) iteration's PV must never leak into the final result.
//!
//! Acceptance baselines measured on `c339b37` (pre-PV): startpos depth 3
//! is 610 nodes, bestmove b1c3, cp 0; queen-win depth 3 is 927 nodes,
//! bestmove e4a4, cp 900. Those exact node counts must hold AFTER PV
//! tracking, proving PV only records results and triggers no extra search.

mod common;

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

/// Test 5 (hardened): a deterministic node budget aborts a *deeper*
/// iteration while the last fully completed iteration's PV is preserved
/// byte-for-byte — never the partial new one.
///
/// startpos uses 20 nodes for depth 1; a budget of 21 lets depth 2
/// acquire at most one more node, then abort.
///
/// The hardening over the old test: instead of only checking `pv` is
/// non-empty and `pv[0] == best_move` (which a partial depth-2 tail
/// that happened to start with the same move would defeat), we record an
/// *independent* full depth-1 result first and demand the aborted run's
/// score / best move / full PV all match it exactly. We also verify the
/// root position is restored after BOTH searches.
#[test]
fn node_budget_aborts_deeper_iteration_preserves_completed_pv() {
    // Baseline: a fully completed depth-1 search, recorded on its own
    // (separate context so its node counter cannot bleed into the aborted
    // run). Also confirms the baseline leaves the root untouched.
    let mut base_pos = parse_fen(START_FEN).unwrap();
    let base_before = to_fen(&base_pos);
    let base = {
        let ctx = fresh_ctx();
        let limits = SearchLimits {
            depth: Some(1),
            ..Default::default()
        };
        search_best_move(&mut base_pos, &limits, &ctx).expect("depth-1 outcome")
    };
    assert_eq!(
        to_fen(&base_pos),
        base_before,
        "depth-1 baseline must restore the root"
    );

    // Aborted run: depth budget open, node budget 21 lets depth 1
    // complete (20 nodes) then aborts depth 2.
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
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

    // The real invariant: the aborted run reports the *completed depth-1*
    // result byte-for-byte. A partial depth-2 PV (even a truncated one
    // whose first move equals the best move) must NOT leak through.
    assert_eq!(
        out.score, base.score,
        "score must match the depth-1 baseline"
    );
    assert_eq!(
        out.best_move, base.best_move,
        "best move must match the depth-1 baseline"
    );
    assert_eq!(
        out.pv, base.pv,
        "aborted run must report the completed depth-1 PV, not a partial depth-2 tail"
    );

    assert_eq!(
        to_fen(&pos),
        before,
        "root position untouched after aborted search"
    );
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

/// Test 8 (hardened): UCI integration. Every completed iteration emits
/// `info ... pv` with a non-empty PV, and the final `bestmove` equals the
/// first move of the last completed iteration's PV. Driven through the real
/// binary via the shared RAII [`common::EngineProcess`] helper, so there is
/// no fixed `sleep` guessing how long depth 3 takes, and a panicking
/// assertion still reaps the child (no stale-exe lock on the next build).
#[test]
fn uci_info_carries_full_pv_and_bestmove_matches() {
    let (mut engine, stdout) = common::spawn_engine();
    let reader = common::spawn_reader(stdout);

    engine.send("position startpos");
    engine.send("go depth 3");

    // Wait for each iteration's info line to actually arrive — no guessing
    // the search duration. 5s per line is generous even under load.
    let d1 =
        common::recv_until(&reader, "info depth 1", Duration::from_secs(5)).expect("depth 1 info");
    let d2 =
        common::recv_until(&reader, "info depth 2", Duration::from_secs(5)).expect("depth 2 info");
    let d3 =
        common::recv_until(&reader, "info depth 3", Duration::from_secs(5)).expect("depth 3 info");
    let best = common::recv_until(&reader, "bestmove", Duration::from_secs(5)).expect("bestmove");

    // Each completed iteration's info carries a non-empty pv. Iterate by
    // reference so `d3` stays owned for the final check below.
    for info in [&d1, &d2, &d3] {
        let pv = info.split(" pv ").nth(1).expect("info carries a pv field");
        assert!(
            !pv.trim().is_empty(),
            "completed-iteration info pv must be non-empty"
        );
    }

    // The final bestmove equals the first pv move of the last iteration.
    let last_pv = d3
        .split(" pv ")
        .nth(1)
        .expect("depth 3 info carries a pv field");
    let first = last_pv.split_whitespace().next().expect("pv non-empty");
    let best_move = best
        .split_whitespace()
        .nth(1)
        .expect("bestmove has a move token");
    assert_eq!(
        first, best_move,
        "bestmove must equal the first pv move of the last completed iteration"
    );
    // `engine` is dropped here: EngineProcess::Drop sends `quit`, closes
    // the pipe, and reaps the child even if an assertion above panicked.
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
