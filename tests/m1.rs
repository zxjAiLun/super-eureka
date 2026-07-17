//! Milestone 1.1: interruptible search core.
//!
//! Correctness contract that must hold no matter where the abort lands:
//!  - a preset stop flag makes the search exit immediately and return a
//!    legal fallback move, leaving the position untouched;
//!  - a node limit aborts mid-iteration but keeps the last *completed*
//!    iteration's result, still leaving the position untouched;
//!  - even a depth-1 that never completes yields a legal fallback move;
//!  - an unmet node limit must not change the result at all.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use chess_engine_demo::chess::types::START_FEN;
use chess_engine_demo::chess::{generate_legal_moves, parse_fen, to_fen};
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits};

#[test]
fn m1_stop_preset_returns_fallback_and_keeps_position() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    // Stop is already requested before the search even starts.
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(true)));
    let limits = SearchLimits {
        depth: Some(4),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx);
    let mv = out
        .expect("a legal fallback move must still be returned")
        .best_move;
    // The fallback must be one of the legal root moves.
    assert!(
        generate_legal_moves(&mut pos).contains(&mv),
        "fallback move must be legal"
    );
    // The board must be byte-for-byte identical to before: no move was
    // left applied, because negamax bails out before making any.
    assert_eq!(
        to_fen(&pos),
        before,
        "position must be untouched after abort"
    );
    // A preset stop must not have counted a single node.
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        0,
        "preset stop must not increment the node counter"
    );
}

#[test]
fn m1_nodes_limit_keeps_last_completed_iteration() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);

    // Calibrate how many nodes a full depth-1 search costs and what its
    // best move is. A node budget comfortably above depth 1 but far
    // below depth 2 guarantees depth 1 completes and depth 2 aborts.
    let probe_ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let probe_limits = SearchLimits {
        depth: Some(1),
        ..Default::default()
    };
    let depth1 = search_best_move(&mut pos, &probe_limits, &probe_ctx)
        .expect("depth 1 completes with no limit");
    let depth1_nodes = probe_ctx.nodes.load(Ordering::Relaxed);
    let expected_mv = depth1.best_move;

    let budget = depth1_nodes + 5; // > depth-1 cost, << depth-2 cost
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits {
        depth: Some(64),
        nodes: Some(budget),
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("a legal move must be returned");
    let mv = out.best_move;
    // We completed depth 1 then aborted during depth 2, so the returned
    // move is exactly depth 1's best move.
    assert_eq!(out.completed_depth, 1, "depth 1 must have completed");
    assert!(out.stopped, "search must be stopped at depth 2");
    assert_eq!(mv, expected_mv, "must keep the last completed iteration");
    // And nothing was corrupted by the mid-search abort; every made move
    // on the aborted path was unmade on the way back up.
    assert_eq!(
        to_fen(&pos),
        before,
        "position must be untouched after abort"
    );
    // The node counter must land exactly on the budget: every node counted
    // was genuinely searched, and it never overshoots (the last node that
    // would exceed the budget is rejected, not counted).
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        budget,
        "node counter must equal the budget exactly"
    );
}

#[test]
fn m1_depth1_interrupted_still_returns_fallback() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    // Budget of 1 node: even depth 1 cannot finish, so the search must
    // fall back to a legal root move.
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits {
        depth: Some(4),
        nodes: Some(1),
    };
    let out =
        search_best_move(&mut pos, &limits, &ctx).expect("a legal fallback move must be returned");
    let mv = out.best_move;
    assert!(
        generate_legal_moves(&mut pos).contains(&mv),
        "fallback move must be legal"
    );
    assert_eq!(
        to_fen(&pos),
        before,
        "position must be untouched after abort"
    );
    // No iteration completed: score must be `None`, never a fabricated 0,
    // and the reported depth must be 0.
    assert!(
        out.score.is_none(),
        "no completed iteration must yield score None, not a fabricated 0"
    );
    assert_eq!(out.completed_depth, 0, "depth 1 never completed");
    assert!(out.stopped, "search must be stopped");
    // Budget 1 is enough to search exactly one node (the first root
    // child); the counter must read exactly 1, never 0 or 2.
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        1,
        "nodes 1 must count exactly one node"
    );
}

#[test]
fn m1_nodes_zero_processes_nothing() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    // Budget of 0 nodes: the search must not process a single node and
    // must fall straight back to a legal root move. The counter stays 0.
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits {
        depth: Some(4),
        nodes: Some(0),
    };
    let out =
        search_best_move(&mut pos, &limits, &ctx).expect("a legal fallback move must be returned");
    let mv = out.best_move;
    assert!(
        generate_legal_moves(&mut pos).contains(&mv),
        "fallback move must be legal"
    );
    assert_eq!(
        to_fen(&pos),
        before,
        "position must be untouched after abort"
    );
    assert!(
        out.score.is_none(),
        "no completed iteration must yield score None, not a fabricated 0"
    );
    assert_eq!(out.completed_depth, 0, "depth 1 never completed");
    assert!(out.stopped, "search must be stopped");
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        0,
        "nodes 0 must process zero nodes"
    );
}

#[test]
fn m1_unmet_node_limit_does_not_change_result() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let clean = search_best_move(
        &mut pos,
        &SearchLimits {
            depth: Some(2),
            ..Default::default()
        },
        &SearchContext::new(Arc::new(AtomicBool::new(false))),
    )
    .expect("depth 2 completes")
    .best_move;
    // A generous budget must let the full depth-2 search finish and agree
    // with the unlimited run; the position stays intact either way.
    let big = search_best_move(
        &mut pos,
        &SearchLimits {
            depth: Some(2),
            nodes: Some(1_000_000),
        },
        &SearchContext::new(Arc::new(AtomicBool::new(false))),
    )
    .expect("depth 2 completes under a large budget")
    .best_move;
    assert_eq!(clean, big, "unmet node limit must not alter the result");
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}
