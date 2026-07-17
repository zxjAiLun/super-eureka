//! M1.3: time control — search-core tests for soft/hard deadlines and the
//! unified depth/nodes/time termination semantics.
//!
//! These tests drive `search_best_move` directly (no binary) with explicit
//! `TimeBudget`s so they are deterministic regardless of host speed.

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
        infinite: false,
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
        infinite: false,
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
            infinite: false,
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
            infinite: false,
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
            infinite: false,
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
            infinite: false,
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
