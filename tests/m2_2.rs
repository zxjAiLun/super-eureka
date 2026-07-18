//! M2.2: basic capture move ordering (MVV-LVA).
//!
//! Acceptance (from the review): ordering must not change the search's
//! best move or score on a fixed-depth search, must not drop any legal
//! move, must keep perft identical, and must not silently explode the node
//! count. It does NOT guarantee nodes never increase — MVV-LVA is a
//! heuristic, and under alpha-beta the visited node set depends on where
//! cutoffs fire, which ordering changes. The pure ordering logic itself is
//! unit-tested in `src/engine/search.rs` (the `tests` submodule).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use chess_engine_demo::chess::move_to_uci;
use chess_engine_demo::chess::parse_fen;
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits};

/// White queen e4 wins the black queen on a4 (the h4 pawn is a lesser
/// capture). A correct search must pick the queen capture at any depth >= 1,
/// and move ordering must not alter that choice.
#[test]
fn ordering_finds_queen_win_tactic() {
    let mut pos = parse_fen("7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1").expect("valid FEN");
    let (ctx, limits) = (
        SearchContext::new(Arc::new(AtomicBool::new(false))),
        SearchLimits {
            depth: Some(4),
            ..Default::default()
        },
    );
    let out = search_best_move(&mut pos, &limits, &ctx).expect("legal move exists");
    assert_eq!(
        move_to_uci(out.best_move),
        "e4a4",
        "best move must be the queen-winning capture"
    );
    let score = out.score.expect("depth 4 yields a real score");
    assert!(
        score > 0,
        "winning the queen must score positive, got {}",
        score
    );
}

/// Smoke bound, NOT a proof of node reduction. MVV-LVA is a heuristic
/// whose effect on the node count depends on where cutoffs land, so this
/// test does not claim ordering reduces nodes. What it guards is a *gross*
/// regression: the search of this small position must not explode into an
/// implausibly large node count. It asserts only `nodes <= CEILING`.
///
/// Reference ceiling for depth 3 on the queen-win position; refresh only if
/// the search logic itself changes, not because ordering "tidied up a bit".
#[test]
fn ordering_node_count_smoke_bound() {
    let mut pos = parse_fen("7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1").expect("valid FEN");
    let (ctx, limits) = (
        SearchContext::new(Arc::new(AtomicBool::new(false))),
        SearchLimits {
            depth: Some(3),
            ..Default::default()
        },
    );
    let _ = search_best_move(&mut pos, &limits, &ctx);
    let nodes = ctx.nodes.load(Ordering::Relaxed);
    assert!(
        nodes <= 50_000,
        "tactical depth-3 node count implausibly high (ordering regression?): {}",
        nodes
    );
}
