//! M2.4 — Piece-Square Table evaluation tests.
//!
//! Per `docs/specs/m2.4-piece-square-evaluation.md`. Every search
//! test passes an explicit `depth` (never `SearchLimits::default()`, which
//! means "iterate forever until stopped" and would hang the test).
//!
//! The node baselines in `pst_fixed_depth_search_baselines` are the NEW
//! M2.4 numbers (PST changes alpha-beta cutoffs, so the old 610/927
//! from M2.3 no longer apply). They are deliberately NOT written under
//! the M2.3 node-count test.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use chess_engine_demo::chess::{move_to_uci, parse_fen, to_fen, START_FEN};
use chess_engine_demo::engine::evaluate;
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits};

fn fresh_ctx() -> Arc<SearchContext> {
    Arc::new(SearchContext::new(Arc::new(AtomicBool::new(false))))
}

/// Test 1 (main proof of `sq ^ 56`): an asymmetric Pawn mirror pair.
/// White pawn d7 and Black pawn d2 must both score 150 and be equal.
/// A correct implementation mirrors Black via `d2 ^ 56 == d7`.
/// (Knight c3/c6 happen to share value 10, so that pair alone does
/// NOT prove the mirror — it is kept only as a sanity check.)
#[test]
fn pst_color_mirror_pawn_pair_is_main_proof() {
    let white_d7 = parse_fen("k7/3P4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let black_d2 = parse_fen("4k3/8/8/8/8/8/3p4/K7 b - - 0 1").unwrap();

    assert_eq!(
        evaluate(&white_d7),
        150,
        "white d7 = 100 + PAWN_PST[d7](50)"
    );
    assert_eq!(
        evaluate(&black_d2),
        150,
        "black d2 mirrors to d7 -> 100 + PAWN_PST[d7](50); omitting the \
         mirror reads PAWN_PST[d2](-20) -> 80, which FAILS this test"
    );
    assert_eq!(evaluate(&white_d7), evaluate(&black_d2));

    // Sanity only: Knight c3/c6 coincidence (both == 10). Passes even
    // if `sq ^ 56` is omitted, so it is NOT a mirror proof.
    let white_n_c3 = parse_fen("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1").unwrap();
    let black_n_c6 = parse_fen("4K3/8/2n5/8/8/8/8/4k3 b - - 0 1").unwrap();
    assert_eq!(evaluate(&white_n_c3), evaluate(&black_n_c6));
}

/// Test 2: flipping only the side field negates the score.
#[test]
fn pst_perspective_negates_on_side_flip() {
    let white_to_move = parse_fen("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1").unwrap();
    let black_to_move = parse_fen("4k3/8/8/8/8/2N5/8/4K3 b - - 0 1").unwrap();
    assert_eq!(
        evaluate(&white_to_move),
        -evaluate(&black_to_move),
        "flipping only the side field must negate the score"
    );
}

/// Test 3: a centralized knight beats an edge knight.
#[test]
fn pst_center_beats_edge() {
    let n_c3 = parse_fen("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1").unwrap(); // N on c3
    let n_a3 = parse_fen("4k3/8/8/8/8/N7/8/4K3 w - - 0 1").unwrap(); // N on a3
    assert!(
        evaluate(&n_c3) - evaluate(&n_a3) > 0,
        "KNIGHT_PST[c3](10) must beat KNIGHT_PST[a3](-30)"
    );
}

/// Test 4: material still dominates; PST never overturns a queen.
#[test]
fn pst_material_still_dominates() {
    let queen_up = parse_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1").unwrap(); // Q a1 + K e1 vs K a8
    let no_queen = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(evaluate(&queen_up), 880, "900 + QUEEN_PST[a1](-20)");
    assert_eq!(evaluate(&no_queen), 0);
    assert!(evaluate(&queen_up) > evaluate(&no_queen));
}

/// Test 5: Pawn fixed value + vertical-flip detection. The Pawn table is
/// strongly asymmetric, so it catches a fully vertically-flipped table
/// (which a near-symmetric Knight table would mask).
#[test]
fn pst_pawn_fixed_value_and_flip_detection() {
    let pos = parse_fen("k7/3P4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(
        evaluate(&pos),
        150,
        "white pawn d7 = 100 + PAWN_PST[d7](50)"
    );
    // If the whole table were flipped vertically, d7 would read d2's -20,
    // giving 80, and this assertion would fail.
}

/// Test 6: `evaluate` is read-only; it never mutates the position.
#[test]
fn evaluate_does_not_mutate_position() {
    let pos = parse_fen("k7/3P4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let before = to_fen(&pos);
    let _ = evaluate(&pos);
    assert_eq!(to_fen(&pos), before, "evaluate must be read-only");
}

/// Test 7: PST really changes the engine's choice. Same position, but
/// the search root move flips from the M2.3 baseline `h1f2` to `h1g3`
/// because KNIGHT_PST[g3](5) > KNIGHT_PST[f2](0). Depth is fixed.
#[test]
fn pst_changes_root_choice_h1f2_to_h1g3() {
    let mut pos = parse_fen("4k3/8/8/8/8/8/K7/7N w - - 0 1").unwrap();
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(1),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
    assert_eq!(
        to_fen(&pos),
        "4k3/8/8/8/8/8/K7/7N w - - 0 1",
        "root position must be restored"
    );
    assert_eq!(
        move_to_uci(out.best_move),
        "h1g3",
        "PST must prefer g3 (bonus 5) over f2 (bonus 0); pure material picked h1f2"
    );
}

/// Test 9 (node baselines, re-attributed here, NOT under the M2.3 test):
/// M2.4 changes node counts because PST changes alpha-beta cutoffs.
/// These are the NEW fixed baselines measured on this commit.
/// Run a fixed depth-3 search and return (node count, best move uci),
/// asserting the root position is restored.
fn depth3_nodes_and_best(fen: &str) -> (u64, String) {
    let mut pos = parse_fen(fen).unwrap();
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");
    let nodes = ctx.nodes.load(Ordering::Relaxed);
    let best = move_to_uci(out.best_move);
    assert_eq!(to_fen(&pos), before, "root position restored");
    (nodes, best)
}

/// Test 9 (node baselines, re-attributed here, NOT under the M2.3 test):
/// M2.4 changes node counts because PST changes alpha-beta cutoffs.
/// These are the NEW fixed baselines measured on this commit.
#[test]
fn pst_fixed_depth_search_baselines() {
    let (startpos_nodes, startpos_best) = depth3_nodes_and_best(START_FEN);
    assert_eq!(
        startpos_nodes, 1149,
        "startpos depth-3 node baseline (PST era)"
    );
    assert_eq!(startpos_best, "b1c3");

    let (queenwin_nodes, queenwin_best) = depth3_nodes_and_best("7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1");
    assert_eq!(
        queenwin_nodes, 963,
        "queen-win depth-3 node baseline (PST era)"
    );
    assert_eq!(queenwin_best, "e4a4");
}
