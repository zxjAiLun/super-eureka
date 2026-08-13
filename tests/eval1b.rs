//! EVAL 1B — exact KQK/KRK mop-up evaluation regressions.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use eureka::chess::{move_to_uci, parse_fen, to_fen};
use eureka::engine::evaluate;
use eureka::engine::search::{search_best_move, SearchContext, SearchLimits, MATE};

#[test]
fn kqk_rewards_edge_and_king_proximity() {
    let edge = parse_fen("7k/8/5K2/8/3Q4/8/8/8 w - - 0 1").unwrap();
    let centre = parse_fen("8/8/5K2/8/3Qk3/8/8/8 w - - 0 1").unwrap();
    let far_king = parse_fen("7k/8/8/8/3Q4/8/8/K7 w - - 0 1").unwrap();

    assert!(
        evaluate(&edge) > evaluate(&centre),
        "KQK should reward driving the weak king toward the edge"
    );
    assert!(
        evaluate(&edge) > evaluate(&far_king),
        "KQK should reward bringing the strong king closer"
    );
}

#[test]
fn krk_rewards_confinement_without_changing_material_semantics() {
    let edge = parse_fen("7k/8/5K2/8/3R4/8/8/8 w - - 0 1").unwrap();
    let centre = parse_fen("8/8/5K2/8/3Rk3/8/8/8 w - - 0 1").unwrap();

    assert!(
        evaluate(&edge) > evaluate(&centre),
        "KRK should reward a rook wall that confines the weak king"
    );
    assert!(
        evaluate(&edge) > 0,
        "KRK must remain a winning material signal"
    );
}

#[test]
fn exact_mopup_does_not_leak_into_positions_with_other_material() {
    let exact = parse_fen("7k/8/5K2/8/3Q4/8/8/8 w - - 0 1").unwrap();
    let with_pawn = parse_fen("7k/8/5K2/8/3Q4/8/P7/8 w - - 0 1").unwrap();
    let with_second_queen = parse_fen("7k/8/5K2/8/3Q4/8/1Q6/8 w - - 0 1").unwrap();

    // Adding a pawn may change the ordinary tapered score, but it must not
    // receive the exact KQK mop-up term merely because a queen is present.
    assert_ne!(evaluate(&exact), evaluate(&with_pawn));
    // A second heavy piece also invalidates the exact signature; it must not
    // be silently ignored by the detector.
    assert_ne!(evaluate(&exact), evaluate(&with_second_queen));
}

#[test]
fn stalemate_is_not_scored_as_a_mopup_win() {
    let stalemate = parse_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1").unwrap();
    assert!(stalemate.is_stalemate());
    assert_eq!(evaluate(&stalemate), 0);
}

#[test]
fn exact_mopup_is_read_only() {
    let pos = parse_fen("7k/8/5K2/8/3R4/8/8/8 w - - 0 1").unwrap();
    let before = to_fen(&pos);
    let _ = evaluate(&pos);
    assert_eq!(to_fen(&pos), before);
}

#[test]
fn kqk_search_chooses_a_confining_queen_move() {
    let mut pos = parse_fen("7k/8/5K2/8/3Q4/8/8/8 w - - 0 1").unwrap();
    let before = to_fen(&pos);
    let ctx = Arc::new(SearchContext::new(Arc::new(AtomicBool::new(false))));
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("search completes");

    assert_eq!(move_to_uci(out.best_move), "d4g1");
    assert!(
        out.score.expect("score") >= MATE - 4,
        "KQK should see a short mate"
    );
    assert_eq!(to_fen(&pos), before, "search must restore the KQK root");
    assert!(ctx.nodes.load(Ordering::Relaxed) > 1);
}

#[test]
fn krk_search_chooses_a_confining_rook_move() {
    let mut pos = parse_fen("7k/8/5K2/8/3R4/8/8/8 w - - 0 1").unwrap();
    let before = to_fen(&pos);
    let ctx = Arc::new(SearchContext::new(Arc::new(AtomicBool::new(false))));
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("search completes");

    assert_eq!(move_to_uci(out.best_move), "f6g6");
    assert!(
        out.score.expect("score") >= MATE - 4,
        "KRK should see a short mate"
    );
    assert_eq!(to_fen(&pos), before, "search must restore the KRK root");
    assert!(ctx.nodes.load(Ordering::Relaxed) > 1);
}
