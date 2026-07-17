//! Search correctness tests — especially the terminal-node bug (P0 #1):
//! checkmate and stalemate sitting on the search horizon must be scored by
//! their game-theoretic value, never by a plain material count.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use chess_engine_demo::chess::parse_fen;
use chess_engine_demo::engine::search;
use chess_engine_demo::engine::search::{SearchContext, SearchLimits};

const ALPHA: i32 = i32::MIN + 1000;
const BETA: i32 = i32::MAX - 1000;

#[test]
fn negamax_detects_mate_at_depth_zero() {
    // Fool's mate position: White to move is already checkmated by Qh4#.
    // A leaf node must report a mate score, not the material balance.
    let mut pos = parse_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        .expect("valid FEN");
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits::default();
    let score = search::negamax(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits)
        .expect("search must not be stopped");
    assert!(
        score <= -(search::MATE - 1000),
        "checkmate on the horizon should score a mate, got {}",
        score
    );
}

#[test]
fn negamax_detects_stalemate_at_depth_zero() {
    // Black to move is stalemated (king on h8, White Kg6/Qf7). The leaf
    // must score exactly 0, never a material evaluation.
    let mut pos = parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").expect("valid FEN");
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits::default();
    let score = search::negamax(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits)
        .expect("search must not be stopped");
    assert_eq!(score, 0, "stalemate on the horizon must score 0");
}

#[test]
fn search_finds_mate_in_one() {
    // White to move has exactly one mate-in-one: Ra8# (back-rank, black pawns
    // on f7/g7/h7 block the king's escape). A depth-1 search must find it.
    let mut pos = parse_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").expect("valid FEN");
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits::default();
    let outcome = search::search_best_move(&mut pos, &limits, &ctx).expect("there is a legal move");
    assert_eq!(
        chess_engine_demo::chess::move_to_uci(outcome.best_move),
        "a1a8",
        "the only mate-in-one is Ra8"
    );
    assert!(
        outcome.score.expect("depth 4 completes with a real score") > search::MATE - 1000,
        "a mate-in-one must score as a mate, got {:?}",
        outcome.score
    );
}

#[test]
fn search_reports_root_checkmate_as_no_move() {
    // White is already checkmated at the root: search should report no move
    // (the UCI layer then emits "bestmove 0000").
    let mut pos = parse_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        .expect("valid FEN");
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = SearchLimits::default();
    assert!(
        search::search_best_move(&mut pos, &limits, &ctx).is_none(),
        "an already-mated root has no best move"
    );
}
