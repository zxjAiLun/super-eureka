//! PERF profiling counters are observational and internally consistent.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use eureka::chess::{parse_fen, to_fen, START_FEN};
use eureka::engine::search::{search_best_move, SearchContext, SearchLimits};

#[test]
fn search_stats_cover_the_search_without_changing_the_root() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let ctx = Arc::new(SearchContext::new_with_profiling(
        Arc::new(AtomicBool::new(false)),
        true,
    ));
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };

    let outcome = search_best_move(&mut pos, &limits, &ctx).expect("startpos is non-terminal");
    let stats = ctx.stats();

    assert_eq!(
        stats.nodes,
        ctx.nodes.load(std::sync::atomic::Ordering::Relaxed)
    );
    assert_eq!(to_fen(&pos), before);
    assert_eq!(outcome.completed_depth, 3);
    assert_eq!(stats.completed_iterations, 3);
    assert!(stats.last_completed_iteration_nodes > 0);
    assert_eq!(stats.aborted_iteration_depth, 0);
    assert_eq!(stats.aborted_iteration_nodes, 0);
    assert!(stats.nodes > 0);
    assert!(stats.qsearch_nodes > 0);
    assert!(stats.eval_calls > 0);
    assert!(stats.legal_move_generations > 0);
    assert!(stats.pseudo_moves >= stats.legal_moves);
    assert_eq!(stats.make_moves, stats.unmake_moves);
    assert!(stats.tt_probes > 0);
    assert_eq!(stats.tt_hits, 0, "public baseline search has TT disabled");
}

#[test]
fn default_search_does_not_collect_diagnostic_counters() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = Arc::new(SearchContext::new(Arc::new(AtomicBool::new(false))));
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };

    search_best_move(&mut pos, &limits, &ctx).expect("startpos is non-terminal");
    let stats = ctx.stats();

    assert!(stats.nodes > 0);
    assert_eq!(stats.qsearch_nodes, 0);
    assert_eq!(stats.eval_calls, 0);
    assert_eq!(stats.legal_move_generations, 0);
    assert_eq!(stats.pseudo_moves, 0);
    assert_eq!(stats.legal_moves, 0);
    assert_eq!(stats.make_moves, 0);
    assert_eq!(stats.unmake_moves, 0);
    assert_eq!(stats.tt_probes, 0);
    assert_eq!(stats.tt_hits, 0);
    assert_eq!(stats.tt_cutoffs, 0);
    assert_eq!(stats.tt_rejected_depth, 0);
    assert_eq!(stats.tt_rejected_bound, 0);
    assert_eq!(stats.tt_rejected_decode, 0);
    assert_eq!(stats.tt_stores, 0);
    assert_eq!(stats.see_calls, 0);
    assert_eq!(stats.see_pruned, 0);
    assert_eq!(stats.qsearch_see_tests, 0);
    assert_eq!(stats.qsearch_see_pruned, 0);
    assert_eq!(stats.qsearch_see_fail_open_promotions, 0);
    assert_eq!(stats.qsearch_checking_captures_kept, 0);
    assert_eq!(stats.qsearch_promotions_kept, 0);
    assert_eq!(stats.qsearch_en_passant_kept, 0);
    assert_eq!(stats.aspiration_retries, 0);
    assert_eq!(stats.aspiration_fail_low, 0);
    assert_eq!(stats.aspiration_fail_high, 0);
    assert_eq!(stats.lmr_reductions, 0);
    assert_eq!(stats.lmr_researches, 0);
    assert_eq!(stats.null_move_attempts, 0);
    assert_eq!(stats.null_move_fail_highs, 0);
    assert_eq!(stats.null_move_researches, 0);
    assert_eq!(stats.futility_pruned, 0);
    assert_eq!(stats.completed_iterations, 0);
    assert_eq!(stats.last_completed_iteration_ms, 0);
    assert_eq!(stats.last_completed_iteration_nodes, 0);
    assert_eq!(stats.aborted_iteration_depth, 0);
    assert_eq!(stats.aborted_iteration_nodes, 0);
}

#[test]
fn profiling_records_an_aborted_iteration() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let ctx = Arc::new(SearchContext::new_with_profiling(
        Arc::new(AtomicBool::new(false)),
        true,
    ));
    let limits = SearchLimits {
        nodes: Some(1),
        ..Default::default()
    };

    let outcome = search_best_move(&mut pos, &limits, &ctx).expect("startpos is non-terminal");
    let stats = ctx.stats();

    assert!(outcome.stopped);
    assert_eq!(outcome.completed_depth, 0);
    assert_eq!(stats.completed_iterations, 0);
    assert_eq!(stats.aborted_iteration_depth, 1);
    assert!(stats.aborted_iteration_nodes > 0);
}
