//! Milestone 2.1: quiescence search correctness.
//!
//! Pure quiescence only — no MVV-LVA, SEE, delta pruning, killer/history,
//! PST, TT, or full PV yet. The point of these tests is to pin down that
//! quiescence *itself* is correct:
//!   - horizon captures are resolved (recapture is seen);
//!   - a side in check never stands pat — it searches every legal evasion;
//!   - checkmate / stalemate reached inside quiescence score correctly;
//!   - quiet promotions and en-passant captures are treated as tactical;
//!   - the search stays interruptible and leaves the board untouched on abort;
//!   - the `MAX_QPLY` cap terminates a would-be infinite check chain.
//!
//! Leaf behaviour is exercised through the public `negamax(depth=0, ...)`
//! path (which delegates to quiescence), and the terminal / cap branches are
//! exercised by calling `quiescence` directly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chess_engine_demo::chess::{generate_legal_moves, parse_fen, to_fen};
use chess_engine_demo::engine::evaluate;
use chess_engine_demo::engine::search::{
    negamax, quiescence, SearchContext, SearchLimits, MATE, MAX_QPLY,
};
use chess_engine_demo::engine::TimeBudget;

const ALPHA: i32 = i32::MIN + 1000;
const BETA: i32 = i32::MAX - 1000;

fn fresh_ctx() -> SearchContext {
    SearchContext::new(Arc::new(AtomicBool::new(false)))
}

fn stopped_ctx() -> SearchContext {
    SearchContext::new(Arc::new(AtomicBool::new(true)))
}

fn ctx_hard_expired() -> SearchContext {
    let now = Instant::now();
    let past = now.checked_sub(Duration::from_millis(1)).unwrap_or(now);
    SearchContext::with_budget(
        Arc::new(AtomicBool::new(false)),
        TimeBudget {
            soft_deadline: None,
            hard_deadline: Some(past),
        },
    )
}

/// Horizon effect: White's only capture (Nf3xe5) *looks* like winning a pawn,
/// but the e5 pawn is defended by the d6 pawn, so after Nxe5 Black plays dxe5
/// and wins the knight. A naive fixed-depth leaf that stopped right after
/// Nxe5 would score White up a pawn (+220). Quiescence must resolve the
/// recapture and reject the losing capture, keeping the stand-pat score.
#[test]
fn quiescence_resolves_horizon_recapture() {
    let mut pos = parse_fen("6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    let stand_pat = evaluate(&pos); // White: N(320) - 2p(200) = +120

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let q = negamax(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    // Quiescence looked past Nxe5 to dxe5 and rejected the losing capture,
    // so the score stays at the stand-pat value — NOT the inflated +pawn.
    assert_eq!(q, stand_pat, "must not overvalue a recaptured capture");
    assert!(
        q < stand_pat + 100,
        "must not report winning the (defended) pawn, got {}",
        q
    );
    // It genuinely searched the capture line (entry node + Nxe5 + dxe5 ...),
    // rather than short-circuiting: more than the single entry node.
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "quiescence must actually search the capture line"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// In check, quiescence must NOT stand pat: it must search the (forced)
/// evasions. Here White (Kg1, Ba8) is in check from the g-file rook (Rg8).
/// Whatever White does to answer the check — a king step or interposing the
/// bishop on g2 — the a8 bishop is attacked by Black's b7 bishop and cannot
/// be saved in one move, so it is lost by force. Static material still counts
/// both bishops on the board (a naive stand-pat-in-check would return that
/// balance), but correct quiescence searches the forced evasion, sees the
/// bishop fall, and returns a score a full minor piece lower.
#[test]
fn quiescence_no_standpat_when_in_check() {
    let mut pos = parse_fen("B5r1/1b6/7k/8/8/8/8/6K1 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    // Both bishops are still on the board here, so the static balance does
    // NOT yet see the forced loss of the a8 bishop.
    let static_eval = evaluate(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check"
    );

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let q = negamax(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    // A stand-pat-in-check bug would return exactly `static_eval`. Correct
    // quiescence searches the forced evasion and discovers the bishop loss,
    // so the score must be strictly worse — by roughly a minor piece.
    assert!(
        q < static_eval,
        "in check, searching the forced evasion must reveal the lost bishop \
         (q={} should be < static {})",
        q,
        static_eval
    );
    assert!(
        q <= static_eval - 300,
        "the forced loss is a whole minor piece, so q ({}) should be at least \
         ~300 below the static balance ({})",
        q,
        static_eval
    );
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "in check, quiescence must search evasions (not stand pat)"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// Checkmate reached in quiescence: the side to move is in check with no
/// legal evasion, so quiescence must return a mate score carrying the ply
/// distance (never a static material count).
#[test]
fn quiescence_scores_checkmate_with_ply_distance() {
    // Fool's mate: White to move is already checkmated (Qh4#).
    let mut pos = parse_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        .expect("valid FEN");
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    let limits = SearchLimits::default();

    let ply = 3u32;
    let score = quiescence(&mut pos, ply, 0, ALPHA, BETA, &ctx, &limits).expect("not stopped");
    assert_eq!(
        score,
        -(MATE - ply as i32),
        "checkmate in quiescence must score a ply-distanced mate"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// Stalemate reached in quiescence: not in check and no legal move must
/// score exactly 0, never a stand-pat material evaluation.
#[test]
fn quiescence_scores_stalemate_zero() {
    // Black to move is stalemated (Kh8; White Kg6, Qf7).
    let mut pos = parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    let limits = SearchLimits::default();

    let score = quiescence(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits).expect("not stopped");
    assert_eq!(score, 0, "stalemate in quiescence must score 0");
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// A *quiet* promotion (`e7e8=Q` onto an empty square) must be treated as
/// tactical and searched. If quiescence judged tacticalness by "target
/// occupied" alone it would skip this move and return the stand-pat pawn
/// score instead of the promoted-queen value.
#[test]
fn quiescence_searches_quiet_promotion() {
    let mut pos = parse_fen("7k/4P3/8/8/8/8/8/K7 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    let stand_pat = evaluate(&pos); // White: pawn(100)

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let q = negamax(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    assert!(
        q > stand_pat,
        "quiet promotion must be searched (q={} should beat stand-pat {})",
        q,
        stand_pat
    );
    assert!(
        q >= 800,
        "promoting to a queen should be worth far more than a pawn, got {}",
        q
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// An en-passant capture must be recognised as tactical (its captured pawn
/// is not on the destination square, so "target occupied" would miss it).
#[test]
fn quiescence_searches_en_passant_capture() {
    // Black has just played ...d7-d5; White can take e5xd6 e.p.
    let mut pos = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    let stand_pat = evaluate(&pos); // material even -> 0

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let q = negamax(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    assert!(
        q > stand_pat,
        "en passant must be searched as a capture (q={} > stand-pat {})",
        q,
        stand_pat
    );
    // M2.4 note: this previously asserted exactly 100 (material-only
    // eval). With piece-square tables the capturing pawn lands on d6,
    // which carries PAWN_PST[d6] = 30, so the quiescence value is
    // 100 (won pawn) + 30 (positional bonus) = 130. The PST
    // table is locked by docs/specs/m2.4, so 130 is stable.
    assert_eq!(
        q, 130,
        "winning the pawn by e.p. is 100 material + PAWN_PST[d6](30) = 130, got {}",
        q
    );
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "the e.p. line must have been searched"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// Interruptibility: a preset stop, an exhausted node budget, and an expired
/// hard deadline must each abort quiescence and leave the board byte-for-byte
/// unchanged (every made move on the aborted path is unmade on the way out).
#[test]
fn quiescence_interrupt_leaves_position_intact() {
    let fen = "6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1"; // has the Nxe5 capture

    // (a) preset stop -> aborts before touching the board.
    {
        let mut pos = parse_fen(fen).unwrap();
        let before = to_fen(&pos);
        let ctx = stopped_ctx();
        let out = quiescence(&mut pos, 0, 0, ALPHA, BETA, &ctx, &SearchLimits::default());
        assert!(out.is_none(), "preset stop must abort quiescence");
        assert_eq!(
            ctx.nodes.load(Ordering::Relaxed),
            0,
            "no node may be counted"
        );
        assert_eq!(to_fen(&pos), before, "position must be untouched");
    }

    // (b) node budget of 1: the entry node is spent, then the recursion into
    // the capture aborts -> None, board restored.
    {
        let mut pos = parse_fen(fen).unwrap();
        let before = to_fen(&pos);
        let ctx = fresh_ctx();
        let limits = SearchLimits {
            depth: None,
            nodes: Some(1),
        };
        let out = quiescence(&mut pos, 0, 0, ALPHA, BETA, &ctx, &limits);
        assert!(out.is_none(), "exhausted node budget must abort");
        assert_eq!(ctx.nodes.load(Ordering::Relaxed), 1, "exactly the budget");
        assert_eq!(to_fen(&pos), before, "position must be untouched");
    }

    // (c) hard deadline already in the past -> aborts before any node.
    {
        let mut pos = parse_fen(fen).unwrap();
        let before = to_fen(&pos);
        let ctx = ctx_hard_expired();
        let out = quiescence(&mut pos, 0, 0, ALPHA, BETA, &ctx, &SearchLimits::default());
        assert!(out.is_none(), "expired hard deadline must abort");
        assert_eq!(to_fen(&pos), before, "position must be untouched");
    }
}

/// The `MAX_QPLY` cap must terminate the recursion: called at the cap, even a
/// position with a pending capture must return the static evaluation without
/// searching any child, and must not corrupt the board.
#[test]
fn quiescence_qply_cap_terminates_without_corruption() {
    let mut pos = parse_fen("6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    let static_eval = evaluate(&pos);

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    // Enter exactly at the cap.
    let out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    assert_eq!(
        out, static_eval,
        "at the qply cap quiescence must return the static eval"
    );
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        1,
        "the cap must prevent searching any child (only the entry node)"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// The blocking bug: at the `MAX_QPLY` cap, a position whose side to move
/// is in check must NOT fall back to a raw `evaluate(pos)` (that would
/// re-introduce stand-pat-on-check — scoring a position with the king still
/// attacked as if it were a quiet leaf). It must still search the evasions.
///
/// Here White (Kg1, Ra2) is in check from the g2 rook (Rg2). The rook is
/// capturable: `Kxg2` wins it. A buggy cap that returns `evaluate(pos)`
/// would report the static balance (material is even, 0); correct behaviour
/// searches the forced evasion and returns the won rook.
#[test]
fn quiescence_qply_cap_preserves_check_evasions() {
    let mut pos = parse_fen("6k1/8/8/8/8/8/R5r1/6K1 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check at the cap"
    );
    let static_eval = evaluate(&pos); // material even -> 0

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    // Enter exactly at the cap, in check.
    let out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    // The forced `Kxg2` wins the rook (+500); a stand-pat-on-check bug
    // would instead return the even material balance (0).
    assert_ne!(
        out, static_eval,
        "cap + in check must search evasions, not stand pat (out={} vs static={})",
        out, static_eval
    );
    assert!(
        out > static_eval + 300,
        "capturing the checking rook is worth far more than the static balance, got {}",
        out
    );
    // Evasions were genuinely searched (entry node + each evasion), not a
    // single-node stand-pat shortcut.
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "cap + in check must search the evasions (not stand pat)"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// Lock the documented cap approximation: an evasion that *itself* gives
/// check (a counter-check) produces a child whose king is also in check.
/// That child has legal moves, so `search_final_evasion_ply` does NOT recurse
/// into it — it is approximated by `-evaluate(child)` at the safety cap. This
/// is the acknowledged, bounded approximation (strict resolution is deferred
/// to repetition / 50-move detection); the test pins the *current* behaviour
/// so a future change to that branch is visible.
///
/// FEN `k7/8/8/8/8/1b6/K7/R7 w - - 0 1`: White (Ka2, Ra1) is in
/// check from the b3 bishop. White can play `Kb1`/`Kb2`, which both evade
/// AND deliver check to the a8 king along the a-file (counter-checks); those
/// children are in check with legal moves, so they are approximated
/// statically. White's best is `Kxb3`, capturing the bishop. A stand-pat-on-
/// check bug (or a non-searching cap) would instead return the static balance
/// (+170: R vs B), but the searched evasion scores the won bishop (+500).
#[test]
fn quiescence_qply_cap_counter_check_uses_static_approx() {
    let mut pos = parse_fen("k7/8/8/8/8/1b6/K7/R7 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check at the cap"
    );
    let static_eval = evaluate(&pos); // White R(500) vs Black B(330) -> +170
    assert_eq!(static_eval, 170, "static balance must be R vs B");

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    // Stand-pat-on-check (or skipping the evasion search) would yield 170.
    // Correct behaviour searches the evasions; `Kxb3` wins the bishop -> 500.
    assert_ne!(
        out, static_eval,
        "cap + in check must search evasions, not stand pat (out={} vs static={})",
        out, static_eval
    );
    assert_eq!(
        out, 500,
        "Kxb3 wins the bishop; counter-check branches are the approx, got {}",
        out
    );
    // Evasions (including the counter-checking Kb1/Kb2) were genuinely
    // searched, not a single-node stand-pat shortcut.
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "cap + in check must search the evasions (incl. counter-checks)"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// At the cap, a position that is already checkmated (no legal evasion) must
/// still return the ply-distanced mate score — the early mate/stalemate
/// detection runs *before* the cap branch, so the cap must not shadow it.
#[test]
fn quiescence_qply_cap_in_check_mate_returns_mate() {
    // Fool's mate: White to move is already checkmated (Qh4#).
    let mut pos = parse_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        .expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check"
    );

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let ply = 3u32;
    let out = quiescence(&mut pos, ply, MAX_QPLY, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    assert_eq!(
        out,
        -(MATE - ply as i32),
        "cap must not shadow the early mate detection"
    );
    // Mate is detected before the cap recursion: only the entry node.
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        1,
        "checkmate is scored before any cap recursion"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// At the cap, a position in check whose only evasions are *quiet king
/// moves* (no capture of the checker, no block) must still search them — the
/// mere presence of >1 node proves evasions were explored rather than the
/// king being stand-patted on an attacked square.
#[test]
fn quiescence_qply_cap_in_check_only_quiet_king_move() {
    // White Kg1 in check from Rg8; only the king can move (Ph2 can neither
    // capture nor block). Every evasion is a quiet king step.
    let mut pos = parse_fen("1k4r1/8/8/8/8/8/7P/6K1 w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check"
    );

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let _out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    // Quiet king steps do not change material, so the score equals the static
    // balance — but the node count proves the evasions were searched.
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "cap + in check must search the quiet king evasions (not stand pat)"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// At the cap, a position in check whose only evasions are *blocks* (the
/// king has no safe square and the checker cannot be captured) must still
/// search them — the block interposes and the score reflects the searched
/// line, not a stand-pat on the attacked king.
#[test]
fn quiescence_qply_cap_in_check_only_block() {
    // White Kh1 in check from Rh8; Rg8 covers every king flight (g1, g2),
    // Rh8 covers h2, so the king cannot move. The only evasions are queen
    // blocks on the h-file (Qh2 / Qh5).
    let mut pos = parse_fen("1k4rr/8/8/8/8/8/4Q3/7K w - - 0 1").expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check"
    );

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();
    let _out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, BETA, &ctx, &limits).expect("not stopped");

    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 1,
        "cap + in check must search the blocking evasions (not stand pat)"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// At the cap + in check, an exhaustion of the node budget mid-loop must
/// abort (`None`) and leave the board byte-for-byte intact — every made move
/// on the aborted path is unmade on the way out.
#[test]
fn quiescence_qply_cap_child_interrupt_recovery() {
    // Same in-check-with-capturable-checker position as the main test, so it
    // has multiple evasions to explore.
    let fen = "6k1/8/8/8/8/8/R5r1/6K1 w - - 0 1";
    let mut pos = parse_fen(fen).expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        pos.is_in_check(pos.side_to_move()),
        "test premise: White is in check"
    );

    let ctx = fresh_ctx();
    // Budget of 2: the cap entry node (1) plus exactly one searched
    // evasion (2); the next child cannot acquire a node and aborts.
    let limits = SearchLimits {
        depth: None,
        nodes: Some(2),
    };
    let out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, BETA, &ctx, &limits);

    assert!(out.is_none(), "exhausted node budget must abort (None)");
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        2,
        "exactly the budget is consumed"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");
}

/// Secondary issue from review: the non-check cap branch used to return a raw
/// `evaluate(pos)`, which could fall below `alpha` or above `beta` and break
/// the fail-hard contract the rest of the tree relies on. At the cap a
/// non-check node must honour fail-hard: a stand-pat that meets/exceeds beta
/// returns beta, and no child is searched (single node).
#[test]
fn quiescence_qply_cap_noncheck_respects_fail_hard() {
    // Non-check position with a positive stand-pat (+120: knight vs 2 pawns).
    let fen = "6k1/8/3p4/4p3/8/5N2/8/6K1 w - - 0 1";
    let mut pos = parse_fen(fen).expect("valid FEN");
    let before = to_fen(&pos);
    assert!(
        !pos.is_in_check(pos.side_to_move()),
        "test premise: White is NOT in check"
    );

    let ctx = fresh_ctx();
    let limits = SearchLimits::default();

    // beta clamp below stand-pat: the cap must return beta, not stand_pat.
    let out = quiescence(&mut pos, 0, MAX_QPLY, ALPHA, 0, &ctx, &limits).expect("not stopped");
    assert_eq!(
        out, 0,
        "non-check cap must return beta on a stand-pat that meets it, got {}",
        out
    );
    // No child searched at the cap: only the entry node.
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        1,
        "the non-check cap must not search any child"
    );
    assert_eq!(to_fen(&pos), before, "position must be untouched");

    // Lower-bound clamp: stand-pat below alpha returns alpha (not stand_pat).
    // Use a fresh context so its node count is independent of the first call
    // (the two segments must not share / accumulate counters).
    let ctx2 = fresh_ctx();
    let mut pos = parse_fen(fen).expect("valid FEN");
    let out = quiescence(&mut pos, 0, MAX_QPLY, 200, BETA, &ctx2, &limits).expect("not stopped");
    assert_eq!(
        out, 200,
        "non-check cap must return alpha when stand-pat is below it, got {}",
        out
    );
}

/// Regression: adding quiescence at the leaves must not break a plain
/// mate-in-one found by iterative deepening, and the returned move / score
/// must still be the mate.
#[test]
fn regression_search_still_finds_mate_in_one() {
    let mut pos = parse_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").expect("valid FEN");
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(4),
        ..Default::default()
    };
    let outcome =
        chess_engine_demo::engine::search::search_best_move(&mut pos, &limits, &ctx).expect("move");
    assert_eq!(
        chess_engine_demo::chess::move_to_uci(outcome.best_move),
        "a1a8",
        "the only mate-in-one is Ra8"
    );
    assert!(
        outcome.score.expect("real score") > MATE - 1000,
        "a mate-in-one must still score as a mate, got {:?}",
        outcome.score
    );
    // Sanity: the position is intact for the caller.
    assert!(!generate_legal_moves(&mut pos).is_empty());
}
