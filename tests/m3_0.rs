//! M3.0 — Game State + Deterministic Zobrist foundation tests (chess-state layer).
//!
//! Covers spec §16.1–§16.7 and §16.9 for the first implementation commit
//! `feat(chess): add deterministic incremental Zobrist state`.
//!
//! Scope note: this file tests the *chess-state* layer only. The following
//! are intentionally NOT covered here (they belong to the deferred
//! `feat(search)` commit):
//!   * §16.8 SearchPath (search-maintained key stack + root-length restore);
//!   * UCI `position ... moves ...` / `ucinewgame` wiring, which is not
//!     implemented until the second commit. The UCI-equivalent invariants
//!     (replace history, reject illegal move, single-element history) ARE
//!     exercised at the `GameState` API level below.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use chess_engine_demo::chess::{
    generate_legal_moves, make_square, move_to_uci, parse_fen, recompute_zobrist, to_fen,
    GameState, Position, START_FEN,
};
use chess_engine_demo::chess::{Color, Move, MoveFlag, Piece, PieceType};
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits};

/// Locked reference value from the spec (§16.1). Determined by the fixed
/// SplitMix64 seed, so it is identical in debug and release.
const STARTPOS_KEY: u64 = 0x61A2_181F_8F2F_9B9C;

fn fresh_ctx() -> Arc<SearchContext> {
    Arc::new(SearchContext::new(Arc::new(AtomicBool::new(false))))
}

/// Find a specific legal move by its UCI string. Panics if the move is not
/// legal in the position (test authoring error, not an engine bug).
fn find_move(pos: &Position, uci_str: &str) -> Move {
    let mut p = *pos;
    let moves = generate_legal_moves(&mut p);
    moves
        .into_iter()
        .find(|m| move_to_uci(*m) == uci_str)
        .unwrap_or_else(|| panic!("move {} not legal in {}", uci_str, to_fen(pos)))
}

// ---------------------------------------------------------------------------
// §16.1 确定性与初始化
// ---------------------------------------------------------------------------

#[test]
fn determinism_and_startpos_key() {
    // 1. startpos key == locked reference value.
    assert_eq!(Position::startpos().zobrist_key(), STARTPOS_KEY);

    // 2. Same FEN re-parsed -> identical key.
    let a = parse_fen(START_FEN).unwrap();
    let b = parse_fen(START_FEN).unwrap();
    assert_eq!(a.zobrist_key(), b.zobrist_key());
    assert_eq!(a.zobrist_key(), STARTPOS_KEY);

    // 3. pos.zobrist_key() == recompute_zobrist(&pos).
    assert_eq!(a.zobrist_key(), recompute_zobrist(&a));

    // 4. debug/release share the same fixed value — guaranteed by the const-fn
    //    derivation; re-parsing twice in the same run must agree.
    let c = parse_fen(START_FEN).unwrap();
    assert_eq!(c.zobrist_key(), STARTPOS_KEY);
}

// ---------------------------------------------------------------------------
// §16.2 Hash component (single-field changes)
// ---------------------------------------------------------------------------

#[test]
fn hash_components_differ_or_same() {
    // side to move changes the key.
    let w = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let b = parse_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();
    assert_ne!(w.zobrist_key(), b.zobrist_key(), "side to move changes key");

    // a piece on a different square changes the key.
    let a = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap(); // K e1
    let c = parse_fen("4k3/8/8/8/8/8/8/3K4 w - - 0 1").unwrap(); // K d1
    assert_ne!(a.zobrist_key(), c.zobrist_key(), "piece square changes key");

    // a different piece type on the same square changes the key.
    // `a` has the white King on e1; `d` has a white Knight on e1
    // (white King moved to a1 so the position stays legal).
    let a = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap(); // K e1
    let d = parse_fen("4k3/8/8/8/8/8/8/K3N3 w - - 0 1").unwrap(); // K a1, N e1
    assert_ne!(a.zobrist_key(), d.zobrist_key(), "piece type changes key");

    // a different piece color on the same square changes the key.
    // `a` has the white King on e1; `e` has a black king on e1
    // (white King moved to a1 so the position stays legal).
    let a = parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap(); // white K e1
    let e = parse_fen("8/8/8/8/8/8/8/K3k3 w - - 0 1").unwrap(); // white K a1, black k e1
    assert_ne!(a.zobrist_key(), e.zobrist_key(), "piece color changes key");

    // castling rights change the key.
    let a = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    let f = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1").unwrap();
    assert_ne!(
        a.zobrist_key(),
        f.zobrist_key(),
        "castling rights change key"
    );

    // a legal en-passant file changes the key.
    let with_ep = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let no_ep = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1").unwrap();
    assert_ne!(
        with_ep.zobrist_key(),
        no_ep.zobrist_key(),
        "effective EP file changes key"
    );

    // halfmove clock must NOT change the key.
    let a = parse_fen(START_FEN).unwrap();
    let b = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 5 1").unwrap();
    assert_eq!(
        a.zobrist_key(),
        b.zobrist_key(),
        "halfmove does NOT change key"
    );

    // fullmove number must NOT change the key.
    let a = parse_fen(START_FEN).unwrap();
    let b = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 10").unwrap();
    assert_eq!(
        a.zobrist_key(),
        b.zobrist_key(),
        "fullmove does NOT change key"
    );

    // an unexecutable EP target (no adjacent pawn) must NOT change the key.
    let a = parse_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1").unwrap();
    let b = parse_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").unwrap();
    assert_eq!(
        a.zobrist_key(),
        b.zobrist_key(),
        "unexecutable EP target does NOT change key"
    );
}

// ---------------------------------------------------------------------------
// §16.3 En-passant — the five fixtures from spec §6.3
// ---------------------------------------------------------------------------

#[test]
fn ep_five_cases() {
    // 1. legal EP: key must differ from the `-` version.
    let with = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let without = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1").unwrap();
    assert_ne!(
        with.zobrist_key(),
        without.zobrist_key(),
        "legal EP must change key"
    );

    // 2. no adjacent pawn: key same as `-`.
    let with = parse_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1").unwrap();
    let without = parse_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1").unwrap();
    assert_eq!(
        with.zobrist_key(),
        without.zobrist_key(),
        "no adjacent pawn -> normalized key"
    );

    // 3. pinned EP (self-check): key same as `-`.
    let with = parse_fen("4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let without = parse_fen("4r1k1/8/8/3pP3/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(
        with.zobrist_key(),
        without.zobrist_key(),
        "pinned EP -> normalized key"
    );

    // 4. EP target occupied: parse_fen must error.
    let r = parse_fen("4k3/8/3n4/3pP3/8/8/8/4K3 w - d6 0 1");
    assert!(r.is_err(), "occupied EP target must error");
    let msg = r.unwrap_err();
    assert!(
        msg.contains("en passant target square must be empty"),
        "got error message: {}",
        msg
    );

    // 5. no enemy pawn on the captured square: normalized key. This is the
    //    user-fixed fixture; assert its preconditions explicitly.
    let with = parse_fen("4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1").unwrap();
    assert_eq!(
        with.board()[make_square(4, 4) as usize],
        Some(Piece::new(Color::White, PieceType::Pawn)),
        "fixture precondition: White pawn on e5"
    );
    assert_eq!(
        with.board()[make_square(3, 4) as usize],
        None,
        "fixture precondition: d5 must be empty"
    );
    let without = parse_fen("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1").unwrap();
    assert_eq!(
        with.zobrist_key(),
        without.zobrist_key(),
        "no enemy pawn on capture square -> normalized key"
    );
}

// ---------------------------------------------------------------------------
// §16.4 Make / unmake special moves
// ---------------------------------------------------------------------------

#[test]
fn make_unmake_roundtrip_per_move_type() {
    // (fen, uci) covering every special move category.
    let cases: &[(&str, &str)] = &[
        (START_FEN, "g1f3"), // quiet
        (
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            "e4d5",
        ), // capture
        (START_FEN, "e2e4"), // double pawn push
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"), // en passant
        // Castling from the start position is NOT legal (f1/g1 are occupied
        // by B/N), so the castling cases use a clear-path back rank.
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"), // white kingside castle
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"), // black queenside castle
        ("8/P7/8/8/8/8/K7/7k w - - 0 1", "a7a8q"),        // quiet promotion
        ("1n6/P7/8/8/8/8/8/k6K w - - 0 1", "a7b8q"),      // capture promotion
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1e2"), // king move loses castling
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "h1g1"), // rook move loses castling
        ("4k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1", "h8h1"),   // capture corner rook
    ];

    for (fen, uci) in cases {
        let mut pos = parse_fen(fen).expect("valid fen");
        let before_fen = to_fen(&pos);
        let before_key = pos.zobrist_key();

        let mv = find_move(&pos, uci);
        let undo = pos.make_move(mv);
        assert_eq!(
            pos.zobrist_key(),
            recompute_zobrist(&pos),
            "incremental key must equal recompute after {} in {}",
            uci,
            fen
        );

        pos.unmake_move(undo);
        assert_eq!(
            to_fen(&pos),
            before_fen,
            "FEN must be restored after {} in {}",
            uci,
            fen
        );
        assert_eq!(
            pos.zobrist_key(),
            before_key,
            "key must be restored after {} in {}",
            uci,
            fen
        );
    }
}

/// §16.4 (last three bullet points): a King move, a Rook move, and
/// capturing the enemy's original corner rook each drop the relevant castling
/// right. This exercises the `update_castling_rights` path that the
/// make/unmake round-trip above only covers implicitly.
#[test]
fn castling_rights_lost_on_king_rook_and_corner_capture() {
    // King move drops BOTH white castling rights.
    let mut pos = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    assert!(
        pos.castling_rights().white_king && pos.castling_rights().white_queen,
        "precondition: both white rights present"
    );
    let mv = find_move(&pos, "e1e2");
    pos.make_move(mv);
    assert!(
        !pos.castling_rights().white_king && !pos.castling_rights().white_queen,
        "king move must drop both white castling rights"
    );

    // Rook move drops only the matching side.
    let mut pos = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    let mv = find_move(&pos, "h1g1");
    pos.make_move(mv);
    assert!(
        !pos.castling_rights().white_king,
        "kingside rook move must drop white kingside"
    );
    assert!(
        pos.castling_rights().white_queen,
        "kingside rook move must keep white queenside"
    );

    // Capturing the enemy corner rook drops the enemy's matching right.
    let mut pos = parse_fen("4k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1").unwrap();
    assert!(
        pos.castling_rights().white_king,
        "precondition: white kingside present before capture"
    );
    let mv = find_move(&pos, "h8h1");
    pos.make_move(mv);
    assert!(
        !pos.castling_rights().white_king,
        "capturing h1 rook must drop white kingside"
    );
}

// ---------------------------------------------------------------------------
// §16.5 Multi-level recursive hash walk
// ---------------------------------------------------------------------------

fn hash_walk(pos: &mut Position, depth: u32) {
    assert_eq!(pos.zobrist_key(), recompute_zobrist(pos));
    if depth == 0 {
        return;
    }
    for mv in generate_legal_moves(pos) {
        let before_fen = to_fen(pos);
        let before_key = pos.zobrist_key();

        let undo = pos.make_move(mv);
        assert_eq!(pos.zobrist_key(), recompute_zobrist(pos));

        hash_walk(pos, depth - 1);

        pos.unmake_move(undo);
        assert_eq!(to_fen(pos), before_fen, "FEN restored by walk");
        assert_eq!(pos.zobrist_key(), before_key, "key restored by walk");
    }
}

#[test]
fn hash_walk_startpos_depth3() {
    let mut pos = parse_fen(START_FEN).unwrap();
    hash_walk(&mut pos, 3);
}

#[test]
fn hash_walk_castling_depth2() {
    let mut pos = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    hash_walk(&mut pos, 2);
}

#[test]
fn hash_walk_enpassant_depth2() {
    let mut pos = parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    hash_walk(&mut pos, 2);
}

#[test]
fn hash_walk_promotion_depth2() {
    let mut pos = parse_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").unwrap();
    hash_walk(&mut pos, 2);
}

// ---------------------------------------------------------------------------
// §16.6 Transposition
// ---------------------------------------------------------------------------

fn apply_uci(pos: &Position, uci_str: &str) -> Position {
    let mut p = *pos;
    let mv = find_move(&p, uci_str);
    p.make_move(mv);
    p
}

#[test]
fn transposition_two_orders_same_final_key() {
    // Two different move orders reaching the same final position.
    let root = parse_fen(START_FEN).unwrap();
    let a = apply_uci(
        &apply_uci(&apply_uci(&apply_uci(&root, "g1f3"), "g8f6"), "b1c3"),
        "b8c6",
    );
    let b = apply_uci(
        &apply_uci(&apply_uci(&apply_uci(&root, "b1c3"), "b8c6"), "g1f3"),
        "g8f6",
    );
    assert_eq!(to_fen(&a), to_fen(&b), "final FEN identical");
    assert_eq!(a.zobrist_key(), b.zobrist_key(), "final key identical");

    // GameState: histories differ, current key is the same.
    let mut gs_a = GameState::from_position(root);
    for u in ["g1f3", "g8f6", "b1c3", "b8c6"] {
        let mv = find_move(gs_a.position(), u);
        gs_a.apply_legal_move(mv).unwrap();
    }
    let mut gs_b = GameState::from_position(root);
    for u in ["b1c3", "b8c6", "g1f3", "g8f6"] {
        let mv = find_move(gs_b.position(), u);
        gs_b.apply_legal_move(mv).unwrap();
    }
    assert_eq!(gs_a.current_key(), gs_b.current_key(), "current key same");
    assert_ne!(
        gs_a.key_history(),
        gs_b.key_history(),
        "histories must differ (different move orders)"
    );
    assert_eq!(gs_a.current_key(), a.zobrist_key());
}

// ---------------------------------------------------------------------------
// §16.7 GameState invariants (API level; UCI wiring deferred)
// ---------------------------------------------------------------------------

#[test]
fn gamestate_invariants() {
    // 1. startpos history len == 1.
    let gs = GameState::startpos();
    assert_eq!(gs.key_history().len(), 1);

    // 2. N legal moves -> history len == N + 1.
    let mut gs = GameState::startpos();
    let mut n = 0u32;
    for u in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        let mv = find_move(gs.position(), u);
        gs.apply_legal_move(mv).unwrap();
        n += 1;
    }
    assert_eq!(gs.key_history().len() as u32, n + 1);

    // 3. history.last == current key.
    assert_eq!(gs.key_history().last().copied(), Some(gs.current_key()));

    // 4. from_position (no moves) -> len 1, and history[0] is the FEN root key.
    let custom = parse_fen("8/8/8/8/8/8/8/4K2k w - - 0 1").unwrap();
    let gs = GameState::from_position(custom);
    assert_eq!(gs.key_history().len(), 1);
    assert_eq!(gs.key_history()[0], custom.zobrist_key());

    // 5. a move appends from the FEN root: history[0] stays the root key.
    let mut gs = GameState::from_position(custom);
    let mv = find_move(gs.position(), "e1e2");
    gs.apply_legal_move(mv).unwrap();
    assert_eq!(gs.key_history().len(), 2);
    assert_eq!(gs.key_history()[0], custom.zobrist_key());

    // 6. building a new GameState replaces the old history (does not append).
    let other = GameState::from_position(parse_fen(START_FEN).unwrap());
    assert_eq!(other.key_history().len(), 1);
    assert_ne!(
        other.current_key(),
        gs.current_key(),
        "fresh GameState must not inherit the previous history"
    );

    // 7/8. an illegal move leaves position, key, and history fully unchanged.
    let mut gs = GameState::startpos();
    let before_fen = to_fen(gs.position());
    let before_key = gs.current_key();
    let before_history: Vec<u64> = gs.key_history().to_vec();
    let illegal = Move {
        from: make_square(3, 0), // d1
        to: make_square(3, 2),   // d3 (blocked by pawn in startpos)
        promotion: None,
        flag: MoveFlag::Normal,
    };
    let res = gs.apply_legal_move(illegal);
    assert!(res.is_err(), "illegal move must be rejected");
    assert_eq!(
        to_fen(gs.position()),
        before_fen,
        "position unchanged after illegal move"
    );
    assert_eq!(
        gs.current_key(),
        before_key,
        "key unchanged after illegal move"
    );
    assert_eq!(
        gs.key_history(),
        &before_history[..],
        "history unchanged after illegal move"
    );
}

// ---------------------------------------------------------------------------
// §16.9 Behavior zero-change (M3.0 must not alter search results)
// ---------------------------------------------------------------------------

#[test]
fn behavior_zero_change_startpos_depth3() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");

    assert_eq!(out.completed_depth, 3);
    assert!(!out.stopped);
    assert_eq!(to_fen(&pos), before, "root position restored");
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        1149,
        "startpos depth-3 node baseline (PST era)"
    );
    assert_eq!(move_to_uci(out.best_move), "b1c3");
    assert_eq!(out.score, Some(50));

    // Full PV vector (not just pv[0]) must match the M2.4 baseline.
    let pv: Vec<String> = out.pv.iter().map(|&m| move_to_uci(m)).collect();
    assert_eq!(pv, vec!["b1c3", "b8c6", "g1f3"]);
}

#[test]
fn behavior_zero_change_queenwin_depth3() {
    let mut pos = parse_fen("7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1").unwrap();
    let before = to_fen(&pos);
    let ctx = fresh_ctx();
    let limits = SearchLimits {
        depth: Some(3),
        ..Default::default()
    };
    let out = search_best_move(&mut pos, &limits, &ctx).expect("outcome");

    assert_eq!(out.completed_depth, 3);
    assert!(!out.stopped);
    assert_eq!(to_fen(&pos), before, "root position restored");
    assert_eq!(
        ctx.nodes.load(Ordering::Relaxed),
        963,
        "queen-win depth-3 node baseline (PST era)"
    );
    assert_eq!(move_to_uci(out.best_move), "e4a4");
    assert_eq!(out.score, Some(890));

    let pv: Vec<String> = out.pv.iter().map(|&m| move_to_uci(m)).collect();
    assert_eq!(pv, vec!["e4a4", "h4h3", "a4h4", "h8g8", "h4h3"]);
}
