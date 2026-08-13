//! EVAL 1A — tapered material/PST and King middlegame/endgame PST tests.

use eureka::chess::{generate_legal_moves, move_to_uci, parse_fen, to_fen};
use eureka::engine::evaluate;

#[test]
fn king_pst_centralises_the_endgame_king() {
    let edge = parse_fen("k7/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let centre = parse_fen("k7/8/8/8/4K3/8/8/8 w - - 0 1").unwrap();

    assert!(
        evaluate(&centre) > evaluate(&edge),
        "endgame King PST must prefer centralisation"
    );
}

#[test]
fn king_pst_is_shelter_seeking_in_the_middlegame() {
    let sheltered = parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    let exposed =
        parse_fen("rnbq1bnr/pppppppp/4k3/8/4K3/8/PPPPPPPP/RNBQ1BNR w KQkq - 0 1").unwrap();

    assert!(
        evaluate(&sheltered) > evaluate(&exposed),
        "sheltered={} exposed={}",
        evaluate(&sheltered),
        evaluate(&exposed)
    );
}

#[test]
fn tapered_interpolation_is_monotonic_between_mg_and_eg() {
    let queen_position = parse_fen("k7/8/8/8/4K3/8/8/3Q4 w - - 0 1").unwrap();
    let pawn_position = parse_fen("k7/8/8/8/4K3/8/8/P7 w - - 0 1").unwrap();

    assert!(evaluate(&queen_position) > evaluate(&pawn_position));
}

#[test]
fn eval_is_read_only_after_tapered_calculation() {
    let position = parse_fen("k7/8/8/8/4K3/8/8/3Q4 w - - 0 1").unwrap();
    let before = to_fen(&position);
    let _ = evaluate(&position);
    assert_eq!(to_fen(&position), before);
}

#[test]
fn eval_returns_to_the_same_value_after_make_unmake() {
    let mut position =
        parse_fen("r3k2r/ppp2ppp/2n1b3/3qp3/3PP3/2N1B3/PPP2PPP/R3K2R w KQkq - 0 1").unwrap();
    let before_fen = to_fen(&position);
    let before_eval = evaluate(&position);
    let moves = generate_legal_moves(&mut position);

    assert!(!moves.is_empty());
    for mv in moves {
        let undo = position.make_move(mv);
        let _child_eval = evaluate(&position);
        position.unmake_move(undo);

        assert_eq!(
            to_fen(&position),
            before_fen,
            "move {} changed the root",
            move_to_uci(mv)
        );
        assert_eq!(
            evaluate(&position),
            before_eval,
            "move {} changed the eval",
            move_to_uci(mv)
        );
    }
}
