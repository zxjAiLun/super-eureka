//! FEN round-trip and basic rules sanity checks.

use chess_engine_demo::chess::fen::{parse_fen, to_fen, print_ascii};
use chess_engine_demo::chess::position::Position;
use chess_engine_demo::chess::types::{move_to_uci, START_FEN, MoveFlag, Piece, PieceType, Color, Move};

#[test]
fn fen_round_trip_startpos() {
    let pos = parse_fen(START_FEN).unwrap();
    assert_eq!(to_fen(&pos), START_FEN);
}

#[test]
fn startpos_has_32_pieces() {
    let pos = parse_fen(START_FEN).unwrap();
    let count = pos.board.iter().filter(|c| c.is_some()).count();
    assert_eq!(count, 32);
}

#[test]
fn startpos_white_to_move() {
    let pos = parse_fen(START_FEN).unwrap();
    assert_eq!(pos.side, Color::White);
    assert!(pos.castling.white_king && pos.castling.white_queen);
    assert!(pos.castling.black_king && pos.castling.black_queen);
    assert_eq!(pos.ep_target, None);
}

#[test]
fn move_to_uci_promotion() {
    let m = Move {
        from: 12, // e2
        to: 4,   // e1 (promotion square in a contrived sense)
        promotion: Some(PieceType::Queen),
        flag: MoveFlag::Promotion(PieceType::Queen),
    };
    assert_eq!(move_to_uci(m), "e2e1q");
}

#[test]
fn simple_make_unmake_restores_board() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let undo = pos.make_move(Move {
        from: 12, // e2
        to: 28,   // e4
        promotion: None,
        flag: MoveFlag::DoublePawnPush,
    });
    assert_eq!(pos.board[28 as usize], Some(Piece::new(Color::White, PieceType::Pawn)));
    assert_eq!(pos.board[12 as usize], None);
    assert_eq!(pos.side, Color::Black);
    pos.unmake_move(undo);
    assert_eq!(to_fen(&pos), START_FEN);
}

#[test]
fn ascii_print_does_not_panic() {
    let pos = parse_fen(START_FEN).unwrap();
    print_ascii(&pos); // just ensure it runs
}
