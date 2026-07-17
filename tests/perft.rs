//! Perft tests — the Phase-2 gate. If any of these fail, movegen has a
//! rules bug and we must NOT proceed to search.

use chess_engine_demo::chess::fen::parse_fen;
use chess_engine_demo::chess::types::START_FEN;

fn perft(fen: &str, depth: u32) -> u64 {
    let mut pos = parse_fen(fen).expect("valid FEN in test");
    pos.perft(depth)
}

#[test]
fn perft_startpos() {
    assert_eq!(perft(START_FEN, 1), 20);
    assert_eq!(perft(START_FEN, 2), 400);
    assert_eq!(perft(START_FEN, 3), 8902);
    assert_eq!(perft(START_FEN, 4), 197281);
}

/// Perft depth 5 (4,865,609 nodes) is the canonical correctness gate, but
/// it is ~300 ms in release and far slower in debug. Gate it so it only runs
/// in release builds (`cargo test --release`), keeping debug CI fast.
#[cfg(not(debug_assertions))]
#[test]
fn perft_startpos_depth_5_release() {
    assert_eq!(perft(START_FEN, 5), 4_865_609);
}

#[test]
fn perft_kiwipete() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    assert_eq!(perft(fen, 1), 48);
    assert_eq!(perft(fen, 2), 2039);
    assert_eq!(perft(fen, 3), 97862);
}

#[test]
fn perft_position_3() {
    let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    assert_eq!(perft(fen, 1), 14);
    assert_eq!(perft(fen, 2), 191);
    assert_eq!(perft(fen, 3), 2812);
    assert_eq!(perft(fen, 4), 43238);
}

#[test]
fn perft_position_4() {
    let fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    assert_eq!(perft(fen, 1), 6);
    assert_eq!(perft(fen, 2), 264);
    assert_eq!(perft(fen, 3), 9467);
}

#[test]
fn perft_position_5() {
    let fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    assert_eq!(perft(fen, 1), 44);
    assert_eq!(perft(fen, 2), 1486);
    assert_eq!(perft(fen, 3), 62379);
}
