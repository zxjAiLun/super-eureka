//! UCI move-string hardening (P1): `find_move` must reject malformed,
//! over-long, and non-ASCII move strings instead of panicking or silently
//! downgrading them.

use chess_engine_demo::chess::fen::parse_fen;
use chess_engine_demo::chess::types::START_FEN;
use chess_engine_demo::uci::find_move;

#[test]
fn find_move_rejects_malformed_move_strings() {
    let mut pos = parse_fen(START_FEN).unwrap();
    let bad = [
        "e2e4x",     // trailing junk after a 4-char move
        "e2e4junk",  // over-long, non-promotion
        "e2e4qjunk", // over-long, looks like promotion but isn't 5 bytes
        "aéa",       // non-ASCII garbage
        "🤖🤖",      // multi-byte UTF-8 (would slice mid-character)
    ];
    for m in bad {
        assert!(find_move(&mut pos, m).is_none(), "should reject {:?}", m);
    }
    // Sanity: a plain legal move is still accepted.
    assert!(find_move(&mut pos, "e2e4").is_some());
}

#[test]
fn find_move_strict_promotion_piece() {
    // White pawn on e7 with e8 empty, ready to promote.
    let mut pos = parse_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    // Legal promotion targets must be accepted.
    for p in ["e7e8q", "e7e8r", "e7e8b", "e7e8n"] {
        assert!(find_move(&mut pos, p).is_some(), "should accept {}", p);
    }
    // 'k' and 'p' are never valid promotion pieces and must be rejected
    // rather than silently downgraded.
    for p in ["e7e8k", "e7e8p", "e7e8x", "e7e8qjunk"] {
        assert!(find_move(&mut pos, p).is_none(), "should reject {}", p);
    }
}
