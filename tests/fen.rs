//! FEN parser hardening (P1): reject malformed input with clear errors
//! instead of panicking, and never accept an impossible board.

use chess_engine_demo::chess::parse_fen;

/// Every entry here must be rejected (return `Err`), never panic.
#[test]
fn parse_fen_rejects_bad_input() {
    let bad = [
        "",                                                               // empty
        "hello",                                                          // not a FEN at all
        "8/8/8/8/8/8/8",                                                  // only 7 ranks
        "8/8/8/8/8/8/8/8/8",                                              // 9 ranks
        "9/8/8/8/8/8/8/8",                                                // run of 9 in a rank
        "8/8/8/8/8/8/8/9",              // run of 9 in the last rank
        "0/8/8/8/8/8/8/8",              // run of 0 is illegal
        "8/8/8/8/8/8/8/8 w KQkq - 0",   // missing fullmove field
        "8/8/8/8/8/8/8/8 w KQkq - 0 0", // fullmove < 1
        "8/8/8/8/8/8/8/RK w - - 0 1",   // no black king
        "8/8/8/8/8/8/8/KK w - - 0 1",   // two white kings
        "8/8/8/8/8/8/8/RN w - - 0 1",   // two rooks, no king
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1", // ep on wrong rank for White
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq e6 0 1", // ep on wrong rank for Black
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", // bad side
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra", // extra field
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1.5", // non-integer fullmove
    ];
    for f in bad {
        assert!(parse_fen(f).is_err(), "FEN should be rejected: {:?}", f);
    }
}

/// A clean FEN with a legal en-passant target is accepted.
#[test]
fn parse_fen_accepts_legal_ep() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1";
    let pos = parse_fen(fen).expect("legal ep target for White to move");
    assert_eq!(
        pos.ep_target(),
        Some(chess_engine_demo::chess::types::parse_square("e6").unwrap())
    );
}

/// parse_fen must never panic, no matter what string it is handed.
#[test]
fn parse_fen_never_panics_on_garbage() {
    let corpus = [
        "",
        "   ",
        "🤖/🤖/🤖/🤖/🤖/🤖/🤖/🤖 w - - 0 1",
        "999999999/8/8/8/8/8/8/8 w - - 0 1",
        "/8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/8/ w - - 0 1",
        "8/8/8/8/8/8/8/RNBQKBNR w - - 0 1",
        "k7/8/8/8/8/8/8/K7 w - - 0 1",
        "8/8/8/8/8/8/8/Rk w - - 0 1 abc def",
        "aaaaaaaa/bbbbbbbb/cccccccc/dddddddd/eeeeeeee/ffffffff/gggggggg/hhhhhhhh w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n",
    ];
    for f in corpus {
        let _ = parse_fen(f); // must not panic
    }
}

/// P0.5 regression: a rank whose run-length exactly fills the 8 squares and
/// is then followed by a piece used to panic on `board[64]`. It must be
/// rejected, never panic.
#[test]
fn parse_fen_does_not_panic_on_overfull_rank() {
    let cases = [
        "8K/8/8/8/8/8/8/k7 w - - 0 1",
        "7KK/8/8/8/8/8/8/k7 w - - 0 1",
        "8p/8/8/8/8/8/8/K6k w - - 0 1",
    ];
    for f in cases {
        // Must not panic on any input.
        let result = std::panic::catch_unwind(|| {
            let _ = parse_fen(f);
        });
        assert!(result.is_ok(), "parse_fen panicked on {:?}", f);
        // And must be reported as an error, not silently accepted.
        assert!(parse_fen(f).is_err(), "FEN should be rejected: {:?}", f);
    }
}

/// v0.1.2 regression: digit run-lengths that overflow the rank must be
/// rejected, never panic. In debug builds the old `file += d as u8`
/// wrapped on input like "8888...8", producing a `u8` overflow panic.
/// Each case has a valid-looking king placement so the only thing wrong is
/// the rank arithmetic.
#[test]
fn parse_fen_digit_overflow_does_not_panic() {
    let malicious = [
        "88/8/8/8/8/8/8/k6K w - - 0 1",
        "888888888888888888888888888888888/8/8/8/8/8/8/k6K w - - 0 1",
        "18181818181818181818181818181818/8/8/8/8/8/8/k6K w - - 0 1",
    ];
    for fen in malicious {
        // catch_unwind guards both behaviours we care about: no panic, AND
        // a clean `Err` (the position is genuinely illegal, not silently
        // accepted as a valid board).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            chess_engine_demo::chess::parse_fen(fen)
        }));
        assert!(
            result.is_ok(),
            "parse_fen panicked on malicious FEN: {}",
            fen
        );
        assert!(
            result.unwrap().is_err(),
            "malicious FEN '{}' should be rejected, not accepted",
            fen
        );
    }
}
