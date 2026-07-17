//! M1.2: real threading + stop.
//!
//! Two layers of coverage:
//!  - a unit test that runs the search on its own thread and flips the
//!    stop flag from a *different* thread (not a preset flag), proving the
//!    search honours an async abort and unwinds promptly;
//!  - integration tests that drive the actual engine binary over UCI and
//!    exercise the command lifecycle: go/stop, go/position/go, go/go,
//!    go/quit, and isready-during-search.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use chess_engine_demo::chess::types::START_FEN;
use chess_engine_demo::chess::{generate_legal_moves, parse_fen, to_fen};
use chess_engine_demo::engine::search::{search_best_move, SearchContext, SearchLimits};

/// The search must honour a stop flag flipped from another thread, not just
/// a flag that was preset before it started.
#[test]
fn search_honours_stop_flipped_from_another_thread() {
    let pos = parse_fen(START_FEN).unwrap();
    let before = to_fen(&pos);
    let stop = Arc::new(AtomicBool::new(false));
    // Share the context via an Arc so the test thread can inspect the node
    // counter while the search thread holds its own cloned handle.
    let ctx = Arc::new(SearchContext::new(stop.clone()));
    let limits = SearchLimits {
        depth: Some(64),
        ..Default::default()
    };
    // The search thread gets its own clone of the position and the context,
    // exactly like M1.2's spawn_search.
    let thread_ctx = ctx.clone();
    let mut thread_pos = pos;
    let handle = thread::spawn(move || search_best_move(&mut thread_pos, &limits, &thread_ctx));

    // Spin until the search has actually consumed some nodes, then flip the
    // flag from THIS test thread (a different thread than the search).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while ctx.nodes.load(Ordering::Relaxed) < 100 {
        if std::time::Instant::now() > deadline {
            break;
        }
        thread::yield_now();
    }
    assert!(
        ctx.nodes.load(Ordering::Relaxed) > 0,
        "search should have started before we stop it"
    );
    stop.store(true, Ordering::SeqCst);

    // join must return promptly: the search must unwind on the flag, not
    // grind through to completion.
    let join_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let outcome = handle.join().expect("search thread must not panic");
    assert!(
        std::time::Instant::now() <= join_deadline,
        "join should return promptly once stop is set"
    );

    let outcome = outcome.expect("a legal best move must be returned");
    assert!(
        generate_legal_moves(&mut pos.clone()).contains(&outcome.best_move),
        "best move must be legal at the search root"
    );
    // The root position must be untouched: the search held its own clone.
    assert_eq!(to_fen(&pos), before, "root position must be unchanged");
}

// ---------------------------------------------------------------------------
// Integration tests: drive the real engine binary over the UCI protocol.
// ---------------------------------------------------------------------------

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// Path to the built engine binary, provided by Cargo at test runtime
/// (the `CARGO_BIN_EXE_*` env var is set when integration tests run,
/// but not at compile time, so we read it at runtime).
fn engine_path() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("CARGO_BIN_EXE_chess-engine-demo")
            .expect("CARGO_BIN_EXE_chess-engine-demo must be set by cargo"),
    )
}

/// Spawn the engine with piped stdin/stdout. Returns the child plus its
/// stdin handle (to feed commands) and raw stdout (to drain in a reader).
fn spawn_engine() -> (
    std::process::Child,
    std::process::ChildStdin,
    std::process::ChildStdout,
) {
    let mut child = Command::new(engine_path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("engine binary must run");
    let stdin = child.stdin.take().expect("stdin should be piped");
    let stdout = child.stdout.take().expect("stdout should be piped");
    (child, stdin, stdout)
}

/// Drain the engine's stdout into a channel so tests can poll with a timeout
/// (std's `read_line` has no deadline of its own).
fn spawn_reader(stdout: std::process::ChildStdout) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut r = BufReader::new(stdout);
        let mut buf = String::new();
        loop {
            buf.clear();
            match r.read_line(&mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    let _ = tx.send(buf.trim_end().to_string());
                }
                Err(_) => break,
            }
        }
    });
    rx
}

/// Collect every `bestmove` line seen on the channel until the engine exits
/// or the timeout elapses.
fn count_bestmoves(rx: &mpsc::Receiver<String>, timeout: Duration) -> usize {
    let deadline = std::time::Instant::now() + timeout;
    let mut count = 0usize;
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                if line.starts_with("bestmove") {
                    count += 1;
                }
            }
            // Channel closed (engine exited) or a 200ms idle gap.
            Err(_) => break,
        }
    }
    count
}

/// Receive lines until one starts with `prefix`, or the channel closes /
/// the timeout elapses. Returns the matched line.
fn recv_until(rx: &mpsc::Receiver<String>, prefix: &str, timeout: Duration) -> Option<String> {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(line) => {
                if line.starts_with(prefix) {
                    return Some(line);
                }
            }
            // Channel closed (engine exited) or a 200ms idle gap.
            Err(_) => return None,
        }
    }
    None
}

#[test]
fn uci_go_then_stop_emits_bestmove() {
    let (mut child, mut stdin, stdout) = spawn_engine();
    let rx = spawn_reader(stdout);
    stdin.write_all(b"uci\n").unwrap();
    stdin.write_all(b"position startpos\n").unwrap();
    stdin.write_all(b"go depth 64\n").unwrap();
    std::thread::sleep(Duration::from_millis(200));
    stdin.write_all(b"stop\n").unwrap();
    stdin.flush().unwrap();

    // Drain until the `bestmove` the stopped search prints as it unwinds
    // (the channel also carries `id name`/`uciok`/`info` lines first).
    let bm = recv_until(&rx, "bestmove", Duration::from_secs(5));
    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();
    let _ = child.wait();
    let bm = bm.expect("stop must eventually produce a bestmove");
    assert!(
        bm.starts_with("bestmove "),
        "expected a bestmove line, got {:?}",
        bm
    );
}

#[test]
fn uci_isready_during_search_is_immediate() {
    let (mut child, mut stdin, stdout) = spawn_engine();
    let rx = spawn_reader(stdout);
    // `go` starts a search on its own thread immediately.
    stdin.write_all(b"go depth 64\n").unwrap();
    stdin.write_all(b"isready\n").unwrap();
    stdin.flush().unwrap();

    // readyok must arrive BEFORE any bestmove (a bestmove only appears
    // after `stop`). If the engine waited for the search, we'd see the
    // bestmove-or-nothing and miss readyok.
    let first = rx.recv_timeout(Duration::from_secs(5));
    stdin.write_all(b"stop\n").unwrap();
    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();
    let _ = child.wait();

    let first = first.expect("isready must get a reply");
    assert_eq!(
        first, "readyok",
        "isready during a search must answer readyok immediately, not block"
    );
}

#[test]
fn uci_go_then_position_then_go() {
    let (mut child, mut stdin, stdout) = spawn_engine();
    let rx = spawn_reader(stdout);
    stdin.write_all(b"position startpos\n").unwrap();
    stdin.write_all(b"go depth 64\n").unwrap();
    std::thread::sleep(Duration::from_millis(150));
    stdin.write_all(b"position startpos moves e2e4\n").unwrap();
    std::thread::sleep(Duration::from_millis(50));
    stdin.write_all(b"go depth 1\n").unwrap();
    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();

    // The first search is stopped by `position` (emits a bestmove), the
    // second search completes depth 1 on the new position (another
    // bestmove). So we expect at least two.
    let count = count_bestmoves(&rx, Duration::from_secs(8));
    let _ = child.wait();
    assert!(
        count >= 2,
        "go->position->go should yield >=2 bestmoves, got {}",
        count
    );
}

#[test]
fn uci_go_then_go_restart() {
    let (mut child, mut stdin, stdout) = spawn_engine();
    let rx = spawn_reader(stdout);
    stdin.write_all(b"position startpos\n").unwrap();
    stdin.write_all(b"go depth 64\n").unwrap();
    std::thread::sleep(Duration::from_millis(150));
    // A second `go` must stop+join the old search (which prints its
    // bestmove) before starting a fresh one (which prints another).
    stdin.write_all(b"go depth 1\n").unwrap();
    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();

    let count = count_bestmoves(&rx, Duration::from_secs(8));
    let _ = child.wait();
    assert!(
        count >= 2,
        "go->go restart should yield >=2 bestmoves, got {}",
        count
    );
}

#[test]
fn uci_go_then_quit() {
    let (mut child, mut stdin, stdout) = spawn_engine();
    let rx = spawn_reader(stdout);
    stdin.write_all(b"position startpos\n").unwrap();
    stdin.write_all(b"go depth 64\n").unwrap();
    std::thread::sleep(Duration::from_millis(150));
    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();

    let bm = recv_until(&rx, "bestmove", Duration::from_secs(5));
    let status = child.wait().expect("engine should exit");
    let bm = bm.expect("quit must flush the final bestmove");
    assert!(
        bm.starts_with("bestmove "),
        "expected a bestmove line, got {:?}",
        bm
    );
    assert!(status.success(), "engine should exit cleanly");
}
