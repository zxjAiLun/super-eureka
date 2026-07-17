//! Shared helpers for UCI integration tests that drive the real engine
//! binary over piped stdin/stdout. This module is included via `mod common;`
//! in integration test files; it is not itself a test target.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

/// Path to the built engine binary. `CARGO_BIN_EXE_*` is set by Cargo at
/// test runtime (not compile time), so read it at runtime.
pub fn engine_path() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("CARGO_BIN_EXE_chess-engine-demo")
            .expect("CARGO_BIN_EXE_chess-engine-demo must be set by cargo"),
    )
}

/// Spawn the engine with piped stdin/stdout. Returns the child plus its
/// stdin writer and raw stdout reader.
pub fn spawn_engine() -> (
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

/// Drain the engine's stdout into a channel so tests can poll with a
/// timeout (std's `read_line` has no deadline of its own).
pub fn spawn_reader(stdout: std::process::ChildStdout) -> mpsc::Receiver<String> {
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

/// Receive lines until one starts with `prefix`, or the channel closes /
/// the timeout elapses. Returns the matched line.
pub fn recv_until(rx: &mpsc::Receiver<String>, prefix: &str, timeout: Duration) -> Option<String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            return None;
        }
        // Wait at most 200ms, OR whatever time is left before the overall
        // timeout — whichever is smaller. A 200ms idle gap is NOT the end:
        // we keep polling until the deadline, so "wait up to 3s" really
        // waits up to 3s (this matters on slow CI / heavy-load machines
        // where a 200ms-only wait produced flaky false failures).
        let wait = std::cmp::min(Duration::from_millis(200), deadline - now);
        match rx.recv_timeout(wait) {
            Ok(line) => {
                if line.starts_with(prefix) {
                    return Some(line);
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            // Engine exited (or the pipe dropped): nothing more will arrive.
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}

/// Send a line to the engine and flush.
pub fn send(stdin: &mut std::process::ChildStdin, line: &str) {
    stdin.write_all(line.as_bytes()).unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.flush().unwrap();
}
