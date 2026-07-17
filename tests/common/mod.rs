//! Shared helpers for UCI integration tests that drive the real engine
//! binary over piped stdin/stdout. This module is included via `mod common;`
//! in integration test files; it is not itself a test target.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
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

/// RAII wrapper around a spawned engine child process.
///
/// Why this exists: integration tests routinely `assert!` mid-flight (e.g.
/// "there must be no bestmove before `stop`"). If an assertion fails, the
/// remaining `stop` / `quit` / `wait` lines never run, and an `go infinite`
/// child keeps searching forever in the background. On Windows those leaked
/// processes hold an open handle to `target/.../chess-engine-demo.exe`, so
/// the *next* build's linker cannot overwrite the file (`os error 5` /
/// `LNK1104`). Cleaning up in `Drop` guarantees the child is always reaped —
/// even when the test panics — which removes the root cause of the recurring
/// stale-exe lock.
pub struct EngineProcess {
    pub child: Child,
    /// `Option` so `Drop` can `take()` it, closing the stdin pipe (which
    /// also signals the engine's read loop to end).
    pub stdin: Option<ChildStdin>,
}

impl EngineProcess {
    /// Send one line to the engine and flush.
    pub fn send(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().expect("stdin available");
        stdin.write_all(line.as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        // Ask the engine to quit gracefully, then close the pipe so its read
        // loop sees EOF. Ignore errors: the child may already be gone.
        if let Some(mut stdin) = self.stdin.take() {
            let _ = stdin.write_all(b"quit\n");
            let _ = stdin.flush();
            // `stdin` drops here -> pipe closes.
        }
        // Give it a brief window to exit on its own; then force-kill so a
        // hung `go infinite` can never leak. Either way we `wait()` to reap
        // the process and release its handle on the .exe.
        for _ in 0..50 {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(_) => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawn the engine with piped stdin/stdout. Returns an [`EngineProcess`]
/// (owns the child + stdin, auto-cleaned on drop) plus the raw stdout reader
/// to hand to [`spawn_reader`].
pub fn spawn_engine() -> (EngineProcess, std::process::ChildStdout) {
    let mut child = Command::new(engine_path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("engine binary must run");
    let stdin = child.stdin.take().expect("stdin should be piped");
    let stdout = child.stdout.take().expect("stdout should be piped");
    (
        EngineProcess {
            child,
            stdin: Some(stdin),
        },
        stdout,
    )
}

/// Handle returned by [`spawn_reader`]: a live channel of parsed output
/// lines plus a one-shot `eof` signal fired when the child's stdout closes.
pub struct ReaderHandle {
    pub lines: mpsc::Receiver<String>,
    /// Signalled (with `()`) exactly once, when the reader thread sees the
    /// child's stdout reach EOF (or error out). This is the reliable reaping
    /// signal: a leaked child keeps the pipe open, so `eof` would never fire.
    /// Being able to spawn a *second* instance does NOT prove the first was
    /// reaped (a stale `.exe` handle only blocks the linker, not a fresh
    /// spawn), so tests must assert on `eof`, not on a second instance.
    // Only the `engine_process_is_reaped_when_test_body_panics` test reads
    // this field; the other test crates compile `ReaderHandle` too, so allow
    // the otherwise-unused field here rather than duplicating the struct.
    #[allow(dead_code)]
    pub eof: mpsc::Receiver<()>,
}

/// Drain the engine's stdout into a channel so tests can poll with a
/// timeout (std's `read_line` has no deadline of its own). The returned
/// [`ReaderHandle`] also exposes an `eof` signal when the pipe closes.
pub fn spawn_reader(stdout: std::process::ChildStdout) -> ReaderHandle {
    let (tx, lines) = mpsc::channel();
    let (eof_tx, eof) = mpsc::channel();
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
        // The child has closed its stdout. Signal EOF so a test can assert the
        // process was actually reaped, rather than merely still running.
        let _ = eof_tx.send(());
    });
    ReaderHandle { lines, eof }
}

/// Receive lines until one starts with `prefix`, or the channel closes /
/// the timeout elapses. Returns the matched line.
pub fn recv_until(handle: &ReaderHandle, prefix: &str, timeout: Duration) -> Option<String> {
    let rx = &handle.lines;
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
