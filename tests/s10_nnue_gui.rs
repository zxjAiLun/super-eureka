//! S10-D GUI integration: configure NNUE entirely through UCI options while
//! keeping search policy independent from evaluator selection. Every search
//! waits for `bestmove` before the RAII cleanup can send `quit`.

#[allow(dead_code)]
mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use common::{spawn_reader, EngineProcess, ReaderHandle};

const MODEL_BYTES: usize = 5_786_544;
const SOURCE_FP32_SHA: [u8; 32] = [
    0x9b, 0xf7, 0xad, 0xdd, 0xf7, 0xb3, 0xb4, 0x4a, 0xff, 0xa5, 0xe2, 0x6d, 0x22, 0x76, 0xb1, 0x3d,
    0x74, 0x56, 0x61, 0x91, 0xa4, 0xeb, 0x4d, 0x00, 0x90, 0xfb, 0xde, 0x5a, 0x7a, 0xfb, 0xc9, 0xfc,
];
const SOURCE_CHECKPOINT_SHA: [u8; 32] = [
    0xd5, 0x9a, 0xd8, 0x52, 0x5c, 0x06, 0xab, 0xe8, 0x03, 0x07, 0xbf, 0xfb, 0x12, 0x1f, 0xf4, 0x97,
    0xa3, 0x6e, 0x94, 0xb1, 0x91, 0xc3, 0xc9, 0xbb, 0x3c, 0x8f, 0x31, 0xe5, 0xcc, 0xe5, 0x50, 0xc7,
];

struct TempEngineDir(PathBuf);

impl TempEngineDir {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "eureka-s10d-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn copy_engine(&self) -> PathBuf {
        let source = common::engine_path();
        let destination = self.0.join(source.file_name().unwrap());
        fs::copy(source, &destination).unwrap();
        destination
    }
}

impl Drop for TempEngineDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_synthetic_model(path: &Path) {
    let mut bytes = Vec::with_capacity(MODEL_BYTES);
    bytes.extend_from_slice(b"EUNN2Q01");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&22_528u32.to_le_bytes());
    bytes.extend_from_slice(&128u32.to_le_bytes());
    bytes.extend_from_slice(&1000.0f32.to_le_bytes());
    bytes.extend_from_slice(&12u32.to_le_bytes());
    bytes.extend_from_slice(&12u32.to_le_bytes());
    bytes.extend_from_slice(&12u32.to_le_bytes());
    bytes.extend_from_slice(&4096u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&SOURCE_FP32_SHA);
    bytes.extend_from_slice(&SOURCE_CHECKPOINT_SHA);
    bytes.resize(MODEL_BYTES, 0);
    fs::write(path, bytes).unwrap();
}

fn spawn_engine_at(path: &Path) -> (EngineProcess, ReaderHandle) {
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("copied engine binary must run");
    let stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let process = EngineProcess {
        child,
        stdin: Some(stdin),
    };
    (process, spawn_reader(stdout))
}

fn recv_through(reader: &ReaderHandle, terminal: &str) -> Vec<String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut lines = Vec::new();
    loop {
        let now = std::time::Instant::now();
        assert!(
            now < deadline,
            "timed out waiting for {terminal}: {lines:?}"
        );
        let line = match reader
            .lines
            .recv_timeout((deadline - now).min(Duration::from_millis(250)))
        {
            Ok(line) => line,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                panic!("engine exited while waiting for {terminal}: {lines:?}")
            }
        };
        let done = line.starts_with(terminal);
        lines.push(line);
        if done {
            return lines;
        }
    }
}

fn bestmove(lines: &[String]) -> &str {
    lines
        .iter()
        .find_map(|line| line.strip_prefix("bestmove "))
        .expect("bestmove output")
}

#[test]
fn no_argument_gui_can_autodiscover_override_and_keep_nnue_across_newgame() {
    let temp = TempEngineDir::new("gui path with spaces");
    let engine = temp.copy_engine();
    let auto_model = temp.0.join("nnue-v2-q01.bin");
    write_synthetic_model(&auto_model);
    let explicit_dir = temp.0.join("explicit model path");
    fs::create_dir(&explicit_dir).unwrap();
    let explicit_model = explicit_dir.join("override.bin");
    write_synthetic_model(&explicit_model);
    let relative_model = temp.0.join("relative-override.bin");
    write_synthetic_model(&relative_model);

    let (mut process, reader) = spawn_engine_at(&engine);
    process.send("uci");
    let handshake = recv_through(&reader, "uciok");
    assert!(handshake.iter().any(|line| {
        line == &format!(
            "option name EvalFile type string default {}",
            auto_model.display()
        )
    }));
    assert!(handshake
        .iter()
        .any(|line| line == "info string evalfile nnue-v2-q01.bin"));
    assert!(handshake
        .iter()
        .any(|line| line == "info string profile current-final"));

    // Auto-loaded model remains inert while Evaluation is classical.
    process.send("position startpos");
    process.send("go depth 2");
    let baseline = recv_through(&reader, "bestmove ");
    let baseline_move = bestmove(&baseline).to_string();
    assert_ne!(baseline_move, "0000");

    // Configure in the GUI-permitted order: evaluation first, then a string path.
    process.send("setoption name Evaluation value nnue");
    process.send(&format!(
        "setoption name EvalFile value {}",
        explicit_model.display()
    ));
    let loaded = recv_through(&reader, "info string EvalFile loaded:");
    assert_eq!(
        loaded.last().unwrap(),
        "info string EvalFile loaded: override.bin"
    );
    process.send("isready");
    recv_through(&reader, "readyok");
    process.send("position startpos");
    process.send("go depth 2");
    assert_ne!(bestmove(&recv_through(&reader, "bestmove ")), "0000");

    // A GUI relative EvalFile follows the same executable-directory-first
    // rule as startup CLI and bench, even when the test runner CWD differs.
    process.send("setoption name EvalFile value relative-override.bin");
    let relative_loaded = recv_through(&reader, "info string EvalFile loaded:");
    assert_eq!(
        relative_loaded.last().unwrap(),
        "info string EvalFile loaded: relative-override.bin"
    );
    process.send("position startpos");
    process.send("go depth 2");
    assert_ne!(bestmove(&recv_through(&reader, "bestmove ")), "0000");

    // SearchProfile remains immutable and its reported identity is unchanged.
    process.send("setoption name SearchProfile value current");
    process.send("uci");
    let second_handshake = recv_through(&reader, "uciok");
    assert!(second_handshake
        .iter()
        .any(|line| line == "info string profile current-final"));

    // ucinewgame resets board/TT only; the selected evaluator survives.
    process.send("ucinewgame");
    process.send("position startpos");
    process.send("go depth 2");
    assert_ne!(bestmove(&recv_through(&reader, "bestmove ")), "0000");

    // Switching back off restores the exact deterministic classical result.
    process.send("setoption name Evaluation value classical");
    process.send("ucinewgame");
    process.send("position startpos");
    process.send("go depth 2");
    assert_eq!(bestmove(&recv_through(&reader, "bestmove ")), baseline_move);
}

#[test]
fn enabled_mode_without_neighbor_or_evalfile_refuses_search() {
    let temp = TempEngineDir::new("no-model");
    let engine = temp.copy_engine();
    let (mut process, reader) = spawn_engine_at(&engine);
    process.send("uci");
    let handshake = recv_through(&reader, "uciok");
    assert!(handshake
        .iter()
        .any(|line| line == "option name EvalFile type string default <empty>"));

    process.send("setoption name Evaluation value invalid-mode");
    process.send("isready");
    let invalid = recv_through(&reader, "readyok");
    assert!(invalid
        .iter()
        .any(|line| line == "info string invalid Evaluation value (expected classical|nnue)"));

    process.send("setoption name Evaluation value nnue");
    process.send("position startpos");
    process.send("go depth 2");
    let output = recv_through(&reader, "bestmove ");
    assert!(output.iter().any(|line| {
        line == "info string Evaluation=nnue requires a loadable EvalFile; refusing to search"
    }));
    assert_eq!(bestmove(&output), "0000");
}
