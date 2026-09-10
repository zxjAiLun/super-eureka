use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const MODEL_BYTES: usize = 5_786_544;
const SOURCE_FP32_SHA: [u8; 32] = [
    0x9b, 0xf7, 0xad, 0xdd, 0xf7, 0xb3, 0xb4, 0x4a, 0xff, 0xa5, 0xe2, 0x6d, 0x22, 0x76, 0xb1, 0x3d,
    0x74, 0x56, 0x61, 0x91, 0xa4, 0xeb, 0x4d, 0x00, 0x90, 0xfb, 0xde, 0x5a, 0x7a, 0xfb, 0xc9, 0xfc,
];
const SOURCE_CHECKPOINT_SHA: [u8; 32] = [
    0xd5, 0x9a, 0xd8, 0x52, 0x5c, 0x06, 0xab, 0xe8, 0x03, 0x07, 0xbf, 0xfb, 0x12, 0x1f, 0xf4, 0x97,
    0xa3, 0x6e, 0x94, 0xb1, 0x91, 0xc3, 0xc9, 0xbb, 0x3c, 0x8f, 0x31, 0xe5, 0xcc, 0xe5, 0x50, 0xc7,
];

fn write_synthetic_model(path: &std::path::Path) {
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
    std::fs::write(path, bytes).unwrap();
}

fn engine_path() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("CARGO_BIN_EXE_eureka").expect("CARGO_BIN_EXE_eureka must be set by cargo"),
    )
}

fn spawn_engine(args: &[&str]) -> (Child, ChildStdin, BufReader<ChildStdout>) {
    let mut child = Command::new(engine_path())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("engine process must start");
    let stdin = child.stdin.take().expect("engine stdin must be piped");
    let stdout = child.stdout.take().expect("engine stdout must be piped");
    (child, stdin, BufReader::new(stdout))
}

fn read_until(reader: &mut BufReader<ChildStdout>, prefix: &str) -> Vec<String> {
    let mut lines = Vec::new();
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .expect("engine stdout must remain readable");
        assert!(read > 0, "engine exited before emitting {}", prefix);
        let line = line.trim_end().to_string();
        let matched = line.starts_with(prefix);
        lines.push(line);
        if matched {
            return lines;
        }
    }
}

fn assert_handshake_profile(
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    expected: &str,
    expected_eval: Option<&str>,
) {
    stdin.write_all(b"uci\n").unwrap();
    stdin.flush().unwrap();
    let lines = read_until(reader, "uciok");
    assert!(
        lines
            .iter()
            .any(|line| line == &format!("info string profile {}", expected)),
        "handshake must report {}: {:?}",
        expected,
        lines
    );
    if let Some(eval) = expected_eval {
        assert!(
            lines
                .iter()
                .any(|line| line == &format!("info string eval {eval}")),
            "handshake must report eval {eval}: {lines:?}"
        );
    }

    stdin.write_all(b"isready\n").unwrap();
    stdin.flush().unwrap();
    read_until(reader, "readyok");
}

fn run_profile_process(
    args: &[&str],
    expected: &str,
    expected_eval: Option<&str>,
    verify_hot_switch: bool,
) {
    let (mut child, mut stdin, mut reader) = spawn_engine(args);
    assert_handshake_profile(&mut stdin, &mut reader, expected, expected_eval);

    stdin.write_all(b"position startpos\ngo depth 2\n").unwrap();
    stdin.flush().unwrap();
    let bestmove = read_until(&mut reader, "bestmove ");
    assert!(
        !bestmove.is_empty(),
        "fixed-depth search must return bestmove"
    );

    if verify_hot_switch {
        stdin
            .write_all(b"setoption name SearchProfile value current\nuci\n")
            .unwrap();
        stdin.flush().unwrap();
        let second_handshake = read_until(&mut reader, "uciok");
        assert!(
            second_handshake
                .iter()
                .any(|line| line == &format!("info string profile {}", expected)),
            "UCI commands must not hot-switch the startup profile: {:?}",
            second_handshake
        );
    }

    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("engine must be reaped");
    assert!(status.success(), "engine quit must succeed: {}", status);
}

fn assert_startup_rejected_with_message(args: &[&str], expected: &str) {
    let output = Command::new(engine_path())
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("engine process must start");
    assert!(!output.status.success(), "startup must reject {:?}", args);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let diagnostics = format!("{stdout}{stderr}");
    assert!(diagnostics.contains(expected), "{args:?}: {diagnostics}");
    assert!(
        !stdout.lines().any(|line| line == "uciok"),
        "rejected startup must fail before uciok: {:?}",
        args
    );
}

fn assert_startup_rejected(args: &[&str]) {
    assert_startup_rejected_with_message(args, "startup_error");
}

#[test]
fn startup_profile_processes_report_identity_search_and_reject_bad_args() {
    run_profile_process(
        &[],
        "current-final",
        Some("handcrafted-v1+integrated-positional"),
        false,
    );
    run_profile_process(
        &["--profile", "current"],
        "current",
        Some("handcrafted-v1"),
        true,
    );
    run_profile_process(
        &["--profile", "current-final"],
        "current-final",
        Some("handcrafted-v1+integrated-positional"),
        true,
    );

    let model = std::env::temp_dir().join(format!(
        "eureka-s2-profile-model-{}.bin",
        std::process::id()
    ));
    write_synthetic_model(&model);
    run_profile_process(
        &[
            "--profile",
            "current-final-s12",
            "--nnue-model",
            model.to_str().unwrap(),
        ],
        "current-final-s12",
        Some("nnue-v2q"),
        false,
    );
    let _ = std::fs::remove_file(&model);

    for historical in [
        "current-final-nnue-v2q-full",
        "current-final-nnue-v2q",
        "current-final-nnue-v2q-material",
        "current-final-nnue-v2q-material-cal-fut",
        "current-final-nnue-v2q-material-r12",
        "current-final-nnue-v2q-material-r12-inc",
    ] {
        assert_startup_rejected_with_message(
            &["--profile", historical],
            "has been removed; checkout the historical commit to reproduce that experiment",
        );
    }

    assert_startup_rejected(&["--profile", "null"]);
    assert_startup_rejected(&["--profile", "current", "--profile", "current-final"]);
    assert_startup_rejected(&["--profile", "not-a-profile"]);
    // Closed experiment names are no longer selectable.
    assert_startup_rejected(&["--profile", "current-lmr"]);
    assert_startup_rejected(&["--profile", "current-threat-aware"]);
    assert_startup_rejected(&["--profile", "current-aspiration"]);
    assert_startup_rejected(&["--profile", "current-qsearch-pruning"]);
}
