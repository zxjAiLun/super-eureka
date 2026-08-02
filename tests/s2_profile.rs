use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

fn engine_path() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("CARGO_BIN_EXE_chess-engine-demo")
            .expect("CARGO_BIN_EXE_chess-engine-demo must be set by cargo"),
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
) {
    stdin.write_all(b"uci\n").unwrap();
    stdin.flush().unwrap();
    let lines = read_until(reader, "uciok");
    assert!(
        lines
            .iter()
            .any(|line| line == &format!("info string search profile {}", expected)),
        "handshake must report {}: {:?}",
        expected,
        lines
    );

    stdin.write_all(b"isready\n").unwrap();
    stdin.flush().unwrap();
    read_until(reader, "readyok");
}

fn run_profile_process(args: &[&str], expected: &str, verify_hot_switch: bool) {
    let (mut child, mut stdin, mut reader) = spawn_engine(args);
    assert_handshake_profile(&mut stdin, &mut reader, expected);

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
                .any(|line| line == &format!("info string search profile {}", expected)),
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

fn assert_startup_rejected(args: &[&str]) {
    let output = Command::new(engine_path())
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("engine process must start");
    assert!(!output.status.success(), "startup must reject {:?}", args);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.lines().any(|line| line == "uciok"),
        "rejected startup must fail before uciok: {:?}",
        args
    );
}

#[test]
fn startup_profile_processes_report_identity_search_and_reject_bad_args() {
    run_profile_process(&[], "current", false);
    run_profile_process(&["--profile", "current"], "current", false);
    run_profile_process(&["--profile", "current-lmr"], "current-lmr", true);
    run_profile_process(
        &["--profile", "current-threat-aware"],
        "current-threat-aware",
        true,
    );
    run_profile_process(
        &["--profile", "current-aspiration"],
        "current-aspiration",
        true,
    );
    run_profile_process(
        &["--profile", "current-qsearch-pruning"],
        "current-qsearch-pruning",
        false,
    );

    assert_startup_rejected(&["--profile", "null"]);
    assert_startup_rejected(&["--profile", "current", "--profile", "current-aspiration"]);
    assert_startup_rejected(&["--profile", "not-a-profile"]);
}
