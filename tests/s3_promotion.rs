use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, PartialEq, Eq)]
struct SearchSnapshot {
    depth: u32,
    score: String,
    nodes: u64,
    bestmove: String,
    pv: Vec<String>,
}

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
        assert!(read > 0, "engine exited before emitting {prefix}");
        let line = line.trim_end().to_string();
        let matched = line.starts_with(prefix);
        lines.push(line);
        if matched {
            return lines;
        }
    }
}

fn token_after<'a>(tokens: &'a [&str], key: &str) -> &'a str {
    let index = tokens
        .iter()
        .position(|token| *token == key)
        .unwrap_or_else(|| panic!("info line is missing {key}: {tokens:?}"));
    tokens
        .get(index + 1)
        .copied()
        .unwrap_or_else(|| panic!("info line is missing value after {key}: {tokens:?}"))
}

fn search_snapshot(args: &[&str], fen: &str) -> SearchSnapshot {
    let (mut child, mut stdin, mut reader) = spawn_engine(args);
    stdin.write_all(b"uci\n").unwrap();
    stdin.flush().unwrap();
    let handshake = read_until(&mut reader, "uciok");
    assert!(
        handshake
            .iter()
            .any(|line| line == "info string search profile current-final"),
        "promoted default identity must be current-final: {handshake:?}"
    );

    stdin.write_all(b"isready\n").unwrap();
    stdin.flush().unwrap();
    read_until(&mut reader, "readyok");

    stdin
        .write_all(format!("position fen {fen}\ngo depth 4\n").as_bytes())
        .unwrap();
    stdin.flush().unwrap();
    let search_lines = read_until(&mut reader, "bestmove ");
    let bestmove_line = search_lines
        .iter()
        .find(|line| line.starts_with("bestmove "))
        .expect("search must emit bestmove");
    let bestmove_tokens: Vec<&str> = bestmove_line.split_whitespace().collect();
    let bestmove = bestmove_tokens
        .get(1)
        .expect("bestmove must contain a move")
        .to_string();

    let info_line = search_lines
        .iter()
        .rfind(|line| line.starts_with("info depth "))
        .expect("search must emit a completed-depth info line");
    let tokens: Vec<&str> = info_line.split_whitespace().collect();
    let depth = tokens[2]
        .parse::<u32>()
        .expect("info depth must be numeric");
    let score_kind = token_after(&tokens, "score");
    let score_value = tokens
        .iter()
        .position(|token| *token == "score")
        .and_then(|index| tokens.get(index + 2))
        .copied()
        .expect("info score must contain a value");
    let score = format!("{score_kind} {score_value}");
    let nodes = token_after(&tokens, "nodes")
        .parse::<u64>()
        .expect("info nodes must be numeric");
    let pv_index = tokens
        .iter()
        .position(|token| *token == "pv")
        .expect("completed info must contain a PV");
    let pv = tokens[pv_index + 1..]
        .iter()
        .map(|move_uci| (*move_uci).to_string())
        .collect();

    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("engine must be reaped");
    assert!(status.success(), "engine quit must succeed: {status}");

    SearchSnapshot {
        depth,
        score,
        nodes,
        bestmove,
        pv,
    }
}

#[test]
fn default_profile_matches_explicit_current_final_on_fixed_searches() {
    let fens = [
        "r1bq1rk1/pp1p1ppp/n1p2n2/4p3/2B1P3/P1N2Q2/1PPP1PPP/R1B1K2R w KQ - 0 1",
        "r3b3/3P4/1k6/8/8/8/4Q3/6K1 w - - 0 1",
        "8/2bp4/4k1q1/8/3Q4/8/8/5b1K w - - 0 1",
    ];

    for fen in fens {
        let default_snapshot = search_snapshot(&[], fen);
        let explicit_snapshot = search_snapshot(&["--profile", "current-final"], fen);
        assert_eq!(
            default_snapshot, explicit_snapshot,
            "default and explicit current-final must have identical fixed-search output for {fen}"
        );
    }
}
