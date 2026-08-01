//! D1.9 external search-safety harness.
//!
//! The manifest is deliberately outside the Rust search implementation. Each
//! case is run through a real UCI process for both Current and the isolated
//! qsearch-pruning candidate. This is a safety gate, not an Elo measurement.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use chess_engine_demo::chess::{generate_legal_moves, move_to_uci, parse_fen};

const MANIFEST: &str = include_str!("data/search_validation.epd");
const PROFILES: [&str; 2] = ["current", "current-qsearch-pruning"];

#[derive(Debug)]
struct Case {
    id: String,
    category: String,
    fen: String,
    allowed_moves: Vec<String>,
    score_class: String,
    depth: u32,
    source: String,
    license: String,
}

#[derive(Debug, Clone, Copy)]
enum Score {
    Cp(i32),
    Mate(i32),
}

#[derive(Debug)]
struct Outcome {
    bestmove: String,
    score: Option<Score>,
}

fn engine_path() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("CARGO_BIN_EXE_chess-engine-demo")
            .expect("CARGO_BIN_EXE_chess-engine-demo must be set by cargo"),
    )
}

fn parse_manifest() -> Vec<Case> {
    MANIFEST
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|(line_no, line)| {
            let fields: Vec<&str> = line.split('|').collect();
            assert_eq!(
                fields.len(),
                8,
                "manifest line {} must have 8 pipe-separated fields",
                line_no + 1
            );
            Case {
                id: fields[0].to_string(),
                category: fields[1].to_string(),
                fen: fields[2].to_string(),
                allowed_moves: fields[3].split(',').map(str::to_string).collect(),
                score_class: fields[4].to_string(),
                depth: fields[5]
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid depth on manifest line {}", line_no + 1)),
                source: fields[6].to_string(),
                license: fields[7].to_string(),
            }
        })
        .collect()
}

fn spawn_engine(profile: &str) -> (Child, ChildStdin, BufReader<ChildStdout>) {
    let mut child = Command::new(engine_path())
        .args(["--profile", profile])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to start {profile}: {error}"));
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

fn parse_info_score(line: &str) -> Option<Score> {
    let words: Vec<&str> = line.split_whitespace().collect();
    words.windows(3).find_map(|window| {
        if window[0] != "score" {
            return None;
        }
        match window[1] {
            "cp" => window[2].parse().ok().map(Score::Cp),
            "mate" => window[2].parse().ok().map(Score::Mate),
            _ => None,
        }
    })
}

fn run_case(case: &Case, profile: &str) -> Outcome {
    let (mut child, mut stdin, mut reader) = spawn_engine(profile);

    stdin.write_all(b"uci\n").unwrap();
    stdin.flush().unwrap();
    let handshake = read_until(&mut reader, "uciok");
    assert!(
        handshake
            .iter()
            .any(|line| line == &format!("info string search profile {profile}")),
        "{} must report profile {profile}: {:?}",
        case.id,
        handshake
    );

    stdin.write_all(b"isready\n").unwrap();
    stdin.flush().unwrap();
    read_until(&mut reader, "readyok");

    writeln!(stdin, "position fen {}", case.fen).unwrap();
    writeln!(stdin, "go depth {}", case.depth).unwrap();
    stdin.flush().unwrap();

    let mut score = None;
    let bestmove;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .expect("engine stdout must remain readable during search");
        assert!(read > 0, "{} exited before bestmove", case.id);
        let line = line.trim_end();
        if line.starts_with("info ") {
            if let Some(parsed) = parse_info_score(line) {
                score = Some(parsed);
            }
        }
        if let Some(move_text) = line.strip_prefix("bestmove ") {
            bestmove = move_text.split_whitespace().next().unwrap().to_string();
            break;
        }
    }

    stdin.write_all(b"quit\n").unwrap();
    stdin.flush().unwrap();
    drop(stdin);
    let status = child.wait().expect("engine must be reaped");
    assert!(
        status.success(),
        "{} {profile} exited unsuccessfully",
        case.id
    );

    Outcome { bestmove, score }
}

fn score_rank(score: Option<Score>) -> i32 {
    match score {
        Some(Score::Mate(moves)) if moves > 0 => 4,
        Some(Score::Mate(_)) => 0,
        Some(Score::Cp(cp)) if cp >= 300 => 3,
        Some(Score::Cp(cp)) if cp >= -150 => 2,
        Some(Score::Cp(_)) => 1,
        None => 0,
    }
}

fn assert_score_class(case: &Case, outcome: &Outcome, profile: &str) {
    match case.score_class.as_str() {
        "terminal-mate" | "terminal-draw" => {
            assert_eq!(
                outcome.bestmove, "0000",
                "{} {profile} terminal case must emit bestmove 0000",
                case.id
            );
        }
        "mate" => assert!(
            matches!(outcome.score, Some(Score::Mate(moves)) if moves > 0),
            "{} {profile} must retain a forced mate, got {:?}",
            case.id,
            outcome.score
        ),
        "winning" => assert!(
            score_rank(outcome.score) >= 3,
            "{} {profile} must remain winning, got {:?}",
            case.id,
            outcome.score
        ),
        "nonlosing" => assert!(
            score_rank(outcome.score) >= 2,
            "{} {profile} must remain non-losing, got {:?}",
            case.id,
            outcome.score
        ),
        other => panic!("{} has unknown score class {other}", case.id),
    }
}

fn assert_legal_and_allowed(case: &Case, outcome: &Outcome, profile: &str) {
    let mut pos = parse_fen(&case.fen).unwrap_or_else(|error| panic!("{}: {error}", case.id));
    let legal_moves: Vec<String> = generate_legal_moves(&mut pos)
        .into_iter()
        .map(move_to_uci)
        .collect();

    if outcome.bestmove == "0000" {
        assert!(
            legal_moves.is_empty(),
            "{} {profile} emitted 0000 with legal moves {:?}",
            case.id,
            legal_moves
        );
        return;
    }

    assert!(
        legal_moves.iter().any(|mv| mv == &outcome.bestmove),
        "{} {profile} emitted illegal bestmove {}",
        case.id,
        outcome.bestmove
    );
    if case.allowed_moves != ["*"] {
        assert!(
            case.allowed_moves.iter().any(|mv| mv == &outcome.bestmove),
            "{} {profile} bestmove {} is outside allowed set {:?}",
            case.id,
            outcome.bestmove,
            case.allowed_moves
        );
    }
}

#[test]
fn validation_manifest_is_pinned_and_well_formed() {
    let cases = parse_manifest();
    assert!(
        cases.len() >= 10,
        "validation corpus must not shrink silently"
    );
    assert!(cases.iter().any(|case| case.category == "tactical-capture"));
    assert!(cases.iter().any(|case| case.category == "mate-sequence"));
    assert!(cases.iter().any(|case| case.category == "underpromotion"));
    assert!(cases.iter().any(|case| case.category == "promotion-race"));
    assert!(cases.iter().any(|case| case.category == "en-passant"));
    assert!(cases.iter().any(|case| case.category == "stalemate"));
    assert!(cases.iter().any(|case| case.category == "endgame-kqk"));
    assert!(cases.iter().any(|case| case.category == "endgame-krk"));
    for case in cases {
        assert!(!case.id.is_empty());
        assert!(!case.fen.is_empty());
        assert!(case.depth > 0);
        assert_eq!(case.source, "project-curated");
        assert_eq!(case.license, "CC0-1.0");
    }
}

#[test]
fn current_qsearch_pruning_passes_external_search_safety_corpus() {
    for case in parse_manifest() {
        let baseline = run_case(&case, PROFILES[0]);
        assert_legal_and_allowed(&case, &baseline, PROFILES[0]);
        assert_score_class(&case, &baseline, PROFILES[0]);

        let candidate = run_case(&case, PROFILES[1]);
        assert_legal_and_allowed(&case, &candidate, PROFILES[1]);
        assert_score_class(&case, &candidate, PROFILES[1]);

        if !case.score_class.starts_with("terminal-") {
            assert!(
                score_rank(candidate.score) >= score_rank(baseline.score),
                "{} candidate downgraded baseline {:?} to {:?}",
                case.id,
                baseline.score,
                candidate.score
            );
        }
    }
}
