//! D1.9 external search-safety harness.
//!
//! The manifest is deliberately outside the Rust search implementation. Each
//! case is run through a real UCI process for both Current and the isolated
//! qsearch-pruning candidate. This is a safety gate, not an Elo measurement.

use std::collections::{HashSet, VecDeque};
use std::ffi::OsStr;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use eureka::chess::{generate_legal_moves, move_to_uci, parse_fen};

const MANIFEST: &str = include_str!("data/search_validation.epd");
const PROFILES: [&str; 2] = ["current", "current-qsearch-pruning"];
const EXPECTED_CORPUS_V2_CASES: usize = 23;
const EXPECTED_D10_IDS: [&str; 11] = [
    "d10-promotion-chain-white",
    "d10-promotion-chain-black",
    "d10-unique-underpromotion",
    "d10-promotion-capture",
    "d10-xray-recapture",
    "d10-pinned-recapture",
    "d10-king-recapture",
    "d10-checking-capture",
    "d10-en-passant-discovered-check",
    "d10-defensive-capture",
    "d10-checking-losing-capture",
];
const UCI_TIMEOUT: Duration = Duration::from_secs(5);
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
const QUIT_TIMEOUT: Duration = Duration::from_secs(5);
const RECENT_OUTPUT_LINES: usize = 20;
const READER_CHANNEL_CAPACITY: usize = 256;

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
    forbidden_moves: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum Score {
    Cp(i32),
    Mate(i32),
}

#[derive(Debug, Clone)]
struct InfoSnapshot {
    depth: u32,
    score: Option<Score>,
    pv: Vec<String>,
}

#[derive(Debug)]
struct Outcome {
    bestmove: String,
    completed_depth: u32,
    score_at_completed_depth: Option<Score>,
    pv_at_completed_depth: Vec<String>,
    recent_stdout: Vec<String>,
    recent_stderr: Vec<String>,
}

#[derive(Debug)]
struct SearchResult {
    bestmove: String,
    completed_depth: u32,
    score_at_completed_depth: Option<Score>,
    pv_at_completed_depth: Vec<String>,
}

type LineReceiver = Receiver<io::Result<String>>;

fn engine_path() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var("CARGO_BIN_EXE_eureka").expect("CARGO_BIN_EXE_eureka must be set by cargo"),
    )
}

fn spawn_line_reader<R: Read + Send + 'static>(reader: R) -> LineReceiver {
    let (sender, receiver) = mpsc::sync_channel(READER_CHANNEL_CAPACITY);
    thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if sender.send(Ok(line.trim_end().to_string())).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error));
                    break;
                }
            }
        }
    });
    receiver
}

struct EngineProcess {
    profile: String,
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: LineReceiver,
    stderr: LineReceiver,
    recent_stdout: VecDeque<String>,
    recent_stderr: VecDeque<String>,
}

impl EngineProcess {
    fn spawn(profile: &str) -> Result<Self, String> {
        Self::spawn_command(profile, engine_path(), &["--profile", profile])
    }

    fn spawn_command<S: AsRef<OsStr>>(
        profile: &str,
        program: S,
        args: &[&str],
    ) -> Result<Self, String> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("failed to start {profile}: {error}"))?;
        let stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{profile}: engine stdin was not piped"));
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{profile}: engine stdout was not piped"));
            }
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{profile}: engine stderr was not piped"));
            }
        };

        Ok(Self {
            profile: profile.to_string(),
            child,
            stdin: Some(stdin),
            stdout: spawn_line_reader(stdout),
            stderr: spawn_line_reader(stderr),
            recent_stdout: VecDeque::new(),
            recent_stderr: VecDeque::new(),
        })
    }

    fn remember(lines: &mut VecDeque<String>, line: String) {
        lines.push_back(line);
        while lines.len() > RECENT_OUTPUT_LINES {
            lines.pop_front();
        }
    }

    fn drain_stderr(&mut self) {
        while let Ok(result) = self.stderr.try_recv() {
            match result {
                Ok(line) => Self::remember(&mut self.recent_stderr, line),
                Err(error) => Self::remember(
                    &mut self.recent_stderr,
                    format!("<stderr reader error: {error}>"),
                ),
            }
        }
    }

    fn diagnostics(&mut self) -> String {
        self.drain_stderr();
        format!(
            "profile={} recent stdout={:?}; recent stderr={:?}",
            self.profile, self.recent_stdout, self.recent_stderr
        )
    }

    fn failure(&mut self, stage: &str, detail: impl std::fmt::Display) -> String {
        let profile = self.profile.clone();
        let diagnostics = self.diagnostics();
        format!("{} during {stage}: {detail}; {}", profile, diagnostics)
    }

    fn send_line(&mut self, line: &str) -> Result<(), String> {
        let result = {
            let stdin = self
                .stdin
                .as_mut()
                .ok_or_else(|| format!("{} stdin is closed", self.profile))?;
            writeln!(stdin, "{line}").and_then(|_| stdin.flush())
        };
        result.map_err(|error| self.failure("sending command", error))
    }

    fn read_line(&mut self, stage: &str, timeout: Duration) -> Result<String, String> {
        match self.stdout.recv_timeout(timeout) {
            Ok(Ok(line)) => {
                Self::remember(&mut self.recent_stdout, line.clone());
                self.drain_stderr();
                Ok(line)
            }
            Ok(Err(error)) => Err(self.failure(stage, error)),
            Err(RecvTimeoutError::Timeout) => Err(self.failure(
                stage,
                format_args!("timed out after {} ms", timeout.as_millis()),
            )),
            Err(RecvTimeoutError::Disconnected) => {
                Err(self.failure(stage, "stdout reader disconnected"))
            }
        }
    }

    fn read_until(&mut self, prefix: &str, timeout: Duration) -> Result<Vec<String>, String> {
        let started = Instant::now();
        let mut lines = Vec::new();
        loop {
            let remaining = timeout
                .checked_sub(started.elapsed())
                .ok_or_else(|| self.failure("waiting for engine output", "deadline expired"))?;
            let line = self.read_line("waiting for engine output", remaining)?;
            let matched = line.starts_with(prefix);
            lines.push(line);
            if matched {
                return Ok(lines);
            }
        }
    }

    fn wait_for_exit(&mut self, timeout: Duration) -> Result<Option<ExitStatus>, String> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    self.drain_stderr();
                    return Ok(Some(status));
                }
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                Ok(None) => return Ok(None),
                Err(error) => return Err(self.failure("checking process status", error)),
            }
        }
    }

    fn cleanup_process(&mut self) {
        let _ = self.send_line("stop");
        let _ = self.send_line("quit");
        if !matches!(self.wait_for_exit(Duration::from_millis(250)), Ok(Some(_))) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        self.drain_stderr();
    }

    fn finish(&mut self) -> Result<(), String> {
        self.send_line("quit")?;
        match self.wait_for_exit(QUIT_TIMEOUT)? {
            Some(status) if status.success() => Ok(()),
            Some(status) => Err(self.failure(
                "quitting engine",
                format_args!("process exited unsuccessfully with {status}"),
            )),
            None => {
                self.cleanup_process();
                Err(self.failure("quitting engine", "process did not exit before deadline"))
            }
        }
    }

    fn read_search_result(&mut self, timeout: Duration) -> Result<SearchResult, String> {
        let started = Instant::now();
        let mut highest_info: Option<InfoSnapshot> = None;
        let bestmove = loop {
            let remaining = timeout.checked_sub(started.elapsed()).ok_or_else(|| {
                self.failure("search", "total search deadline expired before bestmove")
            })?;
            let line = self.read_line("search", remaining)?;
            if line.starts_with("info ") {
                if let Some(info) = parse_info(&line) {
                    let replace = highest_info
                        .as_ref()
                        .map(|previous| {
                            info.depth > previous.depth
                                || (info.depth == previous.depth && info.score.is_some())
                        })
                        .unwrap_or(true);
                    if replace {
                        highest_info = Some(info);
                    }
                }
            }
            if let Some(move_text) = line.strip_prefix("bestmove ") {
                break move_text
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| self.failure("parsing bestmove", "missing move text"))?
                    .to_string();
            }
        };

        Ok(SearchResult {
            bestmove,
            completed_depth: highest_info.as_ref().map(|info| info.depth).unwrap_or(0),
            score_at_completed_depth: highest_info.as_ref().and_then(|info| info.score),
            pv_at_completed_depth: highest_info
                .as_ref()
                .map(|info| info.pv.clone())
                .unwrap_or_default(),
        })
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        let still_running = self.child.try_wait().ok().flatten().is_none();
        if still_running {
            self.cleanup_process();
        }
    }
}

fn parse_manifest() -> Vec<Case> {
    MANIFEST
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .map(|(line_no, line)| {
            let fields: Vec<&str> = line.split('|').collect();
            assert!(
                (8..=9).contains(&fields.len()),
                "manifest line {} must have 8 or 9 pipe-separated fields",
                line_no + 1
            );
            let forbidden_moves = fields
                .get(8)
                .filter(|field| !field.is_empty() && **field != "-")
                .map(|field| field.split(',').map(str::to_string).collect())
                .unwrap_or_default();
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
                forbidden_moves,
            }
        })
        .collect()
}

fn parse_info(line: &str) -> Option<InfoSnapshot> {
    let words: Vec<&str> = line.split_whitespace().collect();
    let depth = words
        .windows(2)
        .find(|window| window[0] == "depth")
        .and_then(|window| window[1].parse().ok())?;
    let score = words.windows(3).find_map(|window| {
        if window[0] != "score" {
            return None;
        }
        match window[1] {
            "cp" => window[2].parse().ok().map(Score::Cp),
            "mate" => window[2].parse().ok().map(Score::Mate),
            _ => None,
        }
    });
    let pv = words
        .iter()
        .position(|word| *word == "pv")
        .map(|index| {
            words[index + 1..]
                .iter()
                .map(|word| word.to_string())
                .collect()
        })
        .unwrap_or_default();
    Some(InfoSnapshot { depth, score, pv })
}

fn run_case(case: &Case, profile: &str) -> Result<Outcome, String> {
    let mut engine = EngineProcess::spawn(profile)?;

    engine.send_line("uci")?;
    let handshake = engine.read_until("uciok", UCI_TIMEOUT)?;
    if !handshake
        .iter()
        .any(|line| line == &format!("info string profile {profile}"))
    {
        return Err(engine.failure(
            "checking profile identity",
            format_args!("expected profile identity in handshake: {handshake:?}"),
        ));
    }

    engine.send_line("isready")?;
    engine.read_until("readyok", UCI_TIMEOUT)?;
    engine.send_line(&format!("position fen {}", case.fen))?;
    engine.send_line(&format!("go depth {}", case.depth))?;

    let result = engine.read_search_result(SEARCH_TIMEOUT)?;
    let recent_stdout = engine.recent_stdout.iter().cloned().collect();
    let recent_stderr = engine.recent_stderr.iter().cloned().collect();

    engine.finish()?;
    Ok(Outcome {
        bestmove: result.bestmove,
        completed_depth: result.completed_depth,
        score_at_completed_depth: result.score_at_completed_depth,
        pv_at_completed_depth: result.pv_at_completed_depth,
        recent_stdout,
        recent_stderr,
    })
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

fn position_facts(case: &Case) -> (Vec<String>, bool) {
    let mut pos = parse_fen(&case.fen).unwrap_or_else(|error| panic!("{}: {error}", case.id));
    let in_check = pos.is_in_check(pos.side_to_move());
    let legal_moves: Vec<String> = generate_legal_moves(&mut pos)
        .into_iter()
        .map(move_to_uci)
        .collect();
    (legal_moves, in_check)
}

fn context(case: &Case, profile: &str, outcome: &Outcome) -> String {
    format!(
        "case={} profile={} depth={} score={:?} pv={:?} recent stdout={:?}; recent stderr={:?}",
        case.id,
        profile,
        outcome.completed_depth,
        outcome.score_at_completed_depth,
        outcome.pv_at_completed_depth,
        outcome.recent_stdout,
        outcome.recent_stderr
    )
}

fn assert_completed_depth(case: &Case, outcome: &Outcome, profile: &str) {
    if !case.score_class.starts_with("terminal-") {
        assert!(
            outcome.completed_depth >= case.depth,
            "{}: requested depth {}, completed depth {}; {}",
            case.id,
            case.depth,
            outcome.completed_depth,
            context(case, profile, outcome)
        );
    }
}

fn assert_score_class(
    case: &Case,
    outcome: &Outcome,
    profile: &str,
    legal_moves: &[String],
    in_check: bool,
) {
    match case.score_class.as_str() {
        "terminal-mate" => {
            assert!(
                legal_moves.is_empty(),
                "{} must have no legal moves",
                case.id
            );
            assert!(in_check, "{} must be checkmate, not stalemate", case.id);
            assert_eq!(
                outcome.bestmove,
                "0000",
                "{} {profile} terminal checkmate must emit bestmove 0000; {}",
                case.id,
                context(case, profile, outcome)
            );
        }
        "terminal-draw" => {
            assert!(
                legal_moves.is_empty(),
                "{} must have no legal moves",
                case.id
            );
            assert!(!in_check, "{} must be stalemate, not checkmate", case.id);
            assert_eq!(
                outcome.bestmove,
                "0000",
                "{} {profile} terminal stalemate must emit bestmove 0000; {}",
                case.id,
                context(case, profile, outcome)
            );
        }
        "mate" => assert!(
            matches!(outcome.score_at_completed_depth, Some(Score::Mate(moves)) if moves > 0),
            "{} {profile} must retain a forced mate; {}",
            case.id,
            context(case, profile, outcome)
        ),
        "winning" => assert!(
            score_rank(outcome.score_at_completed_depth) >= 3,
            "{} {profile} must remain winning; {}",
            case.id,
            context(case, profile, outcome)
        ),
        "nonlosing" => assert!(
            score_rank(outcome.score_at_completed_depth) >= 2,
            "{} {profile} must remain non-losing; {}",
            case.id,
            context(case, profile, outcome)
        ),
        "losing" => assert!(
            matches!(outcome.score_at_completed_depth, Some(Score::Cp(cp)) if cp < -150),
            "{} {profile} must remain in the pinned centipawn-loss class; {}",
            case.id,
            context(case, profile, outcome)
        ),
        other => panic!("{} has unknown score class {other}", case.id),
    }
}

fn assert_legal_and_allowed(case: &Case, outcome: &Outcome, profile: &str, legal_moves: &[String]) {
    if outcome.bestmove == "0000" {
        assert!(
            legal_moves.is_empty(),
            "{} {profile} emitted 0000 with legal moves {:?}; {}",
            case.id,
            legal_moves,
            context(case, profile, outcome)
        );
        return;
    }

    assert!(
        legal_moves.iter().any(|mv| mv == &outcome.bestmove),
        "{} {profile} emitted illegal bestmove {}; {}",
        case.id,
        outcome.bestmove,
        context(case, profile, outcome)
    );
    assert!(
        !case
            .forbidden_moves
            .iter()
            .any(|mv| mv == &outcome.bestmove),
        "{} {profile} emitted forbidden bestmove {}; forbidden set {:?}; {}",
        case.id,
        outcome.bestmove,
        case.forbidden_moves,
        context(case, profile, outcome)
    );
    if case.allowed_moves != ["*"] {
        assert!(
            case.allowed_moves.iter().any(|mv| mv == &outcome.bestmove),
            "{} {profile} bestmove {} is outside allowed set {:?}; {}",
            case.id,
            outcome.bestmove,
            case.allowed_moves,
            context(case, profile, outcome)
        );
    }
}

fn assert_pv(case: &Case, outcome: &Outcome, profile: &str, legal_moves: &[String]) {
    if case.score_class.starts_with("terminal-") {
        return;
    }
    assert!(
        !outcome.pv_at_completed_depth.is_empty(),
        "{} {profile} has no PV at completed depth; {}",
        case.id,
        context(case, profile, outcome)
    );
    assert_eq!(
        outcome.pv_at_completed_depth[0],
        outcome.bestmove,
        "{} {profile} PV first move does not match bestmove; {}",
        case.id,
        context(case, profile, outcome)
    );
    assert!(
        legal_moves
            .iter()
            .any(|mv| mv == &outcome.pv_at_completed_depth[0]),
        "{} {profile} PV first move is illegal; {}",
        case.id,
        context(case, profile, outcome)
    );
}

fn assert_manifest_case(case: &Case) {
    assert!(!case.id.is_empty());
    assert!(!case.fen.is_empty());
    assert!(case.depth > 0);
    assert_eq!(case.source, "project-curated");
    assert_eq!(case.license, "CC0-1.0");
    assert!(!case.allowed_moves.is_empty());
    assert!(
        case.allowed_moves == ["*"] || case.allowed_moves.iter().all(|mv| !mv.is_empty()),
        "{} has an empty allowed-move entry",
        case.id
    );
    let (legal_moves, in_check) = position_facts(case);
    assert!(
        case.forbidden_moves.iter().all(|mv| !mv.is_empty()),
        "{} has an empty forbidden-move entry",
        case.id
    );
    assert!(
        case.forbidden_moves
            .iter()
            .all(|mv| legal_moves.contains(mv)),
        "{} forbidden set contains an illegal move: {:?} vs {:?}",
        case.id,
        case.forbidden_moves,
        legal_moves
    );
    if case.allowed_moves != ["*"] {
        assert!(
            case.forbidden_moves
                .iter()
                .all(|mv| !case.allowed_moves.contains(mv)),
            "{} allowed and forbidden sets overlap",
            case.id
        );
    }
    if case.score_class.starts_with("terminal-") {
        assert!(
            case.forbidden_moves.is_empty(),
            "{} terminal case cannot have forbidden moves",
            case.id
        );
    }
    if case.allowed_moves == ["*"] {
        assert!(
            !legal_moves.is_empty(),
            "{} wildcard must be non-terminal",
            case.id
        );
    } else if case.score_class.starts_with("terminal-") {
        assert_eq!(case.allowed_moves, ["0000"], "{} terminal marker", case.id);
        assert!(legal_moves.is_empty(), "{} must be terminal", case.id);
        match case.score_class.as_str() {
            "terminal-mate" => assert!(in_check, "{} must be checkmate", case.id),
            "terminal-draw" => assert!(!in_check, "{} must be stalemate", case.id),
            _ => unreachable!(),
        }
    } else {
        assert!(
            case.allowed_moves.iter().all(|mv| legal_moves.contains(mv)),
            "{} allowed set contains an illegal move: {:?} vs {:?}",
            case.id,
            case.allowed_moves,
            legal_moves
        );
    }
}

#[test]
fn validation_manifest_is_pinned_and_well_formed() {
    let cases = parse_manifest();
    assert!(
        cases.len() == EXPECTED_CORPUS_V2_CASES,
        "validation corpus cardinality changed: expected {}, got {}",
        EXPECTED_CORPUS_V2_CASES,
        cases.len()
    );
    let mut ids = HashSet::new();
    for case in &cases {
        assert!(
            ids.insert(case.id.clone()),
            "duplicate manifest id {}",
            case.id
        );
        assert_manifest_case(case);
    }
    assert!(cases.iter().any(|case| case.category == "tactical-capture"));
    assert!(cases.iter().any(|case| case.category == "mate-sequence"));
    assert!(cases.iter().any(|case| case.category == "underpromotion"));
    assert!(cases.iter().any(|case| case.category == "promotion-race"));
    assert!(cases.iter().any(|case| case.category == "en-passant"));
    assert!(cases.iter().any(|case| case.category == "stalemate"));
    assert!(cases.iter().any(|case| case.category == "endgame-kqk"));
    assert!(cases.iter().any(|case| case.category == "endgame-krk"));
    for category in [
        "promotion-chain",
        "unique-underpromotion",
        "promotion-capture",
        "xray-recapture",
        "pinned-recapture",
        "king-recapture",
        "en-passant-discovered-check",
        "defensive-capture",
    ] {
        assert!(
            cases.iter().any(|case| case.category == category),
            "D1.10 corpus is missing category {category}"
        );
    }
    let actual_d10_ids: HashSet<String> = cases
        .iter()
        .filter(|case| case.id.starts_with("d10-"))
        .map(|case| case.id.clone())
        .collect();
    let expected_d10_ids: HashSet<String> = EXPECTED_D10_IDS
        .iter()
        .map(|id| (*id).to_string())
        .collect();
    assert_eq!(
        actual_d10_ids, expected_d10_ids,
        "D1.10 case IDs changed; update the pinned corpus deliberately"
    );
    assert!(
        cases
            .iter()
            .any(|case| case.category == "checking-losing-capture"),
        "D1.10 corpus is missing the negative-SEE checking-capture case"
    );
}

#[test]
fn current_qsearch_pruning_passes_external_search_safety_corpus() {
    for case in parse_manifest() {
        let (baseline_legal, baseline_in_check) = position_facts(&case);
        let baseline = run_case(&case, PROFILES[0])
            .unwrap_or_else(|error| panic!("baseline failed for {}: {error}", case.id));
        assert_completed_depth(&case, &baseline, PROFILES[0]);
        assert_legal_and_allowed(&case, &baseline, PROFILES[0], &baseline_legal);
        assert_pv(&case, &baseline, PROFILES[0], &baseline_legal);
        assert_score_class(
            &case,
            &baseline,
            PROFILES[0],
            &baseline_legal,
            baseline_in_check,
        );

        let (candidate_legal, candidate_in_check) = position_facts(&case);
        let candidate = run_case(&case, PROFILES[1])
            .unwrap_or_else(|error| panic!("candidate failed for {}: {error}", case.id));
        assert_completed_depth(&case, &candidate, PROFILES[1]);
        assert_legal_and_allowed(&case, &candidate, PROFILES[1], &candidate_legal);
        assert_pv(&case, &candidate, PROFILES[1], &candidate_legal);
        assert_score_class(
            &case,
            &candidate,
            PROFILES[1],
            &candidate_legal,
            candidate_in_check,
        );

        if !case.score_class.starts_with("terminal-") {
            assert!(
                score_rank(candidate.score_at_completed_depth)
                    >= score_rank(baseline.score_at_completed_depth),
                "{} candidate downgraded baseline {:?} to {:?}; baseline={}, candidate={}",
                case.id,
                baseline.score_at_completed_depth,
                candidate.score_at_completed_depth,
                context(&case, PROFILES[0], &baseline),
                context(&case, PROFILES[1], &candidate)
            );
        }
    }
}

fn spawn_continuous_info_engine() -> Result<EngineProcess, String> {
    #[cfg(windows)]
    {
        let script = "Write-Output 'id name D19DeadlineFake'; Write-Output 'id author tests'; Write-Output 'info string search profile current'; Write-Output 'uciok'; Write-Output 'readyok'; while ($true) { Write-Output 'info depth 1 score cp 0 pv a2a3'; Start-Sleep -Milliseconds 10 }";
        EngineProcess::spawn_command(
            "current",
            "powershell.exe",
            &[
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
        )
    }
    #[cfg(not(windows))]
    {
        let script = "printf '%s\\n' 'id name D19DeadlineFake' 'id author tests' 'info string search profile current' 'uciok' 'readyok'; while :; do printf '%s\\n' 'info depth 1 score cp 0 pv a2a3'; sleep 0.01; done";
        EngineProcess::spawn_command("current", "sh", &["-c", script])
    }
}

#[test]
fn search_total_deadline_kills_continuous_info_engine() {
    let mut engine = spawn_continuous_info_engine().expect("deadline fake must start");
    engine
        .read_until("uciok", UCI_TIMEOUT)
        .expect("deadline fake must complete UCI handshake");
    engine
        .read_until("readyok", UCI_TIMEOUT)
        .expect("deadline fake must complete readiness");

    let started = Instant::now();
    let error = engine
        .read_search_result(Duration::from_millis(250))
        .expect_err("continuous info without bestmove must hit total deadline");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "search deadline was not bounded: {elapsed:?}; {error}"
    );
    assert!(
        error.contains("search") && error.contains("profile=current"),
        "deadline error lacks process context: {error}"
    );

    engine.cleanup_process();
    assert!(
        engine
            .child
            .try_wait()
            .expect("deadline fake status must be readable")
            .is_some(),
        "deadline fake must be killed and reaped"
    );
}
