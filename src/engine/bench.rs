//! M4.0 deterministic search benchmark harness.
//!
//! This module does **not** change search semantics. It only drives the
//! *existing* history-aware, TT-aware search entry
//! (`search::search_best_move_with_history_and_tt`) and records
//! correctness + performance fields for later comparison.
//!
//! It cleanly separates:
//! - fixed-depth node count  -> search-tree efficiency
//! - fixed-node elapsed / NPS -> per-node throughput
//! - TT disabled / cold / warm -> baseline, first search, persistent-TT gain
//! - correctness fields (score, bestmove, PV, full restoration)
//! - machine-dependent fields (elapsed, NPS)

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::chess::fen::{parse_fen, to_fen};
use crate::chess::move_to_uci;
use crate::chess::movegen::generate_legal_moves;
use crate::chess::position::Position;
use crate::chess::types::START_FEN;
use crate::chess::ZobristKey;
use crate::engine::search::{
    search_best_move_with_history_and_tt, SearchContext, SearchLimits, SearchOutcome, MATE,
};
use crate::engine::tt::{TranspositionTable, MATE_THRESHOLD};

/// A benchmark mode. Selecting `All` expands to the three concrete modes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BenchMode {
    Disabled,
    Cold,
    Warm,
    All,
}

impl BenchMode {
    fn as_str(self) -> &'static str {
        match self {
            BenchMode::Disabled => "disabled",
            BenchMode::Cold => "cold",
            BenchMode::Warm => "warm",
            BenchMode::All => "all",
        }
    }
}

/// Which suite is being run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Suite {
    Smoke,
    Standard,
    Throughput,
}

impl Suite {
    fn as_str(self) -> &'static str {
        match self {
            Suite::Smoke => "smoke",
            Suite::Standard => "standard",
            Suite::Throughput => "throughput",
        }
    }
}

/// The search limit actually applied to a run.
#[derive(Clone, Copy)]
enum LimitKind {
    Depth(u32),
    Nodes(u64),
}

/// A single-position benchmark fixture.
struct Fixture {
    id: &'static str,
    fen: &'static str,
    /// Depth used for smoke/standard (ignored for the throughput suite, which
    /// always applies a node budget).
    limit: LimitKind,
    /// Optional explicit history (Zobrist keys of prior positions leading to
    /// `fen`). When `None`, the harness derives `[pos.zobrist_key()]` which
    /// satisfies the search contract and is sufficient for a fixed-position
    /// baseline.
    history: Option<Vec<ZobristKey>>,
    /// When `Some`, the exact correctness fields are asserted. Used only for the
    /// two locked regression fixtures (disabled mode).
    locked: Option<Locked>,
}

/// Exact locked expectation for a regression fixture.
struct Locked {
    nodes: u64,
    /// cp score (mate fixtures are never locked).
    score: i32,
    best_move: &'static str,
    pv: &'static [&'static str],
}

/// Parsed CLI configuration.
struct BenchArgs {
    suite: Suite,
    mode: BenchMode,
    repeat: u32,
    /// Throughput node budget (default 100_000).
    nodes: u64,
}

/// One measured search result.
struct BenchResult {
    suite: &'static str,
    fixture: &'static str,
    mode: &'static str,
    repeat: u32,
    limit: String,
    score: Option<i32>,
    best_move: String,
    completed_depth: u32,
    stopped: bool,
    nodes: u64,
    elapsed_us: u128,
    nps: u64,
    pv: String,
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

fn parse_args(args: &[String]) -> Result<BenchArgs, String> {
    let mut it = args.iter();
    let suite_kw = it
        .next()
        .ok_or_else(|| {
            "bench: missing suite (expected smoke|standard|throughput|help)".to_string()
        })?
        .clone();

    let suite = match suite_kw.as_str() {
        "smoke" => Suite::Smoke,
        "standard" => Suite::Standard,
        "throughput" => Suite::Throughput,
        other => {
            return Err(format!(
                "bench: unknown suite '{}' (expected smoke|standard|throughput|help)",
                other
            ));
        }
    };

    let mut mode = match suite {
        Suite::Smoke => BenchMode::Disabled,
        Suite::Standard => BenchMode::All,
        Suite::Throughput => BenchMode::Disabled,
    };
    let mut repeat = match suite {
        Suite::Smoke => 1,
        Suite::Standard => 1,
        Suite::Throughput => 3,
    };
    let mut nodes = 100_000u64;

    while let Some(tok) = it.next() {
        match tok.as_str() {
            "--mode" => {
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --mode requires a value".to_string())?
                    .clone();
                mode = match v.as_str() {
                    "disabled" => BenchMode::Disabled,
                    "cold" => BenchMode::Cold,
                    "warm" => BenchMode::Warm,
                    "all" => BenchMode::All,
                    other => {
                        return Err(format!(
                            "bench: invalid --mode '{}' (expected disabled|cold|warm|all)",
                            other
                        ));
                    }
                };
            }
            "--repeat" => {
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --repeat requires a value".to_string())?
                    .clone();
                let n: u32 = v
                    .parse()
                    .map_err(|_| format!("bench: --repeat '{}' is not a positive integer", v))?;
                if n == 0 {
                    return Err("bench: --repeat must be >= 1".to_string());
                }
                repeat = n;
            }
            "--nodes" => {
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --nodes requires a value".to_string())?
                    .clone();
                let n: u64 = v
                    .parse()
                    .map_err(|_| format!("bench: --nodes '{}' is not a positive integer", v))?;
                if n == 0 {
                    return Err("bench: --nodes must be >= 1".to_string());
                }
                nodes = n;
            }
            other => {
                return Err(format!("bench: unknown argument '{}'", other));
            }
        }
    }

    Ok(BenchArgs {
        suite,
        mode,
        repeat,
        nodes,
    })
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn smoke_fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            id: "startpos",
            fen: START_FEN,
            limit: LimitKind::Depth(3),
            history: None,
            locked: Some(Locked {
                nodes: 1149,
                score: 50,
                best_move: "b1c3",
                pv: &["b1c3", "b8c6", "g1f3"],
            }),
        },
        Fixture {
            id: "queen-win",
            // Exact locked position from tests/m2_4.rs (do NOT approximate).
            fen: "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1",
            limit: LimitKind::Depth(3),
            history: None,
            locked: Some(Locked {
                nodes: 963,
                score: 890,
                best_move: "e4a4",
                pv: &["e4a4", "h4h3", "a4h4", "h8g8", "h4h3"],
            }),
        },
    ]
}

fn standard_fixtures() -> Vec<Fixture> {
    vec![
        // 1. startpos
        Fixture {
            id: "startpos",
            fen: START_FEN,
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        // 2. locked queen-win (reuse exact FEN; not locked at depth 4)
        Fixture {
            id: "queen-win",
            fen: "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        // 3. open tactical middlegame (Italian, uncastled)
        Fixture {
            id: "open-tactical",
            fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        // 4. quiet / closed middlegame (closed center)
        Fixture {
            id: "closed-quiet",
            fen: "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        // 5. exposed/attacking king (black king still in center)
        Fixture {
            id: "exposed-king",
            fen: "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 6 5",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        // 6. high-branching movegen (many pieces, many captures available)
        Fixture {
            id: "high-branch",
            fen: "r3k2r/pppb1ppp/2np1n2/2q1p3/3pP3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 0 1",
            limit: LimitKind::Depth(3),
            history: None,
            locked: None,
        },
        // 7. rook-pawn endgame (low branching)
        Fixture {
            id: "rook-pawn",
            fen: "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1",
            limit: LimitKind::Depth(5),
            history: None,
            locked: None,
        },
        // 8. KQK low-branch endgame
        Fixture {
            id: "kqk",
            fen: "7k/8/8/8/8/8/3QK3/8 w - - 0 1",
            limit: LimitKind::Depth(5),
            history: None,
            locked: None,
        },
        // 9. KRK low-branch endgame
        Fixture {
            id: "krk",
            fen: "7k/8/8/8/8/8/3RK3/8 w - - 0 1",
            limit: LimitKind::Depth(5),
            history: None,
            locked: None,
        },
        // 10. halfmove / repetition context-sensitive (high halfmove_clock in FEN)
        Fixture {
            id: "halfmove-ctx",
            fen: "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 80 5",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
    ]
}

fn throughput_fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            id: "startpos",
            fen: START_FEN,
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        Fixture {
            id: "open-tactical",
            fen: "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
        Fixture {
            id: "queen-win",
            fen: "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1",
            limit: LimitKind::Depth(4),
            history: None,
            locked: None,
        },
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Effective history for a fixture: explicit if provided, else the single root key.
fn effective_history(fx: &Fixture, pos: &Position) -> Vec<ZobristKey> {
    match &fx.history {
        Some(h) => h.clone(),
        None => vec![pos.zobrist_key()],
    }
}

fn limits_for(lk: LimitKind) -> SearchLimits {
    match lk {
        LimitKind::Depth(d) => SearchLimits {
            depth: Some(d),
            nodes: None,
        },
        LimitKind::Nodes(n) => SearchLimits {
            depth: None,
            nodes: Some(n),
        },
    }
}

/// Snapshot of the root position before a search (for full-restoration checks).
struct Snapshot {
    fen: String,
    zobrist: ZobristKey,
}

fn fmt_score(score: Option<i32>) -> String {
    match score {
        None => "none".to_string(),
        // Mate scores sit at/above MATE_THRESHOLD.
        Some(v) if v >= MATE_THRESHOLD => format!("mate:{}", MATE - v),
        Some(v) if v <= -MATE_THRESHOLD => format!("mate:-{}", MATE + v),
        Some(v) => format!("cp:{}", v),
    }
}

fn median_u128(v: &[u128]) -> u128 {
    if v.is_empty() {
        return 0;
    }
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    }
}

fn median_u64(v: &[u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    }
}

/// Format one result line. Stable key order, integers, quoted PV.
fn format_result_line(r: &BenchResult) -> String {
    format!(
        "bench_result suite={} fixture={} mode={} repeat={} limit={} score={} bestmove={} completed_depth={} stopped={} nodes={} elapsed_us={} nps={} pv=\"{}\"",
        r.suite,
        r.fixture,
        r.mode,
        r.repeat,
        r.limit,
        fmt_score(r.score),
        r.best_move,
        r.completed_depth,
        r.stopped,
        r.nodes,
        r.elapsed_us,
        r.nps,
        r.pv
    )
}

/// Validate a completed run: position fully restored, bestmove/PV legal,
/// fixed-depth complete-search invariants, and any locked exact fields.
#[allow(clippy::too_many_arguments)]
fn validate(
    fx: &Fixture,
    mode: BenchMode,
    snap: &Snapshot,
    pos: &Position,
    hist: &[ZobristKey],
    outcome: &SearchOutcome,
    nodes: u64,
    actual_limit: LimitKind,
) -> Result<(), String> {
    // Position must be fully restored to the root.
    if to_fen(pos) != snap.fen {
        return Err(format!(
            "fixture {}: position not restored after search",
            fx.id
        ));
    }
    if pos.zobrist_key() != snap.zobrist {
        return Err(format!(
            "fixture {}: root Zobrist not restored after search",
            fx.id
        ));
    }
    // History contract.
    if hist.is_empty() || hist.last() != Some(&snap.zobrist) {
        return Err(format!("fixture {}: history contract violated", fx.id));
    }
    // bestmove legal at root.
    let legal: Vec<String> = generate_legal_moves(&mut pos.clone())
        .iter()
        .map(|m| move_to_uci(*m))
        .collect();
    let bm = move_to_uci(outcome.best_move);
    if !legal.contains(&bm) {
        return Err(format!("fixture {}: bestmove {} not legal", fx.id, bm));
    }
    // PV legal replay.
    let mut pv_pos =
        parse_fen(fx.fen).map_err(|e| format!("fixture {}: re-parse FEN failed: {}", fx.id, e))?;
    for m in &outcome.pv {
        let uci = move_to_uci(*m);
        let now: Vec<String> = generate_legal_moves(&mut pv_pos)
            .iter()
            .map(|x| move_to_uci(*x))
            .collect();
        if !now.contains(&uci) {
            return Err(format!("fixture {}: PV move {} illegal", fx.id, uci));
        }
        pv_pos.make_move(*m);
    }
    // Fixed-depth complete-search invariants (disabled only).
    if let LimitKind::Depth(d) = actual_limit {
        if mode == BenchMode::Disabled {
            if outcome.stopped {
                return Err(format!(
                    "fixture {} (disabled depth {}): stopped on a fixed-depth complete search",
                    fx.id, d
                ));
            }
            if outcome.completed_depth != d {
                return Err(format!(
                    "fixture {} (disabled depth {}): completed_depth {} != requested",
                    fx.id, d, outcome.completed_depth
                ));
            }
            if outcome.score.is_none() {
                return Err(format!(
                    "fixture {} (disabled depth {}): score is None",
                    fx.id, d
                ));
            }
        }
    }
    // Locked exact assertions (disabled only).
    if mode == BenchMode::Disabled {
        if let Some(locked) = &fx.locked {
            if outcome.score != Some(locked.score) {
                return Err(format!(
                    "fixture {}: locked score {} != {}",
                    fx.id,
                    outcome.score.unwrap_or(0),
                    locked.score
                ));
            }
            if bm != locked.best_move {
                return Err(format!(
                    "fixture {}: locked bestmove {} != {}",
                    fx.id, bm, locked.best_move
                ));
            }
            if nodes != locked.nodes {
                return Err(format!(
                    "fixture {}: locked nodes {} != {}",
                    fx.id, nodes, locked.nodes
                ));
            }
            let pv_uci: Vec<String> = outcome.pv.iter().map(|m| move_to_uci(*m)).collect();
            let want: Vec<String> = locked.pv.iter().map(|s| s.to_string()).collect();
            if pv_uci != want {
                return Err(format!(
                    "fixture {}: locked PV {:?} != {:?}",
                    fx.id, pv_uci, want
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core run
// ---------------------------------------------------------------------------

/// Run one (fixture, mode, repeat) measurement.
fn run_one(
    cfg: &BenchArgs,
    fx: &Fixture,
    mode: BenchMode,
    repeat: u32,
) -> Result<BenchResult, String> {
    // The limit applied: throughput always uses a node budget; otherwise the
    // fixture's depth.
    let actual_limit = match cfg.suite {
        Suite::Throughput => LimitKind::Nodes(cfg.nodes),
        _ => fx.limit,
    };

    // Build the TT for this mode.
    let mut tt = match mode {
        BenchMode::Disabled => TranspositionTable::disabled(),
        _ => TranspositionTable::new_mb(16)
            .map_err(|e| format!("fixture {}: failed to allocate 16MB TT: {}", fx.id, e))?,
    };

    // Warm-up (warm mode only) — not counted in the measured result.
    if mode == BenchMode::Warm {
        let mut pos =
            parse_fen(fx.fen).map_err(|e| format!("fixture {}: invalid FEN: {}", fx.id, e))?;
        let hist = effective_history(fx, &pos);
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = limits_for(actual_limit);
        search_best_move_with_history_and_tt(&mut pos, &hist, &limits, &ctx, &mut tt);
    }

    // Measured run: fresh Position / SearchContext / history. Same TT for warm.
    let mut pos =
        parse_fen(fx.fen).map_err(|e| format!("fixture {}: invalid FEN: {}", fx.id, e))?;
    let hist = effective_history(fx, &pos);
    let snap = Snapshot {
        fen: to_fen(&pos),
        zobrist: pos.zobrist_key(),
    };
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = limits_for(actual_limit);

    let start = Instant::now();
    let outcome = search_best_move_with_history_and_tt(&mut pos, &hist, &limits, &ctx, &mut tt)
        .ok_or_else(|| format!("fixture {}: no legal moves (terminal root)", fx.id))?;
    let elapsed = start.elapsed();
    let nodes = ctx.nodes.load(Ordering::Relaxed);

    validate(fx, mode, &snap, &pos, &hist, &outcome, nodes, actual_limit)?;

    let elapsed_us = elapsed.as_micros();
    let nps = if elapsed_us > 0 {
        ((nodes as u128 * 1_000_000) / elapsed_us) as u64
    } else {
        0
    };
    let pv_uci: Vec<String> = outcome.pv.iter().map(|m| move_to_uci(*m)).collect();
    let limit_str = match actual_limit {
        LimitKind::Depth(d) => format!("depth:{}", d),
        LimitKind::Nodes(n) => format!("nodes:{}", n),
    };

    Ok(BenchResult {
        suite: cfg.suite.as_str(),
        fixture: fx.id,
        mode: mode.as_str(),
        repeat,
        limit: limit_str,
        score: outcome.score,
        best_move: move_to_uci(outcome.best_move),
        completed_depth: outcome.completed_depth,
        stopped: outcome.stopped,
        nodes,
        elapsed_us,
        nps,
        pv: pv_uci.join(" "),
    })
}

/// Check deterministic fields for consistency across repeats of the same
/// (fixture, mode). On mismatch, emit a clear bench_error (warning).
fn check_determinism(results: &[BenchResult]) {
    use std::collections::HashMap;
    let mut groups: HashMap<(&str, &str), Vec<&BenchResult>> = HashMap::new();
    for r in results {
        groups.entry((r.fixture, r.mode)).or_default().push(r);
    }
    for (key, group) in &groups {
        if group.len() <= 1 {
            continue;
        }
        let nodes_set: BTreeSet<u64> = group.iter().map(|r| r.nodes).collect();
        let score_set: BTreeSet<String> = group.iter().map(|r| fmt_score(r.score)).collect();
        let bm_set: BTreeSet<&str> = group.iter().map(|r| r.best_move.as_str()).collect();
        let cd_set: BTreeSet<u32> = group.iter().map(|r| r.completed_depth).collect();
        if nodes_set.len() > 1 || score_set.len() > 1 || bm_set.len() > 1 || cd_set.len() > 1 {
            eprintln!(
                "bench_error determinism: fixture={} mode={} differs across repeats \
                 (nodes={:?} score={:?} bestmove={:?} depth={:?})",
                key.0, key.1, nodes_set, score_set, bm_set, cd_set
            );
        }
    }
}

fn print_summary(suite: Suite, modes: &[BenchMode], fixtures: &[Fixture], results: &[BenchResult]) {
    let aggregate_nodes: u64 = results.iter().map(|r| r.nodes).sum();
    let mut elapsed_vec: Vec<u128> = results.iter().map(|r| r.elapsed_us).collect();
    let mut nps_vec: Vec<u64> = results.iter().map(|r| r.nps).collect();
    elapsed_vec.sort_unstable();
    nps_vec.sort_unstable();
    let median_elapsed = median_u128(&elapsed_vec);
    let median_nps = median_u64(&nps_vec);

    let mode_str = if modes.len() == 1 {
        modes[0].as_str()
    } else {
        "all"
    };

    check_determinism(results);

    println!(
        "bench_summary suite={} mode={} fixture_count={} measured_run_count={} aggregate_nodes={} median_elapsed_us={} median_nps={}",
        suite.as_str(),
        mode_str,
        fixtures.len(),
        results.len(),
        aggregate_nodes,
        median_elapsed,
        median_nps
    );
}

/// Public, strictly limited entry used by `main.rs`.
pub fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "help" {
        print_help();
        return Ok(());
    }
    let cfg = parse_args(args)?;
    let fixtures: Vec<Fixture> = match cfg.suite {
        Suite::Smoke => smoke_fixtures(),
        Suite::Standard => standard_fixtures(),
        Suite::Throughput => throughput_fixtures(),
    };
    let modes: Vec<BenchMode> = match cfg.mode {
        BenchMode::All => vec![BenchMode::Disabled, BenchMode::Cold, BenchMode::Warm],
        m => vec![m],
    };

    let mut results: Vec<BenchResult> = Vec::new();
    for fx in &fixtures {
        for &mode in &modes {
            for r in 0..cfg.repeat {
                let res = run_one(&cfg, fx, mode, r + 1)?;
                println!("{}", format_result_line(&res));
                results.push(res);
            }
        }
    }
    print_summary(cfg.suite, &modes, &fixtures, &results);
    Ok(())
}

fn print_help() {
    println!("bench - M4.0 deterministic search measurement harness");
    println!();
    println!("USAGE:");
    println!("  cargo run --release -- bench <suite> [options]");
    println!();
    println!("SUITES:");
    println!("  smoke       fixed-depth disabled baseline on locked fixtures (depth 3)");
    println!("  standard    10 single-position fixtures, modes per --mode (default all)");
    println!("  throughput  fixed-node NPS measurement (default nodes 100000, repeat 3)");
    println!("  help        this message");
    println!();
    println!("OPTIONS:");
    println!("  --mode <disabled|cold|warm|all>  default: smoke=disabled, standard=all, throughput=disabled");
    println!("  --repeat <N>                       default: smoke=1, standard=1, throughput=3");
    println!("  --nodes <N>                       throughput node budget (default 100000)");
    println!();
    println!("OUTPUT PREFIXES: bench_result / bench_summary / bench_error");
}

// ---------------------------------------------------------------------------
// Tests — fast only; no full suites, no wall-clock assertions.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn all_ids() -> Vec<&'static str> {
        let mut ids: Vec<&'static str> = Vec::new();
        for f in smoke_fixtures() {
            ids.push(f.id);
        }
        for f in standard_fixtures() {
            ids.push(f.id);
        }
        for f in throughput_fixtures() {
            ids.push(f.id);
        }
        ids
    }

    #[test]
    fn fixture_ids_unique() {
        // IDs are unique within each suite. The same position may be
        // reused across suites (e.g. startpos in smoke/standard/throughput),
        // so uniqueness is checked per suite, not globally.
        for (name, list) in [
            ("smoke", smoke_fixtures()),
            ("standard", standard_fixtures()),
            ("throughput", throughput_fixtures()),
        ] {
            let ids: Vec<&str> = list.iter().map(|f| f.id).collect();
            let set: BTreeSet<&str> = ids.iter().copied().collect();
            assert_eq!(set.len(), ids.len(), "{} fixture IDs must be unique", name);
        }
    }

    #[test]
    fn all_fens_parse() {
        for f in all_ids() {
            let fx = all_by_id(f);
            assert!(
                parse_fen(fx.fen).is_ok(),
                "fixture {} FEN must parse: {}",
                f,
                fx.fen
            );
        }
    }

    #[test]
    fn history_nonempty_last_matches() {
        // Derived (None) history.
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: None,
            locked: None,
        };
        let pos = parse_fen(fx.fen).unwrap();
        let h = effective_history(&fx, &pos);
        assert!(!h.is_empty());
        assert_eq!(h.last(), Some(&pos.zobrist_key()));

        // Explicit history with a matching final key.
        let key = pos.zobrist_key();
        let fx2 = Fixture {
            id: "t2",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: Some(vec![key]),
            locked: None,
        };
        let h2 = effective_history(&fx2, &pos);
        assert_eq!(h2, vec![key]);
    }

    #[test]
    fn parse_valid_defaults() {
        let a = parse_args(&["smoke".to_string()]).unwrap();
        assert_eq!(a.suite, Suite::Smoke);
        assert_eq!(a.mode, BenchMode::Disabled);
        assert_eq!(a.repeat, 1);
        assert_eq!(a.nodes, 100_000);

        let b = parse_args(&["standard".to_string()]).unwrap();
        assert_eq!(b.suite, Suite::Standard);
        assert_eq!(b.mode, BenchMode::All);
        assert_eq!(b.repeat, 1);

        let c = parse_args(&["throughput".to_string()]).unwrap();
        assert_eq!(c.suite, Suite::Throughput);
        assert_eq!(c.mode, BenchMode::Disabled);
        assert_eq!(c.repeat, 3);
        assert_eq!(c.nodes, 100_000);
    }

    #[test]
    fn parse_valid_flags() {
        let a = parse_args(&[
            "standard".to_string(),
            "--mode".to_string(),
            "warm".to_string(),
            "--repeat".to_string(),
            "2".to_string(),
            "--nodes".to_string(),
            "50000".to_string(),
        ])
        .unwrap();
        assert_eq!(a.mode, BenchMode::Warm);
        assert_eq!(a.repeat, 2);
        assert_eq!(a.nodes, 50_000);
    }

    #[test]
    fn parse_invalid_no_panic() {
        let cases: Vec<Vec<String>> = vec![
            vec!["bogus".to_string()],
            vec!["smoke".to_string(), "--mode".to_string(), "x".to_string()],
            vec!["smoke".to_string(), "--repeat".to_string(), "0".to_string()],
            vec![
                "smoke".to_string(),
                "--repeat".to_string(),
                "abc".to_string(),
            ],
            vec![
                "throughput".to_string(),
                "--nodes".to_string(),
                "0".to_string(),
            ],
            vec!["smoke".to_string(), "--bogus".to_string()],
        ];
        for c in cases {
            assert!(parse_args(&c).is_err(), "expected error for args {:?}", c);
        }
    }

    #[test]
    fn median_correct() {
        assert_eq!(median_u128(&[1, 2, 3]), 2);
        assert_eq!(median_u128(&[1, 2, 3, 4]), 2);
        assert_eq!(median_u128(&[]), 0);
        assert_eq!(median_u64(&[10, 20, 30]), 20);
    }

    #[test]
    fn score_format_correct() {
        assert_eq!(fmt_score(None), "none");
        assert_eq!(fmt_score(Some(0)), "cp:0");
        assert_eq!(fmt_score(Some(50)), "cp:50");
        assert_eq!(fmt_score(Some(890)), "cp:890");
        assert_eq!(fmt_score(Some(MATE - 1)), "mate:1");
        assert_eq!(fmt_score(Some(-(MATE - 1))), "mate:-1");
        assert_eq!(fmt_score(Some(MATE_THRESHOLD)), "mate:1000");
    }

    #[test]
    fn result_line_format_stable() {
        let r = BenchResult {
            suite: "standard",
            fixture: "startpos",
            mode: "disabled",
            repeat: 1,
            limit: "depth:4".to_string(),
            score: Some(0),
            best_move: "b1c3".to_string(),
            completed_depth: 4,
            stopped: false,
            nodes: 8453,
            elapsed_us: 413_000,
            nps: 20467,
            pv: "b1c3 b8c6 g1f3".to_string(),
        };
        let line = format_result_line(&r);
        assert!(line.starts_with("bench_result "));
        // Fixed key order.
        let i = |k: &str| line.find(k).unwrap_or(usize::MAX);
        assert!(i("suite=") < i("fixture="));
        assert!(i("fixture=") < i("mode="));
        assert!(i("mode=") < i("repeat="));
        assert!(i("repeat=") < i("limit="));
        assert!(i("limit=") < i("score="));
        assert!(i("score=") < i("bestmove="));
        assert!(i("bestmove=") < i("completed_depth="));
        assert!(i("completed_depth=") < i("stopped="));
        assert!(i("stopped=") < i("nodes="));
        assert!(i("nodes=") < i("elapsed_us="));
        assert!(i("elapsed_us=") < i("nps="));
        assert!(i("nps=") < i("pv="));
        assert!(line.contains("score=cp:0"));
        assert!(line.ends_with("pv=\"b1c3 b8c6 g1f3\""));
    }

    #[test]
    fn tiny_search_validates_restore() {
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: None,
            locked: None,
        };
        let cfg = BenchArgs {
            suite: Suite::Standard,
            mode: BenchMode::Disabled,
            repeat: 1,
            nodes: 100_000,
        };
        let r = run_one(&cfg, &fx, BenchMode::Disabled, 1).unwrap();
        assert_eq!(r.completed_depth, 1);
        assert!(!r.stopped);
        assert!(r.score.is_some());
        // run_one validates bestmove/PV legality and full restoration internally.
        assert!(!r.best_move.is_empty());
    }

    // Small helper to look up a fixture by id without leaking the lists.
    fn all_by_id(id: &str) -> Fixture {
        for f in smoke_fixtures() {
            if f.id == id {
                return f;
            }
        }
        for f in standard_fixtures() {
            if f.id == id {
                return f;
            }
        }
        for f in throughput_fixtures() {
            if f.id == id {
                return f;
            }
        }
        panic!("unknown fixture id {}", id);
    }
}
