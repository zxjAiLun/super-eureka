//! M4.0 deterministic search benchmark harness.
//!
//! This module does **not** change search semantics. It only *drives*
//! existing search entries and records correctness + performance fields for
//! later comparison:
//! - `reference` (the default) calls the exact M4.0 entry
//!   ([`search_best_move_with_history_and_tt`]), preserving the historical
//!   baseline byte-for-byte;
//! - `m4.1` and `current` call the profile-aware entry. `m4.1` selects
//!   [`SearchProfile::M41Reference`] (M4.1 full-window quiet ordering,
//!   no root/non-root PVS); `current` selects [`SearchProfile::Current`]
//!   (M4.1 ordering + PVS). `reference` uses the exact M4.0 entry.
//!
//! The harness itself never alters search behavior.
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
use std::time::{Duration, Instant};

use crate::chess::fen::{parse_fen, to_fen};
use crate::chess::move_to_uci;
use crate::chess::movegen::{generate_legal_moves, generate_pseudo_moves};
use crate::chess::position::Position;
use crate::chess::types::{MoveFlag, START_FEN};
use crate::chess::Move;
use crate::chess::ZobristKey;
use crate::engine::eval::evaluate_components_white;
use crate::engine::search::{
    search_best_move_with_history_and_tt, search_best_move_with_history_tt_and_profile,
    SearchContext, SearchDiagnostics, SearchLimits, SearchOutcome, SearchProfile, SearchStats,
    MATE,
};
use crate::engine::time::{compute_budget, TimeInput};
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
    Profile,
    Ablation,
}

impl Suite {
    fn as_str(self) -> &'static str {
        match self {
            Suite::Smoke => "smoke",
            Suite::Standard => "standard",
            Suite::Throughput => "throughput",
            Suite::Profile => "profile",
            Suite::Ablation => "ablation",
        }
    }
}

/// The search limit actually applied to a run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LimitKind {
    Depth(u32),
    Nodes(u64),
    Movetime(u64),
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
#[derive(Clone)]
struct BenchArgs {
    suite: Suite,
    mode: BenchMode,
    repeat: u32,
    /// Throughput/profile node budget (default 100_000).
    nodes: u64,
    /// Search profile (default reference == M4.0 baseline behavior).
    profile: SearchProfile,
    /// Optional throughput/profile fixture filter.
    fixture: Option<&'static str>,
    /// Optional one-off FEN for a profile run. The CLI process owns the
    /// leaked string for its lifetime; this keeps the existing static Fixture
    /// representation intact for the built-in suites.
    custom_fen: Option<&'static str>,
    /// Optional fixed limit used only by the profile suite.
    profile_limit: Option<LimitKind>,
    /// Limit selector used only by the ablation suite.
    ablation_limit: Option<LimitKind>,
    /// S4.0B bench-only diagnostic overrides (feature ablation). Applied only
    /// to a CurrentFinal profile search; never reachable through UCI.
    diag_lmr: bool,
    diag_futility: bool,
    diag_null: bool,
    diag_qsee: bool,
    /// S4.1c Phase B: every root move gets a full-window child search (no
    /// root scout + conditional re-search). Diagnostic only.
    diag_root_full_window: bool,
    /// S4.3A: sampled wall-time attribution rate (e.g. 256 = 1/256 calls).
    /// Profile suite only; never set on the production UCI path.
    timing_sample: Option<u32>,
    /// S4.0B: force the root to search only this move (UCI), e.g. the teacher move.
    forced_root: Option<String>,
    /// S4.0B: record the 1-based root rank of this move (UCI) under the normal
    /// ordering, before forced-root filtering.
    target_root: Option<String>,
}

/// Render a `SearchProfile` as its CLI string (also used in bench output).
fn profile_str(p: SearchProfile) -> &'static str {
    match p {
        SearchProfile::M4Reference => "reference",
        SearchProfile::M41Reference => "m4.1",
        SearchProfile::PvsReference => "pvs",
        SearchProfile::SeeCandidate => "see",
        SearchProfile::AspirationCandidate => "aspiration",
        SearchProfile::LmrCandidate => "lmr",
        SearchProfile::NullMoveCandidate => "null",
        SearchProfile::FutilityCandidate => "futility",
        SearchProfile::Current => "current",
        SearchProfile::CurrentLmr => "current-lmr",
        SearchProfile::CurrentThreatAware => "current-threat-aware",
        SearchProfile::CurrentThreatAwareNoQchecks => "current-threat-aware-no-qchecks",
        SearchProfile::CurrentThreatAwareEvalOrder => "current-threat-aware-eval-order",
        SearchProfile::CurrentThreatAwareEvalOnly => "current-threat-aware-eval-only",
        SearchProfile::CurrentThreatAwareOrderOnly => "current-threat-aware-order-only",
        SearchProfile::CurrentEval2 => "current-eval2",
        SearchProfile::CurrentQsearchMovegen => "current-qsearch-movegen",
        SearchProfile::CurrentQsearchPruning => "current-qsearch-pruning",
        SearchProfile::CurrentQsearchFastPruning => "current-qsearch-fast-pruning",
        SearchProfile::CurrentAspiration => "current-aspiration",
        SearchProfile::CurrentAspirationLmr => "current-aspiration-lmr",
        SearchProfile::CurrentAspirationLmrFutility => "current-aspiration-lmr-futility",
        SearchProfile::CurrentAspirationLmrFutilitySee => "current-aspiration-lmr-futility-see",
        SearchProfile::CurrentFinal => "current-final",
        SearchProfile::CurrentFinalRootHistory => "current-final-root-history",
        SearchProfile::CurrentFinalRootPrevScore => "current-final-root-prev-score",
        SearchProfile::CurrentFinalLegalityFast => "current-final-legality-fast",
        SearchProfile::CurrentFinalSingleBuffer => "current-final-single-buffer",
        SearchProfile::CurrentFinalSingleGeneration => "current-final-single-generation",
        SearchProfile::CurrentFinalQsearchLazy => "current-final-qsearch-lazy",
        SearchProfile::CurrentFinalQsearchDelta => "current-final-qsearch-delta",
    }
}

/// One measured search result.
struct BenchResult {
    suite: &'static str,
    fixture: &'static str,
    mode: &'static str,
    profile: &'static str,
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
    target_root_rank: u32,
    stats: SearchStats,
}
// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

fn parse_args(args: &[String]) -> Result<BenchArgs, String> {
    let mut it = args.iter();
    let suite_kw = it
        .next()
        .ok_or_else(|| {
            "bench: missing suite (expected smoke|standard|throughput|profile|ablation|help)"
                .to_string()
        })?
        .clone();

    let suite = match suite_kw.as_str() {
        "smoke" => Suite::Smoke,
        "standard" => Suite::Standard,
        "throughput" => Suite::Throughput,
        "profile" => Suite::Profile,
        "ablation" => Suite::Ablation,
        other => {
            return Err(format!(
                "bench: unknown suite '{}' (expected smoke|standard|throughput|profile|ablation|help)",
                other
            ));
        }
    };

    let mut mode = match suite {
        Suite::Smoke => BenchMode::Disabled,
        Suite::Standard => BenchMode::All,
        Suite::Throughput => BenchMode::Disabled,
        Suite::Profile => BenchMode::Disabled,
        Suite::Ablation => BenchMode::Disabled,
    };
    let mut repeat = match suite {
        Suite::Smoke => 1,
        Suite::Standard => 1,
        Suite::Throughput => 3,
        Suite::Profile => 1,
        Suite::Ablation => 1,
    };
    let mut nodes = 100_000u64;
    let mut profile = SearchProfile::M4Reference;
    let mut fixture: Option<&'static str> = None;
    let mut custom_fen: Option<&'static str> = None;
    let mut profile_limit: Option<LimitKind> = None;
    let mut ablation_limit: Option<LimitKind> = None;
    let mut diag_lmr = false;
    let mut diag_futility = false;
    let mut diag_null = false;
    let mut diag_qsee = false;
    let mut diag_root_full_window = false;
    let mut timing_sample: Option<u32> = None;
    let mut forced_root: Option<String> = None;
    let mut target_root: Option<String> = None;

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
                match suite {
                    Suite::Ablation => {
                        if ablation_limit.replace(LimitKind::Nodes(n)).is_some() {
                            return Err(
                                "bench: ablation accepts exactly one of --nodes|--depth|--movetime"
                                    .to_string(),
                            );
                        }
                    }
                    Suite::Profile => {
                        if profile_limit.replace(LimitKind::Nodes(n)).is_some() {
                            return Err(
                                "bench: profile accepts exactly one of --nodes|--depth|--movetime"
                                    .to_string(),
                            );
                        }
                    }
                    _ => {}
                }
            }
            "--depth" => {
                if suite != Suite::Ablation && suite != Suite::Profile {
                    return Err("bench: --depth is only valid for profile or ablation".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --depth requires a value".to_string())?
                    .clone();
                let n: u32 = v
                    .parse()
                    .map_err(|_| format!("bench: --depth '{}' is not a positive integer", v))?;
                if n == 0 {
                    return Err("bench: --depth must be >= 1".to_string());
                }
                if suite == Suite::Ablation {
                    if ablation_limit.replace(LimitKind::Depth(n)).is_some() {
                        return Err(
                            "bench: ablation accepts exactly one of --nodes|--depth|--movetime"
                                .to_string(),
                        );
                    }
                } else if profile_limit.replace(LimitKind::Depth(n)).is_some() {
                    return Err(
                        "bench: profile accepts exactly one of --nodes|--depth|--movetime"
                            .to_string(),
                    );
                }
            }
            "--movetime" => {
                if suite != Suite::Ablation && suite != Suite::Profile {
                    return Err(
                        "bench: --movetime is only valid for profile or ablation".to_string()
                    );
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --movetime requires a value".to_string())?
                    .clone();
                let n: u64 = v
                    .parse()
                    .map_err(|_| format!("bench: --movetime '{}' is not a positive integer", v))?;
                if n == 0 {
                    return Err("bench: --movetime must be >= 1".to_string());
                }
                if suite == Suite::Ablation {
                    if ablation_limit.replace(LimitKind::Movetime(n)).is_some() {
                        return Err(
                            "bench: ablation accepts exactly one of --nodes|--depth|--movetime"
                                .to_string(),
                        );
                    }
                } else if profile_limit.replace(LimitKind::Movetime(n)).is_some() {
                    return Err(
                        "bench: profile accepts exactly one of --nodes|--depth|--movetime"
                            .to_string(),
                    );
                }
            }
            "--profile" => {
                if suite == Suite::Ablation {
                    return Err(
                        "bench: --profile is not valid for ablation; it runs all cumulative profiles"
                            .to_string(),
                    );
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --profile requires a value".to_string())?
                    .clone();
                profile = match v.as_str() {
                    "reference" => SearchProfile::M4Reference,
                    "m4.1" => SearchProfile::M41Reference,
                    "pvs" => SearchProfile::PvsReference,
                    "see" => SearchProfile::SeeCandidate,
                    "aspiration" => SearchProfile::AspirationCandidate,
                    "lmr" => SearchProfile::LmrCandidate,
                    "null" => SearchProfile::NullMoveCandidate,
                    "futility" => SearchProfile::FutilityCandidate,
                    "current" => SearchProfile::Current,
                    "current-lmr" => SearchProfile::CurrentLmr,
                    "current-threat-aware" => SearchProfile::CurrentThreatAware,
                    "current-threat-aware-no-qchecks" => SearchProfile::CurrentThreatAwareNoQchecks,
                    "current-threat-aware-eval-order" => SearchProfile::CurrentThreatAwareEvalOrder,
                    "current-threat-aware-eval-only" => SearchProfile::CurrentThreatAwareEvalOnly,
                    "current-threat-aware-order-only" => SearchProfile::CurrentThreatAwareOrderOnly,
                    "current-eval2" => SearchProfile::CurrentEval2,
                    "current-qsearch-movegen" => SearchProfile::CurrentQsearchMovegen,
                    "current-qsearch-pruning" => SearchProfile::CurrentQsearchPruning,
                    "current-qsearch-fast-pruning" => SearchProfile::CurrentQsearchFastPruning,
                    "current-aspiration" => SearchProfile::CurrentAspiration,
                    "current-aspiration-lmr" => SearchProfile::CurrentAspirationLmr,
                    "current-aspiration-lmr-futility" => {
                        SearchProfile::CurrentAspirationLmrFutility
                    }
                    "current-aspiration-lmr-futility-see" => {
                        SearchProfile::CurrentAspirationLmrFutilitySee
                    }
                    "current-final" => SearchProfile::CurrentFinal,
                    "current-final-root-history" => SearchProfile::CurrentFinalRootHistory,
                    "current-final-root-prev-score" => SearchProfile::CurrentFinalRootPrevScore,
                    "current-final-legality-fast" => SearchProfile::CurrentFinalLegalityFast,
                    "current-final-single-buffer" => SearchProfile::CurrentFinalSingleBuffer,
                    "current-final-single-generation" => {
                        SearchProfile::CurrentFinalSingleGeneration
                    }
                    "current-final-qsearch-lazy" => SearchProfile::CurrentFinalQsearchLazy,
                    "current-final-qsearch-delta" => SearchProfile::CurrentFinalQsearchDelta,
                    other => {
                        return Err(format!(
                            "bench: invalid --profile '{}' (expected reference|m4.1|pvs|see|aspiration|lmr|null|futility|current|current-lmr|current-threat-aware|current-threat-aware-no-qchecks|current-threat-aware-eval-order|current-threat-aware-eval-only|current-threat-aware-order-only|current-eval2|current-qsearch-movegen|current-qsearch-pruning|current-qsearch-fast-pruning|current-aspiration|current-aspiration-lmr|current-aspiration-lmr-futility|current-aspiration-lmr-futility-see|current-final|current-final-root-history|current-final-root-prev-score|current-final-legality-fast|current-final-single-buffer|current-final-single-generation|current-final-qsearch-lazy|current-final-qsearch-delta)",
                            other
                        ));
                    }
                };
            }
            "--fixture" => {
                if suite != Suite::Throughput && suite != Suite::Profile && suite != Suite::Ablation
                {
                    return Err(
                        "bench: --fixture is only valid for throughput, profile, or ablation"
                            .to_string(),
                    );
                }
                if custom_fen.is_some() {
                    return Err("bench: --fixture cannot be combined with --fen".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --fixture requires a value".to_string())?
                    .clone();
                if fixture.is_some() {
                    return Err("bench: --fixture may be specified only once".to_string());
                }
                fixture = Some(match v.as_str() {
                    "startpos" => "startpos",
                    "open-tactical" => "open-tactical",
                    "queen-win" => "queen-win",
                    "closed-quiet" => "closed-quiet",
                    "exposed-king" => "exposed-king",
                    "high-branch" => "high-branch",
                    "rook-pawn" => "rook-pawn",
                    "kqk" => "kqk",
                    "krk" => "krk",
                    "halfmove-ctx" => "halfmove-ctx",
                    other => {
                        return Err(format!(
                            "bench: invalid --fixture '{}' (expected standard/profile fixture id)",
                            other
                        ));
                    }
                });
            }
            "--fen" => {
                if suite != Suite::Profile {
                    return Err("bench: --fen is only valid for profile".to_string());
                }
                if fixture.is_some() {
                    return Err("bench: --fen cannot be combined with --fixture".to_string());
                }
                if custom_fen.is_some() {
                    return Err("bench: --fen may be specified only once".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --fen requires a value".to_string())?
                    .clone();
                if v.trim().is_empty() {
                    return Err("bench: --fen requires a non-empty FEN".to_string());
                }
                custom_fen = Some(Box::leak(v.into_boxed_str()));
            }
            "--diag" => {
                if suite != Suite::Profile {
                    return Err("bench: --diag is only valid for profile".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --diag requires a value".to_string())?
                    .clone();
                match v.as_str() {
                    "no-lmr" => diag_lmr = true,
                    "no-futility" => diag_futility = true,
                    "no-null" => diag_null = true,
                    "no-qsee" => diag_qsee = true,
                    "root-full-window" => diag_root_full_window = true,
                    other => {
                        return Err(format!(
                            "bench: invalid --diag '{}' (expected no-lmr|no-futility|no-null|no-qsee|root-full-window)",
                            other
                        ));
                    }
                }
            }
            "--forced-root" => {
                if suite != Suite::Profile {
                    return Err("bench: --forced-root is only valid for profile".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --forced-root requires a value".to_string())?
                    .clone();
                if forced_root.is_some() {
                    return Err("bench: --forced-root may be specified only once".to_string());
                }
                forced_root = Some(v);
            }
            "--target-root" => {
                if suite != Suite::Profile {
                    return Err("bench: --target-root is only valid for profile".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --target-root requires a value".to_string())?
                    .clone();
                if target_root.is_some() {
                    return Err("bench: --target-root may be specified only once".to_string());
                }
                target_root = Some(v);
            }
            "--timing-sample" => {
                if suite != Suite::Profile {
                    return Err("bench: --timing-sample is only valid for profile".to_string());
                }
                let v = it
                    .next()
                    .ok_or_else(|| "bench: --timing-sample requires a value".to_string())?
                    .clone();
                let n: u32 = v.parse().map_err(|_| {
                    format!("bench: --timing-sample '{}' is not a positive integer", v)
                })?;
                if n == 0 {
                    return Err("bench: --timing-sample must be >= 1".to_string());
                }
                if timing_sample.is_some() {
                    return Err("bench: --timing-sample may be specified only once".to_string());
                }
                timing_sample = Some(n);
            }
            other => {
                return Err(format!("bench: unknown argument '{}'", other));
            }
        }
    }

    if suite == Suite::Ablation && mode != BenchMode::Disabled {
        return Err("bench: ablation uses fixed disabled TT mode".to_string());
    }

    Ok(BenchArgs {
        suite,
        mode,
        repeat,
        nodes,
        profile,
        fixture,
        custom_fen,
        profile_limit: if suite == Suite::Profile {
            Some(profile_limit.unwrap_or(LimitKind::Nodes(nodes)))
        } else {
            None
        },
        ablation_limit: if suite == Suite::Ablation {
            Some(ablation_limit.unwrap_or(LimitKind::Nodes(nodes)))
        } else {
            None
        },
        diag_lmr,
        diag_futility,
        diag_null,
        diag_qsee,
        diag_root_full_window,
        forced_root,
        target_root,
        timing_sample,
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
                nodes: 969,
                score: 990,
                best_move: "e4a4",
                pv: &["e4a4", "h4h3", "a4h4", "h8g7", "h4h3"],
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

/// Dispatch one search through the selected profile.
///
/// - `reference` calls the exact M4.0 entry
///   ([`search_best_move_with_history_and_tt`]), preserving the historical
///   baseline byte-for-byte;
/// - `m4.1` and `current` call the profile-aware entry. `m4.1` selects
///   [`SearchProfile::M41Reference`] (M4.1 full-window quiet ordering,
///   no PVS); `current` selects [`SearchProfile::Current`] (M4.1 ordering
///   + PVS). `reference` uses the exact M4.0 entry.
///
/// Commit 5 keeps `search.rs` untouched: it only selects which *existing*
/// entry to drive, and never alters search semantics.
fn search_one(
    pos: &mut Position,
    hist: &[ZobristKey],
    limits: &SearchLimits,
    ctx: &SearchContext,
    tt: &mut TranspositionTable,
    profile: SearchProfile,
) -> Option<SearchOutcome> {
    if profile == SearchProfile::M4Reference {
        search_best_move_with_history_and_tt(pos, hist, limits, ctx, tt)
    } else {
        search_best_move_with_history_tt_and_profile(pos, hist, limits, ctx, tt, profile)
    }
}

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
        LimitKind::Movetime(_) => SearchLimits {
            depth: None,
            nodes: None,
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
    let line = format!(
        "bench_result suite={} fixture={} mode={} profile={} repeat={} limit={} score={} bestmove={} completed_depth={} stopped={} nodes={} elapsed_us={} nps={} pv=\"{}\" target_root_rank={}",
        r.suite,
        r.fixture,
        r.mode,
        r.profile,
        r.repeat,
        r.limit,
        fmt_score(r.score),
        r.best_move,
        r.completed_depth,
        r.stopped,
        r.nodes,
        r.elapsed_us,
        r.nps,
        r.pv,
        r.target_root_rank
    );
    let line = if r.suite == "profile" || r.suite == "ablation" {
        format!("{} elapsed_ms={}", line, r.elapsed_us / 1_000)
    } else {
        line
    };
    if r.suite == "profile" || r.suite == "ablation" {
        format!(
            "{} total_nodes={} completed_iterations={} nodes_per_completed_depth={} qsearch_ratio={:.6} effective_branching_factor={:.6} last_completed_iteration_ms={} last_completed_iteration_nodes={} aborted_iteration_depth={} aborted_iteration_nodes={} qsearch_nodes={} eval_calls={} legal_move_generations={} pseudo_moves={} legal_moves={} make_moves={} unmake_moves={} tt_probes={} tt_hits={} tt_cutoffs={} tt_rejected_depth={} tt_rejected_bound={} tt_rejected_decode={} tt_stores={} see_calls={} see_pruned={} qsearch_see_tests={} qsearch_see_pruned={} qsearch_see_fail_open_promotions={} qsearch_checking_captures_kept={} qsearch_promotions_kept={} qsearch_en_passant_kept={} check_extensions={} single_evasion_extensions={} qsearch_check_moves={} threat_ordered_moves={} root_reorders={} aspiration_retries={} aspiration_fail_low={} aspiration_fail_high={} lmr_reductions={} lmr_researches={} null_move_attempts={} null_move_fail_highs={} null_move_researches={} futility_pruned={} legality_fast_accepts={} legality_fallback_probes={} legality_fallback_in_check={} legality_fallback_king={} legality_fallback_pinned={} legality_fallback_en_passant={} legality_fallback_castle={} single_buffer_writes={}",
            line,
            r.nodes,
            r.stats.completed_iterations,
            nodes_per_completed_depth(r.nodes, r.completed_depth),
            qsearch_ratio(r.stats.qsearch_nodes, r.nodes),
            effective_branching_factor(r.nodes, r.completed_depth),
            r.stats.last_completed_iteration_ms,
            r.stats.last_completed_iteration_nodes,
            r.stats.aborted_iteration_depth,
            r.stats.aborted_iteration_nodes,
            r.stats.qsearch_nodes,
            r.stats.eval_calls,
            r.stats.legal_move_generations,
            r.stats.pseudo_moves,
            r.stats.legal_moves,
            r.stats.make_moves,
            r.stats.unmake_moves,
            r.stats.tt_probes,
            r.stats.tt_hits,
            r.stats.tt_cutoffs,
            r.stats.tt_rejected_depth,
            r.stats.tt_rejected_bound,
            r.stats.tt_rejected_decode,
            r.stats.tt_stores,
            r.stats.see_calls,
            r.stats.see_pruned,
            r.stats.qsearch_see_tests,
            r.stats.qsearch_see_pruned,
            r.stats.qsearch_see_fail_open_promotions,
            r.stats.qsearch_checking_captures_kept,
            r.stats.qsearch_promotions_kept,
            r.stats.qsearch_en_passant_kept,
            r.stats.check_extensions,
            r.stats.single_evasion_extensions,
            r.stats.qsearch_check_moves,
            r.stats.threat_ordered_moves,
            r.stats.root_reorders,
            r.stats.aspiration_retries,
            r.stats.aspiration_fail_low,
            r.stats.aspiration_fail_high,
            r.stats.lmr_reductions,
            r.stats.lmr_researches,
            r.stats.null_move_attempts,
            r.stats.null_move_fail_highs,
            r.stats.null_move_researches,
            r.stats.futility_pruned,
            r.stats.legality_fast_accepts,
            r.stats.legality_fallback_probes,
            r.stats.legality_fallback_in_check,
            r.stats.legality_fallback_king,
            r.stats.legality_fallback_pinned,
            r.stats.legality_fallback_en_passant,
            r.stats.legality_fallback_castle,
            r.stats.single_buffer_writes,
        ) + format_s7_attribution(&r.stats).as_str()
    } else {
        line
    }
}

/// S7.0 depth-attribution counters, appended to the `bench_result` line for
/// the profile/ablation suites (profiling enabled). Observation-only.
fn format_s7_attribution(stats: &SearchStats) -> String {
    format!(
        " seldepth={} main_seldepth={} qsearch_seldepth={} beta_cutoffs={} beta_cutoff_idx_0={} beta_cutoff_idx_1={} beta_cutoff_idx_2_3={} beta_cutoff_idx_4_7={} beta_cutoff_idx_8_15={} beta_cutoff_idx_16p={} cutoff_tt_move={} cutoff_tactical={} cutoff_killer={} cutoff_quiet={} moves_searched={} pv_nodes={} in_check_nodes={} depth_bucket_0={} depth_bucket_1={} depth_bucket_2={} depth_bucket_3={} depth_bucket_4_5={} depth_bucket_6_7={} depth_bucket_8p={} searched_hist_1={} searched_hist_2={} searched_hist_3_4={} searched_hist_5_8={} searched_hist_9_16={} searched_hist_17p={} tt_hit_exact={} tt_hit_lower={} tt_hit_upper={} lmr_reduction_r1={} lmr_reduction_r2={} lmr_reduced_improves_alpha={} null_fail_lows={} futility_considered={} qsearch_standpat_cutoffs={} qsearch_standpat_alpha_raises={} qsearch_moves_searched={} qsearch_in_check_entries={} qsearch_lazy_has_any_probes={} qsearch_lazy_standpat_cutoffs_before_movegen={} qsearch_lazy_qply_returns_before_movegen={} qsearch_lazy_tactical_generations={} delta_tests={} delta_pruned={} delta_pruned_pawn={} delta_pruned_minor={} delta_pruned_rook={} delta_pruned_queen={} delta_qply_0_1={} delta_qply_2_3={} delta_qply_4p={}",
        stats.seldepth,
        stats.main_seldepth,
        stats.qsearch_seldepth,
        stats.beta_cutoffs,
        stats.beta_cutoff_idx_0,
        stats.beta_cutoff_idx_1,
        stats.beta_cutoff_idx_2_3,
        stats.beta_cutoff_idx_4_7,
        stats.beta_cutoff_idx_8_15,
        stats.beta_cutoff_idx_16p,
        stats.cutoff_tt_move,
        stats.cutoff_tactical,
        stats.cutoff_killer,
        stats.cutoff_quiet,
        stats.moves_searched,
        stats.pv_nodes,
        stats.in_check_nodes,
        stats.depth_bucket_0,
        stats.depth_bucket_1,
        stats.depth_bucket_2,
        stats.depth_bucket_3,
        stats.depth_bucket_4_5,
        stats.depth_bucket_6_7,
        stats.depth_bucket_8p,
        stats.searched_hist_1,
        stats.searched_hist_2,
        stats.searched_hist_3_4,
        stats.searched_hist_5_8,
        stats.searched_hist_9_16,
        stats.searched_hist_17p,
        stats.tt_hit_exact,
        stats.tt_hit_lower,
        stats.tt_hit_upper,
        stats.lmr_reduction_r1,
        stats.lmr_reduction_r2,
        stats.lmr_reduced_improves_alpha,
        stats.null_fail_lows,
        stats.futility_considered,
        stats.qsearch_standpat_cutoffs,
        stats.qsearch_standpat_alpha_raises,
        stats.qsearch_moves_searched,
        stats.qsearch_in_check_entries,
        stats.qsearch_lazy_has_any_probes,
        stats.qsearch_lazy_standpat_cutoffs_before_movegen,
        stats.qsearch_lazy_qply_returns_before_movegen,
        stats.qsearch_lazy_tactical_generations,
        stats.qsearch_delta_tests,
        stats.qsearch_delta_pruned,
        stats.qsearch_delta_pruned_pawn,
        stats.qsearch_delta_pruned_minor,
        stats.qsearch_delta_pruned_rook,
        stats.qsearch_delta_pruned_queen,
        stats.qsearch_delta_qply_0_1,
        stats.qsearch_delta_qply_2_3,
        stats.qsearch_delta_qply_4p,
    )
}

#[inline]
fn nodes_per_completed_depth(nodes: u64, completed_depth: u32) -> u64 {
    if completed_depth == 0 {
        0
    } else {
        nodes / u64::from(completed_depth)
    }
}

#[inline]
fn qsearch_ratio(qsearch_nodes: u64, nodes: u64) -> f64 {
    if nodes == 0 {
        0.0
    } else {
        qsearch_nodes as f64 / nodes as f64
    }
}

#[inline]
fn effective_branching_factor(nodes: u64, completed_depth: u32) -> f64 {
    if nodes == 0 || completed_depth == 0 {
        0.0
    } else {
        (nodes as f64).powf(1.0 / f64::from(completed_depth))
    }
}

/// Check the fixed-depth complete-search invariants for a *completed* search.
/// Used for both the measured run and the warm-up, in EVERY mode
/// (disabled / cold / warm). A fixed-depth search must reach the requested
/// depth, must not stop early, and must return a real (non-None) score.
/// This is what keeps the cold/warm node counts a trustworthy regression
/// base — a partial fixed-depth result would otherwise be mistaken for a
/// genuine node-count improvement.
fn check_fixed_depth_complete(
    fx: &Fixture,
    label: &str,
    out: &SearchOutcome,
    d: u32,
) -> Result<(), String> {
    if out.stopped {
        return Err(format!(
            "fixture {} ({} depth {}): stopped on a fixed-depth complete search",
            fx.id, label, d
        ));
    }
    if out.completed_depth != d {
        return Err(format!(
            "fixture {} ({} depth {}): completed_depth {} != requested",
            fx.id, label, d, out.completed_depth
        ));
    }
    if out.score.is_none() {
        return Err(format!(
            "fixture {} ({} depth {}): score is None",
            fx.id, label, d
        ));
    }
    Ok(())
}

/// Validate a completed run: position fully restored, bestmove/PV legal,
/// fixed-depth complete-search invariants, and any locked exact fields.
#[allow(clippy::too_many_arguments)]
fn validate(
    fx: &Fixture,
    mode: BenchMode,
    profile: SearchProfile,
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
    // Fixed-depth complete-search invariants. Enforced for EVERY mode
    // (disabled / cold / warm): a fixed-depth *measured* search must reach
    // the requested depth, must not stop early, and must return a real
    // (non-None) score. This is what keeps the cold/warm node counts a
    // trustworthy regression base — a partial fixed-depth result would
    // otherwise be mistaken for a genuine node-count improvement.
    if let LimitKind::Depth(d) = actual_limit {
        check_fixed_depth_complete(fx, mode.as_str(), outcome, d)?;
    }
    // Node-budget measured-run invariant. There is no deadline and the stop
    // flag starts false, so a budgeted run must stop exactly at the budget:
    // consumed nodes must not exceed the budget, the outcome must report
    // stopped, and the consumed count must equal the budget (atomic budget
    // semantic). Unlike the determinism check (a soft warning), a budget
    // violation is a fatal validation error: a partial/over-budget result
    // would be mistaken for a genuine throughput measurement, so `validate`
    // returns `Err`, `run_one` propagates it, the offending `bench_result`
    // is never printed, it is excluded from `bench_summary`, and the CLI
    // exits non-zero.
    if let LimitKind::Nodes(n) = actual_limit {
        if nodes > n {
            return Err(format!(
                "fixture {}: node count {} exceeds budget {}",
                fx.id, nodes, n
            ));
        }
        if !outcome.stopped {
            return Err(format!(
                "fixture {}: node-budget search did not report stopped",
                fx.id
            ));
        }
        if nodes != n {
            return Err(format!(
                "fixture {}: node count {} != requested budget {}",
                fx.id, nodes, n
            ));
        }
    }
    // Locked exact assertions (disabled mode, reference profile only).
    // `1149`/`963` precise locks belong to `M4Reference`; under `Current`
    // a fixed-depth run may legitimately produce different node counts,
    // bestmoves, and PVs, so the lock must not be applied.
    if mode == BenchMode::Disabled && profile == SearchProfile::M4Reference {
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

/// Run + validate the warm-up search (warm mode only). The warm-up is NOT
/// counted in the measured result; it only populates the (shared) TT. If it
/// does not complete successfully, we return a clear error and the caller
/// (`run_one`) propagates it, so the measured run never executes on a
/// half-warmed TT — which would corrupt the cold/warm comparison.
fn run_warmup(
    fx: &Fixture,
    actual_limit: LimitKind,
    profile: SearchProfile,
    tt: &mut TranspositionTable,
) -> Result<(), String> {
    let mut pos =
        parse_fen(fx.fen).map_err(|e| format!("fixture {}: invalid FEN: {}", fx.id, e))?;
    let hist = effective_history(fx, &pos);
    let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
    let limits = limits_for(actual_limit);
    let out = search_one(&mut pos, &hist, &limits, &ctx, tt, profile).ok_or_else(|| {
        format!(
            "fixture {} (warm-up): no legal moves (terminal root)",
            fx.id
        )
    })?;
    if out.score.is_none() {
        return Err(format!("fixture {} (warm-up): score is None", fx.id));
    }
    // A fixed-depth warm-up must finish the requested depth. A node-budget
    // warm-up (throughput suite) may stop mid-iteration at the budget, so
    // we do NOT require a specific completed_depth there.
    if let LimitKind::Depth(d) = actual_limit {
        check_fixed_depth_complete(fx, "warm-up", &out, d)?;
    }
    Ok(())
}

/// Run one (fixture, mode, repeat) measurement.
fn run_one(
    cfg: &BenchArgs,
    fx: &Fixture,
    mode: BenchMode,
    repeat: u32,
) -> Result<BenchResult, String> {
    // The limit applied: throughput/profile always uses a node budget; otherwise the
    // fixture's depth.
    let actual_limit = match cfg.suite {
        Suite::Throughput => LimitKind::Nodes(cfg.nodes),
        Suite::Profile => cfg
            .profile_limit
            .expect("profile limit is populated by parse_args"),
        Suite::Ablation => cfg
            .ablation_limit
            .expect("ablation limit is populated by parse_args"),
        _ => fx.limit,
    };

    // Build the TT for this mode.
    let mut tt = match mode {
        BenchMode::Disabled => TranspositionTable::disabled(),
        _ => TranspositionTable::new_mb(16)
            .map_err(|e| format!("fixture {}: failed to allocate 16MB TT: {}", fx.id, e))?,
    };

    // Warm-up (warm mode only) — not counted in the measured result.
    // The warm-up must itself complete a full fixed-depth search (or, for the
    // node-budget throughtput suite, finish with a usable score); if it does
    // not, we must NOT treat `tt` as a fully-warmed table, so we bail with
    // a clear error instead of measuring on a half-warmed TT.
    if mode == BenchMode::Warm {
        run_warmup(fx, actual_limit, cfg.profile, &mut tt)?;
    }

    // Measured run: fresh Position / SearchContext / history. Same TT for warm.
    let mut pos =
        parse_fen(fx.fen).map_err(|e| format!("fixture {}: invalid FEN: {}", fx.id, e))?;
    let hist = effective_history(fx, &pos);
    let snap = Snapshot {
        fen: to_fen(&pos),
        zobrist: pos.zobrist_key(),
    };
    let profiling = matches!(cfg.suite, Suite::Profile | Suite::Ablation);
    let stop = Arc::new(AtomicBool::new(false));
    let mut ctx = match actual_limit {
        LimitKind::Movetime(ms) => {
            let budget = compute_budget(
                &TimeInput {
                    movetime: Some(Duration::from_millis(ms)),
                    ..TimeInput::default()
                },
                Instant::now(),
            );
            SearchContext::with_budget_and_profiling(stop, budget, profiling)
        }
        _ => SearchContext::new_with_profiling(stop, profiling),
    };

    // S4.0B bench-only diagnostics. Only active for a profile search; the
    // production UCI path never constructs a `SearchContext` with these set.
    if cfg.diag_lmr
        || cfg.diag_futility
        || cfg.diag_null
        || cfg.diag_qsee
        || cfg.diag_root_full_window
        || cfg.forced_root.is_some()
        || cfg.target_root.is_some()
    {
        let legal = generate_legal_moves(&mut pos.clone());
        let resolve = |label: &str, uci: &str| -> Result<Move, String> {
            legal
                .iter()
                .copied()
                .find(|m| move_to_uci(*m) == uci)
                .ok_or_else(|| {
                    format!("--{} move {} is not legal in fixture {}", label, uci, fx.id)
                })
        };
        let forced_root_move = match &cfg.forced_root {
            Some(u) => Some(resolve("forced-root", u)?),
            None => None,
        };
        let target_root_move = match &cfg.target_root {
            Some(u) => Some(resolve("target-root", u)?),
            None => None,
        };
        ctx.diagnostics = Some(SearchDiagnostics {
            forced_root_move,
            target_root_move,
            root_full_window: cfg.diag_root_full_window,
            disable_lmr: cfg.diag_lmr,
            disable_futility: cfg.diag_futility,
            disable_null_move: cfg.diag_null,
            disable_qsearch_see: cfg.diag_qsee,
        });
    }

    // S4.3A: sampled wall-time attribution (bench-only, never UCI).
    if let Some(rate) = cfg.timing_sample {
        ctx.sampled_timing = true;
        ctx.sample_rate = rate;
        ctx.full_legal_sub.sample_rate = rate;
    }

    let limits = limits_for(actual_limit);

    let start = Instant::now();
    let outcome = search_one(&mut pos, &hist, &limits, &ctx, &mut tt, cfg.profile)
        .ok_or_else(|| format!("fixture {}: no legal moves (terminal root)", fx.id))?;
    let elapsed = start.elapsed();
    let nodes = ctx.nodes.load(Ordering::Relaxed);
    let stats = ctx.stats();

    if let Some(_rate) = cfg.timing_sample {
        let t = ctx.sampled_timings();
        let legality_make = ctx.legality_probe_make.load(Ordering::Relaxed);
        let legality_unmake = ctx.legality_probe_unmake.load(Ordering::Relaxed);
        let search_edge_make = stats.make_moves.saturating_sub(legality_make);
        let search_edge_unmake = stats.unmake_moves.saturating_sub(legality_unmake);
        let fl_sub = ctx.full_legal_sub.snapshot();
        let s5 = (
            ctx.probe_child_generations.load(Ordering::Relaxed),
            ctx.main_edge_probe_generations.load(Ordering::Relaxed),
            ctx.qsearch_edge_probe_generations.load(Ordering::Relaxed),
            ctx.root_edge_probe_generations.load(Ordering::Relaxed),
            ctx.negamax_body_generations.load(Ordering::Relaxed),
            ctx.root_generations.load(Ordering::Relaxed),
            ctx.final_evasion_generations.load(Ordering::Relaxed),
        );
        println!(
            "bench_timing nodes={} elapsed_us={} legality_probe_make={} legality_probe_unmake={} search_edge_make={} search_edge_unmake={} full_legal_probe_make={} full_legal_probe_unmake={} tactical_probe_make={} tactical_probe_unmake={} evasion_probe_make={} evasion_probe_unmake={} has_any_probe_make={} has_any_probe_unmake={} movegen_legal_calls={} movegen_legal_samples={} movegen_legal_ns={} movegen_tactical_calls={} movegen_tactical_samples={} movegen_tactical_ns={} movegen_evasion_calls={} movegen_evasion_samples={} movegen_evasion_ns={} movegen_has_any_calls={} movegen_has_any_samples={} movegen_has_any_ns={} eval_calls={} eval_samples={} eval_ns={} ordering_calls={} ordering_samples={} ordering_ns={} see_calls={} see_samples={} see_ns={} tt_calls={} tt_samples={} tt_ns={} fl_sub_calls={} fl_sub_samples={} fl_pseudo_gen_ns={} fl_check_state_ns={} fl_pin_scan_ns={} fl_pin_scan_calls={} fl_in_check_calls={} probe_child_generations={} main_edge_probe_generations={} qsearch_edge_probe_generations={} root_edge_probe_generations={} negamax_body_generations={} root_generations={} final_evasion_generations={}",
            nodes,
            elapsed.as_micros(),
            legality_make,
            legality_unmake,
            search_edge_make,
            search_edge_unmake,
            ctx.full_legal_probe_make.load(Ordering::Relaxed),
            ctx.full_legal_probe_unmake.load(Ordering::Relaxed),
            ctx.tactical_probe_make.load(Ordering::Relaxed),
            ctx.tactical_probe_unmake.load(Ordering::Relaxed),
            ctx.evasion_probe_make.load(Ordering::Relaxed),
            ctx.evasion_probe_unmake.load(Ordering::Relaxed),
            ctx.has_any_probe_make.load(Ordering::Relaxed),
            ctx.has_any_probe_unmake.load(Ordering::Relaxed),
            t.movegen_legal.0,
            t.movegen_legal.1,
            t.movegen_legal.2,
            t.movegen_tactical.0,
            t.movegen_tactical.1,
            t.movegen_tactical.2,
            t.movegen_evasion.0,
            t.movegen_evasion.1,
            t.movegen_evasion.2,
            t.movegen_has_any.0,
            t.movegen_has_any.1,
            t.movegen_has_any.2,
            t.eval.0,
            t.eval.1,
            t.eval.2,
            t.ordering.0,
            t.ordering.1,
            t.ordering.2,
            t.see.0,
            t.see.1,
            t.see.2,
            t.tt.0,
            t.tt.1,
            t.tt.2,
            fl_sub.calls,
            fl_sub.samples,
            fl_sub.pseudo_gen_ns,
            fl_sub.check_state_ns,
            fl_sub.pin_scan_ns,
            fl_sub.pin_scan_calls,
            fl_sub.in_check_calls,
            s5.0,
            s5.1,
            s5.2,
            s5.3,
            s5.4,
            s5.5,
            s5.6,
        );
    }

    validate(
        fx,
        mode,
        cfg.profile,
        &snap,
        &pos,
        &hist,
        &outcome,
        nodes,
        actual_limit,
    )?;

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
        LimitKind::Movetime(ms) => format!("movetime_ms:{}", ms),
    };

    Ok(BenchResult {
        suite: cfg.suite.as_str(),
        fixture: fx.id,
        mode: mode.as_str(),
        profile: profile_str(cfg.profile),
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
        target_root_rank: ctx.target_root_rank.load(Ordering::Relaxed),
        stats,
    })
}

/// S4.3A movegen sub-attribution microbench: isolate pseudo generation,
/// make+unmake pair, is_square_attacked, and per-move legality filtering on
/// ONE representative FEN. Supporting evidence only (cache/context differs
/// from real search).
fn run_microbench(args: &[String]) -> Result<(), String> {
    let mut fen: Option<String> = None;
    let mut repeats: u64 = 200_000;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "microbench: --fen requires a value".to_string())?
                    .clone();
                if fen.is_some() {
                    return Err("microbench: --fen may be specified only once".to_string());
                }
                fen = Some(value);
            }
            "--repeats" => {
                let value = it
                    .next()
                    .ok_or_else(|| "microbench: --repeats requires a value".to_string())?
                    .clone();
                let n: u64 = value.parse().map_err(|_| {
                    format!(
                        "microbench: --repeats '{}' is not a positive integer",
                        value
                    )
                })?;
                if n == 0 {
                    return Err("microbench: --repeats must be >= 1".to_string());
                }
                repeats = n;
            }
            other => {
                return Err(format!(
                    "microbench: unknown argument '{}' (expected --fen <fen> [--repeats N])",
                    other
                ));
            }
        }
    }
    let fen = fen.ok_or_else(|| "microbench: --fen is required".to_string())?;
    let mut pos = parse_fen(&fen).map_err(|e| format!("microbench: invalid FEN: {}", e))?;
    let enemy = pos.side.opposite();
    let king_sq = pos.king_square(pos.side);
    let mut moves: Vec<Move> = Vec::with_capacity(64);

    // warmup
    for _ in 0..10_000 {
        moves.clear();
        generate_pseudo_moves(&pos, &mut moves);
    }
    let first = moves[0];
    std::hint::black_box(first);

    // a) pseudo move generation only
    let t0 = Instant::now();
    for _ in 0..repeats {
        moves.clear();
        generate_pseudo_moves(&pos, &mut moves);
    }
    let pseudo_ns = t0.elapsed().as_nanos() as f64 / repeats as f64;

    // b) make+unmake pair alone
    let t0 = Instant::now();
    for _ in 0..repeats {
        let undo = pos.make_move(first);
        pos.unmake_move(undo);
    }
    let make_pair_ns = t0.elapsed().as_nanos() as f64 / repeats as f64;

    // c) is_square_attacked alone
    let t0 = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(pos.is_square_attacked(king_sq, enemy));
    }
    let attack_ns = t0.elapsed().as_nanos() as f64 / repeats as f64;

    // d) legality filtering over a fixed subset of the pseudo list:
    //    make -> king-attack test -> unmake, averaged per move
    let subset: Vec<Move> = moves.iter().take(8).copied().collect();
    let filter_reps = repeats / 8;
    let t0 = Instant::now();
    let mut ok = 0u64;
    for _ in 0..filter_reps {
        for &m in &subset {
            let undo = pos.make_move(m);
            if !pos.is_square_attacked(pos.king_square(pos.side), pos.side.opposite()) {
                ok += 1;
            }
            pos.unmake_move(undo);
        }
    }
    std::hint::black_box(ok);
    let filter_per_move_ns =
        t0.elapsed().as_nanos() as f64 / (filter_reps as f64 * subset.len() as f64);

    // S4.4A e) promoted fast path: eligibility tests + legal-Vec push per
    // move, no make/unmake. Mirrors the generator's non-check branch (the
    // pin bit test is a single AND, ~1ns, omitted from the model).
    let fast_reps = repeats / 8;
    let mut acc: Vec<Move> = Vec::with_capacity(64);
    let t0 = Instant::now();
    let mut accepted = 0u64;
    for _ in 0..fast_reps {
        for &m in &subset {
            let is_ep = m.flag == MoveFlag::EnPassant;
            let is_castle = matches!(m.flag, MoveFlag::KingCastle | MoveFlag::QueenCastle);
            let is_king = !is_castle && m.from == king_sq;
            let _is_pinned = false;
            if is_ep || is_castle || is_king {
                accepted += 1;
            } else {
                acc.push(m);
            }
        }
        acc.clear();
    }
    std::hint::black_box(accepted);
    std::hint::black_box(acc.len());
    let fast_accept_per_move_ns =
        t0.elapsed().as_nanos() as f64 / (fast_reps as f64 * subset.len() as f64);

    println!(
        "microbench fen=\"{}\" pseudo_moves={} pseudo_ns={:.1} make_pair_ns={:.1} attack_ns={:.1} filter_per_move_ns={:.1} fast_accept_per_move_ns={:.1}",
        fen,
        moves.len(),
        pseudo_ns,
        make_pair_ns,
        attack_ns,
        filter_per_move_ns,
        fast_accept_per_move_ns
    );
    Ok(())
}

/// S4.2A: `bench eval-breakdown --fen <fen> [--repeats N]` emits every dormant
/// positional evaluation component in WHITE perspective (positive = favours
/// White), phase-interpolated, plus the base material/PST lane. With
/// `--repeats N` it also times N calls of the base `evaluate` vs the full
/// component breakdown (relative eval cost estimate). Diagnostic only.
fn run_eval_breakdown(args: &[String]) -> Result<(), String> {
    let mut fen: Option<String> = None;
    let mut repeats: Option<u32> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-breakdown: --fen requires a value".to_string())?
                    .clone();
                if fen.is_some() {
                    return Err("eval-breakdown: --fen may be specified only once".to_string());
                }
                fen = Some(value);
            }
            "--repeats" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-breakdown: --repeats requires a value".to_string())?
                    .clone();
                let n: u32 = value.parse().map_err(|_| {
                    format!(
                        "eval-breakdown: --repeats '{}' is not a positive integer",
                        value
                    )
                })?;
                if n == 0 {
                    return Err("eval-breakdown: --repeats must be >= 1".to_string());
                }
                if repeats.is_some() {
                    return Err("eval-breakdown: --repeats may be specified only once".to_string());
                }
                repeats = Some(n);
            }
            other => {
                return Err(format!(
                    "eval-breakdown: unknown argument '{}' (expected --fen <fen> [--repeats N])",
                    other
                ));
            }
        }
    }
    let fen = fen.ok_or_else(|| "eval-breakdown: --fen is required".to_string())?;
    let pos = parse_fen(&fen).map_err(|e| format!("eval-breakdown: invalid FEN: {}", e))?;
    let cps = evaluate_components_white(&pos);
    let side = if pos.side == crate::chess::types::Color::White {
        "w"
    } else {
        "b"
    };
    println!(
        "eval_breakdown fen=\"{}\" side={} phase={} material_pst={} pawn_structure={} mobility={} piece_activity={} rook_activity={} development_space={} king_safety={}",
        fen,
        side,
        cps.phase,
        cps.material_pst,
        cps.pawn_structure,
        cps.mobility,
        cps.piece_activity,
        cps.rook_activity,
        cps.development_space,
        cps.king_safety,
    );
    if let Some(n) = repeats {
        let base_start = std::time::Instant::now();
        let mut acc = 0i64;
        for _ in 0..n {
            acc = acc.wrapping_add(i64::from(crate::engine::eval::evaluate(&pos)));
        }
        let base_ms = base_start.elapsed().as_millis();
        let comp_start = std::time::Instant::now();
        let mut cacc = 0i64;
        for _ in 0..n {
            let c = evaluate_components_white(&pos);
            cacc = cacc.wrapping_add(
                i64::from(c.material_pst) + i64::from(c.mobility) + i64::from(c.pawn_structure),
            );
        }
        let comp_ms = comp_start.elapsed().as_millis();
        let ratio = if base_ms > 0 {
            comp_ms as f64 / base_ms as f64
        } else {
            0.0
        };
        println!(
            "eval_timing repeats={} base_eval_ms={} components_ms={} ratio={:.3} acc={} cacc={}",
            n, base_ms, comp_ms, ratio, acc, cacc
        );
    }
    Ok(())
}

/// Check deterministic fields for consistency across repeats of the same
/// (fixture, mode). On mismatch, emit a clear bench_error (warning).
fn check_determinism(results: &[BenchResult]) {
    use std::collections::HashMap;
    let mut groups: HashMap<(&str, &str, &str), Vec<&BenchResult>> = HashMap::new();
    for r in results {
        groups
            .entry((r.profile, r.fixture, r.mode))
            .or_default()
            .push(r);
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
                "bench_error determinism: profile={} fixture={} mode={} differs across repeats \
                 (nodes={:?} score={:?} bestmove={:?} depth={:?})",
                key.0, key.1, key.2, nodes_set, score_set, bm_set, cd_set
            );
        }
    }
}

fn fixtures_for(cfg: &BenchArgs) -> Vec<Fixture> {
    match cfg.suite {
        Suite::Smoke => smoke_fixtures(),
        Suite::Standard => standard_fixtures(),
        Suite::Throughput => throughput_fixtures()
            .into_iter()
            .filter(|fx| cfg.fixture.is_none_or(|id| fx.id == id))
            .collect(),
        Suite::Profile => {
            if let Some(fen) = cfg.custom_fen {
                vec![Fixture {
                    id: "custom",
                    fen,
                    limit: LimitKind::Depth(1),
                    history: None,
                    locked: None,
                }]
            } else {
                standard_fixtures()
                    .into_iter()
                    .filter(|fx| cfg.fixture.is_none_or(|id| fx.id == id))
                    .collect()
            }
        }
        Suite::Ablation => standard_fixtures()
            .into_iter()
            .filter(|fx| cfg.fixture.is_none_or(|id| fx.id == id))
            .collect(),
    }
}

fn ablation_profiles() -> [SearchProfile; 5] {
    [
        SearchProfile::Current,
        SearchProfile::CurrentAspiration,
        SearchProfile::CurrentAspirationLmr,
        SearchProfile::CurrentAspirationLmrFutility,
        SearchProfile::CurrentAspirationLmrFutilitySee,
    ]
}

fn print_summary(
    suite: Suite,
    modes: &[BenchMode],
    fixtures: &[Fixture],
    results: &[BenchResult],
    profile: &'static str,
) {
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
        "bench_summary suite={} profile={} mode={} fixture_count={} measured_run_count={} aggregate_nodes={} median_elapsed_us={} median_nps={}",
        suite.as_str(),
        profile,
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
    if args[0] == "eval-breakdown" {
        return run_eval_breakdown(&args[1..]);
    }
    if args[0] == "eval-features" {
        return run_eval_features(&args[1..]);
    }
    if args[0] == "eval-features-batch" {
        return run_eval_features_batch(&args[1..]);
    }
    if args[0] == "eval-features-schema" {
        return run_eval_features_schema(&args[1..]);
    }
    if args[0] == "microbench" {
        return run_microbench(&args[1..]);
    }
    let cfg = parse_args(args)?;
    let fixtures = fixtures_for(&cfg);
    let modes: Vec<BenchMode> = match cfg.mode {
        BenchMode::All => vec![BenchMode::Disabled, BenchMode::Cold, BenchMode::Warm],
        m => vec![m],
    };

    let profiles: Vec<SearchProfile> = if cfg.suite == Suite::Ablation {
        ablation_profiles().to_vec()
    } else {
        vec![cfg.profile]
    };

    let mut results: Vec<BenchResult> = Vec::new();
    for profile in profiles {
        let mut profile_cfg = cfg.clone();
        profile_cfg.profile = profile;
        for fx in &fixtures {
            for &mode in &modes {
                for r in 0..profile_cfg.repeat {
                    let res = run_one(&profile_cfg, fx, mode, r + 1)?;
                    println!("{}", format_result_line(&res));
                    results.push(res);
                }
            }
        }
    }
    let summary_profile = if cfg.suite == Suite::Ablation {
        "all"
    } else {
        profile_str(cfg.profile)
    };
    print_summary(cfg.suite, &modes, &fixtures, &results, summary_profile);
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
    println!(
        "  profile     search-cost counters across standard fixtures (fixed nodes by default)"
    );
    println!("  ablation    all cumulative candidate profiles on standard fixtures");
    println!("  help        this message");
    println!();
    println!("OPTIONS:");
    println!("  --mode <disabled|cold|warm|all>  default: smoke=disabled, standard=all, throughput=disabled");
    println!("  --repeat <N>                       default: smoke=1, standard=1, throughput=3");
    println!("  --nodes <N>                       throughput/profile node budget (default 100000); ablation limit");
    println!("  --depth <N>                       profile or ablation fixed-depth limit");
    println!("  --movetime <MS>                   profile or ablation fixed-time limit");
    println!("  --fixture <fixture-id>             throughput/profile/ablation filter");
    println!("  --fen <FEN>                        profile one-off FEN (mutually exclusive with --fixture)");
    println!(
        "  --profile <reference|m4.1|pvs|see|aspiration|lmr|null|futility|current|current-lmr|current-threat-aware|current-threat-aware-no-qchecks|current-threat-aware-eval-order|current-threat-aware-eval-only|current-threat-aware-order-only|current-eval2|current-qsearch-movegen|current-qsearch-pruning|current-qsearch-fast-pruning|current-aspiration|current-aspiration-lmr|current-aspiration-lmr-futility|current-aspiration-lmr-futility-see|current-final>  search profile (default reference == M4.0 baseline)"
    );
    println!();
    println!("OUTPUT PREFIXES: bench_result / bench_summary / bench_error");
}

// ---------------------------------------------------------------------------
// Tests — fast only; no full suites, no wall-clock assertions.
// ---------------------------------------------------------------------------

/// S6.1A: `bench eval-features --fen <fen>` - deterministic sparse
/// FeatureSetV1 export (JSON line). Observation-only.
fn run_eval_features(args: &[String]) -> Result<(), String> {
    let mut fen: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-features: --fen requires a value".to_string())?
                    .clone();
                fen = Some(value);
            }
            other => {
                return Err(format!(
                    "eval-features: unknown argument '{}' (expected --fen <fen>)",
                    other
                ));
            }
        }
    }
    let fen = fen.ok_or_else(|| "eval-features: --fen is required".to_string())?;
    let mut pos = parse_fen(&fen).map_err(|e| format!("eval-features: {e}"))?;
    println!("{}", features_line(&mut pos, &fen));
    Ok(())
}

/// S6.1A: `bench eval-features-batch --epd <file>` - one JSON line per
/// position (deterministic order). Observation-only.
fn run_eval_features_batch(args: &[String]) -> Result<(), String> {
    let mut epd: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--epd" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-features-batch: --epd requires a value".to_string())?
                    .clone();
                epd = Some(value);
            }
            other => {
                return Err(format!(
                    "eval-features-batch: unknown argument '{}' (expected --epd <file>)",
                    other
                ));
            }
        }
    }
    let epd = epd.ok_or_else(|| "eval-features-batch: --epd is required".to_string())?;
    let text = std::fs::read_to_string(&epd)
        .map_err(|e| format!("eval-features-batch: cannot read {epd}: {e}"))?;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fen = if line.contains('|') {
            line.split('|').nth(2).unwrap_or(line).trim()
        } else {
            line.split(';').next().unwrap_or(line).trim()
        };
        let mut pos = parse_fen(fen).map_err(|e| format!("eval-features-batch: {e}"))?;
        println!("{}", features_line(&mut pos, fen));
    }
    Ok(())
}

/// S6.1A: `bench eval-features-schema` - the CANONICAL schema content
/// (stable ids/names). The real SHA-256 of this exact string is computed by
/// the freeze tooling (tools/s6/freeze_schema.py) and compiled in as
/// `FEATURE_SCHEMA_SHA256`; the engine never computes a hash at runtime.
fn run_eval_features_schema(_args: &[String]) -> Result<(), String> {
    use crate::engine::features::{feature_name, FEATURE_COUNT};
    let mut features = String::new();
    for id in 0..FEATURE_COUNT {
        if id > 0 {
            features.push(',');
        }
        features.push_str(&format!(
            "{{\"id\":{},\"name\":\"{}\"}}",
            id,
            feature_name(id)
        ));
    }
    let mut schema = String::from("{\"feature_schema\":\"s6-feature-v1\",");
    schema.push_str("\"schema_version\":1,\"feature_count\":");
    schema.push_str(&FEATURE_COUNT.to_string());
    schema.push_str(",\"features\":[");
    schema.push_str(&features);
    schema.push_str("]}");
    println!("{schema}");
    Ok(())
}

/// Deterministic JSON line for one position's FeatureSetV1.
fn features_line(pos: &mut crate::chess::position::Position, fen: &str) -> String {
    use crate::engine::features::{extract_features_v1, feature_name};
    let f = extract_features_v1(pos);
    let sparse = f.sparse();
    let mut out = String::new();
    out.push_str(&format!(
        "{{\"fen\":\"{}\",\"phase\":{},\"features\":[",
        fen.replace('"', "\\\""),
        f.phase
    ));
    for (i, fv) in sparse.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"id\":{},\"name\":\"{}\",\"value\":{}}}",
            fv.id,
            feature_name(fv.id),
            fv.value
        ));
    }
    out.push_str("]}");
    out
}

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
        // --profile defaults to reference (M4.0 baseline).
        assert_eq!(c.profile, SearchProfile::M4Reference);

        let d = parse_args(&["ablation".to_string()]).unwrap();
        assert_eq!(d.suite, Suite::Ablation);
        assert_eq!(d.mode, BenchMode::Disabled);
        assert_eq!(d.repeat, 1);
        assert_eq!(d.ablation_limit, Some(LimitKind::Nodes(100_000)));
    }

    #[test]
    fn ablation_accepts_one_fixed_limit_kind() {
        let depth = parse_args(&[
            "ablation".to_string(),
            "--depth".to_string(),
            "6".to_string(),
            "--fixture".to_string(),
            "startpos".to_string(),
        ])
        .unwrap();
        assert_eq!(depth.ablation_limit, Some(LimitKind::Depth(6)));
        assert_eq!(depth.fixture, Some("startpos"));

        let profile = parse_args(&[
            "profile".to_string(),
            "--depth".to_string(),
            "6".to_string(),
            "--fixture".to_string(),
            "startpos".to_string(),
        ])
        .unwrap();
        assert_eq!(profile.profile_limit, Some(LimitKind::Depth(6)));
        assert_eq!(profile.fixture, Some("startpos"));

        let profile_movetime = parse_args(&[
            "profile".to_string(),
            "--mode".to_string(),
            "cold".to_string(),
            "--movetime".to_string(),
            "1000".to_string(),
            "--fixture".to_string(),
            "startpos".to_string(),
        ])
        .unwrap();
        assert_eq!(profile_movetime.mode, BenchMode::Cold);
        assert_eq!(
            profile_movetime.profile_limit,
            Some(LimitKind::Movetime(1000))
        );

        let custom = parse_args(&[
            "profile".to_string(),
            "--fen".to_string(),
            "7k/8/8/8/8/8/4Q3/6K1 w - - 0 1".to_string(),
            "--depth".to_string(),
            "2".to_string(),
        ])
        .unwrap();
        assert!(custom.custom_fen.is_some());
        assert!(custom.fixture.is_none());
        assert_eq!(fixtures_for(&custom).len(), 1);
        assert_eq!(fixtures_for(&custom)[0].id, "custom");

        let movetime = parse_args(&[
            "ablation".to_string(),
            "--movetime".to_string(),
            "1000".to_string(),
        ])
        .unwrap();
        assert_eq!(movetime.ablation_limit, Some(LimitKind::Movetime(1000)));

        assert!(parse_args(&[
            "standard".to_string(),
            "--depth".to_string(),
            "6".to_string(),
        ])
        .is_err());
        assert!(parse_args(&[
            "ablation".to_string(),
            "--profile".to_string(),
            "current".to_string(),
        ])
        .is_err());
        assert!(parse_args(&[
            "ablation".to_string(),
            "--nodes".to_string(),
            "1000".to_string(),
            "--depth".to_string(),
            "6".to_string(),
        ])
        .is_err());
        assert!(parse_args(&[
            "profile".to_string(),
            "--nodes".to_string(),
            "1000".to_string(),
            "--depth".to_string(),
            "6".to_string(),
        ])
        .is_err());
        assert!(parse_args(&[
            "profile".to_string(),
            "--fixture".to_string(),
            "startpos".to_string(),
            "--fen".to_string(),
            "7k/8/8/8/8/8/4Q3/6K1 w - - 0 1".to_string(),
        ])
        .is_err());
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
        // unspecified --profile stays at the reference default.
        assert_eq!(a.profile, SearchProfile::M4Reference);
    }

    #[test]
    fn parse_valid_throughput_fixture_filters() {
        for id in ["startpos", "open-tactical", "queen-win"] {
            let args = vec![
                "throughput".to_string(),
                "--fixture".to_string(),
                id.to_string(),
            ];
            let cfg = parse_args(&args).unwrap();
            assert_eq!(cfg.suite, Suite::Throughput);
            assert_eq!(cfg.fixture, Some(id));
            let fixtures = fixtures_for(&cfg);
            assert_eq!(fixtures.len(), 1);
            assert_eq!(fixtures[0].id, id);
        }

        let cfg = parse_args(&["throughput".to_string()]).unwrap();
        assert_eq!(cfg.fixture, None);
        let fixtures = fixtures_for(&cfg);
        assert_eq!(fixtures.len(), 3);
        assert_eq!(fixtures[0].id, "startpos");
        assert_eq!(fixtures[1].id, "open-tactical");
        assert_eq!(fixtures[2].id, "queen-win");
    }

    #[test]
    fn parse_valid_profile() {
        let r = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "current".to_string(),
        ])
        .unwrap();
        assert_eq!(r.profile, SearchProfile::Current);

        let f = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "reference".to_string(),
        ])
        .unwrap();
        assert_eq!(f.profile, SearchProfile::M4Reference);

        // Commit 5 exposes `m4.1` on the CLI; it maps to the M4.1 full-window
        // reference profile (quiet ordering, no PVS).
        let m = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "m4.1".to_string(),
        ])
        .unwrap();
        assert_eq!(m.profile, SearchProfile::M41Reference);

        let a = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "aspiration".to_string(),
        ])
        .unwrap();
        assert_eq!(a.profile, SearchProfile::AspirationCandidate);

        let q = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "current-qsearch-movegen".to_string(),
        ])
        .unwrap();
        assert_eq!(q.profile, SearchProfile::CurrentQsearchMovegen);

        let p = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "current-qsearch-pruning".to_string(),
        ])
        .unwrap();
        assert_eq!(p.profile, SearchProfile::CurrentQsearchPruning);

        let p = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "current-qsearch-fast-pruning".to_string(),
        ])
        .unwrap();
        assert_eq!(p.profile, SearchProfile::CurrentQsearchFastPruning);

        let e2 = parse_args(&[
            "standard".to_string(),
            "--profile".to_string(),
            "current-eval2".to_string(),
        ])
        .unwrap();
        assert_eq!(e2.profile, SearchProfile::CurrentEval2);

        for (name, expected) in [
            (
                "current-threat-aware-eval-only",
                SearchProfile::CurrentThreatAwareEvalOnly,
            ),
            (
                "current-threat-aware-order-only",
                SearchProfile::CurrentThreatAwareOrderOnly,
            ),
        ] {
            let parsed = parse_args(&[
                "profile".to_string(),
                "--profile".to_string(),
                name.to_string(),
            ])
            .unwrap();
            assert_eq!(parsed.profile, expected);
        }

        for (name, expected) in [
            ("current-aspiration", SearchProfile::CurrentAspiration),
            (
                "current-aspiration-lmr",
                SearchProfile::CurrentAspirationLmr,
            ),
            (
                "current-aspiration-lmr-futility",
                SearchProfile::CurrentAspirationLmrFutility,
            ),
            (
                "current-aspiration-lmr-futility-see",
                SearchProfile::CurrentAspirationLmrFutilitySee,
            ),
            ("current-final", SearchProfile::CurrentFinal),
        ] {
            let parsed = parse_args(&[
                "profile".to_string(),
                "--profile".to_string(),
                name.to_string(),
            ])
            .unwrap();
            assert_eq!(parsed.profile, expected);
        }
    }

    #[test]
    fn profile_str_maps_all_variants() {
        // Compile-time exhaustiveness: every `SearchProfile` variant maps to a
        // stable CLI string. Commit 5 exposes `m4.1` on the CLI (see
        // `parse_valid_profile`); it round-trips through `profile_str`.
        assert_eq!(profile_str(SearchProfile::M4Reference), "reference");
        assert_eq!(profile_str(SearchProfile::M41Reference), "m4.1");
        assert_eq!(profile_str(SearchProfile::PvsReference), "pvs");
        assert_eq!(profile_str(SearchProfile::SeeCandidate), "see");
        assert_eq!(
            profile_str(SearchProfile::AspirationCandidate),
            "aspiration"
        );
        assert_eq!(profile_str(SearchProfile::LmrCandidate), "lmr");
        assert_eq!(profile_str(SearchProfile::NullMoveCandidate), "null");
        assert_eq!(profile_str(SearchProfile::FutilityCandidate), "futility");
        assert_eq!(profile_str(SearchProfile::Current), "current");
        assert_eq!(profile_str(SearchProfile::CurrentEval2), "current-eval2");
        assert_eq!(
            profile_str(SearchProfile::CurrentThreatAware),
            "current-threat-aware"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentThreatAwareNoQchecks),
            "current-threat-aware-no-qchecks"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentThreatAwareEvalOrder),
            "current-threat-aware-eval-order"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentThreatAwareEvalOnly),
            "current-threat-aware-eval-only"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentThreatAwareOrderOnly),
            "current-threat-aware-order-only"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentQsearchMovegen),
            "current-qsearch-movegen"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentQsearchPruning),
            "current-qsearch-pruning"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentQsearchFastPruning),
            "current-qsearch-fast-pruning"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentAspiration),
            "current-aspiration"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentAspirationLmr),
            "current-aspiration-lmr"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentAspirationLmrFutility),
            "current-aspiration-lmr-futility"
        );
        assert_eq!(
            profile_str(SearchProfile::CurrentAspirationLmrFutilitySee),
            "current-aspiration-lmr-futility-see"
        );
        assert_eq!(profile_str(SearchProfile::CurrentFinal), "current-final");
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
            vec![
                "smoke".to_string(),
                "--profile".to_string(),
                "foo".to_string(),
            ],
            vec![
                "throughput".to_string(),
                "--fixture".to_string(),
                "unknown".to_string(),
            ],
            vec!["throughput".to_string(), "--fixture".to_string()],
            vec![
                "throughput".to_string(),
                "--fixture".to_string(),
                "startpos".to_string(),
                "--fixture".to_string(),
                "queen-win".to_string(),
            ],
            vec![
                "smoke".to_string(),
                "--fixture".to_string(),
                "startpos".to_string(),
            ],
            vec![
                "standard".to_string(),
                "--fixture".to_string(),
                "startpos".to_string(),
            ],
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
            profile: "reference",
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
            target_root_rank: 0,
            stats: SearchStats::default(),
        };
        let line = format_result_line(&r);
        assert!(line.starts_with("bench_result "));
        // Fixed key order.
        let i = |k: &str| line.find(k).unwrap_or(usize::MAX);
        assert!(i("suite=") < i("fixture="));
        assert!(i("fixture=") < i("mode="));
        assert!(i("mode=") < i("profile="));
        assert!(i("profile=") < i("repeat="));
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
        assert!(line.ends_with("target_root_rank=0"));
    }

    #[test]
    fn ablation_result_includes_elapsed_ms_and_search_counters() {
        let r = BenchResult {
            suite: "ablation",
            fixture: "startpos",
            mode: "disabled",
            profile: "current-aspiration-lmr-futility-see",
            repeat: 1,
            limit: "nodes:1000".to_string(),
            score: Some(0),
            best_move: "b1c3".to_string(),
            completed_depth: 3,
            stopped: true,
            nodes: 1000,
            elapsed_us: 12_345,
            nps: 81_004,
            pv: "b1c3 b8c6".to_string(),
            target_root_rank: 0,
            stats: SearchStats {
                qsearch_nodes: 12,
                aspiration_retries: 3,
                lmr_reductions: 4,
                lmr_researches: 1,
                futility_pruned: 5,
                see_calls: 6,
                qsearch_see_tests: 7,
                qsearch_see_pruned: 2,
                qsearch_see_fail_open_promotions: 8,
                qsearch_checking_captures_kept: 3,
                qsearch_promotions_kept: 4,
                qsearch_en_passant_kept: 5,
                ..SearchStats::default()
            },
        };
        let line = format_result_line(&r);
        assert!(line.contains("elapsed_ms=12"));
        assert!(line.contains("qsearch_nodes=12"));
        assert!(line.contains("aspiration_retries=3"));
        assert!(line.contains("lmr_reductions=4"));
        assert!(line.contains("futility_pruned=5"));
        assert!(line.contains("see_calls=6"));
        assert!(line.contains("qsearch_see_tests=7"));
        assert!(line.contains("qsearch_see_pruned=2"));
        assert!(line.contains("qsearch_see_fail_open_promotions=8"));
        assert!(line.contains("qsearch_checking_captures_kept=3"));
        assert!(line.contains("qsearch_promotions_kept=4"));
        assert!(line.contains("qsearch_en_passant_kept=5"));
        assert!(line.contains("check_extensions=0"));
        assert!(line.contains("single_evasion_extensions=0"));
        assert!(line.contains("qsearch_check_moves=0"));
        assert!(line.contains("threat_ordered_moves=0"));
        assert!(line.contains("root_reorders=0"));
        assert!(line.contains("total_nodes=1000"));
        assert!(line.contains("completed_iterations=0"));
        assert!(line.contains("qsearch_ratio=0.012000"));
        assert!(line.contains("effective_branching_factor=10.000000"));
        assert!(line.contains("tt_rejected_depth=0"));
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
            profile: SearchProfile::M4Reference,
            fixture: None,
            custom_fen: None,
            profile_limit: None,
            ablation_limit: None,
            diag_lmr: false,
            diag_futility: false,
            diag_null: false,
            diag_qsee: false,
            diag_root_full_window: false,
            timing_sample: None,
            forced_root: None,
            target_root: None,
        };
        let r = run_one(&cfg, &fx, BenchMode::Disabled, 1).unwrap();
        assert_eq!(r.completed_depth, 1);
        assert!(!r.stopped);
        assert!(r.score.is_some());
        // run_one validates bestmove/PV legality and full restoration internally.
        assert!(!r.best_move.is_empty());
    }

    fn profile_cfg(
        limit: LimitKind,
        diag_lmr: bool,
        forced_root: Option<&str>,
        target_root: Option<&str>,
    ) -> BenchArgs {
        BenchArgs {
            suite: Suite::Profile,
            mode: BenchMode::Disabled,
            repeat: 1,
            nodes: 100_000,
            profile: SearchProfile::CurrentFinal,
            fixture: None,
            custom_fen: None,
            profile_limit: Some(limit),
            ablation_limit: None,
            diag_lmr,
            diag_futility: false,
            diag_null: false,
            diag_qsee: false,
            diag_root_full_window: false,
            timing_sample: None,
            forced_root: forced_root.map(|s| s.to_string()),
            target_root: target_root.map(|s| s.to_string()),
        }
    }

    #[test]
    fn diag_target_root_records_rank_before_forced_filter() {
        let fx = Fixture {
            id: "startpos",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: None,
            locked: None,
        };
        let cfg = profile_cfg(LimitKind::Depth(1), false, Some("b1c3"), Some("b1c3"));
        let r = run_one(&cfg, &fx, BenchMode::Disabled, 1).unwrap();
        assert_eq!(r.best_move, "b1c3", "forced root must pick the forced move");
        assert_eq!(
            r.target_root_rank, 1,
            "b1c3 is rank 1 under the static root ordering at startpos"
        );
    }

    #[test]
    fn diag_disable_lmr_zeroes_lmr_reductions() {
        let fx = Fixture {
            id: "startpos",
            fen: START_FEN,
            limit: LimitKind::Depth(5),
            history: None,
            locked: None,
        };
        let no_lmr = profile_cfg(LimitKind::Depth(5), true, None, None);
        let r_no = run_one(&no_lmr, &fx, BenchMode::Disabled, 1).unwrap();
        assert_eq!(
            r_no.stats.lmr_reductions, 0,
            "no-lmr disables late-move reductions"
        );
        assert_eq!(
            r_no.stats.lmr_researches, 0,
            "no-lmr disables LMR re-searches"
        );

        let base = profile_cfg(LimitKind::Depth(5), false, None, None);
        let r_base = run_one(&base, &fx, BenchMode::Disabled, 1).unwrap();
        assert!(
            r_base.stats.lmr_reductions > 0,
            "CurrentFinal still applies LMR when no diagnostic is set"
        );
    }

    #[test]
    fn cold_depth1_completes() {
        // P1 hardening: a fixed-depth COLD search must still reach the
        // requested depth, not stop early, and return a real score. `run_one`
        // now enforces fixed-depth completeness for every mode (not just
        // disabled) via `validate` -> `check_fixed_depth_complete`.
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: None,
            locked: None,
        };
        let cfg = BenchArgs {
            suite: Suite::Standard,
            mode: BenchMode::Cold,
            repeat: 1,
            nodes: 100_000,
            profile: SearchProfile::M4Reference,
            fixture: None,
            custom_fen: None,
            profile_limit: None,
            ablation_limit: None,
            diag_lmr: false,
            diag_futility: false,
            diag_null: false,
            diag_qsee: false,
            diag_root_full_window: false,
            timing_sample: None,
            forced_root: None,
            target_root: None,
        };
        let r = run_one(&cfg, &fx, BenchMode::Cold, 1).unwrap();
        assert_eq!(r.completed_depth, 1);
        assert!(!r.stopped);
        assert!(r.score.is_some());
        // bestmove present; PV legal replay (validate already enforces this,
        // re-asserted here for clarity).
        assert!(!r.best_move.is_empty());
        let mut pv_pos = parse_fen(fx.fen).unwrap();
        for m_uci in r.pv.split_whitespace() {
            let legal: Vec<String> = generate_legal_moves(&mut pv_pos)
                .iter()
                .map(|m| move_to_uci(*m))
                .collect();
            assert!(
                legal.contains(&m_uci.to_string()),
                "cold PV move {} illegal",
                m_uci
            );
            let m = *generate_legal_moves(&mut pv_pos)
                .iter()
                .find(|m| move_to_uci(**m) == m_uci)
                .expect("cold PV move must be legal");
            pv_pos.make_move(m);
        }
    }

    #[test]
    fn warm_depth1_warmup_and_measured_completes() {
        // P1 hardening: warm mode runs a warm-up (validated to complete the
        // requested depth) and then a measured run on the same TT. Both must
        // finish depth 1 with a real score; `run_one` bails if either fails,
        // so the measured run never executes on a half-warmed TT.
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: None,
            locked: None,
        };
        let cfg = BenchArgs {
            suite: Suite::Standard,
            mode: BenchMode::Warm,
            repeat: 1,
            nodes: 100_000,
            profile: SearchProfile::M4Reference,
            fixture: None,
            custom_fen: None,
            profile_limit: None,
            ablation_limit: None,
            diag_lmr: false,
            diag_futility: false,
            diag_null: false,
            diag_qsee: false,
            diag_root_full_window: false,
            timing_sample: None,
            forced_root: None,
            target_root: None,
        };
        let r = run_one(&cfg, &fx, BenchMode::Warm, 1).unwrap();
        assert_eq!(r.completed_depth, 1);
        assert!(!r.stopped);
        assert!(r.score.is_some());
        assert!(!r.best_move.is_empty());
        let mut pv_pos = parse_fen(fx.fen).unwrap();
        for m_uci in r.pv.split_whitespace() {
            let legal: Vec<String> = generate_legal_moves(&mut pv_pos)
                .iter()
                .map(|m| move_to_uci(*m))
                .collect();
            assert!(
                legal.contains(&m_uci.to_string()),
                "warm PV move {} illegal",
                m_uci
            );
            let m = *generate_legal_moves(&mut pv_pos)
                .iter()
                .find(|m| move_to_uci(**m) == m_uci)
                .expect("warm PV move must be legal");
            pv_pos.make_move(m);
        }
    }

    // ---- P1 hardening: smoke exact-lock is reference-only ----

    #[test]
    fn smoke_defaults_to_reference_profile() {
        let a = parse_args(&["smoke".to_string()]).unwrap();
        assert_eq!(a.suite, Suite::Smoke);
        assert_eq!(a.profile, SearchProfile::M4Reference);
    }

    #[test]
    fn smoke_reference_locks_exactly() {
        // `bench smoke` (and `bench smoke --profile reference`) must still
        // enforce the exact EVAL 1A 1149 / 969 locks.
        for fx in smoke_fixtures() {
            let cfg = BenchArgs {
                suite: Suite::Smoke,
                mode: BenchMode::Disabled,
                repeat: 1,
                nodes: 100_000,
                profile: SearchProfile::M4Reference,
                fixture: None,
                custom_fen: None,
                profile_limit: None,
                ablation_limit: None,
                diag_lmr: false,
                diag_futility: false,
                diag_null: false,
                diag_qsee: false,
                diag_root_full_window: false,
                timing_sample: None,
                forced_root: None,
                target_root: None,
            };
            let r = run_one(&cfg, &fx, BenchMode::Disabled, 1).unwrap();
            let locked = fx.locked.expect("smoke fixture must be locked");
            assert_eq!(r.nodes, locked.nodes, "{} reference nodes", fx.id);
            assert_eq!(
                r.best_move, locked.best_move,
                "{} reference bestmove",
                fx.id
            );
            assert_eq!(r.score, Some(locked.score), "{} reference score", fx.id);
            let pv_uci: Vec<String> = r.pv.split_whitespace().map(|s| s.to_string()).collect();
            let want: Vec<String> = locked.pv.iter().map(|s| s.to_string()).collect();
            assert_eq!(pv_uci, want, "{} reference PV", fx.id);
        }
    }

    #[test]
    fn smoke_current_runs_both_fixtures() {
        // `bench smoke --profile current` runs Current on both fixtures:
        // fixed-depth completes, score present, bestmove legal, position
        // restored, and the emitted profile is `current`. The exact
        // reference lock must NOT be applied.
        for fx in smoke_fixtures() {
            let cfg = BenchArgs {
                suite: Suite::Smoke,
                mode: BenchMode::Disabled,
                repeat: 1,
                nodes: 100_000,
                profile: SearchProfile::Current,
                fixture: None,
                custom_fen: None,
                profile_limit: None,
                ablation_limit: None,
                diag_lmr: false,
                diag_futility: false,
                diag_null: false,
                diag_qsee: false,
                diag_root_full_window: false,
                timing_sample: None,
                forced_root: None,
                target_root: None,
            };
            let r = run_one(&cfg, &fx, BenchMode::Disabled, 1)
                .unwrap_or_else(|e| panic!("current smoke {} must complete: {}", fx.id, e));
            assert_eq!(r.profile, "current");
            if let LimitKind::Depth(d) = fx.limit {
                assert_eq!(r.completed_depth, d, "{} current depth", fx.id);
            }
            assert!(!r.stopped, "{} current must not stop early", fx.id);
            assert!(r.score.is_some(), "{} current score", fx.id);
            assert!(!r.best_move.is_empty());
        }
    }

    #[test]
    fn current_profile_ignores_reference_lock() {
        // Direct contract test: with a deliberately-wrong locked block,
        // `validate` under `Current` must NOT error, while under
        // `M4Reference` it must. This proves the lock is scoped to the
        // reference profile regardless of Current's actual output.
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Depth(3),
            history: None,
            locked: Some(Locked {
                nodes: 1, // wrong on purpose
                score: 999,
                best_move: "e2e4",
                pv: &["e2e4"],
            }),
        };
        let mut pos = parse_fen(fx.fen).unwrap();
        let hist = effective_history(&fx, &pos);
        let snap = Snapshot {
            fen: to_fen(&pos),
            zobrist: pos.zobrist_key(),
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = limits_for(fx.limit);

        // Current: wrong lock must be ignored.
        let mut tt = TranspositionTable::disabled();
        let out = search_one(
            &mut pos,
            &hist,
            &limits,
            &ctx,
            &mut tt,
            SearchProfile::Current,
        )
        .unwrap();
        let nodes = ctx.nodes.load(Ordering::Relaxed);
        let cur = validate(
            &fx,
            BenchMode::Disabled,
            SearchProfile::Current,
            &snap,
            &pos,
            &hist,
            &out,
            nodes,
            fx.limit,
        );
        assert!(
            cur.is_ok(),
            "Current profile must not enforce reference lock: {:?}",
            cur.err()
        );

        // Reference: same wrong lock must be enforced (negative control).
        let mut pos2 = parse_fen(fx.fen).unwrap();
        let hist2 = effective_history(&fx, &pos2);
        let snap2 = Snapshot {
            fen: to_fen(&pos2),
            zobrist: pos2.zobrist_key(),
        };
        let ctx2 = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let mut tt2 = TranspositionTable::disabled();
        let out2 = search_one(
            &mut pos2,
            &hist2,
            &limits,
            &ctx2,
            &mut tt2,
            SearchProfile::M4Reference,
        )
        .unwrap();
        let nodes2 = ctx2.nodes.load(Ordering::Relaxed);
        let refr = validate(
            &fx,
            BenchMode::Disabled,
            SearchProfile::M4Reference,
            &snap2,
            &pos2,
            &hist2,
            &out2,
            nodes2,
            fx.limit,
        );
        assert!(
            refr.is_err(),
            "M4Reference profile must enforce the exact lock"
        );
    }

    #[test]
    fn node_budget_measured_hits_exact() {
        // Small node-budget measured run must stop exactly at the budget.
        // Not a 100k run; just enough to exercise the atomic budget semantic.
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Depth(1),
            history: None,
            locked: None,
        };
        let n: u64 = 50;
        let cfg = BenchArgs {
            suite: Suite::Throughput,
            mode: BenchMode::Disabled,
            repeat: 1,
            nodes: n,
            profile: SearchProfile::M4Reference,
            fixture: None,
            custom_fen: None,
            profile_limit: None,
            ablation_limit: None,
            diag_lmr: false,
            diag_futility: false,
            diag_null: false,
            diag_qsee: false,
            diag_root_full_window: false,
            timing_sample: None,
            forced_root: None,
            target_root: None,
        };
        let r = run_one(&cfg, &fx, BenchMode::Disabled, 1).unwrap();
        assert!(r.nodes <= n, "node budget must not be exceeded");
        assert_eq!(r.nodes, n, "node budget must be hit exactly");
        assert!(r.stopped, "budgeted run must report stopped");
    }

    #[test]
    fn node_budget_invalid_results_rejected() {
        // A node-budget violation must be a FATAL validation error, not a
        // warning: `validate` returns Err, so `run_one` would abort and
        // the CLI would exit non-zero instead of emitting untrustworthy
        // throughput data. We test `validate` directly against a real
        // position with two deliberately wrong (nodes, stopped) pairs.
        let fx = Fixture {
            id: "t",
            fen: START_FEN,
            limit: LimitKind::Nodes(50),
            history: None,
            locked: None,
        };
        let mut pos = parse_fen(fx.fen).unwrap();
        let hist = effective_history(&fx, &pos);
        let snap = Snapshot {
            fen: to_fen(&pos),
            zobrist: pos.zobrist_key(),
        };
        let ctx = SearchContext::new(Arc::new(AtomicBool::new(false)));
        let limits = limits_for(fx.limit);
        let mut tt = TranspositionTable::disabled();
        let out = search_one(
            &mut pos,
            &hist,
            &limits,
            &ctx,
            &mut tt,
            SearchProfile::M4Reference,
        )
        .unwrap();
        let nodes = ctx.nodes.load(Ordering::Relaxed);

        // Case A: budget not fully consumed (nodes < budget), stopped == true.
        let under = validate(
            &fx,
            BenchMode::Disabled,
            SearchProfile::M4Reference,
            &snap,
            &pos,
            &hist,
            &out,
            nodes.saturating_sub(1),
            fx.limit,
        );
        assert!(
            under.is_err(),
            "under-budget (nodes != budget) must be rejected: {:?}",
            under.err()
        );

        // Case B: budget consumed but stop flag not reported.
        let bad_stopped = SearchOutcome {
            best_move: out.best_move,
            score: out.score,
            completed_depth: out.completed_depth,
            stopped: false,
            pv: out.pv,
        };
        let no_stop = validate(
            &fx,
            BenchMode::Disabled,
            SearchProfile::M4Reference,
            &snap,
            &pos,
            &hist,
            &bad_stopped,
            nodes,
            fx.limit,
        );
        assert!(
            no_stop.is_err(),
            "missing stopped flag must be rejected: {:?}",
            no_stop.err()
        );
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
