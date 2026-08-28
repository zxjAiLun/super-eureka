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
use crate::engine::eval::{evaluate_components_white, Eval2Mask};
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
    /// S7.5B-0: post-A bounded check2 opportunity/probe-cost attribution.
    diag_s75b_probe: bool,
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
        SearchProfile::CurrentFinalLmrNullWindow => "current-final-lmr-null-window",
        SearchProfile::CurrentFinalSingleEvasion => "current-final-single-evasion",
        SearchProfile::CurrentFinalBoundedCheck2 => "current-final-bounded-check2",
        SearchProfile::CurrentFinalPhaseAffine => "current-final-phase-affine",
        SearchProfile::CurrentFinalEval2 => "current-final-eval2",
        SearchProfile::CurrentFinalNoPawnStructure => "current-final-no-pawn-structure",
        SearchProfile::CurrentFinalNoMobility => "current-final-no-mobility",
        SearchProfile::CurrentFinalNoPieceActivity => "current-final-no-piece-activity",
        SearchProfile::CurrentFinalNoRookActivity => "current-final-no-rook-activity",
        SearchProfile::CurrentFinalNoDevelopmentSpace => "current-final-no-development-space",
        SearchProfile::CurrentFinalNoKingSafety => "current-final-no-king-safety",
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
    let mut diag_s75b_probe = false;
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
                    "current-final-lmr-null-window" => SearchProfile::CurrentFinalLmrNullWindow,
                    "current-final-single-evasion" => SearchProfile::CurrentFinalSingleEvasion,
                    "current-final-bounded-check2" => SearchProfile::CurrentFinalBoundedCheck2,
                    "current-final-phase-affine" => SearchProfile::CurrentFinalPhaseAffine,
                    "current-final-eval2" => SearchProfile::CurrentFinalEval2,
                    "current-final-no-pawn-structure" => SearchProfile::CurrentFinalNoPawnStructure,
                    "current-final-no-mobility" => SearchProfile::CurrentFinalNoMobility,
                    "current-final-no-piece-activity" => SearchProfile::CurrentFinalNoPieceActivity,
                    "current-final-no-rook-activity" => SearchProfile::CurrentFinalNoRookActivity,
                    "current-final-no-development-space" => {
                        SearchProfile::CurrentFinalNoDevelopmentSpace
                    }
                    "current-final-no-king-safety" => SearchProfile::CurrentFinalNoKingSafety,
                    other => {
                        return Err(format!(
                            "bench: invalid --profile '{}' (expected reference|m4.1|pvs|see|aspiration|lmr|null|futility|current|current-lmr|current-threat-aware|current-threat-aware-no-qchecks|current-threat-aware-eval-order|current-threat-aware-eval-only|current-threat-aware-order-only|current-eval2|current-qsearch-movegen|current-qsearch-pruning|current-qsearch-fast-pruning|current-aspiration|current-aspiration-lmr|current-aspiration-lmr-futility|current-aspiration-lmr-futility-see|current-final|current-final-root-history|current-final-root-prev-score|current-final-legality-fast|current-final-single-buffer|current-final-single-generation|current-final-qsearch-lazy|current-final-qsearch-delta|current-final-lmr-null-window|current-final-single-evasion|current-final-bounded-check2|current-final-phase-affine|current-final-eval2|current-final-no-pawn-structure|current-final-no-mobility|current-final-no-piece-activity|current-final-no-rook-activity|current-final-no-development-space|current-final-no-king-safety)",
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
                    "s75b-probe" => diag_s75b_probe = true,
                    other => {
                        return Err(format!(
                            "bench: invalid --diag '{}' (expected no-lmr|no-futility|no-null|no-qsee|root-full-window|s75b-probe)",
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
        diag_s75b_probe,
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
            + format_s72_attribution(&r.stats).as_str()
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

/// S7.2 move-ordering attribution counters, appended to the `bench_result`
/// line for the profile/ablation suites (profiling enabled).
/// OBSERVATION ONLY — no ordering/search semantic changes.
fn format_s72_attribution(stats: &SearchStats) -> String {
    format!(
        " s72_cat_tt={} s72_cat_promo={} s72_cat_capture={} s72_cat_k0={} s72_cat_k1={} s72_cat_hist_quiet={} s72_cat_other_quiet={} s72_nodes_with_quiet={} s72_quiet_available={} s72_quiet_searched={} s72_qsr_r0={} s72_qsr_r1={} s72_qsr_r2_3={} s72_qsr_r4_7={} s72_qsr_r8_15={} s72_qsr_r16p={} s72_qsh_le0={} s72_qsh_1_15={} s72_qsh_16_63={} s72_qsh_64_255={} s72_qsh_256p={} s72_qcg_0={} s72_qcg_1={} s72_qcg_2_3={} s72_qcg_4_7={} s72_qcg_8_15={} s72_qcg_16p={} s72_qcr_r0={} s72_qcr_r1={} s72_qcr_r2_3={} s72_qcr_r4_7={} s72_qcr_r8_15={} s72_qcr_r16p={} s72_qch_le0={} s72_qch_1_15={} s72_qch_16_63={} s72_qch_64_255={} s72_qch_256p={} s72_k0_present={} s72_k0_searched={} s72_k0_cutoffs={} s72_k1_present={} s72_k1_searched={} s72_k1_cutoffs={} s72_k_absent={} s72_tt_present={} s72_tt_searched={} s72_tt_cutoffs={} s72_tt_first_cutoff={} s72_tt_improves_alpha={} s72_cs_1={} s72_cs_2={} s72_cs_3_4={} s72_cs_5_8={} s72_cs_9_16={} s72_cs_17p={} s72_fail_low_nodes={} s72_fail_low_searched_sum={} s72_lmr_q_faillow={} s72_lmr_q_research={} s72_lmr_q_cutoff={} s72_dc_d1={} s72_dc_d2={} s72_dc_d3={} s72_dc_d4_5={} s72_dc_d6_7={} s72_dc_d8p={} s72_dc_late5_d1={} s72_dc_late5_d2={} s72_dc_late5_d3={} s72_dc_late5_d4_5={} s72_dc_late5_d6_7={} s72_dc_late5_d8p={} s72_dfl_d1={} s72_dfl_d2={} s72_dfl_d3={} s72_dfl_d4_5={} s72_dfl_d6_7={} s72_dfl_d8p={} s72_dqs_d1={} s72_dqs_d2={} s72_dqs_d3={} s72_dqs_d4_5={} s72_dqs_d6_7={} s72_dqs_d8p={} s72_dqc_d1={} s72_dqc_d2={} s72_dqc_d3={} s72_dqc_d4_5={} s72_dqc_d6_7={} s72_dqc_d8p={}",
        stats.s72_cutoff_category[0],
        stats.s72_cutoff_category[1],
        stats.s72_cutoff_category[2],
        stats.s72_cutoff_category[3],
        stats.s72_cutoff_category[4],
        stats.s72_cutoff_category[5],
        stats.s72_cutoff_category[6],
        stats.s72_nodes_with_quiet_moves,
        stats.s72_quiet_available,
        stats.s72_quiet_searched,
        stats.s72_quiet_searched_rank[0],
        stats.s72_quiet_searched_rank[1],
        stats.s72_quiet_searched_rank[2],
        stats.s72_quiet_searched_rank[3],
        stats.s72_quiet_searched_rank[4],
        stats.s72_quiet_searched_rank[5],
        stats.s72_quiet_searched_hist[0],
        stats.s72_quiet_searched_hist[1],
        stats.s72_quiet_searched_hist[2],
        stats.s72_quiet_searched_hist[3],
        stats.s72_quiet_searched_hist[4],
        stats.s72_quiet_cutoff_gidx[0],
        stats.s72_quiet_cutoff_gidx[1],
        stats.s72_quiet_cutoff_gidx[2],
        stats.s72_quiet_cutoff_gidx[3],
        stats.s72_quiet_cutoff_gidx[4],
        stats.s72_quiet_cutoff_gidx[5],
        stats.s72_quiet_cutoff_rank[0],
        stats.s72_quiet_cutoff_rank[1],
        stats.s72_quiet_cutoff_rank[2],
        stats.s72_quiet_cutoff_rank[3],
        stats.s72_quiet_cutoff_rank[4],
        stats.s72_quiet_cutoff_rank[5],
        stats.s72_quiet_cutoff_hist[0],
        stats.s72_quiet_cutoff_hist[1],
        stats.s72_quiet_cutoff_hist[2],
        stats.s72_quiet_cutoff_hist[3],
        stats.s72_quiet_cutoff_hist[4],
        stats.s72_killer[0],
        stats.s72_killer[1],
        stats.s72_killer[2],
        stats.s72_killer[3],
        stats.s72_killer[4],
        stats.s72_killer[5],
        stats.s72_killer[6],
        stats.s72_tt_hash[0],
        stats.s72_tt_hash[1],
        stats.s72_tt_hash[2],
        stats.s72_tt_hash[3],
        stats.s72_tt_hash[4],
        stats.s72_cutoff_searched[0],
        stats.s72_cutoff_searched[1],
        stats.s72_cutoff_searched[2],
        stats.s72_cutoff_searched[3],
        stats.s72_cutoff_searched[4],
        stats.s72_cutoff_searched[5],
        stats.s72_fail_low_nodes,
        stats.s72_fail_low_searched_sum,
        stats.s72_lmr[0],
        stats.s72_lmr[1],
        stats.s72_lmr[2],
        stats.s72_d_cutoffs[0],
        stats.s72_d_cutoffs[1],
        stats.s72_d_cutoffs[2],
        stats.s72_d_cutoffs[3],
        stats.s72_d_cutoffs[4],
        stats.s72_d_cutoffs[5],
        stats.s72_d_cutoff_late5[0],
        stats.s72_d_cutoff_late5[1],
        stats.s72_d_cutoff_late5[2],
        stats.s72_d_cutoff_late5[3],
        stats.s72_d_cutoff_late5[4],
        stats.s72_d_cutoff_late5[5],
        stats.s72_d_fail_low[0],
        stats.s72_d_fail_low[1],
        stats.s72_d_fail_low[2],
        stats.s72_d_fail_low[3],
        stats.s72_d_fail_low[4],
        stats.s72_d_fail_low[5],
        stats.s72_d_quiet_searched[0],
        stats.s72_d_quiet_searched[1],
        stats.s72_d_quiet_searched[2],
        stats.s72_d_quiet_searched[3],
        stats.s72_d_quiet_searched[4],
        stats.s72_d_quiet_searched[5],
        stats.s72_d_quiet_cutoffs[0],
        stats.s72_d_quiet_cutoffs[1],
        stats.s72_d_quiet_cutoffs[2],
        stats.s72_d_quiet_cutoffs[3],
        stats.s72_d_quiet_cutoffs[4],
        stats.s72_d_quiet_cutoffs[5],
    ) + format_s73_attribution(stats).as_str()
        + format_s74_attribution(stats).as_str()
        + format_s75_attribution(stats).as_str()
}

/// S7.3 selectivity-attribution counters, appended after the S7.2 block for
/// the profile/ablation suites (profiling enabled). Observation-only.
fn format_s73_attribution(stats: &SearchStats) -> String {
    let nocut_hist: Vec<String> = (0..6)
        .map(|i| {
            format!(
                "{}={}",
                hist_bucket_label(i),
                stats.s73_nocut_searched_hist[i]
            )
        })
        .collect();
    let q4p_idx: Vec<String> = (0..5)
        .map(|i| format!("{}={}", idx_bucket_label(i), stats.s73_q4p_quiet_idx[i]))
        .collect();
    let q4p_red: Vec<String> = (0..3)
        .map(|i| format!("r{}={}", i, stats.s73_q4p_quiet_red[i]))
        .collect();
    let q4p_red_idx: Vec<String> = (0..15)
        .map(|i| {
            format!(
                "r{}_{}={}",
                i / 5,
                idx_bucket_label(i % 5),
                stats.s73_q4p_quiet_red_idx[i]
            )
        })
        .collect();
    let q4p_cut_red: Vec<String> = (0..3)
        .map(|i| format!("r{}={}", i, stats.s73_q4p_quiet_cutoff_red[i]))
        .collect();
    let q4p_cut_idx: Vec<String> = (0..5)
        .map(|i| {
            format!(
                "{}={}",
                idx_bucket_label(i),
                stats.s73_q4p_quiet_cutoff_idx[i]
            )
        })
        .collect();
    let q4p_faillow_red: Vec<String> = (0..3)
        .map(|i| format!("r{}={}", i, stats.s73_q4p_scout_faillow_red[i]))
        .collect();
    format!(
        " s73_loop_nodes={} s73_nocut_pv={} s73_nocut_nonpv={} \
         s73_nocut_incheck={} s73_nocut_null_attempted={} \
         s73_nocut_searched_sum={} s73_nocut_searched_hist:{} \
         s73_null_eligible={} s73_fut_quiet_kept={} \
         s73_q4p_quiet_searched={} s73_q4p_quiet_idx:{} \
         s73_q4p_quiet_red:{} s73_q4p_quiet_red_idx:{} \
         s73_q4p_quiet_cutoff_red:{} s73_q4p_quiet_cutoff_idx:{} \
         s73_q4p_scout_faillow_red:{} s73_q4p_quiet_researched={}",
        stats.s73_loop_nodes,
        stats.s73_nocut_pv,
        stats.s73_nocut_nonpv,
        stats.s73_nocut_incheck,
        stats.s73_nocut_null_attempted,
        stats.s73_nocut_searched_sum,
        nocut_hist.join(","),
        stats.s73_null_eligible,
        stats.s73_fut_quiet_kept,
        stats.s73_q4p_quiet_searched,
        q4p_idx.join(","),
        q4p_red.join(","),
        q4p_red_idx.join(","),
        q4p_cut_red.join(","),
        q4p_cut_idx.join(","),
        q4p_faillow_red.join(","),
        stats.s73_q4p_quiet_researched,
    )
}

fn hist_bucket_label(i: usize) -> &'static str {
    ["le0", "h1_15", "h16_63", "h64_255", "h256p", "unused"][i]
}

/// S7.4A LMR-on-null-window mechanism counters, appended after the S7.3
/// block. `proposed` is the theoretical `late_move_reduction()` value BEFORE
/// window routing; `applied_existing_pvs` counts real reductions in the Scout
/// branch; `suppressed_by_null_window` counts the S7.3 diagnosis population;
/// the `nw_*` family counts the candidate's null-window reduced search /
/// fail-low / re-search / verified-cutoff funnel.
fn format_s74_attribution(stats: &SearchStats) -> String {
    let nw_depth: Vec<String> = (0..4)
        .map(|i| {
            format!(
                "d{}={}",
                ["4", "5", "6", "7p"][i],
                stats.s74_lmr_nw_depth[i]
            )
        })
        .collect();
    let nw_idx: Vec<String> = (0..4)
        .map(|i| {
            format!(
                "i{}={}",
                ["3_4", "5_7", "8_15", "16p"][i],
                stats.s74_lmr_nw_idx[i]
            )
        })
        .collect();
    format!(
        " s74_lmr_proposed={} s74_lmr_proposed_r1={} s74_lmr_proposed_r2={} \
         s74_lmr_applied_existing_pvs={} s74_lmr_suppressed_by_null_window={} \
         s74_lmr_applied_null_window={} s74_lmr_nw_depth:{} s74_lmr_nw_idx:{} \
         s74_lmr_nw_fail_low={} s74_lmr_nw_research={} \
         s74_lmr_nw_research_entered={} \
         s74_lmr_nw_verified_cutoff={}",
        stats.s74_lmr_proposed,
        stats.s74_lmr_proposed_r1,
        stats.s74_lmr_proposed_r2,
        stats.s74_lmr_applied_existing_pvs,
        stats.s74_lmr_suppressed_by_null_window,
        stats.s74_lmr_applied_null_window,
        nw_depth.join(","),
        nw_idx.join(","),
        stats.s74_lmr_nw_fail_low,
        stats.s74_lmr_nw_research,
        stats.s74_lmr_nw_research_entered,
        stats.s74_lmr_nw_verified_cutoff,
    )
}

fn idx_bucket_label(i: usize) -> &'static str {
    ["i0", "i1", "i2_3", "i4_7", "i8p"][i]
}

/// S7.5-0 forcing-opportunity funnel (OBSERVATION ONLY, profiling-gated).
/// Main and qsearch are reported as separate families so the two trees can be
/// analyzed independently.
fn format_s75_attribution(stats: &SearchStats) -> String {
    let chain: Vec<String> = (0..3)
        .map(|i| {
            format!(
                "chain{}={}",
                ["1", "2", "3p"][i],
                stats.s75_main_single_evasion_chain[i]
            )
        })
        .collect();
    format!(
        " s75_main_nodes={} s75_main_in_check_nodes={}          s75_main_single_evasion_nodes_raw={}          s75_main_single_evasion_actionable_depth1={}          s75_main_single_evasion_actionable_depth2plus={}          s75_main_single_evasion_depth3plus={}          s75_main_single_evasion_chain:{}          s75_main_checking_edges_searched={} s75_main_check_child_entered={}          s75_main_check_child_movegen={} s75_main_check_child_terminal_0={}          s75_main_check_child_evasions_1={} s75_main_check_child_evasions_2={}          s75_main_check_child_evasions_3plus={}          s75_main_depth1_nodes={} s75_main_depth1_in_check={}          s75_main_depth1_single_evasion={}          s75_main_depth1_entered_from_checking_edge={}          s75_q_nodes={} s75_q_in_check_nodes={}          s75_q_single_evasion_nodes_raw={} s75_q_single_evasion_qply0={}          s75_q_single_evasion_qply1plus={}          s75_q_checking_edges_searched={} s75_q_check_child_entered={}          s75_q_check_child_movegen={} s75_q_check_child_terminal_0={}          s75_q_check_child_evasions_1={} s75_q_check_child_evasions_2={}          s75_q_check_child_evasions_3plus={}          s75a_extension_applied_total={} s75a_extension_applied_depth1={}          s75a_extension_budget_2_to_1={} s75a_extension_budget_1_to_0={}          s75a_opportunity_blocked_budget_0={}          s75b_checking_edges={} s75b_check2_child_seen={}          s75b_check2_at_parent_depth1={} s75b_check2_at_parent_depth2plus={}          s75b_check2_budget2={} s75b_check2_budget1={} s75b_check2_budget0={}          s75b_extension_opportunities={} s75b_extension_applied={} s75b_extension_blocked_budget0={} s75b_extension_blocked_a_overlap={}          s75b_check2_followed_by_single_evasion={}          s75b_single_evasion_followed_by_check2={}          s75b_probe_calls={} s75b_probe_pseudo_moves={}          s75b_probe_legality_tests={} s75b_probe_claim_skipped={}",
        stats.s75_main_nodes,
        stats.s75_main_in_check_nodes,
        stats.s75_main_single_evasion_nodes_raw,
        stats.s75_main_single_evasion_actionable_depth1,
        stats.s75_main_single_evasion_actionable_depth2plus,
        stats.s75_main_single_evasion_depth3plus,
        chain.join(","),
        stats.s75_main_checking_edges_searched,
        stats.s75_main_check_child_entered,
        stats.s75_main_check_child_movegen,
        stats.s75_main_check_child_terminal_0,
        stats.s75_main_check_child_evasions_1,
        stats.s75_main_check_child_evasions_2,
        stats.s75_main_check_child_evasions_3plus,
        stats.s75_main_depth1_nodes,
        stats.s75_main_depth1_in_check,
        stats.s75_main_depth1_single_evasion,
        stats.s75_main_depth1_entered_from_checking_edge,
        stats.s75_q_nodes,
        stats.s75_q_in_check_nodes,
        stats.s75_q_single_evasion_nodes_raw,
        stats.s75_q_single_evasion_qply0,
        stats.s75_q_single_evasion_qply1plus,
        stats.s75_q_checking_edges_searched,
        stats.s75_q_check_child_entered,
        stats.s75_q_check_child_movegen,
        stats.s75_q_check_child_terminal_0,
        stats.s75_q_check_child_evasions_1,
        stats.s75_q_check_child_evasions_2,
        stats.s75_q_check_child_evasions_3plus,
        stats.s75a_extension_applied_total,
        stats.s75a_extension_applied_depth1,
        stats.s75a_extension_budget_2_to_1,
        stats.s75a_extension_budget_1_to_0,
        stats.s75a_opportunity_blocked_budget_0,
        stats.s75b_checking_edges,
        stats.s75b_check2_child_seen,
        stats.s75b_check2_at_parent_depth1,
        stats.s75b_check2_at_parent_depth2plus,
        stats.s75b_check2_budget2,
        stats.s75b_check2_budget1,
        stats.s75b_check2_budget0,
        stats.s75b_extension_opportunities,
        stats.s75b_extension_applied,
        stats.s75b_extension_blocked_budget0,
        stats.s75b_extension_blocked_a_overlap,
        stats.s75b_check2_followed_by_single_evasion,
        stats.s75b_single_evasion_followed_by_check2,
        stats.s75b_probe_calls,
        stats.s75b_probe_pseudo_moves,
        stats.s75b_probe_legality_tests,
        stats.s75b_probe_claim_skipped,
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
        || cfg.diag_s75b_probe
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
            s75b_probe: cfg.diag_s75b_probe,
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

/// S9-B: `bench eval-cost [--repeats N] [--rounds R] [--fen <fen>]` runs hardened pure evaluation
/// timing across profiles (Current, CurrentFinal, and all 6 Leave-One-Out candidates) to isolate
/// raw evaluator compute costs from tree search effects.
fn run_eval_cost(args: &[String]) -> Result<(), String> {
    let mut custom_fen: Option<String> = None;
    let mut repeats = 50_000u32;
    let mut rounds = 5u32;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-cost: --fen requires a value".to_string())?
                    .clone();
                if custom_fen.is_some() {
                    return Err("eval-cost: --fen may be specified only once".to_string());
                }
                custom_fen = Some(value);
            }
            "--repeats" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-cost: --repeats requires a value".to_string())?
                    .clone();
                let n: u32 = value.parse().map_err(|_| {
                    format!("eval-cost: --repeats '{}' is not a positive integer", value)
                })?;
                if n == 0 {
                    return Err("eval-cost: --repeats must be >= 1".to_string());
                }
                repeats = n;
            }
            "--rounds" => {
                let value = it
                    .next()
                    .ok_or_else(|| "eval-cost: --rounds requires a value".to_string())?
                    .clone();
                let r: u32 = value.parse().map_err(|_| {
                    format!("eval-cost: --rounds '{}' is not a positive integer", value)
                })?;
                if r == 0 {
                    return Err("eval-cost: --rounds must be >= 1".to_string());
                }
                rounds = r;
            }
            other => {
                return Err(format!(
                    "eval-cost: unknown argument '{}' (expected [--repeats N] [--rounds R] [--fen <fen>])",
                    other
                ));
            }
        }
    }

    // 12-position representative evaluation benchmark corpus (excluding exact mop-up)
    let fens: Vec<(&str, String)> = if let Some(ref fen) = custom_fen {
        vec![("custom", fen.clone())]
    } else {
        vec![
            ("01-startpos", START_FEN.to_string()),
            (
                "02-open-italian",
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5".to_string(),
            ),
            (
                "03-closed-french",
                "rnbqk2r/ppp2ppp/4pn2/3p4/1bPP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 2 5".to_string(),
            ),
            (
                "04-sicilian-dragon",
                "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP3PP/2KR1B1R w - - 0 10".to_string(),
            ),
            (
                "05-queens-gambit",
                "r1bqk2r/pp1nbppp/2p1pn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQkq - 4 7".to_string(),
            ),
            (
                "06-tactical-complex",
                "r2q1rk1/1bp1bppp/p1np1n2/1p2p3/4P3/1B1P1NN1/PPP2PPP/R1BQ1RK1 w - - 0 10"
                    .to_string(),
            ),
            (
                "07-king-attack",
                "r1b2rk1/pp1n1ppp/2p1p3/3p4/2PP4/2N1PNP1/PP3PBP/R2Q1RK1 b - - 0 10".to_string(),
            ),
            (
                "08-queenless-mid",
                "r1b2rk1/pp2bppp/2n1pn2/2p5/2B5/2N1PN2/PP3PPP/R1BR2K1 w - - 2 11".to_string(),
            ),
            (
                "09-rook-heavy",
                "2r2rk1/1p3ppp/p3p3/3p4/8/2PR4/PP3PPP/4R1K1 w - - 0 20".to_string(),
            ),
            (
                "10-minor-piece-end",
                "8/5pk1/4p1p1/3n3p/3N3P/5PP1/4K3/8 w - - 0 35".to_string(),
            ),
            (
                "11-pawn-endgame",
                "8/5p2/4p1p1/3k3p/7P/4KPP1/8/8 w - - 0 40".to_string(),
            ),
            (
                "12-asymmetric-imbalance",
                "r4rk1/1pp1qppp/p1np1n2/4p3/2B1P1b1/2NP1N2/PPP2PPP/R2QR1K1 w - - 0 10".to_string(),
            ),
        ]
    };

    let base_profiles: [(&str, Option<Eval2Mask>); 8] = [
        ("current-classical", None),
        ("current-final-all", Some(Eval2Mask::ALL)),
        ("current-final-no-pawn", Some(Eval2Mask::NO_PAWN_STRUCTURE)),
        ("current-final-no-mobility", Some(Eval2Mask::NO_MOBILITY)),
        (
            "current-final-no-piece-act",
            Some(Eval2Mask::NO_PIECE_ACTIVITY),
        ),
        (
            "current-final-no-rook-act",
            Some(Eval2Mask::NO_ROOK_ACTIVITY),
        ),
        (
            "current-final-no-dev-space",
            Some(Eval2Mask::NO_DEVELOPMENT_SPACE),
        ),
        (
            "current-final-no-king-safe",
            Some(Eval2Mask::NO_KING_SAFETY),
        ),
    ];

    println!(
        "eval_cost_header repeats={} rounds={} fens={}",
        repeats,
        rounds,
        fens.len()
    );

    let mut aggregate_times_ns: std::collections::HashMap<&'static str, Vec<f64>> =
        std::collections::HashMap::new();

    for (fname, fen_str) in &fens {
        let pos = parse_fen(fen_str).map_err(|e| format!("eval-cost: invalid FEN: {}", e))?;

        // 1. Warm-up evaluation loop (10,000 passes to warm icache/dcache)
        let warm_pos = std::hint::black_box(&pos);
        for _ in 0..10_000 {
            std::hint::black_box(crate::engine::eval::evaluate(warm_pos));
            std::hint::black_box(crate::engine::eval::evaluate_integrated_positional(
                warm_pos,
            ));
        }

        // Store round measurements: profile_name -> Vec<ns_per_eval>
        let mut profile_round_ns: std::collections::HashMap<&'static str, Vec<f64>> =
            std::collections::HashMap::new();

        for round in 0..rounds {
            // Round-rotated execution order to eliminate thermal and drift bias
            let shift = (round as usize) % base_profiles.len();
            let mut rotated_profiles = base_profiles;
            rotated_profiles.rotate_left(shift);

            for (pname, mask_opt) in &rotated_profiles {
                let start = std::time::Instant::now();
                let mut acc = 0i64;
                let b_pos = std::hint::black_box(&pos);
                match mask_opt {
                    None => {
                        for _ in 0..repeats {
                            let s = crate::engine::eval::evaluate(b_pos);
                            acc = acc.wrapping_add(i64::from(std::hint::black_box(s)));
                        }
                    }
                    Some(mask) => {
                        for _ in 0..repeats {
                            let s = crate::engine::eval::evaluate_integrated_positional_masked(
                                b_pos, *mask,
                            );
                            acc = acc.wrapping_add(i64::from(std::hint::black_box(s)));
                        }
                    }
                }
                std::hint::black_box(acc);
                let elapsed_ns = start.elapsed().as_nanos();
                let ns_per_eval = elapsed_ns as f64 / repeats as f64;
                profile_round_ns.entry(pname).or_default().push(ns_per_eval);
                aggregate_times_ns
                    .entry(pname)
                    .or_default()
                    .push(ns_per_eval);
            }
        }

        for (pname, _) in &base_profiles {
            let mut samples = profile_round_ns[pname].clone();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_ns = samples[samples.len() / 2];
            let mut abs_devs: Vec<f64> = samples.iter().map(|s| (s - median_ns).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad_ns = abs_devs[abs_devs.len() / 2];
            let evals_per_sec = if median_ns > 0.0 {
                1_000_000_000.0 / median_ns
            } else {
                0.0
            };
            println!(
                "eval_cost_fixture fixture={} profile={} median_ns={:.1} mad_ns={:.2} min_ns={:.1} max_ns={:.1} evals_per_sec={:.0}",
                fname,
                pname,
                median_ns,
                mad_ns,
                samples.first().copied().unwrap_or(0.0),
                samples.last().copied().unwrap_or(0.0),
                evals_per_sec
            );
        }
    }

    println!("--- AGGREGATE EVALUATION COMPUTE BENCHMARK SUMMARY ---");
    let full_median_agg = {
        let mut s = aggregate_times_ns["current-final-all"].clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        s[s.len() / 2]
    };

    for (pname, _) in &base_profiles {
        let mut samples = aggregate_times_ns[pname].clone();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ns = samples[samples.len() / 2];
        let mut abs_devs: Vec<f64> = samples.iter().map(|s| (s - median_ns).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad_ns = abs_devs[abs_devs.len() / 2];
        let marginal_cost_saved_ns = full_median_agg - median_ns;
        println!(
            "eval_cost_summary profile={} median_ns={:.1} mad_ns={:.2} marginal_saved_ns={:.1} (relative to full)",
            pname,
            median_ns,
            mad_ns,
            marginal_cost_saved_ns
        );
    }

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
    println!("{}", eval_breakdown_line(&pos, &fen));
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
    if args[0] == "eval-cost" {
        return run_eval_cost(&args[1..]);
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
    if args[0] == "phase-affine-batch" {
        return run_phase_affine_batch(&args[1..]);
    }
    if args[0] == "phase-affine-microbench" {
        return run_phase_affine_microbench(&args[1..]);
    }
    if args[0] == "nnue-features" {
        return run_nnue_features(&args[1..]);
    }
    if args[0] == "nnue-features-batch" {
        return run_nnue_features_batch(&args[1..]);
    }
    if args[0] == "nnue-feature-cost" {
        return run_nnue_feature_cost(&args[1..]);
    }
    if args[0] == "nnue-probe" {
        return run_nnue_probe(&args[1..]);
    }
    if args[0] == "nnue-probe-batch" {
        return run_nnue_probe_batch(&args[1..]);
    }
    if args[0] == "nnue-probe-microbench" {
        return run_nnue_probe_microbench(&args[1..]);
    }
    if args[0] == "nnue-v2-probe" {
        return run_nnue_v2_probe(&args[1..]);
    }
    if args[0] == "nnue-v2-probe-batch" {
        return run_nnue_v2_probe_batch(&args[1..]);
    }
    if args[0] == "nnue-v2q-probe" {
        return run_nnue_v2q_probe(&args[1..]);
    }
    if args[0] == "nnue-v2q-probe-batch" {
        return run_nnue_v2q_probe_batch(&args[1..]);
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
    println!("  eval-cost   raw isolated evaluator timing across feature family masks");
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
        "  --profile <reference|m4.1|pvs|see|aspiration|lmr|null|futility|current|current-lmr|current-threat-aware|current-threat-aware-no-qchecks|current-threat-aware-eval-order|current-threat-aware-eval-only|current-threat-aware-order-only|current-eval2|current-qsearch-movegen|current-qsearch-pruning|current-qsearch-fast-pruning|current-aspiration|current-aspiration-lmr|current-aspiration-lmr-futility|current-aspiration-lmr-futility-see|current-final|current-final-bounded-check2|current-final-phase-affine|current-final-eval2|current-final-no-pawn-structure|current-final-no-mobility|current-final-no-piece-activity|current-final-no-rook-activity|current-final-no-development-space|current-final-no-king-safety>  search profile (default reference == M4.0 baseline)"
    );
    println!();
    println!("OUTPUT PREFIXES: bench_result / bench_summary / bench_error");
    println!();
    println!("EXPORT COMMANDS:");
    println!("  nnue-features --fen <fen>          one sparse NnueFeatureSetV1 JSON line");
    println!("  nnue-features-batch --batch <file> one JSON line per record (id|fen or plain fen)");
    println!("  nnue-probe --model <bin> --fen <fen>  S6-N2 full-refresh inference, one JSON line");
    println!("  nnue-probe-batch --model <bin> --batch <file>  one JSON line per record");
    println!("  nnue-probe-microbench --model <bin> --batch <file> --iterations <N>  cost probe");
    println!("  nnue-v2-probe --model <bin> --fen <fen>  S10-B4 V2 full-refresh FP32 inference, one JSON line");
    println!("  nnue-v2-probe-batch --model <bin> --batch <file>  one JSON line per record");
    println!("  nnue-v2q-probe --model <bin> --fen <fen>  S10-B5 V2 quantized integer inference, one JSON line");
    println!("  nnue-v2q-probe-batch --model <bin> --batch <file>  one JSON line per record");
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

/// S6-N1 / S10-A: `bench nnue-features --fen <fen> [--feature-set v1|v2]` - deterministic sparse
/// NNUE export (JSON line). `active_features_for()` in [`crate::engine::nnue`] is the single
/// encoding source of truth; this is only a CLI bridge. Observation-only; never wired into evaluation/search.
fn run_nnue_features(args: &[String]) -> Result<(), String> {
    let mut fen: Option<String> = None;
    let mut feature_set = crate::engine::nnue::NnueFeatureSet::V1;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-features: --fen requires a value".to_string())?
                    .clone();
                fen = Some(value);
            }
            "--feature-set" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-features: --feature-set requires a value".to_string())?;
                match value.to_lowercase().as_str() {
                    "v1" => feature_set = crate::engine::nnue::NnueFeatureSet::V1,
                    "v2" => feature_set = crate::engine::nnue::NnueFeatureSet::V2,
                    other => {
                        return Err(format!(
                            "nnue-features: unknown feature set '{other}' (expected v1|v2)"
                        ))
                    }
                }
            }
            other => {
                return Err(format!(
                    "nnue-features: unknown argument '{}' (expected --fen <fen> [--feature-set v1|v2])",
                    other
                ));
            }
        }
    }
    let fen = fen.ok_or_else(|| "nnue-features: --fen is required".to_string())?;
    println!("{}", nnue_features_for_fen(&fen, feature_set)?);
    Ok(())
}

fn nnue_features_for_fen(
    fen: &str,
    feature_set: crate::engine::nnue::NnueFeatureSet,
) -> Result<String, String> {
    let pos = parse_fen(fen).map_err(|e| format!("nnue-features: {e}"))?;
    Ok(nnue_features_line(&pos, fen, None, feature_set))
}

/// S6-N1 / S10-A: `bench nnue-features-batch --batch <file> [--feature-set v1|v2]` - one JSON line per
/// input record, in deterministic file order. Each non-empty, non-comment
/// line is either `position_id|fen` (the id round-trips into the output) or a
/// plain FEN (no position_id emitted).
fn run_nnue_features_batch(args: &[String]) -> Result<(), String> {
    let mut batch: Option<String> = None;
    let mut feature_set = crate::engine::nnue::NnueFeatureSet::V1;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-features-batch: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            "--feature-set" => {
                let value = it.next().ok_or_else(|| {
                    "nnue-features-batch: --feature-set requires a value".to_string()
                })?;
                match value.to_lowercase().as_str() {
                    "v1" => feature_set = crate::engine::nnue::NnueFeatureSet::V1,
                    "v2" => feature_set = crate::engine::nnue::NnueFeatureSet::V2,
                    other => {
                        return Err(format!(
                            "nnue-features-batch: unknown feature set '{other}' (expected v1|v2)"
                        ))
                    }
                }
            }
            other => {
                return Err(format!(
                    "nnue-features-batch: unknown argument '{}' (expected --batch <file> [--feature-set v1|v2])",
                    other
                ));
            }
        }
    }
    let batch = batch.ok_or_else(|| "nnue-features-batch: --batch is required".to_string())?;
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("nnue-features-batch: cannot read {batch}: {e}"))?;
    print!("{}", nnue_features_batch_from_text(&text, feature_set)?);
    Ok(())
}

/// S6-C1: batch diagnostic for the frozen phase-affine calibration.
///
/// Emits base and calibrated cp for each position so the Python verifier can
/// compare the Rust runtime against the frozen formula without re-implementing
/// the classical evaluator. `evaluate_breakdown` runs once per position, and
/// `calibrated_cp` is produced by the SAME function the search dispatch uses.
fn run_phase_affine_batch(args: &[String]) -> Result<(), String> {
    let mut batch: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "phase-affine-batch: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            other => {
                return Err(format!(
                    "phase-affine-batch: unknown argument '{}' (expected --batch <file>)",
                    other
                ));
            }
        }
    }
    let batch = batch.ok_or_else(|| "phase-affine-batch: --batch is required".to_string())?;
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("phase-affine-batch: cannot read {batch}: {e}"))?;
    print!("{}", phase_affine_batch_from_text(&text)?);
    Ok(())
}

fn phase_affine_batch_from_text(text: &str) -> Result<String, String> {
    use crate::engine::eval::{evaluate_breakdown_public, phase_affine_bucket_public};
    let mut out = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (position_id, fen) = match line.split_once('|') {
            Some((id, fen)) => (id.trim(), fen.trim()),
            None => ("", line),
        };
        let pos = parse_fen(fen).map_err(|e| format!("phase-affine-batch: {e}: '{fen}'"))?;
        let (phase, base_cp) = evaluate_breakdown_public(&pos);
        let calibrated_cp = crate::engine::eval::evaluate_phase_affine(&pos);
        out.push_str(&format!(
            "{{\"position_id\":\"{}\",\"fen\":\"{}\",\"phase\":{},\"bucket\":{},\
\"base_cp\":{},\"calibrated_cp\":{},\"correction_cp\":{}}}\n",
            position_id,
            fen,
            phase,
            phase_affine_bucket_public(phase),
            base_cp,
            calibrated_cp,
            calibrated_cp - base_cp
        ));
    }
    Ok(out)
}

/// S6-C1: median-of-5 microbench comparing the calibrated evaluator against the
/// base classical evaluator. Loading and parsing happen OUTSIDE the timed
/// region; only the evaluation loops are measured.
fn run_phase_affine_microbench(args: &[String]) -> Result<(), String> {
    use crate::engine::eval::{evaluate, evaluate_phase_affine};
    let mut batch: Option<String> = None;
    let mut iterations: usize = 200;
    let mut rounds: usize = 5;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--batch" => {
                batch = Some(
                    it.next()
                        .ok_or_else(|| {
                            "phase-affine-microbench: --batch requires a value".to_string()
                        })?
                        .clone(),
                );
            }
            "--iterations" => {
                iterations = it
                    .next()
                    .ok_or_else(|| {
                        "phase-affine-microbench: --iterations requires a value".to_string()
                    })?
                    .parse()
                    .map_err(|e| format!("phase-affine-microbench: bad --iterations: {e}"))?;
            }
            "--rounds" => {
                rounds = it
                    .next()
                    .ok_or_else(|| {
                        "phase-affine-microbench: --rounds requires a value".to_string()
                    })?
                    .parse()
                    .map_err(|e| format!("phase-affine-microbench: bad --rounds: {e}"))?;
            }
            other => {
                return Err(format!(
                    "phase-affine-microbench: unknown argument '{}'",
                    other
                ));
            }
        }
    }
    let batch = batch.ok_or_else(|| "phase-affine-microbench: --batch is required".to_string())?;
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("phase-affine-microbench: cannot read {batch}: {e}"))?;
    // Parse everything up front: parsing is not part of the measurement.
    let mut positions = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fen = match line.split_once('|') {
            Some((_, fen)) => fen.trim(),
            None => line,
        };
        positions
            .push(parse_fen(fen).map_err(|e| format!("phase-affine-microbench: {e}: '{fen}'"))?);
    }
    if positions.is_empty() {
        return Err("phase-affine-microbench: batch has no positions".to_string());
    }
    // Every call black-boxes BOTH its input and its result, so the optimizer
    // cannot hoist, CSE, or delete repeated evaluations of the same position.
    let time_base = |positions: &[Position], iterations: usize| -> u128 {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for pos in positions {
                std::hint::black_box(evaluate(std::hint::black_box(pos)));
            }
        }
        start.elapsed().as_nanos()
    };
    let time_candidate = |positions: &[Position], iterations: usize| -> u128 {
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for pos in positions {
                std::hint::black_box(evaluate_phase_affine(std::hint::black_box(pos)));
            }
        }
        start.elapsed().as_nanos()
    };

    let mut base_ns = Vec::with_capacity(rounds);
    let mut cand_ns = Vec::with_capacity(rounds);
    let mut order = Vec::with_capacity(rounds);
    for round in 0..rounds {
        // Warm up BOTH evaluators outside the timed region so neither pays for
        // cold caches or first-touch branch prediction.
        let _ = time_base(&positions, 1);
        let _ = time_candidate(&positions, 1);
        // Alternate which evaluator runs first: a fixed order would hand a
        // systematic advantage to one side and bias the <=1.10 gate.
        let (base, cand) = if round % 2 == 0 {
            let base = time_base(&positions, iterations);
            let cand = time_candidate(&positions, iterations);
            (base, cand)
        } else {
            let cand = time_candidate(&positions, iterations);
            let base = time_base(&positions, iterations);
            (base, cand)
        };
        order.push(if round % 2 == 0 {
            "base_first"
        } else {
            "candidate_first"
        });
        base_ns.push(base);
        cand_ns.push(cand);
    }
    base_ns.sort_unstable();
    cand_ns.sort_unstable();
    let base_median = base_ns[rounds / 2];
    let cand_median = cand_ns[rounds / 2];
    let evals = (positions.len() * iterations) as f64;
    println!(
        "{{\"positions\":{},\"iterations\":{},\"rounds\":{},\
\"warmup_per_round\":true,\"order_alternated\":true,\"round_order\":{:?},\
\"base_median_ns\":{},\"candidate_median_ns\":{},\
\"base_ns_per_eval\":{:.4},\"candidate_ns_per_eval\":{:.4},\"ratio\":{:.6}}}",
        positions.len(),
        iterations,
        rounds,
        order,
        base_median,
        cand_median,
        base_median as f64 / evals,
        cand_median as f64 / evals,
        cand_median as f64 / base_median as f64
    );
    Ok(())
}

/// Process one batch input text (used by both the CLI bridge and the tests).
fn nnue_features_batch_from_text(
    text: &str,
    feature_set: crate::engine::nnue::NnueFeatureSet,
) -> Result<String, String> {
    let mut out = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (position_id, fen) = match line.split_once('|') {
            Some((id, fen)) => (Some(id.trim()), fen.trim()),
            None => (None, line),
        };
        let pos = parse_fen(fen).map_err(|e| format!("nnue-features-batch: {e}: '{fen}'"))?;
        out.push_str(&nnue_features_line(&pos, fen, position_id, feature_set));
        out.push('\n');
    }
    Ok(out)
}

/// S4.2A: one `eval_breakdown` line for a position. `base_eval_stm` is the
/// exact side-to-move-perspective score from `crate::engine::eval::evaluate`,
/// so Python consumers never re-implement the classical evaluator.
fn eval_breakdown_line(pos: &Position, fen: &str) -> String {
    let cps = evaluate_components_white(pos);
    let side = if pos.side == crate::chess::types::Color::White {
        "w"
    } else {
        "b"
    };
    format!(
        "eval_breakdown fen=\"{}\" side={} phase={} material_pst={} pawn_structure={} mobility={} piece_activity={} rook_activity={} development_space={} king_safety={} base_eval_stm={}",
        json_escape(fen),
        side,
        cps.phase,
        cps.material_pst,
        cps.pawn_structure,
        cps.mobility,
        cps.piece_activity,
        cps.rook_activity,
        cps.development_space,
        cps.king_safety,
        crate::engine::eval::evaluate(pos),
    )
}

/// JSON string escaping for export lines: `"` -> `\"`, `\` -> `\\`, and every
/// control character U+0000..=U+001F is escaped (`\b \f \n \r \t` short forms,
/// otherwise `\u00xx`). Identity for all other bytes, so normal FENs and ids
/// produce byte-identical output.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Deterministic JSON line for one position's sparse NNUE export:
/// `{"position_id":?, "fen": "...", "white": [u16...], "black": [u16...]}`.
fn nnue_features_line(
    pos: &Position,
    fen: &str,
    position_id: Option<&str>,
    feature_set: crate::engine::nnue::NnueFeatureSet,
) -> String {
    use crate::engine::nnue::{active_features_for, NnuePerspective};
    let white = active_features_for(pos, NnuePerspective::White, feature_set);
    let black = active_features_for(pos, NnuePerspective::Black, feature_set);
    let escaped_fen = json_escape(fen);
    let escaped_id = json_escape(position_id.unwrap_or(""));
    let mut out = String::from("{");
    if position_id.is_some() {
        out.push_str(&format!("\"position_id\":\"{escaped_id}\","));
    }
    out.push_str(&format!("\"fen\":\"{escaped_fen}\","));
    out.push_str("\"white\":[");
    for (i, v) in white.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.push_str("],\"black\":[");
    for (i, v) in black.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.push_str("]}");
    out
}

/// S10-A: `bench nnue-feature-cost --batch <file> [--feature-set v1|v2|both] [--repeats N] [--rounds R]`
/// Measures pure in-memory Rust feature extraction cost (`ns/position`) across pre-parsed positions.
/// All file I/O and FEN parsing happen outside the timed region.
fn run_nnue_feature_cost(args: &[String]) -> Result<(), String> {
    let mut batch: Option<String> = None;
    let mut mode = "both";
    let mut repeats = 100u32;
    let mut rounds = 7u32;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-feature-cost: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            "--feature-set" => {
                let value = it.next().ok_or_else(|| {
                    "nnue-feature-cost: --feature-set requires a value".to_string()
                })?;
                match value.to_lowercase().as_str() {
                    "v1" => mode = "v1",
                    "v2" => mode = "v2",
                    "both" => mode = "both",
                    other => {
                        return Err(format!(
                            "nnue-feature-cost: unknown feature set '{other}' (expected v1|v2|both)"
                        ))
                    }
                }
            }
            "--repeats" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-feature-cost: --repeats requires a value".to_string())?;
                repeats = value
                    .parse()
                    .map_err(|_| format!("nnue-feature-cost: invalid --repeats '{value}'"))?;
            }
            "--rounds" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-feature-cost: --rounds requires a value".to_string())?;
                rounds = value
                    .parse()
                    .map_err(|_| format!("nnue-feature-cost: invalid --rounds '{value}'"))?;
            }
            other => {
                return Err(format!(
                    "nnue-feature-cost: unknown argument '{other}' (expected --batch <file> [--feature-set v1|v2|both] [--repeats N] [--rounds R])"
                ));
            }
        }
    }

    let batch_path = batch.ok_or_else(|| "nnue-feature-cost: --batch is required".to_string())?;
    let text = std::fs::read_to_string(&batch_path)
        .map_err(|e| format!("nnue-feature-cost: cannot read {batch_path}: {e}"))?;

    // Pre-parse all positions completely OUTSIDE the timed region
    let mut positions = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fen = if line.starts_with('{') {
            // JSON record: extract "fen" field
            if let Some(idx) = line.find("\"fen\":") {
                let rest = &line[idx + 6..].trim_start();
                if let Some(start_quote) = rest.find('"') {
                    let after_quote = &rest[start_quote + 1..];
                    if let Some(end_quote) = after_quote.find('"') {
                        &after_quote[..end_quote]
                    } else {
                        line
                    }
                } else {
                    line
                }
            } else {
                line
            }
        } else {
            match line.split_once('|') {
                Some((_, fen)) => fen.trim(),
                None => line,
            }
        };
        let pos = parse_fen(fen).map_err(|e| format!("nnue-feature-cost: {e}: '{fen}'"))?;
        positions.push(pos);
    }

    if positions.is_empty() {
        return Err("nnue-feature-cost: batch file contains 0 valid positions".to_string());
    }

    let num_positions = positions.len();

    // Warm-up loop
    for pos in &positions {
        let bp = std::hint::black_box(pos);
        std::hint::black_box(crate::engine::nnue::active_features_v1(
            bp,
            crate::engine::nnue::NnuePerspective::White,
        ));
        std::hint::black_box(crate::engine::nnue::active_features_v1(
            bp,
            crate::engine::nnue::NnuePerspective::Black,
        ));
        std::hint::black_box(crate::engine::nnue::active_features_v2(
            bp,
            crate::engine::nnue::NnuePerspective::White,
        ));
        std::hint::black_box(crate::engine::nnue::active_features_v2(
            bp,
            crate::engine::nnue::NnuePerspective::Black,
        ));
    }

    println!(
        "nnue_feature_cost_header positions={} repeats={} rounds={} mode={}",
        num_positions, repeats, rounds, mode
    );

    let sets_to_run: Vec<crate::engine::nnue::NnueFeatureSet> = match mode {
        "v1" => vec![crate::engine::nnue::NnueFeatureSet::V1],
        "v2" => vec![crate::engine::nnue::NnueFeatureSet::V2],
        _ => vec![
            crate::engine::nnue::NnueFeatureSet::V1,
            crate::engine::nnue::NnueFeatureSet::V2,
        ],
    };

    let mut round_ns_per_pos: std::collections::HashMap<&'static str, Vec<f64>> =
        std::collections::HashMap::new();

    for round in 0..rounds {
        let mut cur_sets = sets_to_run.clone();
        if round % 2 == 1 && cur_sets.len() == 2 {
            cur_sets.swap(0, 1);
        }

        for &fset in &cur_sets {
            let label = match fset {
                crate::engine::nnue::NnueFeatureSet::V1 => "v1",
                crate::engine::nnue::NnueFeatureSet::V2 => "v2",
            };

            let start = std::time::Instant::now();
            let mut total_active: usize = 0;

            for _ in 0..repeats {
                for pos in &positions {
                    let bp = std::hint::black_box(pos);
                    let w = crate::engine::nnue::active_features_for(
                        bp,
                        crate::engine::nnue::NnuePerspective::White,
                        fset,
                    );
                    let b = crate::engine::nnue::active_features_for(
                        bp,
                        crate::engine::nnue::NnuePerspective::Black,
                        fset,
                    );
                    total_active += w.len() + b.len();
                }
            }

            std::hint::black_box(total_active);
            let elapsed_ns = start.elapsed().as_nanos();
            let total_evals = (num_positions as u64) * (repeats as u64);
            let ns_per_pos = elapsed_ns as f64 / total_evals as f64;
            round_ns_per_pos.entry(label).or_default().push(ns_per_pos);
        }
    }

    for &label in &["v1", "v2"] {
        if let Some(samples) = round_ns_per_pos.get_mut(label) {
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = samples[samples.len() / 2];
            let mut abs_devs: Vec<f64> = samples.iter().map(|s| (s - median).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = abs_devs[abs_devs.len() / 2];
            println!(
                "nnue_feature_cost_result feature_set={} median_ns_per_pos={:.2} mad_ns={:.2} min_ns={:.2} max_ns={:.2}",
                label, median, mad, samples.first().unwrap_or(&0.0), samples.last().unwrap_or(&0.0)
            );
        }
    }

    Ok(())
}

/// S6-N2: `bench nnue-probe --model <bin> --fen <fen>` - one JSON line with
/// the scaled prediction and centipawn prediction from the local probe
/// artifact. Full-refresh inference; observation-only.
fn run_nnue_probe(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut fen: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-probe: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-probe: --fen requires a value".to_string())?
                    .clone();
                fen = Some(value);
            }
            other => {
                return Err(format!(
                    "nnue-probe: unknown argument '{}' (expected --model <bin> --fen <fen>)",
                    other
                ));
            }
        }
    }
    let model_path = model.ok_or_else(|| "nnue-probe: --model is required".to_string())?;
    let fen = fen.ok_or_else(|| "nnue-probe: --fen is required".to_string())?;
    let model =
        crate::engine::nnue_probe::NnueProbeModelV1::load(std::path::Path::new(&model_path))?;
    let pos = parse_fen(&fen).map_err(|e| format!("nnue-probe: {e}"))?;
    println!("{}", nnue_probe_line(&model, &pos, &fen, None));
    Ok(())
}

/// S6-N2: `bench nnue-probe-batch --model <bin> --batch <file>` - one JSON
/// line per input record (`position_id|fen` or plain FEN), deterministic
/// order, full-refresh inference.
fn run_nnue_probe_batch(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut batch: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-probe-batch: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-probe-batch: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            other => {
                return Err(format!(
                    "nnue-probe-batch: unknown argument '{}' (expected --model <bin> --batch <file>)",
                    other
                ));
            }
        }
    }
    let model_path = model.ok_or_else(|| "nnue-probe-batch: --model is required".to_string())?;
    let batch = batch.ok_or_else(|| "nnue-probe-batch: --batch is required".to_string())?;
    let model =
        crate::engine::nnue_probe::NnueProbeModelV1::load(std::path::Path::new(&model_path))?;
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("nnue-probe-batch: cannot read {batch}: {e}"))?;
    print!("{}", nnue_probe_batch_from_text(&model, &text)?);
    Ok(())
}

/// Process one batch input text (shared by the CLI bridge and tests).
fn nnue_probe_batch_from_text(
    model: &crate::engine::nnue_probe::NnueProbeModelV1,
    text: &str,
) -> Result<String, String> {
    let mut out = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (position_id, fen) = match line.split_once('|') {
            Some((id, fen)) => (Some(id.trim()), fen.trim()),
            None => (None, line),
        };
        let pos = parse_fen(fen).map_err(|e| format!("nnue-probe-batch: {e}: '{fen}'"))?;
        out.push_str(&nnue_probe_line(model, &pos, fen, position_id));
        out.push('\n');
    }
    Ok(out)
}

/// Deterministic JSON line for one position's full-refresh probe prediction:
/// `{"position_id":?, "fen": "...", "scaled_prediction": f32,
///   "prediction_cp": f32}`.
fn nnue_probe_line(
    model: &crate::engine::nnue_probe::NnueProbeModelV1,
    pos: &Position,
    fen: &str,
    position_id: Option<&str>,
) -> String {
    let scaled = model.evaluate_scaled(pos);
    let cp = model.evaluate_cp(pos);
    let escaped_fen = json_escape(fen);
    let escaped_id = json_escape(position_id.unwrap_or(""));
    let mut out = String::from("{");
    if position_id.is_some() {
        out.push_str(&format!("\"position_id\":\"{escaped_id}\","));
    }
    out.push_str(&format!(
        "\"fen\":\"{escaped_fen}\",\"scaled_prediction\":{scaled},\"prediction_cp\":{cp}}}"
    ));
    out
}

/// S10-B4: `bench nnue-v2-probe --model <bin> --fen <fen>` - one JSON line
/// with the full-refresh FP32 V2 inference result.
fn run_nnue_v2_probe(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut fen: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2-probe: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2-probe: --fen requires a value".to_string())?
                    .clone();
                fen = Some(value);
            }
            other => {
                return Err(format!(
                    "nnue-v2-probe: unknown argument '{}' (expected --model <bin> --fen <fen>)",
                    other
                ));
            }
        }
    }
    let model_path = model.ok_or_else(|| "nnue-v2-probe: --model is required".to_string())?;
    let fen = fen.ok_or_else(|| "nnue-v2-probe: --fen is required".to_string())?;
    let model =
        crate::engine::nnue_v2_runtime::NnueV2Model::load(std::path::Path::new(&model_path))?;
    let pos = parse_fen(&fen).map_err(|e| format!("nnue-v2-probe: {e}"))?;
    println!("{}", nnue_v2_probe_line(&model, &pos, &fen, None));
    Ok(())
}

/// S10-B4: `bench nnue-v2-probe-batch --model <bin> --batch <file>` - one
/// JSON line per input record (`position_id|fen` or plain FEN),
/// deterministic order, full-refresh FP32 inference.
fn run_nnue_v2_probe_batch(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut batch: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2-probe-batch: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2-probe-batch: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            other => {
                return Err(format!(
                    "nnue-v2-probe-batch: unknown argument '{}' (expected --model <bin> --batch <file>)",
                    other
                ));
            }
        }
    }
    let model_path =
        model.ok_or_else(|| "nnue-v2-probe-batch: --model is required".to_string())?;
    let batch = batch.ok_or_else(|| "nnue-v2-probe-batch: --batch is required".to_string())?;
    let model =
        crate::engine::nnue_v2_runtime::NnueV2Model::load(std::path::Path::new(&model_path))?;
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("nnue-v2-probe-batch: cannot read {batch}: {e}"))?;
    print!("{}", nnue_v2_probe_batch_from_text(&model, &text)?);
    Ok(())
}

/// Process one batch input text (shared by the CLI bridge and tests).
fn nnue_v2_probe_batch_from_text(
    model: &crate::engine::nnue_v2_runtime::NnueV2Model,
    text: &str,
) -> Result<String, String> {
    let mut out = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (position_id, fen) = match line.split_once('|') {
            Some((id, fen)) => (Some(id.trim()), fen.trim()),
            None => (None, line),
        };
        let pos =
            parse_fen(fen).map_err(|e| format!("nnue-v2-probe-batch: {e}: '{fen}'"))?;
        out.push_str(&nnue_v2_probe_line(model, &pos, fen, position_id));
        out.push('\n');
    }
    Ok(out)
}

/// Deterministic JSON line for one position's full-refresh V2 prediction:
/// `{"position_id":?, "fen": "...", "scaled_prediction": f32,
///   "prediction_cp": f32}`.
fn nnue_v2_probe_line(
    model: &crate::engine::nnue_v2_runtime::NnueV2Model,
    pos: &Position,
    fen: &str,
    position_id: Option<&str>,
) -> String {
    let scaled = model.evaluate_scaled(pos);
    let cp = model.evaluate_cp(pos);
    let escaped_fen = json_escape(fen);
    let escaped_id = json_escape(position_id.unwrap_or(""));
    let mut out = String::from("{");
    if position_id.is_some() {
        out.push_str(&format!("\"position_id\":\"{escaped_id}\","));
    }
    out.push_str(&format!(
        "\"fen\":\"{escaped_fen}\",\"scaled_prediction\":{scaled},\"prediction_cp\":{cp}}}"
    ));
    out
}

/// S10-B5: `bench nnue-v2q-probe --model <bin> --fen <fen>` - one JSON line
/// with the quantized integer inference result (raw integer output + cp).
fn run_nnue_v2q_probe(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut fen: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2q-probe: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--fen" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2q-probe: --fen requires a value".to_string())?
                    .clone();
                fen = Some(value);
            }
            other => {
                return Err(format!(
                    "nnue-v2q-probe: unknown argument '{}' (expected --model <bin> --fen <fen>)",
                    other
                ));
            }
        }
    }
    let model_path = model.ok_or_else(|| "nnue-v2q-probe: --model is required".to_string())?;
    let fen = fen.ok_or_else(|| "nnue-v2q-probe: --fen is required".to_string())?;
    let model = crate::engine::nnue_v2q_runtime::NnueV2QuantizedModel::load(
        std::path::Path::new(&model_path))?;
    let pos = parse_fen(&fen).map_err(|e| format!("nnue-v2q-probe: {e}"))?;
    println!("{}", nnue_v2q_probe_line(&model, &pos, &fen, None));
    Ok(())
}

/// S10-B5: `bench nnue-v2q-probe-batch --model <bin> --batch <file>` - one
/// JSON line per input record (`position_id|fen` or plain FEN),
/// deterministic order, full-refresh integer inference.
fn run_nnue_v2q_probe_batch(args: &[String]) -> Result<(), String> {
    let mut model: Option<String> = None;
    let mut batch: Option<String> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2q-probe-batch: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-v2q-probe-batch: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            other => {
                return Err(format!(
                    "nnue-v2q-probe-batch: unknown argument '{}' (expected --model <bin> --batch <file>)",
                    other
                ));
            }
        }
    }
    let model_path =
        model.ok_or_else(|| "nnue-v2q-probe-batch: --model is required".to_string())?;
    let batch = batch.ok_or_else(|| "nnue-v2q-probe-batch: --batch is required".to_string())?;
    let model = crate::engine::nnue_v2q_runtime::NnueV2QuantizedModel::load(
        std::path::Path::new(&model_path))?;
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("nnue-v2q-probe-batch: cannot read {batch}: {e}"))?;
    print!("{}", nnue_v2q_probe_batch_from_text(&model, &text)?);
    Ok(())
}

/// Process one batch input text (shared by the CLI bridge and tests).
fn nnue_v2q_probe_batch_from_text(
    model: &crate::engine::nnue_v2q_runtime::NnueV2QuantizedModel,
    text: &str,
) -> Result<String, String> {
    let mut out = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (position_id, fen) = match line.split_once('|') {
            Some((id, fen)) => (Some(id.trim()), fen.trim()),
            None => (None, line),
        };
        let pos =
            parse_fen(fen).map_err(|e| format!("nnue-v2q-probe-batch: {e}: '{fen}'"))?;
        out.push_str(&nnue_v2q_probe_line(model, &pos, fen, position_id));
        out.push('\n');
    }
    Ok(out)
}

/// Deterministic JSON line for one position's quantized prediction:
/// `{"position_id":?, "fen": "...", "raw_output": i32, "prediction_cp": f32}`.
fn nnue_v2q_probe_line(
    model: &crate::engine::nnue_v2q_runtime::NnueV2QuantizedModel,
    pos: &Position,
    fen: &str,
    position_id: Option<&str>,
) -> String {
    let raw = model.evaluate_raw(pos);
    let cp = model.evaluate_cp(pos);
    let escaped_fen = json_escape(fen);
    let escaped_id = json_escape(position_id.unwrap_or(""));
    let mut out = String::from("{");
    if position_id.is_some() {
        out.push_str(&format!("\"position_id\":\"{escaped_id}\","));
    }
    out.push_str(&format!(
        "\"fen\":\"{escaped_fen}\",\"raw_output\":{raw},\"prediction_cp\":{cp}}}"
    ));
    out
}

/// S6-N2: `bench nnue-probe-microbench --model <bin> --batch <file>
/// --iterations <N>` - five rounds measuring, per position visit: feature
/// extraction (both perspectives), full full-refresh NNUE, and classical
/// evaluate. Model + FEN parsing happen outside the timed region; black_box
/// and checksums prevent dead-code elimination. Cost-only; no promotion gate.
fn run_nnue_probe_microbench(args: &[String]) -> Result<(), String> {
    use std::time::Instant;

    let mut model: Option<String> = None;
    let mut batch: Option<String> = None;
    let mut iterations: Option<u64> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-probe-microbench: --model requires a value".to_string())?
                    .clone();
                model = Some(value);
            }
            "--batch" => {
                let value = it
                    .next()
                    .ok_or_else(|| "nnue-probe-microbench: --batch requires a value".to_string())?
                    .clone();
                batch = Some(value);
            }
            "--iterations" => {
                let value = it
                    .next()
                    .ok_or_else(|| {
                        "nnue-probe-microbench: --iterations requires a value".to_string()
                    })?
                    .clone();
                let n: u64 = value.parse().map_err(|_| {
                    format!(
                        "nnue-probe-microbench: --iterations '{value}' is not a positive integer"
                    )
                })?;
                if n == 0 {
                    return Err("nnue-probe-microbench: --iterations must be >= 1".to_string());
                }
                iterations = Some(n);
            }
            other => {
                return Err(format!(
                    "nnue-probe-microbench: unknown argument '{}' (expected --model <bin> --batch <file> --iterations <N>)",
                    other
                ));
            }
        }
    }
    let model_path =
        model.ok_or_else(|| "nnue-probe-microbench: --model is required".to_string())?;
    let batch = batch.ok_or_else(|| "nnue-probe-microbench: --batch is required".to_string())?;
    let iterations =
        iterations.ok_or_else(|| "nnue-probe-microbench: --iterations is required".to_string())?;
    let model =
        crate::engine::nnue_probe::NnueProbeModelV1::load(std::path::Path::new(&model_path))?;

    // Parse every FEN BEFORE timing.
    let text = std::fs::read_to_string(&batch)
        .map_err(|e| format!("nnue-probe-microbench: cannot read {batch}: {e}"))?;
    let mut positions: Vec<Position> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fen = line.split_once('|').map(|(_, f)| f.trim()).unwrap_or(line);
        positions.push(parse_fen(fen).map_err(|e| format!("nnue-probe-microbench: {e}: '{fen}'"))?);
    }
    if positions.is_empty() {
        return Err("nnue-probe-microbench: batch contains no positions".to_string());
    }
    let n_positions = positions.len() as u64;
    let visits = iterations * n_positions;
    let rounds = 5;
    let mut features_ns: Vec<f64> = Vec::new();
    let mut nnue_ns: Vec<f64> = Vec::new();
    let mut classical_ns: Vec<f64> = Vec::new();
    let mut checksum = 0u64;
    for _ in 0..rounds {
        // 1. Feature extraction (both perspectives).
        let mut featsum = 0usize;
        let start = Instant::now();
        for _ in 0..iterations {
            for pos in &positions {
                let white = crate::engine::nnue::active_features(
                    pos,
                    crate::engine::nnue::NnuePerspective::White,
                );
                let black = crate::engine::nnue::active_features(
                    pos,
                    crate::engine::nnue::NnuePerspective::Black,
                );
                featsum = featsum.wrapping_add(white.len() + black.len());
            }
        }
        let f_ns = start.elapsed().as_nanos() as f64 / visits as f64;
        std::hint::black_box(featsum);

        // 2. Full full-refresh NNUE inference.
        let mut pred_bits = 0u64;
        let start = Instant::now();
        for _ in 0..iterations {
            for pos in &positions {
                pred_bits = pred_bits.wrapping_add(model.evaluate_scaled(pos).to_bits() as u64);
            }
        }
        let n_ns = start.elapsed().as_nanos() as f64 / visits as f64;
        std::hint::black_box(pred_bits);

        // 3. Classical evaluation.
        let mut eval_acc = 0i64;
        let start = Instant::now();
        for _ in 0..iterations {
            for pos in &positions {
                eval_acc = eval_acc.wrapping_add(i64::from(crate::engine::eval::evaluate(pos)));
            }
        }
        let c_ns = start.elapsed().as_nanos() as f64 / visits as f64;
        std::hint::black_box(eval_acc);

        features_ns.push(f_ns);
        nnue_ns.push(n_ns);
        classical_ns.push(c_ns);
        checksum = checksum
            .wrapping_add(featsum as u64)
            .wrapping_add(pred_bits)
            .wrapping_add(eval_acc as u64);
        println!(
            "nnue_probe_microbench_round {{\"round\":{},\"features_ns_per_call\":{:.2},\"nnue_ns_per_call\":{:.2},\"classical_ns_per_call\":{:.2}}}",
            features_ns.len(),
            f_ns,
            n_ns,
            c_ns
        );
    }
    let sorted = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let f_med = sorted(&mut features_ns);
    let n_med = sorted(&mut nnue_ns);
    let c_med = sorted(&mut classical_ns);
    println!(
        "nnue_probe_microbench_summary {{\"rounds\":{},\"positions\":{},\"iterations\":{},\"features_ns_per_call_median\":{:.2},\"nnue_ns_per_call_median\":{:.2},\"classical_ns_per_call_median\":{:.2},\"nnue_over_classical_ratio\":{:.3},\"features_share_of_nnue\":{:.3},\"checksum\":{}}}",
        rounds,
        n_positions,
        iterations,
        f_med,
        n_med,
        c_med,
        n_med / c_med,
        f_med / n_med,
        checksum
    );
    Ok(())
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

        let b = parse_args(&[
            "profile".to_string(),
            "--diag".to_string(),
            "s75b-probe".to_string(),
        ])
        .unwrap();
        assert!(b.diag_s75b_probe);
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
            (
                "current-final-bounded-check2",
                SearchProfile::CurrentFinalBoundedCheck2,
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
        assert_eq!(
            profile_str(SearchProfile::CurrentFinalBoundedCheck2),
            "current-final-bounded-check2"
        );
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
                s75b_check2_child_seen: 9,
                s75b_probe_calls: 10,
                s75b_probe_legality_tests: 11,
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
        assert!(line.contains("s75b_check2_child_seen=9"));
        assert!(line.contains("s75b_probe_calls=10"));
        assert!(line.contains("s75b_probe_legality_tests=11"));
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
            diag_s75b_probe: false,
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
            diag_s75b_probe: false,
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
            diag_s75b_probe: false,
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
            diag_s75b_probe: false,
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
                diag_s75b_probe: false,
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
                diag_s75b_probe: false,
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
            diag_s75b_probe: false,
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

    // ---- S6-N1 NNUE sparse exporter tests ----

    fn json_array(values: &[u16]) -> String {
        let mut out = String::from("[");
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&v.to_string());
        }
        out.push(']');
        out
    }

    #[test]
    fn nnue_single_export_matches_active_features_exact() {
        use crate::engine::nnue::{
            active_features_v1, active_features_v2, NnueFeatureSet, NnuePerspective,
        };
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1",
        ];
        for fen in fens {
            let pos = parse_fen(fen).unwrap();
            // Test V1 (legacy default)
            let line_v1 = nnue_features_line(&pos, fen, None, NnueFeatureSet::V1);
            let white_v1 = active_features_v1(&pos, NnuePerspective::White);
            let black_v1 = active_features_v1(&pos, NnuePerspective::Black);
            let expected_v1 = format!(
                "{{\"fen\":\"{fen}\",\"white\":{},\"black\":{}}}",
                json_array(&white_v1),
                json_array(&black_v1)
            );
            assert_eq!(
                line_v1, expected_v1,
                "V1 export must equal direct active_features for {fen}"
            );

            // Test V2
            let line_v2 = nnue_features_line(&pos, fen, None, NnueFeatureSet::V2);
            let white_v2 = active_features_v2(&pos, NnuePerspective::White);
            let black_v2 = active_features_v2(&pos, NnuePerspective::Black);
            let expected_v2 = format!(
                "{{\"fen\":\"{fen}\",\"white\":{},\"black\":{}}}",
                json_array(&white_v2),
                json_array(&black_v2)
            );
            assert_eq!(
                line_v2, expected_v2,
                "V2 export must equal direct active_features_v2 for {fen}"
            );
        }
    }

    #[test]
    fn nnue_batch_export_is_deterministic_and_roundtrips_position_id() {
        use crate::engine::nnue::NnueFeatureSet;
        let text = concat!(
            "# comment line\n",
            "startpos_id|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n",
        );
        let first = nnue_features_batch_from_text(text, NnueFeatureSet::V1).unwrap();
        let second = nnue_features_batch_from_text(text, NnueFeatureSet::V1).unwrap();
        assert_eq!(first, second, "batch export must be deterministic");

        let lines: Vec<&str> = first.lines().collect();
        assert_eq!(lines.len(), 2, "comment lines are skipped");
        assert!(
            lines[0].contains("\"position_id\":\"startpos_id\""),
            "position_id must round-trip"
        );
        assert!(
            !lines[1].contains("position_id"),
            "plain FEN lines must not emit position_id"
        );
        for line in &lines {
            assert!(line.contains("\"white\":["));
            assert!(line.contains("\"black\":["));
        }

        let first_v2 = nnue_features_batch_from_text(text, NnueFeatureSet::V2).unwrap();
        let second_v2 = nnue_features_batch_from_text(text, NnueFeatureSet::V2).unwrap();
        assert_eq!(first_v2, second_v2, "V2 batch export must be deterministic");
    }

    #[test]
    fn nnue_export_rejects_malformed_fen() {
        use crate::engine::nnue::NnueFeatureSet;
        let err = nnue_features_for_fen("this is not a fen", NnueFeatureSet::V1).unwrap_err();
        assert!(err.contains("nnue-features"), "single export error: {err}");
        let err = nnue_features_batch_from_text("bad|this is not a fen\n", NnueFeatureSet::V1)
            .unwrap_err();
        assert!(
            err.contains("nnue-features-batch"),
            "batch export error: {err}"
        );
    }

    /// Minimal VALID probe artifact: all-zero weights, head_bias = 1.5.
    fn synthetic_probe_artifact_bytes() -> Vec<u8> {
        use crate::engine::nnue::NNUE_INPUTS;
        use crate::engine::nnue_probe::{
            NNUE_PROBE_MAGIC, NNUE_PROBE_TARGET_SCALE, NNUE_PROBE_VERSION, NNUE_PROBE_WIDTH,
        };
        let mut out = Vec::new();
        out.extend_from_slice(&NNUE_PROBE_MAGIC);
        out.extend_from_slice(&NNUE_PROBE_VERSION.to_le_bytes());
        out.extend_from_slice(&(NNUE_INPUTS as u32).to_le_bytes());
        out.extend_from_slice(&(NNUE_PROBE_WIDTH as u32).to_le_bytes());
        out.extend_from_slice(&NNUE_PROBE_TARGET_SCALE.to_le_bytes());
        out.extend_from_slice(&[0u8; 32]);
        for _ in 0..NNUE_INPUTS * NNUE_PROBE_WIDTH {
            out.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH {
            out.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for _ in 0..NNUE_PROBE_WIDTH * 2 {
            out.extend_from_slice(&0.0f32.to_le_bytes());
        }
        out.extend_from_slice(&1.5f32.to_le_bytes());
        out
    }

    #[test]
    fn nnue_probe_batch_is_deterministic_and_roundtrips_position_id() {
        use crate::engine::nnue_probe::NnueProbeModelV1;
        let model = NnueProbeModelV1::from_bytes(&synthetic_probe_artifact_bytes()).unwrap();
        let text = concat!(
            "# comment\n",
            "startpos_id|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n",
            "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1\n",
        );
        let first = nnue_probe_batch_from_text(&model, text).unwrap();
        let second = nnue_probe_batch_from_text(&model, text).unwrap();
        assert_eq!(first, second, "batch probe must be deterministic");
        let lines: Vec<&str> = first.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("\"position_id\":\"startpos_id\""),
            "position_id must round-trip: {}",
            lines[0]
        );
        assert!(!lines[1].contains("position_id"));
        for line in &lines {
            // head_bias 1.5 -> scaled 1.5, cp 1500.0 (shortest round-trip).
            assert!(
                line.contains("\"scaled_prediction\":1.5"),
                "scaled prediction: {line}"
            );
            assert!(
                line.contains("\"prediction_cp\":1500"),
                "cp prediction: {line}"
            );
        }
    }

    #[test]
    fn nnue_probe_rejects_invalid_model_and_fen() {
        use crate::engine::nnue_probe::NnueProbeModelV1;
        let mut bad = synthetic_probe_artifact_bytes();
        bad[0] = b'X';
        assert!(NnueProbeModelV1::from_bytes(&bad).is_err());
        let truncated = &synthetic_probe_artifact_bytes()[..100];
        assert!(NnueProbeModelV1::from_bytes(truncated).is_err());
        let model = NnueProbeModelV1::from_bytes(&synthetic_probe_artifact_bytes()).unwrap();
        let err = nnue_probe_batch_from_text(&model, "bad|not a fen\n").unwrap_err();
        assert!(err.contains("nnue-probe-batch"), "batch error: {err}");
        let err = nnue_probe_batch_from_text(&model, "bogus fen here\n").unwrap_err();
        assert!(err.contains("nnue-probe-batch"), "batch error: {err}");
    }

    #[test]
    fn nnue_export_escapes_quotes_backslashes_and_control_chars() {
        use crate::engine::nnue::NnueFeatureSet;
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let pos = parse_fen(fen).unwrap();
        // Backslash, quote, tab, and U+001F in the position_id.
        let line = nnue_features_line(&pos, fen, Some("id\\x\"y\tz\u{1f}"), NnueFeatureSet::V1);
        assert!(
            line.contains("\"position_id\":\"id\\\\x\\\"y\\tz\\u001f\""),
            "escaped id: {line}"
        );
        // No raw control characters anywhere in the export.
        for c in line.chars() {
            assert!((c as u32) >= 0x20, "raw control char in export: {:?}", c);
        }
        // No unescaped backslash or quote: every occurrence is part of an
        // escape sequence (`\\`, `\"`).
        let mut chars = line.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                let next = chars.next().expect("escape has a continuation");
                assert!(
                    matches!(next, '\\' | '"' | 'b' | 'f' | 'n' | 'r' | 't' | 'u'),
                    "invalid escape \\{next}"
                );
            }
        }
    }

    #[test]
    fn nnue_export_accepts_tab_separated_fen_and_escapes_it() {
        use crate::engine::nnue::NnueFeatureSet;
        // parse_fen uses split_whitespace, so tab-separated fields parse.
        let tab_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR\tw\tKQkq\t-\t0\t1";
        let pos = parse_fen(tab_fen).unwrap();
        let line = nnue_features_line(&pos, tab_fen, None, NnueFeatureSet::V1);
        assert!(line.contains("\\t"), "tab must be escaped: {line}");
        assert!(!line.contains('\t'), "raw tab in export: {line}");
        // The emitted fen field is exactly json_escape(tab_fen): a JSON parser
        // decodes it back to the original tab-separated FEN.
        let start = line.find("\"fen\":\"").expect("fen field") + 7;
        let end = line[start..].find("\",\"white\"").expect("white field") + start;
        assert_eq!(&line[start..end], json_escape(tab_fen));
    }

    #[test]
    fn nnue_export_normal_input_bytes_are_stable() {
        use crate::engine::nnue::NnueFeatureSet;
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let pos = parse_fen(fen).unwrap();
        let line = nnue_features_line(&pos, fen, Some("startpos_id"), NnueFeatureSet::V1);
        // Exact literal from the pre-escape-helper bridge (04396a0 output):
        // identity escaping keeps normal bytes unchanged.
        assert!(line.starts_with(
            "{\"position_id\":\"startpos_id\",\"fen\":\"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\",\"white\":["
        ));
        assert!(line.ends_with("]}"));
        assert!(
            !line.contains("\\u00"),
            "no control escapes for normal input"
        );
        assert!(
            !line.contains("\\\\"),
            "no backslash escapes for normal input"
        );
    }

    #[test]
    fn eval_breakdown_base_eval_stm_matches_evaluate_exactly() {
        use crate::engine::eval::evaluate;
        let fens = [
            // KQK
            "7k/8/8/8/8/8/3QK3/8 w - - 0 1",
            // KRK
            "7k/8/8/8/8/8/3RK3/8 w - - 0 1",
            // Ordinary fixture
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ];
        for fen in fens {
            let pos = parse_fen(fen).unwrap();
            let line = eval_breakdown_line(&pos, fen);
            let expected = format!("base_eval_stm={}", evaluate(&pos));
            assert!(
                line.contains(&expected),
                "line must contain {expected}: {line}"
            );
            // The value is signed: never appears as a bare substring of another field.
            assert!(line.contains(" base_eval_stm="));
        }
    }
}
