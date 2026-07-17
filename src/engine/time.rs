//! Time control: turn UCI time parameters into soft/hard deadlines.
//!
//! Two deadlines with deliberately different semantics:
//!  - `soft_deadline`: checked only *between* fully completed iterations.
//!    When it fires we keep the last completed result and do not start a
//!    deeper iteration — starting one more partial iteration risks blowing
//!    the clock for no guaranteed gain.
//!  - `hard_deadline`: checked at *every* node entry. When it fires the
//!    search unwinds immediately, discarding the partial iteration, so we
//!    never overrun the clock.
//!
//! The first version is intentionally simple: fixed fractions and a safety
//! reserve. Being correct (never overrunning the clock) matters far more
//! here than clever time management; dynamic allocation based on position
//! complexity can come later.

use std::time::{Duration, Instant};

/// A pair of deadlines for one search. `None` means "no deadline" on that
/// axis (e.g. `go infinite` or a depth-only search).
#[derive(Clone, Copy, Debug)]
pub struct TimeBudget {
    pub soft_deadline: Option<Instant>,
    pub hard_deadline: Option<Instant>,
}

/// Parsed UCI time parameters for the side to move. `None` means the field
/// was absent from the `go` command. The caller picks `wtime`/`btime`
/// (and `winc`/`binc`) for the side to move before filling this in, so this
/// struct stays free of `Color` and of UCI parsing details.
#[derive(Clone, Copy, Default, Debug)]
pub struct TimeInput {
    pub movetime: Option<Duration>,
    pub remaining: Option<Duration>,
    pub increment: Option<Duration>,
    pub movestogo: Option<u32>,
}

/// Default move horizon when the GUI gives no `movestogo`.
const DEFAULT_MOVES_LEFT: u32 = 30;

/// Compute soft/hard deadlines from the time input. Returns a budget with
/// both deadlines `None` when no time control was supplied (so the caller
/// treats that as "search until stopped / depth / nodes").
pub fn compute_budget(input: &TimeInput, now: Instant) -> TimeBudget {
    if let Some(movetime) = input.movetime {
        return movetime_budget(movetime, now);
    }
    if let Some(remaining) = input.remaining {
        return clock_budget(
            remaining,
            input.increment.unwrap_or(Duration::ZERO),
            input.movestogo,
            now,
        );
    }
    TimeBudget {
        soft_deadline: None,
        hard_deadline: None,
    }
}

/// `go movetime MS`: soft ~ 90% of MS, hard = MS minus a small reserve so
/// thread exit + output never land on the very last millisecond.
fn movetime_budget(movetime: Duration, now: Instant) -> TimeBudget {
    let reserve = std::cmp::max(Duration::from_millis(1), movetime / 50);
    let hard = now + movetime.saturating_sub(reserve);
    let soft = now + (movetime * 9 / 10);
    TimeBudget {
        soft_deadline: Some(soft),
        hard_deadline: Some(hard),
    }
}

/// Clock mode: spread the usable time over the estimated remaining moves,
/// add half the increment, and keep a reserve so we never lose on time.
fn clock_budget(
    remaining: Duration,
    increment: Duration,
    movestogo: Option<u32>,
    now: Instant,
) -> TimeBudget {
    let reserve = std::cmp::max(Duration::from_millis(10), remaining / 50);
    let usable = match remaining.checked_sub(reserve) {
        Some(u) if !u.is_zero() => u,
        // Almost out of time: stop right away rather than risk flag-fall.
        _ => {
            return TimeBudget {
                soft_deadline: Some(now),
                hard_deadline: Some(now),
            }
        }
    };
    let moves_left = movestogo.unwrap_or(DEFAULT_MOVES_LEFT).max(1);
    let allocation = usable / moves_left + increment / 2;
    if allocation.is_zero() {
        return TimeBudget {
            soft_deadline: Some(now),
            hard_deadline: Some(now),
        };
    }
    let soft = now + (allocation * 9 / 10);
    let hard = now + allocation;
    TimeBudget {
        soft_deadline: Some(soft),
        hard_deadline: Some(hard),
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for deadline arithmetic. They check the *relationships*
    //! (soft < hard, within bounds, reserve applied) rather than exact
    //! millisecond values, so they are stable across host speed.

    use super::*;

    fn within(d: Duration, low_ms: u64, high_ms: u64) -> bool {
        let ms = d.as_millis() as u64;
        ms >= low_ms && ms <= high_ms
    }

    #[test]
    fn no_time_input_yields_no_deadlines() {
        let b = compute_budget(&TimeInput::default(), Instant::now());
        assert!(b.soft_deadline.is_none());
        assert!(b.hard_deadline.is_none());
    }

    #[test]
    fn movetime_sets_soft_and_hard_with_reserve() {
        let now = Instant::now();
        let b = compute_budget(
            &TimeInput {
                movetime: Some(Duration::from_millis(100)),
                ..Default::default()
            },
            now,
        );
        let soft = b.soft_deadline.expect("soft");
        let hard = b.hard_deadline.expect("hard");
        // soft ~ 90ms, hard = 100 - max(1, 2) = 98ms.
        assert!(soft < hard, "soft must be before hard");
        assert!(within(soft - now, 89, 91), "soft ~90ms");
        assert!(
            within(hard - now, 97, 99),
            "hard ~98ms (movetime minus reserve)"
        );
        assert!(
            hard - now <= Duration::from_millis(100),
            "hard must not exceed movetime"
        );
    }

    #[test]
    fn clock_spreads_over_moves_with_increment() {
        let now = Instant::now();
        let b = compute_budget(
            &TimeInput {
                remaining: Some(Duration::from_millis(1000)),
                increment: Some(Duration::from_millis(100)),
                movestogo: Some(10),
                ..Default::default()
            },
            now,
        );
        let soft = b.soft_deadline.expect("soft");
        let hard = b.hard_deadline.expect("hard");
        // reserve = max(10, 20) = 20; usable = 980; base = 98; alloc = 98 + 50 = 148.
        // soft ~ 133ms, hard = 148ms.
        assert!(soft < hard);
        assert!(within(hard - now, 147, 149), "hard ~148ms");
        assert!(within(soft - now, 132, 134), "soft ~133ms");
        assert!(
            hard - now < Duration::from_millis(1000),
            "must not use whole clock"
        );
    }

    #[test]
    fn clock_default_moves_left_is_30() {
        let now = Instant::now();
        let b = compute_budget(
            &TimeInput {
                remaining: Some(Duration::from_millis(1000)),
                increment: Some(Duration::from_millis(0)),
                movestogo: None,
                ..Default::default()
            },
            now,
        );
        // reserve 20, usable 980, moves 30 -> base ~32ms. hard ~32ms.
        let hard = b.hard_deadline.expect("hard");
        assert!(
            within(hard - now, 31, 33),
            "hard ~32ms with default 30 moves"
        );
    }

    #[test]
    fn clock_almost_out_of_time_stops_now() {
        let now = Instant::now();
        let b = compute_budget(
            &TimeInput {
                remaining: Some(Duration::from_millis(5)),
                ..Default::default()
            },
            now,
        );
        // reserve = max(10, 0) = 10 > 5 -> usable underflows -> immediate stop.
        let hard = b.hard_deadline.expect("hard");
        assert!(
            hard <= now,
            "tiny remaining must yield an already-passed hard deadline"
        );
    }
}
