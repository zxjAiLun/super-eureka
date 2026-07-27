# Aspiration Windows — independent candidate record

Status: `IMPLEMENTED — PENDING independent review`

This milestone starts from the approved SEE baseline `52b891edfda0ece45af3f883e16530edf984146b` and adds only the aspiration-window candidate. It does not enable LMR, null-move pruning, futility pruning, or any evaluation change.

## Scope

The candidate is selected with `bench ... --profile aspiration`. It keeps the existing PVS and move-ordering path, then applies a narrow root window around the previous completed iteration's score from depth 2 onward.

- The first iteration always uses a full window.
- Scores in the mate range use a full window.
- Fail-low and fail-high windows are retried with a doubled radius.
- A failed window is never stored as an exact root TT result.
- The final successful iteration is stored as the exact root result.
- The default `reference` profile is unchanged and remains the comparison baseline.

The implementation records `aspiration_retries`, `aspiration_fail_low`, and `aspiration_fail_high`. A retry is counted exactly once and the two failure counters partition the retries.

## Fixed smoke evidence

The required correctness gates remain unchanged:

```text
perft(5) = 4,865,609
bench smoke reference = startpos 1149 / queen-win 969
```

At a fixed 100,000-node budget on the `startpos` profile fixture, the candidate reported:

```text
profile = aspiration
completed_depth = 6
nodes = 100,000
score = cp 0
bestmove = b1c3
aspiration_retries = 3
aspiration_fail_low = 2
aspiration_fail_high = 1
```

This is a node/depth observation only. It is not an Elo claim and is not sufficient to accept the candidate as a stronger engine.

## Verification

The isolated worktree passed:

```text
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --release
cargo run --release -- perft 5
cargo run --release -- bench smoke
git diff --check
```

The unit coverage includes CLI profile isolation, full-window first iteration, retry partitioning, score parity against the M4.1 full-window profile, and position restoration. Independent Elo/SPRT comparison remains required before this feature can be marked `APPROVED`.
