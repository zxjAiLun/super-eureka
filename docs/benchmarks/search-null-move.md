# Null-Move Pruning — independent candidate record

Status: `IMPLEMENTED — PENDING independent review`

This milestone starts from the LMR-approved tip `a8865f12f3fd81b8e06a9de69a0658af24cddd28` and adds only the null-move candidate. It does not enable futility pruning or any evaluation change.

## Scope and safety guards

The candidate is selected with `bench ... --profile null`. A null search is attempted only when:

- the profile is null-move-enabled;
- the node is not in check;
- the depth is at least 5;
- the node has a null-window search (`beta == alpha + 1`);
- the position has at least four non-pawn pieces;
- the score is outside the near-mate lower boundary.

The null position flips the side to move, clears en passant, advances the halfmove/fullmove clocks, and recomputes the Zobrist key. A null fail-high is never accepted directly: the real position is searched again at full depth with null disabled at that node, while descendants retain their normal eligibility. This prevents consecutive null attempts and keeps the board/path state recoverable.

The default `reference` profile does not use null-move pruning.

## Fixed smoke evidence

The required correctness gates remain unchanged:

```text
perft(5) = 4,865,609
bench smoke reference = startpos 1149 / queen-win 969
```

At a fixed 100,000-node budget on the `startpos` profile fixture, the candidate reported:

```text
profile = null
completed_depth = 6
nodes = 100,000
score = cp 0
bestmove = b1c3
null_move_attempts = 20
null_move_cutoffs = 12
null_move_researches = 12
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

Unit coverage includes null-position hash/state reconstruction, no-check and low-material guards, depth/window guards, and verification-search accounting. Dedicated zugzwang fixtures and independent Elo/SPRT comparison remain required before this feature can be marked `APPROVED`.
