# Null-Move Probe — independent candidate record

Status: `PENDING independent review — semantics fix-forward in 234755c`

This candidate is a direct child of implementation commit
`a8865f12f3fd81b8e06a9de69a0658af24cddd28`. The parent LMR candidate was
not approved; its evidence was incomplete. Fix-forward commit `234755c`
corrects the probe-child semantics. This milestone adds only the null-move
candidate. It does not enable futility pruning or any evaluation change.

## Scope and safety guards

The candidate is selected with `bench ... --profile null`. It is a verified
null-move probe, not an unconditional pruning cutoff. A null search is
attempted only when:

- the profile is null-move-enabled;
- the node is not in check;
- the depth is at least 5;
- the node has a null-window search (`beta == alpha + 1`);
- the position has at least four non-pawn pieces;
- the score is outside the near-mate lower boundary.

The null position flips the side to move, clears en passant, advances the halfmove/fullmove clocks, and recomputes the Zobrist key. A null fail-high is never accepted directly: the real position is searched again at full depth with null disabled at that node. The immediate null-probe child is also entered with null disabled, while ordinary descendants retain their normal eligibility. This prevents consecutive null attempts and keeps the board/path state recoverable.

The default `reference` profile does not use null-move probes.

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
null_move_fail_highs = 12
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

Unit coverage includes null-position hash/state reconstruction, no-check and low-material guards, depth/window guards, explicit non-reentrant probe-child semantics, and verification-search accounting. Dedicated zugzwang fixtures, independent review, and Elo/SPRT comparison remain required before enabling this candidate in `Current`.
