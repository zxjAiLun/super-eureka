# Futility Pruning — independent candidate record

Status: `PENDING independent review — path-evidence fix-forward in ea6f5f9`

This candidate is a direct child of implementation commit
`34e8d7655a01a6ec660eda136187b8e37c2bbc85`. The parent null-move candidate
was not approved; it remains a rejected candidate pending further review.
This milestone adds only the futility candidate. It does not change
evaluation or enable any later feature.

## Scope and safety guards

The candidate is selected with `bench ... --profile futility`. Futility is computed only at shallow, non-PV, non-check nodes with enough non-pawn material. A quiet move may be skipped only when its static evaluation plus a fixed depth-scaled margin cannot reach alpha.

The following are always retained:

- the first/PV move;
- captures, en passant, promotions, and checking moves;
- mate-range windows and scores;
- advanced pawn pushes on the final three ranks, conservatively covering promotion races.

The default `reference` profile does not use futility pruning.

## Fixed smoke evidence

The required correctness gates remain unchanged:

```text
perft(5) = 4,865,609
bench smoke reference = startpos 1149 / queen-win 969
```

At a fixed 100,000-node budget on the `startpos` profile fixture, the candidate reported:

```text
profile = futility
completed_depth = 6
nodes = 100,000
score = cp 0
bestmove = b1c3
futility_pruned = 6,080
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

Unit coverage includes profile isolation, non-PV and reference guards, tactical/checking protection, mate-range protection, and advanced-pawn promotion-threat protection. Fix-forward commit `ea6f5f9` additionally exercises a real futility-pruning search and root state restoration. Independent review and Elo/SPRT comparison remain required before enabling this candidate in `Current`.
