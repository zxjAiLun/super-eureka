# Late-Move Reductions — independent candidate record

Status: `PENDING independent review — fix-forward evidence in ea6f5f9`

This candidate is a direct child of implementation commit
`2dcf5f054a34fe0241c37e378c7a9ff87fa51d35`. The parent aspiration candidate
was approved only as an isolated candidate, not as an approved `Current`
baseline. This milestone adds only the LMR candidate. It does not enable
null-move pruning, futility pruning, or any evaluation change.

## Scope and safety guards

The candidate is selected with `bench ... --profile lmr`. It keeps the existing PVS and M4.1 move-ordering path, and reduces only later quiet moves when all of the following hold:

- the profile is LMR-enabled;
- the node is not in check;
- the search depth is at least 4;
- the move is not among the first three moves;
- the move is not a capture, en passant, promotion, or checking move;
- the position has at least four non-pawn pieces.

The reduced search is a scout. If it improves alpha, including a fail-high result, the same move is searched again at the full depth before it can update the PV, score, or cutoff state. The first/PV move and all tactical moves retain full depth. The default `reference` profile does not use LMR.

## Fixed smoke evidence

The required correctness gates remain unchanged:

```text
perft(5) = 4,865,609
bench smoke reference = startpos 1149 / queen-win 969
```

At a fixed 100,000-node budget on the `startpos` profile fixture, the candidate reported:

```text
profile = lmr
completed_depth = 6
nodes = 100,000
score = cp 0
bestmove = b1c3
lmr_reductions = 108
lmr_researches = 7
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

Unit coverage includes profile isolation, PV/first-move protection, in-check protection, low-material protection, and full-depth re-search accounting. Fix-forward commit `ea6f5f9` additionally exercises reductions and full-depth re-searches in a real search while checking root state restoration. Independent review and Elo/SPRT comparison remain required before enabling this candidate in `Current`.
