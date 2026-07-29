# SEARCH — aspiration and selective-pruning candidates

Status: **INCONCLUSIVE — local correctness/profile gates pass; Elo gate unavailable**  
Date: 2026-07-26  
Build: release, current local EVAL 1B tree

This record covers the next independent depth 7–8 candidates after EVAL 1A/B,
profiling, and SEE/qsearch shrinking. The bench runner now exposes `pvs`,
`see`, `aspiration`, `lmr`, `null`, and `futility` profiles so each row can be
measured from the same PVS baseline; `current` remains a combined diagnostic
stack only.

- aspiration windows on later Current-profile iterations;
- conservative LMR for quiet later moves;
- a null-move probe only at Current non-PV null-window nodes;
- shallow quiet futility filtering.

The reference profile remains unchanged. No bitboards, incremental evaluation,
continuation history, or multi-threading is included.

## Safety contract

The candidates are deliberately guarded as follows:

- LMR is disabled for the reference profiles, PV moves, in-check nodes,
  shallow nodes, tactical moves, checking moves, and low-material positions.
- Null move is disabled for PV/full-window nodes, in-check nodes, shallow or
  low-material positions, and mate-near windows. A null fail-high is never
  accepted directly: the real position is fully re-searched before returning.
- Futility applies only to later quiet moves in shallow Current null-window
  nodes with enough non-pawn material. The first move, captures, promotions,
  en-passant moves, and checking moves are retained.
- Aspiration retries discard failed-window results and store an exact root TT
  result only after the score is strictly inside the window or a full-window
  retry completes.

## Local correctness evidence

The following passed after the candidates were added:

- full Rust debug and release test suites;
- `cargo fmt --all -- --check`;
- `cargo clippy --all-targets -- -D warnings`;
- `perft(5) = 4,865,609`;
- current-EVAL reference-search smoke locks: startpos `1149` nodes and
  queen-win `969` nodes;
- Current fixed-depth standard fixtures retained the M4.1 reference scores in
  the local comparison at their configured depths;
- null-position test preserves the board, flips side, clears en-passant
  context, and recomputes the expected Zobrist key;
- pruning guard tests reject PV, in-check, and low-material reductions.

These are not proof of game-theoretic correctness or Elo strength. They are
regression evidence for the guarded candidate implementation.

## Fixed 100,000-node Current profile

Command:

```text
cargo run --release -- bench profile --profile current --nodes 100000
```

Selected counters from the latest run:

| Fixture | Completed depth | qsearch nodes | aspiration retries | LMR reductions | LMR re-searches | null attempts | null re-searches | futility pruned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| startpos | 6 | 78,351 | 3 | 596 | 8 | 25 | 12 | 6,251 |
| queen-win | 6 | 88,480 | 2 | 0 | 0 | 0 | 0 | 0 |
| open-tactical | 5 | 86,012 | 0 | 402 | 9 | 12 | 2 | 48,246 |
| closed-quiet | 5 | 80,528 | 3 | 497 | 6 | 18 | 8 | 9,164 |
| exposed-king | 5 | 90,223 | 0 | 73 | 1 | 0 | 0 | 26,869 |
| high-branch | 6 | 78,174 | 0 | 1,617 | 12 | 45 | 5 | 74,328 |
| rook-pawn | 6 | 83,114 | 0 | 0 | 0 | 0 | 0 | 0 |
| kqk | 6 | 72,360 | 0 | 0 | 0 | 0 | 0 | 0 |
| krk | 6 | 83,973 | 0 | 0 | 0 | 0 | 0 | 0 |

The ten-fixture median was about `719 ms / 139k nodes/s` in this run. The
fixed-node result is machine-dependent and is not a rating measurement.
The lower qsearch/tree counts on several Current fixtures are useful
diagnostics, but the extra static checks and selective decisions still need
independent strength validation.

## Independent candidate profiles

The runner also measures each feature separately from the `pvs` baseline:

```text
bench profile --profile pvs|see|aspiration|lmr|null|futility \
              --nodes 100000 --repeat 1 --fixture open-tactical
```

Latest local release measurements for the open-tactical fixture were:

| Profile | Depth | qsearch nodes | eval calls | SEE calls | SEE pruned | aspiration retries | LMR reductions | null attempts | futility pruned | NPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pvs | 4 | 93,752 | 83,022 | 0 | 0 | 0 | 0 | 0 | 0 | 68,724 |
| see | 5 | 91,638 | 81,296 | 38,363 | 13,906 | 0 | 0 | 0 | 0 | 63,024 |
| aspiration | 4 | 93,585 | 83,183 | 0 | 0 | 0 | 0 | 0 | 0 | 67,920 |
| lmr | 4 | 93,926 | 83,144 | 0 | 0 | 0 | 27 | 0 | 0 | 67,552 |
| null | 4 | 93,752 | 83,022 | 0 | 0 | 0 | 0 | 0 | 0 | 67,471 |
| futility | 5 | 91,316 | 85,863 | 0 | 0 | 0 | 0 | 0 | 39,204 | 71,208 |

These rows establish profile isolation and provide profiling evidence only.
They are not Elo results, and the zero retry/attempt rows mean that the
selected fixture did not exercise those guarded branches. `Current` remains a
combined diagnostic stack; it must not be treated as an accepted replacement
for any single candidate.

## Decision and next boundary

`INCONCLUSIVE` is the only justified status:

- local correctness gates: **PASS**;
- fixed-depth/profile diagnostics: **MEASURED**, not an acceptance result;
- Elo/SPRT: **not accepted**, because the current E1 runner is not yet a
  trusted statistical gate.

The candidate stack must not be described as a 2500+ engine or as a proven Elo
gain. Before accepting it, run a trusted external or formally validated
GSPRT match against the immediately preceding profile, then review every
selective feature independently. Continuation history/countermove and deeper
pruning remain outside this candidate.
