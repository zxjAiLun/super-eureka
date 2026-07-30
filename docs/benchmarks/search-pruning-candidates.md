# SEARCH — aspiration and selective-pruning candidates

Status: **INCONCLUSIVE — local correctness/profile gates pass; Elo gate unavailable**  
Date: 2026-07-26  
Build: release, current local EVAL 1B tree

This record covers the next independent depth 7–8 candidates after EVAL 1A/B,
profiling, and SEE/qsearch shrinking. The bench runner now exposes `pvs`,
`see`, `aspiration`, `lmr`, `null`, and `futility` profiles so each row can be
measured from the same PVS baseline. `current` is the approved M4.1 ordering +
PVS production path and keeps every SEARCH 1 candidate disabled; cumulative
`current-aspiration-*` profiles are separate tournament candidates.

- aspiration windows on later candidate-profile iterations;
- conservative LMR for quiet later moves;
- a null-move probe only at candidate non-PV null-window nodes;
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
- Futility applies only to later quiet moves in shallow candidate null-window
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

## Historical pre-fix-forward combined diagnostic table

The table that was formerly labelled `Current` came from an uncommitted
pre-fix-forward worktree in which several candidate switches were combined.
It is retained only as superseded diagnostic evidence and cannot be reproduced
by `--profile current` at this SHA. It is not evidence about the approved
production profile and must not be used for an Elo or promotion decision.

Historical command:

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

The ten-fixture median and counters in that historical table are machine-
dependent and are not rating measurements. To reproduce current-tip behavior,
run the six isolated profiles below and record the resulting `bench_result`
lines instead.

## Independent candidate profiles

The runner also measures each feature separately from the `pvs` baseline:

```text
bench profile --profile pvs|see|aspiration|lmr|null|futility \
              --nodes 100000 --repeat 1 --fixture open-tactical
```

Current-tip release measurements for the open-tactical fixture were:

| Profile | Depth | qsearch nodes | eval calls | SEE calls | SEE pruned | aspiration retries | LMR reductions | null attempts | futility pruned | NPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pvs | 4 | 93,752 | 83,022 | 0 | 0 | 0 | 0 | 0 | 0 | 166,090 |
| see | 4 | 93,439 | 82,799 | 56,479 | 0 | 0 | 0 | 0 | 0 | 150,805 |
| aspiration | 4 | 93,585 | 83,183 | 0 | 0 | 0 | 0 | 0 | 0 | 163,076 |
| lmr | 4 | 93,926 | 83,144 | 0 | 0 | 0 | 26 | 0 | 0 | 162,329 |
| null | 4 | 93,752 | 83,022 | 0 | 0 | 0 | 0 | 0 | 0 | 151,719 |
| futility | 5 | 91,316 | 85,863 | 0 | 0 | 0 | 0 | 0 | 39,202 | 157,217 |

These rows establish profile isolation and provide profiling evidence only.
They are not Elo results. The current tip must be measured with `pvs`,
`aspiration`, `lmr`, `null`, `futility`, and `see` independently; the
historical combined table above must not be described as `Current`.

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
