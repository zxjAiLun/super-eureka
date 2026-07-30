# PERF — Search cost profile before Depth 7–8 work

Status: **MEASURED — diagnostic only**
Date: 2026-07-26
Build: release, current local EVAL 1B tree
Command:

```text
cargo run --release -- bench profile --nodes 100000
```

The profile suite uses a fixed 100,000-node budget, the disabled/reference
search profile, and the ten standard positions. It records per-search counters
from the same `SearchContext`, not wall-time-only guesses. The counters are
observational and do not change search ordering or limits.

Diagnostic collection is explicitly enabled only by `bench profile`. Ordinary
UCI searches, smoke/standard benches, throughput benches, and the public search
API leave these counters at zero; the always-on node counter remains part of
the search stop contract. This keeps atomic diagnostic updates out of the
production hot path while preserving the same result and node behavior.

## Counter contract

Each `bench_result suite=profile` line reports:

| Counter | Meaning |
|---|---|
| `total_nodes` | All counted search-node entries, including ordinary search, qsearch, and any partial final iteration |
| `qsearch_nodes` | Nodes entered by quiescence, including depth-0 handoff nodes |
| `completed_iterations` / `completed_depth` | Number of completed iterative-deepening passes and their deepest completed depth |
| `last_completed_iteration_ms` / `last_completed_iteration_nodes` | Wall time and nodes spent by the most recent completed pass |
| `aborted_iteration_depth` / `aborted_iteration_nodes` | Depth and node delta of a pass that unwound before completion; zero when no pass was aborted |
| `eval_calls` | Calls to the static evaluator |
| `legal_move_generations` | Search-side legal move-list generations |
| `pseudo_moves` / `legal_moves` | Totals produced and retained by those generations |
| `make_moves` / `unmake_moves` | Movegen legality probes plus search edges |
| `tt_probes` / `tt_hits` / `tt_cutoffs` / `tt_stores` | Search TT activity |
| `tt_rejected_depth` / `tt_rejected_bound` / `tt_rejected_decode` | Matching TT entries that could not cut off because of insufficient depth, an unsatisfied bound, or score decode failure |
| `qsearch_ratio` | `qsearch_nodes / nodes` for the run |
| `nodes_per_completed_depth` | Total nodes divided by completed depth, or zero when no depth completed |
| `effective_branching_factor` | `nodes ^ (1 / completed_depth)`, a coarse cross-fixture indicator |

Movegen legality checks are included in the make/unmake totals, so the latter
are not merely PV-edge counts. The counters are per run and reset for every
fixture/repeat.

## Fixed-budget observations

All rows below consumed exactly 100,000 nodes. `q/n` is
`qsearch_nodes / nodes`; elapsed time and NPS are machine-dependent.

| Fixture | Completed depth | q/n | eval calls | legal generations | pseudo moves | legal moves |
|---|---:|---:|---:|---:|---:|---:|
| startpos | 5 | 89.0% | 86,794 | 269,933 | 6,957,920 | 6,816,727 |
| queen-win | 6 | 86.5% | 70,245 | 250,093 | 4,860,813 | 3,870,508 |
| open-tactical | 3 | 96.7% | 81,580 | 221,146 | 7,593,634 | 6,587,499 |
| closed-quiet | 4 | 90.3% | 88,790 | 257,657 | 7,574,241 | 7,452,714 |
| exposed-king | 4 | 96.0% | 80,406 | 223,219 | 8,115,505 | 6,927,320 |
| high-branch | 4 | 94.7% | 88,950 | 221,774 | 8,552,367 | 8,128,240 |
| rook-pawn | 6 | 79.2% | 75,125 | 272,544 | 3,434,949 | 3,071,904 |
| kqk | 6 | 72.3% | 60,453 | 261,958 | 3,187,519 | 2,942,381 |
| krk | 6 | 83.0% | 75,569 | 276,460 | 2,084,066 | 1,863,045 |
| halfmove-ctx | 4 | 96.0% | 80,406 | 223,219 | 8,115,505 | 6,927,320 |

The first-order conclusion is limited but clear: qsearch accounts for roughly
72–97% of entered nodes in this fixed-budget sample, and legal move generation
does millions of pseudo/legal checks per 100,000 search nodes. This identifies
qsearch and movegen as profiling targets; it does **not** by itself prove that
any particular pruning rule is safe or improves Elo.

For TT visibility, a separate cold/current startpos run produced:

```text
qsearch_nodes=80914 eval_calls=78579
tt_probes=77027 tt_hits=16887 tt_cutoffs=0 tt_stores=77021
completed_depth=6 nodes=100000
```

The disabled/reference rows intentionally have zero TT hits and isolate the
pre-pruning search cost. The cold/current row is a diagnostic comparison, not
a replacement for the approved M4.2 benchmark lineage.

## D1.1 fixed-depth telemetry

D1.1 adds an observational fixed-depth mode so the cost of completing depth 6,
7, and 8 can be measured directly instead of inferring it from a node cap:

```text
cargo run --release -- bench profile --depth 6 --repeat 1
cargo run --release -- bench profile --depth 7 --repeat 1
cargo run --release -- bench profile --depth 8 --repeat 1
```

Use `--fixture startpos` (or another standard fixture id) when a focused run is
needed. The profile suite still uses the reference search profile by default;
`--profile current` is an explicit diagnostic comparison and does not alter the
UCI `Current` production path. A fixed-depth run must complete the requested
depth or report a bench error rather than being treated as a valid cost point.

The current engine does not compute a separate selective depth (`seldepth`)
metric. D1.1 intentionally does not print a fabricated value; `seldepth` will
be added only when the search tracks the deepest actually entered ply with a
defined contract.

These measurements are for search-cost diagnosis only. They do not authorize
qsearch changes, pruning, a profile promotion, or an Elo/SPRT conclusion.

## D1.2 candidate comparison

The isolated D1.2 candidate is selected only by the bench profile name:

```text
cargo run --release -- bench profile --profile current --depth 6 --repeat 1
cargo run --release -- bench profile --profile current-qsearch-movegen --depth 6 --repeat 1
```

`current-qsearch-movegen` preserves the `Current` search strategy and changes
only qsearch move generation. It is not the default profile and is not promoted
into the UCI `Current` path. Under non-check qsearch nodes it generates legal
captures, en passant, and all promotions directly; under check it uses the
complete legal-evasion path. When no tactical move exists it performs an
early-stop legal-move probe so stalemate remains a zero score.

The D1.2 correctness comparison requires identical completed depth, score,
best move, PV, total nodes, qsearch nodes, evaluator calls, and terminal/draw
behavior. A successful performance result should reduce pseudo-move and
make/unmake work without reducing the searched node set. No SEE pruning, delta
pruning, incremental evaluation, bitboard rewrite, or TT-key change belongs in
this profile.

## Depth 7–8 boundary

No search optimization is accepted by this record. The next independent
milestone is qsearch shrinking with a correctness gate (SEE or an equivalent
safe capture filter), followed by one pruning feature at a time. Each change
must preserve perft, legal PVs, terminal/draw behavior, and the existing
benchmark locks before any Elo/SPRT experiment is interpreted.
