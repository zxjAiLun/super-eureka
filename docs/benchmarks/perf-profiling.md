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

## Counter contract

Each `bench_result suite=profile` line reports:

| Counter | Meaning |
|---|---|
| `qsearch_nodes` | Nodes entered by quiescence, including depth-0 handoff nodes |
| `eval_calls` | Calls to the static evaluator |
| `legal_move_generations` | Search-side legal move-list generations |
| `pseudo_moves` / `legal_moves` | Totals produced and retained by those generations |
| `make_moves` / `unmake_moves` | Movegen legality probes plus search edges |
| `tt_probes` / `tt_hits` / `tt_cutoffs` / `tt_stores` | Search TT activity |

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

## Depth 7–8 boundary

No search optimization is accepted by this record. The next independent
milestone is qsearch shrinking with a correctness gate (SEE or an equivalent
safe capture filter), followed by one pruning feature at a time. Each change
must preserve perft, legal PVs, terminal/draw behavior, and the existing
benchmark locks before any Elo/SPRT experiment is interpreted.
