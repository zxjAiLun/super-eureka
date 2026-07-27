# SEARCH — SEE qsearch shrink candidate

Status: **INCONCLUSIVE — correctness gate passed; performance/Elo gate not accepted**
Date: 2026-07-26
Build: release, current local EVAL 1B tree

The candidate implementation is commit `52b891edfda0ece45af3f883e16530edf984146b`, whose parent profiling candidate `16b6b63` was rejected because profiling was not gated off the production path. Fix-forward commit `3d52517` makes SEE conservative. Neither the original candidate nor its fix-forward is an approved strength baseline. This is an independent diagnostic candidate following the qsearch profile in [`perf-profiling.md`](perf-profiling.md). It adds a correctness-first static exchange evaluation (SEE) over the existing array-board representation and uses it only to order qsearch captures. Every legal capture remains in the qsearch, including exchanges with a negative static score; promotions are retained. When the side to move is in check, qsearch still searches every legal evasion.

SEE ordering is enabled only by `SearchProfile::SeeCandidate`; the M4.0,
M4.1, PVS, and public qsearch reference paths explicitly keep the pre-SEE
move order so their counters remain valid independent baselines. `Current` is
not a SEE profile.

No aspiration window, null move, LMR, futility pruning, or other search change is included. SEE is not a claim of Elo gain, and this record does not promote the candidate to an accepted engine baseline.

## Correctness gate

The candidate passed the following local checks:

- SEE distinguishes a winning queen capture from a defended losing capture.
- SEE does not mutate the input position.
- Existing qsearch horizon and interruption tests pass with ordering-only SEE;
  the defended capture is still searched rather than hard-pruned.
- Full debug and release Rust test suites pass.
- `perft(5) = 4,865,609` is unchanged.
- The independent SEE profile remains legal and deterministic on the smoke
  positions:

```text
startpos depth 3: 632 nodes, bestmove b1c3, score cp 50
queen-win depth 3: 557 nodes, bestmove e4a4, score cp 990
```

The historical pre-EVAL/M4.0 smoke locks (`1149` / `963` nodes) and the
current EVAL 1A reference-search locks (`1149` / `969` nodes) are recorded in
the corresponding baseline documents; they are not rewritten by this
candidate profile.

These are correctness and regression observations only; they are not Elo measurements.

## Fixed-budget profile

Command:

```text
cargo run --release -- bench profile --profile see --nodes 100000
```

The profile is compared with the pre-SEE record in [`perf-profiling.md`](perf-profiling.md). Both use the same fixed node budget and fixture family. The counters are diagnostic and the elapsed time is machine-dependent.

Historical pre-fix fixed-budget rows (retained as superseded diagnostic
evidence):

| Fixture | qsearch nodes | eval calls | SEE calls | SEE pruned | completed depth |
|---|---:|---:|---:|---:|---:|
| startpos | 88,143 | 86,311 | 14,057 | 4,982 | 5 |
| queen-win | 86,487 | 70,178 | 6,512 | 259 | 6 |
| open-tactical | 93,987 | 76,562 | 81,060 | 26,773 | 3 |
| closed-quiet | 88,638 | 87,037 | 32,734 | 11,772 | 4 |
| exposed-king | 93,753 | 75,225 | 71,304 | 24,141 | 4 |
| high-branch | 89,950 | 83,598 | 119,234 | 61,716 | 4 |
| rook-pawn | 79,216 | 75,125 | 107 | 0 | 6 |
| kqk | 72,334 | 60,453 | 0 | 0 | 6 |
| krk | 82,953 | 75,569 | 9 | 0 | 6 |

The table above is retained as historical pre-fix diagnostic evidence only;
its nonzero `SEE pruned` values must not be read as an accepted result. The
fix-forward candidate now records SEE calls for ordering and leaves
`SEE pruned` at zero. Its wall-time remains machine-dependent and the extra
plain-board exchange accounting is not yet a proven strength or throughput
improvement.

## Decision

`INCONCLUSIVE` is the only justified status at this stage:

- correctness and perft gates: **PASS**;
- fixed-budget profiling: **diagnostic only**, with an unfavorable current wall-time trade-off;
- Elo/SPRT: **not run / not interpretable**, because the project’s E1 statistical runner is not yet a trusted acceptance gate.

The plain-board SEE was subsequently optimized to scan only attackers of the
exchange square, removing the temporary full pseudo-move list at every
exchange layer. Aspiration, guarded LMR, verified null probing, and shallow
futility are recorded separately in [`search-aspiration.md`](search-aspiration.md),
[`search-lmr.md`](search-lmr.md), [`search-null-move.md`](search-null-move.md),
and [`search-futility.md`](search-futility.md). None of those features is an
accepted Elo result or enabled in `Current`.
