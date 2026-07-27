# Cumulative Search Ablation Suite

Status: `INFRASTRUCTURE ONLY — no Elo or production decision`

Parent: `3884d8f` cumulative candidate profiles.

The `ablation` bench suite runs the five cumulative profiles in a fixed
order, using the existing standard fixture set and disabled TT mode:

```text
current
current-aspiration
current-aspiration-lmr
current-aspiration-lmr-futility
current-aspiration-lmr-futility-see
```

It does not enable any profile in UCI `Current`, and it does not include the
null probe. The suite is intended to produce comparable machine-readable
rows for later A/B selection; node or wall-time differences are not Elo
claims.

## Commands

```text
cargo run --release -- bench ablation --nodes 100000
cargo run --release -- bench ablation --depth 6
cargo run --release -- bench ablation --movetime 1000
```

Only one limit selector may be supplied. `--fixture` may restrict the suite
to one of the standard fixture IDs. The suite uses `disabled` TT mode and
enables diagnostic counters for every measured row.

## Machine-readable output

Each `bench_result` row contains:

```text
profile fixture score bestmove pv completed_depth nodes qsearch_nodes
elapsed_ms nps stopped aspiration_retries lmr_reductions lmr_researches
futility_pruned see_calls
```

The complete counter set also includes evaluation, move-generation,
make/unmake, TT, null-probe, and SEE-pruned fields. PV legality, root
Position/Zobrist restoration, fixed-depth completion, and exact node-budget
validation use the existing bench checks. Determinism is checked separately
for each `(profile, fixture, mode)` group.

The suite is a comparison data source only. Formal acceptance remains the
separate fastchess/OpenBench match process described by the S2 plan.
