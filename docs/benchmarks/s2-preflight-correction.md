# S2 preflight correction

Status: infrastructure prepared; no new long match started.

This correction supersedes the proposed `2,000 games @ 10+0.1` aspiration
SPRT. That run is cancelled. The time control for the next formal candidate
test remains undecided between `1:00+0`, `2:00+1`, `3:00+0`, and `3:00+2`.

The repository profile for that historical run is explicitly marked
`historical-cancelled`. `run_fastchess.py` requires an explicit
`--profile-name`; non-active profiles can be inspected with `--dry-run` but
cannot launch a match. Its committed concurrency is `1`; any future host
parallelism must be an explicit, separately measured invocation.

## Preserved ultrafast dataset

The historical `10+0.1` 400-game result remains at:

```text
results/fastchess-sprt-400-20260729/
```

It is retained as a valid protocol/ultrafast dataset, not as a promotion
decision. The historical run completed 400 games with Current first in the
Fastchess command, so its candidate-oriented SPRT direction must not be
reused. Its observed result was 154 Current wins, 174 losses, and 72 draws;
the candidate-oriented point estimate was approximately `+17.4 Elo`, with a
wide uncertainty interval.

The automatic postmortem was run with:

```text
python tools/analyze_fastchess_pgn.py \
  --pgn results/fastchess-sprt-400-20260729/games.pgn \
  --manifest results/fastchess-sprt-400-20260729/manifest.json \
  --output-dir results/postmortem-400-20260729 \
  --top 50 --min-loss-cp 150 --max-depth 4 --time-threshold-s 0.35
```

The generated output is intentionally ignored by Git and contains
`candidates.jsonl`, `summary.json`, and `report.md`. It was regenerated with
the corrected analyzer semantics.

```text
games                 = 400
moves                 = 60,728
moves with eval       = 54,382
search-info moves     = 54,328
parse errors          = 0
candidate records     = 9,109
horizon/time flags    = 358
shallow candidates    = 3,547
long-think diagnostics = 14
short-think diagnostics = 9,095
time-pressure flags   = 0
time-pressure unknown = 9,109
mate transitions      = 552
passed-pawn context   = 8,939
promotion-race flags  = 8,685
```

The old PGN has no remaining-clock (`timeleft`) field. Therefore its
time-pressure status is unknown: the historical `14` count described moves
whose think time was at least `0.35s`, not moves made with little clock time.
Actual clock-pressure flags require new PGN telemetry (`time_left_ms`, and
when the initial clock is known, `time_left_ratio`). These are diagnostic
candidates, not automatic labels of engine blunders.
For each move, the analyzer compares the current node's pre-move UCI score
with the next node's score after converting it to the mover's point of view.
Mate transitions are explicitly flagged rather than silently presented as
ordinary centipawn losses. The report keeps FEN, PV, depth, think time, nodes,
NPS, hashfull, optional `time_left_ms`, `time_left_ratio`, and
`fastchess_latency_delta_ms` (elapsed wall time minus the engine-reported
search time; not pure IPC latency)
fields, and passed-pawn fields; book-only
comments are treated as lacking search depth/time data.

## Book and endgame-suite preparation

The selected official Stockfish books are pinned to commit
`65815ccdbc7727cd4f6aee252ba8f67fb740e92f` in `books/manifest.json`. The five
EPD suites and the existing `8moves_v3.pgn` were downloaded and verified into
`books/cache/`. The default tournament book remains `8moves_v3.pgn`; adding
the suites does not change a tournament automatically.

Optional Syzygy support is represented by
`tools/syzygy_manifest.json` and `tools/check_syzygy.py`. It is disabled and
has no configured directory in this environment. A complete 5-piece set must
be placed on an explicitly chosen local volume before any future engine or
adjudication integration is considered.

## Timing and host capacity

The Fastchess command now requests:

```text
timeleft=true
latency=true
```

The existing UCI info fields (`score`, `depth`, `seldepth`, `time`, `nodes`,
`nps`, `hashfull`, and `pv`) remain diagnostic data; no Rust search change was
needed. The historical PGN predates the two new Fastchess fields.

`tools/probe_concurrency.py --json` reported:

```text
physical cores       = 14
logical processors   = 20
recommended process concurrency = 13
engine thread model  = single-threaded
```

This is a host-capacity recommendation for future Fastchess sampling, not an
authorization to start a match. The engine's internal thread count remains
one, and `Current` remains unchanged.

## Next gate

Before the next formal comparison, choose the time control, run a short
candidate-first protocol canary with a fresh output directory, and record the
chosen concurrency. No aspiration promotion or search-profile change is
authorized by this preflight work.
