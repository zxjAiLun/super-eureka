# S2 formal-evaluation activation and 2+1 canary

Status: preflight canary completed. This document does not promote
Aspiration, modify `Current`, or authorize the bounded SPRT profile.

## Fixed baseline and active profile

The engine baseline is the approved `f4cbbd1605e6b78075fc2220e78845595d107095`
implementation. The same release binary was started twice with different
immutable startup arguments:

```text
baseline: target/release/chess-engine-demo.exe --profile current
candidate: target/release/chess-engine-demo.exe --profile current-aspiration
```

The active repository profile is `s2-current-vs-aspiration-2m1s`:

```text
baseline        Current / current
candidate       Aspiration / current-aspiration
opening book    stockfish-8moves-v3 (8moves_v3.pgn)
opening order   random, 16 plies
time control    2:00+1
Hash            64 MB
engine threads  1
profile rounds  1000 (2000 games maximum for a future SPRT)
concurrency    1 committed default; corrected empirical selection = 12 (CLI override only)
seed            2026073001 (fresh relative to the cancelled 20260728 run)
SPRT            [0, 5], alpha=beta=0.05, logistic model
```

The canary used a fresh seed `2026073002`, `decision-mode=fixed`, eight
opening pairs, and explicit `--concurrency 1`. It did not run SPRT. A later
corrected empirical probe may select a higher explicit concurrency for a new
canary; this committed profile remains at `1`.

## Empirical concurrency probe

The probe used the release `Current` binary, cold TT per search, the
open-tactical and high-branch fixtures, and one worker process per requested
concurrency point. The corrected matrix covered all requested points at
100,000 nodes, three measured repeats, and one warmup. For every successful
search, the requested node target was counted as work; the final complete
`info nodes` values remain diagnostics only:

The machine-readable artifact is `results/concurrency-probe-corrected-steady.json`.

| Workers | Median worker NPS | Relative worker speed | Aggregate NPS | Aggregate ratio | p95/median duration | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 55,871 | 1.000 | 55,871 | 1.00 | 1.00 | yes |
| 2 | 57,221 | 1.020 | 113,436 | 2.03 | 1.01 | yes |
| 4 | 40,597 | 0.730 | 159,259 | 2.85 | 1.02 | yes |
| 8 | 48,386 | 0.870 | 367,234 | 6.57 | 1.05 | yes |
| 12 | 39,623 | 0.710 | 456,081 | 8.16 | 1.02 | yes |
| 13 | 27,953 | 0.500 | 345,497 | 6.18 | 1.05 | yes |

All points completed without worker failure. The selector uses a minimum
relative worker speed of `0.50`, a p95/median duration ceiling of `1.35`, and
then chooses the eligible point with the highest aggregate throughput ratio.
It therefore selects `concurrency=12`; the repository profile remains at `1`
and the value is used only as an explicit Fastchess CLI override. This is not
an engine multithreading implementation and not an Elo result.

## Candidate-first fixed canary

Fastchess was invoked with candidate as player 1 and baseline as player 2:

```text
P1 = Aspiration
P2 = Current
tc = 2:00+1
rounds = 8
games = 16
concurrency = 1
```

The final artifact reports:

| Integrity field | Result |
| --- | --- |
| execution status | `COMPLETED` |
| decision | `NOT_APPLICABLE` (fixed canary; no SPRT) |
| games | 16/16 |
| pairs | 8/8 opening pairs, two games each |
| stopped early | false |
| return code | 0 |
| Fastchess stderr | empty |
| PGN parse errors | 0 |
| white colors | Aspiration 8, Current 8 |
| black colors | Aspiration 8, Current 8 |
| results | 5 decisive wins for White, 4 decisive wins for Black, 7 draws |
| candidate score | 7.5/16 = 46.875% |

The candidate-perspective game totals are four wins, five losses and seven
draws. This is a protocol canary only; the small fixed sample supplies no
promotion decision and no Elo claim.

The pre-game identity probes and every final manifest entry agree:

```text
baseline reported profile  = current
candidate reported profile = current-aspiration
UCI name                   = ChessEngineDemo
UCI author                 = Rust-learner
```

Both roles used the same release file and the same SHA-256
`3332837180c57054ae7ed6b611cd94340207254aa932ea95865e0c8f0bad5add`; the
startup argv is the intentional difference. Fastchess was the pinned
`1.8.2-alpha` binary with SHA-256
`c6d7e4458a58025983c81bf38c524cb213e6f27559657dcfd72543392a431a71`.

## Clock and search telemetry

All 1,823 search-info moves had clock telemetry. Values below use the
corrected explicit-unit parser (`tl` and `latency` are seconds in PGN and are
stored as milliseconds):

| Metric | Aspiration | Current |
| --- | ---: | ---: |
| search-info moves | 911 | 912 |
| depth p10 / p50 / p90 | 5 / 6 / 7 | 5 / 6 / 7 |
| think-time p50 | 2.327 s | 2.3255 s |
| nodes p50 | 65,134 | 67,423.5 |
| NPS p50 | 65,619 | 65,728.5 |
| minimum time left | 17,612 ms | 17,529 ms |
| median time left | 54,983 ms | 54,814.5 ms |
| `fastchess_latency_delta_ms` p50 / p95 | 1,276 / 3,014 ms | 1,198 / 2,855.75 ms |
| moves below 1,000 ms | 0 | 0 |

The analyzer reported 183 candidate records, one shallow/horizon flag, six
mate transitions, 170 passed-pawn contexts, and 161 promotion-race flags.
Its `long_think=183` count means move time was at least the 0.35-second
diagnostic threshold; it does not mean the engine was in time trouble. No
move crossed the 1,000 ms remaining-clock pressure threshold.

## Intermediate concurrency-8 canary

An earlier steady-state probe selected `concurrency=8`, so a fresh explicit
candidate-first fixed canary was run with seed `2026073003`. It used the same
release binary and profile pair, `2:00+1`, Hash 64 MB, and eight concurrent
Fastchess games. It completed 16/16 with zero protocol errors, zero PGN parse
errors, and candidate score 56.25%. A repeated steady-state matrix later
selected `concurrency=12`; this remains valid load evidence but is superseded
as the final selected-concurrency canary below.

| Integrity field | Result |
| --- | --- |
| execution status | `COMPLETED` |
| decision | `NOT_APPLICABLE` (fixed canary; no SPRT) |
| games | 16/16 |
| stopped early | false |
| return code / stderr | 0 / empty |
| PGN parse errors | 0 |
| Fastchess roles | P1 Aspiration / P2 Current |
| colors | 8 games each color assignment |
| result | Aspiration 5 wins, 3 losses, 8 draws |
| candidate score | 9.0/16 = 56.25% |
| explicit concurrency | 8 |

The pre-game probes reported `current` and `current-aspiration` respectively;
both roles used the same binary SHA-256
`3332837180c57054ae7ed6b611cd94340207254aa932ea95865e0c8f0bad5add`. The
timing summary had depth p10/p50/p90 `5/6/8` for both roles, zero moves below
1,000 ms remaining, and renamed telemetry fields
`fastchess_latency_delta_ms_p50/p95` of `1160/3034.3 ms` for Aspiration and
`1087.5/2929.5 ms` for Current. These values include unfinished-search tail,
scheduling, and protocol overhead; they are not pure IPC latency.

This canary validated the then-selected host load and profile identity only.
It did not promote Aspiration or provide an Elo estimate.

## Final selected-concurrency canary

The repeated steady-state matrix selected `concurrency=12`. A second fresh
candidate-first fixed canary used seed `2026073004`, the same `2:00+1` and
Hash 64 MB settings, and an explicit concurrency of 12:

| Integrity field | Result |
| --- | --- |
| execution status | `COMPLETED` |
| decision | `NOT_APPLICABLE` (fixed canary; no SPRT) |
| games | 16/16 |
| stopped early | false |
| return code / stderr | 0 / empty |
| PGN parse errors | 0 |
| Fastchess roles | P1 Aspiration / P2 Current |
| colors | 8 games each color assignment |
| result | Aspiration 8 wins, 4 losses, 4 draws |
| candidate score | 10.0/16 = 62.50% |
| explicit concurrency | 12 |

The identity probes reported `current` and `current-aspiration`; both roles
used the same release binary and the same SHA-256 as the intermediate canary.
The timing summary had depth p10/p50/p90 `4/5/7` for both roles, zero moves
below 1,000 ms remaining, and
`fastchess_latency_delta_ms` p50/p95 of `1033/2869.1 ms` for Aspiration and
`1081/2830.8 ms` for Current. These fields include unfinished-search tail,
scheduling, and protocol overhead; they are not pure IPC latency.

This final canary validates the selected host load and profile identity only.
It does not promote Aspiration or provide an Elo estimate.

## Books, EPD suites, and Syzygy preparation

The pinned `official-stockfish/books` manifest was verified into ignored local
cache files: `8moves_v3.pgn`, UHO 40-60, closed positions, two endgame EPD
suites, and the stalemate suite. The tournament default remains
`stockfish-8moves-v3`; EPD files are diagnostic suites only.

A complete 3-5-piece Syzygy set was downloaded outside the repository from
the public [Sesse Syzygy 3-4-5 directory](http://tablebases.sesse.net/syzygy/3-4-5/)
to:

```text
E:\AUbuntuProject\data\syzygy-3-4-5
```

The local checker found 290 files: 145 `.rtbw` and 145 `.rtbz`, with piece
counts 10 three-piece, 60 four-piece and 220 five-piece files. Total size is
`983,957,920` bytes. A per-file SHA-256 manifest is stored beside the data.
Syzygy remains disabled, diagnostic-only, and absent from Fastchess tablebase
adjudication.

## Stop boundary

```text
Infrastructure: APPROVED for the next review
Current: unchanged
Canary: protocol, telemetry, and selected-concurrency load PASS
Selected explicit Fastchess concurrency: 12
Aspiration promotion: not decided
Formal 2,000-game candidate-first SPRT: not started
```

The formal candidate-first SPRT remains a separate run and must use a fresh
output directory, fresh seed, and explicit `--concurrency 12`. This canary does
not authorize changing `Current`.
