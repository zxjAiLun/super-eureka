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
concurrency    1 by empirical host probe; CLI override remains explicit
seed            2026073001 (fresh relative to the cancelled 20260728 run)
SPRT            [0, 5], alpha=beta=0.05, logistic model
```

The canary used a fresh seed `2026073002`, `decision-mode=fixed`, eight
opening pairs, and explicit `--concurrency 1`. It did not run SPRT.

## Empirical concurrency probe

The probe used the release `Current` binary, cold TT per search, the
open-tactical and high-branch fixtures, and one worker process per requested
concurrency point. The reduced matrix covered all requested points at
100,000 nodes, one measured repeat, and no warmup:

| Workers | Median worker NPS | Relative to 1 | Eligible |
| ---: | ---: | ---: | :---: |
| 1 | 115,938 | 1.000 | yes |
| 2 | 86,910 | 0.750 | no |
| 4 | 63,116 | 0.544 | no |
| 8 | 33,332 | 0.288 | no |
| 12 | 23,316 | 0.201 | no |
| 13 | 24,249 | 0.209 | no |

A selected full confirmation at 1,000,000 nodes, three measured repeats and
one warmup covered points 1 and 2:

| Workers | Median worker NPS | Relative to 1 | p95/median duration | Eligible |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 21,030.8 | 1.000 | 1.0000 | yes |
| 2 | 14,453.9 | 0.687 | 1.0007 | no |

Both full-confirmation points had zero worker failures. The probe therefore
recommends `concurrency=1`. This is a host tournament-throughput result, not
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
| latency p50 / p95 | 1,276 / 3,014 ms | 1,198 / 2,855.75 ms |
| moves below 1,000 ms | 0 | 0 |

The analyzer reported 183 candidate records, one shallow/horizon flag, six
mate transitions, 170 passed-pawn contexts, and 161 promotion-race flags.
Its `long_think=183` count means move time was at least the 0.35-second
diagnostic threshold; it does not mean the engine was in time trouble. No
move crossed the 1,000 ms remaining-clock pressure threshold.

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
Canary: protocol and telemetry PASS
Aspiration promotion: not decided
Formal 2,000-game SPRT: not started
```

The next action requires an independent review of this two-commit change. A
formal candidate-first SPRT may only be started after that review; this canary
does not itself authorize changing `Current`.
