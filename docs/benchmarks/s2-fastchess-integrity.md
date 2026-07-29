# S2 Fastchess integrity closeout

Status: `CANARY PASSED — no Elo or production decision`

This document records the fix-forward boundary for the formal Fastchess
wrapper. It does not report Elo, approve a candidate profile, or change the
`Current` production profile.

## Execution and decision are separate

`tools/run_fastchess.py` writes two independent fields:

```text
execution_status = PREPARED | RUNNING | COMPLETED | INTEGRITY_FAIL | LAUNCH_FAIL | INTERRUPTED
decision         = PASS | REJECTED | INCONCLUSIVE | NOT_APPLICABLE
```

Only a completed Fastchess run with an explicit SPRT boundary may produce
`PASS` or `REJECTED`. A completed fixed-game canary uses
`decision = NOT_APPLICABLE`. A completed SPRT run without a boundary is
`INCONCLUSIVE`. A non-zero Fastchess exit is an execution integrity failure;
it is never reported as a candidate rejection.

## Artifact isolation and provenance

The wrapper refuses an output directory that already contains `manifest.json`
or `games.pgn`. Every run therefore gets a fresh artifact directory, and the
Fastchess command uses:

```text
-pgnout ... append=false
```

The opening-book `format` is taken from the verified book manifest. It is not
inferred from the selected engine profile. The pinned Stockfish PGN manifest
hash matches the archive bytes as downloaded; the book contains 34,700
positions at 16 plies.

Before Fastchess starts, both final engine argv values receive a read-only UCI
identity probe. The manifest records the reported UCI name, author, and search
profile, and the run stops with `INTEGRITY_FAIL` before game 1 if either
reported profile differs from the requested profile.

The manifest records `engine_thread_model = single-threaded`; no unsupported
`Threads` UCI option is sent to the engine. SPRT runs use `rounds_max` and
`games_max` and are initialized with `games_completed = null` and
`stopped_early = null`. After a completed process, the wrapper records the
actual PGN game count and whether the run stopped before `games_max`.

For SPRT direction, Fastchess player 1 is always
`engine_b_candidate` and player 2 is always `engine_a_baseline`. Therefore
Fastchess `H1 accepted` maps to candidate `PASS`, while `H0 accepted` maps to
candidate `REJECTED`.

## Acceptance boundary

The next authorized action is a four-game fixed canary:

```text
Current + Aspiration (P1)  vs  Current (P2)
```

It must complete with `execution_status = COMPLETED`, four parseable PGN
games, strict paired colors, matching expected/reported startup profiles, and
no Fastchess integrity warning, crash, illegal move, or timeout. This canary
is protocol and artifact validation only; it is not an Elo claim.

The historical 2026-07-29 canary completed before this SPRT-direction
fix-forward with:

```text
Fastchess: v1.8.2-alpha
decision_mode: fixed
rounds_max: 2
games_max / games_completed: 4 / 4
execution_status: COMPLETED
decision: NOT_APPLICABLE
stopped_early: false
legacy Fastchess P1/P2 profiles: current / current-aspiration (both UCI-reported)
PGN parse: 4 games
stderr: empty
```

That artifact remains accepted as a protocol and artifact check only. Its
SPRT orientation is historical and must not be reused for a formal candidate
decision. It is not a relative Elo estimate and does not authorize enabling
aspiration in `Current`.
