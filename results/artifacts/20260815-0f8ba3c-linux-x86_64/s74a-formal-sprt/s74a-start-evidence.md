# S7.4A — Formal Pentanomial SPRT: START EVIDENCE

Persisted BEFORE the worker starts the first pair. No pilot data is seeded:
Ptnml starts [0,0,0,0,0], LLR=0, pairs=0.

## Candidate perspective (frozen)

```
Candidate perspective is Arena engine_a throughout scheduling,
pentanomial aggregation, LLR computation, and final W/D/L reporting.
Pair game 1 gives engine_a White; pair game 2 reverses colors.
```

## Artifact

```
build ID:      20260815-0f8ba3c-linux-x86_64
source SHA:    0f8ba3c76fe7515f9f1bb4cc092ff3d5a9ed3416
binary SHA256: c6a29d372efc6373406e3aafab309aa0eb0013c881834496cd2c659af5d52c4c
candidate:     s74a-lmr-null-window-20260815-0f8ba3c-linux-x86_64  (--profile current-final-lmr-null-window)
baseline:      s74a-current-final-20260815-0f8ba3c-linux-x86_64  (--profile current-final)
same binary both sides, SHA re-verified by Arena install/probe path
```

## Statistical contract (S4.3D/S4.4D verbatim)

```
model:       pentanomial
elo_model:   logistic
H0:          +0.0 Elo
H1:          +8.0 Elo
alpha:       0.05
beta:        0.05
bounds:      [-2.9444389791664403, +2.9444389791664403]
max_pairs:   3000
decision:    LLR >= upper -> ACCEPT_H1; LLR <= lower -> ACCEPT_H0;
             pairs == 3000 -> MAX_PAIRS
```

## Match contract

```
TC: 10+0.1 (blitz_10_01) | Hash 16 MB | Threads 1 | Concurrency 1
strict reversal: game 1 candidate(engine_a) White / baseline Black;
                 game 2 baseline White / candidate Black
```

## Openings

```
opening set:  stockfish-8moves-v3
opening SHA:  5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e
plies:        16
seed:         20260813
historical completed-pair + active-frozen exclusion FENs: 1987
fresh eligible pool after exclusion:                       32713
frozen count:                                              3000
frozen indices SHA256:                                     85e1dce45404f131620113b652c4646d95dd293da49a13a92926dae75c235b33
```

Active tournaments whose full frozen opening lists were also excluded
(before S7.4A starts, any of their remaining pairs cannot overlap):
```
[{'id': '34278872-6af1-4f13-885e-e597f4158054', 'name': '0813rapid5+3', 'status': 'RUNNING', 'frozen_indices': 100}]
```

## Integrity smoke (not counted)

Direct cutechess-cli 1+0.05, one opening, same binary/profiles,
strict reversal, concurrency 1. Result 1-1 (one Black-mates each);
no crash, no illegal move, no protocol failure.

```
...      S7.4A-Candidate playing Black: 1 - 0 - 0  [1.000] 1
...      White vs Black: 0 - 2 - 0  [0.000] 2
Elo difference: 0.0 +/- nan, LOS: 50.0 %, DrawRatio: 0.0 %
SPRT: llr 0 (0.0%), lbound -inf, ubound inf

Player: S7.4A-Candidate
   "Loss: Black mates": 1
   "Win: Black mates": 1
Player: S7.4A-Baseline
   "Loss: Black mates": 1
   "Win: Black mates": 1
Finished match
13510 >S7.4A-Baseline(1): quit
13510 >S7.4A-Candidate(0): quit
```

## Tournament identity

```
name:          s74a-formal-sprt
tournament ID: 2cf04fe6-2120-45c1-852b-e2462e3f62d9
status:        QUEUED
```
