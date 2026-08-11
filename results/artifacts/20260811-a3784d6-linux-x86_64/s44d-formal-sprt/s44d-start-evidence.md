# S4.4D — Formal Pentanomial SPRT for SingleBuffer: START EVIDENCE

Persisted BEFORE any game runs. S4.4C pilot data is NOT seeded into the LLR;
the formal test starts from Ptnml=[0,0,0,0,0], LLR=0, pairs=0.

## Frozen candidate artifact (reused, NOT rebuilt)

```text
build ID:      20260811-a3784d6-linux-x86_64
source SHA:    a3784d6947da1b4fa0d4c5594c45c875c50583c1
binary SHA-256: 753962f0f5381c41b36d73d150529eb739a4a90fa3bb41863594ac81d88f4fd0
lineage:       2db6f6a (S4.4B candidate) + UCI-only wiring 9f88e67/a3784d6
```

## Competitors (same exact binary, both sides)

```text
candidate: preset s44b-single-buffer  ["--profile","current-final-single-buffer"]
baseline:  preset s44b-current-final  ["--profile","current-final"]
both resolve to build 20260811-a3784d6-linux-x86_64 / binary 753962f0...
Arena Elo: OFF. No EngineVersions created.
```

## Statistical contract

```text
model:       pentanomial
elo_model:   logistic
H0:          +0.0 Elo
H1:          +8.0 Elo
alpha:       0.05
beta:        0.05
lower bound: -2.9444389791664403
upper bound: +2.9444389791664403
max_pairs:   3000
decision:    LLR >= upper -> ACCEPT_H1; LLR <= lower -> ACCEPT_H0;
             pairs == 3000 -> MAX_PAIRS
```

## Match contract

```text
TC: 10+0.1 (blitz_10_01) | Hash 16 MB | Threads 1 | Concurrency 1
color: strict reversal (2 games per pair, one opening, reversed colors)
opening plies: 16 | Arena Elo: OFF
```

## Openings

```text
opening set:  stockfish-8moves-v3
opening SHA-256: 5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e
seed:         20260813
frozen count: 3000 (frozen_opening_indices.json)
historical exclusions: 713 (S4.3B 50 + S4.3C 200 + S4.3D 263 + S4.4C 200,
                      derived from actual stored pair records, one-pass FEN
                      computation against the frozen book SHA)
overlap with historical completed openings: 0
duplicates inside frozen list: 0
```

## SPRT implementation

The already-approved S4.3D pentanomial plumbing
(chessarena/services/sprt.py, differential-tested against
fishtest LLRcalc; scheduler _maybe_sprt runs after every VERIFIED pair,
persists sprt.json). No new statistics code.

## Integrity fail-closed

Pair counted only when: same immutable build, exact binary SHA, correct
candidate/baseline profiles, correct opening index + SHA, strict color
reversal, both games complete + verified, legal moves, valid PGN, valid
return codes, no crash, no unexpected forfeit. Any failure: INTEGRITY_FAIL.

## Tournament identity

```text
name:          s44d-formal-sprt
tournament ID: 178d3f86-8877-4215-998c-fc2e897aa254
status:        DRAFT (to be QUEUED after this commit)
```
