# S4.4C — Single-Buffer 200-Pair Arena Screen

Fixed 200-pair / 400-game sanity screen. No SPRT (no H0/H1 predeclared).

## Engine artifact

```text
artifact build ID:     20260811-a3784d6-linux-x86_64
candidate source:      a3784d6947da1b4fa0d4c5594c45c875c50583c1
                       (= 2db6f6a S4.4B candidate + two UCI-wiring-only
                       fixes 9f88e67/a3784d6; see provenance.txt)
binary SHA-256:        753962f0f5381c41b36d73d150529eb739a4a90fa3bb41863594ac81d88f4fd0
artifact-record commit: 1680de0 (chore(s44c))
cross-artifact CurrentFinal tree check: 30/30 exact (nodes/score/bestmove/PV
                       at depth 6, vs production 20260811-26604c4)
```

## Arena

```text
Arena commits:         none required (server-side registration only)
tournament ID:         e19dc52a-e0d3-4ce0-97d7-0adf339a60b9 (s44c-single-buffer-200pairs)
opening set:           stockfish-8moves-v3
opening SHA-256:       5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e
opening plies:         16
seed:                  20260812
frozen indices:        200 (frozen_opening_indices.json)
historical exclusion:  513 completed pair openings
                       (S4.3B 50 + S4.3C 200 + S4.3D 263, from pair records)
overlap with historical: 0
```

## Contract

```text
candidate:  s44b-single-buffer   (--profile current-final-single-buffer)
baseline:   s44b-current-final   (--profile current-final)
same binary 20260811-a3784d6-linux-x86_64 on both sides
pairs:      200 / games: 400
TC:         10+0.1 (blitz_10_01)
Hash:       16 MB
Threads:    1
Concurrency:1
color:      strict reversal
Arena Elo:  OFF (experimental A/B only)
```

## Result (candidate perspective)

```text
W / L / D:     165 / 155 / 80
score:         51.25% (205/400)

Ptnml [LL,DL,DD+WL,WD,WW]: [15, 16, 131, 20, 18]

pair outcomes:
  2-0:        18
  1.5-0.5:    20
  1-1:        131   (22 DD + 109 WL)
  0.5-1.5:    16
  0-2:        15

candidate White:  78W / 43D / 79L  = 49.75%
candidate Black:  87W / 37D / 76L  = 52.75%
baseline  White:  76W / 37D / 87L  = 47.25%
baseline  Black:  79W / 43D / 78L  = 50.25%

logistic Elo point estimate: +8.7
(descriptive only; NOT a promotion rule)
```

## Integrity

```text
verified pairs:  200/200
verified games:  400/400
crashes:         0
forfeits:        0
illegal:         0
integrity failures: 0
```

## Interpretation (predeclared labels)

candidate score 51.25% falls in (50.0%, 52.0%):

```text
VERDICT: INCONCLUSIVE_POSITIVE
```

Do not reject. A ~7% throughput optimization with a +8.7 Elo point estimate
(in the expected +5..15 band) cannot be cleanly resolved by a 400-game
screen; return for a separately designed formal SPRT (hypothesis sizing to
be chosen after this screen, not reusing the S4.3D +10/+30 defaults blindly).

Candidate is FROZEN (no source changes, no reserve/bitboard/tactical work,
no opening-list changes, no side tuning on this evidence).
