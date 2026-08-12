# S5.0C — SingleGeneration 200-Pair Same-Tree Sanity Screen

Fixed 200-pair / 400-game sanity screen (no SPRT, per the adjusted
same-tree throughput policy).

## Artifact

```text
build ID:     20260812-710400a-linux-x86_64
source SHA:   710400abda8cc1558d5ad846f93464b1ac48bb7f (S4.4E promotion)
binary SHA:   6053901ecb786ac0ed32d23d32a4f8bffbbf6dbe0bd4d70d4173a3fdf808a310
cross-check:  current-final vs current-final-single-buffer 30/30 exact
              (nodes/score/bestmove/PV at depth 6) - artifact rebuilt from
              the promotion source
```

## Competitors

```text
candidate:  s50b-single-generation  (--profile current-final-single-generation)
            = promoted CurrentFinal (LegalityFast + SingleBuffer)
              + has-any child terminal probe
baseline:   s50b-current-final      (--profile current-final)
same exact binary both sides; Arena Elo OFF; no EngineVersion created
```

## Contract

```text
200 pairs / 400 games | 10+0.1 | Hash 16 MB | Threads 1 | Concurrency 1
strict color reversal | opening stockfish-8moves-v3 (sha 5835239f...)
16 plies | seed 2026081202 | historical exclusions 1647 (S4.3B/C/D +
S4.4C/D completed records; overlap 0, duplicates 0)
```

## Result (candidate perspective)

```text
W / L / D:     169 / 152 / 79
score:         52.125%
Ptnml [LL,DL,DD+WL,WD,WW]: [12, 28, 114, 23, 23]
candidate White:  91W / 34D / 75L  = 54.00%
candidate Black:  78W / 45D / 77L  = 50.25%
baseline  White:  77W / 45D / 78L
baseline  Black:  75W / 34D / 91L
```

## Integrity

```text
verified pairs:  200/200
verified games:  400/400
crashes: 0  forfeits: 0  illegal: 0  integrity failures: 0
```

## Verdict (predeclared)

candidate score 52.125% >= 48.0% with zero integrity failures:

```text
SANITY_PASS
promotion-qualified under the same-tree throughput policy
(NO further formal SPRT)
```

STOP: no auto-promotion of SingleGeneration; promotion decision is a
separately reviewed step (S5.0D).
