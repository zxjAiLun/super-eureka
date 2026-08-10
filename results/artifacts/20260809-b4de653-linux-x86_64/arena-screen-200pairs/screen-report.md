# S4.3B Arena screen — 200 pairs (s43c-legality-fast-200pairs)

tournament_id: 543510bd-979c-4341-952a-0dcca7fc06cc
status: COMPLETED (200/200 pairs, 400 games)
time control: 10+0.1 (blitz_10_01), Hash 16MB, 1 thread, strict color reversal

engine build: 20260809-b4de653-linux-x86_64 (source b4de653)
  candidate profile: current-final-legality-fast
  baseline profile:  current-final
(same binary; the only difference is the unpinned non-check legality fast path)

## Result (candidate perspective)

- wins 185 / losses 142 / draws 73  ->  score 55.38% (221.5 / 400)
- as White:  103W / 37D / 60L  (60.75%)
- as Black:   82W / 36D / 82L  (50.0%)

## Verification

- 200/200 pairs verified, return_code 0, moves_legal true.

## Prior screen

- s43b-legality-fast-screen (87e32233): 50 pairs / 100 games -> 62.0% (49W/25L/26D)

## Quick statistical read (not a formal SPRT)

- 400 games at 55.38% vs the 50% null: z ~= 2.15 sigma, one-sided LOS ~= 98%
  (normal approximation). Clearly positive direction, consistent with the
  50-pair screen (62%).

## Reading

Per the S4.3B gate this is meaningfully positive, so the candidate advances
toward a larger paired test / formal SPRT before any promotion. Formal SPRT
and promotion still require the pre-declared Arena authorization contract;
CurrentFinal remains the production default.

Artifacts pulled from the Arena server (hash-matched to artifact-manifest.json):
summary.json, combined.pgn, artifact-manifest.json
