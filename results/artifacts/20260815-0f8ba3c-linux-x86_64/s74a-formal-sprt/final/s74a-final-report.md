# S7.4A — Formal Pentanomial SPRT: FINAL REPORT (S7.4A APPROVED_FOR_PROMOTION)

tournament: `2cf04fe6-2120-45c1-852b-e2462e3f62d9` (s74a-formal-sprt)
terminal: `SPRT_ACCEPT_H1`

## Frozen contract

```text
model:       pentanomial
elo_model:   logistic
H0:          +0.0 Elo
H1:          +8.0 Elo
alpha:       0.05
beta:        0.05
bounds:      [-2.9444389791664403, +2.9444389791664403]
max_pairs:   3000
```

Candidate perspective is frozen as Arena `engine_a` throughout scheduling,
pentanomial aggregation, LLR computation, and W/D/L reporting. Pair game 1
gives `engine_a` White; pair game 2 reverses colors. Start evidence was
persisted before game 1 (`/var/lib/chessarena/s74a_start_evidence.md`); no
pilot data was seeded: Ptnml started [0,0,0,0,0], LLR 0.

## Final result

```text
pairs / games:        541 / 1082
candidate W/D/L:      474 / 239 / 369
candidate score:      (474 + 119.5) / 1082 = 54.8521%
Ptnml [LL,LD,DD+WL,WD,WW]: [54, 85, 196, 114, 92]  (sum 541)
final LLR:            +2.9515668774152664
decision:             ACCEPT_H1
logistic Elo point estimate: +33.82 (descriptive only)
```

## First-crossing audit (replayed pair sequence; no new games)

Arena pair indices are zero-based, so `pair_index 540` is the 541st pair.

```text
pair_index 539 (N-1): Ptnml [54,85,196,114,91]
                       pairs = 540
                       LLR   = 2.897378214998293  < +2.9444389791664403  CONTINUE  ✓
pair_index 540 (N):   outcome category 4 (WW: candidate wins both games)
                       Ptnml [54,85,196,114,92]
                       pairs = 541
                       LLR   = 2.9515668774152664  >= +2.9444389791664403  ACCEPT_H1  ✓ FIRST CROSSING
```

No statistical parameters changed. The first boundary crossing happened on
the first verified complete pair whose LLR reached the upper bound, which is
also the terminal pair. The persisted
`/var/lib/chessarena/runs/2cf04fe6-2120-45c1-852b-e2462e3f62d9/sprt.json`
matches this replay exactly (pairs=541, llr=2.9515668774152664).

## Color split (candidate perspective)

```text
candidate White:  239W / 125D / 177L  = (239 + 62.5) / 541 = 55.7301%
candidate Black:  235W / 114D / 192L  = (235 + 57.0) / 541 = 53.9741%
baseline  White:  192W / 114D / 235L  = 46.0259%
baseline  Black:  177W / 125D / 239L  = 44.2699%
```

## Provenance

```text
candidate: preset s74a-lmr-null-window-20260815-0f8ba3c-linux-x86_64
           (--profile current-final-lmr-null-window)
baseline:  preset s74a-current-final-20260815-0f8ba3c-linux-x86_64
           (--profile current-final)
build:     20260815-0f8ba3c-linux-x86_64
source SHA: 0f8ba3c76fe7515f9f1bb4cc092ff3d5a9ed3416
binary SHA: c6a29d372efc6373406e3aafab309aa0eb0013c881834496cd2c659af5d52c4c
           (same exact binary both sides; verified in every pair command)
opening:   stockfish-8moves-v3
opening SHA: 5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e
seed:      20260813, plies 16, frozen openings 3000
frozen indices SHA256: 85e1dce45404f131620113b652c4646d95dd293da49a13a92926dae75c235b33
historical completed-pair records excluded: 2034
active rapid tournament frozen openings also excluded: 100
unique excluded FEN: 1987, fresh eligible pool: 32713
TC:        blitz_10_01 (10+0.1), Hash 16 MB, Threads 1, Concurrency 1,
           strict reversal, Arena Elo OFF
```

## Integrity

```text
verified pairs:       541 / 541
verified games:       1082 / 1082
pair attempts:        all attempt 1
pair return codes:    all 0
moves_legal:          541 / 541
crashes:              0
forfeits:             0
illegal:              0
verifier failures:    0
integrity failures:   0
```

## Final artifacts

```text
results/artifacts/20260815-0f8ba3c-linux-x86_64/s74a-formal-sprt/final/
  sprt.json              fc8800074b91f3371c2ac4f4a069df14bf2d3a2e16d7b6422f6bffdb7ecdd9ac
  summary.json           6fb635c92205e8ce413af4d55eebd3668f148c19ff6cceb5af0a64d66a0b55bc
  combined.pgn           fa924c2454c539374a022f949c3c5aa265048c51ff34c08fc37670cfa84b1e56
  artifact-manifest.json d2e15ff8207b8e579d9a300a8b7c9aac23456fd497d0421d5849ce4e01497c18
  pair_sequence.json     1d23d4ec2bca379bf321747d49795c0b56d2485cf452851c2579278e6d12c3ad
```

`pair_sequence.json` was regenerated as a pure DB replay of the 541 verified
`COMPLETED` PairJob rows (no games re-run, no Arena production-code change)
and independently re-aggregates to the persisted Ptnml
`[54,85,196,114,92]` and final LLR `2.9515668774152664`.

## Verdict

```text
S7.4A APPROVED_FOR_PROMOTION
```

STOP after this terminal: **no auto-promotion**. `CurrentFinal` is NOT
promoted here. Next step is independent review of the terminal SPRT evidence;
any promotion commit requires explicit authorization and must leave the
promotion commit's scope to the default-profile selector only, preserving the
frozen source/binary lineage at `0f8ba3c`.
