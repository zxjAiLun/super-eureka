# S4.4D — Formal Pentanomial SPRT: FINAL REPORT (FORMAL_SPRT_PASS)

tournament: `178d3f86-8877-4215-998c-fc2e897aa254` (s44d-formal-sprt)
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

Start evidence committed before any game: `021a7ee docs(s44d)` (frozen 3000
indices, 713 historical exclusions, seed 20260813, zero overlap/duplicates).
S4.4C pilot data was NOT seeded: Ptnml started [0,0,0,0,0], LLR 0.

## Final result

```text
pairs / games:        934 / 1868
candidate W/L/D:      775 / 700 / 393
candidate score:      (775 + 196.5) / 1868 = 52.0075%
Ptnml [LL,DL,DD+WL,WD,WW]: [62, 84, 587, 119, 82]  (sum 934)
final LLR:            +2.953047447812923
decision:             ACCEPT_H1
logistic Elo point estimate: +13.96 (descriptive only)
```

## First-crossing audit (replayed pair sequence; no new games)

```text
pair 932 (N-1): Ptnml [62,84,587,118,82]
                LLR   2.900248074541206  < +2.9444389791664403  CONTINUE  ✓
pair 933 (N):   outcome category 3 (candidate 1.5-0.5: one win, one draw)
                Ptnml [62,84,587,119,82]
                LLR   2.953047447812923  >= +2.9444389791664403  ACCEPT_H1  ✓ FIRST CROSSING
```

No statistical parameters changed; the first boundary crossing happened on the
first verified complete pair where LLR reached the upper bound. The persisted
`/var/lib/chessarena/runs/<tid>/sprt.json` at terminal matches this replay
exactly (pairs=934, llr=2.953047447812923).

## Color split (candidate perspective)

```text
candidate White:  386W / 188D / 360L  = (386+94)/934 = 51.39%
candidate Black:  389W / 205D / 340L  = (389+102.5)/934 = 52.62%
baseline  White:  340W / 205D / 389L
baseline  Black:  360W / 188D / 386L
```

## Provenance

```text
candidate: preset s44b-single-buffer  (--profile current-final-single-buffer)
baseline:  preset s44b-current-final  (--profile current-final)
build:     20260811-a3784d6-linux-x86_64
binary:    753962f0f5381c41b36d73d150529eb739a4a90fa3bb41863594ac81d88f4fd0
           (same exact binary both sides; verified in every pair command)
opening:   stockfish-8moves-v3, sha 5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e
seed:      20260813, plies 16
historical exclusions: 713 (S4.3B 50 + S4.3C 200 + S4.3D 263 + S4.4C 200)
overlap:   0, duplicates: 0
TC:        10+0.1, Hash 16MB, Threads 1, Concurrency 1, strict reversal,
           Arena Elo OFF
```

## Integrity

```text
verified pairs: 934 / 934
verified games: 1868 / 1868
crashes:        0
forfeits:       0
illegal:        0
verifier failures: 0
integrity failures: 0
```

## Verdict

```text
FORMAL_SPRT_PASS
PROMOTION_EVIDENCE_SATISFIED
```

STOP after this terminal: no auto-promotion. Next step is the independently
reviewed S4.4E (SingleBuffer -> production CurrentFinal), per the standing
promotion discipline.
