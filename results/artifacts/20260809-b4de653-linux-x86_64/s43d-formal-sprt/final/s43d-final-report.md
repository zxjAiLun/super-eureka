# S4.3D — Formal Pentanomial SPRT: FINAL REPORT (FORMAL_SPRT_PASS)

tournament: `86835da4-bdb4-4514-a950-7a5ecf1f132a` (s43d-formal-sprt)
terminal: `SPRT_ACCEPT_H1`

## Frozen contract

- model: pentanomial
- elo model: logistic
- H0: +10 Elo, H1: +30 Elo
- alpha: 0.05, beta: 0.05
- bounds: [-2.9444389791664403, +2.9444389791664403]
- max_pairs: 2000 (safety ceiling; not reached)

## Final result

- pairs / games: **263 / 526**
- candidate W/L/D: **249 / 180 / 97**
- candidate score: (249 + 97/2) / 526 = **56.558935%**
- **Ptnml: [28, 27, 111, 42, 55]** (sum 263)
- **final LLR: +2.975553182682472**
- decision: **ACCEPT_H1**
- integrity: **526/526 verified, crash 0, forfeit 0, verifier failure 0**

## First-crossing audit (replay of the finished pair sequence; no new games)

Replayed the 263-pair sequence through the same differential-tested
pentanomial implementation (pair_points_index + pentanomial_llr):

```text
pair 262 cumulative Ptnml: [28, 27, 111, 42, 54]
pair 262 LLR:              2.846894523039176   < +2.9444389791664403  ✓

pair 263 outcome category: 4 (WW, candidate wins both)
pair 263 cumulative Ptnml: [28, 27, 111, 42, 55]
pair 263 LLR:              2.975553182682472   >= +2.9444389791664403  ✓ FIRST CROSSING
```

No statistical parameters changed; the first boundary crossing happened on the
first verified complete pair where LLR reached the upper bound.

## Provenance

- candidate source: `b4de6532bc239d0ae4c5729a41dea890b2e0a815`
- build: `20260809-b4de653-linux-x86_64`
- binary sha256: `c6a08996d14c4746df77a783a81d660ba9f0db0ae8fc1bcd1239896f8c7e607f`
- candidate preset: `s43b-legality-fast` (--profile current-final-legality-fast)
- baseline preset: `s43b-current-final` (--profile current-final)
- opening: stockfish-8moves-v3
- opening sha: `5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e`
- seed: `20260811`, opening plies: 16, historical exclusions: 250
- TC: 10+0.1, Hash 16MB, Threads 1, concurrency 1, strict color reversal

## Verdict

```text
FORMAL_SPRT_PASS
PROMOTION_EVIDENCE_SATISFIED
```

Evidence files in this directory: sprt.json, summary.json,
artifact-manifest.json, combined.pgn, pair_sequence.json (per-pair candidate
outcomes used for the crossing replay).
