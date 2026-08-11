# S4.4B — Full-Legal Single-Buffer A/B gate

ATTRIBUTION/throughput candidate gate — no Arena match, no
promotion, no chess semantics change.

## Candidate

```text
profile: current-final-single-buffer
change:  full-legal materializes ONE Vec<Move>; pseudo moves are
         generated into it, legality filter compacts IN PLACE
         (stable read/write indices), then truncate. The second
         Vec<Move> allocation/materialization is eliminated.
rules:   EXACTLY the promoted LegalityFast eligibility/fallback
         (in-check probe-all; else pin mask + EP/castle/king/pin
         fallback probes, fast accept otherwise). No capacity
         policy change (Vec::new(), no reserve/smallvec/arena).
```

## Contract

```text
corpus:   E:\AUbuntuProject\project\chessenginedemo\tools\data\s4_compute_positions.epd
limit:    fixed depth 6
TT:       16 MB cold per position
threads:  1
sampler:  disabled (sparse timing off)
reps:     5 interleaved (per-position adjacent, within-pair order alternates per rep)
```

## Fixed-depth search-tree equivalence (same binary A/B)

nodes / score / bestmove / PV identical on all 30
positions x 5 reps: PASS

## Aggregate throughput

| metric | current-final | single-buffer | delta |
|---|---|---|---|
| wall s | 65.47 | 60.92 | -6.95% |
| NPS | 277,747 | 298,490 | +7.47% |
| median per-position paired wall delta | | | -7.06% |

## Per-repetition

| rep | order | wall A (s) | wall B (s) | delta | NPS A | NPS B |
|---|---|---|---|---|---|---|
| 1 | A->B | 12.65 | 11.65 | -7.86% | 287,583 | 312,121 |
| 2 | B->A | 12.69 | 11.82 | -6.90% | 286,528 | 307,777 |
| 3 | A->B | 12.77 | 12.09 | -5.39% | 284,705 | 300,935 |
| 4 | B->A | 13.17 | 12.30 | -6.61% | 276,174 | 295,712 |
| 5 | A->B | 14.19 | 13.07 | -7.90% | 256,319 | 278,291 |

## Mechanism counters (candidate, aggregate)

```text
full legal calls                     45,516,620
pseudo moves (full legal)            1,081,176,005
legal moves retained                 885,082,740
single-buffer truncations            45,516,620  (= full legal calls)
compaction writes (write != read)    98,044,775  (9.1% of pseudo moves)

baseline (two-buffer):
  pseudo Vec materializations        45,516,620
  legal Vec materializations         45,516,620
candidate (one-buffer):
  Vec materializations               45,516,620  (-50% vs baseline)
```

## Correctness gates (debug + release)

```text
ordered move differential (21 fixture classes)   PASS
1500-position reachable legal-walk differential  PASS
perft differential (5 standard fixtures)         PASS
fixed-depth search-tree equivalence (7 positions) PASS
cargo fmt / clippy -D warnings / cargo test      see STOP summary
```

## Verdict

aggregate wall delta -6.95% -> **PROMISING (aggregate wall reduction >= 3%) — QUALIFIED_FOR_ARENA_SCREEN**

STOP after this gate: no Arena artifact, no promotion, no Vec
reserve, no bitboards/piece lists, no tactical/evasion/has-any
changes, no search/eval changes.

