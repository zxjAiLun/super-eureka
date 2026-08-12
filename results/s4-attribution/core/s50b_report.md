# S5.0B — Single-Generation Child Probe A/B gate

Same-tree throughput candidate — no chess semantics change.

## Candidate

```text
profile: current-final-single-generation
change:  probe_child_draw uses has_any_legal_move (emptiness
         boolean) instead of a full legal list discarded on
         Continue. The entered body keeps its single full
         generation. S5.0A: 64.8% of full-legal calls were
         discarded probe lists (35.2% duplicate body re-gen,
         29.6% qsearch probe lists).
equivalence: has_any false <=> no legal move -> identical
         terminal/claim decisions -> fixed-depth tree identical
         by construction.
```

## Contract

```text
corpus:   E:\AUbuntuProject\project\chessenginedemo\tools\data\s4_compute_positions.epd
limit:    fixed depth 6
TT:       16 MB cold per position
threads:  1
sampler:  disabled (perf) (sparse timing off)
reps:     5 interleaved paired
```

## Fixed-depth search-tree equivalence (same binary A/B)

nodes / score / bestmove / PV identical on all positions x reps: PASS

## Aggregate throughput

| metric | current-final | single-generation | delta |
|---|---|---|---|
| wall s | 88.77 | 76.35 | -13.99% |
| NPS | 204,861 | 238,182 | +16.26% |
| median per-position paired wall delta | | | -13.92% |

## Per-repetition

| rep | order | wall A (s) | wall B (s) | delta | NPS A | NPS B |
|---|---|---|---|---|---|---|
| 1 | A->B | 13.62 | 11.54 | -15.26% | 267,041 | 315,140 |
| 2 | B->A | 16.16 | 13.82 | -14.51% | 225,062 | 263,256 |
| 3 | A->B | 20.34 | 17.23 | -15.26% | 178,826 | 211,029 |
| 4 | B->A | 17.64 | 15.59 | -11.65% | 206,135 | 233,322 |
| 5 | A->B | 21.01 | 18.17 | -13.50% | 173,143 | 200,161 |

## Mechanism counters (one timing-enabled corpus pass per profile)

```text
movegen_legal_calls          A=   5,613,489   B=   1,977,926
probe_child_generations      A=   3,635,563   B=           0
negamax_body_generations     A=   1,977,896   B=   1,977,896
movegen_has_any_calls        A=     495,041   B=   4,130,604
```

## Verdict

aggregate wall delta -13.99% -> **PROMISING (aggregate wall reduction >= 3%)**

