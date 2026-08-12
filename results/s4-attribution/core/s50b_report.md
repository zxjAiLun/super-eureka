# S5.0B — Single-Generation Child Probe A/B gate (INCREMENTAL after S4.4E promotion)

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
| wall s | 84.40 | 74.20 | -12.09% |
| NPS | 215,476 | 245,100 | +13.75% |
| median per-position paired wall delta | | | -10.14% |

## Per-repetition

| rep | order | wall A (s) | wall B (s) | delta | NPS A | NPS B |
|---|---|---|---|---|---|---|
| 1 | A->B | 15.27 | 13.66 | -10.55% | 238,195 | 266,279 |
| 2 | B->A | 14.84 | 13.53 | -8.80% | 245,109 | 268,749 |
| 3 | A->B | 14.88 | 13.25 | -10.90% | 244,479 | 274,394 |
| 4 | B->A | 19.46 | 15.92 | -18.21% | 186,891 | 228,502 |
| 5 | A->B | 19.95 | 17.83 | -10.62% | 182,302 | 203,968 |

## Mechanism counters (one timing-enabled corpus pass per profile)

```text
movegen_legal_calls          A=   5,613,489   B=   1,977,926
probe_child_generations      A=   3,635,563   B=           0
negamax_body_generations     A=   1,977,896   B=   1,977,896
movegen_has_any_calls        A=     495,041   B=   4,130,604
```

## Verdict

aggregate wall delta -12.09% -> **PROMISING (aggregate wall reduction >= 3%)**

