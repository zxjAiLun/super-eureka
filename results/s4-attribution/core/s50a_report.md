# S5.0A — Duplicate Child Full-Legal Generation Attribution

ATTRIBUTION ONLY — no optimization candidate.

## Contract

```text
profile:  current-final (promoted, legality fast)
corpus:   30-position S4 corpus
limit:    fixed depth 6
TT:       16 MB cold | threads 1 | sparse sampler 256/1
repeat:   3 full corpus repetitions
```

## The duplicated path

```text
parent makes a move
  probe_child_draw(child):  generate FULL legal list  (discarded on Continue)
    -> terminal / intended-claim detection needs ONLY the emptiness
  negamax entered body:     generate FULL legal list AGAIN  (the duplicate)
qsearch edges:              generate FULL legal list, body replaces it with
                            tactical/evasion lists (never used)
```

## Full-legal generation accounting (aggregate, 3 reps)

```text
total full-legal calls                 16,840,467
  probe_child_generations (discarded)  10,906,689
    main edges                         5,911,653
    qsearch edges                      4,976,472
    root edges                         18,564
  negamax body generations (duplicate) 5,933,688
  root generations (once per search)   90
  final-evasion generations (single)   0

DUPLICATES (body re-gen of a just-probed position):
  5,933,688 calls = 35.2% of full-legal calls
DISCARDED qsearch probe lists (body uses tactical/evasion):
  4,976,472 calls = 29.6% of full-legal calls
TOTAL waste: 10,910,160 calls = 64.8% of full-legal calls
```

## Wall impact (same per-call cost assumption)

```text
movegen_legal bucket wall     34.08 s  (60.8% of elapsed)
duplicate-regeneration wall   12.01 s  (21.4% of elapsed)
qsearch discarded wall        10.07 s  (18.0% of elapsed)
TOTAL waste wall              22.08 s  (39.4% of elapsed)
```

## Per repetition

| rep | full-legal calls | duplicates | qsearch discarded | waste % of calls |
|---|---|---|---|---|
| 1 | 5,613,489 | 1,977,896 | 1,658,824 | 64.8% |
| 2 | 5,613,489 | 1,977,896 | 1,658,824 | 64.8% |
| 3 | 5,613,489 | 1,977,896 | 1,658,824 | 64.8% |

## Reading / S5.0B implication

- The child probe needs ONLY the emptiness boolean (terminal) and the
  claim check (list-independent): `has_any_legal_move` is exactly
  equivalent for both.
- Replacing the probe's full-legal generation with a has-any probe
  removes ALL probe generations (main + qsearch + root edges) while
  the entered body keeps its single full generation.
- This is a fixed-depth tree-IDENTICAL change by construction (empty
  list iff no legal move).

