# S7.5B — Bounded Check2 Extension Candidate Verdict

STATUS: **REJECTED / CLOSED**
Arena: **NO-GO**
S7.5C: **NO-GO**
S7.5 forcing lane: **CLOSED FOR NOW**

## Frozen contract

```text
S7.5B-0 observation:  APPROVED / CLOSED (54bed4b evidence)
base:                 CurrentFinal (production)
candidate profile:    current-final-bounded-check2
predicate:            uses_bounded_check2_extension()
rule:                 checking child with exactly 2 legal evasions
A budget:             S75A_FORCING_BUDGET = 2 (unchanged)
B budget:             1, independent from A (FROZEN)
scope:                main search only, depth > 0, Continue probe
stacking:             same-edge A-overlap blocks B
TT:                   (position, A budget, B budget); B=0 legacy-compatible
```

## Implementation timeline

```text
f66ab37  implementation v1
         P1: reduced scout paths omitted the frozen B +1 offset
3f1a5eb  hybrid gate evidence
         VALID FOR THE HYBRID IMPLEMENTATION, NOT THE FROZEN DECISION
207dc50  Repair 1 (APPROVED)
         ChildExtension { depth, budgets, check2_extended }
         both reduced scouts -> reduced_child_depth()
         full / re-search / verification -> child_depth
619ea3b  repaired gate evidence (APPROVED)
```

The P1 was a real structural fix, but under the current LMR policy
`late_move_reduction()` returns 0 for every checking move, so B-qualified
edges never carry `reduction > 0`. The repair is therefore defensive/structural;
it explains why pre- and post-repair G3 node totals are identical.

## G0 production invariance

```text
baseline binary:  54bed4b CurrentFinal
repaired binary:  207dc50 CurrentFinal
30 S4, depth 6, exact nodes/qsearch_nodes/score/bestmove/PV/seldepth:
30/30 PASS, 0 mismatches
```

## G1 / G2 implementation gates

```text
policy isolation:      PASS (CurrentFinal + exactly one B policy bit)
budget / TT tests:     PASS
reduced-depth invariant: PASS  (6,1)->4/5  (6,2)->3/4
fmt / clippy:          clean
cargo test:            328/328 debug, 328/328 release
```

## G3 fixed-depth explosion fuse

| depth | nodes ratio B/A | wall ratio B/A | pass |
|---|---:|---:|---|
| 6 | 1.073763 | 1.081074 | PASS |
| 7 | 1.056450 | 1.081562 | PASS |

## G4 fixed-wall cost gate

| metric | 1000 ms | 3000 ms |
|---|---:|---:|
| completed | 80/80 | 80/80 |
| median depth A/B | 7 / 7 | 8 / 8 |
| mean depth A/B | 7.350 / 7.213 | 8.450 / 8.350 |
| B opportunities | 200180 | 633425 |
| B applied | 167558 | 526892 |
| B budget blocked | 32622 | 106533 |
| A-overlap blocked | 0 | 0 |
| losing >=2 plies | 0 | 0 |
| verdict | PASS | PASS |

## G5 fixed-depth teacher gate (d6)

```text
evaluable:              176 / 176 (2 terminal N/A)
teacher bestmove:
  baseline matches:     37
  candidate matches:    39
  candidate-only:       3
  baseline-only:        1
cp improvement/regression:
  >=100cp:              0 / 1
  >=300cp:              0 / 1
  >=500cp:              0 / 0
mate hard rejects:      0
verdict:                FAIL (frozen CP correctness rule)
```

## G5W same-time tactical effectiveness (120 R2)

| metric | 1000 ms | 3000 ms |
|---|---:|---:|
| completed | 120/120 | 120/120 |
| candidate wins | 2 | 1 |
| candidate losses | 4 | 2 |
| hard rejects | 0 | 0 |
| verdict | FAIL | FAIL |

Both fixed-time windows show losses > wins with no hard tactical catastrophe.
There is no positive teacher-directed signal for a selective extension
candidate; the hybrid evidence already hinted at this, and the contract-correct
run confirms it.

## G6 depth-stability diagnostic

```text
bestmove changes:  baseline 15, candidate 17
>=200cp reversals: baseline 2, candidate 1
mate transitions:  baseline 1, candidate 2
verdict:           DIAGNOSTIC ONLY / NO VETO
```

## Verdict

```text
S7.5B candidate:   REJECTED / CLOSED
Arena:             NO-GO
S7.5C:             NO-GO
S7.5 forcing lane: CLOSED FOR NOW
```

The frozen first candidate (exactly 2 evasions, independent budget 1) was
evaluated completely and contract-correctly: cost is acceptable, strength
direction is negative. Accept the negative result; do not parameter-fish
(budget 2, evasions <= 2, depth-1-only, shared budget, etc.) after viewing
the failing corpus.

Search conclusion:

```text
"side to move in check with exactly one legal evasion"  -> valuable signal (S7.5A, PRODUCTION)
"check given, opponent has exactly two legal evasions"  -> not a reliable signal (S7.5B, REJECTED)
```

The `current-final-bounded-check2` profile remains bench-only as a
historical/bench identity; it must never be promoted and must not leak into
the next candidate lane.

## Next lane

```text
S6 NNUE production implementation
preflight: feature encoding / artifact format / incremental accumulator
          ownership / classical-NNUE profile isolation / exact fallback
base:     main = 619ea3b, CurrentFinal = S7.4A + S7.5A production semantics
```
