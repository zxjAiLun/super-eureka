# S7.5A Repair 1 — Fixed-Wall Evidence Repair Verdict

STATUS: **REPAIR COMPLETE**

## Repair scope

```text
search implementation:  UNCHANGED (60799c6)
budget:                 S75A_FORCING_BUDGET = 2, FROZEN
engine used:            61f973c final-clean binary
                        e5c85c0951fb456956f5b7203a41859e841c71b1fbc7a5a46cee2bbe9075eaaf
```

Tool repairs:

1. `_g5w_classify()` now follows the frozen priority exactly:
   - teacher mate correctness first
   - teacher bestmove second
   - non-mate cp error >=100 only when bestmove state is tied
   - mate distance is informational only
2. G4 and G5W now rotate A/B execution order deterministically:
   - even position index: A -> B
   - odd position index: B -> A

Execution-order audit:

```text
G4:  A->B 80, B->A 80
G5W: A->B 120, B->A 120
```

## G4 repaired fixed-wall cost gate

| metric | 1000 ms | 3000 ms |
|---|---:|---:|
| completed | 80/80 | 80/80 |
| median depth A/B | 7 / 7 | 8 / 8 |
| mean depth A/B | 7.062 / 7.025 | 8.175 / 8.137 |
| median seldepth A/B | 18.0 / 18.5 | 20.0 / 20.0 |
| extensions applied | 22,704 | 70,773 |
| depth-1 extensions | 17,465 | 52,548 |
| budget 2->1 | 21,437 | 66,096 |
| budget 1->0 | 1,267 | 4,677 |
| budget-0 blocked | 337 | 975 |
| losing >=2 plies | 0 / 80 | 1 / 80 |
| verdict | PASS | PASS |

## G5W repaired same-time tactical gate

| metric | 1000 ms | 3000 ms |
|---|---:|---:|
| completed | 120/120 | 120/120 |
| candidate wins | 5 | 4 |
| candidate losses | 1 | 0 |
| hard rejects | 0 | 0 |
| mate-distance closer/farther | 0 / 0 | 0 / 0 |
| verdict | PASS | PASS |

## Verdict after Repair 1

```text
G4:   PASS
G5W:  PASS
S7.5A overall:  GATE CHAIN PASS AFTER REPAIR
Arena:          HOLD for independent review decision
S7.5B:          NO-GO
```

The repaired classifier and rotated execution order preserve the original
positive signal: candidate remains net positive at 1s and strictly positive
at 3s, with zero hard rejects.
