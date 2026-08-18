# S7.5B Repair 1 Gate Report

STATUS: **EVIDENCE APPROVED — CANDIDATE REJECTED / CLOSED**

```text
S7.5B-0:      APPROVED / OBSERVATION COMPLETE / CLOSED
implementation: Repair 1, 207dc508b5e59bd12fe19cad5a155cb2f64c30f1
baseline:       54bed4b
candidate:      current-final-bounded-check2
binary SHA256:  2f7d4b1b8f2f2a9185d5b64c5e78b1c588cfe7b5cbf9a5f6305415fe258f4812
verdict:        see s75b-verdict.md
```

Repair 1 applies the frozen B offset to both non-root reduced scout paths:
ordinary PVS LMR and S7.4A caller-null-window. The full/re-search/verification
paths continue to use the full `child_depth`.

## Gate Summary

| gate | result | evidence |
|---|---|---|
| G0 | PASS | 30/30 S4 d6 exact cross-source matches vs `54bed4b` |
| G1 | PASS | 328 debug/release tests; candidate policy isolation |
| G2 | PASS | Budget/TT tests plus reduced-depth invariant: `(6,1)->4/5`, `(6,2)->3/4` |
| G3 | PASS | d6 nodes 1.073763x, d7 nodes 1.056450x; wall below 2x |
| G4 | PASS | Median root depth unchanged at 1s and 3s; no drop >=2 |
| G5 | FAIL | Candidate-only 3 vs baseline-only 1; cp regressions at >=100 and >=300: 1 each |
| G5W | FAIL | 1s: 2 wins / 4 losses; 3s: 1 win / 2 losses |
| G6 | DIAGNOSTIC | No additional hard-reject verdict |

## G4 Funnel

| limit | B opportunities | B applied | B budget blocked | A-overlap blocked |
|---|---:|---:|---:|---:|
| 1s | 200180 | 167558 | 32622 | 0 |
| 3s | 633425 | 526892 | 106533 | 0 |

The P1 contract mismatch is repaired, and this evidence is the decisive,
contract-correct evaluation of the frozen S7.5B candidate: cost gates pass,
strength gates fail. The candidate is **REJECTED / CLOSED**; see
`s75b-verdict.md`. Arena stays **NO-GO**; S7.5C stays **NO-GO**.
