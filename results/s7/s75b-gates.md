# S7.5B Candidate Gate Report

STATUS: **HYBRID_CANDIDATE_REJECTED / NOT DECISIVE FOR FROZEN S7.5B**

```text
source head:    f66ab37aabcd1450a7e4c499649446bbd15d4923
candidate:      current-final-bounded-check2
baseline:       current-final
binary SHA256:  17838687bab2ee335d3fd48c5898c9e94313fe1fbff0e691add552e83a74e79f
implementation: f66ab37 (pre-Repair 1)
gate commit:    3f1a5eb
```

This evidence is valid for the hybrid implementation in `f66ab37`, but is not
the frozen S7.5B decision. That implementation omitted the B depth offset from
the two non-root reduced scout paths and consumed B budget without searching
the promised extra ply on those edges.

## Gate Summary

| gate | result | evidence |
|---|---|---|
| G0 | NOT RUN | The prior report did not execute the frozen 30-position exact cross-source check |
| G1 | PASS | CurrentFinal plus one B policy bit |
| G2 | PASS | A/B budget, overlap, and 2D TT-context tests |
| G3 | PASS | S7 d6 ratio 1.073763, d7 ratio 1.056450 |
| G4 | PASS | Median depth unchanged at 1s and 3s; no depth drop >=2 |
| G5 | FAIL | One >=100cp and >=300cp regression, no cp improvement |
| G5W | FAIL | 3s: candidate wins 1, losses 2 |
| G6 | DIAGNOSTIC | No additional hard rejection signal |

## G4 Funnel

| limit | B opportunities | B applied | B budget blocked | A-overlap blocked |
|---|---:|---:|---:|---:|
| 1s | 208203 | 174639 | 33564 | 0 |
| 3s | 661047 | 548563 | 112484 | 0 |

S7.5B remains **NEEDS_REPAIR**. Do not start Arena SPRT, close S7.5B, or move to
S7.5C from this result.
