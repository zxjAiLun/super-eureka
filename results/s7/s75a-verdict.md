# S7.5A — Single-Evasion Extension Candidate Verdict

STATUS: **GATE CHAIN COMPLETE — RETURNED FOR INDEPENDENT REVIEW**
Do not start S7.5B or Arena until explicit review GO.

## Frozen contract

```text
base:                    909cfc8
implementation:          60799c6
production reference:    990aed6 PRODUCTION_PROFILE
baseline profile:        current-final
candidate profile:       current-final-single-evasion
S75A_FORCING_BUDGET:     2 (FROZEN)
rule:                    main-search-only single-evasion extension
TT:                      budget-aware key, CurrentFinal normal reuse policy
                         (legacy exact-depth path NOT inherited)
```

## G0 production invariance

```text
CurrentFinal 30 S4 d6, dirty-gate binary vs 990aed6:
30/30 exact nodes/qsearch_nodes/score/bestmove/PV/seldepth
```

## G1/G2 implementation gates

```text
policy isolation test:   PASS (CurrentFinal + exactly S75A predicate)
edge/budget unit test:   PASS
TT budget-context test:  PASS
fmt:                     clean
clippy -D warnings:      clean
cargo test:              318/318 debug, 318/318 release
```

## G3 fixed-depth explosion fuse

| depth | nodes ratio B/A | wall ratio B/A | pass |
|---|---:|---:|---|
| 6 | 1.020151 | 1.018919 | PASS |
| 7 | 0.991776 | 1.007798 | PASS |

## G4 fixed-wall cost gate

| metric | 1000 ms | 3000 ms |
|---|---:|---:|
| completed | 80/80 | 80/80 |
| median depth A/B | 7 / 7 | 8 / 8 |
| mean depth A/B | 7.225 / 7.200 | 8.400 / 8.350 |
| median seldepth A/B | 18.5 / 19.0 | 21 / 21 |
| extensions applied | 26,935 | 84,483 |
| depth-1 extensions | 20,731 | 62,824 |
| budget 2->1 | 25,317 | 78,717 |
| budget 1->0 | 1,618 | 5,766 |
| budget-0 blocked | 390 | 1,530 |
| losing >=2 plies | 0 / 80 | 1 / 80 |
| verdict | PASS | PASS |

## G5 fixed-depth teacher gate (d6)

```text
evaluable:              176 / 176 (2 terminal N/A)
teacher bestmove:
  baseline matches:     34
  candidate matches:    37
  candidate-only:       3
  baseline-only:        0
cp improvement/regression:
  >=100cp:              1 / 0
  >=300cp:              0 / 0
  >=500cp:              0 / 0
mate hard rejects:      0
verdict:                PASS
```

## G5W same-time tactical effectiveness (120 R2)

| metric | 1000 ms | 3000 ms |
|---|---:|---:|
| completed | 120/120 | 120/120 |
| candidate wins | 6 | 4 |
| candidate losses | 2 | 0 |
| hard rejects | 0 | 0 |
| verdict | PASS | PASS |

## G6 depth-stability diagnostic (80 S7 d6 -> d7)

```text
bestmove changes:       baseline 15, candidate 15
>=200cp reversals:      baseline 1, candidate 2
mate transitions:       baseline 0, candidate 1
candidate transition:
  s7_tac_01 d6 cp:418 -> d7 mate:9 (same bestmove a5a2)
  direction: IMPROVEMENT (baseline d7 remains cp:407)
  hard contradiction:  none
verdict:               DIAGNOSTIC ONLY / NO VETO
```

## Evidence binary bridge

```text
gate runs used dirty pre-commit binary:
  11a4ebb0df8b6937ab2dbc7dd7ca83cf5f18ee174a25280f6314804bbd00bcfd
final clean release binary:
  source 61f973c3482d7dccbce040b41587f2a91a3506d5
  sha256 e5c85c0951fb456956f5b7203a41859e841c71b1fbc7a5a46cee2bbe9075eaaf
  UCI dirty false
30 S4 d6 bridge (both profiles, dirty-gate -> final-clean):
  30/30 exact
```

## Verdict

```text
S7.5A GATE CHAIN: PASS
NEXT ACTION:      STOP, independent review
S7.5B / Arena:    NO-GO until review
```

The candidate passes every frozen gate without approaching the 2x explosion
fuse. Fixed-wall cost is neutral-to-slightly-positive, fixed-depth teacher
quality improves, and same-time R2 tactical classification is positive at
both 1s and 3s. This is gate evidence only, not an Arena authorization.
