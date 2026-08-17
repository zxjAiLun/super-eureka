# S7.5B-0 - Post-A Bounded-Check Attribution

STATUS: **OBSERVATION COMPLETE**

## Contract

```text
production baseline:   4e1cb93d26b1638c98e6b07d3576b37be8a96ba6
profile:               current-final
diagnostic:            s75b-probe
binary SHA256:         0d0b917540789d039a4df2c9579897590f852d0be5216e35d9674fd7a5b02c89
search semantics:      ZERO CHANGE
corpora:               80 S7 x d6/d7 + 120 R2 x d8
```

## Main S7.5B Funnel

| metric | S7 d6 | S7 d7 | R2 d8 |
|---|---:|---:|---:|
| positions | 80 | 80 | 120 |
| checking edges | 235106 | 919451 | 6558282 |
| check2 child seen | 40211 | 149212 | 1392457 |
| check2 parent depth 1 | 31986 | 117667 | 1097368 |
| check2 parent depth 2+ | 8225 | 31545 | 295089 |
| check2 budget 2 | 36388 | 136329 | 980896 |
| check2 budget 1 | 3259 | 11123 | 363295 |
| check2 budget 0 | 564 | 1760 | 48266 |
| check2 -> single-evasion | 0 | 1 | 29 |
| single-evasion -> check2 | 0 | 0 | 9 |
| probe calls | 232472 | 910516 | 6381076 |
| probe pseudo moves | 6723115 | 23724962 | 175234160 |
| probe legality tests | 4755529 | 16421119 | 121391367 |
| claim-skipped checking edges | 0 | 0 | 812 |

## Probe Cost

Estimated cost uses the pinned supporting model:
`643 ns * probe_calls + 32.2 ns * legality_tests`.

| metric | S7 d6 | S7 d7 | R2 d8 |
|---|---:|---:|---:|
| estimated probe ms | 302.608 | 1114.222 | 8011.834 |
| estimated ns per probe | 1301.7 | 1223.7 | 1255.6 |
| estimated share of elapsed | 0.0129 | 0.0125 | 0.0190 |
| check2 share of checking edges | 0.1710 | 0.1623 | 0.2123 |

## Findings

- check2 is 17.1%, 16.2%, and 21.2% of checking edges in S7 d6, S7 d7, and R2 d8.
- Parent depth 1 accounts for 79.5%, 78.9%, and 78.8% of check2 events.
- Remaining A budget 2 accounts for 90.5%, 91.4%, and 70.4% of check2 events.
- check2/single-evasion adjacency is near zero: 0/0, 1/0, and 29/9 in the two reported directions.
- The bounded probe ran on 98.9%, 99.0%, and 97.3% of checking edges; the remainder were terminal or claim-skipped.
- Estimated probe cost is 1.3%, 1.3%, and 1.9% of elapsed time under the pinned supporting model.

## Interpretation

The tables answer the B-0 questions without enabling any B extension:

- check2 population remaining under the post-A production tree;
- parent-depth and remaining-A-budget distribution;
- adjacency between check2 and single-evasion opportunities;
- bounded eligibility probe volume and estimated legality cost.

No B budget or implementation recommendation is frozen by this file.
