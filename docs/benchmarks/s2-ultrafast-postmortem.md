# S2 historical ultrafast postmortem

Status: diagnostic archive only. This report does not authorize a profile
change, does not estimate a playing strength, and does not treat node or
depth changes as Elo.

## Source and scope

The source is the completed 400-game `10+0.1` match from the historical
profile. That profile is cancelled and fail-closed; it is not the active
evaluation configuration. The PGN and manifest remain preserved under the
preflight result directory, while the analyzer output was regenerated after
clock-pressure semantics were corrected.

The source match had candidate `Aspiration` and baseline `Current`, with 200
opening pairs and strict color reversal. Its old Fastchess ordering must not
be reused as evidence of candidate-oriented SPRT configuration.

## Aggregate results

| Metric | Value |
| --- | ---: |
| Games | 400 |
| Moves | 60,728 |
| Moves with evaluation | 54,382 |
| Search-info moves | 54,328 |
| PGN parse errors | 0 |
| Candidate records | 9,109 |
| Horizon/time flags | 358 |
| Shallow candidates | 3,547 |
| Long-think diagnostics | 14 |
| Short-think diagnostics | 9,095 |
| Time-pressure candidates | 0 |
| Clock-pressure status | unknown for all 9,109 records |
| Passed-pawn-context candidates | 8,939 |
| Promotion-race flags | 8,685 |
| Mate-transition candidates | 552 |

The old PGN has no `timeleft` telemetry. Consequently, `long_think` and
`short_think` describe the time spent on a move, not remaining-clock pressure;
the report does not infer time trouble from those fields.

The analyzer's adjacent-evaluation heuristic is useful for candidate
screening, but it is not an objective blunder label. In particular, the large
record count includes promotion and mate-transition contexts and must not be
read as 9,109 independent engine errors.

## Top 20 passed-pawn/promotion-context records

These are the largest `eval_loss_cp` records among entries with passed-pawn
context or promotion-race flags. They are retained as reproducible positions
for later investigation, not as ground-truth tactical annotations.

| Game | Ply | Mover | Move | Result | Loss (cp) | Depth | Promotion race | FEN before move |
| ---: | ---: | --- | --- | --- | ---: | ---: | :---: | --- |
| 231 | 129 | Current | `Kg2` (`f1g2`) | 0-1 | 100117 | 3 | no | `4r1k1/5p2/R2p3p/2pP2p1/2Pq2P1/PQ5P/2P5/5K2 w - - 7 65` |
| 184 | 109 | Aspiration | `Qxh7` (`h6h7`) | 0-1 | 99972 | 5 | no | `2r5/1p2kp1p/7Q/8/4p1p1/4P3/2r1qPPP/5RKR w - - 24 55` |
| 296 | 124 | Current | `Ke7` (`f7e7`) | 1-0 | 99970 | 5 | no | `8/1pr2kp1/p2R3p/3P4/2q5/P3KQ1P/6P1/8 b - - 1 62` |
| 229 | 91 | Current | `Rb3` (`d3b3`) | 0-1 | 99967 | 5 | no | `8/6pp/8/1B2k3/1P6/3R1PKP/8/8 w - - 4 46` |
| 33 | 83 | Current | `Bd7` (`a4d7`) | 0-1 | 99890 | 5 | no | `5rk1/7p/p2p4/Q1pPp3/B1P3P1/P5nP/5r2/6K1 w - - 0 42` |
| 15 | 81 | Current | `Qa8+` (`a6a8`) | 0-1 | 99869 | 5 | no | `6k1/5p1p/Q7/4p1p1/1P2Pq2/P4B1K/2r3PP/8 w - - 0 41` |
| 247 | 50 | Aspiration | `f6` (`f7f6`) | 1-0 | 99707 | 3 | no | `3r1rk1/5p2/pqb4Q/3pR3/2pP4/2P2B2/P4PPP/3R2K1 b - - 2 25` |
| 230 | 82 | Current | `R8e4` (`e8e4`) | 1-0 | 99688 | 3 | no | `4r1k1/2q3pp/3R4/3R1PP1/8/1QP5/4rPKP/8 b - - 0 41` |
| 268 | 98 | Current | `Qb6` (`c5b6`) | 1-0 | 99654 | 3 | yes | `4r1k1/3b4/3PpQ2/1pq1Pp2/2p5/2R5/P4PP1/3R2K1 b - - 19 49` |
| 348 | 52 | Current | `Kxh7` (`h8h7`) | 1-0 | 99653 | 5 | yes | `1rb1r2k/1p2P2R/p2q4/2pB2P1/2Pp4/P4QP1/1P5P/4R1K1 b - - 0 26` |
| 338 | 114 | Current | `g4` (`g5g4`) | 1-0 | 99628 | 3 | no | `8/1p3r1p/p2R4/6pk/2q1P3/P6P/1Q3PB1/7K b - - 0 57` |
| 286 | 64 | Current | `Qb5` (`a5b5`) | 1-0 | 99613 | 3 | yes | `3r1r2/ppp2k2/2n2P2/q1P3R1/3P1Q2/P7/1PB3P1/3R2K1 b - - 2 32` |
| 117 | 35 | Current | `Kf1` (`g1f1`) | 0-1 | 99597 | 5 | no | `r4rk1/p2n1ppp/1p6/3p1P2/3P2n1/P1BB4/1PQ2P1q/R2R2K1 w - - 0 18` |
| 364 | 76 | Current | `Kb8` (`c7b8`) | 1-0 | 99532 | 5 | no | `8/p1k2Q2/6p1/4N3/8/q5P1/2b2P1P/6K1 b - - 3 38` |
| 213 | 155 | Current | `Kc2` (`d1c2`) | 0-1 | 99520 | 5 | no | `8/3B4/5R1p/1P6/1bkp1P2/7P/8/3Kr3 w - - 13 78` |
| 121 | 69 | Current | `Kh2` (`g2h2`) | 0-1 | 99490 | 3 | no | `4r2k/1p2b1pp/2p5/p1P3P1/2QP2q1/5rB1/6K1/1R2R3 w - - 8 35` |
| 69 | 91 | Current | `Rxd5` (`d1d5`) | 0-1 | 99330 | 5 | no | `4R3/5p1p/5kp1/3pn3/8/7P/1r4r1/3R3K w - - 0 46` |
| 301 | 171 | Current | `Kg8` (`g7g8`) | 0-1 | 99325 | 5 | no | `8/6K1/4k2p/8/8/8/4r3/8 w - - 12 86` |
| 181 | 181 | Current | `Kg3` (`f2g3`) | 0-1 | 99317 | 6 | no | `8/8/7p/8/4k3/8/2r2K2/8 w - - 20 91` |
| 82 | 79 | Aspiration | `Kh2` (`h1h2`) | 0-1 | 99300 | 3 | no | `4r1k1/p4rp1/1p5p/2p5/P4p1R/2QP1q2/1PP5/R6K w - - 4 40` |

## Decision boundary

The historical 400-game sample remains a valid protocol/PGN dataset. Its
old SPRT orientation was invalid for candidate promotion, so its formal
decision is `INCONCLUSIVE`. `Current` remains unchanged. The active follow-up
uses a fresh seed and candidate-first Fastchess ordering under `2:00+1`.
