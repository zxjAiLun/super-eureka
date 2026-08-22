# S6-C1 Arena Screen - FAIL

STATUS: **S6_C1_ARENA_SCREEN_FAIL**

The candidate is far weaker in real play than the baseline, so the formal SPRT
was NOT started.

## Result

| | value |
|---|---:|
| pairs / games | 200 / 400 |
| candidate W / L / D | 107 / 213 / 80 |
| candidate score | **36.7500%** |
| gate | >= 48.0% |
| verdict | **FAIL** |
| Elo estimate | **-94.3** |

## Integrity - all 15 checks passed

Every check passed, which is what makes the score trustworthy: this is a real
strength loss, not a broken harness.

- completed_pairs_200
- games_400
- wld_sums_to_400
- candidate_is_engine_a
- engine_a_profile
- engine_b_profile
- verified_games_400
- failure_reason_none
- no_missing_results
- strict_colour_swap
- pairwise_alternation
- arena_elo_disabled
- status_completed
- fen_overlap_zero
- pgn_files_200

W/L/D were also recomputed independently from the 400 PGN results and matched
the database exactly (107/213/80).

## Configuration

Same binary on both sides, differing only in `--profile`:

```text
build_id      20260822-91ca15c-linux-x86_64
binary sha256 7535ec09414eed2ae3fab230cf8f111d25667921a80ebcc4f7a08a38311fc4ad
engine_a      s6-c1-phase-affine-91ca15c  --profile current-final-phase-affine
engine_b      20260822-91ca15c-linux-x86_64  --profile current-final
TC            10+0.1   threads 1   hash 16MB   concurrency 1
openings      stockfish-8moves-v3, 16 plies, seed 2026082201
excluded      3337 historical + smoke FENs (overlap verified 0)
Arena Elo     disabled
```

## Why this matters

The calibration did exactly what it was fitted to do - it improved agreement
with the Stockfish 18 teacher on held-out, game-disjoint positions (clipped MAE
162.142 -> 155.103, RMSE 231.462 -> 215.082, N3E). It still loses about 94 Elo.

Teacher imitation is not Elo. Rescaling the evaluation changes the score scale
that pruning, aspiration windows, null-move, qsearch and TT entries are all
calibrated against, and that cost dwarfs the accuracy gain. The prime suspect is
the `zero`-phase slope of 2.51, which more than doubles endgame scores; it was
deliberately kept unmodified because trimming a frozen fitted parameter would
have turned a falsifiable test into an unfalsifiable one.

This is the outcome the S6-C1 authorization was designed to allow: the candidate
was isolated behind its own profile, the default never changed, and Arena
answered the question that bench metrics could not.

## Outcome

NO_PROMOTION. Formal SPRT not started. `CurrentFinal` remains
`PRODUCTION_PROFILE` and production baseline `bde9085` is untouched.
