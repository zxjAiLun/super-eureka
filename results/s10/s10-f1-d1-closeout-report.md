# S10-F1-D1 Closeout — Material-Residual NNUE 300k vs CurrentFinal (same binary)

**Status: CLOSED / SPRT_ACCEPT_H0 — candidate REJECTED; material-anchor
hypothesis STRONGLY SUPPORTED**

## Frozen protocol (see s10-f1-d1-provenance.md, commit aa29cda)

```
tournament   0c70792f-67b5-4022-8dbf-2e8378619a88
experiment   s10-f1-d1-nnue-v2q-material-300k01 (confirmation)
binary       db6c459 / 2f8ef902... (BOTH arms; only the evaluator differs)
candidate    current-final-nnue-v2q-material + model 77404d04...
             (EUNN2Q01 v2 target_mode=material_residual)
baseline     current-final (no model argv)
TC           bullet_1_0
SPRT         pentanomial logistic, elo0=0, elo1=+10, alpha=beta=0.05,
             max 1000 pairs
openings     stockfish-8moves-v3, plies 16, seed 2026083101,
             1004 excluded FENs (full old S10-D1 set + A5 smoke)
             — verified zero overlap with prior samples
arena_elo    off
```

## Terminal result

```
decision     ACCEPT_H0 (LLR -2.964421622611167 < -2.9444389791664403)
pairs        211 / 1000
games        422 (133 W / 211 L / 78 D)
score        40.7583%
ptnml        [46, 48, 75, 22, 20]
approx Elo   ~ -65 (pair-level 95% CI roughly -95 .. -36)

integrity    422/422 games verified; 0 retried pairs (attempt > 1: 0);
             0 failed/invalid pairs; opening indices at terminal match
             the frozen snapshot SHA exactly (c5a6aee4...); no worker
             restarts, model re-hash failures, or provenance mismatches
duration     2026-08-31T14:38:41Z .. 2026-09-01T03:27:10Z (~12.8h)
```

## Verdicts

```
S10-F1-D1 strength gate:              FAIL / ACCEPT_H0
S10-F1 material-anchor hypothesis:     STRONGLY SUPPORTED
```

The 300k material-residual candidate is NOT production-eligible and does
not advance to a slower-TC confirmation (the 3+2 gate exists only for
bullet ACCEPT_H1; running it on a formally rejected candidate would
spend machine time without changing any decision).

## Historical comparison (NOT a matched Elo decomposition)

```
old S10-D1 pure NNUE (9ef078f, seed 2026083001 openings):
  104 pairs, 18 W / 176 L / 14 D = 12.02% (~ -346 Elo), ACCEPT_H0

F1-D1 material-residual (db6c459, seed 2026083101 openings):
  211 pairs, 40.76% (~ -65 Elo), ACCEPT_H0
```

The two runs differ in engine commit, opening sample, and evaluator
delivery; the ~281 Elo recovery (~81% of the previously estimated gap)
is therefore a HISTORICAL COMPARISON, not a strict causal Elo
attribution. Within F1 itself the controlled evidence remains: same
300k/teacher/architecture/seeds, target decomposition only, offline
composed MAE 165.3 -> 149.4, holdout 149.86, forensic dxc6 +13 -> -261,
counterfactual removal medians 320/487/872 at 100% direction.

The remaining ~-65 Elo gap is NOT yet attributable. Candidate
explanations (trapped pieces, mobility, king safety, piece
interaction) are plausible but unproven; capacity, teacher
approximation, ±2000 target clipping, and evaluator/search calibration
are not excluded. The next controlled step is the 1M scale probe.

## Next (frozen decision)

```
3+2 64-pair confirmation        SKIP (bullet already ACCEPT_H0)
E1 (1M dataset build)           REPAIR REQUIRED — current 4eda8fe1...
                                 not approved as E2 input (SEED_1M
                                 priority instead of frozen B1
                                 position_id continuation; verifier
                                 missing the independent 12-cell exact
                                 check)
E2 (SF18 1M labeling)           GO immediately after repaired E1 PASS
                                 (unchanged teacher contract; PLUS the
                                 free parent-300k exact oracle: the
                                 300000 nested positions must reproduce
                                 the old B2 labels field-for-field,
                                 300000/300000 exact)
E3 (1M training)                MATERIAL-RESIDUAL ONLY — no pure-NNUE 1M.
                                 F1 recipe frozen verbatim (FT128,
                                 256-32-32-1, AdamW 1e-3/1e-5, SmoothL1
                                 beta 0.1, batch 256, epochs 100,
                                 patience 15, seeds 20260818/19/20)
E3 offline gates                3-seed median val < 149.83
                                 selected val      < 149.40
                                 winner holdout    < 149.86, ratio <= 1.15
                                 (fail any -> STOP, no quant/Arena,
                                 go capacity experiment)
WIDTH/CAPACITY                  HOLD until E3 result
```

## Artifacts

```
results/s10/s10-f1-d1-provenance.md        (pre-results freeze, aa29cda)
results/s10/s10-f1-d1-frozen-snapshot.json  (opening indices, both arms)
results/s10/s10-f1-d1-sprt.json             (server terminal SPRT record;
                                             indices match frozen snapshot)
server: /var/lib/chessarena/runs/0c70792f-.../sprt.json (authoritative)
```
