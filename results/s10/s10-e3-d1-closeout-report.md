# S10-E3-D1 Closeout — 1M Material-Residual NNUE vs CurrentFinal (same binary)

**Status: CLOSED / SPRT_ACCEPT_H0 — candidate REJECTED (not production-
eligible); data-scale effect measured: 40.76% -> 42.87% (+2.1pp, ~ -65 ->
~ -50 Elo). Offline MAE improvement did NOT convert proportionally to
playing strength.**

## Frozen protocol (see s10-e3-d1-provenance.md, commit a85e7dc)

```
tournament   21d96a29-06ff-4458-b6f0-e7678ac34c43
experiment   s10-e3-d1-nnue-v2q-material-1m01 (confirmation)
binary       bffb53c / 41d64a9d... (BOTH arms)
candidate    current-final-nnue-v2q-material + model daddd085...
             (1M material-residual, Windows SF18 teacher)
baseline     current-final (no model argv)
TC           bullet_1_0
SPRT         pentanomial logistic, elo0=0, elo1=+10, alpha=beta=0.05,
             max 1000 pairs
openings     stockfish-8moves-v3, plies 16, seed 2026090201,
             2004 excluded FENs (old D1 + F1-D1 + A5), verified zero
             overlap; terminal indices match the frozen snapshot exactly
arena_elo    off
```

## Terminal result

```
decision     ACCEPT_H0 (LLR -2.9558 < -2.9444)
pairs        270 / 1000
games        540 (192 W / 269 L / 79 D)
score        42.87%  (~ -50 Elo)
ptnml        [62, 44, 103, 31, 30]

integrity    540/540 games verified; 0 retried pairs; no failed pairs;
             opening drift 0 (indices == frozen snapshot); no worker
             restarts; model re-hash clean
duration     ~10.5h
```

## The three-step chain (formal Arena, same-binary Route B each time)

```
pure 300k NNUE (old D1):        12.02%  (~ -346 Elo)
residual 300k (F1-D1):          40.76%  (~ -65 Elo)
residual 1M   (E3-D1):          42.87%  (~ -50 Elo)
```

### Interpretation (against the frozen result paths)

```
ACCEPT_H0 but clearly above F1's 40.76%
-> data scale helped but is NOT sufficient
-> next experiment: capacity / width
-> do NOT continue blind 3M scaling
```

The 300k -> 1M step moved +2.1 score points (~ +15 Elo) while offline
composed MAE improved 149.4 -> 138.6 (~10.8 cp). The MAE gain is real
and clean (same teacher, nested data, control reproduced F1 exactly),
but playing strength converges much more slowly than average eval
accuracy at this scale. The remaining ~ -50 Elo gap behaves like a
representation/capacity problem, not a data-volume problem.

## Verdicts

```
S10-E3-D1 strength gate:          FAIL / ACCEPT_H0
S10-E3 data-scale hypothesis:     PARTIALLY SUPPORTED (+2.1pp; far from
                                  closing the gap)
3+2 confirmation:                 SKIP (bullet already ACCEPT_H0)
1M -> 3M blind scaling:           REJECTED
width/capacity experiment:        NEXT (on the 1M Windows labels)
```

## Artifacts

```
results/s10/s10-e3-d1-provenance.md         (pre-results freeze, a85e7dc)
results/s10/s10-e3-d1-frozen-snapshot.json   (openings, both arms)
results/s10/s10-e3-d1-sprt.json              (server terminal record)
server: /var/lib/chessarena/runs/21d96a29.../sprt.json (authoritative)
```
