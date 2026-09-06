# S10-G1-D1 Closeout — FT256 vs FT128-v3 Capacity Arena (same binary)

**Status: CLOSED / SPRT_ACCEPT_H0 — capacity-width route REJECTED at this
architecture; width ladder STOPPED.**

## Frozen protocol (see s10-g1-d1-provenance.md, commit 6c961cd)

```
tournament   03431578-f9d9-4a18-ba49-0110817dfdaa
experiment   s10-g1-d1-ft256-vs-ft128-1m01 (confirmation)
binary       792a8e4 / 909b5bd3... (BOTH arms)
profile      current-final-nnue-v2q-material (BOTH arms)

A candidate   FT256-v3, model aed00d05... (G1 seed 20260819,
              val 137.811 / holdout 137.048)
B control     FT128-v3 twin, model a9f19bfd... (payload byte-identical
              to the E3 v2 artifact daddd085...; val 138.578 /
              holdout 138.693)

TC           bullet_1_0
SPRT         pentanomial logistic, elo0=0, elo1=+10, alpha=beta=0.05,
             max_pairs=1000
openings     stockfish-8moves-v3, plies 16, seed 2026090301,
             3004 excluded FENs (actual union of old D1 + F1-D1 +
             E3-D1 + A5), verified zero overlap; terminal indices
             match the frozen snapshot exactly
arena_elo    disabled
```

## Terminal result

```
decision     ACCEPT_H0 (LLR -2.9802 < -2.9444)
pairs        866 / 1000
games        1732 (670 W / 728 L / 334 D)
score        48.33%  (~ -12 Elo for FT256)
ptnml        [149, 147, 310, 133, 127]

integrity    1732/1732 games verified; 0 retried pairs; opening
             drift 0 (indices == frozen snapshot); no anomalies
duration     ~36h
```

## Interpretation (frozen result paths)

```
ACCEPT_H0, point estimate slightly below 50%
-> FT256's +1.12cp offline accuracy did NOT survive the -18.9% NPS
   cost; net capacity value ≈ -12 Elo (within noise of zero, but
   decisively not >= 0)
-> STOP the FT width ladder (no FT384/512)
-> representation remains the open frontier (consistent with the
   H0/I1 audit chain conclusions)
```

## The capacity experiment ladder (closed)

```
G1-A  offline:   FT256 five gates ALL PASS (+1.12cp, 3/3 seeds)
G1-B  runtime:   v3 width-aware format, all three gates PASS
                 (backward-compat bit-exact, twin payload-identical,
                 FT256 parity/forensics PASS); paired NPS -18.9%
G1-D1 Arena:     ACCEPT_H0 @ 866 pairs, 48.33%
```

Complete chain: offline improvement confirmed, runtime cost measured
precisely, and the matched same-binary Arena priced the trade — the
accuracy gain does not cover the speed loss. The width route is
closed with full evidence at every layer.

## Verdicts

```
G1-D1 strength gate      FAIL / ACCEPT_H0
FT width ladder          STOPPED (no FT384/FT512 fishing)
G1-D2 (FT256 vs CurrentFinal)  NOT RUN (contingent on D1 ACCEPT_H1)
3+2 confirmation         N/A
```

## Artifacts

```
results/s10/s10-g1-d1-provenance.md          (pre-results freeze, 6c961cd)
results/s10/s10-g1-d1-frozen-snapshot.json
results/s10/s10-g1-d1-sprt.json              (terminal server record)
server: /var/lib/chessarena/runs/03431578.../sprt.json (authoritative)
```
