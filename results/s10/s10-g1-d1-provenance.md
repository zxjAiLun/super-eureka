# S10-G1-D1 Deployment & Opening Provenance (frozen)

**Status: RUNNING — AUTHORITATIVE (no interim interpretation)**

Frozen at completed_pairs = 0 of 1000.

## Tournament

```
tournament_id    03431578-f9d9-4a18-ba49-0110817dfdaa
name             S10-G1 D1 FT256 vs FT128-v3 same-binary
experiment_id    s10-g1-d1-ft256-vs-ft128-1m01 (stage: confirmation)
time_control     bullet_1_0
SPRT             pentanomial logistic, elo0=0, elo1=+10, alpha=beta=0.05,
                 max_pairs=1000
arena_elo        disabled
```

## Engines (same binary, same profile; ONLY the FT width / model differs)

```
binary           792a8e4 / 909b5bd336c1db62a2c94b6086787b2c89c09d172798e
                 fcdbf56a5740340a598 (BOTH arms, three-way SHA match)
profile          current-final-nnue-v2q-material (BOTH arms)

engine_a (candidate)
  preset_id      s10g1-formal-792a8e4-ft256-aed00d05
  model          nnue-v2q-material-ft256.bin
                 SHA aed00d055e8910ef43bcf6e23b910df42fa34310d069eff60c68
                 b002827be0ed (EUNN2Q01 v3, ft_width=256,
                 material_residual)
  provenance     1M material-residual, seed 20260819,
                 val 137.811 / holdout 137.048

engine_b (control)
  preset_id      s10g1-formal-792a8e4-ft128v3-a9f19bfd
  model          nnue-v2q-material-ft128v3.bin
                 SHA a9f19bfdff07eb04aecf22ed746b718652b06b3c6ec6fec1420d
                 8aad72ae3b0a (EUNN2Q01 v3, ft_width=128,
                 material_residual)
  provenance     payload byte-identical FT128 twin of the E3 v2 artifact
                 daddd085... (same E3 seed 20260820 checkpoint; only the
                 version word differs v2->v3), val 138.578 /
                 holdout 138.693
```

The A/B arms share data, labels, teacher, representation, material
anchor, quantization scheme, AND artifact format generation — the only
structural variable is the FT width.

## Runtime provenance

```
git             792a8e479ff3b505dac7ae86fe206281af726523
perf audit      same-machine 2M-node ABBAAB, old-binary vs new-binary
                FT128: ratio 0.981 (gate >= 0.95 PASS; the historical
                228k reading was cross-session environment variance)
paired cost     FT256 vs FT128 (interleaved, same session): -18.9% NPS
                — the exact trade this Arena prices
startup gates   both arms live-launched on the server; v3 headers
                verified (ft_width 256/128, target_mode
                material_residual); model SHAs exact; smoke search on
                an excluded A5 FEN returned bestmove on both arms
```

## Opening sample (fresh evidence)

```
opening_set     stockfish-8moves-v3 (sha 5835239f...)
format/plies    pgn / 16
seed            2026090301
indices         1000, unique
indices_sha256  a9fff0273ca2ed68a998e7b28b16cfee6d8d00387bc2c7f57c851c
                99f207d6e6
excluded        ACTUAL UNION of old pure D1 (1000, seed 2026083001) +
                F1-D1 (1000, seed 2026083101) + E3-D1 (1000, seed
                2026090201) + A5 smoke (4) = 3004 unique FENs
excluded_sha    5658989fd0d435934ff2eb55661d17767faa4de8fe336f462a9557782
                a21961d
verified        overlap with each prior set = 0; A5 = 0
                (checked post-creation against the frozen snapshot)
```

## Result paths (frozen)

```
ACCEPT_H1            -> FT256 vs same-binary CurrentFinal next
ACCEPT_H0 ~ 50%      -> STOP width ladder; representation next
ACCEPT_H0 small >50% -> STOP width ladder; representation next
ACCEPT_H0 < 50%      -> accuracy gain killed by runtime cost;
                        capacity route stopped
MAX_PAIRS            -> frozen statistical evidence, no extension
```

No interim interpretation. Escalation only on terminal
SPRT/MAX_PAIRS or infrastructure anomaly.
