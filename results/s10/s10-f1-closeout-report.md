# S10-F1 Closeout — Material-Anchored Residual NNUE (same 300k)

**Status: CLOSED / PASS — B4/B5/F2 quant gates all PASS; Arena GO**

## Chain of evidence

```
Forensic (4.d5 ...Bg7?? 5.dxc6)   pure NNUE +13cp on a black-down-knight
                                   position -> material blindness
F0 distribution audit              down-a-minor MAE 210cp, down-rook+ 256cp
                                   vs balanced 153cp; +142cp signed bias
                                   pulling losing positions toward zero
F1 training                        same 300k, same SF18 labels, same
                                   architecture (22528-128-32-32-1), same
                                   three seeds; ONLY the target decomposes:
                                   R = clamp(T, ±2000) - M (no second clip)
F1-H winner holdout                seed 20260820: val 149.40 / holdout
                                   149.86 (ratio 1.0031, gate <=1.15 PASS;
                                   < B3 winner holdout 166.076 PASS)
B4 export                          FP32 artifact 238e17be...; semantic
                                   target_mode header (EUNN2Q01 v2)
B5 PTQ + parity                    quantized artifact 77404d04...; raw
                                   residual parity 1000/1000 exact; composed
                                   parity 1000/1000 exact; FP32-vs-quant
                                   residual error mean 0.129cp / max 0.78cp;
                                   proven overflow bounds re-run:
                                   l1_mac 1.23e9 < i32::MAX 2.15e9 (43%
                                   headroom), all four bounds pass
F2 quantized forensic              after_dxc6 = -261.16cp (was +13);
                                   removal medians 320/487/872cp at 100%
                                   direction — zero degradation vs FP32
```

## Three-seed validation composed MAE (correction of the earlier report)

The B3 pure-NNUE baselines are PER-SEED (the earlier draft wrongly copied
seed 20260818's value to all three rows):

| seed | B3 pure NNUE val MAE | F1 material-residual val MAE |
|---|---|---|
| 20260818 | 165.324615 | 149.866 |
| 20260819 | 165.569229 | 149.827 |
| 20260820 | 166.474075 | 149.399 (winner) |

Winner selection used validation composed MAE only; the single holdout
evaluation was performed AFTER selection and changed no parameter.

## Bucket MAE + signed bias (seed 20260820, 100 shared validation bases)

| bucket | B3 q01 MAE (bias) | F1 FP32 MAE (bias) |
|---|---|---|
| balanced ±250 | 159.9 (-7) | 120.9 (-14) |
| stm down minor | 253.7 (+142) | 227.3 (+38) |
| stm down rook+ | 202.1 (+49) | 246.4 (-194) |
| stm up minor | 163.6 (-58) | 133.9 (-71) |
| stm up rook+ | 134.8 (-112) | 191.6 (+120) |
| overall | 171.9 | 142.2 |

The +142cp pull-to-zero bias in the down-a-minor bucket — the statistical
form of the forensic bug — is largely eliminated (+38). The up-rook+ /
down-rook+ buckets now show a counter-bias (the fixed anchor overshoots
when the model previously under-reacted); recorded, accepted for the
Arena candidate.

## Artifact format (EUNN2Q01 v2)

```
header offset 40 (reserved u32 in v1) = target_mode
  0 = cp                 (output IS the eval; the B3/B5 recipe)
  1 = material_residual  (output is a cp residual; runtime composes
                          material_cp_stm + residual)
```

v1 artifacts load with implicit mode `cp`. The loader, the UCI startup
path, and the bench profile path all fail closed on a profile/artifact
mode mismatch:

```
pure profile    + residual artifact  -> refused
material profile + cp artifact       -> refused
material profile + residual artifact -> ok (profile
  current-final-nnue-v2q-material)
```

Runtime composition (evaluate_profiled, search.rs):

```
eval = material_cp_stm(pos) + nnue_residual_cp(pos)
material_cp_stm = P=100 N=320 B=330 R=500 Q=900 (PieceType::value),
                  stm perspective, kings 0 — NO Eval2 positional terms
KQK/KRK exact mop-up override keeps priority (unchanged)
```

Python<->Rust material cross-check: 270,000/270,000 records exact
(fail-closed inside the trainer on every run).

## Residual target distribution (recorded before training, never clipped)

```
min -2170  p01 -544  p05 -306  median +37  p95 +499  p99 +808  max +2590
|R| > 2000: 4 records    > 3000: 0    > 4000: 0
```

## Permanent regression fixtures

* `4.d5 ...Bg7?? 5.dxc6` four-position forensic suite
  (tools/s10/f1_gates.py, f2_quant_forensic.py)
* 100-base counterfactual removal suite (8 removal kinds, 437 variants)
* gates: direction >= 99%, minor >= 250cp, rook >= 400cp, queen >= 700cp,
  after_dxc6 <= -150cp

## Artifacts

```
data/s10/f1/seed-20260818/  checkpoint + summary (composed MAE 149.866)
data/s10/f1/seed-20260819/  checkpoint + summary (composed MAE 149.827)
data/s10/f1/seed-20260820/  checkpoint + summary (winner, holdout 149.859)
                            nnue-v2-f32-residual.bin  (238e17be...)
                            nnue-v2-q01-material.bin  (77404d04..., v2,
                                                       target_mode=1)
results/s10/s10-f0-material-audit.json
results/s10/s10-f1-gates.json (+ s20260818, s20260819)
results/s10/s10-f1-quant-parity.json
results/s10/s10-f2-quant-forensic.json
```

## Next

Same-binary Arena D1 protocol (frozen):

```
A = current-final-nnue-v2q-material + 77404d04...
B = CurrentFinal
bullet_1_0, pentanomial, elo0=0, elo1=+10, alpha=beta=.05,
max_pairs=1000, NEW opening seed (not 2026083001),
experiment id: s10-f1-d1-nnue-v2q-material-300k01
```
