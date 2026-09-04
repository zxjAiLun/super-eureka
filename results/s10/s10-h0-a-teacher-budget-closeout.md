# S10-H0-A Teacher-Budget Closeout

**Status: CLOSED / DECISIVE — full-dataset 64k relabel REJECTED;
100k paired causal probe SKIPPED; 256k/1M sweeps SKIPPED.**

## Scope

This closeout covers ONLY the teacher search-budget question. It does
not answer representation, search-leaf distribution, search/eval
calibration, or decision-quality questions (H0-B..E remain open).

## Experiments

### H0-A1 — teacher-drift diagnosis (PASS / INFORMATIVE)

Frozen matched corpus: 500 validation positions (seed 2026090401,
ordered position_id SHA recorded in the artifact), each with:

```
Y16  = E2 teacher label   (SF18 Windows c86215fa..., nodes 16384,
                           Threads=1, Hash=64, MultiPV=1, ucinewgame)
YD20 = depth-20 reference (same binary/contract)
SFSE = 'go depth 1' search score (NOT a bare static-eval probe; SF18's
       static evaluation is itself an NNUE, and depth-1 search already
       includes qsearch, so this column is a correlated shallow-search
       reference, not an independent architecture yardstick)
F128 = Eureka FT128-v3 (1M material-residual, E3 seed 20260820)
F256 = Eureka FT256-v3 (G1 seed 20260819)
```

All scores unified: STM perspective, ±2000 clamp (the training-time
clip — the clamped view is what the network was asked to learn), mate
positions excluded from CP metrics (none overlapped: 500/500 both-cp).

Matrix (cp MAE / median / p95, clamped):

```
              MAE    median   p95    pearson   sign@50
Y16<->YD20    96.8    28      553    0.870     100%
SFSE<->Y16    76.4    40      293    0.921     98.3%
SFSE<->YD20  145.9    56      614    0.803     98.5%
F128<->Y16   137.8    93      431    0.818     90.6%
F128<->YD20  202.3   115      660    0.673     89.8%
F256<->Y16   140.9    92      425    —         93.1%
F256<->YD20  206.5   126      712    —         93.2%
```

Clamp sensitivity: unclamped Y16<->YD20 = 133.5 (three positions with
|YD20| ~ 7500); both recorded.

Phase-stratified Y16<->YD20 drift: high 21.9 / mid 65.6 / low 133.3 /
zero 291.6 — the shallow-teacher drift concentrates in the endgame
phases, where 16k nodes often reach only depth 1-2.

Deep-correction correlation: corr(F128 err, YD20 - Y16) = -0.19 —
weakly negative; no systematic anti-correlation signature.

Affine calibration F128->YD20 (fit 500 / test 500): a=0.90, b=10.7;
MAE 197.6 -> 200.2. No calibration gain: the error is structural, not
scale/offset.

### H0-A2 — 64k engineering pilot (PASS / DECISIVE)

One new budget tier on the SAME frozen 500 corpus (no resampling,
no new depth-20, no other tiers):

```
Y64 = SF18 Windows, nodes 65536, identical contract
cost: 303s wall @ 1.6 pos/s (single worker)
```

```
              MAE    median   p95
Y16<->YD20    96.8    28      553
Y64<->YD20    79.2    20      457
Y16<->Y64     25.5    14       87
```

Closure (1 - MAE64/MAE16 vs D20):

```
overall 18.2%
high    16.4%
mid     38.7%
low      5.4%     <- drift 133cp, barely moves
zero    13.6%     <- drift 292cp, barely moves
```

Agreement Y16 vs Y64: bestmove 379/500 (75.8%), WDL 263/500 (52.6%),
sign@|x|>=50 388/368 (some pairs double-counted by the >=50-or rule;
the strict both>=50 subset is the honest one: 368 pairs).

## Decision (frozen rules)

```
16k -> 64k = 4x labeling compute
overall deep-reference drift closure = 18.2%
low/zero closure = 5.4% / 13.6%

=> full-dataset 64k relabel              REJECTED
=> 100k 16k-vs-64k causal training probe SKIPPED
=> 256k / 1M budget sweeps               SKIPPED
```

## Causal language constraints (frozen wording)

1. NOT "endgame shallow-search error is not a budget problem." The
   evidence supports only: **under a 4x budget increase (16k -> 64k),
   low/zero-phase drift convergence is very limited; therefore 64k is
   not a cost-effective general teacher budget.** 256k was not tested.

2. NO offline-MAE-to-Elo extrapolation. The rejection rests on the
   compute/closure ratio alone (4x compute for 18.2% closure, with the
   largest-error phases barely improving), not on any predicted Elo.

3. NO "~56cp architecture gap" claim. The SFSE column is a depth-1
   SEARCH score (SF18's static evaluation is itself an NNUE, and
   depth-1 already includes qsearch), correlated with the shallow
   teacher through shared leaves; MAEs cannot be subtracted into an
   architecture contribution. Supported conclusion: **Eureka F128/F256
   error against the deep reference is substantially larger than the
   shallow teacher's own budget drift, so teacher depth is not the
   sole remaining error source; representation / training / search
   interaction remain the open variables.**

## What H0-A actually bought us

The previously attractive hypothesis — "raise the teacher budget,
labels get much better, retrain" — is now measured and dead at the
first economically plausible tier. This avoids burning ~4x of the E2
labeling compute (65B vs 16B nodes) on a relabel that demonstrably
would not fix the phases where the drift lives.

## Open questions (explicitly NOT answered here)

```
H0-B  representation collision (HalfKAv2_hm information loss:
      castling rights, en-passant, rule50, repetition context —
      two positions with identical features but different teacher
      scores give a mathematical lower bound no training recipe
      can cross)
H0-C  search-leaf distribution shift (Lichess game positions vs
      Eureka's own main/qsearch leaves)
H0-D  search/evaluator calibration (futility/null/LMR/qsearch
      stand-pat tuned around HCE score distributions)
H0-E  sibling ranking / decision quality
```

## Artifacts

```
tools/s10/h0_a1_audit.py          7-column audit + diagnostics
tools/s10/h0_a2_pilot.py          64k tier pilot
results/s10/s10-h0-a1-audit.json  A1 matrix + strata + calibration
results/s10/s10-h0-a2-pilot.json  A2 numbers + full Y64 labels
```
