# S10-H0 Master Root-Cause Closeout

**Status: CLOSED / MULTIFACTORIAL — no single dominant cause established;
several attractive single-cause hypotheses experimentally rejected or shown
insufficient. Next phase frozen: S10-I1 Task-Aligned NNUE Training.**

## Scope and question

H0 asked why the NNUE candidate's steady offline improvement
(composed CP MAE 165.3 → 149.4 → 138.6 → 137.0 across B3 → F1 → E3 → G1)
converted to only ~12% → 43% Arena score against same-binary CurrentFinal.
The audit chain tested, in controlled single-variable experiments, the
five most plausible root causes.

## Verdict table

| Branch | Method | Result | Implication |
|---|---|---|---|
| **H0-A Teacher budget** | 500-corpus drift diagnosis + 64k pilot | 16k↔D20 drift 96.8cp (clamped; 133.5 unclamped), concentrated in endgames (zero-phase 292cp). 64k (4× compute) closes only 18.2% overall — low 5.4% / zero 13.6% | Teacher-budget scaling is not an economical fix; full 64k relabel rejected, 256k/1M sweeps skipped |
| **H0-B Representation collision** | 1M natural census + symmetry-orbit repair + targeted counterfactuals | All 2,814 natural collision groups are the V2 representation's own legal symmetries (H/C/HC); zero unexplained aliasing; global MAE floor 0.087cp. Metadata counterfactuals: EP sharpest (p95 262cp, 5 sign flips), castling moderate (p95 74), rule50 mild | Exact information loss is NOT a major MAE source. EP (none + 8 files categorical) retained as a future feature candidate |
| **H0-C Deployment distribution shift** | 128-root eval-site capture, 1,024 sites vs 5 matched controls, robustness repair | Search-site MAE 194–202cp vs phase/material/STM-matched ordinary 155–164cp; R = 1.18–1.25 across 5 controls; min delta +30.1cp; not driven by outlier roots | The shift is real and robust. Evaluator calls see an out-of-distribution manifold |
| **H0-C2 Search-site augmentation** | 256 train roots → 20k unique sites, paired 120k causal training (identical core/distribution/recipe) | B fits its own 20k training sites to 112.0cp (A: 242.7) yet transfers −1.6cp to held-out search sites; ordinary validation +3.3cp worse. Gate: FAIL | Not an absorption failure — the search-site manifold is highly heterogeneous; 20k sites from 256 roots do not generalize to new trees. Root DIVERSITY, not per-tree density, is the scarce resource |
| **H0-D Search/eval calibration** | Feature-gated bidirectional shadow instrumentation at the exact production predicate sites (futility move-gate, both qsearch stand-pat sites); 128 roots × 2 arms × 50k nodes | Gate decision disagreement 13.7–17.7% (hard 5.2–9.3%), median slack delta 108–130cp; D1 rates shift (futility prune/eval −24%, standpat cut +9.9pp, aspiration fails ~2×) | A real, systematic calibration mismatch — HCE-tuned selectivity interprets the NNUE score distribution differently |
| **H0-D3 Futility causal probe** | Single additive K*=+75cp derived offline from D2 records (median per-root optimum); replay gate passed (−28.1%/−21.7% disagreement both trees); new profile, 500-position × 100k-node paired bestmove vs YD20 | 230 both / 4 A-only / 1 B-only / 265 neither; NET −3. Gate: FAIL, no Arena | Significantly repaired gate DECISIONS did not repair move quality. "Large mismatch ≠ strength cause" — at least for futility-only, at this budget |
| **H0-E Decision quality** | 256 search-site parents (64/phase), ALL 6,142 legal siblings SF-32k-constrained (~250M nodes, 1.5% of E2 compute); static + 100k-search regret; teacher-gap stratification | STATIC: regret 224.4→210.1 (−14.3cp), pairwise 48.8→58.1% (+9.3pp, n=75,818), acc@20 37.5→42.6%. SEARCH: regret 77.6→69.6 (−8.0cp), acc@20 62.5→67.2%, top-1 48.8→53.1%. Gains consistent across phases; largest regret gains at teacher-gap ≥100 (+31cp) | Scalar-CP improvement DOES convert to better sibling ranking, but only partially; full search further attenuates it (~half). MIXED per frozen gates, tilted positive |

## Core conclusion (frozen wording)

> The remaining NNUE strength deficit cannot be attributed to a single
> failure point. Scalar CP accuracy improvements DO convert into better
> sibling move-ranking, but the conversion is partial; the complete search
> further compresses the evaluator advantage. Meanwhile teacher budget,
> exact representation collisions, naive search-site augmentation, and
> futility-only calibration each failed as a sufficient single-variable
> explanation.

Explicitly NOT claimed: any quantitative decomposition of the Elo gap
by multiplying the observed attenuation ratios (different statistics,
different teachers — the H0-E sibling teacher is SF-32k-constrained;
the scalar-MAE teacher is Y16 single-position).

## Evidence-chain notes

* The H0-E corpus is NOT dominated by near-ties (teacher-gap median 33cp;
  66/256 parents gap ≥100), so D3's 265/500 "neither matches" was partly
  — but only partly — the over-strict exact-match metric.
* C2's postmortem rejects absorption failure on SEEN sites (112cp fit)
  but does not establish capacity sufficiency for generalization over
  the search-site manifold.
* D0 (free): on the same 1,024 search sites, F128 aligns to the Y16
  teacher 66.4cp BETTER than CurrentFinal HCE (202.1 vs 268.5, n=984
  CP-comparable) — scoped to teacher alignment, not true-strength
  superiority.
* All instrumentation is cargo-feature-gated
  (diagnostic_eval_site_capture, diagnostic_search_calibration);
  production builds carry zero diagnostic code; the G1-D1 server binary
  was never touched.

## Concurrent experiment (orthogonal)

G1-D1 (FT256-vs-FT128 formal Arena) remains running. Its terminal
result — either direction — does not affect any H0 conclusion and will
be interpreted separately.

## STOP / HOLD list (frozen until new causal evidence)

```
- teacher budget expansion (64k/256k/1M relabeling)
- blind 2M/3M Lichess scalar positions
- FT384/FT512 width fishing
- per-gate margin tuning (futility done — negative; qsearch/null/LMR/aspiration HOLD)
- same-root search-site density augmentation
```

## NEXT (frozen)

```
S10-I1  Task-Aligned NNUE Training

I1-A  Sibling-ranking auxiliary objective pilot
      L = L_cp(material-residual scalar head, unchanged)
        + λ_rank * L_sibling_rank
        (teacher-gap-weighted pairwise logistic over siblings;
         parent's children scored by the SAME position evaluator:
         score(move) = -V(child))

      New sibling corpus: ~10,000 parents,
        50% ordinary game positions / 50% Eureka search-site parents,
        search-site half drawn from THOUSANDS of distinct roots
        (max 1–2 parents per root — root diversity over tree density),
        top-K=8 siblings per parent, SF-32k constrained, frozen once.

HOLD:  WDL auxiliary, EP feature, phase heads, further calibration,
       further width scaling — each behind I1-A's result.
```

## Guiding principle (the one-line takeaway)

> The next-generation objective is no longer to grind single-position
> CP MAE from 137 toward 135, but to improve the model's ability to
> RANK the moves that matter at a decision point — and to guarantee that
> ability generalizes to new search trees through high-root-diversity
> data.

## Artifact index

```
H0-A  results/s10/s10-h0-a-teacher-budget-closeout.md
      results/s10/s10-h0-a1-audit.json, s10-h0-a2-pilot.json
H0-B  results/s10/s10-h0-b1-census.json, s10-h0-b2-probe.json,
      s10-h0-b-repair1.json
H0-C  results/s10/s10-h0-c-shift.json, s10-h0-c-repair1.json
H0-C2 results/s10/s10-h0-c2-pilot.json, s10-h0-c2-postmortem.json
H0-D  results/s10/s10-h0-d0-hce-baseline.json,
      s10-h0-d12-shadow.json,
      s10-h0-d3-replay-gate.json, s10-h0-d3-bestmove.json
H0-E  results/s10/s10-h0-e-sibling-ranking.json
tools tools/s10/h0_*.py (all harnesses, deterministic)
```
