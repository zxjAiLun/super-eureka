# S18-QC1: NNUE search-boundary experiment (pre-run plan)

> Historical pre-run plan retained unchanged below. Final outcome: mechanism
> diagnosis PASS, cost PASS but expensive, timed screen FAIL (99/42/115),
> candidate KILLED; feature remains OFF. The pinned-d3 `krk-mopup` difference
> was explicitly retained, not converted to a 23/23 pass. See
> [final closeout](../dev-log/2026-09-19-s18-qc1-quiet-checks.md).

## Decision to make

Not another fixture-only closeout: decide whether bounded quiet checks should
advance as an NNUE search candidate, or be rejected on safety/cost; then define
the next search adaptation stage. This is not a model-training experiment.

## Single candidate

Compile-time `experimental_nnue_qchecks` (OFF by default), active only for
CurrentFinal with an NNUE state. Add legal quiet checks at **qply 0 and 1**,
after stand-pat fails high cutoff, alongside existing tactical moves. No
quiet checks at qply >=2. All evasions still searched, current MAX_QPLY/draw/
stop/NNUE-stack rules unchanged. No SEE filter on new quiet checks. Use existing
full legal generator and make/unmake check test first: wall overhead is part
of acceptance, not hidden by counting only recursive nodes. Do not change
NNUE weights, material scaling, futility/null/LMR or time management this round.

## Same-source arms

A default current source (S14 + AVX2); B same source with just the feature.
Record source state + tracked diff hash + binary/model hashes; no claim of
clean committed source if worktree is dirty. Fresh TT per root, Hash16,
single-thread, full frozen history for the three cloud roots. No deployment.

## Permanent regression

Game127 exact quiet-mate boundary and both legal evasions. Model-independent
synthetic NNUE keeps the semantic test durable without a 12 MB model dependency.
Verify the new move set's budget, HCE/rollback exclusion, legal PV/mate,
board/history restoration and existing terminal/draw/abort regressions.
Baseline retains a labelled known limitation; the feature must resolve it.

## Measurements (frozen before candidate execution)

1. Three cloud roots, fixed depths 5/6/7, max 15s; fixed node caps
   50k/200k/1M, max 15s. Completed iterations only; capped depth != achieved.
2. Game127 boundary d1 and mate endpoint d1.
3. All 23 `tests/data/search_validation.epd` tactical/terminal/EP/promotion/
   sacrifice cases with S14 NNUE at declared depth: report absolute results
   and paired regressions, never blame pre-existing NNUE failures on QC1.
4. All 32 `tests/data/external_validation_v1.epd` at declared depth4 for
   generic cost and legality (NO answer-key/strength claim). Interleave arms;
   measure recursive nodes AND process search wall time. Repeat this fixed
   cost panel 3 times in reversed AB/BA order (not cherry-pick rounds).
5. Default + feature Rust release suites (build jobs <=2); no destructive
   fault injection or tree-wide audit project.

## Predeclared advancement criteria

- Boundary becomes correct at d1; no illegal move/PV/terminal/stack regression.
- No additional failures among the 23 tactical cases relative to baseline;
  absolute pre-existing misses reported separately.
- g127 avoids the known sacrifice earlier/by no greater fixed-node budget;
  g175/g187 are out-of-target transfer observations, not guaranteed fixes.
- Generic 32-position cost: median candidate/base nodes <=1.50 and median
  search wall <=1.50; p90 nodes <=3.0. These are screening thresholds, not
  promises of NPS or Elo. If exceeded, DO NOT adopt/promote broad QC1; diagnose
  cost concentration before a separately declared narrower successor.
- Passing this screen authorizes recommendation for a timed match, not an
  automatic production switch or claimed strength gain.

## Required final output

Cause evidence; bounded candidate pass/fail and cost; what this says about
HCE-era search policies under NNUE; a prioritized next-stage search plan with
adoption/stop gates. Distinguish correctness repair, performance engineering,
and playing-strength evidence. No endless parameter sweep or fixture-only milestone.
