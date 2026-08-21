# S6-N3E Closure and S6-C1 Authorization

Written: 2026-08-21, opened at commit
`a49b3924ac27bd8086e89120d87ee9819420053a`.

## 1. Cloud verdict on S6-N3E

```text
N3E_MEASUREMENT_ACCEPTED
NNUE_RESIDUAL_RUNTIME_REJECTED
POSITION_SPECIFIC_INCREMENT_NOT_DEMONSTRATED
PHASE_AFFINE_CLASSICAL_CANDIDATE_AUTHORIZED
```

`NNUE_RUNTIME_NOT_AUTHORIZED` stands. On the game-disjoint confirmation set the
NNUE hybrid improved on the 8-parameter calibrator by only 0.78% clipped MAE,
the paired bootstrap CI crossed zero (`[-2.495, +0.074]` cp), RMSE was worse
(218.473 vs 215.082), and the `|teacher CP| = 0-100` bucket lagged badly
(86.746 vs 77.686, +11.66%). That is not enough to move a 655k-parameter table
into the Rust runtime, and nowhere near enough for search/UCI or the production
baseline.

### 1.1 Correction to an overstated claim

An earlier summary said N3E "proved the NNUE correction has no position
structure". That is too strong and is corrected here.

`corr(correction, classical) = -0.046` only rules out an obvious LINEAR
relationship between the correction and the classical score. It does not show
the correction is structureless in general. The NNUE still scored 1.208 cp lower
clipped MAE than the calibrator on the confirmation set - it simply failed to
clear the pre-registered statistical and practical thresholds.

The accurate statement is: **position-specific gain was not demonstrated, and
what gain exists does not justify the NNUE's complexity and runtime cost.**

N3D is NOT retracted. N3D asked whether the hybrid beats RAW classical on
unseen games; the answer remains yes. N3E shows that the large majority of that
N3D gain is explainable by cheap phase calibration.

## 2. P2 provenance correction (semantics only, no re-run)

The historical N3E result records `bindings.engine_invoked = false`. That field
is too broad and is hereby clarified rather than rewritten.

What actually happened: the runner calls `diag.load_prepared()`, which invokes
the engine's `bench nnue-features-batch` to export NNUE feature indices. What is
genuinely true is that the engine was NEVER invoked to recompute base
evaluations - every `base_eval_stm` came from the two committed, SHA-validated
classical caches.

- The measurement is NOT contaminated: the engine binary SHA
  `05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66` is bound in
  the result and covers both uses.
- The frozen contract was NOT violated: it forbade recomputing base
  evaluations, which did not happen.
- `results/s6/s6-n3e-residual-specificity.json` and `.md` are NOT re-run,
  regenerated, or edited. They keep the coarse field exactly as recorded; this
  document is the authority on how to read it.

Future runs of `tools/s6/run_n3e_residual_specificity.py` emit the split form
instead:

```json
"engine_invocation": {
  "nnue_feature_export": true,
  "base_eval_recomputation": false
}
```

## 3. S6-N branch closed

The S6-N NNUE line is closed. No residual exporter, no Rust NNUE runtime, no
NNUE in search/UCI, no NNUE promotion. The existing bench-only NNUE probe
commands remain as historical diagnostics.

## 4. S6-C1 authorization: Phase-Affine Classical Calibration

Authorized as an ISOLATED classical-evaluation candidate profile, for
implementation and Arena testing only.

Rationale: on the game-disjoint confirmation set the phase-affine calibrator
moved classical clipped MAE 162.142 -> 155.103 and clipped RMSE 231.462 ->
215.082, at 8 parameters and negligible runtime cost. That is worth a real
playing test.

### 4.1 Hard limits

- Teacher imitation is NOT Elo. Calibration changes the score scale seen by
  pruning, aspiration windows, null-move, qsearch and TT, so the only claim
  authorized is "implement and send to Arena".
- The candidate must NOT become the default. `CurrentFinal` stays
  `PRODUCTION_PROFILE`; no-argument startup and `current-final` behavior must be
  bit-identical to before.
- Production baseline `bde9085` is untouched.
- N3E JSON/Markdown and the canonical checkpoint stay byte-identical.
- No promotion may be claimed in this repository before Arena cloud review.

### 4.2 Frozen parameters

Taken verbatim from `results/s6/s6-n3e-residual-specificity.json`, selected
calibrator `phase_affine` (fitted on the N3B train split, selected on the N3B
validation split). No retraining, no retuning, no trimming.

Phase order and boundaries are reused from N3E exactly:
`high = 18..24`, `mid = 8..17`, `low = 1..7`, `zero = 0`.

| phase | fitted slope u | factor = 1 + u | fitted bias b | bias cp |
|---|---:|---:|---:|---:|
| high | -0.2613823636 | 0.7386176364 | 0.036717418 | +36.717418 |
| mid | -0.1061941736 | 0.8938058264 | 0.0503747198 | +50.374720 |
| low | -0.0117859984 | 0.9882140016 | 0.0217028031 | +21.702803 |
| zero | +1.5102040829 | 2.5102040829 | -0.000464891 | -0.464891 |

Runtime uses deterministic integer fixed point, scale `1_000_000`, with `i64`
intermediates. Floating point is forbidden in the Rust runtime formula.

```text
factor         = [738618, 893806, 988214, 2510204]
bias_scaled_cp = [36717418, 50374720, 21702803, -464891]

calibrated_cp  = round_symmetric((base_cp * factor + bias_scaled_cp) / 1_000_000)
```

The `zero` bucket slope of 2.51 is aggressive and is deliberately KEPT. It was
fitted on only 538 N3B zero-phase records and is the single most likely source
of Arena regression, but hand-trimming it would break the frozen-parameter
contract and turn a clean test into an unfalsifiable one. Arena decides.

### 4.3 Scope

`SearchProfile::CurrentFinalPhaseAffine` inherits every `CurrentFinal` search
policy bit and differs ONLY in evaluator dispatch. A centralized test compares
the full resolved `SearchFeaturePolicy` plus every search feature boolean
between the two profiles and requires exact equality apart from the evaluator
selector, so the candidate cannot drift from production by copy-paste.

Both profiles are exposed from the SAME frozen binary; Arena receives
`current-final` as baseline and `current-final-phase-affine` as candidate.
