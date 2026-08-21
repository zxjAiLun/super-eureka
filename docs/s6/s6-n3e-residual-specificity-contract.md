# S6-N3E Residual Specificity Audit - Measurement Contract

Written: 2026-08-21, opened at commit `de09d486e6931e7c215b4b070613da21eb090739`
(clean worktree, `HEAD == origin/main`).

## 1. What this is, and what it is not

This is a **post-confirmation mechanism audit**. It was opened AFTER the
S6-N3D confirmation result was produced, reviewed and accepted, and it is
written with full knowledge of that result. That ordering is disclosed here on
purpose: N3E is NOT a pre-registered prediction about whether the residual
works, it is a targeted attempt to falsify one specific alternative
explanation for a gain that has already been measured.

N3E does NOT rewrite, amend, supersede, or reinterpret any N3D artifact:

- `results/s6/s6-n3d-residual-confirmation.json` / `.md` stay byte-identical.
- `results/s6/s6-n3d-construction-attempt-g1000.json` /
  `-g1400.json` stay byte-identical.
- `results/s6/s6-n3d-classical-cache-confirm01.json` stays byte-identical.
- `results/s6/s6-n3c-classical-cache.json` stays byte-identical.
- The N3D verdict `RESIDUAL_CONFIRMED_AWAITING_RUNTIME_REVIEW` remains the
  standing verdict on the question N3D asked (hybrid vs raw classical on
  unseen games). N3E asks a DIFFERENT question and can only change what is
  authorized NEXT, never what N3D measured.

Cloud review accepted the N3D measurement at `de09d48` but withheld
authorization for residual exporter, Rust runtime, search, UCI and Arena work.

## 2. The alternative explanation being tested

N3D established that `classical + residual` beats raw classical on
fingerprint-disjoint games (clipped MAE 162.142 -> 153.895, -5.09%; paired
95% CI [-9.564, -6.935] cp). It did NOT establish that the gain is
position-specific.

The published N3D record already shows the concern directly. On the 6979
eligible confirmation positions:

```text
classical_prediction_stats  mean -36.915   std 281.1628
hybrid  ("residual")        mean   4.4677  std 282.8293
mean(hybrid) - mean(classical) = +41.3827 cp
std moves only 281.1628 -> 282.8293
```

So the entire measured improvement is consistent with the NNUE having learned
a nearly CONSTANT global correction of about **+41.38 cp** - i.e. the
discovery that this engine's classical evaluation is systematically pessimistic
relative to a 16384-node Stockfish 18 teacher. If that is what happened, the
same gain is obtainable with a single added constant, at zero runtime cost, and
no NNUE runtime work is justified.

This suspicion is reinforced by a fact already recorded in the N3C closure and
the N3D result: every residual run early-stopped at `best_epoch = 1`, so the
model is barely trained and a near-constant output is exactly what one would
expect.

N3E therefore compares the NNUE hybrid against a ladder of deliberately cheap,
parameter-poor calibrators. The NNUE must beat the best cheap calibrator, not
merely beat raw classical.

## 3. Frozen inputs

Nothing below is refitted, reselected, relabelled or regenerated.

| item | frozen value |
|---|---|
| N3B dataset | `s6-eval-v1-multisource-pilot01`, dataset_sha256 `5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af` |
| N3B labels | labels_sha256 `e6f036f426db8a5fffc6c28baa6ae5333b0fe441bd9eec13f56d4dda989896d9` |
| N3C classical cache | `results/s6/s6-n3c-classical-cache.json` sha256 `c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727` |
| N3D confirmation dataset | `s6-eval-v1-residual-confirm01-g1400`, dataset_sha256 `3deff6a4a5cbafcdceb02b2b2c3d06ea0cd061e127cb66f24be4d2bc81d2c43d` |
| N3D confirmation labels | labels_sha256 `e1c25844fd785d46625f6a2a24edaa1a2e8fbd2863f57edfc3f3769723e8edfb` |
| N3D confirmation cache | `results/s6/s6-n3d-classical-cache-confirm01.json` sha256 `126f7c82a5dfb29dbb4750b6a979da16652dca2bd850d1b8551c9361b3e5b169` |
| canonical checkpoint | `data/s6/models/s6-n3d-residual-w16-s20260818.pt` sha256 `5033d47cb101d96057e13aae9d3819d48fa8079e90bda8eae8cd935ac1006c55` |
| engine binary | `target/release/eureka` sha256 `05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66` |
| teacher | Stockfish 18 `6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9`, Threads=1 Hash=64 MultiPV=1 nodes=16384 |
| device | CPU only, `torch.set_num_threads(1)`, CUDA forbidden |

### 3.1 Explicitly forbidden in N3E

- Retraining or fine-tuning the NNUE in any form.
- Re-selecting width or seed (stays width 16, seed 20260818).
- Modifying the confirmation dataset, its labels, or the teacher pass.
- Re-invoking the engine to recompute `base_eval_stm`; the two committed
  classical caches are the ONLY source of base evaluations, and each is
  validated against its bound dataset SHA and engine binary SHA before use.
- Regenerating or editing any existing N3D JSON/Markdown artifact.

## 4. Cheap calibration candidates

Let `x = base_eval_stm / 1000` and the fitting target
`z = clamp(teacher_cp_stm - base_eval_stm, -2000, 2000) / 1000`. Every
calibrator predicts a correction in the same scaled units; cp is
`correction * 1000`, and the calibrated prediction is
`base_eval_stm + correction * 1000` - structurally identical to how the NNUE
hybrid is formed, so the comparison is apples to apples.

| candidate | form | parameters |
|---|---|---:|
| `identity` | `correction = 0` | 0 |
| `global_bias` | `correction = b` | 1 |
| `global_affine` | `correction = u*x + b` | 2 |
| `phase_bias` | `correction = b[phase]`, phase in high/mid/low/zero | 4 |
| `phase_affine` | `correction = u[phase]*x + b[phase]` | 8 |

Fitting protocol, fixed:

- CPU, float64, full-batch `torch.optim.LBFGS`, zero initialization,
  `SmoothL1` with `beta = 0.1`, `max_iter = 500`,
  `line_search_fn = "strong_wolfe"`. No random number generation anywhere.
- Parameters are fitted on the **N3B train split only**.
- The candidate is selected on **N3B validation SmoothL1 only**. On a tie
  within `1e-9`, the candidate with FEWER parameters wins, resolved in the
  order `identity -> global_bias -> global_affine -> phase_bias ->
  phase_affine`.
- N3B holdout and the N3D confirmation set never take part in fitting or
  selection.
- A missing phase bucket in the fitting data, or any non-finite fitted
  parameter, fails closed.

## 5. Verdict gates (frozen before the run)

All of the following must hold to emit
`POSITION_SPECIFIC_GAIN_SUPPORTED` / `BENCH_ONLY_RUNTIME_AUTHORIZED`.
`calibrator` below means the validation-selected calibrator.

| # | gate | threshold |
|---:|---|---|
| 1 | NNUE hybrid clipped MAE vs calibrator on N3B validation AND N3B holdout | hybrid not worse on either |
| 2 | NNUE hybrid clipped MAE vs calibrator on the N3D eligible confirmation set | `>= 2%` improvement |
| 3 | paired per-position delta `abs(hybrid - teacher) - abs(calibrator - teacher)`, `numpy.random.default_rng(20260821)`, 10000 bootstrap resamples | 95% CI **upper** bound `< 0` cp |
| 4 | hybrid clipped RMSE vs calibrator clipped RMSE (confirmation set) | `hybrid <= calibrator` |
| 5 | per phase bucket with `n >= 200`, hybrid vs calibrator clipped MAE | not worse by more than `2%` |
| 6 | per raw `abs(teacher CP)` bucket with `n >= 200`, hybrid vs calibrator clipped MAE | not worse by more than `2%` |
| 7 | provenance, classical-cache, checkpoint and finiteness gates | all must pass |

Clipped metrics clamp BOTH prediction and raw teacher target to +-2000 cp,
exactly as `train_nnue_probe.clipped_metrics` already does, so gate 3's mean
delta equals `hybrid_clipped_MAE - calibrator_clipped_MAE` by construction.
Buckets use the existing frozen definitions: phase `high 18-24 / mid 8-17 /
low 1-7 / zero 0`, and `abs(teacher CP)` `0-100 / 100-300 / 300-1000 /
1000-inf` assigned by RAW target magnitude.

Outcome mapping - two branches only, no retry and no retuning:

- all gates pass -> status `POSITION_SPECIFIC_GAIN_SUPPORTED`, authorization
  `BENCH_ONLY_RUNTIME_AUTHORIZED`. N3F bench-only artifact parity/cost work may
  proceed in the same round, and nothing beyond bench.
- any gate fails -> status `CHEAP_CALIBRATION_SUFFICIENT`, authorization
  `NNUE_RUNTIME_NOT_AUTHORIZED`, return code 2. The NNUE residual runtime
  sub-branch CLOSES. No exporter and no Rust runtime file may be created; the
  selected calibrator's parameters and effect are recorded instead, and the
  production baseline `bde9085` stays untouched.

## 6. Reported quantities

For each of N3B validation, N3B holdout, and the N3D eligible confirmation
set: clipped MAE and clipped RMSE for raw classical, for every calibrator
candidate, for the validation-selected calibrator, and for the NNUE hybrid.

Additionally, to characterise the mechanism directly:

- `correction = hybrid - classical` in cp: mean, std, p10, p50, p90. A
  near-zero std is the signature of the cheap-global-shift explanation.
- Pearson correlation between `correction` and `classical`.
- the fraction of the original NNUE gain over raw classical that the selected
  calibrator already captures.

Prediction-statistics fields are named `classical_prediction_stats`,
`correction_prediction_stats` and `hybrid_prediction_stats`. The old
`residual_prediction_stats` name is retired: it was ambiguous between the
scaled residual output and the final hybrid prediction, and N3E needs the two
separated.

## 7. Run discipline

Code is committed first; then exactly ONE official run. Provenance requires a
clean tracked worktree and index, the trainer blob and the runner's own blob
byte-identical to `HEAD`, and the canonical checkpoint SHA verified BEFORE the
torch payload is deserialized.
