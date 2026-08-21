# S6-N3C Closure and S6-N3D Residual Confirmation Authorization

Written: 2026-08-21. Opened at commit
`0652b5b652591dc5e2fd2001cfe219198692f291` (clean worktree,
`HEAD == origin/main`).

This document CLOSES S6-N3C and AUTHORIZES S6-N3D. It is an independent
record: `results/s6/s6-n3c-generalization-diagnostics.json` and its `.md`
companion are NOT modified, amended, or deleted by this closure. Their
recorded verdict field remains the literal string `CLOUD_VERDICT_PENDING`
as written at run time; the cloud ruling below is the authority that
supersedes that pending state, and it lives here, not by rewriting the
measurement record.

## 1. Cloud verdict on S6-N3C

```text
ABSOLUTE_REPLACEMENT_REJECTED
RESIDUAL_PATH_PROMISING
INDEPENDENT_CONFIRMATION_REQUIRED
```

### 1.1 Evidence the ruling rests on

All values are from the unmodified N3C record (run git
`9ecc87fe95c03c02b56d88acaaa83f54001b784b`, diagnostics script sha256
`591d4f31bb094664f2a25f128bbf3dcd6fdba075759c3517890d92a2f1dbb01a`).

`ABSOLUTE_REPLACEMENT_REJECTED` — an NNUE probe predicting the absolute
teacher score is worse than the existing classical evaluation on the new
distribution, and the three benign explanations were eliminated:

| control | finding | signal |
|---|---|---|
| A | current trainer replays the frozen N1 checkpoint bit-exactly (state tensors `exact_equal=True`, prediction max abs delta 0.0 cp, holdout MAE 141.586 == 141.586) | `TRAINER_REGRESSION=false` |
| C | no narrower mixed-family width clears +5% / 2-of-3 seeds; validation selects width 32 | `OVERPARAMETERIZED_SPARSE_TABLE=false` |
| D | family-isolated training is WORSE than mixed (arena +2.8..+9.7 cp, lichess +4.7..+9.5 cp) | `FAMILY_INTERFERENCE=false` |
| B | on identity-filtered N3B rows both NNUE generations trail classical by far more than 5% (validation: classical 167.907 vs new_n3b 247.528 vs old_n1 264.925; holdout: classical 170.287 vs 245.779 vs 263.736) | `DISTRIBUTION_REPRESENTATION_GAP=true` |

`RESIDUAL_PATH_PROMISING` — control E replaced the absolute target with the
classical residual and beat classical on BOTH splits at both tested widths:

| width | residual val MAE | residual holdout MAE | classical val | classical holdout |
|---:|---:|---:|---:|---:|
| 8 | 159.993 | 157.580 | 167.588 | 170.036 |
| 16 | 159.507 | 158.938 | 167.588 | 170.036 |

Validation-selected width is 16 (159.507 < 159.993). Per-seed width-16
validation MAE: 159.685 / 159.713 / 159.123; holdout 159.071 / 158.964 /
158.779. Improvement at width 16: ~4.8% validation, ~6.5% holdout.

`INDEPENDENT_CONFIRMATION_REQUIRED` — the control-E numbers were produced on
the SAME games, the same sampling, the same teacher pass, and the same
classical cache that the residual target itself was derived from. Every
split in that measurement descends from the N3B game pool, so validation and
holdout there are not independent evidence about unseen games. Nothing may be
promoted on that basis. The N3D confirmation below is the only measurement
allowed to speak to generalization.

### 1.2 One honest caveat carried forward

Every control-E run early-stopped at `best_epoch = 1` (patience 15, so
epochs 2..16 all failed to improve validation loss). The residual gain is
therefore achieved by a barely-trained model and may be close to a learned
shrinkage of the classical score rather than a rich correction. This is NOT
a reason to retune anything — optimizer, schedule, width, and seed are frozen
below — but the confirmation report must not describe the result as evidence
of a deep learned correction.

## 2. Frozen S6-N3D candidate

Every field below is FROZEN. No tuning, no re-selection, no substitution.

| item | frozen value |
|---|---|
| architecture | 40960 x 16 shared feature table, shared 16-dim accumulator bias, ReLU via `clamp-min(0)`, head `concat(own_accumulator, opponent_accumulator)` = 32 -> 1 linear |
| width | 16 |
| seed | 20260818 |
| target | `clamp(teacher_cp_stm - base_eval_stm, -2000, 2000) / 1000` |
| inference | `base_eval_stm + residual * 1000` |
| optimizer | AdamW, lr 1e-3, weight decay 1e-5 (unchanged from N1/N3B) |
| loss | SmoothL1, beta 0.1 (unchanged) |
| batch / epochs / patience | 256 / 100 / 15 (unchanged) |
| device | CPU only; CUDA is forbidden |

Reused, already-validated SHAs (N3C values; none of these is recomputed
from a new source):

```text
N3B dataset       s6-eval-v1-multisource-pilot01
  dataset_sha256  5501240e9fd30414cde204038ea0b1e94d20f0029cbeb796d69885375a0683af
  labels_sha256   e6f036f426db8a5fffc6c28baa6ae5333b0fe441bd9eec13f56d4dda989896d9
teacher           Stockfish 18, binary sha256
                  6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9
                  Threads=1, Hash=64, MultiPV=1, UCI_ShowWDL=true, go nodes 16384
engine binary     target/release/eureka
                  05b822b49940a74019b497c123c9085f27a1bf4cf472e05dabf22a5d533d8c66
classical cache   results/s6/s6-n3c-classical-cache.json
                  c40a38ab4796e0aca68131c17a713a3a31ab9834c741c9221a7a8d1317cf5727
                  (header binds the N3B dataset SHA + the engine binary SHA;
                   20542 usable positions)
```

### 2.1 Explicitly disallowed

- Re-selecting width 8 on the strength of its better N3C holdout mean
  (157.580 vs 158.938). Validation selected 16; holdout was not and is not a
  selection surface. Width 8 stays unused.
- Changing width, seed, optimizer, target definition, or any threshold.
- Expanding or relabeling the N3B dataset.
- Any runtime / search / UCI / exporter work.
- CUDA.

## 3. Frozen S6-N3D confirmation gates

These gates are fixed HERE and in
`tools/s6/run_n3d_residual_confirmation.py` BEFORE any confirmation label or
metric exists. The confirmation is a single run against a disk-loaded
canonical checkpoint: no training, no width or seed re-selection, no repeat
attempts.

Let `classical = base_eval_stm` and
`residual_prediction = base_eval_stm + residual * 1000`. Clipped metrics
clamp BOTH the prediction and the raw teacher target to +-2000 cp, exactly as
`train_nnue_probe.clipped_metrics` already does.

| # | gate | threshold |
|---:|---|---|
| 1 | overall clipped MAE improvement of residual over classical | `>= 2%` |
| 2 | paired per-position absolute-error delta `abs(residual_prediction - teacher) - abs(classical - teacher)`, deterministic paired bootstrap, NumPy `default_rng(20260821)`, 10000 resamples | 95% CI **upper** bound `< 0` cp |
| 3 | residual clipped RMSE vs classical clipped RMSE | `residual <= classical` |
| 4 | per phase bucket with `n >= 200`, residual MAE vs classical MAE | not worse by more than `2%` |
| 5 | per `abs(teacher CP)` bucket with `n >= 200`, residual MAE vs classical MAE | not worse by more than `2%` |
| 6 | all reported metrics finite; checkpoint disk roundtrip passes; identity-overlap and classical-cache gates pass | all must hold |

Gate 2 operates on the SAME clipped pair as gate 1, so
`mean(delta) == residual_clipped_MAE - classical_clipped_MAE` by
construction. Buckets follow the existing frozen definitions: phase
`high 18-24 / mid 8-17 / low 1-7 / zero 0`, and `abs(teacher CP)`
`0-100 / 100-300 / 300-1000 / 1000-inf` assigned by RAW target magnitude.

Outcome mapping — there is no third branch and no retry:

- any gate fails -> status `CONFIRMATION_FAIL`, process return code 2. Do NOT
  change the seed, the width, the thresholds, or the selected games. The NNUE
  residual sub-branch ends.
- all gates pass -> status `RESIDUAL_CONFIRMATION_PASS`, verdict
  `RESIDUAL_CONFIRMED_AWAITING_RUNTIME_REVIEW`. The next round may then begin
  bench-only residual artifact parity/cost work — and nothing beyond that
  without a further ruling.

## 4. Confirmation data contract

The confirmation source must consist of games that no N3B model has seen.

| item | value |
|---|---|
| archive | local, already-verified `lichess_db_standard_rated_2026-07.pgn.zst`, official sha256 `68738b1c448f051dc8d42db645d5b01749988a3bc1c24981adfe44ea92060dc7` — reused, never re-downloaded |
| selection seed | `CONFIRM_SEED = 20260821` |
| games | 1000, selected after excluding the 2000 games already used by N3B |
| exclusion key | deterministic game fingerprint: sha256 of canonical JSON over EXACTLY `initial_fen` (standard initial position when the PGN carries no `FEN` header), `result`, and the mainline UCI move list |
| source id | `lichess-standard-rated-confirm-v1` |
| source family | `lichess-standard-rated-v1` |
| dataset | `s6-eval-v1-residual-confirm01`, sampling v2, existing `build_dataset` contract |
| dataset role | EVALUATION ONLY — no confirmation record may enter training or early stopping |

Streaming discipline: memory stays O(1000) (fingerprint sets and counters
only, never a list of parsed games), a fresh `StringExporter` is created per
written game, the archive is read by exactly one sequential reader, and after
selection stops the remaining compressed bytes are still drained through the
`HashingReader` so the official archive SHA is verified over the WHOLE file.

Identity audit against the FULL N3B dataset, all four required before any
inference:

1. source game fingerprint intersection == 0;
2. exact `position_id` overlap removed before any inference;
3. selection reads neither teacher labels nor model predictions;
4. eligible usable positions `>= 5000` and position `retained_fraction >= 0.90`.

Teacher labeling for the confirmation dataset uses the same Stockfish 18
binary SHA `6b087694...` with `Threads=1`, `Hash=64`, `MultiPV=1`,
`go nodes 16384`, a fresh-second-pass audit of 1000/1000 with 0 mismatches,
and a labeled `verify_dataset` pass. The N3B dataset is NOT relabeled. A NEW
classical cache is built for the confirmation positions, its header binding
the confirmation dataset SHA and the SAME engine binary SHA `05b822b...`;
duplicate, missing, or non-finite values fail closed.

Committed vs local: the confirmation classical cache is committed alongside
the results. The raw confirmation PGN, the confirmation `labels.jsonl`, and
the canonical residual checkpoint remain gitignored, exactly as their N3A/N3B
counterparts are.

## 5. Existing-split reporting is diagnostic only

The canonical checkpoint step also reports classical vs classical+residual on
the EXISTING N3B validation and holdout splits, grouped by family, phase
bucket, and `abs(teacher CP)` bucket, with per-position absolute-error delta
mean, median, and a deterministic paired bootstrap 95% CI.

Those numbers are diagnostic context only. They are NOT a confirmation gate
and cannot promote anything, because they are drawn from the same game pool
the residual target was fitted against. Only the S6-N3D confirmation set,
whose games are fingerprint-disjoint from N3B, can satisfy section 3.

## 6. Stop line

After the confirmation result is recorded, STOP with a clean worktree and
`HEAD == origin/main`. Do not enter residual exporter, Rust runtime, search,
or Arena work. If the confirmation PASSES, the next round begins bench-only
residual artifact parity/cost. If it FAILS, the NNUE residual sub-branch is
over.
