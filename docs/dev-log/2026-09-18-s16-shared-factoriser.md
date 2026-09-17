# S16 — Training-time shared feature table (shared factoriser)

**Status: CLOSED at PARITY. S14 remains the production net.**

## One-line result

**S16 验证损失改善，但本次初筛没有显示棋力提升。**

`−8.1 Elo` 的 95% 区间 `[−39.0, +22.8]` **跨零**，因此**不能**写成"已确认比 S14 弱"；
只能说**本次初筛没有观察到提升**。

## What was tried

A **training-only** cross-king-position shared feature table, in the
Bullet shared-factoriser style. The V2 feature `i` receives

```text
W_effective[i] = W[i] + F[i % 704]        for i in [0, 22528)
W_effective[i] = W[i]                      for i in [22528, 23296)   (R12 rows)
```

- `W` = existing per-king-position table `[22528, 256]` (upstream V2 table)
- `F` = new shared table `[704, 256]`, `704 = 11 * 64`, `22528 = 32 * 704`
- **+180,224 params (~3% of the base FT table), zero added inference ops**
- Both perspectives share the **same** `F`; no shared bias
- **R12 rows get no shared term**
- FT width stays **256**; export stays a single `[23296, 256]` table

The shared table is **merged into the exported table in float, then quantized
once**, so the deployed model is an ordinary single-table v5 artifact and the
Rust search / incremental update / model format are untouched.

## Result

Screen protocol: **256 games = 128 fresh opening pairs**, `10+0.1`, Hash 16,
1 search thread, concurrency 6, both colours, **same binary and same profile
(`current-final`) — the only difference under test is `--nnue-model`.**

| | candidate | baseline |
|---|---|---|
| net | S16 `7d442e9cc31ab6c8cee3cd625f0ab0cdf54be58844a8e37e10ac789c4a53c35a` | S14 `329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0` |
| engine | `a541e09f2cf42f0ed0be60b9cc4102de8eb2c7c0ffbccf805b5d073c64450a96` | same |

```text
candidate_W 102 / draws 46 / candidate_L 108
score        125 / 256 = 48.83%
Elo          -8.1 +/- 30.9   -> 95% CI [-39.0, +22.8]  (contains 0)
pentanomial   [16, 10, 78, 12, 12]   pairs 128
verdict      PARITY      (<48% FAIL / 48-52% PARITY / >52% PROMISING)
```

Report: `results/s16/screen-report.json`
Openings used: `results/s16/screen-openings.epd` (book lines **321-448**)

### Candidate comparison vs S14 (same opening block 321-448, comparable)

| candidate | training | best_val_loss | score vs S14 | Elo | verdict |
|---|---|---|---|---|---|
| S15-short | 60M pres. / 3 passes | 0.0090439 | 47.07% | −20.4 ± 32.5 | FAIL |
| S15-long | 400M / 20 passes | 0.0090580 | 51.76% | +12.2 ± 36.5 | PARITY |
| **S16 (shared)** | 361M / 18 passes (3600 s wall) | **0.0087879** | **48.83%** | **−8.1 ± 30.9** | **PARITY** |

**S16 has the lowest validation loss of the three but did not gain strength.**
This is the second time `val loss` failed to predict playing strength, so the
standing rule holds: **use canonical validation loss to pick the checkpoint,
use games to decide promotion.** Do not promote on loss.

`−8.1 Elo`'s interval contains zero, and the point estimate is below S14, so the
honest reading is *"no improvement observed in this screen"* — not *"confirmed
weaker."*

## Training

```text
checkpoint   data/s16/run2/checkpoint_s16_v2r12_s20260915.pt
pool         20,967,387 positions (743 shards, reused from S15 — no new data)
seed         20260915
best_step    55500
best_val_loss 0.0087879      best_val_mae 131.76
budget_hit   wall_clock      (3600 s, not presentations)
presentations 361,260,427    (18 passes)
steps        352,794         (cosine aligned to 400M = 390,625 steps; not fully annealed)
```

## Export

```text
artifact  data/s16/run2/nnue-s16-shared-v5.bin
sha256    7d442e9cc31ab6c8cee3cd625f0ab0cdf54be58844a8e37e10ac789c4a53c35a
size      11,936,916 bytes   (v5 / EUNN2Q01 — same size and format as S14/S15 artifacts)
```

- merge happens in **float**, quantisation is applied **once**
- merged float range `[-11589.3, +8605.8]` — **already × 4096**, i.e. in
  quantised units; it maps cleanly into int16 with **zero clipping**
- the export does an **explicit int16 range check that raises** rather than
  silently clipping

`tools/s12/s12_export.py` was **not** modified: it rejects the S16 checkpoint by
schema (it does not know `ft_shared`), and that refusal is correct — allowing it
through would export an **unmerged** model.

## A bug caught, and the guard that caught it

The **first** S16 run is **invalid**.

`ft_shared` is a single `nn.Parameter`, so AdamW updated the **sentinel row
704** every step. That sentinel is what R12 features route to, so for the whole
run **every R12 feature received a spurious shared contribution**. The
measured sentinel row was non-zero on all 256 entries (range −0.674 to +0.161).
Its `val 0.008728` is therefore **not comparable** to S15L and is not evidence
of anything.

- **Direct cause of the drift: R12 features being routed onto that row and
  producing gradient there. AdamW weight decay alone does not turn a zero into
  a non-zero.**
- Caught by the **export-side assertion** that the sentinel row must be exactly
  zero. The export **failed closed**.
- **Fix:** re-pin the sentinel to zero after every `optimizer.step()`, plus a
  save-time assertion. Negative control: without the fix the row drifts to
  `2.97e-02` after 5 steps; with it, `0.000e+00` for the whole run.
- **Lesson: zero-init is not stay-zero.** An invariant like this must be
  enforced on **both** the training side and the export side.

The invalid checkpoint is kept locally, out of Git, renamed
`data/s16/run/INVALID-sentinel-bug_checkpoint_s16_v2r12_s20260915.pt`.

## Verification (all green, with negative controls)

| # | check | result |
|---|---|---|
| 1 | factorised vs merged **float** eval, **on the final trained checkpoint** | **PASS** — `max\|diff\| = 3.87e-07`; shared table non-zero (absmax 2.2214); sample contained **99,792 R12 rows** |
| 2 | Python integer eval vs Rust **exact**, incl. material-imbalanced | **PASS** — **7/7 exact** (K+P, K+R, K+R vs K, bare kings, opening, middlegame) |
| 3 | full vs incremental **bit-exact** + UCI load | **PASS** — `nnue-v2q-r12-parity` `passed:true`, 0 lane / 0 raw mismatches over 120 transitions + 80 incremental transitions + 10 directed fixtures; **field-for-field identical to the S14 control** |

Check 1 must run on the **final** checkpoint: an earlier version of this check
ran before the training loop, when the shared table was still all-zero, so it
could not certify the merge for a non-zero table.

**Negative controls prove check 1 is load-bearing** — deliberately wrong merges
are all caught:

```text
R12 rows also get the shared term      max|diff| = 8.38e-01
period 703 instead of 704              max|diff| = 6.71e-01
repeat 31 times instead of 32          max|diff| = 1.53e-01
```

### Two invocation traps worth recording

1. **`bench nnue-features` defaults to feature set V1** (`bench.rs`), so it
   **must** be given `--feature-set v2r12` for a V2R12 model. Omit it and you
   get indices in a different space (e.g. `35135`) that look like corruption but
   are just the wrong encoder. **Do not patch this with an offset or modulo.**
2. **Rounding happens on the un-shifted `z`**, per
   `src/engine/nnue_v2q_runtime.rs`:
   `raw = (z + d//2)//d if z >= 0 else -((-z + d//2)//d)`.
   Shifting first and then rounding the integer discards the remainder and
   **can never match** the runtime.

### Not a defect: the stale accumulator audit

`bench nnue-v2q-accumulator-audit` reports 100% mismatch on S16 — but it
**fails identically on the S14 production net** (30,708 lane / 120 raw) and on
S15L. Its source walks the **original V2 update path** and cannot validate the
R12 incremental stack. **Use `nnue-v2q-r12-parity`** (check 3), which passes.
Note `"S14 also fails"` is **only** a hint of a shared cause; it is not on its
own proof that the tool is wrong.

## Scope discipline this round

- **No retraining** after the screen. **No new data conversion.** 21M pool and
  caches reused as-is.
- **No promotion.** S14 stays in production.
- Opening block 321-448 **reused** for this regression screen (the earlier
  "every screen needs a brand-new block" requirement was withdrawn by the
  owner). Reused blocks are fine for regime-triaging a candidate; if a
  candidate ever looks promising, widen to independent openings.

## Reproduce

```bash
# 1. train (pool reused; ~1 h wall, stops on 3600 s regardless of presentations)
python tools/s16/s16_train.py --presentations 400000000 --max-seconds 3600 \
    --seed 20260915 --out data/s16/run2

# 2. export (float merge -> single quantisation, int16 range check, sentinel assert)
python tools/s16/s16_export.py \
    --checkpoint data/s16/run2/checkpoint_s16_v2r12_s20260915.pt \
    --out data/s16/run2/nnue-s16-shared-v5.bin

# 3. verification
python tools/s16/s16_verify_merge_final.py \
    data/s16/run2/checkpoint_s16_v2r12_s20260915.pt data/s15/cache/cache-000.ekc 4096
python tools/s16/s16_verify_int.py \
    data/s16/run2/nnue-s16-shared-v5.bin data/s16/fens.txt
./target/release/eureka.exe bench nnue-v2q-r12-parity \
    --model data/s16/run2/nnue-s16-shared-v5.bin --games 3 --plies 40

# 4. screen vs S14 (256 games, openings 321-448)
python tools/s15/s15_screen.py \
    --candidate-art data/s16/run2/nnue-s16-shared-v5.bin \
    --label-cand S16 --out results/s16 --concurrency 6
```

## Where this leaves the line

Across S15 (more data) and S16 (more data + longer training + training-time
shared weights) on the **current FT256 recipe**, **no candidate cleared the
>52% bar**. The next step should be a **structurally stronger change**, not
further tuning of these same knobs — e.g. a **single FT512 attempt**, reusing
the existing data and training pipeline, with deployment support and the real
speed cost confirmed **before** committing a training budget.
