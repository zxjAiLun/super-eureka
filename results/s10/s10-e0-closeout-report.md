# S10-E0 Closeout — Scale-Provenance Generalization + Semantic Audit

**Status: CLOSED / PASS — 1M data-scale probe unblocked**

## What was done

Two hard-coded scale provenance contracts were generalized so the 1M
data-scale probe (S10-E1) can run under the same fail-closed discipline
as the 300k run, plus a one-shot 10k semantic audit to rule out an
STM-perspective bug as the cause of the D1 rejection (12.02% score,
ACCEPT_H0).

## E0-1: teacher contract (tools/s10/train_nnue.py)

`FROZEN_TEACHER_CONTRACT` no longer pins the 300k scale:
`labeled_positions` must equal the dataset manifest's `records_total`
(cross-checked in `load_dataset` against the teacher manifest) and
`labels_sha256` is verified against the actual `labels.jsonl` bytes on
disk. The frozen artifact identity (engine, binary SHA, nodes, options,
fresh-second-pass audit) is unchanged. The old 300k labels SHA is kept
as a documented historical reference.

## E0-2: format-contract loader (src/engine/nnue_v2q_runtime.rs)

The loader no longer requires the artifact header's source SHAs to equal
the B3 checkpoint (`d59ad852…`) / B4 FP32 artifact (`9bf7addd…`). It
still fully validates magic / version / inputs=22528 / ft_width=128 /
scales / quant shifts and recomputes the proven i32 MAC bounds from the
payload; the header SHAs are preserved for provenance output. Model
IDENTITY is now enforced by the consumer — the Arena D0 immutable
model-artifact SHA gate pins the exact bytes a tournament may launch
with. This is not a runtime optimization: the same binary can now load
different training iterations of the same architecture.

## E0-3: bit-exact regression

The frozen `b51a79b1…` model was probed over the same deterministic
10k validation sample (seed 2026083003) with the PRE-change binary
(server build `20260829-9ef078f-linux-x86_64`) and the POST-change
binary: **10,000/10,000 raw outputs bit-exact.**

## E0-4: 10k STM semantic audit (tools/s10/e0_semantic_audit.py)

Deterministic 10k sample from the 300k validation split (seed
2026083003), White-STM n=4921 / Black-STM n=5079:

```
                        teacher vs quantized   teacher vs fp32
stm w   MAE cp              171.013               171.011
        signed bias cp       +13.999               +14.040
        pearson              0.73477               0.734759
        spearman             0.705047              0.705062
        sign agreement       0.7726                0.7718
        pred at clip         0                     0

stm b   MAE cp              162.841               162.835
        signed bias cp        +9.489                +9.524
        pearson              0.755989               0.755996
        spearman             0.717826               0.717818
        sign agreement       0.7694                0.7694
        pred at clip         0                     0

fp32 vs quantized (both colors): MAE ~0.25 cp, pearson 0.999999
```

Verdict: **PASS** — no direction reversal on either color, no
×1000/÷1000 scale error, neither color is broken, no output clipping,
and the quantized model tracks the FP32 model to ~0.25 cp. The D1
rejection is a genuine strength deficit of the 300k recipe, not an STM
sign bug.

Observation (informational only): both colors show a mild positive
signed bias (predictions smaller in magnitude than teacher) and ~77%
sign agreement consistent with the known ~165 cp validation MAE.

## Test evidence

```
cargo test --release --lib            425 passed
tools/s10/test_train_nnue.py           18 passed (contract gates updated
                                        for per-run scale fields)
tools/s10/test_eval_nnue_checkpoint.py 10 passed
```

New/changed tests:
- loader accepts a future-iteration source SHA (and still preserves
  header SHAs for provenance);
- teacher contract accepts a 1M labeled_positions, still requires the
  field's presence/type;
- audit report written to results/s10/s10-e0-semantic-audit.json.
