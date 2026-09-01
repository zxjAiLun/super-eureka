#!/usr/bin/env pwsh
# S10-E3: 1M material-residual training launcher (FROZEN until E2 gates PASS).
#
# F1 recipe verbatim — the ONLY change is the dataset:
#   dataset      s10-eval-v2-1m01 (labels SHA verified by the trainer)
#   target_mode  material-residual
#   architecture 22528 -> FT128 -> 256 -> 32 -> 32 -> 1, ClippedReLU
#   AdamW lr 1e-3 wd 1e-5, SmoothL1 beta 0.1, batch 256,
#   max epochs 100, patience 15
#   seeds 20260818 / 20260819 / 20260820
#
# Gates (frozen vs F1 baselines):
#   3-seed median validation composed MAE < 149.83
#   selected validation                 < 149.40
#   winner holdout                      < 149.86, ratio <= 1.15
#   (any fail -> STOP; no quant, no Arena; capacity experiment next)
#
# USAGE (only after E2 Gate A + Gate B PASS):
#   pwsh tools/s10/e3_train_1m.ps1            # all three seeds
#   pwsh tools/s10/e3_train_1m.ps1 20260818   # single seed

param(
    [Parameter(Position = 0)]
    [int[]]$Seeds = @(20260818, 20260819, 20260820)
)

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\..\.."

$dataset = "data/s10/s10-eval-v2-1m01"
$engine = "target/release/eureka.exe"

# FAIL-CLOSED preconditions: E2 artifacts must exist and be frozen.
#
# MEMORY NOTE: this machine has 16GB RAM / 4GB VRAM. The 1M run peaks at
# ~3.5-4GB host memory (records + feature export + three EncodedSplits).
# Run ONLY after the E2 labeling process has exited (it holds ~1GB+ for
# 13h); never concurrently with other agents' builds. VRAM is NOT the
# constraint (embedding_bag keeps the on-GPU working set tiny).
foreach ($f in @(
    "$dataset/labels.jsonl",
    "$dataset/teacher_manifest.json"
)) {
    if (-not (Test-Path $f)) {
        Write-Error "FAIL CLOSED: $f missing (E2 not complete?)"
        exit 1
    }
}

# The authoritative 1M dataset SHA (E1 Repair 1, commit 0a53ae5).
$expected = "1504803a179e7601149862aa47ba6d3ed77977d976397bc780ec92d0d8880560"
$manifest = Get-Content "$dataset/dataset_manifest.json" | ConvertFrom-Json
if ($manifest.dataset_sha256 -ne $expected) {
    Write-Error "FAIL CLOSED: dataset SHA $($manifest.dataset_sha256) != $expected"
    exit 1
}

foreach ($seed in $Seeds) {
    Write-Host "=== E3 seed $seed ==="
    python tools/s10/train_nnue.py `
        --dataset $dataset `
        --engine $engine `
        --feature-set v2 `
        --seed $seed `
        --output "data/s10/e3/seed-$seed" `
        --target-mode material-residual
    if ($LASTEXITCODE -ne 0) {
        Write-Error "seed $seed FAILED"
        exit $LASTEXITCODE
    }
}

Write-Host "=== E3 training complete; report composed MAEs per seed ==="
foreach ($seed in $Seeds) {
    $s = Get-Content "data/s10/e3/seed-$seed/training_summary_v2_s$seed.json" | ConvertFrom-Json
    Write-Host ("seed {0}: best_epoch={1} composed_val_MAE={2}" -f $seed, $s.training.best_epoch, [math]::Round($s.training.best_val_mae, 3))
}
