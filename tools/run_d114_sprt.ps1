param(
    [ValidateSet("Smoke", "Sprt")]
    [string]$Mode = "Smoke",
    [string]$CutechessPath = "tools/.cache/cutechess-1.5.1-win64/cutechess-cli.exe",
    [string]$EnginePath = "target/release/chess-engine-demo.exe",
    [string]$OutputDir = "results/d1.14/$($Mode.ToLower())"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Get-Command python -ErrorAction Stop
$arguments = @(
    "tools/run_d114_sprt.py",
    "--mode", $Mode,
    "--cutechess", (Join-Path $repoRoot $CutechessPath),
    "--engine", (Join-Path $repoRoot $EnginePath),
    "--output-dir", (Join-Path $repoRoot $OutputDir)
)

& $python.Source @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
