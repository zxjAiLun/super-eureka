import json
import statistics
from pathlib import Path

def analyze():
    bakeoff_dir = Path("results/s10/bakeoff")
    v1_runs = [json.loads((bakeoff_dir / f"training_summary_v1_s{seed}.json").read_text()) for seed in [20260818, 20260819, 20260820]]
    v2_runs = [json.loads((bakeoff_dir / f"training_summary_v2_s{seed}.json").read_text()) for seed in [20260818, 20260819, 20260820]]

    v1_val_maes = [r["training"]["best_val_mae"] for r in v1_runs]
    v2_val_maes = [r["training"]["best_val_mae"] for r in v2_runs]

    v1_val_losses = [r["training"]["best_val_loss"] for r in v1_runs]
    v2_val_losses = [r["training"]["best_val_loss"] for r in v2_runs]

    print("=== Validation MAE (cp) ===")
    print(f"V1 seeds: {v1_val_maes}")
    print(f"V1 median: {statistics.median(v1_val_maes):.3f} mean: {statistics.mean(v1_val_maes):.3f}")
    print(f"V2 seeds: {v2_val_maes}")
    print(f"V2 median: {statistics.median(v2_val_maes):.3f} mean: {statistics.mean(v2_val_maes):.3f}")
    diff_pct = (statistics.median(v2_val_maes) - statistics.median(v1_val_maes)) / statistics.median(v1_val_maes) * 100
    print(f"Median MAE diff (V2 vs V1): {diff_pct:+.3f}% (threshold is <= +2.0%)")

    print("\n=== Coverage & Diagnostics ===")
    print(f"V1 train unique features: {v1_runs[0]['coverage_diagnostics']['train_observed_unique']}/{v1_runs[0]['architecture']['num_inputs']}")
    print(f"V2 train unique features: {v2_runs[0]['coverage_diagnostics']['train_observed_unique']}/{v2_runs[0]['architecture']['num_inputs']}")
    print(f"V1 val unseen activation rate: {v1_runs[0]['coverage_diagnostics']['val_unseen_activation_rate']*100:.3f}%")
    print(f"V2 val unseen activation rate: {v2_runs[0]['coverage_diagnostics']['val_unseen_activation_rate']*100:.3f}%")
    print(f"V1 val positions with unseen rate: {v1_runs[0]['coverage_diagnostics']['val_positions_with_unseen_rate']*100:.3f}%")
    print(f"V2 val positions with unseen rate: {v2_runs[0]['coverage_diagnostics']['val_positions_with_unseen_rate']*100:.3f}%")

    print("\n=== Parameters & Footprint ===")
    print(f"V1 FT params: {v1_runs[0]['architecture']['ft_param_count']:,} ({v1_runs[0]['architecture']['ft_fp32_mib']:.2f} MiB)")
    print(f"V2 FT params: {v2_runs[0]['architecture']['ft_param_count']:,} ({v2_runs[0]['architecture']['ft_fp32_mib']:.2f} MiB)")
    ft_reduction = (v1_runs[0]['architecture']['ft_param_count'] - v2_runs[0]['architecture']['ft_param_count']) / v1_runs[0]['architecture']['ft_param_count'] * 100
    print(f"FT parameter reduction: -{ft_reduction:.2f}%")

if __name__ == "__main__":
    analyze()
