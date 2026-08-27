import json
import statistics
from pathlib import Path

def analyze_replay():
    replay_dir = Path("results/s10/deterministic_replay")
    v1_runs = [json.loads((replay_dir / f"training_summary_v1_s{seed}.json").read_text()) for seed in [20260818, 20260819, 20260820]]
    v2_runs = [json.loads((replay_dir / f"training_summary_v2_s{seed}.json").read_text()) for seed in [20260818, 20260819, 20260820]]

    v1_val_maes = [r["training"]["best_val_mae"] for r in v1_runs]
    v2_val_maes = [r["training"]["best_val_mae"] for r in v2_runs]

    v1_val_losses = [r["training"]["best_val_loss"] for r in v1_runs]
    v2_val_losses = [r["training"]["best_val_loss"] for r in v2_runs]

    print("=== Deterministic Replay Validation MAE (cp) ===")
    for seed, v1_m, v2_m in zip([20260818, 20260819, 20260820], v1_val_maes, v2_val_maes):
        print(f"Seed {seed}: V1 = {v1_m:.4f} cp | V2 = {v2_m:.4f} cp")

    v1_med = statistics.median(v1_val_maes)
    v2_med = statistics.median(v2_val_maes)
    ratio = v2_med / v1_med
    delta_pct = (ratio - 1.0) * 100

    print(f"\nV1 Median Val MAE: {v1_med:.4f} cp")
    print(f"V2 Median Val MAE: {v2_med:.4f} cp")
    print(f"V2 / V1 Ratio: {ratio:.5f} (Delta: {delta_pct:+.3f}%, Threshold: <= +2.00%)")
    print(f"Primary Gate Passed: {ratio <= 1.02}")

if __name__ == "__main__":
    analyze_replay()
