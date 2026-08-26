import json
import statistics
from pathlib import Path

def extract():
    holdout_dir = Path("results/s10/stage2_holdout")
    runs = [json.loads((holdout_dir / f"training_summary_v2_s{seed}.json").read_text()) for seed in [20260818, 20260819, 20260820]]

    holdout_maes = [r["holdout_metrics"]["holdout_mae"] for r in runs]
    holdout_losses = [r["holdout_metrics"]["holdout_loss"] for r in runs]
    val_maes = [r["training"]["best_val_mae"] for r in runs]

    print("=== Stage 2 Confirmatory Holdout Results for V2 ===")
    for seed, v_mae, h_mae, h_loss in zip([20260818, 20260819, 20260820], val_maes, holdout_maes, holdout_losses):
        print(f"Seed {seed}: Val MAE = {v_mae:.3f} cp | Holdout MAE = {h_mae:.3f} cp | Holdout Loss = {h_loss:.6f}")

    print(f"\nHoldout Median MAE: {statistics.median(holdout_maes):.3f} cp")
    print(f"Holdout Mean MAE: {statistics.mean(holdout_maes):.3f} cp")

if __name__ == "__main__":
    extract()
