#!/usr/bin/env python3
"""S6-N2 Python<->Rust full-refresh inference parity gate + cost evidence.

Pipeline:
  1. validate the frozen checkpoint SHA and metadata;
  2. export sparse features for every non-null-CP position via the Rust
     `nnue-features-batch` bridge (encoding source of truth stays in Rust);
  3. compute Python predictions from the checkpoint state_dict (float32);
  4. compute Rust predictions via `nnue-probe-batch` (independent full-refresh
     inference from Position);
  5. compare on all 5891 CP rows; run the microbench; assemble
     results/s6/s6-n2-runtime-probe.json.

Hard gate: rows == 5891, NaN/Inf == 0, max |diff| <= 0.1 cp, mean <= 0.01 cp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_nnue_probe as exp  # noqa: E402
import train_nnue_probe as probe  # noqa: E402

EXPECTED_CP_ROWS = 5891
MAX_ABS_DIFF_CP = 0.1
MEAN_ABS_DIFF_CP = 0.01


def index_unique_rust_rows(rows: list[dict], expected: int) -> dict[str, dict]:
    """Validate raw rust rows: exact expected count, then reject any
    duplicate position_id. Returns the id-indexed rows."""
    if len(rows) != expected:
        raise SystemExit(
            f"PIPELINE_FAILURE: rust probe rows {len(rows)} != expected {expected}")
    indexed: dict[str, dict] = {}
    for row in rows:
        pid = row["position_id"]
        if pid in indexed:
            raise SystemExit(
                f"PIPELINE_FAILURE: duplicate position_id in rust probe rows "
                f"{pid[:16]}")
        indexed[pid] = row
    return indexed


def parity_passes(parity: dict) -> bool:
    """Hard gate: exact row count, no NaN/Inf, tight cp bounds."""
    return (
        parity["rows"] == EXPECTED_CP_ROWS
        and parity["nan_inf"] == 0
        and parity["max_abs_diff_cp"] <= MAX_ABS_DIFF_CP
        and parity["mean_abs_diff_cp"] <= MEAN_ABS_DIFF_CP)


def engine_batch(engine: Path, args: list[str]) -> list[dict]:
    proc = subprocess.run([str(engine), *args], capture_output=True, text=True,
                          timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"PIPELINE_FAILURE: {args[0]} exit {proc.returncode}: "
                         f"{proc.stderr[:400]}")
    rows: list[dict] = []
    for line in proc.stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_microbench(engine: Path, artifact: Path, batch: Path, iterations: int) -> dict:
    proc = subprocess.run(
        [str(engine), "bench", "nnue-probe-microbench", "--model",
         str(artifact), "--batch", str(batch),
         "--iterations", str(iterations)],
        capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"PIPELINE_FAILURE: microbench exit {proc.returncode}: "
                         f"{proc.stderr[:400]}")
    rounds: list[dict] = []
    summary: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("nnue_probe_microbench_round "):
            rounds.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("nnue_probe_microbench_summary "):
            summary = json.loads(line.split(" ", 1)[1])
    return {"rounds": rounds, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--microbench-batch", type=Path, required=True)
    ap.add_argument("--microbench-iterations", type=int, default=1000)
    ap.add_argument("--out", type=Path,
                    default=Path("results/s6/s6-n2-runtime-probe.json"))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    engine = args.engine.resolve()

    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    artifact_sha = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    engine_binary_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    git_sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    exp.validate_checkpoint(args.checkpoint)  # frozen SHA + metadata binding

    dataset = probe.load_dataset(args.dataset)
    labels = dataset["labels"]
    cp_records = [r for r in dataset["records"]
                  if labels[r["position_id"]]["teacher_cp_stm"] is not None]
    if len(cp_records) != EXPECTED_CP_ROWS:
        raise SystemExit(
            f"PIPELINE_FAILURE: {len(cp_records)} CP rows != {EXPECTED_CP_ROWS}")

    # ---- Python predictions from the checkpoint (indices via Rust bridge) ----
    exported = probe.export_all_features(engine, dataset["records"])
    split = probe.prepare_split(exported, cp_records, labels)
    ckpt = torch.load(args.checkpoint, weights_only=True)
    model = probe.NnueProbe()
    model.load_state_dict(ckpt["state_dict"])
    py_pred, _ = probe.evaluate_split(model, split)
    py_cp = [p * probe.TARGET_SCALE for p in py_pred.tolist()]

    # ---- Rust predictions (independent full-refresh inference) ----
    with tempfile.TemporaryDirectory(prefix="s6-n2-parity-") as tmp:
        batch_path = Path(tmp) / "batch.txt"
        batch_path.write_text(
            "".join(f"{r['position_id']}|{r['fen']}\n" for r in cp_records),
            encoding="utf-8")
        rust_rows = engine_batch(engine, [
            "bench", "nnue-probe-batch", "--model", str(args.artifact),
            "--batch", str(batch_path)])
    rust_by_id = index_unique_rust_rows(rust_rows, EXPECTED_CP_ROWS)
    if set(rust_by_id) != set(split["pids"]):
        raise SystemExit("PIPELINE_FAILURE: rust probe id set mismatch")

    diffs: list[float] = []
    nan_inf = 0
    sign_mismatch = 0
    for i, pid in enumerate(split["pids"]):
        py_c = py_cp[i]
        rs_c = float(rust_by_id[pid]["prediction_cp"])
        if not (py_c == py_c and rs_c == rs_c) \
                or not (abs(py_c) != float("inf") and abs(rs_c) != float("inf")):
            nan_inf += 1
            continue
        diffs.append(abs(py_c - rs_c))
        if (py_c > 0) != (rs_c > 0):
            sign_mismatch += 1

    diffs_sorted = sorted(diffs)
    def percentile(p: float) -> float:
        if not diffs_sorted:
            return 0.0
        idx = min(len(diffs_sorted) - 1, int(p / 100.0 * len(diffs_sorted)))
        return diffs_sorted[idx]

    parity = {
        "rows": len(diffs),
        "expected_rows": EXPECTED_CP_ROWS,
        "nan_inf": nan_inf,
        "sign_mismatch": sign_mismatch,
        "max_abs_diff_cp": round(max(diffs), 6) if diffs else None,
        "mean_abs_diff_cp": round(statistics.mean(diffs), 6) if diffs else None,
        "p50_abs_diff_cp": round(percentile(50), 6),
        "p95_abs_diff_cp": round(percentile(95), 6),
        "p99_abs_diff_cp": round(percentile(99), 6),
    }
    parity["pass"] = parity_passes(parity)

    def build_result(microbench: dict | None) -> dict:
        return {
            "status": "PARITY_PASS" if parity["pass"] else "PARITY_FAIL",
            "cost_status": "COST_MEASURED" if microbench is not None
            else "NOT_RUN_PARITY_FAIL",
            "gate": {
                "rows_required": EXPECTED_CP_ROWS,
                "max_abs_diff_cp_required": MAX_ABS_DIFF_CP,
                "mean_abs_diff_cp_required": MEAN_ABS_DIFF_CP,
            },
            "hashes": {
                "checkpoint_sha256": checkpoint_sha,
                "artifact_sha256": artifact_sha,
                "engine_binary_sha256": engine_binary_sha,
                "engine_git_sha": git_sha,
                "dataset_sha256": dataset["dataset_sha"],
                "labels_sha256": dataset["labels_sha"],
            },
            "artifact": {
                "path": str(args.artifact),
                "format": exp.MAGIC.decode(),
                "version": exp.FORMAT_VERSION,
                "inputs": exp.EXPECTED_INPUTS,
                "width": exp.EXPECTED_WIDTH,
                "target_scale": exp.TARGET_SCALE,
                "header_bytes": exp.HEADER_BYTES,
                "total_bytes": exp.TOTAL_BYTES,
                "offsets": exp.artifact_offsets(),
            },
            "parity": parity,
            "microbench": microbench,
        }

    if not parity["pass"]:
        # Write the failing record and exit BEFORE any microbench run.
        result = build_result(None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "parity": parity}),
              flush=True)
        return 2

    micro = run_microbench(engine, args.artifact, args.microbench_batch,
                           args.microbench_iterations)
    result = build_result(micro)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "parity": parity}, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
