"""S10-E0 semantic audit: 10k-position STM-perspective numerical audit.

Deterministically samples 10k positions (seed 2026083003) from the frozen
300k validation split, then computes for each:
  * teacher_cp_stm (Stockfish 18 side-to-move score, from labels.jsonl)
  * FP32 NNUE prediction (torch, from the frozen B4 artifact)
  * quantized NNUE prediction (Rust binary, nnue-v2q-probe-batch, from b51)

Grouped by White-STM / Black-STM, reports MAE, signed bias, Pearson/Spearman
correlation, sign agreement, and output clipping/saturation counts.

ABORT criteria (frozen): direction reversal on one color, x1000//1000-scale
errors, one color clearly broken. Everything else is informational.

Usage:
    python tools/s10/e0_semantic_audit.py \
        --dataset data/s10/s10-eval-v1-300k01 \
        --fp32 data/s10/b3/seed-20260818/nnue-v2-f32.bin \
        --quantized data/s10/b3/seed-20260818/nnue-v2-q01.bin \
        --engine target/release/eureka \
        --out results/s10/s10-e0-semantic-audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def load_validation(dataset_dir: Path) -> list[dict]:
    records = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    labels = {}
    for line in (dataset_dir / "labels.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    out = []
    for r in records:
        if r.get("split") != "validation":
            continue
        lab = labels.get(r["position_id"])
        if lab is None or lab.get("teacher_cp_stm") is None:
            continue
        out.append(
            {
                "position_id": r["position_id"],
                "fen": r["fen"],
                "stm": "w" if " w " in r["fen"] else "b",
                "teacher_cp_stm": lab["teacher_cp_stm"],
            }
        )
    return out


def fp32_predictions(
    engine: Path, checkpoint: Path, sample: list[dict]
) -> list[float]:
    """Run the frozen B3 checkpoint through the training-path torch model,
    using the engine's feature exporter as the single source of truth
    (same path as parity_nnue_v2.py)."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.s10.train_nnue import (
        EncodedSplit,
        NnueModel,
        NNUE_INPUTS_V2,
        export_features_from_engine,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    exported = export_features_from_engine(engine, sample, "v2")
    items = []
    for r in sample:
        exp = exported[r["position_id"]]
        stm_is_white = r["fen"].split()[1] == "w"
        items.append({
            "stm": exp["white"] if stm_is_white else exp["black"],
            "nstm": exp["black"] if stm_is_white else exp["white"],
        })
    enc = EncodedSplit([
        {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items
    ])
    with torch.no_grad():
        preds = model(
            enc.stm_indices, enc.stm_offsets,
            enc.nstm_indices, enc.nstm_offsets,
        ).numpy()
    return [float(p) * 1000.0 for p in preds]


def quantized_predictions(
    engine: Path, model: Path, batch_lines: list[str]
) -> dict[str, int]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(batch_lines))
        batch_path = fh.name
    try:
        result = subprocess.run(
            [
                str(engine),
                "bench",
                "nnue-v2q-probe-batch",
                "--model",
                str(model),
                "--batch",
                batch_path,
            ],
            capture_output=True,
            text=True,
            timeout=3600,
            check=True,
        )
    finally:
        Path(batch_path).unlink(missing_ok=True)
    out = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["position_id"]] = rec["raw_output"]
    return out


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


def stats_block(teacher, pred) -> dict:
    n = len(teacher)
    mae = sum(abs(t - p) for t, p in zip(teacher, pred)) / n
    bias = sum(t - p for t, p in zip(teacher, pred)) / n
    agree = sum(
        1
        for t, p in zip(teacher, pred)
        if (t > 0) == (p > 0) or (t == 0 and p == 0)
    )
    # Clipping: the training target is clipped at +-2000 cp; count predictions
    # at/beyond the clip boundary.
    clipped = sum(1 for p in pred if abs(p) >= 2000)
    return {
        "n": n,
        "mae_cp": round(mae, 3),
        "signed_bias_cp": round(bias, 3),
        "pearson": round(pearson(teacher, pred), 6)
        if pearson(teacher, pred) is not None
        else None,
        "spearman": round(spearman(teacher, pred), 6)
        if spearman(teacher, pred) is not None
        else None,
        "sign_agreement": round(agree / n, 4),
        "predictions_at_clip": clipped,
        "teacher_mean_cp": round(sum(teacher) / n, 2),
        "pred_mean_cp": round(sum(pred) / n, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026083003)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pool = load_validation(args.dataset)
    if len(pool) < args.n:
        raise SystemExit(f"validation pool too small: {len(pool)} < {args.n}")
    sample = random.Random(args.seed).sample(pool, args.n)
    print(f"sampled {len(sample)} validation positions "
          f"(w={sum(1 for s in sample if s['stm'] == 'w')}, "
          f"b={sum(1 for s in sample if s['stm'] == 'b')})")

    batch_lines = [
        f"{s['position_id']}|{s['fen']}" for s in sample
    ]
    print("running quantized Rust probe...")
    q_raw = quantized_predictions(args.engine, args.quantized, batch_lines)
    # raw_output is the quantized integer raw value; cp = raw / 2^12 * 1000
    q_cp = {pid: raw / 4096.0 * 1000.0 for pid, raw in q_raw.items()}

    print("running FP32 torch probe...")
    fp_pred = fp32_predictions(args.engine, args.checkpoint, sample)

    by_pid = {s["position_id"]: s for s in sample}
    report = {
        "schema_version": 1,
        "sample": {
            "n": args.n,
            "seed": args.seed,
            "source_split": "validation",
            "dataset": str(args.dataset),
        },
        "abort_criteria": {
            "direction_reversal_one_color": None,
            "scale_error_x1000": None,
            "one_color_broken": None,
        },
    }
    for stm in ("w", "b"):
        ids = [s["position_id"] for s in sample if s["stm"] == stm]
        teacher = [by_pid[i]["teacher_cp_stm"] for i in ids]
        quant = [q_cp[i] for i in ids]
        fp32 = [p for p, s in zip(fp_pred, sample) if s["stm"] == stm]
        report[f"stm_{stm}"] = {
            "teacher_vs_quantized": stats_block(teacher, quant),
            "teacher_vs_fp32": stats_block(teacher, fp32),
            "fp32_vs_quantized": stats_block(fp32, quant),
        }
        print(f"--- stm {stm} ---")
        print(json.dumps(report[f"stm_{stm}"], indent=1))

    # Automatic abort checks
    aborts = report["abort_criteria"]
    for stm in ("w", "b"):
        blk = report[f"stm_{stm}"]["teacher_vs_quantized"]
        if blk["spearman"] is not None and blk["spearman"] < 0:
            aborts["direction_reversal_one_color"] = (
                f"stm_{stm} spearman {blk['spearman']}"
            )
        ratio = blk["pred_mean_cp"] / blk["teacher_mean_cp"] \
            if blk["teacher_mean_cp"] else None
        if ratio is not None and (abs(ratio) > 100 or abs(ratio) < 0.01):
            aborts["scale_error_x1000"] = f"stm_{stm} mean ratio {ratio:.3f}"
        if blk["sign_agreement"] < 0.5:
            aborts["one_color_broken"] = (
                f"stm_{stm} sign agreement {blk['sign_agreement']}"
            )
    report["verdict"] = "ABORT" if any(
        v is not None for v in aborts.values()
    ) else "PASS"
    print("verdict:", report["verdict"], json.dumps(aborts))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.out}")
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
