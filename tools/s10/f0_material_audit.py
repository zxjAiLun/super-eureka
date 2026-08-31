"""S10-F0 material-distribution audit of the frozen 300k dataset.

Deterministic, labels-only analysis (no Stockfish, no GPU, no new data):

For every usable record (teacher_cp_stm not None) in the frozen 300k:
  * raw material balance in cp from the engine's canonical piece values
    (P=100 N=320 B=330 R=500 Q=900 — src/chess/types.rs PieceType::value),
    computed stm-perspective: positive = stm up material;
  * the SAME value white-perspective, for the histogram;
  * teacher_cp_stm from labels.jsonl.

Outputs (results/s10/s10-f0-material-audit.json):
  * |material| histograms (0-50 / 51-150 / 151-250 / 251-400 / 401-600 / 601+)
    by split and by phase bucket;
  * counts for |material| >= 250 / 450 / 800;
  * pearson/spearman corr(material_stm, teacher_cp_stm) per split;
  * NNUE quantized-model MAE per material bucket on a deterministic 10k
    validation sample (seed 2026083101), bucketed by stm material
    (down-a-minor / down-a-rook+ / balanced) — the frozen B5 artifact,
    probed via the engine's own nnue-v2q-probe-batch.

The F0 prediction to test: balanced positions dominate; big-imbalance
positions are rare; NNUE MAE is worst in the down-a-minor / down-a-rook+
buckets.

Usage:
    python tools/s10/f0_material_audit.py \
        --dataset data/s10/s10-eval-v1-300k01 \
        --quantized data/s10/b3/seed-20260818/nnue-v2-q01.bin \
        --engine target/release/eureka \
        --sample-n 10000 \
        --out results/s10/s10-f0-material-audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Frozen canonical piece values (src/chess/types.rs). DO NOT tune here.
PIECE_CP = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900}

HIST_EDGES = [0, 50, 150, 250, 400, 600]
THRESHOLDS = [250, 450, 800]


def material_balance_white(fen_board: str) -> int:
    """Raw material balance (white - black) in cp from the board field."""
    w = 0
    b = 0
    for ch in fen_board:
        if ch.isupper():
            cp = PIECE_CP.get(ch.lower())
            if cp is not None:
                w += cp
        else:
            cp = PIECE_CP.get(ch)
            if cp is not None:
                b += cp
    return w - b


def stm_material(fen: str) -> int:
    """Material balance from the side-to-move's perspective (+ = stm up)."""
    parts = fen.split()
    white = material_balance_white(parts[0])
    return white if parts[1] == "w" else -white


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
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


def hist_bucket(a: int) -> str:
    for i in range(len(HIST_EDGES) - 1):
        if HIST_EDGES[i] <= a <= HIST_EDGES[i + 1]:
            return f"{HIST_EDGES[i]}-{HIST_EDGES[i+1]}"
    return f"{HIST_EDGES[-1]}+"


def load_records(dataset_dir: Path) -> list[dict]:
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
        lab = labels.get(r["position_id"])
        if lab is None or lab.get("teacher_cp_stm") is None:
            continue
        out.append(
            {
                "position_id": r["position_id"],
                "fen": r["fen"],
                "split": r["split"],
                "phase": r["phase"],
                "teacher_cp_stm": lab["teacher_cp_stm"],
            }
        )
    return out


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--sample-n", type=int, default=10000)
    parser.add_argument("--sample-seed", type=int, default=2026083101)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.dataset)
    if not records:
        raise SystemExit("FAIL CLOSED: 0 usable records with teacher_cp_stm")
    print(f"loaded {len(records)} usable records")

    # --- distribution audit (all usable records, every split) -------------
    hist_by_split: dict[str, Counter] = defaultdict(Counter)
    hist_by_phase: dict[str, Counter] = defaultdict(Counter)
    thresholds_by_split: dict[str, Counter] = defaultdict(Counter)
    thresholds_by_phase: dict[str, Counter] = defaultdict(Counter)
    mat_stm: dict[str, list[int]] = defaultdict(list)
    teacher: dict[str, list[int]] = defaultdict(list)
    n_total = 0

    for r in records:
        m = stm_material(r["fen"])
        a = abs(m)
        bucket = hist_bucket(a)
        hist_by_split[r["split"]][bucket] += 1
        hist_by_phase[r["phase"]][bucket] += 1
        for t in THRESHOLDS:
            if a >= t:
                thresholds_by_split[r["split"]][f">={t}"] += 1
                thresholds_by_phase[r["phase"]][f">={t}"] += 1
        mat_stm[r["split"]].append(m)
        teacher[r["split"]].append(r["teacher_cp_stm"])
        n_total += 1

    def counter_to_pct(c: Counter, denom: int) -> dict:
        return {
            k: {"n": c[k], "pct": round(100.0 * c[k] / denom, 3)}
            for k in sorted(c)
        }

    split_sizes = Counter(r["split"] for r in records)
    phase_sizes = Counter(r["phase"] for r in records)

    correlations = {}
    for split in sorted(mat_stm):
        xs = mat_stm[split]
        ys = teacher[split]
        correlations[split] = {
            "n": len(xs),
            "pearson": round(pearson(xs, ys), 6)
            if pearson(xs, ys) is not None
            else None,
            "spearman": round(spearman(xs, ys), 6)
            if spearman(xs, ys) is not None
            else None,
        }

    # --- NNUE MAE by stm material bucket (validation sample) --------------
    validation = [r for r in records if r["split"] == "validation"]
    sample = random.Random(args.sample_seed).sample(
        validation, min(args.sample_n, len(validation))
    )
    print(f"sampled {len(sample)} validation positions for NNUE probing")

    batch_lines = [f"{s['position_id']}|{s['fen']}" for s in sample]
    print("running quantized Rust probe...")
    q_raw = quantized_predictions(args.engine, args.quantized, batch_lines)

    # raw_output -> cp: raw / 2^FT_SHIFT * target_scale (FT_SHIFT=12, scale=1000)
    errors_by_bucket: dict[str, list[float]] = defaultdict(list)
    probe_failures = 0
    for s in sample:
        raw = q_raw.get(s["position_id"])
        if raw is None:
            probe_failures += 1
            continue
        pred_cp = raw / 4096.0 * 1000.0
        m = stm_material(s["fen"])
        if m <= -500:
            bucket = "stm_down_rook_plus"
        elif m <= -250:
            bucket = "stm_down_minor"
        elif m >= 500:
            bucket = "stm_up_rook_plus"
        elif m >= 250:
            bucket = "stm_up_minor"
        else:
            bucket = "balanced_-250_250"
        errors_by_bucket[bucket].append(
            abs(pred_cp - s["teacher_cp_stm"])
        )

    mae_by_bucket = {}
    for bucket in sorted(errors_by_bucket):
        errs = errors_by_bucket[bucket]
        mae_by_bucket[bucket] = {
            "n": len(errs),
            "mae_cp": round(sum(errs) / len(errs), 3),
        }

    report = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "quantized_model": str(args.quantized),
        "piece_values_cp": PIECE_CP,
        "records_usable": n_total,
        "hist_edges": HIST_EDGES,
        "hist_by_split": {
            split: {
                "split_size": split_sizes[split],
                "buckets": counter_to_pct(
                    hist_by_split[split], split_sizes[split]
                ),
            }
            for split in sorted(hist_by_split)
        },
        "hist_by_phase": {
            phase: {
                "phase_size": phase_sizes[phase],
                "buckets": counter_to_pct(
                    hist_by_phase[phase], phase_sizes[phase]
                ),
            }
            for phase in sorted(hist_by_phase)
        },
        "thresholds_by_split": {
            split: counter_to_pct(
                thresholds_by_split[split], split_sizes[split]
            )
            for split in sorted(thresholds_by_split)
        },
        "thresholds_by_phase": {
            phase: counter_to_pct(
                thresholds_by_phase[phase], phase_sizes[phase]
            )
            for phase in sorted(thresholds_by_phase)
        },
        "correlations": correlations,
        "nnue_mae_by_material_bucket": mae_by_bucket,
        "nnue_sample": {
            "n": len(sample),
            "seed": args.sample_seed,
            "probe_failures": probe_failures,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")

    # Console summary
    print("\n=== |material| histogram (validation) ===")
    for bucket, v in report["hist_by_split"]["validation"]["buckets"].items():
        print(f"  {bucket:>10}: {v['n']:>7}  ({v['pct']:6.2f}%)")
    print("\n=== thresholds (validation) ===")
    for k, v in report["thresholds_by_split"]["validation"].items():
        print(f"  {k}: {v['n']} ({v['pct']}%)")
    print("\n=== corr(material_stm, teacher_cp_stm) ===")
    for split, c in correlations.items():
        print(f"  {split}: pearson={c['pearson']} spearman={c['spearman']}")
    print("\n=== NNUE quantized MAE by material bucket (validation) ===")
    for bucket, v in mae_by_bucket.items():
        print(f"  {bucket:>22}: n={v['n']:>6}  MAE={v['mae_cp']:8.2f}cp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
