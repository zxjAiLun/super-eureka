"""S10-F1 gates: forensic fixture + counterfactual material sanity + bucket MAE.

Runs on the FP32 torch checkpoints (material-residual mode), composed as
`eval = material_cp_stm + residual_prediction * 1000`.

Gates (frozen):
  1. Forensic `4.d5 ...Bg7?? 5.dxc6` fixture: after_dxc6 must evaluate as
     clearly worse for Black (the composed eval must be <= -150cp; the old
     pure NNUE said +13).
  2. Counterfactual material removal over >=100 validation base positions:
       direction accuracy  >= 99%
       minor removal median |delta| >= 250cp
       rook removal        >= 400cp
       queen removal       >= 700cp
  3. Bucket MAE + signed bias comparison, pure 300k NNUE (frozen B3 q01
     numbers from F0 re-computed here on the same sample) vs
     material-residual on identical validation positions.

Usage:
    python tools/s10/f1_gates.py \
        --checkpoint data/s10/f1/seed-20260820/checkpoint_v2_s20260820.pt \
        --baseline-quantized data/s10/b3/seed-20260818/nnue-v2-q01.bin \
        --dataset data/s10/s10-eval-v1-300k01 \
        --engine target/release/eureka \
        --out results/s10/s10-f1-gates.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FORENSIC_FENS = {
    "P0": "r1bqkbnr/ppp1pp1p/2np2p1/3P4/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 4",
    "after_Bg7": (
        "r1bqk1nr/ppp1ppbp/2np2p1/3P4/2P5/2N5/"
        "PP2PPPP/R1BQKBNR w KQkq - 1 5"
    ),
    "after_dxc6": (
        "r1bqk1nr/ppp1ppbp/2Pp2p1/8/2P5/2N5/"
        "PP2PPPP/R1BQKBNR b KQkq - 0 5"
    ),
    "after_bxc6": (
        "r1bqk1nr/p1p1ppbp/2pp2p1/8/2P5/2N5/"
        "PP2PPPP/R1BQKBNR w KQkq - 0 6"
    ),
}

REMOVALS = [
    ("own_n", "n", True), ("opp_n", "n", False),
    ("own_b", "b", True), ("opp_b", "b", False),
    ("own_r", "r", True), ("opp_r", "r", False),
    ("own_q", "q", True), ("opp_q", "q", False),
]

GATES = {
    "direction_accuracy_min": 0.99,
    "minor_removal_median_abs_delta_min": 250.0,
    "rook_removal_median_abs_delta_min": 400.0,
    "queen_removal_median_abs_delta_min": 700.0,
    "after_dxc6_black_must_be_worse_than_max": -150.0,
}


def load_checkpoint(path: Path):
    import torch

    from tools.s10.train_nnue import NNUE_INPUTS_V2, NnueModel

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NnueModel(num_inputs=NNUE_INPUTS_V2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def composed_predictions(model, engine: Path, fens: list[str]):
    """FP32 residual predictions via the engine feature exporter, composed
    with the engine's canonical material (material-batch, cross-checked)."""
    import torch

    from tools.s10.train_nnue import (
        export_features_from_engine,
        material_cp_stm_python,
    )

    records = [
        {"position_id": f"c{i}", "fen": f} for i, f in enumerate(fens)
    ]
    exported = export_features_from_engine(engine, records, "v2")

    # material from the engine (single source of truth), cross-checked
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("".join(f"{r['position_id']}|{r['fen']}\n" for r in records))
        batch_path = fh.name
    try:
        proc = subprocess.run(
            [str(engine), "bench", "material-batch", "--batch", batch_path],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    finally:
        Path(batch_path).unlink(missing_ok=True)
    material = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        material[rec["position_id"]] = rec["material_cp_stm"]
    for r in records:
        assert material[r["position_id"]] == material_cp_stm_python(r["fen"])

    items = []
    for i, r in enumerate(records):
        exp = exported[r["position_id"]]
        stm_white = r["fen"].split()[1] == "w"
        items.append({
            "stm": exp["white"] if stm_white else exp["black"],
            "nstm": exp["black"] if stm_white else exp["white"],
            "material": float(material[r["position_id"]]),
        })

    from tools.s10.train_nnue import EncodedSplit

    enc = EncodedSplit([
        {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items
    ])
    with torch.no_grad():
        preds = model(
            enc.stm_indices, enc.stm_offsets, enc.nstm_indices, enc.nstm_offsets
        ).numpy()
    return [
        it["material"] + float(p) * 1000.0 for it, p in zip(items, preds)
    ]


def counterfactuals(board_fen: str):
    """Generate legal (board-valid) removal variants of one FEN."""
    import chess

    out = []
    for name, letter, own in REMOVALS:
        piece_type = {
            "n": chess.KNIGHT, "b": chess.BISHOP,
            "r": chess.ROOK, "q": chess.QUEEN,
        }[letter]
        b = chess.Board(board_fen)
        stm = b.turn
        color = stm if own else (not stm)
        candidates = [
            sq for sq, pc in b.piece_map().items()
            if pc.piece_type == piece_type and pc.color == color
        ]
        if not candidates:
            continue
        # deterministic: remove the first candidate
        b.remove_piece_at(candidates[0])
        if not b.is_valid():
            continue
        out.append((name, b.fen()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-quantized", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026083102)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)

    # --- 1. forensic fixture -----------------------------------------------
    forensic = {}
    fens = list(FORENSIC_FENS.values())
    preds = composed_predictions(model, args.engine, fens)
    for name, p in zip(FORENSIC_FENS, preds):
        forensic[name] = round(p, 2)
    print("forensic (composed stm cp):", forensic)

    # --- 2. counterfactual material gate -----------------------------------
    records = []
    labels = {}
    for line in (args.dataset / "part-0000.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            records.append(json.loads(line))
    for line in (args.dataset / "labels.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    validation = [
        r for r in records
        if r["split"] == "validation"
        and labels[r["position_id"]].get("teacher_cp_stm") is not None
    ]
    bases = random.Random(args.base_seed).sample(validation, args.n_bases)

    # Build the full FEN list: base + variants, one batch.
    all_fens = []
    base_idx = {}
    variant_idx = []
    for r in bases:
        base_idx[r["position_id"]] = len(all_fens)
        all_fens.append(r["fen"])
        for name, vf in counterfactuals(r["fen"]):
            variant_idx.append((r["position_id"], name, len(all_fens)))
            all_fens.append(vf)

    print(f"probing {len(all_fens)} positions ("
          f"{args.n_bases} bases + {len(variant_idx)} variants)...")
    preds = composed_predictions(model, args.engine, all_fens)

    deltas: dict[str, list[float]] = defaultdict(list)
    directions: dict[str, list[bool]] = defaultdict(list)
    for pid, name, vi in variant_idx:
        d = preds[vi] - preds[base_idx[pid]]
        deltas[name].append(d)
        # removing own piece should lower stm eval; opp piece raise it
        expected_positive = name.startswith("opp")
        directions[name].append((d > 0) if expected_positive else (d < 0))

    def median(v):
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    removal_stats = {}
    for name in deltas:
        removal_stats[name] = {
            "n": len(deltas[name]),
            "median_abs_delta": round(median(
                [abs(d) for d in deltas[name]]), 2),
            "direction_accuracy": round(
                sum(directions[name]) / len(directions[name]), 4),
        }

    def group_stats(prefixes):
        ds = [d for name in prefixes for d in deltas.get(name, [])]
        dir_ok = [ok for name in prefixes for ok in directions.get(name, [])]
        if not ds:
            return None
        return {
            "n": len(ds),
            "median_abs_delta": round(median([abs(d) for d in ds]), 2),
            "direction_accuracy": round(sum(dir_ok) / len(dir_ok), 4),
        }

    grouped = {
        "minor_removal": group_stats(["own_n", "opp_n", "own_b", "opp_b"]),
        "rook_removal": group_stats(["own_r", "opp_r"]),
        "queen_removal": group_stats(["own_q", "opp_q"]),
    }
    print("removal groups:", json.dumps(grouped, indent=1))

    # --- gates ---------------------------------------------------------------
    gate_results = {
        "forensic_after_dxc6": {
            "value": forensic["after_dxc6"],
            "gate_max": GATES["after_dxc6_black_must_be_worse_than_max"],
            "pass": forensic["after_dxc6"]
            <= GATES["after_dxc6_black_must_be_worse_than_max"],
        },
    }
    for key, group in (("minor", "minor_removal"),
                       ("rook", "rook_removal"),
                       ("queen", "queen_removal")):
        gate_results[f"{group}"] = {
            "stats": grouped[group],
            "median_gate": GATES[f"{key}_removal_median_abs_delta_min"],
            "direction_gate": GATES["direction_accuracy_min"],
            "pass": (
                grouped[group]["median_abs_delta"]
                >= GATES[f"{key}_removal_median_abs_delta_min"]
                and grouped[group]["direction_accuracy"]
                >= GATES["direction_accuracy_min"]
            ),
        }
    all_pass = all(g["pass"] for g in gate_results.values())
    print("GATES:", "ALL PASS" if all_pass else "FAIL")
    for k, g in gate_results.items():
        print(f"  {k}: {'PASS' if g['pass'] else 'FAIL'}")

    # --- 3. bucket MAE + signed bias vs pure NNUE ---------------------------
    # Pure NNUE baseline: frozen q01 on the same validation records.
    val_ids = [r["position_id"] for r in bases]
    val_fens = [r["fen"] for r in bases]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(f"{i}|{f}" for i, f in zip(val_ids, val_fens)))
        qbatch = fh.name
    try:
        proc = subprocess.run(
            [str(args.engine), "bench", "nnue-v2q-probe-batch",
             "--model", str(args.baseline_quantized), "--batch", qbatch],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    finally:
        Path(qbatch).unlink(missing_ok=True)
    baseline_cp = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            rec = json.loads(line)
            baseline_cp[rec["position_id"]] = rec["raw_output"] / 4096.0 * 1000.0

    from tools.s10.train_nnue import material_cp_stm_python

    # NOTE: preds is indexed over all_fens (bases INTERLEAVED with their
    # variants). Map each base by its recorded index, never by list position.
    f1_cp = {
        pid: preds[base_idx[pid]] for pid in val_ids
    }
    errors_by_bucket = {"baseline": defaultdict(list),
                        "f1": defaultdict(list)}
    signed_by_bucket = {"baseline": defaultdict(list),
                        "f1": defaultdict(list)}
    for r in bases:
        pid = r["position_id"]
        teacher = float(labels[pid]["teacher_cp_stm"])
        m = material_cp_stm_python(r["fen"])
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
        for arm, cp in (("baseline", baseline_cp[pid]), ("f1", f1_cp[pid])):
            errors_by_bucket[arm][bucket].append(abs(cp - teacher))
            signed_by_bucket[arm][bucket].append(cp - teacher)

    bucket_table = {}
    for bucket in sorted(errors_by_bucket["f1"]):
        row = {}
        for arm in ("baseline", "f1"):
            errs = errors_by_bucket[arm][bucket]
            sgn = signed_by_bucket[arm][bucket]
            row[arm] = {
                "n": len(errs),
                "mae_cp": round(sum(errs) / len(errs), 2),
                "signed_bias_cp": round(sum(sgn) / len(sgn), 2),
            }
        bucket_table[bucket] = row
    overall = {}
    for arm in ("baseline", "f1"):
        errs = [e for b in errors_by_bucket[arm] for e in errors_by_bucket[arm][b]]
        sgn = [e for b in signed_by_bucket[arm] for e in signed_by_bucket[arm][b]]
        overall[arm] = {
            "n": len(errs),
            "mae_cp": round(sum(errs) / len(errs), 2),
            "signed_bias_cp": round(sum(sgn) / len(sgn), 2),
        }
    print("bucket table:")
    for bucket, row in bucket_table.items():
        print(f"  {bucket:>22}: baseline MAE={row['baseline']['mae_cp']:8.2f} "
              f"bias={row['baseline']['signed_bias_cp']:8.2f} | "
              f"F1 MAE={row['f1']['mae_cp']:8.2f} "
              f"bias={row['f1']['signed_bias_cp']:8.2f}")
    print(f"  {'overall':>22}: baseline MAE={overall['baseline']['mae_cp']:8.2f} "
          f"| F1 MAE={overall['f1']['mae_cp']:8.2f}")

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "baseline_quantized": str(args.baseline_quantized),
        "forensic_composed_cp": forensic,
        "counterfactual_removal_stats": removal_stats,
        "counterfactual_groups": grouped,
        "gates": gate_results,
        "all_gates_pass": all_pass,
        "bucket_table": bucket_table,
        "overall": overall,
        "sample": {
            "n_bases": args.n_bases,
            "base_seed": args.base_seed,
            "n_variants": len(variant_idx),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
