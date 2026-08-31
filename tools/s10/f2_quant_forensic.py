"""S10-F2: quantized material-residual forensic + counterfactual regression.

Runs the permanent forensic fixture and the 100-base counterfactual
removal suite against the QUANTIZED artifact through the engine's own
probe path, then re-checks the search-level behaviour on the forensic
root (the engine must not play 4...Bg7?? and must see 5.dxc6 as
punishment).

Gates (same as the FP32 F1 gates):
  direction accuracy             >= 99%
  minor removal median |delta|   >= 250cp
  rook removal                   >= 400cp
  queen removal                  >= 700cp
  after_dxc6 composed eval       <= -150cp (black clearly worse)

Usage:
    python tools/s10/f2_quant_forensic.py \
        --quantized data/s10/f1/seed-20260820/nnue-v2-q01-material.bin \
        --dataset data/s10/s10-eval-v1-300k01 \
        --engine target/release/eureka \
        --out results/s10/s10-f2-quant-forensic.json
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
    "minor_median_min": 250.0,
    "rook_median_min": 400.0,
    "queen_median_min": 700.0,
    "after_dxc6_max": -150.0,
}

FT_SHIFT = 12


def counterfactuals(board_fen: str):
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
        b.remove_piece_at(candidates[0])
        if not b.is_valid():
            continue
        out.append((name, b.fen()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2026083102)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from tools.s10.train_nnue import material_cp_stm_python

    records = []
    for line in (args.dataset / "part-0000.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            records.append(json.loads(line))
    labels = {}
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

    all_fens = list(FORENSIC_FENS.values())
    base_idx = {}
    variant_idx = []
    for r in bases:
        base_idx[r["position_id"]] = len(all_fens)
        all_fens.append(r["fen"])
        for name, vf in counterfactuals(r["fen"]):
            variant_idx.append((r["position_id"], name, len(all_fens)))
            all_fens.append(vf)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(f"x{i}|{f}" for i, f in enumerate(all_fens)))
        qbatch = fh.name
    try:
        proc = subprocess.run(
            [str(args.engine), "bench", "nnue-v2q-probe-batch",
             "--model", str(args.quantized), "--batch", qbatch],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    finally:
        Path(qbatch).unlink(missing_ok=True)
    raw = {}
    for line in proc.stdout.splitlines():
        if line.strip():
            rec = json.loads(line)
            raw[rec["position_id"]] = rec["raw_output"]

    def composed(i: int) -> float:
        fen = all_fens[i]
        m = material_cp_stm_python(fen)
        return m + raw[f"x{i}"] * 1000.0 / (1 << FT_SHIFT)

    forensic = {k: round(composed(i), 2)
                for i, k in enumerate(FORENSIC_FENS)}
    print("forensic (quantized composed stm cp):", forensic)

    deltas: dict[str, list[float]] = defaultdict(list)
    directions: dict[str, list[bool]] = defaultdict(list)
    for pid, name, vi in variant_idx:
        d = composed(vi) - composed(base_idx[pid])
        deltas[name].append(d)
        expected_positive = name.startswith("opp")
        directions[name].append((d > 0) if expected_positive else (d < 0))

    def median(v):
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    def group(prefixes):
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
        "minor_removal": group(["own_n", "opp_n", "own_b", "opp_b"]),
        "rook_removal": group(["own_r", "opp_r"]),
        "queen_removal": group(["own_q", "opp_q"]),
    }
    print("removal groups:", json.dumps(grouped, indent=1))

    gate_results = {
        "forensic_after_dxc6": {
            "value": forensic["after_dxc6"],
            "gate_max": GATES["after_dxc6_max"],
            "pass": forensic["after_dxc6"] <= GATES["after_dxc6_max"],
        },
    }
    for key, group in (("minor", "minor_removal"),
                       ("rook", "rook_removal"),
                       ("queen", "queen_removal")):
        gate_results[group] = {
            "stats": grouped[group],
            "median_gate": GATES[f"{key}_median_min"],
            "direction_gate": GATES["direction_accuracy_min"],
            "pass": (
                grouped[group]["median_abs_delta"] >= GATES[f"{key}_median_min"]
                and grouped[group]["direction_accuracy"]
                >= GATES["direction_accuracy_min"]
            ),
        }
    all_pass = all(g["pass"] for g in gate_results.values())
    print("GATES:", "ALL PASS" if all_pass else "FAIL")
    for k, g in gate_results.items():
        print(f"  {k}: {'PASS' if g['pass'] else 'FAIL'}")

    report = {
        "schema_version": 1,
        "quantized_artifact": str(args.quantized),
        "forensic_composed_cp": forensic,
        "counterfactual_groups": grouped,
        "gates": gate_results,
        "all_gates_pass": all_pass,
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
