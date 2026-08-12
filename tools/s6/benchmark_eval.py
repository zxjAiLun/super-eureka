#!/usr/bin/env python3
"""S6.0 CurrentFinal baseline evaluation benchmark + teacher challenge set.

On the dataset HOLDOUT, compare:
  - CurrentFinal production static eval (material_pst component)
  - the dormant integrated E2 evaluator (sum of all components)
against frozen teacher labels (cp/mate/wdl) and game outcomes.

Metrics: MAE, RMSE, median absolute error, Pearson, Spearman, sign
agreement (|teacher_cp| >= 50/100/200), Texel logistic loss vs game
outcome, 3-way cross-entropy vs teacher WDL. All metrics phase-bucketed.

Also builds data/s6/s6-teacher-challenge-v1.jsonl (S4 disagreement
positions + fixed tactical/endgame challenge cases; never in training)
and labels it with the same frozen teacher.

Observation-only: no engine/evaluator changes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import chess

CHALLENGE_SOURCES = [
    ("s4-disagreements", "results/s4-attribution/quality/disagreements.jsonl", "jsonl"),
    ("threat-awareness", "tests/data/threat-awareness.epd", "epd"),
    ("external-validation-v1", "tests/data/external_validation_v1.epd", "epd"),
    ("search-validation", "tests/data/search_validation.epd", "epd"),
]
PHASE_BUCKETS = {"high": (18, 24), "mid": (8, 17), "low": (1, 7), "zero": (0, 0)}
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def phase_of(board: chess.Board) -> int:
    total = 0
    for _, piece in board.piece_map().items():
        total += PHASE_WEIGHTS.get(piece.piece_type, 0)
    return min(24, total)


def bucket_of(phase: int) -> str:
    for name, (lo, hi) in PHASE_BUCKETS.items():
        if lo <= phase <= hi:
            return name
    return "mid"


def sigmoid(cp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
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
    return pearson(rank(x), rank(y))


def run_eval_breakdown(engine: Path, fen: str) -> dict[str, int] | None:
    out = subprocess.run(
        [str(engine), "bench", "eval-breakdown", "--fen", fen],
        capture_output=True, text=True, timeout=60,
    ).stdout
    for line in out.splitlines():
        if line.startswith("eval_breakdown "):
            kv = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            return {k: int(v) for k, v in kv.items() if k not in ("fen", "side")}
    return None


def metrics(pred: list[float], tgt: list[float]) -> dict:
    errs = [a - b for a, b in zip(pred, tgt)]
    mae = statistics.mean(abs(e) for e in errs)
    rmse = math.sqrt(statistics.mean(e * e for e in errs))
    med = statistics.median(abs(e) for e in errs)
    sign = sum(1 for a, b in zip(pred, tgt) if (a > 0) == (b > 0))
    return {
        "n": len(pred),
        "mae": mae,
        "rmse": rmse,
        "median_abs": med,
        "pearson": pearson(pred, tgt),
        "spearman": spearman(pred, tgt),
        "sign_agreement": sign / len(pred) if pred else float("nan"),
    }


def texel_loss(pred: list[float], outcomes: list[float]) -> float:
    total = 0.0
    for p, o in zip(pred, outcomes):
        s = sigmoid(p)
        s = min(max(s, 1e-6), 1 - 1e-6)
        total += -(o * math.log(s) + (1 - o) * math.log(1 - s))
    return total / len(pred) if pred else float("nan")


def ce_vs_teacher_wdl(pred: list[float], wdls: list[list[int]]) -> float:
    total = 0.0
    n = 0
    for p, w in zip(pred, wdls):
        if w is None or sum(w) == 0:
            continue
        probs = [x / sum(w) for x in w]
        s = sigmoid(p)
        pv = [s * s, 2 * s * (1 - s), (1 - s) * (1 - s)]  # win/draw/loss
        total += -sum(q * math.log(max(qq, 1e-9)) for q, qq in zip(probs, pv))
        n += 1
    return total / n if n else float("nan")


def build_challenge() -> list[dict]:
    out: list[dict] = []
    for source_id, path, kind in CHALLENGE_SOURCES:
        if kind == "jsonl":
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                fen = rec["fen"].split(";")[0].strip()
                out.append({"source_id": source_id, "fen": fen})
        else:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    # pipe format: id|class|fen|tag|...
                    fields = line.split("|")
                    fen = fields[2] if len(fields) > 2 else fields[0]
                else:
                    fen = line.split(";")[0]
                out.append({"source_id": source_id, "fen": fen.strip()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="dataset dir")
    parser.add_argument("--out", default="results/s6")
    parser.add_argument("--label-challenge", action="store_true",
                        help="also teacher-label the challenge corpus")
    args = parser.parse_args(sys.argv[1:])
    engine = args.engine.resolve()
    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- challenge corpus (always built; never in training) ----
    challenge = build_challenge()
    # keep every PARSEABLE position (challenge set may legitimately include
    # in-check / tactical cases; only unparseable FENs are dropped)
    valid: list[dict] = []
    for rec in challenge:
        try:
            chess.Board(rec["fen"])
            valid.append(rec)
        except ValueError:
            continue
    challenge = valid
    challenge_path = Path("data/s6/s6-teacher-challenge-v1.jsonl")
    challenge_path.parent.mkdir(parents=True, exist_ok=True)
    challenge_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                for r in challenge),
        encoding="utf-8")
    print(f"challenge corpus: {len(challenge)} positions")

    if args.label_challenge:
        from label_teacher import Teacher
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        teacher = Teacher(True)
        labeled = []
        failed = []
        for i, rec in enumerate(challenge):
            if i % 100 == 0:
                print(f"  challenge {i}/{len(challenge)}", flush=True)
            try:
                lbl = teacher.label(rec["fen"])
                labeled.append({**rec, **lbl})
            except RuntimeError:
                # teacher died (bad fen or transient): respawn and retry once
                try:
                    teacher.close()
                except Exception:
                    pass
                teacher = Teacher(True)
                try:
                    lbl = teacher.label(rec["fen"])
                    labeled.append({**rec, **lbl})
                except RuntimeError as exc:
                    failed.append({"fen": rec["fen"], "source": rec["source_id"],
                                   "error": str(exc)})
        teacher.close()
        challenge = labeled
        challenge_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                    for r in challenge),
            encoding="utf-8")
        print(f"challenge labeled: {len(challenge)}  failed: {len(failed)}")
        if failed:
            print("failed:", json.dumps(failed[:5], indent=1))

    # ---- holdout ----
    labels: dict[str, dict] = {}
    labels_path = dataset_dir / "labels.jsonl"
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    records = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["split"] == "holdout":
                    records.append(rec)
    print(f"holdout positions: {len(records)}")

    cf_pred: list[float] = []
    e2_pred: list[float] = []
    tgt: list[float] = []
    outcomes: list[float] = []
    wdls: list[list[int] | None] = []
    buckets: dict[str, list] = {b: [] for b in PHASE_BUCKETS}
    skipped = {"mate": 0, "no_label": 0, "eval_fail": 0}
    for i, rec in enumerate(records):
        lbl = labels.get(rec["position_id"])
        if lbl is None or (lbl["teacher_cp_stm"] is None and lbl["teacher_mate"] is None):
            skipped["no_label"] += 1
            continue
        comp = run_eval_breakdown(engine, rec["fen"])
        if comp is None:
            skipped["eval_fail"] += 1
            continue
        board = chess.Board(rec["fen"])
        sign = 1 if board.turn == chess.WHITE else -1
        cf = sign * comp["material_pst"]
        e2 = sign * sum(
            comp[k] for k in ("material_pst", "pawn_structure", "mobility",
                              "piece_activity", "rook_activity",
                              "development_space", "king_safety"))
        if lbl["teacher_cp_stm"] is None:
            # mate labels: keep in dataset, exclude from cp regression
            skipped["mate"] += 1
            continue
        cf_pred.append(float(cf))
        e2_pred.append(float(e2))
        tgt.append(float(lbl["teacher_cp_stm"]))
        outcome = rec["game_result_white"]
        outcomes.append(outcome if board.turn == chess.WHITE else 1.0 - outcome)
        wdls.append(lbl.get("teacher_wdl_stm"))
        buckets[bucket_of(rec["phase"])].append((cf, e2, lbl["teacher_cp_stm"]))

    base = {
        "evaluator": "current-final-material-pst",
        "metrics": metrics(cf_pred, tgt),
        "texel_logistic_loss": texel_loss(cf_pred, outcomes),
        "ce_teacher_wdl": ce_vs_teacher_wdl(cf_pred, wdls),
        "phase_buckets": {
            b: metrics([x[0] for x in v], [x[2] for x in v])
            for b, v in buckets.items() if v
        },
        "sign_agreement_thresholds": {
            t: round(
                sum(1 for a, b in zip(cf_pred, tgt)
                    if abs(b) >= t and (a > 0) == (b > 0))
                / sum(1 for b in tgt if abs(b) >= t), 4)
            for t in (50, 100, 200)
        },
    }
    e2 = {
        "evaluator": "dormant-integrated-e2 (diagnostic only)",
        "metrics": metrics(e2_pred, tgt),
        "texel_logistic_loss": texel_loss(e2_pred, outcomes),
        "ce_teacher_wdl": ce_vs_teacher_wdl(e2_pred, wdls),
        "phase_buckets": {
            b: metrics([x[1] for x in v], [x[2] for x in v])
            for b, v in buckets.items() if v
        },
    }
    result = {
        "dataset_id": dataset_dir.name,
        "holdout_count": len(records),
        "evaluated": len(tgt),
        "skipped": skipped,
        "current_final": base,
        "dormant_e2": e2,
        "challenge": {
            "path": str(challenge_path),
            "count": len(challenge),
        },
    }
    (out_dir / "baseline_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
