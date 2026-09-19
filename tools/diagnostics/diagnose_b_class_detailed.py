#!/usr/bin/env python3
"""Recompute the CLOSED B-class probe summary from its frozen 40-row evidence.

This is an evidence reader, not a search launcher. The old collector's
`ground_truth` field is only a same-engine bounded-search reference score;
it did not capture alpha/beta, completed depth, bounds or repetition history.
No heuristic or new sample is produced here. Usage: python <this-file>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/diagnostics/b_class_detailed_diagnosis.json"


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("empty labeled probe")
    if len({r["fen"] for r in records}) != len(records):
        raise ValueError("duplicate FEN in frozen sample")
    for r in records:
        # Keep legacy numbers intact; do not relabel mate sentinels as cp truth.
        if r["optimism_gap"] != r["full_eval"] - r["ground_truth"]:
            raise ValueError("inconsistent optimism gap")
        label = r["full_eval"] >= -150 and r["ground_truth"] <= -300
        if r["is_false_optimism"] != label:
            raise ValueError("inconsistent historical proxy label")
        if r["is_severe_optimism"] != (r["optimism_gap"] >= 300):
            raise ValueError("inconsistent historical gap flag")
    positives = sum(r["is_false_optimism"] for r in records)
    rules = [
        ("deficit >= 300 & residual >= 200", lambda r: r["material_deficit"] >= 300 and r["residual_cp"] >= 200),
        ("deficit >= 300 & residual >= 300", lambda r: r["material_deficit"] >= 300 and r["residual_cp"] >= 300),
        ("deficit >= 300 & residual >= 400", lambda r: r["material_deficit"] >= 300 and r["residual_cp"] >= 400),
        ("deficit >= 200 & net_comp >= -100", lambda r: r["material_deficit"] >= 200 and r["net_compensation"] >= -100),
        ("deficit >= 300 & net_comp >= -100", lambda r: r["material_deficit"] >= 300 and r["net_compensation"] >= -100),
        ("deficit >= 300 & net_comp >= 0", lambda r: r["material_deficit"] >= 300 and r["net_compensation"] >= 0),
        ("residual >= 300", lambda r: r["residual_cp"] >= 300),
        ("residual >= 400", lambda r: r["residual_cp"] >= 400),
        ("residual >= 500", lambda r: r["residual_cp"] >= 500),
    ]
    counts = []
    for name, rule in rules:
        tp = sum(rule(r) and r["is_false_optimism"] for r in records)
        fp = sum(rule(r) and not r["is_false_optimism"] for r in records)
        fn = positives - tp
        counts.append({"rule": name, "TP": tp, "FP": fp, "FN": fn,
                       "TN": len(records) - tp - fp - fn,
                       "precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / positives if positives else None})
    return {
        "labeled_rows": len(records), "proxy_positive_rows": positives,
        "label": "full_eval >= -150 and reference_score <= -300 (not a measured cutoff)",
        "rules": counts,
        "gap_ge_500": [{"fen": r["fen"], "reference_score": r["ground_truth"],
                        "legacy_gap": r["optimism_gap"],
                        "mate_sentinel": abs(r["ground_truth"]) == 10000}
                       for r in records if r["optimism_gap"] >= 500],
        "limitations": [
            "No actual cutoff/window telemetry; recall undefined when no proxy positives.",
            "Same-engine reference, not game-theoretic ground truth or independent oracle.",
            "Legacy log says d6; call requested d8; completed depth was not persisted.",
            "Unordered-set sampling was not reproducible; only this frozen evidence is replayable.",
        ],
        "conclusion": "No useful scalar separation observed in the labeled probe.",
        "decision": "STOP scalar stand-pat heuristic work (insufficient supporting evidence).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, default=EVIDENCE)
    args = ap.parse_args()
    raw = args.evidence.read_bytes()
    report = summarize(json.loads(raw))
    report["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
