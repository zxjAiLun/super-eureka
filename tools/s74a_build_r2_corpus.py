#!/usr/bin/env python3
"""S7.4A Repair 1: build the deterministic R2 tactical/horizon corpus.

Sources (EXISTING project data only, no hand-designed positions):
  1. data/s6/s6-teacher-challenge-v1.jsonl
     - every teacher_mate-labelled row (5; depth 8 REQUIRED in the gate)
     - every |teacher_cp_stm| >= 500 row (32)
  2. data/s6/s6-eval-v1-core-shard01/ labels joined to part-*.jsonl FENs
     - teacher_mate rows with mate distance 1..8
     - |teacher_cp_stm| >= 500 rows
     Deterministic order: sorted by position_id, then truncated to fill the
     target size (120).

Rows with teacher_bestmove "(none)"/None (terminal / no-move fixtures) are
excluded. Writes data/s7/s74a-r2-tactical-corpus.jsonl and prints its SHA256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO / "data/s6/s6-teacher-challenge-v1.jsonl"
SHARD = REPO / "data/s6/s6-eval-v1-core-shard01"
OUT = REPO / "data/s7/s74a-r2-tactical-corpus.jsonl"
TARGET = 120


def cp_val(v) -> int | None:
    s = str(v)
    if not s.lstrip("-").isdigit():
        return None
    return int(s)


def mate_val(v) -> int | None:
    s = str(v)
    if not s.lstrip("-").isdigit():
        return None
    d = int(s)
    return d if d != 0 else None


def main() -> int:
    rows: list[dict] = []
    seen_fens: set[str] = set()

    # Source 1: teacher challenge.
    challenge = [json.loads(l) for l in
                 CHALLENGE.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, r in enumerate(challenge):
        bm = r.get("teacher_bestmove")
        if not bm or bm == "(none)":
            continue
        m = mate_val(r.get("teacher_mate"))
        c = cp_val(r.get("teacher_cp_stm"))
        if (m is not None and abs(m) <= 8) or (c is not None and abs(c) >= 500):
            fen = r["fen"]
            if fen in seen_fens:
                continue
            seen_fens.add(fen)
            rows.append({"fen": fen, "source": "s6-challenge",
                         "teacher_bestmove": bm,
                         "teacher_mate": r.get("teacher_mate"),
                         "teacher_cp_stm": r.get("teacher_cp_stm"),
                         "sort_key": f"challenge-{i:04d}"})

    n_challenge = len(rows)

    # Source 2: eval shard labels joined to part FENs.
    labels = {}
    for l in SHARD.joinpath("labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        labels[r["position_id"]] = r
    part_rows = []
    for part in sorted(SHARD.glob("part-*.jsonl")):
        for l in part.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            part_rows.append(json.loads(l))

    shard_sel = []
    for p in part_rows:
        lab = labels.get(p.get("position_id"))
        if lab is None:
            continue
        bm = lab.get("teacher_bestmove")
        if not bm or bm == "(none)":
            continue
        m = mate_val(lab.get("teacher_mate"))
        c = cp_val(lab.get("teacher_cp_stm"))
        if (m is not None and abs(m) <= 8) or (c is not None and abs(c) >= 500):
            fen = p["fen"]
            if fen in seen_fens:
                continue
            shard_sel.append({"fen": fen, "source": "s6-eval-shard",
                              "teacher_bestmove": bm,
                              "teacher_mate": lab.get("teacher_mate"),
                              "teacher_cp_stm": lab.get("teacher_cp_stm"),
                              "sort_key": p["position_id"]})

    shard_sel.sort(key=lambda r: r["sort_key"])
    need = TARGET - len(rows)
    if need > 0:
        for r in shard_sel[:need]:
            seen_fens.add(r["fen"])
            rows.append(r)

    rows.sort(key=lambda r: r["sort_key"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    blob = OUT.read_bytes()
    print(f"corpus rows: {len(rows)} (challenge={n_challenge}, shard={len(rows)-n_challenge})")
    print(f"mate-labelled: {sum(1 for r in rows if mate_val(r['teacher_mate']) is not None)}")
    print(f"sha256: {hashlib.sha256(blob).hexdigest()}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
