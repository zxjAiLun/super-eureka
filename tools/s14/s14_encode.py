#!/usr/bin/env python3
"""S14-A Layer 2: encode assembled Fishtest records into training shards.

Per record (compact JSONL.gz, one line):
    stm:   comma-joined sparse V2+R12 indices (mover perspective)
    nstm:  comma-joined indices (opponent perspective)
    m:     material_cp_stm
    b:     MaterialCount output bucket (occ incl kings - 2) // 4
    cp:    teacher_cp_stm (clamped +-2000; raw kept in the raw field)
    raw:   teacher_cp_raw (audit)

Feature extraction + material: the engine's own batch exporters (single
semantic source). Order: assembled-00..04, line order (deterministic).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, r"tools\s12")
import s12_train  # noqa: E402  (output_bucket)
import chess  # noqa: E402

ENGINE = r"target\release\eureka.exe"
SRC = sorted(glob.glob(r"data\s14\assembled\assembled-*.jsonl.gz"))
OUT = Path(r"data\s14\encoded")
CHUNK = 500_000
OUT.mkdir(parents=True, exist_ok=True)


def run_batch(cmd_extra: list, batch: Path):
    proc = subprocess.run([ENGINE, "bench"] + cmd_extra
                          + ["--batch", str(batch)],
                          capture_output=True, text=True, timeout=7200,
                          check=True)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
            out[rec["position_id"]] = rec
    return out


def main() -> int:
    shard_no = 0
    t0 = time.time()
    total = 0
    buf = []
    for src in SRC:
        with gzip.open(src, "rt") as fh:
            pending = []
            for line in fh:
                r = json.loads(line)
                pending.append(r)
                if len(pending) >= CHUNK:
                    shard_no = flush(pending, shard_no, t0)
                    total += len(pending)
                    pending = []
            if pending:
                shard_no = flush(pending, shard_no, t0)
                total += len(pending)
    print(f"ENCODED {total} records in {(time.time()-t0)/60:.1f} min "
          f"-> {shard_no} shards")
    return 0


def flush(records: list, shard_no: int, t0: float) -> int:
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False,
            encoding="utf-8") as fh:
        for i, r in enumerate(records):
            fh.write(f"p{i}|{r['fen_before']}\n")
        batch = Path(fh.name)
    try:
        feats = run_batch(["nnue-features-batch", "--feature-set", "v2r12"],
                          batch)
        mats = run_batch(["material-batch"], batch)
    finally:
        batch.unlink(missing_ok=True)

    path = OUT / f"encoded-{shard_no:03d}.jsonl.gz"
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for i, r in enumerate(records):
            f = feats[f"p{i}"]
            m = mats[f"p{i}"]["material_cp_stm"]
            stm_white = r["fen_before"].split()[1] == "w"
            white, black = f["white"], f["black"]
            stm = white if stm_white else black
            nstm = black if stm_white else white
            board = chess.Board(r["fen_before"])
            b = s12_train.output_bucket(len(board.piece_map()))
            out.write(json.dumps({
                "stm": ",".join(map(str, stm)),
                "nstm": ",".join(map(str, nstm)),
                "m": m, "b": b,
                "cp": max(-2000.0, min(2000.0, r["teacher_cp_raw"])),
                "raw": r["teacher_cp_raw"],
            }, separators=(",", ":")) + "\n")
            n += 1
    shard_no += 1
    el = time.time() - t0
    print(f"  encoded shard {shard_no}: {n} records "
          f"({n/el/1000:.0f}k/s avg)", flush=True)
    return shard_no


if __name__ == "__main__":
    raise SystemExit(main())
