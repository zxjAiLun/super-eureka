#!/usr/bin/env python3
"""S15 gate: does the CURRENT engine binary reproduce the S14 feature
encoding bit-for-bit?

Why this exists: the S14 fresh-Fishtest positions were encoded with the
engine binary that existed on 2026-09-09 (`target/release/eureka.exe`).
The expanded-data round encodes NEW positions with the CURRENT binary.
If `--feature-set v2r12` indices, `material_cp_stm`, or the output bucket
changed in the meantime, the new data would silently mix two different
label/feature conventions into one pool.

Method (no heuristics, exact equality):
  * take the first N records of an `assembled` shard (raw FENs, in scan
    order) -- these are the SAME positions, in the SAME order, as the
    corresponding `encoded` shard (encoder CHUNK == 500,000 and both
    streams preserve order);
  * re-encode those FENs with the CURRENT engine binary via
    `bench nnue-features-batch --feature-set v2r12` + `bench material-batch`;
  * compare stm / nstm / m / b field-by-field against the stored shard.

Exit 0 = identical. Any mismatch prints the first few offenders and exits 1.

Usage:
  python tools/s14/s15_encoding_gate.py [--n 20000]
        [--assembled data/s14/assembled/assembled-00.jsonl.gz]
        [--encoded   data/s14/encoded/encoded-000.jsonl.gz]
        [--engine    target/release/eureka.exe]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/s12"))
import s12_train  # noqa: E402


def run_batch(engine: Path, cmd_extra: list, batch: Path) -> dict:
    proc = subprocess.run(
        [str(engine), "bench"] + cmd_extra + ["--batch", str(batch)],
        capture_output=True, text=True, timeout=7200, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
            out[rec["position_id"]] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--assembled", type=Path,
                    default=ROOT / "data/s14/assembled/assembled-00.jsonl.gz")
    ap.add_argument("--encoded", type=Path,
                    default=ROOT / "data/s14/encoded/encoded-000.jsonl.gz")
    ap.add_argument("--engine", type=Path,
                    default=ROOT / "target/release/eureka.exe")
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")

    if not args.engine.exists():
        raise SystemExit(f"FAIL CLOSED: engine not found: {args.engine}")

    engine_sha = hashlib.sha256(args.engine.read_bytes()).hexdigest()

    # ---- read N raw FENs (scan order) and their stored encoding ----
    fens: list[str] = []
    with gzip.open(args.assembled, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fens.append(json.loads(line)["fen_before"])
            if len(fens) >= args.n:
                break

    stored: list[dict] = []
    with gzip.open(args.encoded, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            stored.append(json.loads(line))
            if len(stored) >= len(fens):
                break

    if not fens:
        raise SystemExit("FAIL CLOSED: assembled shard is empty")
    if len(stored) != len(fens):
        raise SystemExit(
            f"FAIL CLOSED: assembled gave {len(fens)} fens but encoded gave "
            f"{len(stored)} records; alignment assumption broken")

    print(f"[gate] engine sha256 = {engine_sha}")
    print(f"[gate] comparing {len(fens)} positions "
          f"({args.assembled.name} vs {args.encoded.name})", flush=True)

    # ---- re-encode with the CURRENT binary ----
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        for i, fen in enumerate(fens):
            fh.write(f"p{i}|{fen}\n")
        batch = Path(fh.name)
    try:
        feats = run_batch(args.engine,
                          ["nnue-features-batch", "--feature-set", "v2r12"],
                          batch)
        mats = run_batch(args.engine, ["material-batch"], batch)
    finally:
        batch.unlink(missing_ok=True)

    # ---- exact comparison ----
    mism_stm = mism_nstm = mism_m = mism_b = 0
    samples: list[str] = []
    for i, fen in enumerate(fens):
        rec = stored[i]
        pid = f"p{i}"
        f = feats[pid]
        stm_white = fen.split()[1] == "w"
        white, black = f["white"], f["black"]
        exp_stm = white if stm_white else black
        exp_nstm = black if stm_white else white
        got_stm = [int(x) for x in rec["stm"].split(",")]
        got_nstm = [int(x) for x in rec["nstm"].split(",")]
        if exp_stm != got_stm:
            mism_stm += 1
            if len(samples) < 3:
                samples.append(f"{pid} stm: current={exp_stm[:8]} "
                               f"stored={got_stm[:8]}")
        if exp_nstm != got_nstm:
            mism_nstm += 1
            if len(samples) < 3:
                samples.append(f"{pid} nstm: current={exp_nstm[:8]} "
                               f"stored={got_nstm[:8]}")
        exp_m = mats[pid]["material_cp_stm"]
        if abs(float(exp_m) - float(rec["m"])) > 1e-6:
            mism_m += 1
            if len(samples) < 3:
                samples.append(f"{pid} m: current={exp_m} stored={rec['m']}")
        npieces = len(chess.Board(fen).piece_map())
        exp_b = s12_train.output_bucket(npieces)
        if exp_b != int(rec["b"]):
            mism_b += 1
            if len(samples) < 3:
                samples.append(f"{pid} b: current={exp_b} stored={rec['b']}")

    total = len(fens)
    bad = mism_stm + mism_nstm + mism_m + mism_b
    print(f"[gate] positions={total}")
    print(f"[gate] stm mismatches   = {mism_stm}")
    print(f"[gate] nstm mismatches  = {mism_nstm}")
    print(f"[gate] material mism    = {mism_m}")
    print(f"[gate] bucket mism      = {mism_b}")
    for s in samples:
        print(f"[gate]   {s}")

    if bad:
        print("[gate] VERDICT: FAIL -- current binary does NOT reproduce the "
              "S14 encoding; do NOT mix new data into the pool")
        return 1
    print("[gate] VERDICT: PASS -- current binary reproduces the S14 v2r12 "
          "encoding exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
