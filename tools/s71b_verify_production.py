#!/usr/bin/env python3
"""S7.1B verification: production `current-final` must be UNCHANGED by the
mere ADDITION of the `current-final-qsearch-delta` candidate.

Runs the 30-position S4 compute corpus at fixed depth 6 (cold TT, threads 1)
and records exact nodes / score / bestmove / PV. Run once BEFORE the
implementation (`--tag before`) and once AFTER (`--tag after`); the two JSON
files must be byte-identical in the gated fields.

Writes results/s7/s71b-verification-<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S4_EPD = REPO / "tools/data/s4_compute_positions.epd"
DEPTH = 6


def run(engine: Path, fen: str) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", "current-final",
           "--depth", str(DEPTH), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True}
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return {"nodes": toks["nodes"], "score": toks["score"],
                    "bestmove": toks["bestmove"], "pv": toks["pv"]}
    return {"error": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--tag", required=True, choices=["before", "after"])
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    rows = []
    for line in S4_EPD.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen = line.split(";", 1)[0].strip()
        rows.append({"fen": fen, "result": run(engine, fen)})
        print(f"s71b_verify tag={args.tag} n={len(rows)}", flush=True)

    out = {"profile": "current-final", "depth": DEPTH, "tag": args.tag,
           "positions": rows}
    path = REPO / f"results/s7/s71b-verification-{args.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"s71b_verify wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s71b_verify_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
