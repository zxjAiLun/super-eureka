#!/usr/bin/env python3
"""S17b timed match: same commit, two builds, SAME model on both sides.

Purpose: measure whether the AVX2 SCReLU head turns into real playing
strength when the saved computation is spent on extra search. This is a
perf comparison, NOT a model comparison:

  * both engines use the SAME S14 artifact (identical sha256);
  * both binaries are built from the SAME commit, differing only in
    `force_scalar_l1` (arm B) vs the AVX2 head (arm A);
  * a fixed time control (not fixed depth), so arm A's savings become
    additional nodes rather than dead time.

Because the two sides are behaviourally identical per node, a PARITY
result does NOT discard the optimisation: the adoption criterion is
"integer-identical + stable whole-search speedup", which this match
informs rather than decides.

Usage:
    python tools/s17/s17_ab_match.py \
        --arm-a data/s17/armA-avx2.exe --arm-b data/s17/armB-scalar.exe \
        --model <s14.bin> --out results/s17-ab --concurrency 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

PROFILE = "current-final"
TIME_CONTROL = "10+0.1"
HASH_MB = 16
ROUNDS = 128          # 128 opening pairs -> 256 games (repeat 2)
BOOK_START, BOOK_END = 321, 448
CUTECHESS = Path("tools/.cache/cutechess-1.5.1-win64/cutechess-cli.exe")
BOOK = Path("results/s3-promotion/run-001/openings.epd")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_openings() -> list[str]:
    lines = [l.strip() for l in BOOK.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    block = lines[BOOK_START - 1:BOOK_END]
    if len(block) != ROUNDS:
        raise SystemExit(f"expected {ROUNDS} openings, got {len(block)}")
    return block


def parse_pgn(pgn: Path) -> list[tuple[str, str, str]]:
    games = []
    white = black = result = None
    for line in pgn.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[White "):
            white = line.split('"')[1]
        elif line.startswith("[Black "):
            black = line.split('"')[1]
        elif line.startswith("[Result "):
            result = line.split('"')[1]
            if white is not None and black is not None:
                games.append((white, black, result))
                white = black = result = None
    return games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", type=Path, required=True,
                    help="AVX2 build (candidate)")
    ap.add_argument("--arm-b", type=Path, required=True,
                    help="force_scalar_l1 build (baseline)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    openings = args.out / "ab-openings.epd"
    openings.write_text("\n".join(read_openings()) + "\n", encoding="utf-8")
    pgnout = args.out / "ab-match.pgn"
    if pgnout.exists() and pgnout.stat().st_size > 0:
        raise SystemExit(f"FAIL CLOSED: {pgnout} exists (non-empty)")

    sha_a = sha256_file(args.arm_a)
    sha_b = sha256_file(args.arm_b)
    sha_m = sha256_file(args.model)
    if sha_a == sha_b:
        raise SystemExit("FAIL CLOSED: the two arms are the same binary")
    print(f"[ab] model  {args.model.name} sha256={sha_m}", flush=True)
    print(f"[ab] arm A  AVX2   sha256={sha_a}", flush=True)
    print(f"[ab] arm B  scalar sha256={sha_b}", flush=True)
    if args.arm_a.suffix == ".exe" and args.arm_b.suffix == ".exe":
        pass

    def engine(name: str, exe: Path) -> list[str]:
        return [
            "-engine", f"name={name}", f"cmd={exe.resolve()}", "proto=uci",
            "arg=--profile", f"arg={PROFILE}",
            "arg=--nnue-model", f"arg={args.model.resolve()}",
        ]

    cmd = [
        str(CUTECHESS.resolve()),
        *engine("A-avx2", args.arm_a),
        *engine("B-scalar", args.arm_b),
        "-variant", "standard",
        "-openings", f"file={openings.resolve()}", "format=epd",
        "order=sequential", "policy=default",
        "-each", f"tc={TIME_CONTROL}", f"option.Hash={HASH_MB}",
        "-rounds", str(ROUNDS), "-repeat", "2",
        "-concurrency", str(args.concurrency),
        "-pgnout", str(pgnout.resolve()),
        "-resultformat", "short",
    ]
    print(f"[ab] openings={ROUNDS} (book {BOOK_START}-{BOOK_END})", flush=True)
    print(f"[ab] launching cutechess ({ROUNDS * 2} games)...", flush=True)
    t0 = time.time()
    log = args.out / "ab-cutechess.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    games = parse_pgn(pgnout)
    W = D = L = 0
    for white, black, res in games:
        if res == "1/2-1/2":
            D += 1
        elif res == "1-0":
            W += 1 if white == "A-avx2" else 0
            L += 1 if white != "A-avx2" else 0
        elif res == "0-1":
            W += 1 if black == "A-avx2" else 0
            L += 1 if black != "A-avx2" else 0
    n = W + D + L
    score = W + 0.5 * D
    pct = score / n * 100 if n else float("nan")
    elo = -400 * math.log10(n / score - 1) if score else float("nan")
    ci = 1.96 * 800 / math.sqrt(n) if n else float("nan")
    verdict = ("FAIL" if pct < 48 else "PARITY" if pct <= 52
               else "PROMISING")

    report = {
        "schema_version": 1,
        "stage": "s17b_ab_speed_match",
        "note": ("same commit, same S14 artifact both sides; the only "
                 "difference is the SCReLU head implementation (AVX2 vs "
                 "scalar). Fixed time control so saved time becomes search."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=Path(__file__).resolve().parents[2]
                                 ).stdout.strip(),
        "protocol": (f"{ROUNDS} opening pairs (book {BOOK_START}-{BOOK_END}), "
                     f"both colours, {TIME_CONTROL}, Hash {HASH_MB}, "
                     f"concurrency {args.concurrency}"),
        "arm_a_avx2_sha256": sha_a,
        "arm_b_scalar_sha256": sha_b,
        "model_sha256": sha_m,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
        "result": {
            "games": n,
            "candidate_W": W,
            "draws": D,
            "candidate_L": L,
            "score_points": f"{score:g} / {n}",
            "score_percent": round(pct, 2),
            "elo_descriptive": round(elo, 1),
            "elo_ci95": round(ci, 1),
        },
        "verdict_bands": "<48% FAIL / 48-52% PARITY / >52% PROMISING",
        "verdict": verdict,
    }
    (args.out / "ab-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(report["result"], indent=1), flush=True)
    print(f"[ab] VERDICT: {verdict} ({pct:.2f}%)  "
          f"elapsed {elapsed/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
