#!/usr/bin/env python3
"""Reproducible S18-QC1 paired screen runner / offline report reader.

QC1 is KILLED: retaining a runner is not permission to re-open the experiment.
256 games, 128 openings (321-448), 10+0.1, Hash16, inherently single-threaded.
CLI launches a match only when explicitly invoked with both arms and a model.
parse_pgn_and_stats / read_cutechess_elo support offline auditing without engines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
_TOOLS = str(Path(__file__).resolve().parents[1])
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from paired_pgn import parse_pgn_and_stats, read_cutechess_elo

PROFILE, TIME_CONTROL, HASH_MB, ROUNDS = "current-final", "10+0.1", 16, 256
BOOK_START, BOOK_END = 321, 448
CUTECHESS = ROOT / "tools/.cache/cutechess-1.5.1-win64/cutechess-cli.exe"
BOOK = ROOT / "results/s3-promotion/run-001/openings.epd"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_openings() -> list[str]:
    lines = [x.strip() for x in BOOK.read_text(encoding="utf-8").splitlines()
             if x.strip() and not x.startswith("#")]
    block = lines[BOOK_START - 1:BOOK_END]
    if len(block) != 128 or len(set(block)) != 128:
        raise ValueError("expected 128 distinct openings")
    return block


def verify_arm(exe: Path, model: Path, tag: str) -> dict:
    cmd = [str(exe.resolve()), "--profile", PROFILE, "--evaluation", "nnue",
           "--nnue-model", str(model.resolve())]
    p = subprocess.run(cmd, input="uci\nisready\nquit\n", capture_output=True,
                       text=True, timeout=120, check=True)
    fields = {}
    for line in (p.stdout + p.stderr).splitlines():
        if line.startswith("info string "):
            key, _, value = line[len("info string "):].partition(" ")
            if key in ("eval", "network", "evalfile", "profile"):
                fields[key] = value
    if ("uciok" not in p.stdout.splitlines() or "readyok" not in p.stdout.splitlines()
            or fields.get("profile") != PROFILE
            or not fields.get("eval", "").startswith("nnue")
            or not fields.get("network", "").startswith("nnue")
            or Path(fields.get("evalfile", "")).name != model.name):
        raise ValueError(f"FAIL CLOSED: {tag} handshake {fields}")
    return fields


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-a", type=Path, required=True, help="QC1 build")
    ap.add_argument("--arm-b", type=Path, required=True, help="baseline build")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source-commit", required=True, help="build source identity (caller supplied)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    if args.concurrency < 1:
        ap.error("concurrency must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    pgn = args.out / "qc1-match.pgn"
    report_path = args.out / "qc1-match-report.json"
    if pgn.exists() or report_path.exists():
        raise ValueError("FAIL CLOSED: refusing to overwrite an existing run")
    if sha256_file(args.arm_a) == sha256_file(args.arm_b):
        raise ValueError("identical arm binaries")
    hs_a = verify_arm(args.arm_a, args.model, "A-qc1")
    hs_b = verify_arm(args.arm_b, args.model, "B-base")
    lines = read_openings()
    openings = args.out / "qc1-openings.epd"
    openings.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def engine(name, exe):
        return ["-engine", f"name={name}", f"cmd={exe.resolve()}", "proto=uci",
                "arg=--profile", f"arg={PROFILE}", "arg=--evaluation", "arg=nnue",
                "arg=--nnue-model", f"arg={args.model.resolve()}"]

    cmd = [str(CUTECHESS), *engine("A-qc1", args.arm_a), *engine("B-base", args.arm_b),
           "-variant", "standard", "-openings", f"file={openings.resolve()}",
           "format=epd", "order=sequential", "policy=default", "-each",
           f"tc={TIME_CONTROL}", f"option.Hash={HASH_MB}", "-rounds", str(ROUNDS),
           "-repeat", "2", "-concurrency", str(args.concurrency),
           "-pgnout", str(pgn.resolve()), "-resultformat", "short"]
    log = args.out / "qc1-cutechess.log"
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=14400)
    elapsed = time.monotonic() - start
    stats = parse_pgn_and_stats(pgn, "A-qc1", "B-base", lines)
    pairs = stats.pop("pair_evidence")
    pgn_sha = sha256_file(pgn)
    (args.out / "qc1-pairs.json").write_text(json.dumps(
        {"schema_version": 1, "pgn_sha256": pgn_sha, "pairs": pairs}, indent=1) + "\n", encoding="utf-8")
    report = {"schema_version": 2, "stage": "s18_qc1_screen",
              "commit": args.source_commit, "source_identity": "caller supplied; check build provenance",
              "protocol": f"256 games / 128 paired openings 321-448 / 10+0.1 / Hash16 / single-thread / concurrency {args.concurrency}",
              "arm_a_qc1_sha256": sha256_file(args.arm_a), "arm_b_base_sha256": sha256_file(args.arm_b),
              "model_sha256": sha256_file(args.model), "handshake_arm_a": hs_a, "handshake_arm_b": hs_b,
              "returncode": 0, "elapsed_seconds": round(elapsed, 1), "result": stats,
              "elo_statistics": read_cutechess_elo(log), "pgn_sha256": pgn_sha,
              "log_sha256": sha256_file(log), "openings_sha256": sha256_file(openings),
              "verdict": "FAIL" if stats["score_percent"] < 48 else "PARITY" if stats["score_percent"] <= 52 else "PROMISING",
              "decision_rule": "<48% KILL / parity unproven / positive screen not production approval"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
