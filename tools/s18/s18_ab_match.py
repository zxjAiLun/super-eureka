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
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
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


def read_cutechess_elo(log: Path) -> dict:
    matches = re.findall(r"^Elo difference: ([-+\d.]+) \+/- ([\d.]+), LOS: ([\d.]+) %.*$",
                         log.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    if not matches:
        raise ValueError("missing cutechess Elo summary; do not invent a CI")
    elo, margin, los = map(float, matches[-1])
    return {"source": "cutechess-cli summary (not a paired-CI recomputation)",
            "elo": elo, "reported_ci95_half_width": margin, "los_percent": los}


def parse_pgn_and_stats(pgn_path: Path, arm_a: str, arm_b: str,
                        expected_fens: list[str] | None = None) -> dict:
    """Validate complete color-swapped pairs; attribute times using actual turn.

    No adjacency assumption: cutechess can finish games out of launch order.
    Timings are rounded PGN move times, not CPU time or total search-node work.
    """
    groups = defaultdict(list)
    times = {arm_a: [], arm_b: []}
    moves = {arm_a: 0, arm_b: 0}
    scores = []
    rounds = set()
    with pgn_path.open(encoding="utf-8") as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            h = game.headers
            if game.errors or set((h.get("White"), h.get("Black"))) != {arm_a, arm_b}:
                raise ValueError("illegal game or unexpected player")
            if h.get("Result") not in ("1-0", "0-1", "1/2-1/2") or "FEN" not in h:
                raise ValueError("unfinished game or missing opening FEN")
            if h.get("Round") in rounds:
                raise ValueError("duplicate game round")
            rounds.add(h.get("Round"))
            score = (0.5 if h["Result"] == "1/2-1/2" else
                     float((h["Result"] == "1-0") == (h["White"] == arm_a)))
            scores.append(score)
            groups[h["FEN"]].append({"round": h["Round"],
                                      "candidate_color": "white" if h["White"] == arm_a else "black",
                                      "result": h["Result"], "candidate_points": score})
            board = game.board()
            for node in game.mainline():
                engine = h["White"] if board.turn == chess.WHITE else h["Black"]
                moves[engine] += 1
                match = re.search(r"(?:^|\s)(\d+(?:\.\d+)?)s(?:\s|$)", node.comment)
                if match:
                    times[engine].append(float(match[1]))
                board.push(node.move)
    if not scores:
        raise ValueError("empty PGN")
    # cutechess format=epd consumes the first four fields and resets counters.
    expected = ({" ".join(f.split()[:4]) + " 0 1" for f in expected_fens}
                if expected_fens is not None else None)
    if expected is not None and (set(groups) != expected
                                or len(scores) != 2 * len(expected_fens)
                                or len(expected) != len(expected_fens)):
        raise ValueError("opening set or game count does not match protocol")
    penta = [0] * 5
    pairs = []
    for fen, pair in sorted(groups.items()):
        if len(pair) != 2 or {g["candidate_color"] for g in pair} != {"white", "black"}:
            raise ValueError("opening is not a complete color-swapped pair")
        penta[round(sum(g["candidate_points"] for g in pair) * 2)] += 1
        pairs.append({"fen": fen, "games": sorted(pair, key=lambda g: g["candidate_color"])})
    points = sum(scores)
    return {"games": len(scores), "candidate_W": scores.count(1.0),
            "draws": scores.count(0.5), "candidate_L": scores.count(0.0),
            "score_points": f"{points:g} / {len(scores)}",
            "score_percent": round(points / len(scores) * 100, 2),
            "pentanomial": {"counts": penta, "labels": ["LL", "LD", "DD/WL", "WD", "WW"],
                            "pairs": len(pairs)},
            "move_time": {name: {"moves": moves[name], "timed_moves": len(ts),
                                 "total_seconds": round(sum(ts), 2),
                                 "avg_move_seconds": round(sum(ts) / len(ts), 6) if ts else None}
                          for name, ts in times.items()},
            "pair_evidence": pairs}


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
