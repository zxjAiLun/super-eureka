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
import sys
import time
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1])
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from paired_pgn import parse_pgn_and_stats, read_cutechess_elo

PROFILE = "current-final"
TIME_CONTROL = "10+0.1"
HASH_MB = 16
ROUNDS = 256         # cutechess counts games per opening, so -rounds 256
                     # with -repeat 2 plays each of the 128 openings twice
                     # (one game per colour) for 256 games total.
                     # Previously this was 128, which stopped the run after
                     # 128 games and left one opening unpaired.
BOOK_START, BOOK_END = 321, 448  # 128 openings
CUTECHESS = Path("tools/.cache/cutechess-1.5.1-win64/cutechess-cli.exe")
BOOK = Path("results/s3-promotion/run-001/openings.epd")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_openings() -> list[str]:
    lines = [l.strip() for l in BOOK.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    block = lines[BOOK_START - 1:BOOK_END]
    expected = BOOK_END - BOOK_START + 1
    if len(block) != expected:
        raise SystemExit(f"expected {expected} openings, got {len(block)}")
    return block


def handshake(exe: Path, model: Path) -> dict[str, str]:
    """Run the real startup handshake and return the reported eval fields.

    A match may only start once both sides report an NNUE evaluator and the
    expected artifact. Naming a model is not the same as selecting the
    evaluator that consumes it, so the selection is read back from the
    process rather than assumed from the command line.
    """
    cmd = [str(exe.resolve()), "--profile", PROFILE,
           "--evaluation", "nnue", "--nnue-model", str(model.resolve())]
    p = subprocess.run(cmd, input="uci\nisready\nquit\n", capture_output=True,
                       text=True, timeout=120)
    fields: dict[str, str] = {}
    for line in (p.stdout + p.stderr).splitlines():
        if line.startswith("info string "):
            body = line[len("info string "):]
            key, _, val = body.partition(" ")
            if key in ("eval", "network", "evalfile", "profile", "source"):
                fields[key] = val
    if p.returncode != 0:
        raise SystemExit(f"FAIL CLOSED: handshake for {exe.name} exited {p.returncode}")
    return fields


def verify_arm(exe: Path, model: Path, tag: str) -> dict[str, str]:
    f = handshake(exe, model)
    print(f"[ab] handshake {tag}: " + "  ".join(f"{k}={v}" for k, v in f.items()),
          flush=True)
    if not f.get("eval", "").startswith("nnue"):
        raise SystemExit(
            f"FAIL CLOSED: {tag} reports eval={f.get('eval')!r}, not NNUE; "
            "the games would not exercise the supplied artifact")
    if not f.get("network", "").startswith("nnue"):
        raise SystemExit(f"FAIL CLOSED: {tag} reports network={f.get('network')!r}")
    if Path(f.get("evalfile", "")).name != model.name:
        raise SystemExit(
            f"FAIL CLOSED: {tag} loaded evalfile={f.get('evalfile')!r}, "
            f"expected {model.name!r}")
    return f


def resolve_source_commit(
    hs_a: dict[str, str],
    hs_b: dict[str, str],
    expected_commit: str | None = None,
) -> tuple[str, str]:
    """Resolve binary source provenance without querying the git repo HEAD."""
    src_a = hs_a.get("source")
    src_b = hs_b.get("source")
    if src_a and src_b:
        if src_a != src_b:
            raise SystemExit(
                f"FAIL CLOSED: source commit mismatch between arms: "
                f"arm A source {src_a!r} != arm B source {src_b!r}")
        if expected_commit and expected_commit != src_a:
            raise SystemExit(
                f"FAIL CLOSED: binary source {src_a!r} does not match "
                f"--source-commit {expected_commit!r}")
        return src_a, "binary_uci_handshake"
    if expected_commit:
        return expected_commit, "caller_supplied"
    raise SystemExit(
        "FAIL CLOSED: binaries do not report source SHA; "
        "--source-commit is required")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", type=Path, required=True,
                    help="AVX2 build (candidate)")
    ap.add_argument("--arm-b", type=Path, required=True,
                    help="force_scalar_l1 build (baseline)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--source-commit", type=str, default=None,
                    help="expected source commit SHA of the binary builds")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    openings = args.out / "ab-openings.epd"
    opening_lines = read_openings()
    openings.write_text("\n".join(opening_lines) + "\n", encoding="utf-8")
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
            # Explicit: --profile alone selects the handcrafted evaluator,
            # so the supplied model would be loaded and never used.
            "arg=--evaluation", "arg=nnue",
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
    print(f"[ab] openings={len(opening_lines)}", flush=True)
    hs_a = verify_arm(args.arm_a, args.model, "A-avx2")
    hs_b = verify_arm(args.arm_b, args.model, "B-scalar")

    source_commit, source_provenance = resolve_source_commit(
        hs_a, hs_b, args.source_commit)

    print(f"[ab] launching cutechess ({ROUNDS} games)...", flush=True)
    t0 = time.time()
    log = args.out / "ab-cutechess.log"
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, timeout=14400)
    elapsed = time.time() - t0

    stats = parse_pgn_and_stats(pgnout, "A-avx2", "B-scalar", expected_fens=opening_lines)
    stats.pop("pair_evidence", None)
    n = stats["games"]
    W = stats["candidate_W"]
    D = stats["draws"]
    L = stats["candidate_L"]
    score = W + 0.5 * D
    pct = stats["score_percent"]
    elo = -400 * math.log10(n / score - 1) if (score and score < n) else float("nan")

    cute_elo = read_cutechess_elo(log)
    elo_summary = (f"Elo difference: {cute_elo['elo']:+.1f} +/- "
                   f"{cute_elo['reported_ci95_half_width']:.1f}, "
                   f"LOS: {cute_elo['los_percent']:.1f} %")
    verdict = ("FAIL" if pct < 48 else "PARITY" if pct <= 52
               else "PROMISING")

    report = {
        "schema_version": 1,
        "stage": "s17b_ab_speed_match",
        "note": ("same commit, same S14 artifact both sides; the only "
                 "difference is the SCReLU head implementation (AVX2 vs "
                 "scalar). Fixed time control so saved time becomes search."),
        "source_commit": source_commit,
        "source_provenance": source_provenance,
        "protocol": (f"{ROUNDS} games over {len(opening_lines)} openings "
                     f"(book {BOOK_START}-{BOOK_END}), both colours, "
                     f"{TIME_CONTROL}, Hash {HASH_MB}, "
                     f"concurrency {args.concurrency}"),
        "arm_a_avx2_sha256": sha_a,
        "arm_b_scalar_sha256": sha_b,
        "model_sha256": sha_m,
        "handshake_arm_a": hs_a,
        "handshake_arm_b": hs_b,
        "returncode": 0,
        "elapsed_seconds": round(elapsed, 1),
        "result": {
            "games": n,
            "candidate_W": W,
            "draws": D,
            "candidate_L": L,
            "score_points": stats["score_points"],
            "score_percent": pct,
            "elo_descriptive": round(elo, 1) if not math.isnan(elo) else None,
            "cutechess_elo": cute_elo,
            "cutechess_elo_summary": elo_summary,
            "pentanomial": stats["pentanomial"],
            "move_time": stats["move_time"],
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
