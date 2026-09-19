#!/usr/bin/env python3
"""S15 data-expansion screen: S15 net vs S14 net, SAME binary, SAME profile.

The single variable under test is the data-supply net. Search config is held
identical on both sides (`--profile current-final`; `current-final` and
`current-final-s12` resolve to the exact same production search semantics --
see `src/engine/search.rs` PRODUCTION_PROFILE), so any score difference is
attributable to the net, not to search.

Frozen protocol (successor to S12-R0 / S13 / S14 screens, only the openings
advance):
  * same binary for both sides; TC 10+0.1; Hash 16; 1 search thread
    (the engine is inherently single-threaded -- no Threads option).
  * openings = lines 321-448 of the frozen s3-promotion book
    (R0 used 1-64, S13 used 65-192, S14 used 193-320 -> these 128 are
    disjoint from all three).
  * cutechess-cli -rounds 256 -repeat 2 order=sequential  ->  256 games,
    128 pairs (same opening, reversed colors).
  * NO SPRT. Report W/D/L + score points + score% + pentanomial + Elo+-CI.
  * Verdict bands (frozen): <48% FAIL / 48-52% PARITY / >52% PROMISING.

Usage:
  python tools/s15/s15_screen.py --concurrency 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1])
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from paired_pgn import parse_pgn_and_stats, paired_elo_ci

BOOK = Path(r"results\s3-promotion\run-001\openings.epd")
BOOK_START = 321   # 1-indexed, inclusive
BOOK_END = 448     # 1-indexed, inclusive (128 lines)
ENGINE = Path(r"target\release\eureka.exe")
ART_CAND = Path(r"data\s15\run\nnue-s15-datasupply-v5.bin")
ART_BASE = Path(r"data\s14\run\seed-20260908\nnue-s14-datasupply-v5.bin")
CUTECHESS = Path(r"tools\.cache\cutechess-1.5.1-win64\cutechess-cli.exe")
OUTDIR = Path(r"results\s15")
TIME_CONTROL = "10+0.1"
HASH_MB = 16
ROUNDS = 256
PROFILE = "current-final"
LABEL_CAND = "S15"
LABEL_BASE = "S14"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def verify_evaluator(engine: Path, model: Path, tag: str) -> dict:
    """Read the evaluator selection back from a real startup handshake.

    The regression this guards against: `--profile current-final` with
    `--nnue-model X` reports eval=handcrafted.../network=none, so a match
    could run HCE on both sides while the report named two different nets.
    Naming a model is not the same as selecting the evaluator that consumes
    it, so the selection is verified per engine before any game starts.
    """
    cmd = [str(engine.resolve()), "--profile", PROFILE,
           "--evaluation", "nnue", "--nnue-model", str(model.resolve())]
    p = subprocess.run(cmd, input="uci\nisready\nquit\n",
                       capture_output=True, text=True, timeout=180)
    fields = {}
    for line in (p.stdout + p.stderr).splitlines():
        if line.startswith("info string "):
            key, _, val = line[len("info string "):].partition(" ")
            if key in ("eval", "network", "evalfile", "profile"):
                fields[key] = val
    if p.returncode != 0:
        raise SystemExit(f"FAIL CLOSED: handshake {tag} exited {p.returncode}")
    if not fields.get("eval", "").startswith("nnue"):
        raise SystemExit(
            f"FAIL CLOSED: {tag} reports eval={fields.get('eval')!r}; "
            "the supplied artifact would not be used")
    if not fields.get("network", "").startswith("nnue"):
        raise SystemExit(f"FAIL CLOSED: {tag} reports network={fields.get('network')!r}")
    if Path(fields.get("evalfile", "")).name != model.name:
        raise SystemExit(
            f"FAIL CLOSED: {tag} loaded evalfile={fields.get('evalfile')!r}, "
            f"expected {model.name!r}")
    return fields


def build_openings(out: Path) -> list[str]:
    lines = BOOK.read_text(encoding="utf-8").splitlines()
    if len(lines) < BOOK_END:
        raise SystemExit(f"FAIL CLOSED: book has {len(lines)} lines < {BOOK_END}")
    sl = lines[BOOK_START - 1:BOOK_END]           # 321..448 inclusive
    if len(sl) != 128:
        raise SystemExit(f"FAIL CLOSED: slice is {len(sl)} lines, expected 128")
    # disjointness from R0 (1-64), S13 (65-192) and S14 (193-320) is
    # guaranteed by the index window; assert the boundaries as a tripwire.
    assert sl[0] == lines[320] and sl[-1] == lines[447]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sl) + "\n", encoding="utf-8")
    return sl


def build_command(openings: Path, pgnout: Path, concurrency: int,
                  art_cand: Path = ART_CAND,
                  art_base: Path = ART_BASE,
                  label_cand: str = LABEL_CAND,
                  label_base: str = LABEL_BASE,
                  engine: Path = ENGINE) -> list[str]:
    eng = str(engine.resolve())
    return [
        str(CUTECHESS.resolve()),
        "-engine", f"name={label_cand}", f"cmd={eng}", "proto=uci",
        "arg=--profile", f"arg={PROFILE}",
        # Explicit: --profile alone selects the handcrafted evaluator, so a
        # supplied --nnue-model would be loaded and never used.
        "arg=--evaluation", "arg=nnue",
        "arg=--nnue-model", f"arg={art_cand.resolve()}",
        "-engine", f"name={label_base}", f"cmd={eng}", "proto=uci",
        "arg=--profile", f"arg={PROFILE}",
        "arg=--evaluation", "arg=nnue",
        "arg=--nnue-model", f"arg={art_base.resolve()}",
        "-variant", "standard",
        "-openings", f"file={openings.resolve()}", "format=epd",
        "order=sequential", "policy=default",
        "-each", f"tc={TIME_CONTROL}", f"option.Hash={HASH_MB}",
        "-rounds", str(ROUNDS), "-repeat", "2",
        "-concurrency", str(concurrency),
        "-pgnout", str(pgnout.resolve()),
        "-resultformat", "short",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUTDIR)
    ap.add_argument("--candidate-art", type=Path, default=ART_CAND,
                    help="candidate net (default: S15 60M-run net)")
    ap.add_argument("--baseline-art", type=Path, default=ART_BASE,
                    help="baseline net (default: S14 production net)")
    ap.add_argument("--engine", type=Path, default=ENGINE,
                    help="engine binary used for BOTH sides")
    ap.add_argument("--label-cand", default=LABEL_CAND,
                    help="candidate engine label in the PGN")
    ap.add_argument("--label-base", default=LABEL_BASE,
                    help="baseline engine label in the PGN")
    args = ap.parse_args()
    art_cand = args.candidate_art
    art_base = args.baseline_art
    engine = args.engine
    label_cand = args.label_cand
    label_base = args.label_base
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    openings = outdir / "screen-openings.epd"
    pgnout = outdir / "screen-match.pgn"
    if pgnout.exists() and pgnout.stat().st_size > 0:
        raise SystemExit(f"FAIL CLOSED: {pgnout} already exists (non-empty)")

    sl = build_openings(openings)
    print(f"[screen] openings={len(sl)} (book {BOOK_START}-{BOOK_END}) -> "
          f"{openings}", flush=True)
    sha_c = sha256_file(art_cand)
    sha_b = sha256_file(art_base)
    eng_sha = sha256_file(engine)
    print(f"[screen] candidate {label_cand} net sha256={sha_c}", flush=True)
    print(f"[screen] baseline  {label_base} net sha256={sha_b}", flush=True)
    print(f"[screen] engine sha256={eng_sha}", flush=True)

    hs_c = verify_evaluator(engine, art_cand, f"candidate {label_cand}")
    hs_b = verify_evaluator(engine, art_base, f"baseline {label_base}")
    print(f"[screen] handshake {label_cand}: " +
          "  ".join(f"{k}={v}" for k, v in hs_c.items()), flush=True)
    print(f"[screen] handshake {label_base}: " +
          "  ".join(f"{k}={v}" for k, v in hs_b.items()), flush=True)

    cmd = build_command(openings, pgnout, args.concurrency, art_cand, art_base,
                        label_cand, label_base, engine)
    print("[screen] launching cutechess (256 games)...", flush=True)
    log = outdir / "screen-cutechess.log"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              text=True, timeout=14400, check=True)
    print(f"[screen] cutechess exit={proc.returncode}", flush=True)

    stats = parse_pgn_and_stats(pgnout, label_cand, label_base, expected_fens=sl)
    pairs = stats.pop("pair_evidence")
    n = stats["games"]
    W = stats["candidate_W"]
    D = stats["draws"]
    L = stats["candidate_L"]
    score_points = W + 0.5 * D
    score_pct = stats["score_percent"]
    penta = stats["pentanomial"]["counts"]
    npairs = stats["pentanomial"]["pairs"]

    elo, elo_ci95 = paired_elo_ci(pairs, score_points, n)

    if score_pct < 48.0:
        verdict = "FAIL"
    elif score_pct <= 52.0:
        verdict = "PARITY"
    else:
        verdict = "PROMISING"

    report = {
        "schema_version": 1,
        "stage": "s15_screen",
        "protocol": (f"256 games = 128 fresh opening pairs (book lines "
                     f"{BOOK_START}-{BOOK_END} of the frozen s3-promotion "
                     f"openings, disjoint from R0 1-64, S13 65-192 and S14 "
                     f"193-320), both colors, same binary, same profile "
                     f"({PROFILE}), {TIME_CONTROL}, Hash {HASH_MB}, "
                     f"1 search thread, concurrency {args.concurrency}"),
        "single_variable": ("both engines identical except --nnue-model; the "
                            "only difference under test is the data-supply "
                            "net"),
        "engines": {
            "candidate": f"{PROFILE} + {label_cand} net ({sha_c[:8]}...)",
            "baseline": f"{PROFILE} + {label_base} net ({sha_b[:8]}...)",
        },
        "artifact_sha256_candidate": sha_c,
        "artifact_sha256_baseline": sha_b,
        "engine_sha256": eng_sha,
        "result": {
            "games": n,
            "games_total_in_pgn": n,
            "unfinished": 0,
            "candidate_W": W,
            "draws": D,
            "candidate_L": L,
            "score_points": f"{score_points:g} / {n}",
            "score_percent": round(score_pct, 2),
            "elo_descriptive": elo,
            "elo_ci95": elo_ci95,
            "pentanomial_0to2": penta,
            "pairs": npairs,
        },
        "verdict_bands": "<48% FAIL / 48-52% PARITY / >52% PROMISING",
        "verdict": verdict,
    }
    (outdir / "screen-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(report["result"], indent=1), flush=True)
    print(f"[screen] VERDICT: {verdict} ({score_pct:.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
