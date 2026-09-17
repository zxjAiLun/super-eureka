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
from pathlib import Path

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
                  label_cand: str = LABEL_CAND,
                  label_base: str = LABEL_BASE) -> list[str]:
    eng = str(ENGINE.resolve())
    return [
        str(CUTECHESS.resolve()),
        "-engine", f"name={label_cand}", f"cmd={eng}", "proto=uci",
        "arg=--profile", f"arg={PROFILE}",
        "arg=--nnue-model", f"arg={art_cand.resolve()}",
        "-engine", f"name={label_base}", f"cmd={eng}", "proto=uci",
        "arg=--profile", f"arg={PROFILE}",
        "arg=--nnue-model", f"arg={ART_BASE.resolve()}",
        "-variant", "standard",
        "-openings", f"file={openings.resolve()}", "format=epd",
        "order=sequential", "policy=default",
        "-each", f"tc={TIME_CONTROL}", f"option.Hash={HASH_MB}",
        "-rounds", str(ROUNDS), "-repeat", "2",
        "-concurrency", str(concurrency),
        "-pgnout", str(pgnout.resolve()),
        "-resultformat", "short",
    ]


def parse_pgn(pgn: Path) -> list[tuple[str, str, str]]:
    """Return per-game (white, black, result) in file order."""
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


def score_for(cand: str, white: str, black: str, result: str):
    """Candidate points for one game (1 win / 0.5 draw / 0 loss), or None
    for an unfinished/unknown result."""
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if white == cand else 0.0
    if result == "0-1":
        return 1.0 if black == cand else 0.0
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUTDIR)
    ap.add_argument("--candidate-art", type=Path, default=ART_CAND,
                    help="candidate net (default: S15 60M-run net)")
    ap.add_argument("--label-cand", default=LABEL_CAND,
                    help="candidate engine label in the PGN")
    ap.add_argument("--label-base", default=LABEL_BASE,
                    help="baseline engine label in the PGN")
    args = ap.parse_args()
    art_cand = args.candidate_art
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
    sha_b = sha256_file(ART_BASE)
    eng_sha = sha256_file(ENGINE)
    print(f"[screen] candidate {label_cand} net sha256={sha_c}", flush=True)
    print(f"[screen] baseline  {label_base} net sha256={sha_b}", flush=True)
    print(f"[screen] engine sha256={eng_sha}", flush=True)

    cmd = build_command(openings, pgnout, args.concurrency, art_cand,
                        label_cand, label_base)
    print("[screen] launching cutechess (256 games)...", flush=True)
    log = outdir / "screen-cutechess.log"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              text=True, timeout=14400)
    print(f"[screen] cutechess exit={proc.returncode}", flush=True)

    games = parse_pgn(pgnout)
    n_total = len(games)
    if n_total != 256:
        print(f"[screen] WARNING: parsed {n_total} games (expected 256)",
              flush=True)
    W = D = L = 0
    unfinished = 0
    pts_per_game = []
    for (white, black, result) in games:
        s = score_for(label_cand, white, black, result)
        pts_per_game.append(s)
        if s is None:
            unfinished += 1
        elif s == 1.0:
            W += 1
        elif s == 0.5:
            D += 1
        else:
            L += 1
    n = W + D + L                       # finished games (score denominator)
    score_points = W + 0.5 * D
    score_pct = 100.0 * score_points / n if n else 0.0

    # pentanomial over consecutive pairs (repeat 2); only complete pairs
    penta = [0, 0, 0, 0, 0]   # [0, 0.5, 1, 1.5, 2]
    pair_norm = []
    for i in range(0, len(pts_per_game) - 1, 2):
        a, b = pts_per_game[i], pts_per_game[i + 1]
        if a is None or b is None:
            continue
        pair = a + b
        idx = int(round(pair * 2))    # 0,1,2,3,4
        penta[idx] += 1
        pair_norm.append(pair / 2.0)  # normalized [0,1]
    npairs = len(pair_norm)

    # Elo + 95% CI from the pentanomial (paired) variance.
    p = score_points / n if n else 0.0
    if 0.0 < p < 1.0:
        elo = -400.0 * math.log10(1.0 / p - 1.0)
    else:
        elo = float("inf") if p >= 1.0 else float("-inf")
    if npairs > 1 and 0.0 < p < 1.0:
        mean = sum(pair_norm) / npairs
        var = sum((x - mean) ** 2 for x in pair_norm) / (npairs - 1)
        se_p = math.sqrt(var / npairs)
        d_elo_dp = 400.0 / (math.log(10) * p * (1.0 - p))
        elo_ci95 = 1.96 * se_p * d_elo_dp
    else:
        elo_ci95 = float("nan")

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
            "games_total_in_pgn": n_total,
            "unfinished": unfinished,
            "candidate_W": W,
            "draws": D,
            "candidate_L": L,
            "score_points": f"{score_points:g} / {n}",
            "score_percent": round(score_pct, 2),
            "elo_descriptive": round(elo, 1),
            "elo_ci95": round(elo_ci95, 1) if not math.isnan(elo_ci95)
            else None,
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
