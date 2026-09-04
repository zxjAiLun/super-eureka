"""S10-H0-B2: targeted counterfactual representation-sensitivity probe.

Uses the SAME production 16k teacher contract (SF18 Windows
c86215fa..., nodes 16384, Threads=1, Hash=64, MultiPV=1,
UCI_ShowWDL=true, ucinewgame). No depth20, no 64k.

Three probes over bases drawn from the frozen 1M (each base already has
a Y16 label):
  castling — remove EXISTING rights only (never add); board/STM/ep/
             halfmove unchanged
  ep       — only positions with a LEGAL en-passant capture present;
             counterpart strips the ep square
  rule50   — naturally high halfmove-clock bases; counterpart resets
             halfmove to 0

Fail-close gate BEFORE labeling: the Eureka F128 production evaluator
(raw NNUE output, then composed) must return BIT-IDENTICAL values for
original and counterpart. Only gate-passing pairs are labeled.

Reports per probe: CP delta (clamped ±2000) median/p90/p95/max,
sign changes, bestmove changes, WDL changes, mate transitions.

Usage:
    python tools/s10/h0_b2_probe.py --out results/s10/s10-h0-b2-probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.train_nnue import export_features_from_engine  # noqa: E402

SF = (r"C:\Users\81489\AppData\Local\Temp\opencode\sf18-win\stockfish"
      r"\stockfish-windows-x86-64-avx2.exe")
EUREKA = r"target\release\eureka.exe"
MODEL128 = (r"data\s10\e3\scale-1m-win\seed-20260820"
            r"\nnue-v2-q01-material-v3twin.bin")
MODEL256 = (r"data\s10\g1\ft256-1m\seed-20260819"
            r"\nnue-v2-q01-material-256.bin")
CLIP = 2000
N_BASES = 256


def sf_label(fen: str) -> dict:
    p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    def send(cmd):
        p.stdin.write(cmd + "\n"); p.stdin.flush()
    send("uci")
    while p.stdout.readline().strip() != "uciok":
        pass
    send("setoption name Threads value 1")
    send("setoption name Hash value 64")
    send("setoption name MultiPV value 1")
    send("setoption name UCI_ShowWDL value true")
    send("ucinewgame")
    send(f"position fen {fen}")
    send("go nodes 16384")
    final = {}
    best = None
    while True:
        line = p.stdout.readline()
        if not line or line.startswith("bestmove"):
            if line.startswith("bestmove"):
                best = line.split()[1]
            break
        if not line.startswith("info") or " pv " not in line:
            continue
        m = re.search(r"score cp (-?\d+)|score mate (-?\d+)", line)
        w = re.search(r"wdl (\d+) (\d+) (\d+)", line)
        pv = re.search(r" pv (\S+)", line)
        if m:
            final["cp"] = int(m.group(1)) if m.group(1) is not None else None
            final["mate"] = (int(m.group(2))
                             if m.group(2) is not None else None)
        if w:
            final["wdl"] = [int(w.group(i)) for i in (1, 2, 3)]
        if pv:
            final["bestmove"] = pv.group(1)
    p.stdin.write("quit\n"); p.stdin.flush()
    p.wait(timeout=30)
    return {"cp": final.get("cp"), "mate": final.get("mate"),
            "bestmove": final.get("bestmove", best),
            "wdl": final.get("wdl", [None, None, None])}


def eureka_raws(fens: list[str], model: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(f"x{i}|{f}" for i, f in enumerate(fens)))
        path = fh.name
    out = subprocess.run([EUREKA, "bench", "nnue-v2q-probe-batch",
                          "--model", model, "--batch", path],
                         capture_output=True, text=True, timeout=1800,
                         check=True)
    Path(path).unlink()
    d = {json.loads(l)["position_id"]: json.loads(l)["raw_output"]
         for l in out.stdout.splitlines() if l.strip()}
    return [d[f"x{i}"] for i in range(len(fens))]


def clamp(v):
    return max(-CLIP, min(CLIP, v))


def make_castling_counterpart(fen: str) -> str | None:
    b = chess.Board(fen)
    rights = b.castling_rights
    if not rights:
        return None
    b2 = b.copy()
    b2.castling_rights = 0  # remove ALL existing rights
    b2.set_fen(b2.fen())    # normalize
    return b2.fen()


def make_ep_counterpart(fen: str) -> str | None:
    b = chess.Board(fen)
    if b.ep_square is None:
        return None
    # must have a LEGAL en-passant capture
    ep_moves = [m for m in b.legal_moves if b.is_en_passant(m)]
    if not ep_moves:
        return None
    b2 = b.copy()
    b2.ep_square = None
    return b2.fen()


def make_rule50_counterpart(fen: str) -> str | None:
    parts = fen.split()
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    if halfmove < 8:
        return None
    parts[4] = "0"
    return " ".join(parts)


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100 * (len(s) - 1)))]


def analyze(pairs: list[dict]) -> dict:
    cp_deltas = []
    sign_changes = 0
    bm_changes = 0
    wdl_changes = 0
    mate_trans = 0
    for p in pairs:
        o, c = p["orig"], p["cf"]
        if o["cp"] is not None and c["cp"] is not None:
            a, b = clamp(o["cp"]), clamp(c["cp"])
            cp_deltas.append(abs(b - a))
            if a >= 50 and b <= -50 or a <= -50 and b >= 50:
                sign_changes += 1
        else:
            mate_trans += 1
        if o["bestmove"] != c["bestmove"]:
            bm_changes += 1
        if list(o["wdl"]) != list(c["wdl"]):
            wdl_changes += 1
    n = len(pairs)
    return {
        "n_pairs": n,
        "cp_delta_median": pct(cp_deltas, 50) if cp_deltas else None,
        "cp_delta_p90": pct(cp_deltas, 90) if cp_deltas else None,
        "cp_delta_p95": pct(cp_deltas, 95) if cp_deltas else None,
        "cp_delta_max": max(cp_deltas) if cp_deltas else None,
        "sign_changes": sign_changes,
        "bestmove_changes": bm_changes,
        "wdl_changes": wdl_changes,
        "mate_or_class_transitions": mate_trans,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for shard in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob(
            "part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    rng = random.Random(2026090501)

    # ---- candidate base pools ----
    print("building candidate pools...", flush=True)
    castling_pool, ep_pool, r50_pool = [], [], []
    for r in records:
        fen = r["fen"]
        parts = fen.split()
        if "K" in parts[2] or "Q" in parts[2] or "k" in parts[2] \
                or "q" in parts[2]:
            castling_pool.append(r)
        if parts[3] != "-":
            ep_pool.append(r)
        if len(parts) > 4 and int(parts[4]) >= 20:
            r50_pool.append(r)
    print(f"pools: castling={len(castling_pool)} ep={len(ep_pool)} "
          f"rule50>=20={len(r50_pool)}", flush=True)

    # ---- build (orig, counterpart) candidates ----
    def build_pairs(pool, maker, n):
        rng.shuffle(pool)
        out = []
        for r in pool:
            if len(out) >= n:
                break
            cf = maker(r["fen"])
            if cf is None or cf == r["fen"]:
                continue
            out.append({"pid": r["position_id"], "orig_fen": r["fen"],
                        "cf_fen": cf})
        return out

    pairs_castling = build_pairs(castling_pool, make_castling_counterpart,
                                 N_BASES)
    pairs_ep = build_pairs(ep_pool, make_ep_counterpart, N_BASES)
    pairs_r50 = build_pairs(r50_pool, make_rule50_counterpart, N_BASES)
    print(f"candidates: castling={len(pairs_castling)} "
          f"ep={len(pairs_ep)} rule50={len(pairs_r50)}", flush=True)

    # ---- fail-close eval gate: F128 bit-exact on (orig, cf) ----
    print("eval gate (F128 bit-exact)...", flush=True)
    all_fens = []
    for plist in (pairs_castling, pairs_ep, pairs_r50):
        for p in plist:
            all_fens.extend([p["orig_fen"], p["cf_fen"]])
    raws = eureka_raws(all_fens, MODEL128)
    # also spot-check F256
    spot = all_fens[:64]
    raws256 = eureka_raws(spot, MODEL256)

    def gate(plist, offset):
        kept = []
        for i, p in enumerate(plist):
            o_raw = raws[offset + 2 * i]
            c_raw = raws[offset + 2 * i + 1]
            if o_raw == c_raw:
                kept.append(p)
        return kept

    off = 0
    kept_c = gate(pairs_castling, off); off += 2 * len(pairs_castling)
    kept_e = gate(pairs_ep, off); off += 2 * len(pairs_ep)
    kept_r = gate(pairs_r50, off); off += 2 * len(pairs_r50)
    print(f"gate passed: castling={len(kept_c)}/{len(pairs_castling)} "
          f"ep={len(kept_e)}/{len(pairs_ep)} "
          f"rule50={len(kept_r)}/{len(pairs_r50)}", flush=True)
    # F256 spot gate
    f256_gate = sum(1 for i in range(0, len(spot), 2)
                    if raws256[i] == raws256[i + 1])
    print(f"F256 spot gate (32 pairs): {f256_gate}/32", flush=True)

    # ---- label the counterparts with the 16k teacher ----
    print("labeling counterparts (16k teacher)...", flush=True)
    t0 = time.time()
    for plist in (kept_c, kept_e, kept_r):
        for p in plist:
            p["cf"] = sf_label(p["cf_fen"])
    wall = time.time() - t0
    print(f"labeled in {wall:.0f}s", flush=True)

    # ---- attach original Y16 labels from the dataset ----
    labels = {}
    for line in Path(
            r"data\s10\s10-eval-v2-1m01\labels.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    for plist in (kept_c, kept_e, kept_r):
        for p in plist:
            lab = labels[p["pid"]]
            p["orig"] = {"cp": lab.get("teacher_cp_stm"),
                         "mate": lab.get("teacher_mate"),
                         "bestmove": lab.get("teacher_bestmove"),
                         "wdl": lab.get("teacher_wdl_stm")}

    results = {
        "castling": analyze(kept_c),
        "ep": analyze(kept_e),
        "rule50": analyze(kept_r),
    }
    print(json.dumps(results, indent=1))

    report = {
        "schema_version": 1,
        "teacher_contract": "SF18 windows c86215fa..., nodes 16384, "
                            "identical to E2 production",
        "n_bases_requested": N_BASES,
        "gate": {
            "castling": f"{len(kept_c)}/{len(pairs_castling)}",
            "ep": f"{len(kept_e)}/{len(pairs_ep)}",
            "rule50": f"{len(kept_r)}/{len(pairs_r50)}",
            "f256_spot": f"{f256_gate}/32",
        },
        "labeling_wall_s": round(wall, 1),
        "results": results,
        "pairs": {
            "castling": kept_c,
            "ep": kept_e,
            "rule50": kept_r,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
