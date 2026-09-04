"""S10-H0-A2 Engineering Pilot: one new budget tier (64k nodes) on the
frozen H0-A1 corpus.

Corpus: the EXACT matched 500 positions from H0-A1 (no resampling).
Existing: Y16 (nodes 16384, from labels.jsonl) + YD20 (depth 20).
New work: Y64 = SF18 Windows teacher binary, Threads=1, Hash=64,
MultiPV=1, UCI_ShowWDL=true, ucinewgame, go nodes 65536.

Reports ONLY:
  cp MAE / median / p95 for Y16<->YD20, Y64<->YD20, Y16<->Y64
  bestmove agreement (Y16 vs Y64)
  WDL agreement (Y16 vs Y64)
  sign agreement |ref|>=50
  phase breakdown high/mid/low/zero
  wall time + positions/sec

No 256k. No 1M. No new depth20. No new sampling. No training.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SF = (r"C:\Users\81489\AppData\Local\Temp\opencode\sf18-win\stockfish"
      r"\stockfish-windows-x86-64-avx2.exe")
CLIP = 2000
NODES = 65536

PHASE_WEIGHTS = {"n": 1, "b": 1, "r": 2, "q": 4}


def phase_of(fen):
    import chess
    b = chess.Board(fen)
    return sum(PHASE_WEIGHTS.get(pc.symbol().lower(), 0)
               for pc in b.piece_map().values())


def phase_bucket(p):
    return "high" if p >= 18 else "mid" if p >= 8 else \
        "low" if p >= 1 else "zero"


def clamp(v):
    return max(-CLIP, min(CLIP, v))


def label_64k(fen):
    """One 64k-node label: same field contract as the E2 teacher."""
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
    send(f"go nodes {NODES}")
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
    return {
        "cp": final.get("cp"),
        "mate": final.get("mate"),
        "bestmove": final.get("bestmove", best),
        "wdl": final.get("wdl", [None, None, None]),
    }


def main():
    prev = json.load(open(
        r"C:\Users\81489\AppData\Local\Temp\opencode\teacher-noise-check.json"
    ))["rows"]
    labels = {}
    with open(r"data\s10\s10-eval-v2-1m01\labels.jsonl",
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                labels[rec["position_id"]] = rec
    assert len(prev) == 500
    # freeze corpus identity
    pid_sha = hashlib.sha256(
        "\n".join(sorted(r["pid"] for r in prev)).encode()).hexdigest()
    print(f"corpus: {len(prev)} positions | ordered pid SHA {pid_sha[:16]}")

    # ---- Y64 labeling ----
    t0 = time.time()
    out_rows = []
    for i, r in enumerate(prev):
        lab = label_64k(r["fen"])
        out_rows.append({"pid": r["pid"], "fen": r["fen"],
                         "y64": lab, "t20": r["t20"]})
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  {i + 1}/500 ({(i + 1)/dt:.1f} pos/s)", flush=True)
    wall = time.time() - t0
    print(f"Y64 wall time: {wall:.0f}s ({500/wall:.1f} pos/s)")

    def mae_stats(pairs):
        pairs = [(a, b) for a, b in pairs
                 if a is not None and b is not None]
        if not pairs:
            return None
        d = [abs(a - b) for a, b in pairs]
        s = sorted(d)
        return {"n": len(pairs), "mae": round(sum(d) / len(d), 1),
                "median": s[len(s) // 2],
                "p95": s[min(len(s) - 1, int(0.95 * len(s)))]}

    def cmp_fields(ra, rb, f):
        return (ra.get(f) == rb.get(f))

    # ---- assemble comparisons ----
    y16_yd20, y64_yd20, y16_y64 = [], [], []
    bm_agree = wdl_agree = sign_agree = sign_n = 0
    phase_rows = {"high": [], "mid": [], "low": [], "zero": []}
    for r in out_rows:
        lab = labels[r["pid"]]
        y16 = lab["teacher_cp_stm"]
        y64 = r["y64"]["cp"]
        yd20 = r["t20"] if not isinstance(r["t20"], list) else None
        y16c = clamp(y16) if y16 is not None else None
        y64c = clamp(y64) if y64 is not None else None
        yd20c = clamp(yd20) if yd20 is not None else None
        if y16c is not None and yd20c is not None:
            y16_yd20.append((y16c, yd20c))
        if y64c is not None and yd20c is not None:
            y64_yd20.append((y64c, yd20c))
        if y16c is not None and y64c is not None:
            y16_y64.append((y16c, y64c))
            if (y16c > 0) == (y64c > 0) and (abs(y16c) >= 50
                                              or abs(y64c) >= 50):
                sign_agree += 1
            if abs(y16c) >= 50 and abs(y64c) >= 50:
                sign_n += 1
        if lab["teacher_bestmove"] == r["y64"]["bestmove"]:
            bm_agree += 1
        if list(lab["teacher_wdl_stm"]) == list(r["y64"]["wdl"]):
            wdl_agree += 1
        ph = phase_bucket(phase_of(r["fen"]))
        phase_rows[ph].append((y16c, y64c, yd20c))

    print()
    print("Y16<->YD20:", mae_stats(y16_yd20))
    print("Y64<->YD20:", mae_stats(y64_yd20))
    print("Y16<->Y64 :", mae_stats(y16_y64))
    print()
    print(f"bestmove agreement (Y16 vs Y64): {bm_agree}/500")
    print(f"WDL agreement (Y16 vs Y64):      {wdl_agree}/500")
    print(f"sign agreement |x|>=50:          {sign_agree}/{sign_n or 1}")

    print()
    print("phase breakdown (MAE):")
    print(f"{'phase':<6} {'n':>4} {'16k-D20':>8} {'64k-D20':>8} {'16k-64k':>8}")
    for ph in ("high", "mid", "low", "zero"):
        g = phase_rows[ph]
        a = mae_stats([(x, z) for x, y, z in g if x is not None and z is not None])
        b = mae_stats([(y, z) for x, y, z in g if y is not None and z is not None])
        c = mae_stats([(x, y) for x, y, z in g if x is not None and y is not None])
        print(f"{ph:<6} {len(g):>4} "
              f"{a['mae'] if a else '-':>8} "
              f"{b['mae'] if b else '-':>8} "
              f"{c['mae'] if c else '-':>8}")

    report = {
        "schema_version": 1,
        "corpus": {"n": 500, "ordered_pid_sha256": pid_sha,
                   "source": "H0-A1 frozen matched corpus (no resampling)"},
        "y64_contract": {"nodes": NODES, "threads": 1, "hash": 64,
                         "multipv": 1, "uci_show_wdl": True,
                         "binary": "SF18 windows-x86-64-avx2 c86215fa..."},
        "overall": {
            "y16_yd20": mae_stats(y16_yd20),
            "y64_yd20": mae_stats(y64_yd20),
            "y16_y64": mae_stats(y16_y64),
        },
        "agreement": {
            "bestmove": f"{bm_agree}/500",
            "wdl": f"{wdl_agree}/500",
            "sign_ge50": f"{sign_agree}/{sign_n}",
        },
        "phase": {
            ph: {"n": len(phase_rows[ph]),
                 "y16_yd20": mae_stats([(x, z) for x, y, z in phase_rows[ph]
                                        if x is not None and z is not None]),
                 "y64_yd20": mae_stats([(y, z) for x, y, z in phase_rows[ph]
                                        if y is not None and z is not None]),
                 "y16_y64": mae_stats([(x, y) for x, y, z in phase_rows[ph]
                                       if x is not None and y is not None])}
            for ph in ("high", "mid", "low", "zero")
        },
        "throughput": {"wall_s": round(wall, 1),
                       "pos_per_s": round(500 / wall, 2)},
        "y64_labels": out_rows,
    }
    Path(r"results\s10\s10-h0-a2-pilot.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nwrote results/s10/s10-h0-a2-pilot.json")


if __name__ == "__main__":
    main()
