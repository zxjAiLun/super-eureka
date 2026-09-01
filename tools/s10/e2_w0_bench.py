"""S10-E2-W0: Windows-native worker throughput benchmark.

Labels the first 1000 corpus positions with N independent persistent
stockfish.exe processes (Threads=1 Hash=64 MultiPV=1) for
N in {1, 2, 4, 6, 8}. Reports aggregate pos/s, wall time, and peak
process RSS.

Usage:
    python tools/s10/e2_w0_bench.py --corpus results/s10/s10-e2-w0-corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.e2_w0_run import NativeTeacher, WIN_EXE, WIN_SHA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--workers", type=int, nargs="+",
                        default=[1, 2, 4, 6, 8])
    args = parser.parse_args()

    fens = [json.loads(l)["fen"] for l in
            args.corpus.read_text(encoding="utf-8").splitlines() if l.strip()]
    fens = fens[: args.positions]

    results = {}
    for n in args.workers:
        local = threading.local()

        def get_teacher():
            if getattr(local, "teacher", None) is None:
                local.teacher = NativeTeacher(WIN_EXE, WIN_SHA)
            return local.teacher

        t0 = time.time()
        peak_rss = [0]

        def work(i_fen):
            i, fen = i_fen
            get_teacher().label(fen)

        def sampler():
            import subprocess as sp
            while not stop[0]:
                out = sp.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Process stockfish* -ErrorAction SilentlyContinue "
                     "| Measure-Object WorkingSet64 -Sum).Sum"],
                    capture_output=True, text=True).stdout.strip()
                try:
                    peak_rss[0] = max(peak_rss[0], int(out))
                except ValueError:
                    pass
                time.sleep(2)

        stop = [False]
        samp = threading.Thread(target=sampler, daemon=True)
        samp.start()
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(work, list(enumerate(fens))))
        stop[0] = True
        dt = time.time() - t0

        t = getattr(local, "teacher", None)
        if t is not None:
            t.close()
        results[n] = {
            "seconds": round(dt, 1),
            "pos_per_s": round(len(fens) / dt, 1),
            "peak_rss_mb": round(peak_rss[0] / 2**20, 0),
        }
        print(f"workers={n}: {len(fens)} positions in {dt:.1f}s "
              f"({len(fens)/dt:.1f} pos/s, peak RSS "
              f"{peak_rss[0]/2**20:.0f} MB)", flush=True)

    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
