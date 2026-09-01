"""S10-E2 Repair: worker-count throughput benchmark.

Labels a fixed slice of the 1M dataset with N independent single-thread
SF18 processes (each: Threads=1 Hash=64 MultiPV=1 nodes=16384,
ucinewgame per position — the frozen teacher contract per position).

NOT part of the labeling pipeline: this is a throwaway measurement that
writes nothing into the dataset directory.

Usage:
    python tools/s10/e2_bench_workers.py \
        --dataset data/s10/s10-eval-v2-1m01 --positions 2000 \
        --workers 1 2 4 6 --teacher-binary /home/sparkle/sf18
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s6.label_teacher import (  # noqa: E402
    DEFAULT_BINARY_SHA256, Teacher, load_records,
)


def bench(records, n_workers, teacher_kwargs, n_positions):
    import threading
    local = threading.local()

    def get_teacher():
        if getattr(local, "teacher", None) is None:
            local.teacher = Teacher(**teacher_kwargs)
        return local.teacher

    try:
        t0 = time.time()

        def work(i_rec):
            i, rec = i_rec
            get_teacher().label(rec["fen"])
            return None

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(work, list(enumerate(records[:n_positions]))))
        dt = time.time() - t0
        return dt, n_positions / dt
    finally:
        t = getattr(local, "teacher", None)
        try:
            if t is not None:
                t.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--workers", type=int, nargs="+",
                        default=[1, 2, 4, 6])
    parser.add_argument("--teacher-binary", default="/home/sparkle/sf18")
    args = parser.parse_args()

    records = load_records(args.dataset)
    teacher_kwargs = dict(
        wsl=True,
        binary=args.teacher_binary,
        expected_binary_sha256=DEFAULT_BINARY_SHA256,
    )

    results = {}
    for n in args.workers:
        dt, rate = bench(records, n, teacher_kwargs, args.positions)
        results[n] = {"seconds": round(dt, 1), "pos_per_s": round(rate, 1)}
        print(f"workers={n}: {args.positions} positions in {dt:.1f}s "
              f"({rate:.1f} pos/s)", flush=True)

    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
