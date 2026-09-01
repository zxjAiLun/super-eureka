"""S10-E2 Repair: single-worker vs N-worker exact equivalence gate.

Takes the first K positions of the dataset, labels them with N
independent single-thread SF18 workers (coordinator pattern), and
compares EVERY field against the single-worker partial stream
(labels.partial.jsonl produced by the serial run — the golden
reference).

Gate: K/K exact on teacher_cp_stm / teacher_mate / teacher_bestmove /
teacher_wdl_stm / nodes.

Usage:
    python tools/s10/e2_equiv_workers.py \
        --dataset data/s10/s10-eval-v2-1m01 --k 5000 --workers 4 \
        --teacher-binary /home/sparkle/sf18
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s6.label_teacher import (  # noqa: E402
    DEFAULT_BINARY_SHA256, Teacher, load_records,
)

FIELDS = ("teacher_cp_stm", "teacher_mate", "teacher_bestmove",
          "teacher_wdl_stm", "nodes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-binary", default="/home/sparkle/sf18")
    parser.add_argument("--partial", default="labels.partial.jsonl")
    args = parser.parse_args()

    records = load_records(args.dataset)[: args.k]

    golden = {}
    with open(args.dataset / args.partial, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                golden[rec["position_id"]] = rec
    have = sum(1 for r in records if r["position_id"] in golden)
    if have < args.k:
        print(f"FATAL: partial stream covers only {have}/{args.k} of the "
              f"first records; equivalence needs the full prefix",
              flush=True)
        return 1

    # One Teacher PER WORKER THREAD. A Teacher object owns a single
    # stdin/stdout pipe pair and is NOT thread-safe; sharing one across
    # threads interleaves reads and swaps labels between positions (the
    # first attempt did exactly that and produced value-swapped mismatches).
    import threading
    local = threading.local()

    def get_teacher():
        if getattr(local, "teacher", None) is None:
            local.teacher = Teacher(
                wsl=True, binary=args.teacher_binary,
                expected_binary_sha256=DEFAULT_BINARY_SHA256)
        return local.teacher

    try:
        results: dict[str, dict] = {}
        lock = threading.Lock()

        def work(i_rec):
            i, rec = i_rec
            lbl = get_teacher().label(rec["fen"])
            with lock:
                results[rec["position_id"]] = lbl

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, list(enumerate(records))))
    finally:
        for t in [getattr(local, "teacher", None)]:
            try:
                if t is not None:
                    t.close()
            except Exception:
                pass

    mismatches = 0
    examples = []
    for rec in records:
        pid = rec["position_id"]
        got = results[pid]
        want = golden[pid]
        for f in FIELDS:
            if got.get(f) != want.get(f):
                mismatches += 1
                if len(examples) < 3:
                    examples.append((pid, f, got.get(f), want.get(f)))
                break

    print(f"equivalence: {args.k - mismatches}/{args.k} exact "
          f"({args.workers} workers vs single-worker golden)", flush=True)
    for ex in examples:
        print(f"  MISMATCH {ex[0]} field {ex[1]}: got {ex[2]} want {ex[3]}",
              flush=True)
    print("EQUIVALENCE", "PASS" if mismatches == 0 else "FAIL")
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
