#!/usr/bin/env python3
"""S15: how much convert progress is safely on disk (restart-safe)?

`s15_convert.py` resumes by test id: on restart it reads the existing shards
and SKIPS any test already present. That means a restart never duplicates
work -- but it also means a test that was only partially written is skipped
in full, so its unflushed remainder is lost.

This reports both sides so the restart decision is quantitative:
  * records on disk (what survives a restart)
  * distinct test ids on disk (what the exclusion set will skip)
  * the log's global accepted counter (what the run has actually parsed)

Usage:
  python tools/s15/s15_resume_check.py
"""

from __future__ import annotations

import glob
import gzip
import json
import re
import sys
from pathlib import Path


def main() -> int:
    prune = "--prune" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    shard_glob = positional[0] if positional else \
        r"data\s15\shards\*.jsonl.gz"
    log_path = Path(positional[1]) if len(positional) > 1 else \
        Path(r"data\s15\convert.stdout.log")

    files = sorted(glob.glob(shard_glob))
    ids: set[str] = set()
    on_disk = 0
    bad: list[str] = []
    per_test: dict[str, int] = {}
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    tid = json.loads(line)["test_id"]
                    ids.add(tid)
                    per_test[tid] = per_test.get(tid, 0) + 1
                    on_disk += 1
        except (EOFError, OSError, gzip.BadGzipFile):
            # A shard being written right now, or one left truncated by a
            # kill. Report it; it must not be counted as good data.
            bad.append(Path(f).name)

    print(f"shards on disk      : {len(files)}")
    print(f"records on disk     : {on_disk:,}")
    print(f"distinct test ids   : {len(ids)}")
    if bad:
        print(f"UNREADABLE shards   : {len(bad)}  {bad[:5]}")
        if prune:
            for name in bad:
                (Path(shard_glob).parent / name).unlink(missing_ok=True)
            print(f"  -> pruned {len(bad)} unreadable shards "
                  f"(their tests will be converted again)")

    parsed = None
    if log_path.exists():
        last = None
        for line in log_path.read_text(encoding="utf-8",
                                        errors="replace").splitlines():
            m = re.search(r"global=(\d+)/\d+ elapsed=(\d+)s", line)
            if m:
                last = (int(m.group(1)), int(m.group(2)))
        if last:
            parsed = last
            print(f"log global accepted : {last[0]:,}  @ {last[1]}s")

    if parsed is not None:
        inflight = parsed[0] - on_disk
        print(f"in-flight (lost)    : {inflight:,} "
              f"({inflight / max(parsed[0], 1) * 100:.2f}% of parsed)")
        print()
        print("On restart: the %d tests below are SKIPPED (no duplication),"
              % len(ids))
        print("but any partially-written test keeps only its flushed part.")
        small = sorted(per_test.items(), key=lambda kv: kv[1])[:5]
        big = sorted(per_test.items(), key=lambda kv: -kv[1])[:3]
        print(f"  smallest tests on disk: "
              f"{[(k[:8], v) for k, v in small]}")
        print(f"  largest  tests on disk: "
              f"{[(k[:8], v) for k, v in big]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
