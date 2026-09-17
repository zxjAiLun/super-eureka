#!/usr/bin/env python3
"""S15: list the Fishtest test_ids already consumed by the S14 round.

Reads the assembled shards (which carry `test_id`) and writes the distinct
set to a JSON file so the expanded convert can skip those tests instead of
re-parsing them (~2h of work that produced the existing 4.22M positions).
"""

from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path

OUT = Path(r"data\s15\used-test-ids.json")


def main() -> int:
    ids: set[str] = set()
    games = 0
    shards = sorted(glob.glob(r"data\s14\assembled\assembled-*.jsonl.gz"))
    for sp in shards:
        with gzip.open(sp, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["test_id"])
                games += 1
        print(f"  {sp}: cumulative positions={games} tests={len(ids)}",
              flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sorted(ids), indent=1), encoding="utf-8")
    print(f"positions={games} distinct test_ids={len(ids)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
