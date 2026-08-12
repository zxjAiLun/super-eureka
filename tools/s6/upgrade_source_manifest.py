#!/usr/bin/env python3
"""Upgrade a legacy source_manifest.json with explicit source_family and
provenance fields (REQUIRED by the FINAL builder contract).

Usage:
  python tools/s6/upgrade_source_manifest.py \
      --manifest data/s6/sources/source_manifest.json \
      --family arena --provenance arena-historical-tournament-pgn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args(sys.argv[1:])

    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, entry in data.items():
        if not entry.get("source_id"):
            entry["source_id"] = f"arena-{key}"
        entry["source_family"] = args.family
        entry["provenance"] = args.provenance
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"upgraded {path} ({len(data)} sources, family={args.family})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
