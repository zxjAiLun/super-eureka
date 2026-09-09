#!/usr/bin/env python3
"""S14-A final assembly: dedup + trim + deterministic cut to 4,220,410.

  1. read all shard records in deterministic order (worker id, shard idx)
  2. drop exact-FEN overlaps with the frozen 1M validation/holdout
     (train overlaps are ALSO dropped — the canonical 779,590 train set
     stays the sole source of those positions; no double-weighting)
  3. dedup within Fishtest (exact FEN) — engine games repeat openings,
     keep the first occurrence in scan order
  4. trim to EXACTLY 4,220,410 (deterministic: scan order)
  5. report stats + write assembly manifest
"""

from __future__ import annotations

import glob
import gzip
import json
import sys
from pathlib import Path

CORPUS = Path(r"data\s10\s10-eval-v2-1m01")
SHARDS = sorted(glob.glob(r"data\s14\shards\*.jsonl.gz"))
TARGET = 4_220_410
OUT = Path(r"data\s14\assembled")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    # frozen corpus FENs (all splits — train overlap dropped too, see
    # docstring; the canonical train set provides those positions)
    corpus_fens = set()
    for p in sorted(CORPUS.glob("part-*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                corpus_fens.add(json.loads(line)["fen"])
    print("corpus fens:", len(corpus_fens), flush=True)

    def shard_order(path: str):
        name = Path(path).name
        wid = int(name.split("-")[1][1:])
        idx = int(name.split("-")[2].split(".")[0])
        return (wid, idx)

    seen = set()
    kept = []
    dropped_corpus = dropped_dup = 0
    for f in sorted(SHARDS, key=shard_order):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                fen = r["fen_before"]
                if fen in corpus_fens:
                    dropped_corpus += 1
                    continue
                if fen in seen:
                    dropped_dup += 1
                    continue
                seen.add(fen)
                kept.append(r)
                if len(kept) >= TARGET:
                    break
        if len(kept) >= TARGET:
            break

    trimmed = kept[:TARGET]
    print(f"kept {len(trimmed)} (dropped: corpus-overlap {dropped_corpus}, "
          f"internal-dup {dropped_dup})", flush=True)

    # write assembled shards (~1M lines each, gz)
    per = 1_000_000
    for i in range(0, len(trimmed), per):
        chunk = trimmed[i:i + per]
        path = OUT / f"assembled-{i//per:02d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for r in chunk:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        print("wrote", path, len(chunk), flush=True)

    # stats
    import statistics
    cps = [r["teacher_cp_raw"] for r in trimmed]
    manifest = {
        "schema_version": 1,
        "stage": "s14_assembly",
        "target": TARGET,
        "kept": len(trimmed),
        "dropped_corpus_overlap": dropped_corpus,
        "dropped_internal_dup": dropped_dup,
        "input_shards": len(SHARDS),
        "scan_order": "worker-id, shard-idx (deterministic)",
        "teacher_cp_raw_stats": {
            "mean": round(statistics.mean(cps), 2),
            "min": min(cps), "max": max(cps),
            "abs_gt_2000": sum(1 for c in cps if abs(c) > 2000),
        },
    }
    (OUT / "assembly-manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
