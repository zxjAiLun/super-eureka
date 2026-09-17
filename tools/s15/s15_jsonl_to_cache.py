#!/usr/bin/env python3
"""S15: convert the S14 encoded JSONL.gz shards into the binary cache.

One-time cost. After this, the 4.22M frozen S14 positions load at I/O
speed on every future candidate instead of being re-parsed from gzip+JSON.

Verification is built in: every written shard is read back and compared
field-by-field against the source lines it came from. Any mismatch aborts.

Usage:
  python tools/s15/s15_jsonl_to_cache.py \
      --glob "data/s14/encoded/encoded-*.jsonl.gz" \
      --out  data/s15/cache
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"tools\s15")
import nnue_cache  # noqa: E402

FEATURE_SET = "v2r12"
MAX_INDEX = 23296  # NNUE_INPUTS_V2R12


def parse_shard(path: Path):
    stm_strs, nstm_strs = [], []
    stm_len, nstm_len = [], []
    material, bucket, cp, raw = [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            s, n = r["stm"], r["nstm"]
            stm_strs.append(s)
            nstm_strs.append(n)
            stm_len.append(s.count(",") + 1)
            nstm_len.append(n.count(",") + 1)
            material.append(float(r["m"]))
            bucket.append(int(r["b"]))
            cp.append(float(r["cp"]))
            raw.append(float(r["raw"]))
    stm_flat = np.fromstring(",".join(stm_strs), dtype=np.int32, sep=",")
    nstm_flat = np.fromstring(",".join(nstm_strs), dtype=np.int32, sep=",")
    stm_off = np.zeros(len(stm_len) + 1, dtype=np.int64)
    nstm_off = np.zeros(len(nstm_len) + 1, dtype=np.int64)
    np.cumsum(np.asarray(stm_len, dtype=np.int64), out=stm_off[1:])
    np.cumsum(np.asarray(nstm_len, dtype=np.int64), out=nstm_off[1:])
    return dict(stm_flat=stm_flat, nstm_flat=nstm_flat,
                stm_off=stm_off, nstm_off=nstm_off,
                material=np.asarray(material, dtype=np.float32),
                bucket=np.asarray(bucket, dtype=np.int8),
                cp=np.asarray(cp, dtype=np.float32),
                raw=np.asarray(raw, dtype=np.float32),
                stm_len=np.asarray(stm_len, dtype=np.int64),
                nstm_len=np.asarray(nstm_len, dtype=np.int64))


def verify_readback(src: dict, shard_path: Path) -> None:
    back = nnue_cache.read_shard(shard_path)
    if not np.array_equal(back["stm_flat"], src["stm_flat"].astype(
            back["stm_flat"].dtype)):
        raise SystemExit(f"FAIL CLOSED: stm_flat mismatch after read-back "
                         f"({shard_path})")
    if not np.array_equal(back["nstm_flat"], src["nstm_flat"].astype(
            back["nstm_flat"].dtype)):
        raise SystemExit(f"FAIL CLOSED: nstm_flat mismatch ({shard_path})")
    if int(back["stm_off"][-1]) != src["stm_flat"].size:
        raise SystemExit(f"FAIL CLOSED: stm_off total mismatch ({shard_path})")
    if not np.array_equal(back["material"], src["material"]):
        raise SystemExit(f"FAIL CLOSED: material mismatch ({shard_path})")
    if not np.array_equal(back["bucket"], src["bucket"]):
        raise SystemExit(f"FAIL CLOSED: bucket mismatch ({shard_path})")
    if not np.array_equal(back["cp"], src["cp"]):
        raise SystemExit(f"FAIL CLOSED: cp mismatch ({shard_path})")
    if not np.array_equal(back["raw"], src["raw"]):
        raise SystemExit(f"FAIL CLOSED: raw mismatch ({shard_path})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default=r"data\s14\encoded\encoded-*.jsonl.gz")
    ap.add_argument("--out", type=Path, default=Path(r"data\s15\cache"))
    args = ap.parse_args()

    srcs = sorted(glob.glob(args.glob))
    if not srcs:
        raise SystemExit(f"FAIL CLOSED: no sources match {args.glob}")
    args.out.mkdir(parents=True, exist_ok=True)

    idx_dtype = nnue_cache.choose_index_dtype(MAX_INDEX)
    manifest = {"schema_version": 1,
                "stage": "s15_jsonl_to_cache",
                "feature_set": FEATURE_SET,
                "max_index": MAX_INDEX,
                "idx_dtype": idx_dtype,
                "sources": [], "shards": [], "positions": 0}
    t0 = time.time()
    for si, sp in enumerate(srcs):
        d = parse_shard(Path(sp))
        if d["stm_flat"].size and int(d["stm_flat"].max()) >= MAX_INDEX:
            raise SystemExit(f"FAIL CLOSED: stm index >= {MAX_INDEX} in {sp}")
        if d["nstm_flat"].size and int(d["nstm_flat"].max()) >= MAX_INDEX:
            raise SystemExit(f"FAIL CLOSED: nstm index >= {MAX_INDEX} in {sp}")
        out_path = args.out / f"cache-{si:03d}.ekc"
        info = nnue_cache.write_shard(
            out_path,
            stm_flat=d["stm_flat"].astype(np.int16),
            nstm_flat=d["nstm_flat"].astype(np.int16),
            stm_off=d["stm_off"], nstm_off=d["nstm_off"],
            material=d["material"], bucket=d["bucket"],
            cp=d["cp"], raw=d["raw"],
            feature_set=FEATURE_SET, max_index=MAX_INDEX,
            idx_dtype=idx_dtype,
            meta={"source": Path(sp).name})
        verify_readback(d, out_path)
        manifest["sources"].append(Path(sp).name)
        manifest["shards"].append({**info, "file": out_path.name})
        manifest["positions"] += info["n"]
        print(f"  {out_path.name}: n={info['n']} "
              f"stm={info['n_stm']} nstm={info['n_nstm']} "
              f"{info['bytes']/2**20:.1f} MiB  (read-back OK)", flush=True)

    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["total_bytes"] = sum(
        s["bytes"] for s in manifest["shards"])
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("positions", "idx_dtype", "total_bytes",
                       "elapsed_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
