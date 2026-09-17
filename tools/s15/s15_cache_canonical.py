#!/usr/bin/env python3
"""S15: cache the canonical CP corpus track (train + validation) as binary.

In the S14 run the canonical track was re-derived on EVERY training run by
shelling out to the engine's batch exporters (`nnue-features-batch`,
`material-batch`). That work is deterministic given (corpus, engine,
feature set), so it belongs in the one-time cache too.

Writes:
  <out>/train.ekc   canonical TRAIN split  (the pool's first track)
  <out>/val.ekc     canonical Y16 validation split (model selection only)
  <out>/meta.json   dataset sha, counts, engine sha, feature set

The engine binary's sha256 is recorded so a cache built with a different
binary is detectable (the S15 encoding gate already proves the current
binary reproduces the S14 v2r12 encoding exactly; this records which
binary was used).

Usage:
  python tools/s15/s15_cache_canonical.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"tools\s12")
sys.path.insert(0, r"tools\s15")
import s12_train  # noqa: E402
import nnue_cache  # noqa: E402
import chess  # noqa: E402

FEATURE_SET = "v2r12"
MAX_INDEX = 23296
CLIP_CP = s12_train.CLIP_CP


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path(r"data\s10\s10-eval-v2-1m01"))
    ap.add_argument("--engine", type=Path,
                    default=Path(r"target\release\eureka.exe"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"data\s15\cache-canonical"))
    args = ap.parse_args()

    if not args.engine.exists():
        raise SystemExit(f"FAIL CLOSED: engine not found: {args.engine}")
    engine_sha = hashlib.sha256(args.engine.read_bytes()).hexdigest()
    idx_dtype = nnue_cache.choose_index_dtype(MAX_INDEX)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds = s12_train.load_dataset(args.dataset)
    records, labels = ds["records"], ds["labels"]
    usable = [(r, labels[r["position_id"]]) for r in records
              if labels[r["position_id"]].get("teacher_cp_stm") is not None]
    export_records = [r for r, _ in usable]
    print(f"[canon] usable={len(usable)}; exporting features via engine...",
          flush=True)
    exported = s12_train.export_features_from_engine(
        args.engine, export_records, "v2r12")
    material_stm = s12_train.export_material_from_engine(
        args.engine, export_records)
    print(f"[canon] engine export done ({time.time()-t0:.0f}s)", flush=True)

    _bucket_cache: dict[str, int] = {}

    def bucket_of(fen: str) -> int:
        b = _bucket_cache.get(fen)
        if b is None:
            b = s12_train.output_bucket(len(chess.Board(fen).piece_map()))
            _bucket_cache[fen] = b
        return b

    splits = {"train": [], "validation": []}
    for r, lbl in usable:
        split = r["split"]
        if split not in splits:
            continue
        pid = r["position_id"]
        exp = exported[pid]
        stm_is_white = r["fen"].split()[1] == "w"
        stm = exp["white"] if stm_is_white else exp["black"]
        nstm = exp["black"] if stm_is_white else exp["white"]
        target_cp = max(-CLIP_CP, min(CLIP_CP, float(lbl["teacher_cp_stm"])))
        splits[split].append((stm, nstm, float(material_stm[pid]),
                              bucket_of(r["fen"]), target_cp))

    manifest = {"schema_version": 1, "stage": "s15_canonical_cache",
                "feature_set": FEATURE_SET, "max_index": MAX_INDEX,
                "idx_dtype": idx_dtype, "engine_sha256": engine_sha,
                "dataset_sha256": ds["dataset_sha"], "splits": {}}

    for split, rows in splits.items():
        if not rows:
            raise SystemExit(f"FAIL CLOSED: empty split {split}")
        stm_flat = np.concatenate(
            [np.asarray(x[0], dtype=np.int16) for x in rows])
        nstm_flat = np.concatenate(
            [np.asarray(x[1], dtype=np.int16) for x in rows])
        stm_off = np.zeros(len(rows) + 1, dtype=np.int64)
        nstm_off = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum([len(x[0]) for x in rows], out=stm_off[1:])
        np.cumsum([len(x[1]) for x in rows], out=nstm_off[1:])
        path = args.out / f"{split}.ekc"
        info = nnue_cache.write_shard(
            path,
            stm_flat=stm_flat, nstm_flat=nstm_flat,
            stm_off=stm_off, nstm_off=nstm_off,
            material=np.asarray([x[2] for x in rows], dtype=np.float32),
            bucket=np.asarray([x[3] for x in rows], dtype=np.int8),
            cp=np.asarray([x[4] for x in rows], dtype=np.float32),
            raw=np.asarray([x[4] for x in rows], dtype=np.float32),
            feature_set=FEATURE_SET, max_index=MAX_INDEX,
            idx_dtype=idx_dtype, meta={"split": split})
        manifest["splits"][split] = {**info, "file": path.name}
        print(f"  {path.name}: n={info['n']} stm={info['n_stm']} "
              f"{info['bytes']/2**20:.1f} MiB", flush=True)

    manifest["elapsed_s"] = round(time.time() - t0, 1)
    (args.out / "meta.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest["splits"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
