#!/usr/bin/env python3
"""S15: compact binary cache for encoded NNUE training positions.

Motivation (measured, S14 run): data loading cost 955 s versus 291 s of
actual training -- the pool was read as gzip'd JSONL, and every position
paid `gzip` + `json.loads` + `",".join` + `np.fromstring` again on every
training run. The same preprocessing cost was paid for a candidate that
trained for five minutes.

This module defines a flat binary shard format that is written ONCE and
then read back with plain sequential I/O, so a candidate pays no parsing
cost. It is deliberately lossless with respect to the training tensors:
the reader reproduces exactly the tensors `s14_train.FlatPool.finalize()`
builds (int sparse indices + CSR offsets + material/bucket/cp scalars),
so the training loop does not change.

Layout of one shard file (little-endian):

    offset 0   : magic  b"EKC1"
    offset 4   : version u32
    offset 8   : header_len u32
    offset 12  : flags u32 (reserved, 0)
    offset 16  : header JSON (header_len bytes, UTF-8)
    then, padded to 64 bytes, the arrays in the order listed by
    header["arrays"], each self-describing (name/dtype/count/offset).

Arrays per shard:
    stm_flat    sparse feature indices, mover perspective
    nstm_flat   sparse feature indices, opponent perspective
    stm_off     CSR offsets for stm_flat, int32, length n+1, local to shard
    nstm_off    CSR offsets for nstm_flat
    material    material_cp_stm, float32
    bucket      MaterialCount output bucket, int8
    cp          clamped teacher CP label, float32
    raw         unclamped teacher CP (audit only), float32

Index dtype is declared per cache and chosen by the writer: int16 when the
feature-set's index ceiling fits (v2r12 has 23,296 < 32,768), otherwise
int32. That halves the resident pool versus a naive int32 pool, which
matters on this 16 GB box.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"EKC1"
VERSION = 1
# Fixed header region. Array offsets are absolute, so the reader seeks by
# explicit offset and the padding is irrelevant to it; fixing the region
# keeps array offsets independent of the header's own length.
HDR_RESERVE = 4096

ARRAY_ORDER = ("stm_flat", "nstm_flat", "stm_off", "nstm_off",
               "material", "bucket", "cp", "raw")

DTYPE_MAP = {
    "int16": np.int16, "int32": np.int32, "int8": np.int8,
    "float32": np.float32, "uint16": np.uint16,
}

FEATURE_SET_UNKNOWN = "unknown"


def choose_index_dtype(max_index: int) -> str:
    """Smallest signed dtype that can hold indices in [0, max_index)."""
    if max_index <= 0:
        raise ValueError("max_index must be positive")
    if max_index < 2 ** 15:
        return "int16"
    if max_index < 2 ** 31:
        return "int32"
    raise ValueError(f"feature index ceiling too large: {max_index}")


def write_shard(path: Path, *, stm_flat, nstm_flat, stm_off, nstm_off,
                material, bucket, cp, raw, feature_set: str,
                max_index: int, idx_dtype: str, meta: dict | None = None):
    """Write one cache shard. Arrays must already be numpy, same length
    convention: stm_off/nstm_off have n+1 entries, scalars have n."""
    path = Path(path)
    arrays = {
        "stm_flat": np.ascontiguousarray(stm_flat),
        "nstm_flat": np.ascontiguousarray(nstm_flat),
        "stm_off": np.ascontiguousarray(stm_off, dtype=np.int32),
        "nstm_off": np.ascontiguousarray(nstm_off, dtype=np.int32),
        "material": np.ascontiguousarray(material, dtype=np.float32),
        "bucket": np.ascontiguousarray(bucket, dtype=np.int8),
        "cp": np.ascontiguousarray(cp, dtype=np.float32),
        "raw": np.ascontiguousarray(raw, dtype=np.float32),
    }
    n = int(arrays["material"].size)
    if n == 0:
        raise ValueError("refusing to write an empty shard")
    if arrays["bucket"].size != n or arrays["cp"].size != n \
            or arrays["raw"].size != n:
        raise ValueError("scalar array length mismatch")
    if arrays["stm_off"].size != n + 1 or arrays["nstm_off"].size != n + 1:
        raise ValueError("offset array must have n+1 entries")
    if int(arrays["stm_off"][-1]) != arrays["stm_flat"].size:
        raise ValueError("stm_off[-1] != len(stm_flat)")
    if int(arrays["nstm_off"][-1]) != arrays["nstm_flat"].size:
        raise ValueError("nstm_off[-1] != len(nstm_flat)")

    # fail closed on out-of-range indices before writing anything
    for key in ("stm_flat", "nstm_flat"):
        a = arrays[key]
        if a.size:
            lo, hi = int(a.min()), int(a.max())
            if lo < 0 or hi >= max_index:
                raise ValueError(
                    f"FAIL CLOSED: {key} index out of range "
                    f"[{lo},{hi}] not in [0,{max_index})")

    header = {
        "version": VERSION,
        "n": n,
        "n_stm": int(arrays["stm_flat"].size),
        "n_nstm": int(arrays["nstm_flat"].size),
        "idx_dtype": idx_dtype,
        "feature_set": feature_set,
        "max_index": int(max_index),
        "meta": meta or {},
    }

    base = HDR_RESERVE
    directory = []
    offset = base
    for name in ARRAY_ORDER:
        a = arrays[name]
        directory.append({"name": name, "dtype": a.dtype.str,
                          "count": int(a.size), "offset": offset})
        offset += a.nbytes
    header["arrays"] = directory
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if 16 + len(blob) > HDR_RESERVE:
        raise ValueError(f"header too large: {len(blob)} bytes")

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<III", VERSION, len(blob), 0))
        fh.write(blob)
        fh.write(b"\x00" * (HDR_RESERVE - 16 - len(blob)))
        for entry in directory:
            fh.write(arrays[entry["name"]].tobytes(order="C"))
    tmp.replace(path)
    return {"n": n, "n_stm": int(arrays["stm_flat"].size),
            "n_nstm": int(arrays["nstm_flat"].size),
            "bytes": path.stat().st_size}


def read_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != MAGIC:
            raise ValueError(f"not an EKC cache shard: {path}")
        version, hdr_len, _flags = struct.unpack("<III", fh.read(12))
        if version != VERSION:
            raise ValueError(f"unsupported cache version {version}: {path}")
        blob = fh.read(hdr_len)
    return json.loads(blob.decode("utf-8"))


def read_shard(path: Path) -> dict:
    """Return {header..., arrays...} as numpy arrays (copies, plain read)."""
    header = read_header(path)
    out = {}
    with open(path, "rb") as fh:
        for entry in header["arrays"]:
            fh.seek(entry["offset"])
            a = np.frombuffer(
                fh.read(entry["count"] * np.dtype(entry["dtype"]).itemsize),
                dtype=np.dtype(entry["dtype"]))
            out[entry["name"]] = a
    return {"header": header, **out}


class CachedPool:
    """Mirror of s14_train.FlatPool's finalized tensor surface."""

    def __init__(self, *, stm_flat, nstm_flat, stm_off, nstm_off,
                 material, bucket, cp, n, feature_set, idx_dtype,
                 shards, sources):
        self.stm_flat = stm_flat
        self.nstm_flat = nstm_flat
        self.stm_off = stm_off
        self.nstm_off = nstm_off
        self.material = material
        self.bucket = bucket
        self.cp = cp
        self.n = n
        self.feature_set = feature_set
        self.idx_dtype = idx_dtype
        self.shards = shards
        self.sources = sources
        self.total_stm = int(stm_flat.numel())
        self.total_nstm = int(nstm_flat.numel())


def load_pool(paths, max_index: int | None = None, device="cpu"):
    """Load cache shards into a CachedPool (CSR, ready for the trainer).

    Validates every shard against the first: same feature_set, same index
    dtype, indices inside the declared ceiling. Offsets are rebuilt globally
    so a shard boundary is invisible to the training loop.
    """
    import torch

    paths = [Path(p) for p in paths]
    if not paths:
        raise SystemExit("FAIL CLOSED: no cache shards to load")

    stm_flats, nstm_flats = [], []
    materials, buckets, cps = [], [], []
    stm_lens, nstm_lens = [], []
    n_total = 0
    ref = None
    for p in paths:
        d = read_shard(p)
        h = d["header"]
        if ref is None:
            ref = (h["feature_set"], h["idx_dtype"], h["max_index"])
        else:
            got = (h["feature_set"], h["idx_dtype"], h["max_index"])
            if got != ref:
                raise SystemExit(
                    f"FAIL CLOSED: shard {p.name} declares {got}, "
                    f"expected {ref}")
        ceiling = max_index if max_index is not None else h["max_index"]
        stm_flat = d["stm_flat"]
        nstm_flat = d["nstm_flat"]
        if stm_flat.size:
            if int(stm_flat.min()) < 0 or int(stm_flat.max()) >= ceiling:
                raise SystemExit(f"FAIL CLOSED: stm index out of range in {p}")
        if nstm_flat.size:
            if int(nstm_flat.min()) < 0 or int(nstm_flat.max()) >= ceiling:
                raise SystemExit(f"FAIL CLOSED: nstm out of range in {p}")
        stm_flats.append(stm_flat)
        nstm_flats.append(nstm_flat)
        materials.append(d["material"])
        buckets.append(d["bucket"])
        cps.append(d["cp"])
        stm_lens.append(np.diff(d["stm_off"].astype(np.int64)))
        nstm_lens.append(np.diff(d["nstm_off"].astype(np.int64)))
        n_total += int(h["n"])

    feature_set, idx_dtype, ceiling = ref
    stm_flat = np.concatenate(stm_flats) if len(stm_flats) > 1 else stm_flats[0]
    nstm_flat = (np.concatenate(nstm_flats) if len(nstm_flats) > 1
                 else nstm_flats[0])
    stm_off = np.zeros(n_total + 1, dtype=np.int64)
    nstm_off = np.zeros(n_total + 1, dtype=np.int64)
    np.cumsum(np.concatenate(stm_lens), out=stm_off[1:])
    np.cumsum(np.concatenate(nstm_lens), out=nstm_off[1:])
    assert int(stm_off[-1]) == stm_flat.size
    assert int(nstm_off[-1]) == nstm_flat.size

    def to_t(a):
        return torch.from_numpy(a).to(device)

    pool = CachedPool(
        stm_flat=to_t(stm_flat),
        nstm_flat=to_t(nstm_flat),
        stm_off=to_t(stm_off),
        nstm_off=to_t(nstm_off),
        material=to_t(np.concatenate(materials)),
        bucket=to_t(np.concatenate(buckets).astype(np.int64)),
        cp=to_t(np.concatenate(cps).astype(np.float32)),
        n=n_total,
        feature_set=feature_set,
        idx_dtype=idx_dtype,
        shards=[p.name for p in paths],
        sources=len(paths),
    )
    return pool
