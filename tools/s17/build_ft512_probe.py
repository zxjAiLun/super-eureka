"""S17 pre-check: build a synthetic FT512 v5 artifact from S14 weights.

Purpose: a *correctness and cost* probe only, NOT a playing-strength
candidate.  The S14 FT256 weights are embedded into a 512-wide layout and
every other channel is zeroed, so the model's *arithmetic shape* is the
full 512-wide computation while its behaviour is still driven by the S14
channels.  The op count is identical to a real FT512 model, so the speed
measurement is representative.

Layout: the first 256 lanes of each row carry the S14 weights, lanes
[256, 512) are zero.  FT width 512 -> dense_in = 1024, so the L1 weights
are [32][1024] with the first 256 (own) and 512..768 (opp) slots taking the
S14 L1 columns and the rest zero.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

INPUTS = 23296
FT256 = 256
FT512 = 512
NUM_BUCKETS = 8
TARGET_SCALE = 1000.0
VERSION = 5
HEAD_KIND_SCRELU_BUCKETS = 1
FEATURE_SET_V2R12 = 2  # must match the Rust NnueFeatureSetId mapping
FT_SHIFT = 12
DENSE_W_SHIFT = 12
DENSE_Z_SHIFT = 12
QA = 4096


def load_v5(path: Path):
    blob = path.read_bytes()
    magic = blob[:8]
    if magic != b"EUNN2Q01":
        raise SystemExit(f"not a v5 artifact: {magic!r}")
    (ver, inputs, ftw, tscale, fts, dws, dzs, qa, tmode, fset, hkind) = \
        struct.unpack("<IIIfIIIIIII", blob[8:52])
    off = 52 + 64
    ft_w = np.frombuffer(blob, dtype="<i2", count=inputs * ftw,
                         offset=off).reshape(inputs, ftw).astype(np.int64)
    off += inputs * ftw * 2
    ft_b = np.frombuffer(blob, dtype="<i4", count=ftw,
                         offset=off).astype(np.int64)
    off += ftw * 4
    l1_w = np.frombuffer(blob, dtype="<i2", count=NUM_BUCKETS * 2 * ftw,
                         offset=off).reshape(NUM_BUCKETS, 2 * ftw).astype(np.int64)
    off += NUM_BUCKETS * 2 * ftw * 2
    l1_b = np.frombuffer(blob, dtype="<i4", count=NUM_BUCKETS,
                         offset=off).astype(np.int64)
    return dict(ver=ver, inputs=inputs, ftw=ftw, ft_w=ft_w, ft_b=ft_b,
                l1_w=l1_w, l1_b=l1_b, fset=fset, hkind=hkind, tscale=tscale,
                tmode=tmode, sha=hashlib.sha256(blob).hexdigest())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="FT256 v5 artifact")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-sha", default="0" * 64)
    args = ap.parse_args()

    m = load_v5(args.src)
    if m["ftw"] != FT256:
        raise SystemExit(f"source is FT{m['ftw']}, expected FT256")
    print(f"source {args.src}")
    print(f"  sha256 {m['sha']}  inputs={m['inputs']} ft_width={m['ftw']}")
    print(f"  target_mode={m['tmode']}  "
          f"({'material_residual' if m['tmode'] == 1 else 'cp'}) "
          f"-- INHERITED from the source, not hard-coded")
    if m["tmode"] not in (0, 1):
        raise SystemExit(f"unexpected target_mode {m['tmode']}")

    # --- FT table: [23296][512], first 256 lanes = S14, rest zero ---
    ft_w = np.zeros((INPUTS, FT512), dtype=np.int64)
    ft_w[:, :FT256] = m["ft_w"]
    ft_b = np.zeros(FT512, dtype=np.int64)
    ft_b[:FT256] = m["ft_b"]

    # --- L1: [8][1024]. own half occupies [0,512), opp half [512,1024).
    # S14's own 256 columns -> lanes [0,256); S14's opp 256 columns (which
    # live at 256..512 in the FT256 layout) -> lanes [512,768).
    l1_w = np.zeros((NUM_BUCKETS, 2 * FT512), dtype=np.int64)
    l1_w[:, :FT256] = m["l1_w"][:, :FT256]
    l1_w[:, FT512:FT512 + FT256] = m["l1_w"][:, FT256:]
    l1_b = m["l1_b"].copy()

    # int16 range check on the widened tables (must not clip)
    for name, arr in (("ft_w", ft_w), ("l1_w", l1_w)):
        lo, hi = int(arr.min()), int(arr.max())
        if lo < -32768 or hi > 32767:
            raise SystemExit(f"FAIL: {name} out of int16 range [{lo},{hi}]")
        print(f"  {name} widened range [{lo}, {hi}]  int16 clean")

    print(f"  ft table {ft_w.shape}  l1 {l1_w.shape}  "
          f"nonzero lanes ft={int((ft_w != 0).any(axis=0).sum())}/512 "
          f"l1={int((l1_w != 0).any(axis=0).sum())}/1024")

    # --- header ---
    hdr = bytearray()
    hdr += b"EUNN2Q01"
    hdr += struct.pack("<IIIfIIIIIII", VERSION, INPUTS, FT512, TARGET_SCALE,
                       FT_SHIFT, DENSE_W_SHIFT, DENSE_Z_SHIFT, QA,
                       m["tmode"], m["fset"], m["hkind"])
    if len(hdr) != 8 + 44:
        raise SystemExit(f"bad header size {len(hdr)}")
    hdr += bytes.fromhex(args.ckpt_sha)[:32]
    hdr += bytes(32)
    if len(hdr) != 8 + 44 + 64:
        raise SystemExit(f"bad header size {len(hdr)}")

    blob = bytes(hdr)
    blob += ft_w.astype("<i2").tobytes()
    blob += ft_b.astype("<i4").tobytes()
    blob += l1_w.astype("<i2").tobytes()
    blob += l1_b.astype("<i4").tobytes()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print()
    print(f"wrote {args.out}")
    print(f"  size   {len(blob)} bytes")
    print(f"  sha256 {hashlib.sha256(blob).hexdigest()}")
    # Verify the written header round-trips, including target_mode: a
    # wrong mode silently changes the evaluation semantics (a
    # material-residual artifact read as cp reports absolute scores).
    chk = load_v5(args.out)
    assert chk["ftw"] == FT512, chk["ftw"]
    assert chk["tmode"] == m["tmode"], (chk["tmode"], m["tmode"])
    print(f"  header round-trip: ft_width={chk['ftw']} "
          f"target_mode={chk['tmode']} (matches source)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
