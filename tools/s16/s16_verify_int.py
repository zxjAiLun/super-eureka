"""S16 export verification: Python integer eval vs Rust runtime, bit-exact.

Mirrors the Rust runtime contract from s12_export's docstring:
  A  = ft_bias + sum(ft_w rows)            (i32, A units)
  y  = clamp(A, 0, qa)^2 // qa             (SCReLU, exact integer division)
  z  = l1_bias[bucket] + sum_j q_w[bucket][j] * y[j]   (i64 accumulate)
  raw= round_half_away(z >> dense_z_shift)
  cp = material_cp_stm + raw * 1000 / 2^12

The material_cp_stm term is NOT compared (it comes from the engine's own
material-batch exporter); we compare `raw_output`, which is the quantity the
network is solely responsible for.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ART = sys.argv[1]
FENS = [ln.strip() for ln in open(sys.argv[2], encoding="utf-8") if ln.strip()]
ENG = r"target\release\eureka.exe"
NUM_BUCKETS = 8

blob = open(ART, "rb").read()
magic = blob[:8]
(ver, inputs, ftw, tscale, fts, dws, dzs, qa, tmode, fset, hkind) = \
    struct.unpack("<IIIfIIIIIII", blob[8:8 + 44])
if magic != b"EUNN2Q01" or ver != 5:
    raise SystemExit(f"unexpected artifact header {magic} v{ver}")
off = 8 + 44 + 64
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
print(f"[artifact] inputs={inputs} ft_width={ftw} "
      f"shifts=({fts},{dws},{dzs}) qa={qa}", file=sys.stderr)


def _run(args, stdin_text=None):
    return subprocess.run(args, capture_output=True, text=True,
                          input=stdin_text, cwd=r"E:\AUbuntuProject\project\chessenginedemo")


def _batch_file(lines):
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def features(fen):
    # S16 is a V2R12 model: the bench default is V1, so the feature set MUST
    # be passed explicitly (bench.rs defaults to NnueFeatureSet::V1).
    p = _run([ENG, "bench", "nnue-features", "--feature-set", "v2r12",
              "--fen", fen])
    for ln in p.stdout.strip().splitlines():
        if ln.startswith("{"):
            return json.loads(ln)
    raise SystemExit("nnue-features gave no JSON: " + p.stdout[:200] + p.stderr[:200])


def bucket_of(fen):
    """(popcount_occ_incl_kings - 2) // 4 -- the frozen bucket rule."""
    board = fen.split()[0]
    n = sum(ch.isalpha() for ch in board)
    return (n - 2) // 4


def material_batch(fens):
    path = _batch_file(fens)
    try:
        p = _run([ENG, "bench", "material-batch", "--batch", path])
    finally:
        os.unlink(path)
    out = {}
    for ln in p.stdout.strip().splitlines():
        if ln.startswith("{"):
            j = json.loads(ln)
            out[j["fen"]] = j
    return out


def rust_probe_batch(fens):
    path = _batch_file(fens)
    try:
        p = _run([ENG, "bench", "nnue-v2q-probe-batch", "--model", ART,
                  "--batch", path])
    finally:
        os.unlink(path)
    out = {}
    for ln in p.stdout.strip().splitlines():
        if ln.startswith("{"):
            j = json.loads(ln)
            out[j["fen"]] = j
    if not out:
        raise SystemExit("probe gave no JSON: " + p.stdout[:300] + p.stderr[:300])
    return out


def _which(feat, side):
    """nnue-features emits the mover's list under "white"/"black"; the stm
    side is the side to move in the FEN."""
    return feat[side]


def screlu(a):
    y = np.clip(a, 0, qa)
    return (y * y) // qa


def py_raw(feat, bucket, stm_color):
    a_stm = ft_b.copy()
    for i in feat[stm_color]:
        a_stm += ft_w[i]
    a_nstm = ft_b.copy()
    for i in feat["white" if stm_color == "black" else "black"]:
        a_nstm += ft_w[i]
    hidden = np.concatenate([screlu(a_stm), screlu(a_nstm)])
    z = int(l1_b[bucket]) + int((l1_w[bucket] * hidden).sum())
    # Round-half-away on the UN-shifted z, matching the runtime exactly
    # (src/engine/nnue_v2q_runtime.rs: rounding is applied to z with
    # denom/2 added BEFORE dividing). Shifting first would discard the
    # remainder and can never match.
    denom = 1 << dzs
    if z >= 0:
        raw = (z + denom // 2) // denom
    else:
        raw = -((-z + denom // 2) // denom)
    return int(raw)


mats = material_batch(FENS)
rust = rust_probe_batch(FENS)

print(f"{'fen':44s} {'bucket':>6s} {'py_raw':>8s} {'rust_raw':>8s}  match")
print("-" * 84)
n_match = 0
for fen in FENS:
    feat = features(fen)
    bucket = bucket_of(fen)
    stm_color = "white" if fen.split()[1] == "w" else "black"
    a = py_raw(feat, bucket, stm_color)
    r = int(rust[fen]["raw_output"])
    ok = (a == r)
    n_match += ok
    print(f"{fen[:44]:44s} {bucket:>6d} {a:>8d} {r:>8d}  {ok}")
print("-" * 84)
print(f"EXACT MATCH: {n_match}/{len(FENS)}")
sys.exit(0 if n_match == len(FENS) else 1)
