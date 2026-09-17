#!/usr/bin/env python3
"""S15: encode converted position shards straight into the binary cache.

This replaces the S14 `assemble` + `encode` pair for NEW data. Instead of
writing an intermediate JSONL.gz of encoded records (which the trainer then
re-parses on every run), it streams the convert output, encodes with the
engine's own exporters, and writes EKC cache shards directly.

Frozen semantics preserved from S14:
  * features: engine `bench nnue-features-batch --feature-set v2r12`
    (mover perspective = stm, opponent = nstm, no sign flip for black)
  * material: engine `bench material-batch` -> material_cp_stm
  * label: teacher_cp_raw clamped to +-2000
  * bucket: (piece_count - 2) // 4, count includes kings
  * dedup: exact-FEN, first occurrence in scan order wins
  * canonical overlap dropped (the canonical corpus stays the sole source
    of its own positions)

Deliberate change vs S14: the bucket's piece count is derived directly from
the FEN board field (count of piece letters) instead of constructing a
`chess.Board`. Same value, much cheaper over millions of positions.
`--check-bucket N` asserts the fast path against chess.Board on the first
N positions.

Resumable: one output cache shard per accepted chunk, and `state.json`
records which input shards are done. On restart the dedup set is rebuilt by
re-reading ONLY the `fen_before` field of already-processed inputs, so
dedup stays exact across restarts.

Usage:
  python tools/s15/s15_encode_cache.py \
      --input-glob "data/s15/shards/*.jsonl.gz" --out data/s15/cache-new
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
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
PIECES = set("PNBRQKpnbrqk")


def fen_hash(fen: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest(), "big")


def piece_count_fast(fen: str) -> int:
    """Count piece letters in the FEN board field (== len(board.piece_map()))."""
    board_field = fen.split(" ", 1)[0]
    return sum(1 for ch in board_field if ch in PIECES)


def run_batch(engine: Path, cmd_extra: list, batch: Path) -> dict:
    proc = subprocess.run(
        [str(engine), "bench"] + cmd_extra + ["--batch", str(batch)],
        capture_output=True, text=True, timeout=14400, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
            out[rec["position_id"]] = rec
    return out


def encode_chunk(engine: Path, fens: list[str]) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        for i, fen in enumerate(fens):
            fh.write(f"p{i}|{fen}\n")
        batch = Path(fh.name)
    try:
        feats = run_batch(engine,
                          ["nnue-features-batch", "--feature-set", FEATURE_SET],
                          batch)
        mats = run_batch(engine, ["material-batch"], batch)
    finally:
        batch.unlink(missing_ok=True)
    return feats, mats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-glob", default=r"data\s15\shards\*.jsonl.gz")
    ap.add_argument("--out", type=Path, default=Path(r"data\s15\cache-new"))
    ap.add_argument("--corpus", type=Path,
                    default=Path(r"data\s10\s10-eval-v2-1m01"))
    ap.add_argument("--engine", type=Path,
                    default=Path(r"target\release\eureka.exe"))
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--check-bucket", type=int, default=3000)
    args = ap.parse_args()

    if not args.engine.exists():
        raise SystemExit(f"FAIL CLOSED: engine not found: {args.engine}")
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.json"
    state = {"processed_inputs": [], "shards": [], "positions": 0,
             "dropped_dup": 0, "dropped_corpus": 0}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"[encode] resuming: {len(state['processed_inputs'])} inputs "
              f"done, {state['positions']} positions", flush=True)

    inputs = sorted(glob.glob(args.input_glob))
    if not inputs:
        raise SystemExit(f"FAIL CLOSED: no inputs match {args.input_glob}")
    done = set(state["processed_inputs"])
    todo = [p for p in inputs if Path(p).name not in done]
    print(f"[encode] inputs total={len(inputs)} todo={len(todo)}", flush=True)

    # ---- dedup set ----
    seen: set[int] = set()
    if not args.no_dedup:
        corpora = sorted(args.corpus.glob("part-*.jsonl"))
        n_corpus = 0
        for p in corpora:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(fen_hash(json.loads(line)["fen"]))
                    n_corpus += 1
        # rebuild from processed inputs so dedup stays exact across restarts
        for name in state["processed_inputs"]:
            path = Path(args.input_glob).parent / name
            if not path.exists():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        seen.add(fen_hash(json.loads(line)["fen_before"]))
        print(f"[encode] dedup set: corpus={n_corpus} "
              f"total={len(seen)}", flush=True)

    shard_idx = len(state["shards"])
    t0 = time.time()
    buf_fens: list[str] = []
    buf_cp: list[float] = []
    buf_raw: list[float] = []
    checked = 0

    def flush_shard():
        nonlocal shard_idx, buf_fens, buf_cp, buf_raw
        if not buf_fens:
            return
        feats, mats = encode_chunk(args.engine, buf_fens)
        stm_parts, nstm_parts = [], []
        stm_len, nstm_len = [], []
        material, bucket, cp, raw = [], [], [], []
        for i, fen in enumerate(buf_fens):
            pid = f"p{i}"
            f = feats[pid]
            stm_white = fen.split()[1] == "w"
            white, black = f["white"], f["black"]
            s = white if stm_white else black
            n = black if stm_white else white
            stm_parts.append(np.asarray(s, dtype=np.int16))
            nstm_parts.append(np.asarray(n, dtype=np.int16))
            stm_len.append(len(s))
            nstm_len.append(len(n))
            material.append(float(mats[pid]["material_cp_stm"]))
            bucket.append(s12_train.output_bucket(piece_count_fast(fen)))
            cp.append(buf_cp[i])
            raw.append(buf_raw[i])
        stm_flat = np.concatenate(stm_parts) if stm_parts else np.zeros(
            0, dtype=np.int16)
        nstm_flat = np.concatenate(nstm_parts) if nstm_parts else np.zeros(
            0, dtype=np.int16)
        stm_off = np.zeros(len(stm_len) + 1, dtype=np.int64)
        nstm_off = np.zeros(len(nstm_len) + 1, dtype=np.int64)
        np.cumsum(stm_len, out=stm_off[1:])
        np.cumsum(nstm_len, out=nstm_off[1:])
        path = args.out / f"cache-new-{shard_idx:04d}.ekc"
        info = nnue_cache.write_shard(
            path, stm_flat=stm_flat, nstm_flat=nstm_flat,
            stm_off=stm_off, nstm_off=nstm_off,
            material=np.asarray(material, dtype=np.float32),
            bucket=np.asarray(bucket, dtype=np.int8),
            cp=np.asarray(cp, dtype=np.float32),
            raw=np.asarray(raw, dtype=np.float32),
            feature_set=FEATURE_SET, max_index=MAX_INDEX,
            idx_dtype=nnue_cache.choose_index_dtype(MAX_INDEX),
            meta={"source": "s15_convert"})
        # read-back verification
        back = nnue_cache.read_shard(path)
        if int(back["stm_off"][-1]) != stm_flat.size:
            raise SystemExit(f"FAIL CLOSED: read-back stm_off mismatch {path}")
        if not np.array_equal(back["bucket"], np.asarray(bucket,
                                                         dtype=np.int8)):
            raise SystemExit(f"FAIL CLOSED: read-back bucket mismatch {path}")
        if not np.array_equal(back["cp"], np.asarray(cp, dtype=np.float32)):
            raise SystemExit(f"FAIL CLOSED: read-back cp mismatch {path}")
        state["shards"].append({**info, "file": path.name})
        state["positions"] += info["n"]
        shard_idx += 1
        el = time.time() - t0
        print(f"  {path.name}: n={info['n']} {info['bytes']/2**20:.1f} MiB "
              f"total={state['positions']} ({el:.0f}s)", flush=True)
        buf_fens, buf_cp, buf_raw = [], [], []

    for src in todo:
        with gzip.open(src, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                fen = r["fen_before"]
                if not args.no_dedup:
                    h = fen_hash(fen)
                    if h in seen:
                        state["dropped_dup"] += 1
                        continue
                    seen.add(h)
                if checked < args.check_bucket:
                    expect = s12_train.output_bucket(
                        len(chess.Board(fen).piece_map()))
                    got = s12_train.output_bucket(piece_count_fast(fen))
                    if expect != got:
                        raise SystemExit(
                            f"FAIL CLOSED: fast bucket {got} != "
                            f"chess.Board bucket {expect} for {fen}")
                    checked += 1
                buf_fens.append(fen)
                rc = max(-CLIP_CP, min(CLIP_CP, float(r["teacher_cp_raw"])))
                buf_cp.append(rc)
                buf_raw.append(float(r["teacher_cp_raw"]))
                if len(buf_fens) >= args.chunk:
                    flush_shard()
        flush_shard()
        state["processed_inputs"].append(Path(src).name)
        state_path.write_text(json.dumps(state, indent=1) + "\n",
                              encoding="utf-8")
    flush_shard()
    state_path.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: state[k] for k in
                      ("positions", "dropped_dup", "dropped_corpus")},
                     indent=1))
    print(f"[encode] shards={len(state['shards'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
