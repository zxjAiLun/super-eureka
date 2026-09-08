#!/usr/bin/env python3
"""S13 minimal candidate identity validation (weights-only change).

Python int vs Rust raw (bit-exact) on a 300-position frozen sample +
UCI smoke. The architecture/runtime parity was proven at S12 (dfe60df);
R1 only swaps weights.
"""
import json
import random
import subprocess
import sys
from importlib import util as ilu
from pathlib import Path

sys.path.insert(0, r"tools\s12")
from s12_parity import load_v5, py_int_raw  # noqa: E402
import s12_train  # noqa: E402

import chess  # noqa: E402

ART = Path(r"data\s13\r1-blend075\seed-20260908\nnue-s13-blend-v5.bin")
ENGINE = Path(r"target\release\eureka.exe")

spec = ilu.spec_from_file_location(
    "_tn", r"tools\s10\train_nnue.py")
_tn = ilu.module_from_spec(spec)
sys.modules.setdefault("_tn", _tn)
spec.loader.exec_module(_tn)

records = []
for p in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob("part-*.jsonl")):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
val = [r for r in records if r["split"] == "validation"]
sample = random.Random(2026090802).sample(val, 300)
exported = _tn.export_features_from_engine(ENGINE, sample, "v2r12")

batch = Path(r"C:\Users\81489\AppData\Local\Temp\opencode\s13-ident.txt")
batch.write_text(
    "".join(f"{r['position_id']}|{r['fen']}\n" for r in sample),
    encoding="utf-8")
out = subprocess.run(
    [str(ENGINE), "bench", "nnue-v2q-probe-batch",
     "--model", str(ART), "--batch", str(batch)],
    capture_output=True, text=True, timeout=600, check=True)
rust = {}
for line in out.stdout.splitlines():
    if line.strip().startswith('{"position_id"'):
        rec = json.loads(line)
        rust[rec["position_id"]] = rec["raw_output"]

q = load_v5(ART)
mism = 0
for r in sample:
    pid = r["position_id"]
    exp = exported[pid]
    stm_white = r["fen"].split()[1] == "w"
    stm = exp["white"] if stm_white else exp["black"]
    nstm = exp["black"] if stm_white else exp["white"]
    b = s12_train.output_bucket(
        len(chess.Board(r["fen"]).piece_map()))
    py = py_int_raw(q, stm, nstm, b)
    if py != rust[pid]:
        mism += 1

print(f"identity PyInt-vs-Rust: n=300 mismatches={mism} "
      f"{'PASS' if mism == 0 else 'FAIL'}")
sys.exit(0 if mism == 0 else 1)
