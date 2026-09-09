#!/usr/bin/env python3
"""S14 minimal candidate identity validation (weights-only change).

The V2R12/FT256/SCReLU/8-bucket architecture + v5 runtime + integer
semantics were fully proven at S12 (parity dfe60df) and re-confirmed at
S13; S14 only swaps the trained WEIGHTS (same recipe, new data). So the
frozen contract for a weights-only artifact is:
  1. PyInt (fp->i16/i32 quantized reference) vs Rust raw output,
     bit-exact on a 300-position frozen validation sample.
  2. UCI smoke: the engine loads the artifact under current-final-s12
     and produces a legal bestmove.
NOT re-run: the full corpus/transition/directed parity battery.
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

ART = Path(r"data\s14\run\seed-20260908\nnue-s14-datasupply-v5.bin")
ENGINE = Path(r"target\release\eureka.exe")
N = 300
SAMPLE_SEED = 2026090901

spec = ilu.spec_from_file_location("_tn", r"tools\s10\train_nnue.py")
_tn = ilu.module_from_spec(spec)
sys.modules.setdefault("_tn", _tn)
spec.loader.exec_module(_tn)

records = []
for p in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob("part-*.jsonl")):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
val = [r for r in records if r["split"] == "validation"]
sample = random.Random(SAMPLE_SEED).sample(val, N)
exported = _tn.export_features_from_engine(ENGINE, sample, "v2r12")

batch = Path(r"C:\Users\81489\AppData\Local\Temp\opencode\s14-ident.txt")
batch.parent.mkdir(parents=True, exist_ok=True)
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
    b = s12_train.output_bucket(len(chess.Board(r["fen"]).piece_map()))
    py = py_int_raw(q, stm, nstm, b)
    if py != rust[pid]:
        mism += 1
ident_pass = mism == 0
print(f"identity PyInt-vs-Rust: n={N} mismatches={mism} "
      f"{'PASS' if ident_pass else 'FAIL'}")

# UCI smoke: load artifact under current-final-s12 and get a bestmove.
uci = subprocess.run(
    [str(ENGINE), "--profile", "current-final-s12", "--nnue-model", str(ART)],
    input="uci\nisready\nposition startpos\ngo depth 10\nquit\n",
    capture_output=True, text=True, timeout=120)
bestmove = next((ln.split()[1] for ln in uci.stdout.splitlines()
                 if ln.startswith("bestmove")), None)
uci_pass = bestmove is not None and bestmove not in ("(none)", "0000")
print(f"uci smoke: bestmove={bestmove} {'PASS' if uci_pass else 'FAIL'}")

sys.exit(0 if (ident_pass and uci_pass) else 1)
