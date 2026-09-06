"""S10-J lockbox scorer — the FROZEN single source of truth for H0-E
lockbox evaluation (256 parents / 6,142 siblings / 75,818 pairs).

Established at J1-0 to end baseline drift (E3 pairwise had been
reported as both 58.1% and 58.3% across scripts). Every future
experiment evaluates through THIS module with a named model path; the
E3 baseline numbers below were produced by this exact code path and
are frozen in results/s10/s10-j0-lockbox-baseline.json.

Usage (library):
    from tools.s10.j0_lockbox import evaluate_checkpoint, E3_BASELINE
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import torch

import importlib.util as _ilu

_HERE = Path(__file__).parent
_spec = _ilu.spec_from_file_location("_tn", str(_HERE / "train_nnue.py"))
_tn = _ilu.module_from_spec(_spec)
import sys as _sys
_sys.modules.setdefault("_tn", _tn)
_spec.loader.exec_module(_tn)

EncodedSplit = _tn.EncodedSplit
export_features_from_engine = _tn.export_features_from_engine
material_cp_stm_python = _tn.material_cp_stm_python
NnueModel = _tn.NnueModel
NNUE_INPUTS_V2 = _tn.NNUE_INPUTS_V2

H0C_CACHE = Path(
    r"C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache")
EUREKA = Path(r"target\release\eureka.exe")
CLIP = 2000

# --- frozen corpus -------------------------------------------------------

def _load_parents():
    return [json.loads(l) for l in
            (H0C_CACHE / "e_search.jsonl").read_text(encoding="utf-8")
            .splitlines() if l.strip()]


def _teacher(sib):
    s = sib["sf"]
    if s.get("cp") is not None:
        return -s["cp"]
    if s.get("mate") is not None:
        return -CLIP if s["mate"] > 0 else CLIP
    return None


def _clamp(v):
    return max(-CLIP, min(CLIP, v))


# --- frozen static scorer ------------------------------------------------

def evaluate_checkpoint(ckpt_path, device="cpu"):
    """Static H0-E lockbox metrics for one torch checkpoint.

    Model score for a parent's move = -(material(child) +
    residual(child)); the child's STM is the opponent.
    """
    parents = _load_parents()
    child_fens = []
    for p in parents:
        b = chess.Board(p["fen"])
        for sib in p["siblings"]:
            b.push(chess.Move.from_uci(sib["uci"]))
            child_fens.append(b.fen())
            b.pop()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    ft_w = int(sd["ft_bias"].shape[0])
    dense_w = int(sd["l1.bias"].shape[0])
    model = NnueModel(num_inputs=NNUE_INPUTS_V2, ft_width=ft_w,
                      dense_width=dense_w)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    records = [{"position_id": f"c{i}", "fen": f}
               for i, f in enumerate(child_fens)]
    exported = export_features_from_engine(EUREKA, records, "v2")
    items = []
    for i, f in enumerate(child_fens):
        exp = exported[f"c{i}"]
        sw = f.split()[1] == "w"
        items.append({"stm": exp["white"] if sw else exp["black"],
                      "nstm": exp["black"] if sw else exp["white"]})
    enc = EncodedSplit([
        {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items])
    with torch.no_grad():
        preds = model(enc.stm_indices, enc.stm_offsets,
                      enc.nstm_indices, enc.nstm_offsets).numpy()
    child_eval = [_clamp(material_cp_stm_python(f) + float(p) * 1000.0)
                  for f, p in zip(child_fens, preds)]

    idx = 0
    for p in parents:
        p["_score"] = []
        for sib in p["siblings"]:
            p["_score"].append(-child_eval[idx])
            idx += 1

    regrets, acc20, acc50, top1 = [], 0, 0, 0
    agree = total = 0
    for p in parents:
        scored = []
        for si, sib in enumerate(p["siblings"]):
            t = _teacher(sib)
            if t is None:
                continue
            scored.append((t, p["_score"][si]))
        if not scored:
            continue
        best = max(v for v, _ in scored)
        pick_v, _ = max(scored, key=lambda x: x[1])
        regrets.append(best - pick_v)
        if best - pick_v <= 20:
            acc20 += 1
        if best - pick_v <= 50:
            acc50 += 1
        if pick_v == best:
            top1 += 1
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                v1, m1 = scored[i]
                v2, m2 = scored[j]
                if abs(v1 - v2) < 20:
                    continue
                if (v1 > v2) == (m1 > m2):
                    agree += 1
                total += 1
    r = sorted(regrets)
    return {
        "n": len(r),
        "mean_regret": round(sum(r) / len(r), 1),
        "median_regret": r[len(r) // 2],
        "acc20": round(100 * acc20 / len(r), 1),
        "acc50": round(100 * acc50 / len(r), 1),
        "top1": round(100 * top1 / len(r), 1),
        "pairwise": round(100 * agree / total, 1),
        "pairs": total,
    }


E3_CHECKPOINT = (r"data\s10\e3\scale-1m-win\seed-20260820"
                 r"\checkpoint_v2_s20260820.pt")

# Frozen once at J1-0 by this exact code path; every future comparison
# uses evaluate_checkpoint() against these numbers.
E3_BASELINE = None  # populated by j0_freeze_baseline.py into the JSON


if __name__ == "__main__":
    print(json.dumps(evaluate_checkpoint(E3_CHECKPOINT), indent=1))
