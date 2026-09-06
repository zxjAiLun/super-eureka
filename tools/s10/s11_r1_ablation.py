"""S11-A Repair 1 / R1-0: zero-cost channel ablations on the FROZEN R6
checkpoint (ac945bc, data/s10/s11/a/seed-20260820).

Inference-only: masks relation rows out of the feature lists BEFORE the
forward pass (equivalent to those inputs not existing, since the model is
a pure position function). No retraining, no SF.

Ablations:
  FULL          (baseline)
  ALL_R6_OFF    (no relation rows at all — the S11-A attribution check)
  -OWN_A  -OPP_A  -A (both A channels)
  -D (both D)   -C (both C)
  LOCAL: drop ONLY the A rows on square c6 (white view / black view / both)

Reports for each: dxc6 forensic composed cp + H0-E lockbox
(pairwise / mean regret / acc@20).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

H0C_CACHE = Path(
    r"C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache")
EUREKA = Path(r"target\release\eureka.exe")
CKPT = r"data\s10\s11\a\seed-20260820\checkpoint_v2r6_s20260820.pt"
REL_BASE = 22528
CLIP = 2000

FORENSIC = {
    "P0": "r1bqkbnr/ppp1pp1p/2np2p1/3P4/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 4",
    "after_Bg7": "r1bqk1nr/ppp1ppbp/2np2p1/3P4/2P5/2N5/PP2PPPP/R1BQKBNR w KQkq - 1 5",
    "after_dxc6": "r1bqk1nr/ppp1ppbp/2Pp2p1/8/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 5",
    "after_bxc6": "r1bqk1nr/p1p1ppbp/2pp2p1/8/2P5/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 6",
}

# channel ids in the R6 sidecar: state*2 + own ; state A=0 D=1 C=2
CH_OWN_A, CH_OWN_D, CH_OWN_C = 0, 2, 4
CH_OPP_A, CH_OPP_D, CH_OPP_C = 1, 3, 5


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    m = NnueModel(num_inputs=int(sd["ft_weights.weight"].shape[0]),
                  ft_width=int(sd["ft_bias"].shape[0]),
                  dense_width=int(sd["l1.bias"].shape[0]))
    m.load_state_dict(sd)
    m.eval()
    return m


def rel_ch(i):
    return (i - REL_BASE) // 64


def rel_sq(i):
    return (i - REL_BASE) % 64


def mask_rows(indices, drop):
    """drop: predicate over (channel, square) -> True to REMOVE."""
    return [i for i in indices if i < REL_BASE
            or not drop(rel_ch(i), rel_sq(i))]


def eval_fens(model, fens, drop):
    records = [{"position_id": f"c{i}", "fen": f} for i, f in enumerate(fens)]
    exported = export_features_from_engine(EUREKA, records, "v2r6")
    items = []
    for i, f in enumerate(fens):
        exp = exported[f"c{i}"]
        sw = f.split()[1] == "w"
        items.append({
            "stm": mask_rows(exp["white"] if sw else exp["black"], drop),
            "nstm": mask_rows(exp["black"] if sw else exp["white"], drop),
        })
    enc = EncodedSplit([
        {**it, "target_scaled": 0.0, "target_cp": 0.0} for it in items])
    with torch.no_grad():
        preds = model(enc.stm_indices, enc.stm_offsets,
                      enc.nstm_indices, enc.nstm_offsets).numpy()
    return preds


def composed(model, fens, drop):
    preds = eval_fens(model, fens, drop)
    return [max(-CLIP, min(CLIP, material_cp_stm_python(f)
                           + float(p) * 1000.0))
            for f, p in zip(fens, preds)]


def _teacher(sib):
    s = sib["sf"]
    if s.get("cp") is not None:
        return -s["cp"]
    if s.get("mate") is not None:
        return -CLIP if s["mate"] > 0 else CLIP
    return None


def lockbox(model, drop):
    parents = [json.loads(l) for l in
              (H0C_CACHE / "e_search.jsonl").read_text(encoding="utf-8")
              .splitlines() if l.strip()]
    child_fens = []
    for p in parents:
        b = chess.Board(p["fen"])
        for sib in p["siblings"]:
            b.push(chess.Move.from_uci(sib["uci"]))
            child_fens.append(b.fen())
            b.pop()
    child_eval = composed(model, child_fens, drop)
    idx = 0
    regrets, acc20, top1 = [], 0, 0
    agree = total = 0
    for p in parents:
        p["_s"] = []
        for sib in p["siblings"]:
            p["_s"].append(-child_eval[idx])
            idx += 1
        scored = []
        for si, sib in enumerate(p["siblings"]):
            t = _teacher(sib)
            if t is None:
                continue
            scored.append((t, p["_s"][si]))
        if not scored:
            continue
        best = max(v for v, _ in scored)
        pick_v, _ = max(scored, key=lambda x: x[1])
        regrets.append(best - pick_v)
        if best - pick_v <= 20:
            acc20 += 1
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
    return {"pairwise": round(100 * agree / total, 1),
            "mean_regret": round(sum(r) / len(r), 1),
            "acc20": round(100 * acc20 / len(r), 1)}


ABLATIONS = {
    "FULL": lambda ch, sq: False,
    "ALL_R6_OFF": lambda ch, sq: True,
    "-OWN_A": lambda ch, sq: ch == CH_OWN_A,
    "-OPP_A": lambda ch, sq: ch == CH_OPP_A,
    "-A": lambda ch, sq: ch in (CH_OWN_A, CH_OPP_A),
    "-D": lambda ch, sq: ch in (CH_OWN_D, CH_OPP_D),
    "-C": lambda ch, sq: ch in (CH_OWN_C, CH_OPP_C),
}

# LOCAL: drop only the A rows on c6 (transformed square). For the
# after_dxc6 position the white-perspective square of c6: white
# perspective is identity, c6 = file2 rank5 = sq 42. Black perspective:
# orient flips rank -> c6 = sq 21. We cannot know which accumulator's
# row drives it, so test each: white-side only, black-side only, both.
# Square identity per side must be computed per position, so the local
# ablations get their own drop functions built per FEN — here we
# approximate with "any A row on the file/rank transform of c6" by
# testing BOTH raw squares {42 (white-view), 21 (black-view)}.
C6_SQS = {42, 21}  # white-view c6 = 42; black-view c6 = 42^56=21? no:
# 42 = rank5*8+2 -> black orient flips rank: (7-5)*8+2 = 18? orient is
# sq ^ 56 = flip rank bits: 42 ^ 56 = 18. Recompute: 42 = 0b101010,
# 56 = 0b111000 -> 42^56 = 0b010010 = 18. So black-view c6 = 18.
C6_SQS = {42, 18}
ABLATIONS["-A@c6(both-sides)"] = (
    lambda ch, sq: ch in (CH_OWN_A, CH_OPP_A) and sq in C6_SQS)


def main():
    model = load_model()
    fens = list(FORENSIC.values())
    print(f"{'ablation':<20} {'dxc6':>8} {'pairwise':>9} "
          f"{'regret':>8} {'acc20':>6}")
    results = {}
    for name, drop in ABLATIONS.items():
        comp = composed(model, fens, drop)
        dxc6 = round(comp[2], 1)
        if name in ("FULL", "ALL_R6_OFF", "-A", "-A@c6(both-sides)"):
            lb = lockbox(model, drop)
            results[name] = {"dxc6": dxc6, **lb}
            print(f"{name:<20} {dxc6:>8} {lb['pairwise']:>9} "
                  f"{lb['mean_regret']:>8} {lb['acc20']:>6}")
        else:
            results[name] = {"dxc6": dxc6}
            print(f"{name:<20} {dxc6:>8} {'-':>9} {'-':>8} {'-':>6}")

    Path(r"results\s10\s10-s11a-r1-ablation.json").write_text(
        json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print("wrote results/s10/s10-s11a-r1-ablation.json")


if __name__ == "__main__":
    main()
