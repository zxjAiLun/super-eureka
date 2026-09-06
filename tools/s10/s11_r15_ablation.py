"""S11-A R1.5-1: R14 inference-only channel ablations.

Zero training, zero SF. Masks relation rows before the forward pass on
the FROZEN R14 checkpoint (94fe03b, data/s11/r1/seed-20260820).

R14 channel layout (see nnue.rs):
  ch 0..4  = OWN_A_{P,N,B,R,Q}
  ch 5..9  = OPP_A_{P,N,B,R,Q}
  ch 10/11 = OWN_D / OPP_D
  ch 12/13 = OWN_C / OPP_C

Ablations: FULL, each typed-A channel, all-A, each type both sides,
A_P local on c6 (both/white/black perspective squares), D, C.
Reports dxc6 composed cp; lockbox for the informative subset.
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
CKPT = r"data\s10\s11\r1\seed-20260820\checkpoint_v2r14_s20260820.pt"
REL_BASE = 22528
CLIP = 2000

FORENSIC_FENS = [
    "r1bqkbnr/ppp1pp1p/2np2p1/3P4/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 4",
    "r1bqk1nr/ppp1ppbp/2np2p1/3P4/2P5/2N5/PP2PPPP/R1BQKBNR w KQkq - 1 5",
    "r1bqk1nr/ppp1ppbp/2Pp2p1/8/2P5/2N5/PP2PPPP/R1BQKBNR b KQkq - 0 5",
    "r1bqk1nr/p1p1ppbp/2pp2p1/8/2P5/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 6",
]

# typed-A channel ids
OWN_A = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q"}
OPP_A = {5: "P", 6: "N", 7: "B", 8: "R", 9: "Q"}
OWN_D, OPP_D, OWN_C, OPP_C = 10, 11, 12, 13

# c6 transformed squares: white-view 42, black-view 18 (42 ^ 56).
C6_WHITE, C6_BLACK = 42, 18


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
    return [i for i in indices if i < REL_BASE
            or not drop(rel_ch(i), rel_sq(i))]


def composed(model, fens, drop):
    records = [{"position_id": f"c{i}", "fen": f}
               for i, f in enumerate(fens)]
    exported = export_features_from_engine(EUREKA, records, "v2r14")
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
    regrets, acc20 = [], 0
    agree = total = 0
    for p in parents:
        scores = []
        for sib in p["siblings"]:
            scores.append(-child_eval[idx])
            idx += 1
        scored = []
        for si, sib in enumerate(p["siblings"]):
            t = _teacher(sib)
            if t is None:
                continue
            scored.append((t, scores[si]))
        if not scored:
            continue
        best = max(v for v, _ in scored)
        pick_v, _ = max(scored, key=lambda x: x[1])
        regrets.append(best - pick_v)
        if best - pick_v <= 20:
            acc20 += 1
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


def main():
    model = load_model()

    def never(ch, sq):
        return False

    ablations = {
        "FULL": never,
        "-OWN_A_P": lambda ch, sq: ch == 0,
        "-OPP_A_P": lambda ch, sq: ch == 5,
        "-A_P": lambda ch, sq: ch in (0, 5),
        "-A_N": lambda ch, sq: ch in (1, 6),
        "-A_B": lambda ch, sq: ch in (2, 7),
        "-A_R": lambda ch, sq: ch in (3, 8),
        "-A_Q": lambda ch, sq: ch in (4, 9),
        "-A (all)": lambda ch, sq: ch <= 9,
        "-D": lambda ch, sq: ch in (10, 11),
        "-C": lambda ch, sq: ch in (12, 13),
        # local: only the A_P rows on c6
        "-A_P@c6(both)": (
            lambda ch, sq: ch in (0, 5) and sq in (C6_WHITE, C6_BLACK)),
        "-A_P@c6(w-side only)": (
            lambda ch, sq: ch in (0, 5) and sq == C6_WHITE),
        "-A_P@c6(b-side only)": (
            lambda ch, sq: ch in (0, 5) and sq == C6_BLACK),
        # any A row on c6 regardless of victim type
        "-A_any@c6": (
            lambda ch, sq: ch <= 9 and sq in (C6_WHITE, C6_BLACK)),
    }

    results = {}
    print(f"{'ablation':<24} {'dxc6':>8}")
    for name, drop in ablations.items():
        comp = composed(model, FORENSIC_FENS, drop)
        dxc6 = round(comp[2], 1)
        entry = {"dxc6": dxc6}
        if name in ("FULL", "-A_P", "-A (all)", "-A_P@c6(both)",
                    "-A_any@c6"):
            lb = lockbox(model, drop)
            entry.update(lb)
            print(f"{name:<24} {dxc6:>8}   pw={lb['pairwise']} "
                  f"regret={lb['mean_regret']} acc20={lb['acc20']}")
        else:
            print(f"{name:<24} {dxc6:>8}")
        results[name] = entry

    Path(r"results\s10\s10-s11r15-ablation.json").write_text(
        json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print("wrote results/s10/s10-s11r15-ablation.json")


if __name__ == "__main__":
    main()
