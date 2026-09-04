"""S10-H0-B Repair 1: symmetry-orbit classification + WDL-EV metrics.

Re-classifies the B1 collision groups against the FULL legal symmetry
orbit of the V2 representation (no new SF work — the B1 report stored
one example pair per class; this tool re-runs the census grouping in
memory and classifies every group against all four transforms):

    I  = identity            (same board + STM)
    H  = horizontal mirror   (file a<->h)
    C  = rank flip + color swap + STM swap (the perspective transform:
         exactly what black-perspective features 'see' vs white)
    HC = H composed with C

Also repairs the B2 WDL reporting: exact tuple inequality -> WDL-EV
distance (EV = (W - L)/1000) and argmax class transitions.

Usage:
    python tools/s10/h0_b_repair1.py --out results/s10/s10-h0-b-repair1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.train_nnue import (  # noqa: E402
    export_features_from_engine, material_cp_stm_python,
)

ENGINE = r"target\release\eureka.exe"


def mop_up_class(fen: str) -> str:
    b = chess.Board(fen)
    if not b.is_valid():
        return "invalid"
    pieces = b.piece_map()
    heavy = [p for p in pieces.values()
             if p.piece_type in (chess.QUEEN, chess.ROOK)]
    if len(pieces) == 3 and len(heavy) == 1:
        return "krk_kqk"
    return "normal"


# ---- the four transforms, returning the transformed BOARD+STM key ----

def key_identity(fen):
    p = fen.split()
    return (p[0], p[1])


def key_h(fen):
    """Horizontal mirror: file a<->h, colors/stm unchanged."""
    b = chess.Board(fen)
    b2 = chess.Board()
    b2.clear_board()
    for sq, pc in b.piece_map().items():
        b2.set_piece_at(sq ^ 7, pc)
    return (b2.board_fen(), fen.split()[1])


def key_c(fen):
    """Rank flip + color swap + STM swap (perspective transform)."""
    b = chess.Board(fen)
    b2 = chess.Board()
    b2.clear_board()
    for sq, pc in b.piece_map().items():
        b2.set_piece_at(sq ^ 56, chess.Piece(pc.piece_type,
                                             not pc.color))
    stm = "b" if fen.split()[1] == "w" else "w"
    return (b2.board_fen(), stm)


def key_hc(fen):
    """H composed with C: sq ^ 63 (full point reflection) + color/stm swap."""
    b = chess.Board(fen)
    b2 = chess.Board()
    b2.clear_board()
    for sq, pc in b.piece_map().items():
        b2.set_piece_at(sq ^ 63, chess.Piece(pc.piece_type,
                                             not pc.color))
    stm = "b" if fen.split()[1] == "w" else "w"
    return (b2.board_fen(), stm)


def metadata_diff(fa: str, fb: str) -> str:
    pa, pb = fa.split(), fb.split()
    if pa[0] == pb[0] and pa[1] == pb[1]:
        diff = []
        if pa[2] != pb[2]:
            diff.append("castling")
        if pa[3] != pb[3]:
            diff.append("ep")
        if len(pa) > 4 and len(pb) > 4 and pa[4] != pb[4]:
            diff.append("halfmove")
        return "I-meta:" + ("+".join(diff) if diff else "fullmove")
    return None


def classify_group_pair(fa: str, fb: str) -> str:
    # same board+STM?
    md = metadata_diff(fa, fb)
    if md:
        return md
    if key_h(fa) == key_identity(fb) or key_h(fb) == key_identity(fa):
        return "H:mirror"
    if key_c(fa) == key_identity(fb) or key_c(fb) == key_identity(fa):
        return "C:perspective"
    if key_hc(fa) == key_identity(fb) or key_hc(fb) == key_identity(fa):
        return "HC:mirror-perspective"
    return "X:unexplained"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for shard in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob(
            "part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    labels = {}
    for line in Path(
            r"data\s10\s10-eval-v2-1m01\labels.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    print(f"records: {len(records)}", flush=True)

    print("exporting features...", flush=True)
    exported = export_features_from_engine(Path(ENGINE), records, "v2")

    groups = defaultdict(list)
    for i, r in enumerate(records):
        exp = exported[r["position_id"]]
        stm_white = r["fen"].split()[1] == "w"
        stm = sorted(exp["white"] if stm_white else exp["black"])
        nstm = sorted(exp["black"] if stm_white else exp["white"])
        sig = (tuple(stm), tuple(nstm),
               material_cp_stm_python(r["fen"]),
               mop_up_class(r["fen"]))
        h = hashlib.sha256(repr(sig).encode()).hexdigest()
        groups[h].append(i)
        if (i + 1) % 200000 == 0:
            print(f"  {i + 1}/{len(records)}", flush=True)

    class_counts = defaultdict(int)
    covered = 0
    class_examples = defaultdict(list)
    for h, idxs in groups.items():
        if len(idxs) < 2:
            continue
        covered += len(idxs)
        fa, fb = records[idxs[0]]["fen"], records[idxs[1]]["fen"]
        cls = classify_group_pair(fa, fb)
        class_counts[cls] += 1
        if len(class_examples[cls]) < 5:
            class_examples[cls].append({"a": fa, "b": fb,
                                        "size": len(idxs)})
    print(f"covered: {covered} | classified groups: "
          f"{sum(class_counts.values())}")
    for cls, n in sorted(class_counts.items()):
        print(f"  {cls}: {n}")

    # ---- B2 WDL repair (from the stored probe pairs) ----
    b2 = json.load(open(r"results\s10\s10-h0-b2-probe.json"))
    wdl_repair = {}
    for probe in ("castling", "ep", "rule50"):
        pairs = b2["pairs"][probe]
        ev_deltas = []
        class_trans = 0
        for p in pairs:
            o, c = p["orig"], p["cf"]
            ow, cw = o.get("wdl"), c.get("wdl")
            if not ow or not cw or None in ow or None in cw:
                continue
            oe = (ow[0] - ow[2]) / 1000
            ce = (cw[0] - cw[2]) / 1000
            ev_deltas.append(abs(ce - oe))
            if max(range(3), key=lambda i: ow[i]) != \
                    max(range(3), key=lambda i: cw[i]):
                class_trans += 1
        ev_sorted = sorted(ev_deltas)

        def pct(v, p):
            return v[min(len(v) - 1,
                         int(p / 100 * (len(v) - 1)))] if v else None
        wdl_repair[probe] = {
            "n": len(ev_deltas),
            "median_abs_d_ev": pct(ev_sorted, 50),
            "p95_abs_d_ev": pct(ev_sorted, 95),
            "max_abs_d_ev": ev_sorted[-1] if ev_sorted else None,
            "argmax_class_transitions": class_trans,
        }
        print(f"wdl-EV {probe}: {wdl_repair[probe]}")

    report = {
        "schema_version": 1,
        "collision_group_classification": {
            "covered_positions": covered,
            "group_counts": dict(class_counts),
            "examples": {k: v for k, v in class_examples.items()},
            "note": "Orbit = I (metadata-only), H (file mirror), "
                    "C (rank flip + color swap + STM swap — the "
                    "black-perspective transform), HC. The previous "
                    "B1 classifier only recognized H; the former "
                    "'C:unexplained' class is now split into C/HC/X.",
        },
        "b2_wdl_repair": wdl_repair,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
