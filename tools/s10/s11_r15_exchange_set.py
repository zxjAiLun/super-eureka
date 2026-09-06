"""S11-A R1.5-3: exchange-motif regression set from EXISTING data.

Deterministic motif detector over positions we already have labels for:
  1. H0-E sibling children (SF-32k constrained, parent POV)
  2. 1M validation positions (Y16)

Motif: side-to-move can immediately capture an opponent piece that is
attacked-undefended (A state) — i.e. a hanging victim — subdivided by
victim type, and matched controls (same source, non-motif).

The victim metadata (type / defended / capture-by-pawn) is REGRESSION
SET ONLY — never a training feature.

Output: results/s10/s10-s11r15-exchange-set.json with the selected
positions + motif metadata, for the E3/R6/R14 signed-bias audit.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

H0C_CACHE = Path(
    r"C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache")
DS = Path(r"data\s10\s10-eval-v2-1m01")

TYPE_ORDER = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4}


def victim_info(board: chess.Board):
    """For the side to move: an immediately capturable OPPONENT piece on an
    attacked-undefended square (pseudo-attack semantics, matching R6/R14),
    with metadata. Returns the best (most valuable) such victim or None."""
    us = board.turn
    them = not us
    best = None
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        victim_sq = mv.to_square
        victim = board.piece_at(victim_sq)
        if victim is None or victim.color != them:
            continue  # en passant or own piece (impossible)
        # victim square attacked by us (it must be, we capture it) —
        # defended = attacked by them (excluding the occupant itself is
        # approximated as in R6/R14: raw is_square_attacked)
        attacked_by_them = board.is_attacked_by(them, victim_sq)
        if attacked_by_them:
            continue  # defended victim: not an A-state motif
        # capture exists onto an undefended victim
        info = {
            "victim_type": victim.symbol().upper(),
            "capture_uci": mv.uci(),
            "capture_by_pawn": board.piece_type_at(mv.from_square)
            == chess.PAWN,
            "from_type": board.piece_type_at(mv.from_square)
            and board.piece_at(mv.from_square).symbol().upper(),
        }
        if best is None or TYPE_ORDER[info["victim_type"]] > \
                TYPE_ORDER[best["victim_type"]]:
            best = info
    return best


def main():
    rng = random.Random(2026090901)

    # 1. H0-E sibling children with teacher scores (child-POV from the
    #    constrained search; we store the child FEN + its own label)
    parents = [json.loads(l) for l in
              (H0C_CACHE / "e_search.jsonl").read_text(encoding="utf-8")
              .splitlines() if l.strip()]
    candidates = []  # (fen, label_cp_child_stm, source)
    for p in parents:
        b = chess.Board(p["fen"])
        for sib in p["siblings"]:
            t = sib["sf"]["cp"]
            if t is None:
                continue
            bb = chess.Board(p["fen"])
            bb.push(chess.Move.from_uci(sib["uci"]))
            candidates.append((bb.fen(), t, "h0e_child"))

    # 2. 1M validation with Y16
    labels = {}
    for line in (DS / "labels.jsonl").read_text(encoding="utf-8") \
            .splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    for shard in sorted(DS.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["split"] != "validation":
                continue
            lab = labels.get(rec["position_id"])
            if lab is None or lab.get("teacher_cp_stm") is None:
                continue
            candidates.append((rec["fen"], lab["teacher_cp_stm"],
                               "validation"))

    print(f"candidate pool: {len(candidates)}")

    motifs = []
    controls = []
    for fen, label, source in candidates:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False) or board.is_check():
            continue
        info = victim_info(board)
        if info is not None:
            motifs.append((fen, label, source, info))
        else:
            controls.append((fen, label, source))

    print(f"motif positions: {len(motifs)} | non-motif: {len(controls)}")
    by_type = Counter(m["victim_type"] for *_, m in motifs)
    print("victim types:", dict(by_type))
    by_pawn = Counter(m["capture_by_pawn"] for *_, m in motifs)
    print("capture-by-pawn:", dict(by_pawn))

    # deterministic selection: 128 total
    #   32 hanging pawn, 32 hanging minor (N/B), 32 hanging R/Q,
    #   32 matched controls (same source mix, non-motif)
    def pick(pool, n, key):
        pool = sorted(pool, key=key)
        rng.shuffle(pool)
        return pool[:n]

    pawn = [m for m in motifs if m[3]["victim_type"] == "P"]
    minor = [m for m in motifs
             if m[3]["victim_type"] in ("N", "B")]
    rq = [m for m in motifs if m[3]["victim_type"] in ("R", "Q")]
    print(f"avail: pawn {len(pawn)}, minor {len(minor)}, R/Q {len(rq)}")

    sel_motif = (pick(pawn, 32, lambda x: x[0])
                 + pick(minor, 32, lambda x: x[0])
                 + pick(rq, 32, lambda x: x[0]))
    # controls matched on source distribution
    src_mix = Counter(m[2] for m in sel_motif)
    sel_controls = []
    for source, n in src_mix.items():
        pool = [c for c in controls if c[2] == source]
        sel_controls.extend(pick(pool, n, lambda x: x[0]))
    sel_controls = sel_controls[:128 - len(sel_motif)] \
        if len(sel_motif) < 128 else sel_controls[:0] if False else \
        sel_controls[:max(0, 128 - len(sel_motif))]

    out = {
        "schema_version": 1,
        "seed": 2026090901,
        "motif_positions": [
            {"fen": f, "label_cp": lab, "source": src, **meta}
            for f, lab, src, meta in sel_motif],
        "control_positions": [
            {"fen": f, "label_cp": lab, "source": src}
            for f, lab, src in sel_controls],
        "pool_stats": {
            "candidates": len(candidates),
            "motifs": len(motifs),
            "by_victim_type": dict(by_type),
            "by_capture_by_pawn": dict(by_pawn),
        },
    }
    Path(r"results\s10\s10-s11r15-exchange-set.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"selected: {len(sel_motif)} motif + {len(sel_controls)} "
          f"control positions")
    print("wrote results/s10/s10-s11r15-exchange-set.json")


if __name__ == "__main__":
    main()
