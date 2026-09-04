"""S10-H0-B1: natural collision census over the frozen 1M dataset.

ZERO new Stockfish work. Uses the existing Y16 labels and the Rust V2
feature exporter as the single representation source of truth.

Production evaluator signature (what the deployed evaluator can
actually distinguish):

    sig = (
        sorted(stm feature indices),
        sorted(nstm feature indices),
        material_cp_stm,                 # canonical anchor term
        evaluator special-branch class,  # KQK/KRK mop-up vs normal
    )

STM is encoded by the [stm, nstm] concat order (not an extra bit), so
the two multisets are position-labeled, not order-insensitive overall.

Group all 1M positions by sig; for groups >= 2 classify:
  A. same board + same STM, metadata-only difference
     (castling / ep / halfmove / fullmove)
  B. horizontal-mirror-equivalent boards (the intentional HalfKAv2_hm
     symmetry — expected, reported separately)
  C. unexplained

Report group counts, coverage, max size, teacher dispersion inside
collision groups, and the empirical MAE lower bounds:

    global floor   = sum_g sum_i |y_i - median(y_g)| / 1,000,000
    subset floor   = sum_g sum_i |y_i - median(y_g)| / covered

Usage:
    python tools/s10/h0_b1_census.py --out results/s10/s10-h0-b1-census.json
"""

from __future__ import annotations

import argparse
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
    """Mirror the search's exact_mop_up branch classes."""
    b = chess.Board(fen)
    if not b.is_valid():
        return "invalid"
    pieces = b.piece_map()
    heavy = [p for p in pieces.values()
             if p.piece_type in (chess.QUEEN, chess.ROOK)]
    if len(pieces) == 3 and len(heavy) == 1:
        return "krk_kqk"
    return "normal"


def board_key(fen: str) -> str:
    """Board+STM identity (ignores castling/ep/halfmove/fullmove)."""
    parts = fen.split()
    return parts[0] + " " + parts[1]


def mirror_fen(fen: str) -> str:
    """Horizontally mirrored board (file a<->h), other fields kept.

    Mirrors the piece placement square-by-square (not string reverse,
    which would also flip digit runs incorrectly).
    """
    parts = fen.split()
    rows = parts[0].split("/")
    out_rows = []
    for row in rows:
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.append(ch)
            else:
                # map file: a<->h  =>  sq ^ 7
                cells.append(ch)
        # rebuild with mirrored squares: expand, mirror, re-pack
        expanded = []
        for ch in row:
            if ch.isdigit():
                expanded.extend([None] * int(ch))
            else:
                expanded.append(ch)
        expanded = expanded[::-1]
        packed = []
        run = 0
        for c in expanded:
            if c is None:
                run += 1
            else:
                if run:
                    packed.append(str(run))
                    run = 0
                packed.append(c)
        if run:
            packed.append(str(run))
        out_rows.append("".join(packed))
    parts[0] = "/".join(out_rows)
    # castling rights swap K<->Q sides
    if len(parts) > 2 and parts[2] != "-":
        m = {"K": "K", "Q": "Q", "k": "k", "q": "q"}
        parts[2] = "".join(sorted(set(parts[2])))
    return " ".join(parts[:2] + parts[2:])


def classify_pair(a_fen: str, b_fen: str) -> str:
    pa, pb = a_fen.split(), b_fen.split()
    if pa[0] == pb[0] and pa[1] == pb[1]:
        # same board, same STM: pure metadata difference
        diff = []
        if pa[2] != pb[2]:
            diff.append("castling")
        if pa[3] != pb[3]:
            diff.append("ep")
        if len(pa) > 4 and len(pb) > 4 and pa[4] != pb[4]:
            diff.append("halfmove")
        return "A:" + ("+".join(diff) if diff else "fullmove-only")
    # mirror check: board of b equals mirror of a (and vice versa)
    if pb[0] == mirror_fen(a_fen).split()[0] or \
            pa[0] == mirror_fen(b_fen).split()[0]:
        return "B:mirror"
    return "C:unexplained"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # ---- load dataset + labels ----
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

    # ---- export features via the Rust engine (representation truth) ----
    print("exporting features (Rust V2 exporter)...", flush=True)
    exported = export_features_from_engine(
        Path(ENGINE), records, "v2")
    print("features exported", flush=True)

    # ---- build signatures ----
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        exp = exported[r["position_id"]]
        stm_white = r["fen"].split()[1] == "w"
        stm = sorted(exp["white"] if stm_white else exp["black"])
        nstm = sorted(exp["black"] if stm_white else exp["white"])
        sig = (
            tuple(stm),
            tuple(nstm),
            material_cp_stm_python(r["fen"]),
            mop_up_class(r["fen"]),
        )
        import hashlib
        h = hashlib.sha256(repr(sig).encode()).hexdigest()
        groups[h].append(i)
        if (i + 1) % 200000 == 0:
            print(f"  signed {i + 1}/{len(records)}", flush=True)

    sizes = defaultdict(int)
    covered = 0
    for g in groups.values():
        sizes[len(g)] += 1
        if len(g) >= 2:
            covered += len(g)
    print(f"groups: {len(groups)} | coverage(>=2): {covered} "
          f"({100 * covered / len(records):.3f}%) | max group: "
          f"{max(sizes)}")

    # ---- collision analysis ----
    CLIP = 2000

    def clamp(v):
        return max(-CLIP, min(CLIP, v))

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def pct(vals, p):
        s = sorted(vals)
        return s[min(len(s) - 1, int(p / 100 * (len(s) - 1)))]

    class_counts = defaultdict(int)
    group_class_examples = defaultdict(list)
    lb_total = 0.0
    cp_groups = 0
    dispersion = []
    sign_disagree = 0
    bm_disagree = 0
    wdl_disagree = 0
    pair_count = 0

    for h, idxs in groups.items():
        if len(idxs) < 2:
            continue
        cps = []
        for i in idxs:
            lab = labels[records[i]["position_id"]]
            if lab.get("teacher_cp_stm") is not None:
                cps.append(clamp(lab["teacher_cp_stm"]))
        if len(cps) >= 2:
            cp_groups += 1
            med = median(cps)
            lb_total += sum(abs(c - med) for c in cps)
            disp = max(cps) - min(cps)
            dispersion.append(disp)
            pos = [c for c in cps if c >= 50]
            neg = [c for c in cps if c <= -50]
            if pos and neg:
                sign_disagree += 1
        # classification on the first pair of the group
        a, b = records[idxs[0]], records[idxs[1]]
        cls = classify_pair(a["fen"], b["fen"])
        class_counts[cls.split(":")[0]] += 1
        group_class_examples[cls].append(
            {"a": a["fen"], "b": b["fen"], "size": len(idxs)})
        # bestmove/wdl disagreement across all pairs
        labs = [labels[records[i]["position_id"]] for i in idxs]
        for x in range(len(labs)):
            for y in range(x + 1, len(labs)):
                pair_count += 1
                if labs[x].get("teacher_bestmove") != \
                        labs[y].get("teacher_bestmove"):
                    bm_disagree += 1
                if list(labs[x].get("teacher_wdl_stm") or []) != \
                        list(labs[y].get("teacher_wdl_stm") or []):
                    wdl_disagree += 1

    global_floor = lb_total / len(records)
    subset_floor = lb_total / covered if covered else 0
    print()
    print(f"collision groups with >=2 cp labels: {cp_groups}")
    print(f"global collision MAE floor:  {global_floor:.3f} cp "
          f"(over all {len(records)})")
    print(f"collision-subset MAE floor:  {subset_floor:.1f} cp "
          f"(over {covered} covered)")
    if dispersion:
        print(f"teacher CP dispersion in groups: "
              f"median {median(dispersion):.0f} "
              f"p90 {pct(dispersion, 90):.0f} "
              f"p95 {pct(dispersion, 95):.0f} "
              f"max {max(dispersion):.0f}")
    print(f"sign disagreements (>=+50 & <=-50 in group): {sign_disagree}")
    print(f"bestmove disagreements: {bm_disagree}/{pair_count} pairs")
    print(f"WDL disagreements:      {wdl_disagree}/{pair_count} pairs")
    print(f"class counts: {dict(class_counts)}")
    for cls, exs in group_class_examples.items():
        print(f"  {cls}: {len(exs)} groups, example: "
              f"{exs[0]['a'][:60]} <-> {exs[0]['b'][:60]}")

    report = {
        "schema_version": 1,
        "records": len(records),
        "groups": len(groups),
        "covered": covered,
        "coverage_pct": round(100 * covered / len(records), 4),
        "max_group_size": max(sizes),
        "group_size_histogram": {str(k): v for k, v in sorted(
            sizes.items()) if k >= 2},
        "cp_groups": cp_groups,
        "global_collision_mae_floor_cp": round(global_floor, 4),
        "collision_subset_mae_floor_cp": round(subset_floor, 1),
        "dispersion": {
            "median": median(dispersion) if dispersion else None,
            "p90": pct(dispersion, 90) if dispersion else None,
            "p95": pct(dispersion, 95) if dispersion else None,
            "max": max(dispersion) if dispersion else None,
        },
        "sign_disagree_groups": sign_disagree,
        "bestmove_disagree_pairs": bm_disagree,
        "wdl_disagree_pairs": wdl_disagree,
        "total_pairs": pair_count,
        "class_counts": dict(class_counts),
        "class_examples": {k: v[:5] for k, v in
                           group_class_examples.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
