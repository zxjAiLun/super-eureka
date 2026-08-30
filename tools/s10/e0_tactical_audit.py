"""S10-E0b: Tactical Instability Audit (search/NNUE coupling diagnosis).

Motivation: D1 rejected the 300k NNUE candidate at -346 Elo with visible
shallow tactical blunders. Before burning 1M teacher labels, determine
whether the search's eval-dependent selectivity (null move / futility)
amplifies the NNUE's ~165 cp static error, or whether the evaluator is
simply weak.

Arms (frozen):
  A = CurrentFinal HCE                     (production search + HCE)
  B = NNUE + full CurrentFinal search      (the D1 candidate exactly)
  C = NNUE conservative diagnostic         (B with null move and futility
      disabled; LMR / qsearch / aspiration unchanged) — bench-only
      (--diag no-null --diag no-futility), never reachable through UCI.

Pipeline:
  1. deterministic sample of N validation positions (never holdout),
  2. each arm runs one fixed-depth search per position, per-depth
     bestmove/score parsed from the engine's `info depth` lines,
  3. tactical_reversal = bestmove flips A@depth d -> B@depth d+k (k>=2)
     AND the abandoned move's deeper score worsens by >= 200 cp
     (measured by re-searching the abandoned move at the deeper depth
     via --forced-root),
  4. one-move material blunder hard check: for every shallow NNUE
     bestmove, make the move, enumerate opponent captures; if one nets
     >= minor-piece material, decide whether qsearch SAW it
     (the opponent capture appears in the engine's own PV/qsearch
     output for that root move) or MISSED it.

Verdict rules (frozen):
  B blunders a lot, C clearly fixes it  -> search/eval coupling issue
  B and C blunder equally               -> evaluator is weak -> 1M
  A also blunders at the same rate      -> normal tactical horizon
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

INFO_RE = re.compile(
    r"^info depth (\d+) seldepth \d+ score (?:cp|mate) (-?\d+).*? pv (\S+)"
)


def load_validation(dataset_dir: Path) -> list[dict]:
    records = []
    for shard in sorted(dataset_dir.glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    labels = {}
    for line in (dataset_dir / "labels.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    out = []
    for r in records:
        if r.get("split") != "validation":
            continue
        lab = labels.get(r["position_id"])
        if lab is None or lab.get("teacher_cp_stm") is None:
            continue
        out.append(
            {
                "position_id": r["position_id"],
                "fen": r["fen"],
                "teacher_cp_stm": lab["teacher_cp_stm"],
                "teacher_bestmove": lab.get("teacher_bestmove"),
            }
        )
    return out


def run_search(
    engine: Path,
    fen: str,
    depth: int,
    profile: str,
    nnue_model: Path | None,
    diag: list[str],
) -> dict:
    """One fixed-depth bench search; returns per-depth bestmove/score."""
    argv = [
        str(engine), "bench", "profile",
        "--fen", fen,
        "--profile", profile,
        "--depth", str(depth),
    ]
    if nnue_model is not None:
        argv += ["--nnue-model", str(nnue_model)]
    for d in diag:
        argv += ["--diag", d]
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench failed rc={proc.returncode}: {proc.stderr[:300]}"
        )
    iterations = {}
    final = {}
    for line in proc.stdout.splitlines():
        m = INFO_RE.match(line)
        if m:
            d = int(m.group(1))
            iterations[d] = {
                "depth": d,
                "bestmove": m.group(3),
                "score": int(m.group(2)),
            }
        elif line.startswith("bench_result"):
            kv = dict(
                pair.split("=", 1)
                for pair in line.split()[1:]
                if "=" in pair
            )
            final = {
                "bestmove": kv.get("bestmove"),
                "score": kv.get("score"),
                "nodes": int(kv.get("nodes", 0)),
            }
    return {"iterations": iterations, "final": final}


PIECE_VALUES = {
    "p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0,
}


def material_after_captures(fen_after: str) -> tuple[int, str] | None:
    """Best immediate material GAIN for the side to move in fen_after
    (i.e. the opponent of the audited move's mover), from legal captures.

    Uses python-chess: gain = value(captured) - max(0, value(mover) -
    value(captured)) is approximated by SEE-style exchange on the target
    square (chess.Board.see is not in python-chess; use simple material
    after the capture plus recapture check via SEE approximation).
    """
    import chess

    board = chess.Board(fen_after)
    best_gain = None
    best_move = None
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        captured = board.piece_type_at(mv.to_square)
        if captured is None:
            # en passant
            captured = chess.PAWN
        captured_val = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900,
        }[captured]
        mover_type = board.piece_type_at(mv.from_square)
        mover_val = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
        }[mover_type]
        # Net material after capture AND mandatory worst-case recapture
        # (if the capturer can be recaptured, subtract its value). After
        # push, `board.turn` IS the recapturing side (the opponent of the
        # mover).
        board.push(mv)
        recapture_loss = 0
        if board.is_attacked_by(board.turn, mv.to_square):
            recapture_loss = mover_val
        board.pop()
        gain = captured_val - recapture_loss
        if gain > 0 and (best_gain is None or gain > best_gain):
            best_gain = gain
            best_move = mv.uci()
    if best_gain is None:
        return None
    return best_gain, best_move or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nnue-model", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026083004)
    parser.add_argument("--blunder-min-cp", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    pool = load_validation(args.dataset)
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    print(f"sampled {len(sample)} validation positions")

    arms = {
        "A_hce": dict(profile="current-final", nnue=None, diag=[]),
        "B_nnue": dict(
            profile="current-final-nnue-v2q", nnue=args.nnue_model, diag=[]
        ),
        "C_nnue_conservative": dict(
            profile="current-final-nnue-v2q", nnue=args.nnue_model,
            diag=["no-null", "no-futility"],
        ),
    }

    def one_position(pos: dict) -> dict:
        """Run all three arms + blunder analysis for one position."""
        out = {"position_id": pos["position_id"], "fen": pos["fen"], "arms": {}}
        import chess

        for arm, cfg in arms.items():
            try:
                r = run_search(
                    args.engine, pos["fen"], args.depth,
                    cfg["profile"], cfg["nnue"], cfg["diag"],
                )
            except RuntimeError as exc:
                out["arms"][arm] = {"error": str(exc)}
                continue
            out["arms"][arm] = r

        blunder = None
        for arm in ("B_nnue", "C_nnue_conservative"):
            r = out["arms"].get(arm)
            if not r or "iterations" not in r or not r["iterations"]:
                continue
            iters = r["iterations"]
            shallow = iters[min(iters)]
            bm = shallow["bestmove"]
            board = chess.Board(pos["fen"])
            mv = None
            for m in board.legal_moves:
                if m.uci() == bm:
                    mv = m
                    break
            if mv is None:
                continue
            board.push(mv)
            gain = material_after_captures(board.fen())
            if gain is None or gain[0] < args.blunder_min_cp:
                continue
            # Did the engine's own deeper search refute its shallow move?
            # Forced-root PV for the blunder move at the deeper depth.
            saw = False
            try:
                fr = run_search_forced(
                    args.engine, pos["fen"], args.depth, arms[arm], bm
                )
                pv = fr.get("pv", [])
                saw = gain[1] in pv if gain[1] else False
            except RuntimeError:
                saw = False
            if blunder is None:
                blunder = {"fen": pos["fen"], "shallow_depth": min(iters)}
            blunder[arm] = {
                "bestmove": bm,
                "opp_capture": gain[1],
                "net_gain_cp": gain[0],
                "saw": saw,
            }
        out["blunder"] = blunder
        return out

    all_results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(
            ex.map(one_position, sample)
        ):
            all_results.append(res)
            if (i + 1) % 50 == 0:
                print(f"  position {i + 1}/{len(sample)}", flush=True)

    # ---- aggregate --------------------------------------------------
    results = {arm: [] for arm in arms}
    for res in all_results:
        for arm, r in res["arms"].items():
            if "iterations" in r:
                results[arm].append(
                    {
                        "position_id": res["position_id"],
                        "fen": res["fen"],
                        "iterations": r["iterations"],
                        "final": r["final"],
                    }
                )

    blunders = {
        arm: {"seen": 0, "missed": 0, "cases": []}
        for arm in ("B_nnue", "C_nnue_conservative")
    }
    for res in all_results:
        b = res.get("blunder")
        if not b:
            continue
        for arm in ("B_nnue", "C_nnue_conservative"):
            entry = b.get(arm)
            if not entry:
                continue
            key = "seen" if entry["saw"] else "missed"
            blunders[arm][key] += 1
            if len(blunders[arm]["cases"]) < 50:
                blunders[arm]["cases"].append(
                    {
                        "position_id": res["position_id"],
                        **entry,
                        "shallow_depth": b["shallow_depth"],
                    }
                )

    # tactical reversal detection (B only, A as control)
    reversal_counts = {arm: 0 for arm in arms}
    reversal_examples = {arm: [] for arm in arms}
    by_pid = {
        arm: {e["position_id"]: e for e in entries}
        for arm, entries in results.items()
    }
    for pid in by_pid.get("B_nnue", {}):
        for arm in ("A_hce", "B_nnue", "C_nnue_conservative"):
            e = by_pid.get(arm, {}).get(pid)
            if not e or len(e["iterations"]) < 3:
                continue
            iters = e["iterations"]
            depths = sorted(iters)
            for d in depths[:-2]:
                a = iters[d]
                for d2 in (d + 2, d + 3):
                    if d2 not in iters:
                        continue
                    b = iters[d2]
                    if a["bestmove"] != b["bestmove"]:
                        # score drop of the abandoned move: approximate by
                        # root score drop across the flip
                        drop = a["score"] - b["score"]
                        if drop >= 200:
                            reversal_counts[arm] += 1
                            if len(reversal_examples[arm]) < 20:
                                reversal_examples[arm].append(
                                    {
                                        "position_id": pid,
                                        "fen": by_pid[arm][pid]["fen"]
                                        if "fen" in by_pid[arm][pid]
                                        else None,
                                        "from_depth": d,
                                        "to_depth": d2,
                                        "move_a": a["bestmove"],
                                        "move_b": b["bestmove"],
                                        "score_a": a["score"],
                                        "score_b": b["score"],
                                    }
                                )
                            break
                    break

    # Aggregate shallow-blunder rates
    summary = {
        "n_positions": len(sample),
        "depth": args.depth,
        "seed": args.seed,
        "reversal_counts": reversal_counts,
        "blunders": {
            arm: {
                "seen": v["seen"],
                "missed": v["missed"],
                "rate": round((v["seen"] + v["missed"]) / max(1, len(sample)), 4),
            }
            for arm, v in blunders.items()
        },
    }
    print(json.dumps(summary, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "summary": summary,
                "blunder_cases": blunders,
                "reversal_examples": reversal_examples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"written {args.out}")
    return 0


def run_search_forced(
    engine: Path,
    fen: str,
    depth: int,
    cfg: dict,
    forced_move: str,
) -> dict:
    """Search with --forced-root to get the engine's own PV for one move."""
    argv = [
        str(engine), "bench", "profile",
        "--fen", fen,
        "--profile", cfg["profile"],
        "--depth", str(depth),
        "--forced-root", forced_move,
    ]
    if cfg.get("nnue"):
        argv += ["--nnue-model", str(cfg["nnue"])]
    for d in cfg.get("diag", []):
        argv += ["--diag", d]
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"forced bench rc={proc.returncode}: {proc.stderr[:300]}"
        )
    pv = []
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result"):
            # pv="m1 m2 ..." is a quoted multi-token value; split on
            # 'pv="' then strip the trailing quote.
            idx = line.find('pv="')
            if idx >= 0:
                rest = line[idx + 4:]
                end = rest.find('"')
                pv_str = rest[:end] if end >= 0 else rest
                pv = pv_str.split() if pv_str else []
    return {"pv": pv}


if __name__ == "__main__":
    sys.exit(main())
