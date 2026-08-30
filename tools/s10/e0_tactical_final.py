"""Final analysis pass over the tactical-audit raw data.

The engine runs already happened (e0_tactical_audit.py wrote per-arm
iterations into memory only); re-derive from the SAME seed/sample and
compare, per position and per depth:
  * A (HCE) vs B (NNUE) vs C (NNUE conservative) bestmove agreement;
  * how often each arm's FINAL bestmove (depth 7) equals a move that
    allows an immediate >=threshold material capture (real blunder,
    detector fixed);
  * how often each arm's FINAL bestmove agrees with the SF18 teacher
    bestmove (the 16k-node searched reference);
  * root-score flips (EXPLORATORY: bestmove flip across >=2 depths with
    a >=200cp ROOT-score drop; NOT the frozen per-move forced-root
    reversal definition — kept out of the authoritative verdict).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_tactical_audit import (  # noqa: E402
    INFO_RE,
    load_validation,
    material_after_captures,
    run_search,
    run_search_forced,
)

import chess  # noqa: E402


def main() -> int:
    dataset = Path(r"data/s10/s10-eval-v1-300k01")
    engine = Path(r"target/release/eureka.exe")
    model = Path(r"data/s10/b3/seed-20260818/nnue-v2-q01.bin")
    n = 400
    depth = 7
    seed = 2026083004
    blunder_min = 300

    pool = load_validation(dataset)
    sample = random.Random(seed).sample(pool, min(n, len(pool)))

    arms = {
        "A_hce": dict(profile="current-final", nnue=None, diag=[]),
        "B_nnue": dict(
            profile="current-final-nnue-v2q", nnue=model, diag=[]
        ),
        "C_nnue_conservative": dict(
            profile="current-final-nnue-v2q", nnue=model,
            diag=["no-null", "no-futility"],
        ),
    }

    stats = {
        arm: {
            "final_blunders": 0,
            "shallow_blunders": 0,
            "teacher_bestmove_agreement": 0,
            "root_score_flips_exploratory": 0,
            "n": 0,
        }
        for arm in arms
    }
    agree_ab_final = 0
    agree_ac_final = 0
    refutation_cases = []

    from concurrent.futures import ThreadPoolExecutor

    def one(pos):
        out = {"fen": pos["fen"], "arms": {}}
        for arm, cfg in arms.items():
            try:
                r = run_search(
                    engine, pos["fen"], depth,
                    cfg["profile"], cfg["nnue"], cfg["diag"],
                )
                out["arms"][arm] = r
            except RuntimeError:
                out["arms"][arm] = None
        out["teacher_bestmove"] = pos["teacher_bestmove"]
        return out

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(one, sample):
            results.append(res)

    for res in results:
        board = chess.Board(res["fen"])
        finals = {}
        for arm in arms:
            r = res["arms"][arm]
            if not r or not r["iterations"]:
                continue
            iters = r["iterations"]
            last = iters[max(iters)]
            finals[arm] = last["bestmove"]
            s = stats[arm]
            s["n"] += 1
            # final-move blunder: does the FINAL bestmove hang material?
            mv = next(
                (m for m in board.legal_moves
                 if m.uci() == last["bestmove"]), None
            )
            if mv is not None:
                b2 = board.copy()
                b2.push(mv)
                g = material_after_captures(b2.fen())
                if g and g[0] >= blunder_min:
                    s["final_blunders"] += 1
            # shallow (depth-1) blunder
            first = iters[min(iters)]
            mv1 = next(
                (m for m in board.legal_moves
                 if m.uci() == first["bestmove"]), None
            )
            if mv1 is not None:
                b1 = board.copy()
                b1.push(mv1)
                g1 = material_after_captures(b1.fen())
                if g1 and g1[0] >= blunder_min:
                    s["shallow_blunders"] += 1
            # teacher BESTMOVE agreement (exact match of the SF18 16k-node
            # searched bestmove; NOT an SF-evaluation regret metric)
            if res["teacher_bestmove"]:
                if last["bestmove"] == res["teacher_bestmove"]:
                    s["teacher_bestmove_agreement"] += 1
            # root-score flips (EXPLORATORY): bestmove flip across >=2 depths with >=200cp root-score drop
            depths = sorted(iters)
            for d in depths:
                for d2 in (d + 2, d + 3):
                    if d2 not in iters:
                        continue
                    a, b = iters[d], iters[d2]
                    if (
                        a["bestmove"] != b["bestmove"]
                        and a["score"] - b["score"] >= 200
                    ):
                        s["root_score_flips_exploratory"] += 1
                    break
                break
        if "A_hce" in finals and "B_nnue" in finals:
            if finals["A_hce"] == finals["B_nnue"]:
                agree_ab_final += 1
        if "A_hce" in finals and "C_nnue_conservative" in finals:
            if finals["A_hce"] == finals["C_nnue_conservative"]:
                agree_ac_final += 1

        # deeper forced-root refutation check: for each of B's shallow
        # material-hang moves, re-search that move forced at full depth
        # and check whether the engine's own PV refutes it with the
        # counter-capture. This is a forced-root PV check — it does NOT
        # prove the depth-1 qsearch itself searched the capture.
        r = res["arms"].get("B_nnue")
        if r and r["iterations"]:
            iters = r["iterations"]
            first = iters[min(iters)]
            mv = next(
                (m for m in board.legal_moves
                 if m.uci() == first["bestmove"]), None
            )
            if mv is not None:
                b1 = board.copy()
                b1.push(mv)
                g = material_after_captures(b1.fen())
                if g and g[0] >= blunder_min:
                    deeper_pv = []
                    last = iters[max(iters)]
                    # engine's own deeper PV (root-level) and the forced PV
                    try:
                        fr = run_search_forced(
                            engine, res["fen"], depth, arms["B_nnue"],
                            first["bestmove"],
                        )
                        deeper_pv = fr.get("pv", [])
                    except RuntimeError:
                        pass
                    saw = bool(g[1]) and g[1] in deeper_pv
                    refutation_cases.append(
                        {
                            "fen": res["fen"],
                            "shallow_bm": first["bestmove"],
                            "opp_capture": g[1],
                            "gain": g[0],
                            "saw": saw,
                            "deeper_pv": deeper_pv[:8],
                        }
                    )

    summary = {
        "n": len(results),
        "depth": depth,
        "arms": stats,
        "agree_AB_final": agree_ab_final,
        "agree_AC_final": agree_ac_final,
        "forced_root_refutation_cases": {
            "seen": sum(1 for c in refutation_cases if c["saw"]),
            "missed": sum(1 for c in refutation_cases if not c["saw"]),
            "note": "forced-root PV membership; does not prove the "
                    "depth-1 qsearch itself searched the capture",
        },
    }
    print(json.dumps(summary, indent=2))
    Path("results/s10/s10-e0-tactical-final.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "forced_root_refutation_cases": refutation_cases[:50],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
