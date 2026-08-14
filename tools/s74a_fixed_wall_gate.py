#!/usr/bin/env python3
"""S7.4A fixed-wall depth gate: current-final vs current-final-lmr-null-window.

80 S7 positions x {1000ms, 3000ms}, A/B interleaved (rotated order), one
fresh engine process per run (cold TT, threads 1). Reports completed-depth
distribution, median/mean depth, seldepth, nodes, bestmove.

Writes results/s7/s74a-fixed-wall-gate.json (incremental, --resume).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
OUT = REPO / "results/s7/s74a-fixed-wall-gate.json"
A = "current-final"
B = "current-final-lmr-null-window"


def run(engine: Path, profile: str, fen: str, movetime_ms: int) -> dict | None:
    """Keep stdin OPEN until the engine emits `bestmove`, then send `quit`.

    Piping all commands at once (including `quit`) makes a UCI-compliant
    engine abort the search immediately, which is why a naive
    `subprocess.run(input=...)` here used to return the depth-1 move.
    """
    import time

    deadline = time.monotonic() + (movetime_ms / 1000.0) + 30.0
    try:
        proc = subprocess.Popen(
            [str(engine), "--profile", profile],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
    except OSError:
        return {"error": True}
    try:
        proc.stdin.write(f"uci\nisready\nposition fen {fen}\n"
                         f"go movetime {movetime_ms}\n")
        proc.stdin.flush()
        depth = seldepth = nodes = None
        bestmove = None
        import threading

        def reader():
            nonlocal depth, seldepth, nodes, bestmove
            assert proc.stdout is not None
            for line in proc.stdout:
                toks = line.split()
                if toks[:2] == ["info", "depth"]:
                    try:
                        depth = int(toks[2])
                        if "seldepth" in toks:
                            seldepth = int(toks[toks.index("seldepth") + 1])
                        if "nodes" in toks:
                            nodes = int(toks[toks.index("nodes") + 1])
                    except (ValueError, IndexError):
                        pass
                elif toks[:1] == ["bestmove"] and len(toks) >= 2:
                    bestmove = toks[1]
                    return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(max(0.1, deadline - time.monotonic()))
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass
        proc.wait(timeout=10)
        return {"depth": depth, "seldepth": seldepth, "nodes": nodes,
                "bestmove": bestmove}
    finally:
        if proc.poll() is None:
            proc.kill()


def depth_bucket(d: int | None) -> str:
    if d is None:
        return "none"
    if d >= 9:
        return "9+"
    return str(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--times", default="1000,3000")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    times = [int(t) for t in args.times.split(",")]
    positions = [json.loads(l) for l in
                 CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    results = {"rows": []}
    done = set()
    if args.resume and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r["id"], r["movetime"]) for r in results["rows"]}

    for t in times:
        for idx, pos in enumerate(positions):
            if (pos["id"], t) in done:
                continue
            order = (A, B) if (idx + t) % 2 == 0 else (B, A)
            runs = {p: run(engine, p, pos["fen"], t) for p in order}
            rec = {"id": pos["id"], "movetime": t, "fen": pos["fen"],
                   A: runs[A], B: runs[B]}
            da, db = runs[A].get("depth"), runs[B].get("depth")
            if da is not None and db is not None:
                rec["depth_gain"] = db - da
            results["rows"].append(rec)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
            print(f"s74a_wall {pos['id']} {t}ms d {da} -> {db} "
                  f"(gain {rec.get('depth_gain', '?')})", flush=True)

    # Aggregate per time control.
    summaries = {}
    for t in times:
        rs = [r for r in results["rows"] if r["movetime"] == t
              and r.get("depth_gain") is not None]
        da = [r[A]["depth"] for r in rs]
        db = [r[B]["depth"] for r in rs]
        gained = sum(1 for r in rs if r["depth_gain"] > 0)
        lost = sum(1 for r in rs if r["depth_gain"] < 0)
        dist_a = {}
        dist_b = {}
        for r in rs:
            for dist, prof in ((dist_a, A), (dist_b, B)):
                b_ = depth_bucket(r[prof]["depth"])
                dist[b_] = dist.get(b_, 0) + 1
        sa = [r[A].get("seldepth") for r in rs if r[A].get("seldepth")]
        sb = [r[B].get("seldepth") for r in rs if r[B].get("seldepth")]
        summaries[f"movetime{t}"] = {
            "positions": len(rs),
            "median_depth_a": statistics.median(da),
            "median_depth_b": statistics.median(db),
            "mean_depth_a": round(statistics.mean(da), 3),
            "mean_depth_b": round(statistics.mean(db), 3),
            "median_seldepth_a": statistics.median(sa) if sa else None,
            "median_seldepth_b": statistics.median(sb) if sb else None,
            "gained": gained, "lost": lost,
            "depth_distribution_a": dist_a, "depth_distribution_b": dist_b,
        }
    results["summary"] = summaries
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"s74a_wall summary: "
          f"{json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in summaries.items()})}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s74a_wall_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
