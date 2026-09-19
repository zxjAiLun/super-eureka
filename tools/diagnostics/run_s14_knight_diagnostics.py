#!/usr/bin/env python3
"""Bounded, serial UCI diagnostics; stdlib-only so the production ELF can run in WSL.

No searchmoves (not implemented by this production UCI). Forced branches are
positions reached by appending legal moves to the frozen full history. Every
search uses a new process/empty TT; this does not recreate historical hot TT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import subprocess
import threading
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info_fields(line):
    out = {"raw": line}
    for field in ("depth", "seldepth", "nodes", "time", "nps"):
        m = re.search(r"\b" + field + r" (\d+)", line)
        if m:
            out[field] = int(m[1])
    m = re.search(r"\bscore (cp|mate) (-?\d+)", line)
    if m:
        out[m[1]] = int(m[2])
    out["bound"] = "lowerbound" if "lowerbound" in line else "upperbound" if "upperbound" in line else None
    out["pv"] = line.split(" pv ", 1)[1].split() if " pv " in line else []
    return out


class Engine:
    def __init__(self, command):
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.q = queue.Queue()
        self.transcript = []
        def read_lines():
            for line in self.p.stdout:
                self.q.put(line.rstrip())
            self.q.put(None)
        self.reader = threading.Thread(target=read_lines, daemon=True)
        self.reader.start()

    def send(self, command):
        self.transcript.append("> " + command)
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()

    def until(self, prefix, seconds):
        deadline = time.monotonic() + seconds
        lines = []
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("waiting for " + prefix)
            try:
                line = self.q.get(timeout=left)
            except queue.Empty as exc:
                raise TimeoutError("waiting for " + prefix) from exc
            if line is None:
                raise RuntimeError("engine EOF before " + prefix)
            self.transcript.append("< " + line)
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def close(self):
        if self.p.poll() is None:
            try:
                self.send("quit")
                self.p.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.p.kill()
                self.p.wait(timeout=3)
        self.p.stdin.close()
        self.reader.join(timeout=1)
        self.p.stdout.close()


def search(exe, model, plan, case, point, go, out, name, wait_seconds=25):
    engine = Engine([str(exe), "--evaluation", "nnue", "--nnue-model", str(model)])
    try:
        engine.send("uci")
        handshake = engine.until("uciok", 10)
        assert "info string source " + plan["source_commit"] in handshake
        assert "info string eval nnue-v2q" in handshake
        assert "info string network nnue-v2q" in handshake
        assert "info string evalfile " + model.name in handshake
        engine.send("setoption name Hash value " + str(plan["hash_mb"]))
        engine.send("ucinewgame")
        engine.send("isready")
        engine.until("readyok", 10)
        history = case["history_uci"] + point.get("forced_uci", [])
        engine.send("position fen " + case["initial_fen"] + (" moves " + " ".join(history) if history else ""))
        start = time.monotonic()
        engine.send(go)
        # Unexpected stalls are errors, not silently accepted partial results.
        # The exact process is cleaned up; follow-ups declare their own cap.
        lines = engine.until("bestmove ", wait_seconds)
        elapsed_ms = (time.monotonic() - start) * 1000
        best = lines[-1].split()[1]
        infos = [info_fields(l) for l in lines if l.startswith("info depth ")]
        if point["checkmate"]:
            assert best == "0000", (name, best)
        else:
            assert best in point["legal_moves"], (name, "illegal bestmove", best)
            assert infos and ("cp" in infos[-1] or "mate" in infos[-1])
        record = {
            "name": name, "game": case["game_number"], "point": point.get("name", "root"),
            "fen": point["fen"], "history_length": len(history), "go": go,
            "white_to_move": point["white_to_move"], "bestmove": best,
            "elapsed_ms": round(elapsed_ms, 3), "last": infos[-1] if infos else None,
            "iterations": infos, "handshake": handshake,
        }
        m = re.search(r"\bdepth (\d+)", go)
        if m:
            record["requested_depth"] = int(m[1])
            record["depth_reached"] = bool(infos and infos[-1].get("depth", 0) >= int(m[1]))
        return record
    finally:
        engine.close()
        (out / (name + ".uci.txt")).write_text("\n".join(engine.transcript) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(args.engine) == plan["binary_sha256"]
    assert sha(args.model) == plan["model_sha256"]
    assert plan["threads"] == 1
    os.nice(10)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "plan.json").write_text(args.plan.read_text())
    identity = {"source_commit": plan["source_commit"], "binary_sha256": sha(args.engine),
                "model_sha256": sha(args.model), "plan_sha256": sha(args.plan),
                "runner_sha256": sha(__file__),
                "host": platform.uname()._asdict(), "niceness": os.nice(0),
                "engine": str(args.engine), "model": str(args.model),
                "tt_policy": plan["tt_policy"], "started_at": time.time()}
    (args.out / "identity.json").write_text(json.dumps(identity, indent=2))
    results = []
    try:
        for case in plan["cases"]:
            root = case["root"]
            branches = case["branches"]
            # First test the reported quiet-mate endpoint directly.
            branches = sorted(branches, key=lambda b: 0 if b["name"] == "pv_endpoint_Kg1" else 1)
            if args.smoke:
                branches = branches[:1]
            for point in branches:
                depths = [1] if point["checkmate"] or args.smoke else plan["branch_depths"]
                for depth in depths:
                    name = f"g{case['game_number']}-{point['name']}-d{depth}"
                    go = f"go depth {depth} movetime {plan['branch_time_cap_ms']}"
                    r = search(args.engine, args.model, plan, case, point, go, args.out, name)
                    results.append(r)
                    with (args.out / "results.jsonl").open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(name, r["bestmove"], r["last"], flush=True)
            if args.smoke:
                break
            jobs = [("historical-clock-cold", case["historical_go"])]
            jobs += [(f"t{ms}", f"go movetime {ms}") for ms in plan["root_time_ms"]]
            jobs += [(f"d{d}", f"go depth {d} movetime {plan['depth_time_cap_ms']}") for d in plan["root_depths"]]
            for label, go in jobs:
                name = f"g{case['game_number']}-root-{label}"
                r = search(args.engine, args.model, plan, case, root, go, args.out, name)
                results.append(r)
                with (args.out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(name, r["bestmove"], r["last"], flush=True)
            for point in [root] + case["branches"]:
                if point["checkmate"]:
                    continue  # Static NNUE is not a game-termination adjudicator.
                command = [str(args.engine), "bench", "nnue-v2q-probe", "--model", str(args.model), "--fen", point["fen"]]
                p = subprocess.run(command, text=True, capture_output=True, timeout=10, check=True)
                probe = json.loads(p.stdout)
                assert probe["fen"] == point["fen"]
                with (args.out / "static.jsonl").open("a") as f:
                    f.write(json.dumps({"game": case["game_number"], "point": point.get("name", "root"),
                                        "white_to_move": point["white_to_move"], "probe": probe}) + "\n")
        (args.out / "complete.json").write_text(json.dumps({"status": "complete", "searches": len(results), "finished_at": time.time()}, indent=2))
    except BaseException as exc:
        (args.out / "error.json").write_text(json.dumps({"error": repr(exc), "completed_searches": len(results)}, indent=2))
        raise


if __name__ == "__main__":
    main()
