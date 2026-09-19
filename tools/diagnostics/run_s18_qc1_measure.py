"""S18-QC1 frozen measurement battery. Two arms, same source diff, S14 model.

Runs: 3 cloud roots (depths 5/6/7 + node caps 50k/200k/1M), the game-127
boundary and mate endpoint, the 23 tactical validation cases with NNUE, and
the 32-position external cost panel x3 interleaved AB/BA rounds.
Writes one JSON; prints a compact pass/fail digest against the predeclared
gates. No engine edits here.
"""
import json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_s14_knight_diagnostics import Engine, info_fields

ROOT = Path(__file__).resolve().parents[2]
TMP = Path("C:/Users/81489/AppData/Local/Temp/eureka-s18-qc1")
MODEL = (ROOT / "data/s14/run/seed-20260908/nnue-s14-datasupply-v5.bin").resolve()
PLAN = json.loads((ROOT / "docs/diagnostics/s14-knight-cases-20260919.json").read_text())


def run_search(exe, pos_cmd, go, wait=45):
    eng = Engine([str(exe), "--profile", "current-final", "--evaluation", "nnue",
                  "--nnue-model", str(MODEL)])
    try:
        eng.send("uci"); eng.until("uciok", 15)
        eng.send("setoption name Hash value 16")
        eng.send("ucinewgame"); eng.send("isready"); eng.until("readyok", 15)
        eng.send(pos_cmd)
        t0 = time.monotonic()
        eng.send(go)
        lines = eng.until("bestmove ", wait)
        wall = (time.monotonic() - t0) * 1000
        best = lines[-1].split()[1]
        infos = [info_fields(l) for l in lines if l.startswith("info depth ")]
        return {"best": best, "wall_ms": round(wall, 1),
                "last": infos[-1] if infos else None}
    finally:
        eng.close()


def parse_epd(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("|")
        out.append({"id": f[0], "fen": f[2], "allowed": f[3].split(","),
                    "score_class": f[4], "depth": int(f[5])})
    return out


def rank(last):
    if last is None or last.get("cp") is None and "mate" not in last:
        return 0
    if "mate" in last:
        return 4 if last["mate"] > 0 else 0
    cp = last.get("cp")
    if cp is None:
        return 0
    return 3 if cp >= 300 else 2 if cp >= -150 else 1


def main():
    arms = {t: TMP / f"{t}.exe" for t in ("base", "qc1")}
    plan_cases = {c["game_number"]: c for c in PLAN["cases"]}

    def pos_of(case, branch=None):
        hist = case["history_uci"] + (branch["forced_uci"] if branch else [])
        return ("position fen " + case["initial_fen"] +
                (" moves " + " ".join(hist) if hist else ""))

    R = {"roots": {}, "boundary": {}, "tactical": {}, "cost": []}

    # 1. Three cloud roots: fixed depths and node caps.
    for g in (127, 175, 187):
        case = plan_cases[g]
        pos = pos_of(case)
        for go in ("go depth 5 movetime 15000", "go depth 6 movetime 15000",
                   "go depth 7 movetime 15000", "go nodes 50000 movetime 15000",
                   "go nodes 200000 movetime 15000", "go nodes 1000000 movetime 15000"):
            R["roots"].setdefault(str(g), {})[go] = {
                t: run_search(arms[t], pos, go) for t in arms}

    # 2. Boundary + mate endpoint at depth 1.
    case127 = plan_cases[127]
    for name in ("after_Qf3_check", "pv_endpoint_Kg1"):
        b = next(x for x in case127["branches"] if x["name"] == name)
        R["boundary"][name] = {t: run_search(arms[t], pos_of(case127, b),
                                             "go depth 1") for t in arms}

    # 3. Tactical corpus with NNUE at declared depth.
    for c in parse_epd(ROOT / "tests/data/search_validation.epd"):
        go = f"go depth {c['depth']} movetime 30000"
        res = {t: run_search(arms[t], "position fen " + c["fen"], go) for t in arms}
        fails = {}
        for t in arms:
            last, best = res[t]["last"], res[t]["best"]
            if c["score_class"] == "terminal-mate" or c["score_class"] == "terminal-draw":
                fails[t] = best != "0000"
            elif c["score_class"] == "losing":
                fails[t] = not (last and last.get("cp") is not None and last["cp"] < -150)
            else:
                need = {"winning": 3, "nonlosing": 2, "mate": 4}[c["score_class"]]
                bad_move = c["allowed"] != ["*"] and best not in c["allowed"]
                fails[t] = rank(last) < need or bad_move
        R["tactical"][c["id"]] = {"fails": fails,
                                  "score_class": c["score_class"],
                                  "best": {t: res[t]["best"] for t in arms}}

    # 4. External cost panel: 32 positions, depth 4, three interleaved rounds.
    ext = parse_epd(ROOT / "tests/data/external_validation_v1.epd")
    for rnd in range(3):
        order = ("base", "qc1") if rnd % 2 == 0 else ("qc1", "base")
        for c in ext:
            row = {"round": rnd, "id": c["id"]}
            for t in order:
                row[t] = run_search(arms[t], "position fen " + c["fen"], "go depth 4")
            R["cost"].append(row)

    out = TMP / "measurements.json"
    out.write_text(json.dumps(R, indent=1))
    print("wrote", out)

    # Digest against the predeclared gates.
    b, q = R["boundary"]["after_Qf3_check"]["base"], R["boundary"]["after_Qf3_check"]["qc1"]
    print("\n[boundary after_Qf3_check d1] base:",
          b["last"].get("cp"), b["best"], "| qc1:",
          q["last"].get("mate", q["last"].get("cp")), q["best"])
    b, q = R["boundary"]["pv_endpoint_Kg1"]["base"], R["boundary"]["pv_endpoint_Kg1"]["qc1"]
    print("[endpoint Kg1 d1] base:", b["last"].get("mate", b["last"].get("cp")),
          b["best"], "| qc1:", q["last"].get("mate", q["last"].get("cp")), q["best"])

    print("\n[roots] (bestmove, score, depth, nodes)")
    for g in ("127", "175", "187"):
        for go, res in R["roots"][g].items():
            if "depth" in go and go != "go depth 5 movetime 15000":
                continue
            f = lambda r: (r["best"], r["last"].get("cp", r["last"].get("mate")),
                           r["last"]["depth"], r["last"]["nodes"])
            print(f"  g{g} {go}: base={f(res['base'])} qc1={f(res['qc1'])}")
    for g in ("127", "175", "187"):
        for go, res in R["roots"][g].items():
            if "nodes" not in go:
                continue
            f = lambda r: (r["best"], r["last"].get("cp", r["last"].get("mate")),
                           r["last"]["depth"], r["last"]["nodes"])
            print(f"  g{g} {go}: base={f(res['base'])} qc1={f(res['qc1'])}")

    tf = {t: [i for i, v in R["tactical"].items() if v["fails"][t]] for t in arms}
    print("\n[tactical fails] base:", tf["base"], " qc1:", tf["qc1"],
          " NEW-FAILS(qc1-base):", [i for i in tf["qc1"] if i not in tf["base"]])

    ratios_n, ratios_w = [], []
    by_id = {}
    for row in R["cost"]:
        by_id.setdefault(row["id"], []).append(row)
    for cid, rows in by_id.items():
        for row in rows:
            n_b, n_q = row["base"]["last"]["nodes"], row["qc1"]["last"]["nodes"]
            w_b, w_q = row["base"]["wall_ms"], row["qc1"]["wall_ms"]
            ratios_n.append(n_q / max(n_b, 1))
            ratios_w.append(w_q / max(w_b, 0.1))
    ratios_n.sort(); ratios_w.sort()
    med = lambda xs: xs[len(xs) // 2]
    p90 = lambda xs: xs[int(len(xs) * 0.9)]
    print(f"\n[cost depth4 x32 x3] nodes ratio qc1/base: median={med(ratios_n):.2f} "
          f"p90={p90(ratios_n):.2f} | wall ratio: median={med(ratios_w):.2f} "
          f"p90={p90(ratios_w):.2f}")
    print("GATE nodes<=1.50:", med(ratios_n) <= 1.50, "p90<=3.0:", p90(ratios_n) <= 3.0,
          "GATE wall<=1.50:", med(ratios_w) <= 1.50)


if __name__ == "__main__":
    main()
