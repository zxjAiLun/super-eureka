"""S10-H0-C: search-site distribution-shift audit.

Phase 1: 128 deterministic validation roots (32 per phase bucket),
50k-node searches each with the production profile+model, capturing
every evaluator call site (FEN + main_static/qsearch_standpat).

Phase 2: deterministic selection of 1024 unique sites (balanced
main/qsearch x high/mid/low/zero, overflow redistributed within the
same phase), labeled with the production 16k teacher.

Phase 3: phase x material-matched control of 1024 validation
positions (existing Y16 labels, zero SF cost).

Verdict (frozen): R = MAE(F128, search-sites) / MAE(F128, control)
  R >= 1.20 or delta >= 20cp  -> SUPPORTED (shift real)
  R <= 1.10 and delta <= 10cp -> NOT SUPPORTED
  else                        -> AMBIGUOUS (extend to 2048 later)

Usage:
    python tools/s10/h0_c_shift.py --phase collect
    python tools/s10/h0_c_shift.py --phase label
    python tools/s10/h0_c_shift.py --phase report
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.train_nnue import export_features_from_engine  # noqa: E402

EUREKA = r"target\release\eureka.exe"
MODEL128 = (r"data\s10\e3\scale-1m-win\seed-20260820"
            r"\nnue-v2-q01-material-v3twin.bin")
MODEL256 = (r"data\s10\g1\ft256-1m\seed-20260819"
            r"\nnue-v2-q01-material-256.bin")
SF = (r"C:\Users\81489\AppData\Local\Temp\opencode\sf18-win\stockfish"
      r"\stockfish-windows-x86-64-avx2.exe")
CLIP = 2000
ROOT_SEED = 2026090403
CACHE = Path(r"C:\Users\81489\AppData\Local\Temp\opencode\h0c-cache")
PHASE_WEIGHTS = {"n": 1, "b": 1, "r": 2, "q": 4}


def phase_of(fen):
    b = chess.Board(fen)
    return sum(PHASE_WEIGHTS.get(pc.symbol().lower(), 0)
               for pc in b.piece_map().values())


def phase_bucket(p):
    return "high" if p >= 18 else "mid" if p >= 8 else \
        "low" if p >= 1 else "zero"


def mat_abs(fen):
    b = chess.Board(fen)
    w = t = 0
    for pc in b.piece_map().values():
        v = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900}.get(
            pc.symbol().lower(), 0)
        if pc.color == chess.WHITE:
            w += v
        else:
            t += v
    return abs(w - t)


def mat_bucket(m):
    return "<=50" if m <= 50 else "51-250" if m <= 250 else \
        "251-500" if m <= 500 else ">500"


def clamp(v):
    return max(-CLIP, min(CLIP, v))


# ---------------------------------------------------------------- phase 1
def collect():
    CACHE.mkdir(exist_ok=True)
    records = []
    for shard in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob(
            "part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["split"] == "validation":
                    records.append(rec)
    by_phase = defaultdict(list)
    for r in records:
        by_phase[phase_bucket(phase_of(r["fen"]))].append(r)
    rng = random.Random(ROOT_SEED)
    roots = []
    for ph in ("high", "mid", "low", "zero"):
        pool = sorted(by_phase[ph], key=lambda r: r["position_id"])
        roots.extend(rng.sample(pool, 32))
    print(f"roots: {len(roots)} (32 x 4 phases)", flush=True)

    site_hits = defaultdict(int)  # fen+site -> hit_count
    t0 = time.time()
    for i, r in enumerate(roots):
        out = subprocess.run(
            [EUREKA, "bench", "eval-site-capture", "--fen", r["fen"],
             "--nodes", "50000",
             "--profile", "current-final-nnue-v2q-material",
             "--nnue-model", MODEL128, "--hash-mb", "32"],
            capture_output=True, text=True, timeout=1800, check=True)
        for line in out.stdout.splitlines():
            if line.startswith('{"fen"'):
                rec = json.loads(line)
                site_hits[(rec["fen"], rec["site"])] += 1
        if (i + 1) % 16 == 0:
            print(f"  searched {i + 1}/128 roots "
                  f"({len(site_hits)} unique sites, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    uniques = [{"fen": f, "site": s, "hits": n}
               for (f, s), n in site_hits.items()]
    (CACHE / "sites.jsonl").write_text(
        "\n".join(json.dumps(u) for u in uniques) + "\n", encoding="utf-8")
    total_calls = sum(site_hits.values())
    print(f"unique sites: {len(uniques)} | total evaluator calls: "
          f"{total_calls}", flush=True)
    # workload prevalence stats (H0-B follow-up)
    ep = sum(u["hits"] for u in uniques if u["fen"].split()[3] != "-")
    any_castling = sum(u["hits"] for u in uniques
                       if u["fen"].split()[2] != "-")
    high_r50 = sum(u["hits"] for u in uniques
                   if int(u["fen"].split()[4]) >= 50)
    print(f"workload prevalence (call-weighted): ep-square {ep} "
          f"({100 * ep / total_calls:.2f}%), castling-rights "
          f"{any_castling} ({100 * any_castling / total_calls:.2f}%), "
          f"halfmove>=50 {high_r50} ({100 * high_r50 / total_calls:.2f}%)")


# ---------------------------------------------------------------- phase 2
def select_1024():
    sites = [json.loads(l) for l in
             (CACHE / "sites.jsonl").read_text(encoding="utf-8")
             .splitlines() if l.strip()]
    rng = random.Random(2026090404)
    cells = defaultdict(list)
    for u in sites:
        cells[(u["site"], phase_bucket(phase_of(u["fen"])))].append(u)
    for k in cells:
        cells[k].sort(key=lambda u: u["fen"])
    selected = []
    # target: 8 cells x 128
    order = [("main_static", p) for p in ("high", "mid", "low", "zero")] + \
            [("qsearch_standpat", p)
             for p in ("high", "mid", "low", "zero")]
    deficit = 0
    picked = {}
    for site_kind, ph in order:
        cell = cells[(site_kind, ph)]
        take = min(128, len(cell))
        rng.shuffle(cell)
        picked[(site_kind, ph)] = cell[:take]
        deficit += 128 - take
        print(f"cell {site_kind}/{ph}: {take}/128")
    # redistribute deficit within the SAME phase, other site kind first
    for (site_kind, ph) in order:
        need = 128 - len(picked[(site_kind, ph)])
        if need <= 0 or deficit <= 0:
            continue
        other = "qsearch_standpat" if site_kind == "main_static" \
            else "main_static"
        pool = [u for u in cells[(other, ph)]
                if u not in picked[(other, ph)]]
        take = min(need, len(pool))
        if take:
            picked[(site_kind, ph)].extend(pool[:take])
            deficit -= take
    for v in picked.values():
        selected.extend(v)
    print(f"selected: {len(selected)}")
    (CACHE / "selected.jsonl").write_text(
        "\n".join(json.dumps(u) for u in selected) + "\n",
        encoding="utf-8")


def sf_label(fen):
    p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    def send(cmd):
        p.stdin.write(cmd + "\n"); p.stdin.flush()
    send("uci")
    while p.stdout.readline().strip() != "uciok":
        pass
    send("setoption name Threads value 1")
    send("setoption name Hash value 64")
    send("setoption name MultiPV value 1")
    send("setoption name UCI_ShowWDL value true")
    send("ucinewgame")
    send(f"position fen {fen}")
    send("go nodes 16384")
    final = {}
    best = None
    while True:
        line = p.stdout.readline()
        if not line or line.startswith("bestmove"):
            if line.startswith("bestmove"):
                best = line.split()[1]
            break
        if not line.startswith("info") or " pv " not in line:
            continue
        m = re.search(r"score cp (-?\d+)|score mate (-?\d+)", line)
        if m:
            final["cp"] = int(m.group(1)) if m.group(1) is not None else None
            final["mate"] = (int(m.group(2))
                             if m.group(2) is not None else None)
        pv = re.search(r" pv (\S+)", line)
        if pv:
            final["bestmove"] = pv.group(1)
    p.stdin.write("quit\n"); p.stdin.flush()
    p.wait(timeout=30)
    return {"cp": final.get("cp"), "mate": final.get("mate"),
            "bestmove": final.get("bestmove", best)}


def label():
    selected = [json.loads(l) for l in
                (CACHE / "selected.jsonl").read_text(encoding="utf-8")
                .splitlines() if l.strip()]
    print(f"labeling {len(selected)} sites with the 16k teacher...",
          flush=True)
    t0 = time.time()
    for i, u in enumerate(selected):
        u["y16"] = sf_label(u["fen"])
        if (i + 1) % 128 == 0:
            print(f"  {i + 1}/{len(selected)} ({time.time() - t0:.0f}s)",
                  flush=True)
    (CACHE / "labeled.jsonl").write_text(
        "\n".join(json.dumps(u) for u in selected) + "\n", encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- phase 3
def report():
    labeled = [json.loads(l) for l in
               (CACHE / "labeled.jsonl").read_text(encoding="utf-8")
               .splitlines() if l.strip()]
    # eureka evals
    def probe(model, tag):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False,
                                         encoding="utf-8") as fh:
            fh.write("\n".join(f"{tag}{i}|{u['fen']}"
                               for i, u in enumerate(labeled)))
            path = fh.name
        out = subprocess.run([EUREKA, "bench", "nnue-v2q-probe-batch",
                              "--model", model, "--batch", path],
                             capture_output=True, text=True, timeout=3600,
                             check=True)
        Path(path).unlink()
        return {json.loads(l)["position_id"]: json.loads(l)["raw_output"]
                for l in out.stdout.splitlines() if l.strip()}

    from tools.s10.train_nnue import material_cp_stm_python
    raw128 = probe(MODEL128, "a")
    raw256 = probe(MODEL256, "b")
    for i, u in enumerate(labeled):
        mat = material_cp_stm_python(u["fen"])
        u["f128"] = clamp(mat + raw128[f"a{i}"] * 1000 // 4096)
        u["f256"] = clamp(mat + raw256[f"b{i}"] * 1000 // 4096)
        u["phase"] = phase_bucket(phase_of(u["fen"]))
        u["mat"] = mat_bucket(mat_abs(u["fen"]))

    # matched control from validation (existing labels)
    records = []
    for shard in sorted(Path(r"data\s10\s10-eval-v2-1m01").glob(
            "part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec["split"] == "validation":
                    records.append(rec)
    labels = {}
    for line in Path(
            r"data\s10\s10-eval-v2-1m01\labels.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[rec["position_id"]] = rec
    # control target cell counts = (phase x mat) distribution of the
    # search-site corpus
    target_cells = defaultdict(int)
    for u in labeled:
        target_cells[(u["phase"], u["mat"])] += 1
    rng = random.Random(2026090405)
    control = []
    pool_by_cell = defaultdict(list)
    for r in records:
        lab = labels.get(r["position_id"])
        if lab is None or lab.get("teacher_cp_stm") is None:
            continue
        pool_by_cell[(phase_bucket(phase_of(r["fen"])),
                      mat_bucket(mat_abs(r["fen"])))].append(r)
    for cell, n in sorted(target_cells.items()):
        pool = pool_by_cell.get(cell, [])
        rng.shuffle(pool)
        control.extend(pool[:n])
    print(f"control: {len(control)} matched positions "
          f"({dict(target_cells)})", flush=True)

    def mae(rows, pred_key, ref_key):
        pairs = [(u[pred_key], u[ref_key]) for u in rows
                 if u.get(pred_key) is not None
                 and u.get(ref_key) is not None]
        if not pairs:
            return None
        return sum(abs(a - b) for a, b in pairs) / len(pairs)

    # control needs F128 too: batch probe the control positions
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(f"c{i}|{r['fen']}"
                           for i, r in enumerate(control)))
        cpath = fh.name
    out = subprocess.run([EUREKA, "bench", "nnue-v2q-probe-batch",
                          "--model", MODEL128, "--batch", cpath],
                         capture_output=True, text=True, timeout=3600,
                         check=True)
    Path(cpath).unlink()
    craw = {json.loads(l)["position_id"]: json.loads(l)["raw_output"]
            for l in out.stdout.splitlines() if l.strip()}
    for i, r in enumerate(control):
        mat = material_cp_stm_python(r["fen"])
        r["f128"] = clamp(mat + craw[f"c{i}"] * 1000 // 4096)
        r["y16"] = clamp(labels[r["position_id"]]["teacher_cp_stm"])
        r["phase"] = phase_bucket(phase_of(r["fen"]))
        r["mat"] = mat_bucket(mat_abs(r["fen"]))

    for u in labeled:
        u["y16c"] = clamp(u["y16"]["cp"]) if u["y16"]["cp"] is not None \
            else None

    m_search = mae(labeled, "f128", "y16c")
    m_control = mae(control, "f128", "y16")
    R = m_search / m_control if m_control else None
    print()
    print(f"MAE(F128, search-sites): {m_search:.1f}")
    print(f"MAE(F128, matched control): {m_control:.1f}")
    print(f"R = {R:.3f}")
    delta = m_search - m_control
    if R >= 1.20 or delta >= 20:
        verdict = "SUPPORTED (distribution shift real)"
    elif R <= 1.10 and delta <= 10:
        verdict = "NOT SUPPORTED (no material shift)"
    else:
        verdict = "AMBIGUOUS (extend to 2048 before concluding)"
    print(f"verdict: {verdict}")

    # main vs qsearch split
    main_rows = [u for u in labeled if u["site"] == "main_static"]
    qs_rows = [u for u in labeled if u["site"] == "qsearch_standpat"]
    print(f"  main_static MAE:    {mae(main_rows, 'f128', 'y16c'):.1f} "
          f"(n={len(main_rows)})")
    print(f"  qsearch MAE:        {mae(qs_rows, 'f128', 'y16c'):.1f} "
          f"(n={len(qs_rows)})")
    for ph in ("high", "mid", "low", "zero"):
        rows = [u for u in labeled if u["phase"] == ph]
        crows = [u for u in control if u["phase"] == ph]
        print(f"  {ph:<6} search {mae(rows, 'f128', 'y16c') or 0:6.1f} | "
              f"control {mae(crows, 'f128', 'y16') or 0:6.1f}")

    out = {
        "schema_version": 1,
        "n_sites": len(labeled),
        "n_control": len(control),
        "mae_search": round(m_search, 1),
        "mae_control": round(m_control, 1),
        "R": round(R, 3),
        "delta": round(delta, 1),
        "verdict": verdict,
        "main_mae": round(mae(main_rows, "f128", "y16c"), 1),
        "qsearch_mae": round(mae(qs_rows, "f128", "y16c"), 1),
        "f256_search_mae": round(mae(labeled, "f256", "y16c"), 1),
        "phase_table": {
            ph: {"search": round(mae([u for u in labeled
                                      if u["phase"] == ph],
                                     "f128", "y16c") or 0, 1),
                 "control": round(mae([u for u in control
                                       if u["phase"] == ph],
                                      "f128", "y16") or 0, 1)}
            for ph in ("high", "mid", "low", "zero")},
        "labeled": labeled,
        "control_pids": [r["position_id"] for r in control],
    }
    Path(r"results\s10\s10-h0-c-shift.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/s10/s10-h0-c-shift.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["collect", "select", "label",
                                            "report"], required=True)
    args = parser.parse_args()
    if args.phase == "collect":
        collect()
    elif args.phase == "select":
        select_1024()
    elif args.phase == "label":
        label()
    elif args.phase == "report":
        report()


if __name__ == "__main__":
    main()
