"""S10-H0-A1: exhaustive audit of the existing 1k corpus.

NO new Stockfish training labels. One new SF pass only: static eval
(depth 1, nodes 1) on the SAME 1000 FENs that already have:
  Y16  = E2 teacher labels (nodes 16384, from labels.jsonl)
  Y20  = depth-20 reference (from the previous sanity run)
  F128 = Eureka FT128-v3 evals (probe)
  F256 = Eureka FT256-v3 evals (probe)

All scores unified: STM perspective, ±2000 clamp, mate positions
handled separately (excluded from CP metrics, counted in class stats).

Outputs:
  * full 7-column metric matrix (Y16<->Y20, SFSE<->Y16, SFSE<->Y20,
    F128<->Y16, F128<->Y20, F256<->Y16, F256<->Y20)
  * Y16 vs Y20 bestmove/WDL/class agreement
  * deep-correction correlation: corr(F - Y16, Y20 - Y16)
  * affine calibration: fit Y20 ~ a*F + b on 500, test on 500
  * stratified breakdowns: phase / material imbalance / |Y20-Y16| bucket
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.s10.train_nnue import material_cp_stm_python  # noqa: E402

SF = (r"C:\Users\81489\AppData\Local\Temp\opencode\sf18-win\stockfish"
      r"\stockfish-windows-x86-64-avx2.exe")
EUREKA = (r"E:\AUbuntuProject\project\chessenginedemo\target\release"
          r"\eureka.exe")
F128 = (r"data\s10\e3\scale-1m-win\seed-20260820"
        r"\nnue-v2-q01-material-v3twin.bin")
F256 = (r"data\s10\g1\ft256-1m\seed-20260819\nnue-v2-q01-material-256.bin")
CLIP = 2000

PHASE_WEIGHTS = {"n": 1, "b": 1, "r": 2, "q": 4}


def phase_of(fen):
    import chess
    b = chess.Board(fen)
    ph = 0
    for sq, pc in b.piece_map().items():
        ph += PHASE_WEIGHTS.get(pc.symbol().lower(), 0)
    return ph


def mat_abs(fen):
    import chess
    b = chess.Board(fen)
    w = t = 0
    for sq, pc in b.piece_map().items():
        v = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900}.get(
            pc.symbol().lower(), 0)
        if pc.color == chess.WHITE:
            w += v
        else:
            t += v
    return abs(w - t)


def sf_static(fen):
    """SF18 static NNUE eval: 'go depth 1', take the FIRST pv info score."""
    p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush()
    while p.stdout.readline().strip() != "uciok":
        pass
    p.stdin.write("setoption name Threads value 1\n")
    p.stdin.write("setoption name Hash value 64\n")
    p.stdin.write("setoption name MultiPV value 1\n")
    p.stdin.write(f"position fen {fen}\n")
    p.stdin.write("go depth 1\n"); p.stdin.flush()
    first = None
    while True:
        line = p.stdout.readline()
        if not line or line.startswith("bestmove"):
            break
        if line.startswith("info depth 1") and " pv " in line:
            m = re.search(r"score cp (-?\d+)", line)
            mm = re.search(r"score mate (-?\d+)", line)
            if m:
                first = int(m.group(1))
            elif mm:
                first = ("mate", int(mm.group(1)))
    p.stdin.write("quit\n"); p.stdin.flush()
    p.wait(timeout=15)
    return first


def clamp(v):
    return max(-CLIP, min(CLIP, v))


def main():
    # ---- load the same 1k corpus ----
    prev = json.load(open(
        r"C:\Users\81489\AppData\Local\Temp\opencode\teacher-noise-check.json"
    ))["rows"]
    labels = {}
    with open(r"data\s10\s10-eval-v2-1m01\labels.jsonl",
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                labels[rec["position_id"]] = rec
    print(f"corpus: {len(prev)} positions", flush=True)

    # ---- SF static pass (the only new SF work) ----
    for i, r in enumerate(prev):
        r["sfse"] = sf_static(r["fen"])
        if (i + 1) % 100 == 0:
            print(f"  sfse {i + 1}/{len(prev)}", flush=True)

    # ---- Eureka evals (batch) ----
    def probe(model, tag):
        with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False,
                encoding="utf-8") as fh:
            fh.write("\n".join(
                f"{tag}{i}|{r['fen']}" for i, r in enumerate(prev)))
            path = fh.name
        out = subprocess.run(
            [EUREKA, "bench", "nnue-v2q-probe-batch", "--model", model,
             "--batch", path],
            capture_output=True, text=True, timeout=1800, check=True)
        Path(path).unlink()
        return {json.loads(l)["position_id"]: json.loads(l)["raw_output"]
                for l in out.stdout.splitlines() if l.strip()}

    raw128 = probe(F128, "a")
    raw256 = probe(F256, "b")

    # ---- unify ----
    rows = []
    for i, r in enumerate(prev):
        pid = r["pid"]
        lab = labels[pid]
        y16 = lab["teacher_cp_stm"]      # already clamped at label time
        y16_mate = lab["teacher_mate"] is not None
        y20 = r["t20"] if not isinstance(r["t20"], list) else None
        y20_mate = isinstance(r["t20"], list)
        sfse = r["sfse"] if not isinstance(r["sfse"], list) else None
        sfse_mate = isinstance(r["sfse"], list)
        mat = material_cp_stm_python(r["fen"])
        f128 = mat + raw128[f"a{i}"] * 1000 // 4096
        f256 = mat + raw256[f"b{i}"] * 1000 // 4096
        rows.append({
            "fen": r["fen"], "pid": pid,
            "y16": None if y16_mate else clamp(y16), "y16_mate": y16_mate,
            "y20": None if y20_mate else (clamp(y20) if y20 is not None else None),
            "y20_mate": y20_mate,
            "sfse": None if sfse_mate else (clamp(sfse) if sfse is not None else None),
            "sfse_mate": sfse_mate,
            "f128": clamp(f128), "f256": clamp(f256),
            "phase": phase_of(r["fen"]),
            "mat": mat_abs(r["fen"]),
            "bm16": lab["teacher_bestmove"],
            "wdl16": lab["teacher_wdl_stm"],
        })

    def pct(v, p):
        s = sorted(v)
        return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]

    def pearson(x, y):
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        vx = sum((a - mx) ** 2 for a in x)
        vy = sum((b - my) ** 2 for b in y)
        return cov / (vx * vy) ** 0.5 if vx and vy else None

    def spearman(x, y):
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        return pearson(ranks(x), ranks(y))

    def pair_metrics(pa, pb):
        """pa, pb: field names. Only rows where BOTH are plain cp."""
        xs, ys = [], []
        for r in rows:
            a, b = r[pa], r[pb]
            if a is None or b is None:
                continue
            xs.append(a); ys.append(b)
        diffs = [b - a for a, b in zip(xs, ys)]
        absd = [abs(d) for d in diffs]
        sign_agree = sum(
            1 for a, b in zip(xs, ys) if abs(a) >= 50 and abs(b) >= 50
            and (a > 0) == (b > 0))
        n_ge50 = sum(1 for a, b in zip(xs, ys)
                     if abs(a) >= 50 and abs(b) >= 50)
        return {
            "n": len(xs),
            "mae": round(sum(absd) / len(absd), 1),
            "median": pct(absd, 50),
            "p90": pct(absd, 90),
            "p95": pct(absd, 95),
            "mean_signed": round(sum(diffs) / len(diffs), 1),
            "pearson": round(pearson(xs, ys), 4),
            "spearman": round(spearman(xs, ys), 4),
            "sign_agree_ge50": (round(sign_agree / n_ge50, 4)
                                if n_ge50 else None),
        }

    # ---- the 7-column matrix ----
    matrix = {}
    for pa, pb in [("y16", "y20"), ("sfse", "y16"), ("sfse", "y20"),
                   ("f128", "y16"), ("f128", "y20"),
                   ("f256", "y16"), ("f256", "y20")]:
        matrix[f"{pa}<->{pb}"] = pair_metrics(pa, pb)
    print(json.dumps(matrix, indent=1))

    # ---- mate / class transitions (Y16 vs Y20) ----
    both_cp = sum(1 for r in rows if r["y16"] is not None
                  and r["y20"] is not None)
    y16m_y20cp = sum(1 for r in rows if r["y16_mate"] and r["y20"] is not None)
    y16cp_y20m = sum(1 for r in rows if r["y16"] is not None and r["y20_mate"])
    both_mate = sum(1 for r in rows if r["y16_mate"] and r["y20_mate"])
    class_stats = {
        "both_cp": both_cp, "y16_mate_y20_cp": y16m_y20cp,
        "y16_cp_y20_mate": y16cp_y20m, "both_mate": both_mate,
        "n": len(rows),
    }
    print("class:", class_stats)

    # ---- deep correction correlation ----
    dc, ne128, ne256 = [], [], []
    for r in rows:
        if r["y16"] is None or r["y20"] is None:
            continue
        dc.append(r["y20"] - r["y16"])
        ne128.append(r["f128"] - r["y16"])
        ne256.append(r["f256"] - r["y16"])
    corr128 = pearson(ne128, dc)
    corr256 = pearson(ne256, dc)
    print(f"deep-correction corr (F128 err, deep corr): {corr128:.4f}")
    print(f"deep-correction corr (F256 err, deep corr): {corr256:.4f}")

    # ---- affine calibration: fit Y20 ~ a*F + b on first half ----
    def affine(fit_pairs, test_pairs):
        xs = [a for a, b in fit_pairs]
        ys = [b for a, b in fit_pairs]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        a = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
        b = my - a * mx
        mae_raw = sum(abs(b0 - (a * x0 + b))
                      for x0, b0 in test_pairs) / len(test_pairs)
        mae_base = sum(abs(b0 - x0) for x0, b0 in test_pairs) / len(test_pairs)
        return {"a": round(a, 4), "b": round(b, 1),
                "test_mae_raw": round(mae_base, 1),
                "test_mae_calibrated": round(mae_raw, 1)}

    pairs128 = [(r["f128"], r["y20"]) for r in rows
                if r["f128"] is not None and r["y20"] is not None]
    cal128 = affine(pairs128[:len(pairs128) // 2],
                    pairs128[len(pairs128) // 2:])
    print("affine F128->Y20:", cal128)

    # ---- stratified: phase / material / |Y20-Y16| buckets ----
    def bucket_phase(p):
        return "high" if p >= 18 else "mid" if p >= 8 else \
               "low" if p >= 1 else "zero"

    def bucket_mat(m):
        return "<=50" if m <= 50 else "51-250" if m <= 250 else \
               "251-500" if m <= 500 else ">500"

    strata = {}
    for dim, fn in (("phase", bucket_phase), ("material", bucket_mat)):
        groups = {}
        for r in rows:
            if r["y16"] is None or r["y20"] is None:
                continue
            k = fn(r[dim] if dim == "phase" else r["mat"])
            groups.setdefault(k, []).append(r)
        strata[dim] = {}
        for k, g in sorted(groups.items()):
            strata[dim][k] = {
                "n": len(g),
                "y16_y20_mae": round(sum(
                    abs(r["y20"] - r["y16"]) for r in g) / len(g), 1),
                "f128_y16_mae": round(sum(
                    abs(r["f128"] - r["y16"]) for r in g) / len(g), 1),
                "f128_y20_mae": round(sum(
                    abs(r["f128"] - r["y20"]) for r in g) / len(g), 1),
                "sfse_y20_mae": round(sum(
                    abs(r["sfse"] - r["y20"]) for r in g
                    if r["sfse"] is not None) / max(
                    sum(1 for r in g if r["sfse"] is not None), 1), 1),
            }
    print("strata:", json.dumps(strata, indent=1))

    # |Y20-Y16| buckets: does the network do worse where the teacher drifts?
    drift_groups = {"0-50": [], "51-150": [], "151-300": [], ">300": []}
    for r in rows:
        if r["y16"] is None or r["y20"] is None:
            continue
        d = abs(r["y20"] - r["y16"])
        k = "0-50" if d <= 50 else "51-150" if d <= 150 else \
            "151-300" if d <= 300 else ">300"
        drift_groups[k].append(r)
    drift = {k: {"n": len(g),
                 "f128_y16_mae": round(sum(
                     abs(r["f128"] - r["y16"]) for r in g) / len(g), 1)
                 if g else None}
             for k, g in drift_groups.items()}
    print("drift buckets:", json.dumps(drift, indent=1))

    out = {
        "schema_version": 1,
        "corpus": {"n": len(rows), "seed": 2026090401},
        "matrix": matrix,
        "class_stats": class_stats,
        "deep_correction_corr": {"f128": round(corr128, 4),
                                 "f256": round(corr256, 4)},
        "affine_f128_to_y20": cal128,
        "strata": strata,
        "drift_buckets": drift,
    }
    Path(r"results\s10").mkdir(exist_ok=True)
    Path(r"results\s10\s10-h0-a1-audit.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/s10/s10-h0-a1-audit.json")


if __name__ == "__main__":
    main()
