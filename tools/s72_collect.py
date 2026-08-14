#!/usr/bin/env python3
"""S7.2 move-ordering attribution collector (OBSERVATION ONLY).

Runs `current-final` over the 80-position S7 corpus (depth 6 + 7) and the
30-position S4 corpus (depth 6), sums the profiling-gated s72_* counters,
derives the opportunity-normalized metrics, applies the predeclared decision
rule, and writes results/s7/s72-ordering-attribution.json plus a markdown
summary at results/s7/s72-ordering-attribution.md.

No ordering/search semantic changes are involved; the tree is untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
S4_EPD = REPO / "tools/data/s4_compute_positions.epd"
OUT_JSON = REPO / "results/s7/s72-ordering-attribution.json"
OUT_MD = REPO / "results/s7/s72-ordering-attribution.md"

INT_FIELDS = [
    "nodes", "beta_cutoffs", "moves_searched",
    "s72_cat_tt", "s72_cat_promo", "s72_cat_capture", "s72_cat_k0",
    "s72_cat_k1", "s72_cat_hist_quiet", "s72_cat_other_quiet",
    "s72_nodes_with_quiet", "s72_quiet_available", "s72_quiet_searched",
    "s72_qsr_r0", "s72_qsr_r1", "s72_qsr_r2_3", "s72_qsr_r4_7",
    "s72_qsr_r8_15", "s72_qsr_r16p",
    "s72_qsh_le0", "s72_qsh_1_15", "s72_qsh_16_63", "s72_qsh_64_255",
    "s72_qsh_256p",
    "s72_qcg_0", "s72_qcg_1", "s72_qcg_2_3", "s72_qcg_4_7",
    "s72_qcg_8_15", "s72_qcg_16p",
    "s72_qcr_r0", "s72_qcr_r1", "s72_qcr_r2_3", "s72_qcr_r4_7",
    "s72_qcr_r8_15", "s72_qcr_r16p",
    "s72_qch_le0", "s72_qch_1_15", "s72_qch_16_63", "s72_qch_64_255",
    "s72_qch_256p",
    "s72_k0_present", "s72_k0_searched", "s72_k0_cutoffs",
    "s72_k1_present", "s72_k1_searched", "s72_k1_cutoffs", "s72_k_absent",
    "s72_tt_present", "s72_tt_searched", "s72_tt_cutoffs",
    "s72_tt_first_cutoff", "s72_tt_improves_alpha",
    "s72_cs_1", "s72_cs_2", "s72_cs_3_4", "s72_cs_5_8", "s72_cs_9_16",
    "s72_cs_17p",
    "s72_fail_low_nodes", "s72_fail_low_searched_sum",
    "s72_lmr_q_faillow", "s72_lmr_q_research", "s72_lmr_q_cutoff",
]

DEPTH_FIELDS = [
    "s72_dc_d1", "s72_dc_d2", "s72_dc_d3", "s72_dc_d4_5", "s72_dc_d6_7",
    "s72_dc_d8p",
    "s72_dc_late5_d1", "s72_dc_late5_d2", "s72_dc_late5_d3",
    "s72_dc_late5_d4_5", "s72_dc_late5_d6_7", "s72_dc_late5_d8p",
    "s72_dfl_d1", "s72_dfl_d2", "s72_dfl_d3", "s72_dfl_d4_5",
    "s72_dfl_d6_7", "s72_dfl_d8p",
    "s72_dqs_d1", "s72_dqs_d2", "s72_dqs_d3", "s72_dqs_d4_5",
    "s72_dqs_d6_7", "s72_dqs_d8p",
    "s72_dqc_d1", "s72_dqc_d2", "s72_dqc_d3", "s72_dqc_d4_5",
    "s72_dqc_d6_7", "s72_dqc_d8p",
]


def run(engine: Path, fen: str, depth: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", "current-final",
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("bench_result "):
            toks = dict(x.split("=", 1) for x in line.split() if "=" in x)
            return toks
    return None


def pct(n: float, d: float) -> float | None:
    return round(n * 100.0 / d, 3) if d else None


def hist_stats(counts: list[int], reps: list[float]):
    """Approximate mean/median/p90 from a bucket histogram."""
    total = sum(counts)
    if total == 0:
        return {"mean": None, "median": None, "p90": None}
    weighted = sum(c * r for c, r in zip(counts, reps))
    mean = weighted / total

    def quantile(q: float) -> float:
        acc = 0
        for c, r in zip(counts, reps):
            acc += c
            if acc >= q * total:
                return r
        return reps[-1]

    return {"mean": round(mean, 3), "median": quantile(0.5), "p90": quantile(0.9)}


def summarize(tot: dict) -> dict:
    cutoffs = tot["beta_cutoffs"]
    cs = [tot["s72_cs_1"], tot["s72_cs_2"], tot["s72_cs_3_4"],
          tot["s72_cs_5_8"], tot["s72_cs_9_16"], tot["s72_cs_17p"]]
    cs_total = sum(cs)
    quiet_cutoffs = sum(tot[f"s72_qcg_{k}"] for k in
                        ("0", "1", "2_3", "4_7", "8_15", "16p"))
    qsh = [tot[f"s72_qsh_{k}"] for k in
           ("le0", "1_15", "16_63", "64_255", "256p")]
    qch = [tot[f"s72_qch_{k}"] for k in
           ("le0", "1_15", "16_63", "64_255", "256p")]
    cats = {
        "tt_hash": (tot["s72_cat_tt"], tot["s72_tt_searched"]),
        "promotion": (tot["s72_cat_promo"], None),
        "capture": (tot["s72_cat_capture"], None),
        "killer0": (tot["s72_cat_k0"], tot["s72_k0_searched"]),
        "killer1": (tot["s72_cat_k1"], tot["s72_k1_searched"]),
        "history_quiet": (tot["s72_cat_hist_quiet"], sum(qsh[1:])),
        "other_quiet": (tot["s72_cat_other_quiet"], qsh[0]),
    }
    late5 = sum(cs[3:])
    late9 = sum(cs[4:])
    return {
        "beta_cutoffs": cutoffs,
        "cutoff_nodes_with_histogram": cs_total,
        "first_move_cutoff_pct": pct(cs[0], cs_total),
        "moves_before_cutoff": hist_stats(cs, [1, 2, 3.5, 6.5, 12.5, 20]),
        "late_cutoff_ge5_pct": pct(late5, cs_total),
        "late_cutoff_ge9_pct": pct(late9, cs_total),
        "cutoff_category_share_pct": {
            k: pct(c, cs_total) for k, (c, _) in cats.items()},
        "cutoff_category_success_rate_pct": {
            k: pct(c, s) if s is not None else None
            for k, (c, s) in cats.items()},
        "quiet_opportunity": {
            "nodes_with_quiet_moves": tot["s72_nodes_with_quiet"],
            "quiet_available": tot["s72_quiet_available"],
            "quiet_searched": tot["s72_quiet_searched"],
            "quiet_cutoffs": quiet_cutoffs,
            "quiet_cutoff_rate_pct": pct(quiet_cutoffs,
                                         tot["s72_quiet_searched"]),
        },
        "quiet_cutoff_global_idx_pct": {
            k: pct(tot[f"s72_qcg_{k}"], quiet_cutoffs) for k in
            ("0", "1", "2_3", "4_7", "8_15", "16p")},
        "quiet_cutoff_quiet_rank_pct": {
            k: pct(tot[f"s72_qcr_{k}"], quiet_cutoffs) for k in
            ("r0", "r1", "r2_3", "r4_7", "r8_15", "r16p")},
        "history_bucket_cutoff_rate_pct": {
            k: pct(c, s) for k, c, s in zip(
                ("le0", "1_15", "16_63", "64_255", "256p"), qch, qsh)},
        "killers": {
            "k0_present": tot["s72_k0_present"],
            "k0_searched": tot["s72_k0_searched"],
            "k0_cutoff_rate_pct": pct(tot["s72_k0_cutoffs"],
                                      tot["s72_k0_searched"]),
            "k1_present": tot["s72_k1_present"],
            "k1_searched": tot["s72_k1_searched"],
            "k1_cutoff_rate_pct": pct(tot["s72_k1_cutoffs"],
                                      tot["s72_k1_searched"]),
            "absent_or_illegal": tot["s72_k_absent"],
        },
        "tt_hash": {
            "present": tot["s72_tt_present"],
            "searched": tot["s72_tt_searched"],
            "cutoffs": tot["s72_tt_cutoffs"],
            "cutoff_rate_pct": pct(tot["s72_tt_cutoffs"],
                                   tot["s72_tt_searched"]),
            "first_move_cutoffs": tot["s72_tt_first_cutoff"],
            "improves_alpha_without_cutoff": tot["s72_tt_improves_alpha"],
        },
        "fail_low": {
            "all_moves_no_cutoff_nodes": tot["s72_fail_low_nodes"],
            "mean_searched": round(
                tot["s72_fail_low_searched_sum"] / tot["s72_fail_low_nodes"],
                3) if tot["s72_fail_low_nodes"] else None,
        },
        "lmr_quiet": {
            "reduced_fail_low": tot["s72_lmr_q_faillow"],
            "reduced_research": tot["s72_lmr_q_research"],
            "reduced_eventual_cutoff": tot["s72_lmr_q_cutoff"],
        },
        "depth_split": {
            "cutoffs": [tot[f"s72_dc_d{k}"] for k in
                        ("1", "2", "3", "4_5", "6_7", "8p")],
            "cutoffs_late5": [tot[f"s72_dc_late5_d{k}"] for k in
                              ("1", "2", "3", "4_5", "6_7", "8p")],
            "fail_low": [tot[f"s72_dfl_d{k}"] for k in
                         ("1", "2", "3", "4_5", "6_7", "8p")],
            "quiet_searched": [tot[f"s72_dqs_d{k}"] for k in
                               ("1", "2", "3", "4_5", "6_7", "8p")],
            "quiet_cutoffs": [tot[f"s72_dqc_d{k}"] for k in
                              ("1", "2", "3", "4_5", "6_7", "8p")],
        },
    }
    return summary


def classify(s: dict) -> str:
    """Section M decision rule (opportunity-normalized, no external 90%)."""
    late5 = s["late_cutoff_ge5_pct"] or 0.0
    first = s["first_move_cutoff_pct"] or 0.0
    hist = s["history_bucket_cutoff_rate_pct"]
    h_hi = hist.get("256p") or 0.0
    h_lo = hist.get("le0") or 0.0
    gradient = h_hi - h_lo
    if late5 >= 20.0 and gradient >= 5.0:
        return "ORDERING_MAJOR"
    if late5 >= 8.0:
        return "ORDERING_MODERATE"
    if first >= 70.0:
        return "ORDERING_NOT_PRIMARY"
    return "ORDERING_MODERATE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path,
                    default=REPO / "target/release/eureka.exe")
    ap.add_argument("--depths", default="6,7")
    args = ap.parse_args(sys.argv[1:])

    engine = args.engine.resolve()
    depths = [int(d) for d in args.depths.split(",")]
    rows_s7 = [json.loads(l) for l in
               CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    s4 = [l.split(";", 1)[0].strip() for l in
          S4_EPD.read_text(encoding="utf-8").splitlines()
          if l.strip() and not l.startswith("#")]

    tot = {k: 0 for k in INT_FIELDS + DEPTH_FIELDS}
    per_run = []

    def collect(fen: str, depth: int, tag: str) -> None:
        toks = run(engine, fen, depth)
        if toks is None:
            print(f"s72_collect {tag} d{depth} FAILED", flush=True)
            return
        for k in tot:
            tot[k] += int(toks.get(k, "0") or 0)
        per_run.append({"tag": tag, "depth": depth, "nodes": toks["nodes"],
                        "score": toks["score"], "bestmove": toks["bestmove"],
                        "pv": toks["pv"]})
        print(f"s72_collect {tag} d{depth} nodes={toks['nodes']}", flush=True)

    for pos in rows_s7:
        for d in depths:
            collect(pos["fen"], d, pos["id"])
    for i, fen in enumerate(s4):
        collect(fen, 6, f"s4_{i:02d}")

    summary = summarize(tot)
    summary["classification"] = classify(summary)
    out = {"totals": tot, "summary": summary, "runs": per_run,
           "corpus_sha":
               "8786ffca6c8e6277b711c990bf9788d88eaedbb0b4b894f85fc2b18de62d5b1b"}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    s = summary
    lines = [
        "# S7.2 Move Ordering Attribution (OBSERVATION ONLY)",
        "",
        "- profile: current-final, cold 16MB TT, threads 1",
        f"- corpus: 80 S7 (d{','.join(map(str, depths))}) + 30 S4 (d6)",
        f"- classification: **{s['classification']}**",
        "",
        "## Headline",
        f"- first-move cutoff: {s['first_move_cutoff_pct']}%",
        f"- moves before cutoff: mean {s['moves_before_cutoff']['mean']} "
        f"median {s['moves_before_cutoff']['median']} "
        f"p90 {s['moves_before_cutoff']['p90']}",
        f"- late cutoff >=5 searched: {s['late_cutoff_ge5_pct']}%",
        f"- late cutoff >=9 searched: {s['late_cutoff_ge9_pct']}%",
        f"- fail-low all-moves nodes: {s['fail_low']['all_moves_no_cutoff_nodes']}"
        f" (mean searched {s['fail_low']['mean_searched']})",
        "",
        "## Cutoff category share / success rate (%)",
    ]
    for k in ("tt_hash", "promotion", "capture", "killer0", "killer1",
              "history_quiet", "other_quiet"):
        lines.append(f"- {k}: share {s['cutoff_category_share_pct'][k]}"
                     f" success {s['cutoff_category_success_rate_pct'][k]}")
    q = s["quiet_opportunity"]
    lines += [
        "",
        "## Quiet opportunity",
        f"- quiet searched {q['quiet_searched']} / available "
        f"{q['quiet_available']}, cutoffs {q['quiet_cutoffs']} "
        f"(rate {q['quiet_cutoff_rate_pct']}%)",
        f"- quiet cutoff global-index histogram: "
        f"{s['quiet_cutoff_global_idx_pct']}",
        f"- quiet cutoff quiet-rank histogram: "
        f"{s['quiet_cutoff_quiet_rank_pct']}",
        "",
        "## History buckets: cutoff rate (%)",
    ]
    for k, v in s["history_bucket_cutoff_rate_pct"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Killers / TT",
        f"- killer0 searched {s['killers']['k0_searched']} "
        f"cutoff rate {s['killers']['k0_cutoff_rate_pct']}%",
        f"- killer1 searched {s['killers']['k1_searched']} "
        f"cutoff rate {s['killers']['k1_cutoff_rate_pct']}%",
        f"- TT hash searched {s['tt_hash']['searched']} "
        f"cutoff rate {s['tt_hash']['cutoff_rate_pct']}%",
        "",
        "## LMR quiet interaction",
        f"- {s['lmr_quiet']}",
        "",
        "## Depth split (cutoffs / late>=5 / fail-low / quiet searched / "
        "quiet cutoffs)",
    ]
    ds = s["depth_split"]
    for i, k in enumerate(("d1", "d2", "d3", "d4_5", "d6_7", "d8p")):
        lines.append(f"- {k}: {ds['cutoffs'][i]} / {ds['cutoffs_late5'][i]}"
                     f" / {ds['fail_low'][i]} / {ds['quiet_searched'][i]}"
                     f" / {ds['quiet_cutoffs'][i]}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"s72_collect done classification={s['classification']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s72_collect_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

