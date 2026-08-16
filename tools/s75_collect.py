#!/usr/bin/env python3
"""S7.5-0 forcing-opportunity attribution collector.

OBSERVATION ONLY. Runs production PRODUCTION_PROFILE (`current-final`) with
profiling enabled and records the S7.5 main/qsearch funnels:

- 80 S7 depth-attribution corpus at depths 6 and 7
- 120 R2 tactical corpus at depth 8

No search semantics are changed. Writes results/s7/s75-0-attribution.json
incrementally; --resume skips already-completed rows.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S7_CORPUS = REPO / "tools/data/s7_depth_attribution_corpus.jsonl"
R2_CORPUS = REPO / "data/s7/s74a-r2-tactical-corpus.jsonl"
OUT = REPO / "results/s7/s75-0-attribution.json"

FIELDS = [
    "nodes", "qsearch_nodes", "elapsed_us", "nps", "seldepth", "score",
    "bestmove", "pv",
    "s75_main_nodes", "s75_main_in_check_nodes",
    "s75_main_single_evasion_nodes_raw",
    "s75_main_single_evasion_actionable_depth1",
    "s75_main_single_evasion_actionable_depth2plus",
    "s75_main_single_evasion_depth3plus",
    "s75_main_single_evasion_chain_chain1",
    "s75_main_single_evasion_chain_chain2",
    "s75_main_single_evasion_chain_chain3plus",
    "s75_main_checking_edges_searched", "s75_main_check_child_entered",
    "s75_main_check_child_movegen", "s75_main_check_child_terminal_0",
    "s75_main_check_child_evasions_1", "s75_main_check_child_evasions_2",
    "s75_main_check_child_evasions_3plus",
    "s75_main_depth1_nodes", "s75_main_depth1_in_check",
    "s75_main_depth1_single_evasion",
    "s75_main_depth1_entered_from_checking_edge",
    "s75_q_nodes", "s75_q_in_check_nodes",
    "s75_q_single_evasion_nodes_raw", "s75_q_single_evasion_qply0",
    "s75_q_single_evasion_qply1plus",
    "s75_q_checking_edges_searched", "s75_q_check_child_entered",
    "s75_q_check_child_movegen", "s75_q_check_child_terminal_0",
    "s75_q_check_child_evasions_1", "s75_q_check_child_evasions_2",
    "s75_q_check_child_evasions_3plus",
]


def run_one(engine: Path, fen: str, depth: int, timeout: int) -> dict | None:
    cmd = [str(engine), "bench", "profile", "--profile", "current-final",
           "--depth", str(depth), "--mode", "cold", "--fen", fen]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"timeout": True}
    if proc.returncode != 0:
        return {"error": True, "stderr": proc.stderr[:500]}
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("bench_result ")), None)
    if line is None:
        return {"error": True, "missing_result": True}
    tokens = shlex.split(line)[1:]
    rec: dict = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        key, value = tok.split("=", 1)
        if key in FIELDS:
            rec[key] = value
        elif key.startswith("s75_main_single_evasion_chain:"):
            payload = tok.split(":", 1)[1]
            for part in payload.split(","):
                label, number = part.split("=", 1)
                if label == "chain3p":
                    label = "chain3plus"
                rec[f"s75_main_single_evasion_chain_{label}"] = number
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--r2-timeout", type=int, default=3600)
    args = ap.parse_args()

    engine = args.engine.resolve()
    s7 = [json.loads(l) for l in
          S7_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    r2 = [json.loads(l) for l in
          R2_CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]

    result: dict = {"rows": []}
    done: set = set()
    if OUT.exists():
        result = json.loads(OUT.read_text(encoding="utf-8"))
        done = {(r.get("corpus"), r.get("id") or r.get("i"), r["depth"])
                for r in result.get("rows", []) if r.get("ok")}

    jobs = [(f"s7:{row['id']}", row["fen"], d) for row in s7 for d in (6, 7)]
    jobs += [(f"r2:{i}", row["fen"], 8) for i, row in enumerate(r2)]

    for key, fen, depth in jobs:
        corpus, ident = key.split(":", 1)
        ident_key = int(ident) if corpus == "r2" else ident
        if (corpus, ident_key, depth) in done:
            continue
        rec = run_one(engine, fen, depth, args.r2_timeout)
        row = {"corpus": corpus, "id": ident_key, "depth": depth, "fen": fen,
               "result": rec, "ok": bool(rec) and not rec.get("error")
               and not rec.get("timeout")}
        result["rows"].append(row)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"s75_0 {corpus} {ident} d{depth} ok={row['ok']} "
              f"nodes={rec.get('nodes') if rec else '?'}", flush=True)

    summarize(result, OUT)
    return 0


def summarize(result: dict, out: Path) -> None:
    rows = [r for r in result["rows"] if r.get("ok")]
    by_key: dict = {}
    for r in rows:
        rec = r["result"]
        key = (r["corpus"], r["depth"])
        d = by_key.setdefault(key, {"positions": 0})
        d["positions"] += 1
        for field in FIELDS:
            val = rec.get(field)
            if val is None:
                continue
            # arrays arrive as e.g. chain1=..,chain2=..,chain3p=..
            if field == "s75_main_single_evasion_chain":
                continue
            try:
                num = int(val)
            except (TypeError, ValueError):
                continue
            d[field] = d.get(field, 0) + num
    # Chain aggregates were already parsed into explicit keys.
    summaries = {}
    for (corpus, depth), d in sorted(by_key.items()):
        n = max(1, d.get("positions", 1))
        total_nodes = d.get("nodes", 0)
        q_nodes = d.get("qsearch_nodes", 0)
        main_nodes = d.get("s75_main_nodes", 0)
        raw_se = d.get("s75_main_single_evasion_nodes_raw", 0)
        summaries[f"{corpus}_d{depth}"] = {
            "positions": d["positions"],
            "nodes": total_nodes,
            "qsearch_nodes": q_nodes,
            "main_nodes": main_nodes,
            "main_in_check_nodes": d.get("s75_main_in_check_nodes", 0),
            "main_single_evasion_raw": raw_se,
            "main_single_evasion_depth1": d.get(
                "s75_main_single_evasion_actionable_depth1", 0),
            "main_single_evasion_depth2plus": d.get(
                "s75_main_single_evasion_actionable_depth2plus", 0),
            "main_single_evasion_depth3plus": d.get(
                "s75_main_single_evasion_depth3plus", 0),
            "main_single_evasion_chain1": d.get(
                "s75_main_single_evasion_chain_chain1", 0),
            "main_single_evasion_chain2": d.get(
                "s75_main_single_evasion_chain_chain2", 0),
            "main_single_evasion_chain3plus": d.get(
                "s75_main_single_evasion_chain_chain3plus", 0),
            "main_checking_edges": d.get("s75_main_checking_edges_searched", 0),
            "main_check_child_entered": d.get("s75_main_check_child_entered", 0),
            "main_check_child_movegen": d.get("s75_main_check_child_movegen", 0),
            "main_check_child_evasions_0": d.get(
                "s75_main_check_child_terminal_0", 0),
            "main_check_child_evasions_1": d.get(
                "s75_main_check_child_evasions_1", 0),
            "main_check_child_evasions_2": d.get(
                "s75_main_check_child_evasions_2", 0),
            "main_check_child_evasions_3plus": d.get(
                "s75_main_check_child_evasions_3plus", 0),
            "main_depth1_nodes": d.get("s75_main_depth1_nodes", 0),
            "main_depth1_in_check": d.get("s75_main_depth1_in_check", 0),
            "main_depth1_single_evasion": d.get(
                "s75_main_depth1_single_evasion", 0),
            "main_depth1_from_checking_edge": d.get(
                "s75_main_depth1_entered_from_checking_edge", 0),
            "q_nodes": d.get("s75_q_nodes", 0),
            "q_in_check_nodes": d.get("s75_q_in_check_nodes", 0),
            "q_single_evasion_raw": d.get("s75_q_single_evasion_nodes_raw", 0),
            "q_single_evasion_qply0": d.get("s75_q_single_evasion_qply0", 0),
            "q_single_evasion_qply1plus": d.get(
                "s75_q_single_evasion_qply1plus", 0),
            "q_checking_edges": d.get("s75_q_checking_edges_searched", 0),
            "q_check_child_entered": d.get("s75_q_check_child_entered", 0),
            "q_check_child_evasions_0": d.get("s75_q_check_child_terminal_0", 0),
            "q_check_child_evasions_1": d.get("s75_q_check_child_evasions_1", 0),
            "q_check_child_evasions_2": d.get("s75_q_check_child_evasions_2", 0),
            "q_check_child_evasions_3plus": d.get(
                "s75_q_check_child_evasions_3plus", 0),
            "elapsed_us": d.get("elapsed_us", 0),
        }
    result["summary_by_corpus_depth"] = summaries
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("s75_0 summary_by_corpus_depth: "
          f"{json.dumps(summaries, indent=2)}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"s75_0_error {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
