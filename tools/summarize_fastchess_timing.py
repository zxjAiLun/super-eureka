#!/usr/bin/env python3
"""Summarize per-engine clock and search telemetry from a Fastchess PGN.

This is a diagnostic report only.  It does not adjudicate games, recalculate
Elo, or make a profile decision.  Fastchess PGN comments are parsed by the
existing analyzer so clock fields retain explicit millisecond semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess.pgn

from analyze_fastchess_pgn import parse_comment


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _summary(values: dict[str, list[float]], threshold_ms: float) -> dict[str, Any]:
    depths = values["depth"]
    think = values["think_time_s"]
    nodes = values["nodes"]
    nps = values["nps"]
    clock = values["time_left_ms"]
    latency = values["fastchess_latency_delta_ms"]
    return {
        "search_info_moves": len(depths),
        "depth_p10": percentile(depths, 0.10),
        "depth_p50": percentile(depths, 0.50),
        "depth_p90": percentile(depths, 0.90),
        "think_time_s_p50": percentile(think, 0.50),
        "nodes_p50": percentile(nodes, 0.50),
        "nps_p50": percentile(nps, 0.50),
        "time_left_ms_min": min(clock) if clock else None,
        "time_left_ms_p50": percentile(clock, 0.50),
        "time_pressure_moves": sum(value <= threshold_ms for value in clock),
        "clock_telemetry_moves": len(clock),
        "fastchess_latency_delta_ms_p50": percentile(latency, 0.50),
        "fastchess_latency_delta_ms_p95": percentile(latency, 0.95),
        "fastchess_latency_telemetry_moves": len(latency),
    }


def summarize(pgn_path: Path, time_pressure_threshold_ms: float = 1000.0) -> dict[str, Any]:
    by_engine: dict[str, dict[str, list[float]]] = {}
    games = 0
    parse_errors = 0
    with pgn_path.open("r", encoding="utf-8", errors="replace") as source:
        while game := chess.pgn.read_game(source):
            games += 1
            parse_errors += len(game.errors)
            white = game.headers.get("White", "White")
            black = game.headers.get("Black", "Black")
            for label in {white, black}:
                by_engine.setdefault(
                    label,
                    {
                        "depth": [],
                        "think_time_s": [],
                        "nodes": [],
                        "nps": [],
                        "time_left_ms": [],
                        "fastchess_latency_delta_ms": [],
                    },
                )
            board = game.board()
            for node in game.mainline():
                label = white if board.turn else black
                _score, info = parse_comment(node.comment)
                if info is not None:
                    target = by_engine[label]
                    target["depth"].append(float(info.depth))
                    target["think_time_s"].append(info.time_s)
                    target["nodes"].append(float(info.nodes))
                    target["nps"].append(float(info.nps))
                    if info.time_left_ms is not None:
                        target["time_left_ms"].append(info.time_left_ms)
                    if info.fastchess_latency_delta_ms is not None:
                        target["fastchess_latency_delta_ms"].append(
                            info.fastchess_latency_delta_ms
                        )
                board.push(node.move)

    return {
        "schema_version": 2,
        "pgn": str(pgn_path.resolve()),
        "games": games,
        "parse_errors": parse_errors,
        "time_pressure_threshold_ms": time_pressure_threshold_ms,
        "engines": {
            label: _summary(values, time_pressure_threshold_ms)
            for label, values in sorted(by_engine.items())
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pgn", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--time-pressure-threshold-ms", type=float, default=1000.0)
    return result


def main() -> int:
    args = parser().parse_args()
    report = summarize(args.pgn, args.time_pressure_threshold_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
