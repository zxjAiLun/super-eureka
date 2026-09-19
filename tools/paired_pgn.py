"""Shared fail-closed PGN pair parser and Cutechess summary reader.

Used by S15, S17, and S18 runners to eliminate redundant parsers,
enforce strict FEN-based pair grouping (immune to cutechess game output order),
and ensure incomplete or failed matches fail closed without writing reports.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import re

import chess
import chess.pgn


def read_cutechess_elo(log: Path) -> dict:
    matches = re.findall(
        r"^Elo difference: ([-+\d.]+) \+/- ([\d.]+), LOS: ([\d.]+) %.*$",
        log.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if not matches:
        raise ValueError("missing cutechess Elo summary; do not invent a CI")
    elo, margin, los = map(float, matches[-1])
    return {
        "source": "cutechess-cli summary (not a paired-CI recomputation)",
        "elo": elo,
        "reported_ci95_half_width": margin,
        "los_percent": los,
    }


def paired_elo_ci(pairs: list[dict], points: float, n_games: int) -> tuple[float, float | None]:
    """Calculate Elo difference and 95% CI from FEN-grouped pair outcomes."""
    p = points / n_games if n_games else 0.0
    if 0.0 < p < 1.0:
        elo = -400.0 * math.log10(1.0 / p - 1.0)
    else:
        elo = float("inf") if p >= 1.0 else float("-inf")
    npairs = len(pairs)
    if npairs > 1 and 0.0 < p < 1.0:
        pair_norms = [
            sum(g["candidate_points"] for g in pair["games"]) / 2.0
            for pair in pairs
        ]
        mean = sum(pair_norms) / npairs
        var = sum((x - mean) ** 2 for x in pair_norms) / (npairs - 1)
        se_p = math.sqrt(var / npairs)
        d_elo_dp = 400.0 / (math.log(10) * p * (1.0 - p))
        elo_ci95 = 1.96 * se_p * d_elo_dp
        return round(elo, 1), round(elo_ci95, 1)
    return round(elo, 1) if not math.isinf(elo) else elo, None


def parse_pgn_and_stats(
    pgn_path: Path,
    arm_a: str,
    arm_b: str,
    expected_fens: list[str] | None = None,
) -> dict:
    """Validate complete color-swapped pairs; attribute times using actual turn.

    No adjacency assumption: cutechess can finish games out of launch order.
    Timings are rounded PGN move times, not CPU time or total search-node work.
    """
    groups = defaultdict(list)
    times = {arm_a: [], arm_b: []}
    moves = {arm_a: 0, arm_b: 0}
    scores = []
    rounds = set()
    with pgn_path.open(encoding="utf-8") as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            h = game.headers
            if game.errors or set((h.get("White"), h.get("Black"))) != {arm_a, arm_b}:
                raise ValueError("illegal game or unexpected player")
            if h.get("Result") not in ("1-0", "0-1", "1/2-1/2") or "FEN" not in h:
                raise ValueError("unfinished game or missing opening FEN")
            if h.get("Round") in rounds:
                raise ValueError("duplicate game round")
            rounds.add(h.get("Round"))
            score = (
                0.5
                if h["Result"] == "1/2-1/2"
                else float((h["Result"] == "1-0") == (h["White"] == arm_a))
            )
            scores.append(score)
            groups[h["FEN"]].append({
                "round": h["Round"],
                "candidate_color": "white" if h["White"] == arm_a else "black",
                "result": h["Result"],
                "candidate_points": score,
            })
            board = game.board()
            for node in game.mainline():
                engine = h["White"] if board.turn == chess.WHITE else h["Black"]
                moves[engine] += 1
                match = re.search(r"(?:^|\s)(\d+(?:\.\d+)?)s(?:\s|$)", node.comment)
                if match:
                    times[engine].append(float(match[1]))
                board.push(node.move)
    if not scores:
        raise ValueError("empty PGN")
    # cutechess format=epd consumes the first four fields and resets counters.
    expected = (
        {" ".join(f.split()[:4]) + " 0 1" for f in expected_fens}
        if expected_fens is not None
        else None
    )
    if expected is not None and (
        set(groups) != expected
        or len(scores) != 2 * len(expected_fens)
        or len(expected) != len(expected_fens)
    ):
        raise ValueError("opening set or game count does not match protocol")
    penta = [0] * 5
    pairs = []
    for fen, pair in sorted(groups.items()):
        if len(pair) != 2 or {g["candidate_color"] for g in pair} != {"white", "black"}:
            raise ValueError("opening is not a complete color-swapped pair")
        penta[round(sum(g["candidate_points"] for g in pair) * 2)] += 1
        pairs.append({"fen": fen, "games": sorted(pair, key=lambda g: g["candidate_color"])})
    points = sum(scores)
    return {
        "games": len(scores),
        "candidate_W": scores.count(1.0),
        "draws": scores.count(0.5),
        "candidate_L": scores.count(0.0),
        "score_points": f"{points:g} / {len(scores)}",
        "score_percent": round(points / len(scores) * 100, 2),
        "pentanomial": {
            "counts": penta,
            "labels": ["LL", "LD", "DD/WL", "WD", "WW"],
            "pairs": len(pairs),
        },
        "move_time": {
            name: {
                "moves": moves[name],
                "timed_moves": len(ts),
                "total_seconds": round(sum(ts), 2),
                "avg_move_seconds": round(sum(ts) / len(ts), 6) if ts else None,
            }
            for name, ts in times.items()
        },
        "pair_evidence": pairs,
    }
