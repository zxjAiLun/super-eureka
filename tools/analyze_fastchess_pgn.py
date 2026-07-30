"""Postmortem analysis for Fastchess PGN engine-info comments.

This tool is deliberately a report generator.  It never changes a PGN, an
engine profile, or a tournament decision.  Fastchess comments are interpreted
using the convention used by this project: ``[%eval ...]`` is from the mover's
point of view, while the fallback score before ``/depth`` is also associated
with the move that just completed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import chess
import chess.pgn


SCORE_TOKEN = r"(?:[+-]?(?:M\d+|\d+(?:\.\d+)?)|#-?\d+)"
EVAL_RE = re.compile(rf"\[%eval\s+(?P<score>{SCORE_TOKEN})\]")
SEARCH_RE = re.compile(
    rf"(?P<score>{SCORE_TOKEN})/(?P<depth>\d+)\s+"
    r"(?P<time>\d+(?:\.\d+)?)s,.*?n=(?P<nodes>\d+)"
    r".*?sd=(?P<seldepth>\d+)\s*,\s*nps=(?P<nps>\d+)"
    r"\s*,\s*hashfull=(?P<hashfull>\d+)\s*,\s*pv=\"(?P<pv>[^\"]*)\""
)
# Fastchess formats these diagnostics with formatTime(): seconds with an
# explicit ``s`` suffix. Keep the suffix required so unitless values are not
# silently interpreted in the wrong unit.
TIMELEFT_RE = re.compile(r"\b(?:tl|timeleft)=(?P<value>-?\d+(?:\.\d+)?)s\b")
LATENCY_RE = re.compile(r"\b(?:lat|latency)=(?P<value>-?\d+(?:\.\d+)?)s\b")

MATE_CP = 100_000


@dataclass(frozen=True)
class ParsedScore:
    raw: str
    cp: Optional[int]
    mate: Optional[int]

    @property
    def comparable_cp(self) -> Optional[int]:
        if self.mate is None:
            return self.cp
        magnitude = MATE_CP - abs(self.mate)
        return magnitude if self.mate > 0 else -magnitude


@dataclass(frozen=True)
class SearchInfo:
    score: ParsedScore
    depth: int
    time_s: float
    nodes: int
    seldepth: int
    nps: int
    hashfull: int
    pv: list[str]
    time_left_ms: Optional[float]
    # Fastchess's value is elapsed wall time minus the engine-reported search
    # time. It is not pure IPC latency, so keep the source and unit explicit.
    fastchess_latency_delta_ms: Optional[float]


def initial_time_ms_from_time_control(time_control: Optional[str]) -> Optional[float]:
    """Parse Fastchess ``minutes+increment`` or ``mm:ss+increment`` TC text."""
    if not time_control or "+" not in time_control:
        return None
    base, _increment = time_control.split("+", 1)
    try:
        if ":" in base:
            minutes, seconds = base.split(":", 1)
            return (float(minutes) * 60.0 + float(seconds)) * 1000.0
        return float(base) * 1000.0
    except ValueError:
        return None


def parse_score(raw: str) -> ParsedScore:
    token = raw.strip()
    if token.startswith("#"):
        mate = int(token[1:] or "1")
        return ParsedScore(raw=token, cp=None, mate=mate)
    if "M" in token:
        sign = -1 if token.startswith("-") else 1
        mate = int(token.lstrip("+-")[1:] or "1")
        return ParsedScore(raw=token, cp=None, mate=sign * mate)
    return ParsedScore(raw=token, cp=round(float(token) * 100), mate=None)


def parse_comment(comment: str) -> tuple[Optional[ParsedScore], Optional[SearchInfo]]:
    eval_match = EVAL_RE.search(comment)
    search_match = SEARCH_RE.search(comment)
    score: Optional[ParsedScore] = None
    if eval_match:
        score = parse_score(eval_match.group("score"))
    elif search_match:
        score = parse_score(search_match.group("score"))
    if not search_match:
        return score, None
    info = SearchInfo(
        score=parse_score(search_match.group("score")),
        depth=int(search_match.group("depth")),
        time_s=float(search_match.group("time")),
        nodes=int(search_match.group("nodes")),
        seldepth=int(search_match.group("seldepth")),
        nps=int(search_match.group("nps")),
        hashfull=int(search_match.group("hashfull")),
        pv=[move for move in search_match.group("pv").split() if move],
        time_left_ms=(
            float(match.group("value")) * 1000.0
            if (match := TIMELEFT_RE.search(comment))
            else None
        ),
        fastchess_latency_delta_ms=(
            float(match.group("value")) * 1000.0
            if (match := LATENCY_RE.search(comment))
            else None
        ),
    )
    return score or info.score, info


def _passed_pawns(board: chess.Board, color: chess.Color) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    enemy = not color
    for square in board.pieces(chess.PAWN, color):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        ahead = range(rank_index + 1, 8) if color == chess.WHITE else range(rank_index - 1, -1, -1)
        blocked = False
        for rank in ahead:
            for file_offset in (-1, 0, 1):
                file = file_index + file_offset
                if 0 <= file < 8 and board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, enemy):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            distance = 7 - rank_index if color == chess.WHITE else rank_index
            result.append(
                {
                    "square": chess.square_name(square),
                    "color": "white" if color == chess.WHITE else "black",
                    "promotion_distance": distance,
                }
            )
    return sorted(result, key=lambda pawn: pawn["square"])


def passed_pawns(board: chess.Board) -> list[dict[str, Any]]:
    return _passed_pawns(board, chess.WHITE) + _passed_pawns(board, chess.BLACK)


def _min_promotion_distance(pawns: Iterable[dict[str, Any]]) -> Optional[int]:
    distances = [int(pawn["promotion_distance"]) for pawn in pawns]
    return min(distances) if distances else None


def _white_pov(score: Optional[ParsedScore], mover: chess.Color) -> Optional[int]:
    if score is None or score.comparable_cp is None:
        return None
    return score.comparable_cp if mover == chess.WHITE else -score.comparable_cp


def _display_score(score: Optional[ParsedScore]) -> Optional[str]:
    return score.raw if score is not None else None


def analyze_game(
    game: chess.pgn.Game,
    game_index: int,
    min_loss_cp: int,
    max_depth: int,
    time_threshold_s: float,
    time_left_threshold_ms: float = 1000.0,
    time_left_ratio_threshold: float = 0.05,
    initial_time_ms: Optional[float] = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    counts = {"moves": 0, "moves_with_eval": 0, "search_infos": 0}
    observations: list[dict[str, Any]] = []
    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        before = board.copy()
        mover = before.turn
        san = before.san(node.move)
        board.push(node.move)
        score, info = parse_comment(node.comment)
        counts["moves"] += 1
        if score is not None:
            counts["moves_with_eval"] += 1
        if info is not None:
            counts["search_infos"] += 1
        observations.append(
            {
                "ply": ply,
                "before": before,
                "after": board.copy(),
                "mover": mover,
                "san": san,
                "node": node,
                "score": score,
                "info": info,
            }
        )

    # Fastchess's UCI score is the position searched immediately before the
    # move.  The next node's score is the resulting position searched by the
    # opponent.  Comparing those adjacent observations avoids confusing the
    # PGN-after-move [%eval] tag with the move's pre-move search score.
    for index, observation in enumerate(observations[:-1]):
        info = observation["info"]
        next_observation = observations[index + 1]
        next_info = next_observation["info"]
        if info is None or next_info is None:
            continue
        before_cp = info.score.comparable_cp
        next_cp = next_info.score.comparable_cp
        if before_cp is None or next_cp is None:
            continue
        mover = observation["mover"]
        after_cp = next_cp if next_observation["mover"] == mover else -next_cp
        eval_loss = before_cp - after_cp

        before = observation["before"]
        after = observation["after"]
        pawns_before = passed_pawns(before)
        pawns_after = passed_pawns(after)
        min_distance_before = _min_promotion_distance(pawns_before)
        min_distance_after = _min_promotion_distance(pawns_after)
        shallow = info.depth <= max_depth
        long_think = info.time_s >= time_threshold_s
        short_think = info.time_s < time_threshold_s
        time_left_ratio = (
            info.time_left_ms / initial_time_ms
            if info.time_left_ms is not None
            and initial_time_ms is not None
            and initial_time_ms > 0
            else None
        )
        time_pressure = (
            None
            if info.time_left_ms is None
            else info.time_left_ms <= time_left_threshold_ms
            or (
                time_left_ratio is not None
                and time_left_ratio <= time_left_ratio_threshold
            )
        )
        promotion_race = bool(pawns_before or pawns_after) and (
            (min_distance_before is not None and min_distance_before <= 2)
            or (min_distance_after is not None and min_distance_after <= 2)
        )
        passed_pawn_context = bool(pawns_before or pawns_after)
        mate_transition = info.score.mate is not None or next_info.score.mate is not None
        horizon_time_blunder = eval_loss >= min_loss_cp and (
            shallow or time_pressure is True
        )

        if eval_loss >= min_loss_cp or promotion_race:
            records.append(
                {
                    "game": game_index,
                    "round": game.headers.get("Round"),
                    "white": game.headers.get("White"),
                    "black": game.headers.get("Black"),
                    "result": game.headers.get("Result"),
                    "ply": observation["ply"],
                    "move_number": before.fullmove_number,
                    "mover": "white" if mover == chess.WHITE else "black",
                    "san": observation["san"],
                    "uci": observation["node"].move.uci(),
                    "fen_before": before.fen(),
                    "eval_before_raw": info.score.raw,
                    "eval_after_raw": next_info.score.raw,
                    "comment_eval_raw": _display_score(observation["score"]),
                    "next_comment_eval_raw": _display_score(next_observation["score"]),
                    "eval_before_cp_mover": before_cp,
                    "eval_after_cp_mover": after_cp,
                    "eval_loss_cp": eval_loss,
                    "depth": info.depth,
                    "seldepth": info.seldepth,
                    "think_time_s": info.time_s,
                    "nodes": info.nodes,
                    "nps": info.nps,
                    "hashfull": info.hashfull,
                    "time_left_ms": info.time_left_ms,
                    "time_left_ratio": time_left_ratio,
                    "fastchess_latency_delta_ms": info.fastchess_latency_delta_ms,
                    "pv": info.pv,
                    "passed_pawns_before": pawns_before,
                    "passed_pawns_after": pawns_after,
                    "promotion_distance_before": min_distance_before,
                    "promotion_distance_after": min_distance_after,
                    "flags": {
                        "shallow": shallow,
                        "long_think": long_think,
                        "short_think": short_think,
                        "time_pressure": time_pressure,
                        "mate_transition": mate_transition,
                        "passed_pawn_context": passed_pawn_context,
                        "promotion_race": promotion_race,
                        "horizon_time_blunder": horizon_time_blunder,
                    },
                }
            )

    return records, counts


def analyze_pgn(
    pgn_path: Path,
    min_loss_cp: int = 150,
    max_depth: int = 4,
    time_threshold_s: float = 0.35,
    time_left_threshold_ms: float = 1000.0,
    time_left_ratio_threshold: float = 0.05,
    initial_time_ms: Optional[float] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    games = 0
    move_counts = {"moves": 0, "moves_with_eval": 0, "search_infos": 0}
    parse_errors = 0
    with pgn_path.open("r", encoding="utf-8", errors="replace") as source:
        while game := chess.pgn.read_game(source):
            games += 1
            if game.errors:
                parse_errors += len(game.errors)
            game_records, counts = analyze_game(
                game,
                games,
                min_loss_cp,
                max_depth,
                time_threshold_s,
                time_left_threshold_ms,
                time_left_ratio_threshold,
                initial_time_ms,
            )
            records.extend(game_records)
            for key, value in counts.items():
                move_counts[key] += value

    records.sort(key=lambda record: (-int(record["eval_loss_cp"]), int(record["game"]), int(record["ply"])))
    summary = {
        "schema_version": 3,
        "pgn": str(pgn_path.resolve()),
        "games": games,
        "parse_errors": parse_errors,
        "moves": move_counts["moves"],
        "moves_with_eval": move_counts["moves_with_eval"],
        "search_infos": move_counts["search_infos"],
        "candidate_records": len(records),
        "horizon_time_blunders": sum(bool(r["flags"]["horizon_time_blunder"]) for r in records),
        "shallow_candidates": sum(bool(r["flags"]["shallow"]) for r in records),
        "long_think_candidates": sum(bool(r["flags"]["long_think"]) for r in records),
        "short_think_candidates": sum(bool(r["flags"]["short_think"]) for r in records),
        "time_pressure_candidates": sum(
            r["flags"]["time_pressure"] is True for r in records
        ),
        "time_pressure_unknown": sum(
            r["flags"]["time_pressure"] is None for r in records
        ),
        "mate_transition_candidates": sum(bool(r["flags"]["mate_transition"]) for r in records),
        "passed_pawn_candidates": sum(bool(r["flags"]["passed_pawn_context"]) for r in records),
        "promotion_race_candidates": sum(bool(r["flags"]["promotion_race"]) for r in records),
        "thresholds": {
            "min_loss_cp": min_loss_cp,
            "max_depth": max_depth,
            "time_threshold_s": time_threshold_s,
            "time_left_threshold_ms": time_left_threshold_ms,
            "time_left_ratio_threshold": time_left_ratio_threshold,
            "initial_time_ms": initial_time_ms,
        },
    }
    return records, summary


def write_report(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any], top: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = dict(summary)
    summary["top_candidates"] = records[:top]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fastchess PGN postmortem",
        "",
        f"- Games: `{summary['games']}`",
        f"- Search-info moves: `{summary['search_infos']}`",
        f"- Candidate records: `{summary['candidate_records']}`",
        f"- Horizon/time flags: `{summary['horizon_time_blunders']}`",
        f"- Long-think diagnostics: `{summary['long_think_candidates']}`",
        f"- Short-think diagnostics: `{summary['short_think_candidates']}`",
        f"- Time-pressure flags: `{summary['time_pressure_candidates']}`",
        f"- Time-pressure unknown: `{summary['time_pressure_unknown']}`",
        f"- Mate-transition candidates: `{summary['mate_transition_candidates']}`",
        f"- Passed-pawn-context candidates: `{summary['passed_pawn_candidates']}`",
        f"- Promotion-race flags: `{summary['promotion_race_candidates']}`",
        "",
        "## Top candidates",
        "",
    ]
    for record in records[:top]:
        flags = ", ".join(name for name, enabled in record["flags"].items() if enabled) or "none"
        lines.append(
            f"- Game {record['game']} ply {record['ply']} {record['mover']} "
            f"`{record['san']}`: loss `{record['eval_loss_cp']} cp`, "
            f"depth `{record['depth']}`, time `{record['think_time_s']}s`, flags `{flags}`"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pgn", type=Path, required=True)
    result.add_argument("--manifest", type=Path, default=None)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--top", type=int, default=50)
    result.add_argument("--min-loss-cp", type=int, default=150)
    result.add_argument("--max-depth", type=int, default=4)
    result.add_argument("--time-threshold-s", type=float, default=0.35)
    result.add_argument("--time-left-threshold-ms", type=float, default=1000.0)
    result.add_argument("--time-left-ratio-threshold", type=float, default=0.05)
    result.add_argument(
        "--initial-time-ms",
        type=float,
        default=None,
        help="initial clock time for optional time-left ratio diagnostics",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = None
    initial_time_ms = args.initial_time_ms
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if initial_time_ms is None:
            initial_time_ms = initial_time_ms_from_time_control(
                manifest.get("time_control")
            )
    records, summary = analyze_pgn(
        args.pgn,
        args.min_loss_cp,
        args.max_depth,
        args.time_threshold_s,
        args.time_left_threshold_ms,
        args.time_left_ratio_threshold,
        initial_time_ms,
    )
    if manifest is not None:
        summary["source_manifest"] = {
            "path": str(args.manifest.resolve()),
            "execution_status": manifest.get("execution_status"),
            "decision": manifest.get("decision"),
            "time_control": manifest.get("time_control"),
            "games_completed": manifest.get("games_completed"),
            "games_max": manifest.get("games_max"),
            "concurrency": manifest.get("concurrency"),
            "engine_a_baseline": manifest.get("engine_a_baseline", {}).get("search_profile"),
            "engine_b_candidate": manifest.get("engine_b_candidate", {}).get("search_profile"),
            "opening_book": manifest.get("opening_book", {}).get("book_id"),
        }
    write_report(args.output_dir, records, summary, args.top)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
