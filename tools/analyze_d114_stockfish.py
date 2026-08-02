#!/usr/bin/env python3
"""Run a CP-only Stockfish postmortem over an aborted D1.14 pilot.

The tool compares the move actually played with Stockfish's best move from the
same position using two independent searches at a fixed node limit.  It keeps
centipawn loss and mate outcomes separate: mate scores are never converted to
an artificial centipawn value.

The first pass is intentionally a bounded screening pass.  It selects stable,
deterministic positions from every complete paired game, prioritising decisive
games, large PGN evaluation swings, common positions in a pair, and a seeded
sample of drawn games.  A second invocation can use ``--review-jsonl`` to
recheck the largest screen losses at a larger node limit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Optional

import chess
import chess.pgn

from verify_d114_match import (
    D114VerificationError,
    load_games,
    load_opening_lines,
    parse_manager_reasons,
    position_key,
    read_json,
    require,
    verify_binary_artifact,
    verify_opening_hash_artifact,
    verify_pairs,
)
from verify_d114_partial import verify_partial


BASELINE_LABEL = "Current"
CANDIDATE_LABEL = "CurrentLmr"
PROFILE_LABELS = (BASELINE_LABEL, CANDIDATE_LABEL)
COMMENT_SCORE_RE = re.compile(r"(?P<score>[+-]?(?:M\d+|\d+(?:\.\d+)?))/(?P<depth>\d+)")
INFO_SCORE_RE = re.compile(r"\bscore\s+(?P<kind>cp|mate)\s+(?P<value>[+-]?\d+)")
MATE_SCORES_ARE_NOT_CP = True


class StockfishAnalysisError(RuntimeError):
    """Raised when a teacher process cannot produce a trustworthy score."""


@dataclass(frozen=True)
class ParsedScore:
    kind: str
    value: int

    @property
    def raw(self) -> str:
        return f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class SearchResult:
    best_move: str
    score: ParsedScore
    depth: int
    pv: tuple[str, ...]


@dataclass(frozen=True)
class CommentScore:
    cp: int
    depth: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_comment_score(comment: str) -> Optional[CommentScore]:
    match = COMMENT_SCORE_RE.search(comment)
    if not match:
        return None
    raw = match.group("score")
    if raw.lstrip("+-").startswith("M"):
        sign = -1 if raw.startswith("-") else 1
        cp = sign * 100_000
    else:
        cp = round(float(raw) * 100.0)
    return CommentScore(cp=cp, depth=int(match.group("depth")))


def parse_info_score(line: str) -> Optional[ParsedScore]:
    match = INFO_SCORE_RE.search(line)
    if not match:
        return None
    return ParsedScore(kind=match.group("kind"), value=int(match.group("value")))


def score_is_positive_mate(score: ParsedScore) -> bool:
    return score.kind == "mate" and score.value > 0


def score_is_negative_mate(score: ParsedScore) -> bool:
    return score.kind == "mate" and score.value < 0


def mate_category(best: ParsedScore, played: ParsedScore) -> Optional[str]:
    if best.kind != "mate" and played.kind != "mate":
        return None
    if score_is_positive_mate(best):
        if not score_is_positive_mate(played):
            return "missed_mate"
        if played.value > best.value:
            return "mate_distance_increase"
        if played.value < best.value:
            return "mate_distance_decrease"
        return "allowed_mate"
    if score_is_negative_mate(best):
        if score_is_positive_mate(played):
            return "mate_reversal"
        if played.kind != "mate":
            return "escaped_losing_mate"
        if played.value > best.value:
            return "mate_distance_decrease"
        if played.value < best.value:
            return "mate_distance_increase"
        return "allowed_mate"
    if score_is_positive_mate(played):
        return "unexpected_mate"
    return "catastrophic_mate_swing"


def centipawn_loss(best: ParsedScore, played: ParsedScore) -> Optional[int]:
    if best.kind != "cp" or played.kind != "cp":
        return None
    return max(0, best.value - played.value)


def classify_cpl(loss: Optional[int]) -> str:
    if loss is None:
        return "mate-swing"
    if loss < 30:
        return "normal"
    if loss < 80:
        return "inaccuracy"
    if loss < 180:
        return "mistake"
    return "blunder"


def percentile(values: list[int], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stage_for_board(board: chess.Board) -> str:
    if board.fullmove_number <= 15:
        return "opening"
    non_pawn_pieces = sum(
        len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    if non_pawn_pieces <= 4 or len(board.piece_map()) <= 10:
        return "endgame"
    return "middlegame"


def is_quiet_move(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and not board.gives_check(move) and move.promotion is None


def _open_pgn(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_complete_games(run_dir: Path) -> tuple[dict[str, Any], list[chess.pgn.Game], dict[str, Any]]:
    run_dir = run_dir.expanduser().resolve()
    partial_summary = verify_partial(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    expected_fens = load_opening_lines(manifest, run_dir)
    reasons = parse_manager_reasons(run_dir)
    games = load_games(run_dir, expected_fens, reasons)
    complete_count = len(games) - (len(games) % 2)
    complete_games: list[chess.pgn.Game] = []
    with _open_pgn(run_dir / "match.pgn.gz") as source:
        number = 0
        while number < complete_count:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            number += 1
            complete_games.append(game)
    require(len(complete_games) == complete_count, "could not reload all complete PGN games")
    verify_pairs(
        games[:complete_count],
        expected_fens[: complete_count // 2],
        complete_count,
        require_sequential_order=False,
    )
    return manifest, complete_games, partial_summary


def build_move_records(games: list[chess.pgn.Game]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pair_games: dict[str, list[int]] = defaultdict(list)
    game_fens: dict[int, set[str]] = defaultdict(set)
    records_by_game: dict[int, list[dict[str, Any]]] = {}
    for game_number, game in enumerate(games, start=1):
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        require({white, black} == {BASELINE_LABEL, CANDIDATE_LABEL},
                f"game {game_number} has unexpected D1.14 labels")
        opening_fen = position_key(game.headers.get("FEN", ""))
        pair_games[opening_fen].append(game_number)
        board = game.board()
        game_records: list[dict[str, Any]] = []
        for ply, node in enumerate(game.mainline(), start=1):
            move = node.move
            require(board.is_legal(move), f"game {game_number} has illegal move {move.uci()}")
            before = board.copy()
            fen = before.fen()
            game_fens[game_number].add(fen)
            mover = before.turn
            profile = white if mover == chess.WHITE else black
            game_records.append(
                {
                    "game": game_number,
                    "pair": None,
                    "opening_fen": opening_fen,
                    "profile": profile,
                    "side": "white" if mover == chess.WHITE else "black",
                    "result": game.headers.get("Result", ""),
                    "ply": ply,
                    "move_number": before.fullmove_number,
                    "fen": fen,
                    "played_move": move.uci(),
                    "san": before.san(move),
                    "stage": stage_for_board(before),
                    "quiet_move": is_quiet_move(before, move),
                    "common_in_pair": False,
                    "comment_score_cp": (
                        parse_comment_score(node.comment).cp
                        if parse_comment_score(node.comment) is not None
                        else None
                    ),
                    "comment_depth": (
                        parse_comment_score(node.comment).depth
                        if parse_comment_score(node.comment) is not None
                        else None
                    ),
                    "self_eval_swing_cp": None,
                    "selection_reasons": [],
                }
            )
            board.push(move)
        records_by_game[game_number] = game_records
        if game_records:
            for index, record in enumerate(game_records[:-1]):
                current = record["comment_score_cp"]
                following = game_records[index + 1]["comment_score_cp"]
                if current is not None and following is not None:
                    record["self_eval_swing_cp"] = current + following
        records.extend(game_records)

    pair_index: dict[str, int] = {
        fen: index for index, fen in enumerate(pair_games, start=1)
    }
    for record in records:
        record["pair"] = pair_index[record["opening_fen"]]
    for opening_fen, game_numbers in pair_games.items():
        if len(game_numbers) != 2:
            continue
        common = game_fens[game_numbers[0]] & game_fens[game_numbers[1]]
        for game_number in game_numbers:
            for record in records_by_game[game_number]:
                if record["fen"] in common:
                    record["common_in_pair"] = True
    return records


def _add_selection(record: dict[str, Any], selected: dict[tuple[int, int], dict[str, Any]], reason: str) -> None:
    key = (int(record["game"]), int(record["ply"]))
    if key not in selected:
        selected[key] = record
    if reason not in selected[key]["selection_reasons"]:
        selected[key]["selection_reasons"].append(reason)


def select_records(
    records: list[dict[str, Any]],
    *,
    max_positions: int,
    per_decisive: int,
    per_draw: int,
    random_draws: int,
    common_per_pair: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_game[int(record["game"])].append(record)
        by_pair[int(record["pair"])].append(record)

    for game_number, game_records in by_game.items():
        decisive = game_records[0]["result"] != "1/2-1/2"
        ranked = sorted(
            game_records,
            key=lambda record: (
                abs(int(record["self_eval_swing_cp"] or 0)),
                int(record["ply"]),
            ),
            reverse=True,
        )
        limit = per_decisive if decisive else per_draw
        for record in ranked[:limit]:
            _add_selection(record, selected, "decisive-swing" if decisive else "draw-swing")
        if not decisive and random_draws:
            remaining = [record for record in game_records if (record["game"], record["ply"]) not in selected]
            for record in rng.sample(remaining, min(random_draws, len(remaining))):
                _add_selection(record, selected, "draw-random")

    for pair_number, pair_records in by_pair.items():
        common = [record for record in pair_records if record["common_in_pair"]]
        common_fens = sorted(
            {record["fen"] for record in common},
            key=lambda fen: max(
                abs(int(record["self_eval_swing_cp"] or 0))
                for record in common
                if record["fen"] == fen
            ),
            reverse=True,
        )
        for fen in common_fens[:common_per_pair]:
            for record in common:
                if record["fen"] == fen:
                    _add_selection(record, selected, "common-position")

    for record in records:
        if abs(int(record["self_eval_swing_cp"] or 0)) >= 100:
            _add_selection(record, selected, "large-self-swing")

    chosen = list(selected.values())
    chosen.sort(
        key=lambda record: (
            0 if "common-position" in record["selection_reasons"] else 1,
            0 if "large-self-swing" in record["selection_reasons"] else 1,
            -abs(int(record["self_eval_swing_cp"] or 0)),
            int(record["game"]),
            int(record["ply"]),
        )
    )
    return chosen[:max_positions]


class StockfishSession:
    def __init__(self, executable: Path, hash_mb: int, threads: int, timeout_s: float) -> None:
        self.executable = executable.expanduser().resolve()
        self.hash_mb = hash_mb
        self.threads = threads
        self.timeout_s = timeout_s
        self.process: Optional[subprocess.Popen[str]] = None
        self.stdout_queue: queue.Queue[Optional[str]] = queue.Queue()
        self.stderr_lines: list[str] = []
        self.identity: list[str] = []
        self.reader_threads: list[threading.Thread] = []

    def __enter__(self) -> "StockfishSession":
        if not self.executable.is_file():
            raise StockfishAnalysisError(f"Stockfish executable does not exist: {self.executable}")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                [str(self.executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise StockfishAnalysisError(f"cannot start Stockfish: {exc}") from exc
        assert self.process.stdout is not None and self.process.stderr is not None
        stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        self.reader_threads = [stdout_thread, stderr_thread]
        self._send("uci")
        while True:
            line = self._read_line(self.timeout_s)
            if line is None:
                raise StockfishAnalysisError("Stockfish closed before uciok")
            if line.startswith("id "):
                self.identity.append(line)
            if line == "uciok":
                break
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send(f"setoption name Threads value {self.threads}")
        self._ready()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.process is None:
            return
        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait(timeout=5)
        for thread in self.reader_threads:
            thread.join(timeout=1)

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in iter(self.process.stdout.readline, ""):
            self.stdout_queue.put(line.rstrip("\r\n"))
        self.stdout_queue.put(None)

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_lines.append(line.rstrip("\r\n"))
            del self.stderr_lines[:-20]

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise StockfishAnalysisError("Stockfish is not running")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_line(self, timeout_s: float) -> Optional[str]:
        try:
            return self.stdout_queue.get(timeout=max(timeout_s, 0.001))
        except queue.Empty as exc:
            diagnostics = "; ".join(self.stderr_lines[-5:])
            raise StockfishAnalysisError(f"Stockfish read timeout; stderr={diagnostics!r}") from exc

    def _ready(self) -> None:
        self._send("isready")
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StockfishAnalysisError("Stockfish readiness timeout")
            line = self._read_line(remaining)
            if line == "readyok":
                return
            if line is None:
                raise StockfishAnalysisError("Stockfish closed while waiting for readyok")

    def search(self, fen: str, nodes: int, search_move: Optional[str] = None) -> SearchResult:
        self._send("ucinewgame")
        self._send("setoption name Clear Hash")
        self._ready()
        self._send(f"position fen {fen}")
        command = f"go nodes {nodes}"
        if search_move is not None:
            command += f" searchmoves {search_move}"
        self._send(command)
        best_info: Optional[tuple[int, ParsedScore, tuple[str, ...]]] = None
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StockfishAnalysisError(
                    f"Stockfish search timeout for {fen} move={search_move}; stderr={self.stderr_lines[-5:]}"
                )
            line = self._read_line(remaining)
            if line is None:
                raise StockfishAnalysisError("Stockfish closed before bestmove")
            if line.startswith("info "):
                info = self._parse_info(line)
                if info is not None:
                    depth, score, pv = info
                    if best_info is None or depth >= best_info[0]:
                        best_info = (depth, score, pv)
            elif line.startswith("bestmove "):
                tokens = line.split()
                best_move = tokens[1] if len(tokens) > 1 else "0000"
                if best_info is None:
                    raise StockfishAnalysisError(f"Stockfish returned no score before {line!r}")
                return SearchResult(
                    best_move=best_move,
                    score=best_info[1],
                    depth=best_info[0],
                    pv=best_info[2],
                )

    @staticmethod
    def _parse_info(line: str) -> Optional[tuple[int, ParsedScore, tuple[str, ...]]]:
        if "lowerbound" in line or "upperbound" in line:
            return None
        tokens = line.split()
        try:
            depth = int(tokens[tokens.index("depth") + 1])
        except (ValueError, IndexError):
            return None
        score = parse_info_score(line)
        if score is None:
            return None
        pv: tuple[str, ...] = ()
        if "pv" in tokens:
            pv = tuple(tokens[tokens.index("pv") + 1 :])
        return depth, score, pv


def score_metrics(best: SearchResult, played: SearchResult, played_move: str) -> dict[str, Any]:
    loss = centipawn_loss(best.score, played.score)
    mate_event = mate_category(best.score, played.score)
    harmful_mate_categories = {
        "missed_mate",
        "mate_distance_increase",
        "catastrophic_mate_swing",
    }
    mate_loss = mate_event in harmful_mate_categories
    return {
        "best_move": best.best_move,
        "best_score": best.score.raw,
        "best_score_kind": best.score.kind,
        "best_score_value": best.score.value,
        "best_depth": best.depth,
        "best_pv": list(best.pv),
        "played_score": played.score.raw,
        "played_score_kind": played.score.kind,
        "played_score_value": played.score.value,
        "played_depth": played.depth,
        "played_pv": list(played.pv),
        "best_move_agreement": best.best_move == played_move,
        "centipawn_loss": loss,
        "classification": (
            classify_cpl(loss)
            if mate_event is None
            else ("mate-swing" if mate_loss else "mate-outcome")
        ),
        "mate_swing": mate_loss,
        "mate_outcome": mate_event is not None,
        "mate_category": mate_event,
    }


def _severity(record: dict[str, Any]) -> tuple[int, int]:
    if record.get("mate_swing"):
        category = record.get("mate_category")
        category_weight = {
            "catastrophic_mate_swing": 4,
            "missed_mate": 3,
            "mate_distance_increase": 2,
        }.get(category, 1)
        return (category_weight, 100_000)
    return (0, int(record.get("centipawn_loss") or 0))


def analyze_selected_records(
    records: list[dict[str, Any]],
    executable: Path,
    nodes: int,
    hash_mb: int,
    threads: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    analyzed: list[dict[str, Any]] = []
    with StockfishSession(executable, hash_mb, threads, timeout_s) as stockfish:
        best_cache: dict[str, SearchResult] = {}
        for index, record in enumerate(records, start=1):
            board = chess.Board(record["fen"])
            move = chess.Move.from_uci(record["played_move"])
            require(board.is_legal(move), f"selected move is illegal: {record['fen']} {move}")
            if record["fen"] not in best_cache:
                best_cache[record["fen"]] = stockfish.search(record["fen"], nodes)
            best = best_cache[record["fen"]]
            if best.best_move == "0000":
                raise StockfishAnalysisError(f"Stockfish returned bestmove 0000 for non-terminal {record['fen']}")
            try:
                best_move = chess.Move.from_uci(best.best_move)
            except ValueError as exc:
                raise StockfishAnalysisError(
                    f"Stockfish returned malformed bestmove {best.best_move!r}"
                ) from exc
            if not board.is_legal(best_move):
                raise StockfishAnalysisError(
                    f"Stockfish returned illegal bestmove {best.best_move} for {record['fen']}"
                )
            played = stockfish.search(record["fen"], nodes, record["played_move"])
            result = dict(record)
            result.update(score_metrics(best, played, record["played_move"]))
            result["stockfish_nodes"] = nodes
            result["stockfish_identity"] = list(stockfish.identity)
            analyzed.append(result)
            if index == 1 or index % 25 == 0 or index == len(records):
                print(f"analyzed {index}/{len(records)} positions", flush=True)
    return analyzed


def profile_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    cpls = [int(record["centipawn_loss"]) for record in records if record.get("centipawn_loss") is not None]
    scored = len(cpls)
    counts = {
        "moves": len(records),
        "cp_scored_moves": scored,
        "mate_swing": sum(bool(record.get("mate_swing")) for record in records),
        "mate_outcomes": sum(bool(record.get("mate_outcome")) for record in records),
        "best_move_matches": sum(bool(record.get("best_move_agreement")) for record in records),
        "inaccuracy": sum(record.get("classification") == "inaccuracy" for record in records),
        "mistake": sum(record.get("classification") == "mistake" for record in records),
        "blunder": sum(record.get("classification") == "blunder" for record in records),
        "cpl_300_plus": sum(value >= 300 for value in cpls),
        "cpl_500_plus": sum(value >= 500 for value in cpls),
        "quiet_blunders": sum(
            bool(record.get("quiet_move")) and record.get("classification") == "blunder"
            for record in records
        ),
    }
    stage = {
        name: sum(record.get("stage") == name for record in records)
        for name in ("opening", "middlegame", "endgame")
    }
    return {
        **counts,
        "mean_cpl": statistics.mean(cpls) if cpls else None,
        "median_cpl": statistics.median(cpls) if cpls else None,
        "p75_cpl": percentile(cpls, 0.75),
        "p90_cpl": percentile(cpls, 0.90),
        "p95_cpl": percentile(cpls, 0.95),
        "best_move_match_rate": counts["best_move_matches"] / len(records) if records else None,
        "inaccuracy_rate": counts["inaccuracy"] / scored if scored else None,
        "mistake_rate": counts["mistake"] / scored if scored else None,
        "blunder_rate": counts["blunder"] / scored if scored else None,
        "stage": stage,
    }


def common_position_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("common_in_pair"):
            grouped[(int(record["pair"]), record["fen"])][record["profile"]] = record
    candidate_better = baseline_better = tied = comparable = 0
    for values in grouped.values():
        candidate = values.get(CANDIDATE_LABEL)
        baseline = values.get(BASELINE_LABEL)
        if candidate is None or baseline is None:
            continue
        candidate_loss = candidate.get("centipawn_loss")
        baseline_loss = baseline.get("centipawn_loss")
        if candidate_loss is None or baseline_loss is None:
            continue
        comparable += 1
        if candidate_loss < baseline_loss:
            candidate_better += 1
        elif candidate_loss > baseline_loss:
            baseline_better += 1
        else:
            tied += 1
    return {
        "common_fen_groups": len(grouped),
        "comparable_cp_groups": comparable,
        "candidate_lower_cpl": candidate_better,
        "baseline_lower_cpl": baseline_better,
        "equal_cpl": tied,
    }


def summary_for_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    partial_summary: dict[str, Any],
    executable: Path,
    nodes: int,
    hash_mb: int,
    threads: int,
    phase: str,
    selection: dict[str, Any],
) -> dict[str, Any]:
    profiles = {
        profile: profile_statistics([record for record in records if record["profile"] == profile])
        for profile in PROFILE_LABELS
    }
    numeric_deltas = {}
    for key in ("mean_cpl", "median_cpl", "p75_cpl", "p90_cpl", "p95_cpl", "blunder_rate", "quiet_blunders", "mate_swing"):
        candidate_value = profiles[CANDIDATE_LABEL].get(key)
        baseline_value = profiles[BASELINE_LABEL].get(key)
        numeric_deltas[key] = (
            candidate_value - baseline_value
            if isinstance(candidate_value, (int, float)) and isinstance(baseline_value, (int, float))
            else None
        )
    return {
        "schema_version": 1,
        "analysis": "D1.14 Stockfish postmortem",
        "phase": phase,
        "score_model": "centipawn-loss and explicit mate outcomes",
        "wdl_enabled": False,
        "wdl_used": False,
        "stockfish": {
            "path": str(executable.resolve()),
            "sha256": sha256_file(executable),
            "nodes_per_search": nodes,
            "hash_mb": hash_mb,
            "threads": threads,
            "identity": selection.get("stockfish_identity", []),
        },
        "source": {
            "run_dir": str(Path(partial_summary["run_dir"]).resolve()),
            "manifest_status": manifest.get("status"),
            "match_decision": manifest.get("decision"),
            "raw_games": partial_summary["raw_games"],
            "complete_pair_games": partial_summary["complete_pair_games"],
            "complete_opening_pairs": partial_summary["complete_opening_pairs"],
            "unpaired_games": partial_summary["unpaired_games"],
        },
        "selection": selection,
        "profiles": profiles,
        "candidate_minus_baseline": numeric_deltas,
        "common_position_comparison": common_position_comparison(records),
        "records": len(records),
    }


def read_records_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StockfishAnalysisError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise StockfishAnalysisError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any], top: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "positions.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    report_lines = [
        "# D1.14 Stockfish postmortem",
        "",
        f"- Phase: `{summary['phase']}`",
        f"- Positions analyzed: `{summary['records']}`",
        f"- Node limit per search: `{summary['stockfish']['nodes_per_search']}`",
        "- Score model: `centipawn loss plus explicit mate outcomes`",
        "- WDL: `not requested and not used`",
        "",
        "## Profile comparison",
        "",
    ]
    for profile, stats in summary["profiles"].items():
        report_lines.append(
            f"- `{profile}`: moves `{stats['moves']}`, mean/median CPL "
            f"`{stats['mean_cpl']}` / `{stats['median_cpl']}`, "
            f"blunders `{stats['blunder']}`, mate swings `{stats['mate_swing']}`, "
            f"best-move match `{stats['best_move_match_rate']}`"
        )
    report_lines.extend(
        [
            "",
            "## Top losses",
            "",
        ]
    )
    ranked = sorted(records, key=_severity, reverse=True)
    for record in ranked[:top]:
        report_lines.append(
            f"- Game {record['game']} ply {record['ply']} `{record['profile']}` "
            f"`{record['san']}`: best `{record['best_move']} {record['best_score']}`, "
            f"played `{record['played_score']}`, CPL `{record['centipawn_loss']}`, "
            f"class `{record['classification']}`, mate `{record['mate_category']}`"
        )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--stockfish", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--review-jsonl", type=Path, default=None)
    command.add_argument("--nodes", type=int, default=25_000)
    command.add_argument("--top", type=int, default=50)
    command.add_argument("--deep-top", type=int, default=150)
    command.add_argument("--max-positions", type=int, default=2_000)
    command.add_argument("--per-decisive", type=int, default=2)
    command.add_argument("--per-draw", type=int, default=1)
    command.add_argument("--random-draws", type=int, default=1)
    command.add_argument("--common-per-pair", type=int, default=3)
    command.add_argument("--seed", type=int, default=11414)
    command.add_argument("--hash-mb", type=int, default=16)
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--timeout-s", type=float, default=30.0)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.nodes <= 0 or args.hash_mb <= 0 or args.threads <= 0:
        print("nodes, hash, and threads must be positive", file=sys.stderr)
        return 2
    try:
        manifest, games, partial_summary = load_complete_games(args.run_dir)
        all_records = build_move_records(games)
        if args.review_jsonl is not None:
            source_records = read_records_jsonl(args.review_jsonl)
            source_records.sort(key=_severity, reverse=True)
            selected = []
            seen: set[tuple[str, str, str]] = set()
            for record in source_records:
                key = (record["fen"], record["profile"], record["played_move"])
                if key in seen:
                    continue
                seen.add(key)
                selected.append(record)
                if len(selected) >= args.deep_top:
                    break
            phase = "deep-review"
            selection = {
                "mode": "top-screen-losses",
                "source_jsonl": str(args.review_jsonl.resolve()),
                "selected": len(selected),
                "requested": args.deep_top,
                "seed": None,
            }
        else:
            selected = select_records(
                all_records,
                max_positions=args.max_positions,
                per_decisive=args.per_decisive,
                per_draw=args.per_draw,
                random_draws=args.random_draws,
                common_per_pair=args.common_per_pair,
                seed=args.seed,
            )
            phase = "screen"
            selection = {
                "mode": "complete-pair-bounded-screen",
                "input_games": len(games),
                "input_moves": len(all_records),
                "selected": len(selected),
                "max_positions": args.max_positions,
                "per_decisive": args.per_decisive,
                "per_draw": args.per_draw,
                "random_draws": args.random_draws,
                "common_per_pair": args.common_per_pair,
                "seed": args.seed,
            }
        analyzed = analyze_selected_records(
            selected,
            args.stockfish,
            args.nodes,
            args.hash_mb,
            args.threads,
            args.timeout_s,
        )
        selection["stockfish_identity"] = analyzed[0].get("stockfish_identity", []) if analyzed else []
        summary = summary_for_records(
            analyzed,
            manifest,
            partial_summary,
            args.stockfish,
            args.nodes,
            args.hash_mb,
            args.threads,
            phase,
            selection,
        )
        write_outputs(args.output_dir, analyzed, summary, args.top)
    except (D114VerificationError, StockfishAnalysisError, OSError, ValueError) as exc:
        print(f"ANALYSIS_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
