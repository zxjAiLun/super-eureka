#!/usr/bin/env python3
"""Deterministic UCI tournament runner for E1 engine comparisons.

Engine A is always the approved baseline and engine B is always the
candidate. Results are paired by opening and colour. This tool is a
fixed-game data collector; its pair counters and Elo interval are diagnostic
only. Formal feature acceptance is delegated to a validated
fastchess/OpenBench/Fishtest workflow.

Example:

    python tools/tournament.py \
        --engine-a target/release/chess-engine-demo.exe \
        --engine-b target/release/chess-engine-demo.exe \
        --games 64 --output-dir tournament-results/smoke
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import queue
import random
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Optional

import chess
import chess.pgn


DEFAULT_GAMES = 2048
DEFAULT_MOVETIME_MS = 100
DEFAULT_MOVE_GRACE_MS = 25
DEFAULT_HASH_MB = 16
DEFAULT_MAX_PLIES = 512
DEFAULT_SEED = 0
DEFAULT_DRAW_RATE = 0.5
STARTUP_TIMEOUT_SECONDS = 5.0
DEFAULT_OPENINGS = Path(__file__).with_name("openings.txt")
PAIR_CATEGORIES = ("0.0", "0.5", "1.0", "1.5", "2.0")
STATISTICAL_PERSPECTIVE = "engine_b_candidate_minus_engine_a_baseline"


class TournamentError(RuntimeError):
    """A protocol, legality, or configuration failure."""


class EngineStartupTimeout(TournamentError):
    """The engine did not complete the UCI startup handshake in time."""

    def __init__(self, message: str, elapsed_ms: float):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


class EngineMoveTimeout(TournamentError):
    """The side to move did not produce bestmove before the host deadline."""

    def __init__(self, message: str, elapsed_ms: float):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


class EngineProtocolError(TournamentError):
    """The UCI child exited or emitted an unusable response."""


class UCIResponseTimeout(TournamentError):
    """Internal timeout while waiting for one UCI response line."""

    def __init__(self, message: str, elapsed_ms: float):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


@dataclasses.dataclass(frozen=True)
class Opening:
    identifier: str
    moves: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ScheduledGame:
    number: int
    pair: int
    opening: Opening
    engine_a_white: bool


@dataclasses.dataclass
class GameRecord:
    number: int
    pair: int
    opening: str
    engine_a_white: bool
    result: str
    reason: str
    plies: int
    moves: list[str]
    elapsed_ms: dict[str, float]
    timeout_engine: Optional[str] = None
    timeout_color: Optional[str] = None
    timeout_elapsed_ms: Optional[float] = None
    error: Optional[str] = None
    error_engine: Optional[str] = None
    error_color: Optional[str] = None
    exit_code: Optional[int] = None
    stderr_log: Optional[str] = None
    stderr_tail: Optional[list[str]] = None

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Return a finite Elo value; boundary scores are Jeffreys-smoothed by callers."""
    if not 0.0 < score < 1.0:
        raise ValueError("score_to_elo requires a score strictly between 0 and 1")
    return 400.0 * math.log10(score / (1.0 - score))


def pair_category(pair_score: float) -> str:
    """Map a candidate two-game score in [0, 2] to a pentanomial category."""
    rounded = round(pair_score * 2.0) / 2.0
    key = f"{rounded:.1f}"
    if key not in PAIR_CATEGORIES:
        raise ValueError(f"invalid pair score {pair_score!r}")
    return key


def candidate_result_for_game(result: str, engine_a_white: bool) -> Optional[str]:
    """Return win/draw/loss strictly from candidate engine B's perspective."""
    if result == "1/2-1/2":
        return "draw"
    if result not in {"1-0", "0-1"}:
        return None
    candidate_is_white = not engine_a_white
    candidate_won = (result == "1-0") == candidate_is_white
    return "win" if candidate_won else "loss"


def game_score(result: str) -> float:
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    if result == "loss":
        return 0.0
    raise ValueError(f"invalid game result {result!r}")


def pair_score_from_results(first: str, second: str) -> float:
    return game_score(first) + game_score(second)


@dataclasses.dataclass
class DiagnosticPentanomialState:
    """Diagnostic pentanomial counters for complete colour-swapped pairs.

    The fixed draw-rate likelihood is retained only as a diagnostic number.
    It is not a GSPRT, has no error-boundary decision, and never stops a
    tournament.  Formal feature acceptance belongs to a validated external
    fastchess/OpenBench/Fishtest workflow.
    """

    elo0: float = 0.0
    elo1: float = 5.0
    draw_rate: float = DEFAULT_DRAW_RATE
    counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: {category: 0 for category in PAIR_CATEGORIES}
    )
    llr: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.draw_rate < 1.0:
            raise ValueError("draw-rate must be strictly between 0 and 1")
        if self.elo1 <= self.elo0:
            raise ValueError("elo1 must be greater than elo0")
        missing = set(PAIR_CATEGORIES) - set(self.counts)
        if missing:
            raise ValueError(f"missing pair categories: {sorted(missing)}")

    @property
    def pairs_completed(self) -> int:
        return sum(self.counts.values())

    def game_probabilities(self, elo: float) -> dict[str, float]:
        score = elo_to_score(elo)
        return {
            "win": (1.0 - self.draw_rate) * score,
            "draw": self.draw_rate,
            "loss": (1.0 - self.draw_rate) * (1.0 - score),
        }

    def pair_probabilities(self, elo: float) -> dict[str, float]:
        p = self.game_probabilities(elo)
        return {
            "0.0": p["loss"] ** 2,
            "0.5": 2.0 * p["loss"] * p["draw"],
            "1.0": p["draw"] ** 2 + 2.0 * p["win"] * p["loss"],
            "1.5": 2.0 * p["win"] * p["draw"],
            "2.0": p["win"] ** 2,
        }

    def update_pair(self, pair_score: float) -> None:
        category = pair_category(pair_score)
        p0 = self.pair_probabilities(self.elo0)[category]
        p1 = self.pair_probabilities(self.elo1)[category]
        if p0 <= 0.0 or p1 <= 0.0:
            raise ValueError(f"pair category has zero model probability: {category}")
        self.counts[category] += 1
        self.llr += math.log(p1 / p0)

    def as_json(self) -> dict[str, Any]:
        return {
            "model": "diagnostic_pentanomial_fixed_draw_rate",
            "diagnostic_only": True,
            "perspective": STATISTICAL_PERSPECTIVE,
            "draw_rate": self.draw_rate,
            "elo0": self.elo0,
            "elo1": self.elo1,
            "counts": dict(self.counts),
            "pairs_completed": self.pairs_completed,
            "llr": self.llr,
            "assumptions": [
                "fixed_draw_rate",
                "independent_game_likelihood_convolution",
            ],
        }


class EngineSession:
    """One short-lived UCI process with host-side deadlines and stderr capture."""

    startup_timeout_seconds = STARTUP_TIMEOUT_SECONDS

    def __init__(
        self,
        command: list[str],
        label: str,
        hash_mb: int,
        stderr_path: Optional[Path] = None,
    ):
        if not command:
            raise ValueError("engine command must not be empty")
        self.command = list(command)
        self.executable = self.command[0]
        self.label = label
        self.hash_mb = hash_mb
        self.stderr_path = stderr_path
        self.process: Optional[subprocess.Popen[str]] = None
        self.returncode: Optional[int] = None
        self._lines: queue.Queue[Optional[str]] = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self.stderr_tail: list[str] = []
        self.last_info: Optional[str] = None
        self.uci_name: Optional[str] = None
        self.uci_author: Optional[str] = None

    def __enter__(self) -> "EngineSession":
        try:
            self.start()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            raise EngineProtocolError(f"{self.label}: session already started")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise EngineProtocolError(
                f"{self.label}: cannot start {self.executable}: {exc}"
            ) from exc

        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._reader = threading.Thread(
            target=self._pump_stdout,
            args=(self.process.stdout,),
            name=f"uci-reader-{self.label}",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._pump_stderr,
            args=(self.process.stderr,),
            name=f"uci-stderr-{self.label}",
            daemon=True,
        )
        self._stderr_reader.start()

        self.send("uci")
        self._read_uci_handshake()
        self.send(f"setoption name Hash value {self.hash_mb}")
        self.send("isready")
        self.wait_for_prefix("readyok", timeout=self.startup_timeout_seconds, startup=True)
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for_prefix("readyok", timeout=self.startup_timeout_seconds, startup=True)

    def _pump_stdout(self, stdout: Any) -> None:
        try:
            for raw in stdout:
                self._lines.put(raw.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _pump_stderr(self, stderr: Any) -> None:
        sink = None
        try:
            if self.stderr_path is not None:
                self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
                sink = self.stderr_path.open("w", encoding="utf-8")
            for raw in stderr:
                line = raw.rstrip("\r\n")
                self.stderr_tail.append(line)
                del self.stderr_tail[:-80]
                if sink is not None:
                    sink.write(line + "\n")
                    sink.flush()
        finally:
            if sink is not None:
                sink.close()

    def _read_uci_handshake(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise EngineStartupTimeout(
                    f"{self.label}: waiting for uciok",
                    self.startup_timeout_seconds * 1000.0,
                )
            try:
                line = self._next_line(remaining)
            except UCIResponseTimeout as exc:
                raise EngineStartupTimeout(
                    f"{self.label}: waiting for uciok", exc.elapsed_ms
                ) from exc
            if line.startswith("id name "):
                self.uci_name = line[8:].strip()
            elif line.startswith("id author "):
                self.uci_author = line[10:].strip()
            elif line.startswith("uciok"):
                return

    def send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise EngineProtocolError(f"{self.label}: process is not running")
        if self.process.poll() is not None:
            raise EngineProtocolError(
                f"{self.label}: process exited with {self.process.returncode}"
            )
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise EngineProtocolError(f"{self.label}: send failed: {exc}") from exc

    def _next_line(self, timeout: float) -> str:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise UCIResponseTimeout(
                f"{self.label}: no UCI response in {timeout:.3f}s",
                timeout * 1000.0,
            ) from exc
        if line is None:
            code = self.process.returncode if self.process else self.returncode
            raise EngineProtocolError(f"{self.label}: stdout closed (exit={code})")
        return line

    def wait_for_prefix(self, prefix: str, timeout: float, startup: bool = False) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                if startup:
                    raise EngineStartupTimeout(
                        f"{self.label}: waiting for {prefix}", timeout * 1000.0
                    )
                raise UCIResponseTimeout(
                    f"{self.label}: waiting for {prefix}", timeout * 1000.0
                )
            try:
                line = self._next_line(remaining)
            except UCIResponseTimeout as exc:
                if startup:
                    raise EngineStartupTimeout(
                        f"{self.label}: waiting for {prefix}", exc.elapsed_ms
                    ) from exc
                raise
            if line.startswith(prefix):
                return line

    def position(self, moves: Iterable[str]) -> None:
        move_list = list(moves)
        command = "position startpos"
        if move_list:
            command += " moves " + " ".join(move_list)
        self.send(command)

    def go_movetime(self, movetime_ms: int, move_grace_ms: int) -> tuple[str, float]:
        self.last_info = None
        self.send(f"go movetime {movetime_ms}")
        sent_at = time.monotonic()
        deadline = sent_at + (movetime_ms + move_grace_ms) / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                elapsed_ms = (time.monotonic() - sent_at) * 1000.0
                raise EngineMoveTimeout(f"{self.label}: host movetime deadline", elapsed_ms)
            try:
                line = self._next_line(remaining)
            except UCIResponseTimeout as exc:
                raise EngineMoveTimeout(
                    f"{self.label}: host movetime deadline", exc.elapsed_ms
                ) from exc
            received_at = time.monotonic()
            if line.startswith("info "):
                self.last_info = line
            elif line.startswith("bestmove"):
                elapsed_ms = (received_at - sent_at) * 1000.0
                if received_at > deadline:
                    raise EngineMoveTimeout(
                        f"{self.label}: bestmove arrived after host deadline", elapsed_ms
                    )
                return parse_bestmove_line(line), elapsed_ms

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("quit\n")
                process.stdin.flush()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            self.returncode = process.returncode
            if self._reader is not None:
                self._reader.join(timeout=1.0)
            if self._stderr_reader is not None:
                self._stderr_reader.join(timeout=1.0)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.process = None


def parse_bestmove_line(line: str) -> str:
    fields = line.split()
    if len(fields) < 2 or fields[0] != "bestmove" or not fields[1]:
        raise EngineProtocolError(f"malformed bestmove line: {line!r}")
    return fields[1]


def score_statistics(pair_totals: Iterable[float]) -> dict[str, Any]:
    """Diagnostic pair-score interval with finite Elo endpoints.

    Wilson's Bernoulli assumptions do not exactly match five-valued pair
    scores, so this interval is deliberately reported as diagnostic only.
    """
    totals = list(pair_totals)
    pairs = len(totals)
    if pairs == 0:
        return {
            "pairs": 0,
            "candidate_score": None,
            "candidate_elo": None,
            "candidate_elo_ci95": None,
            "ci_method": "approximate_pair_wilson",
            "ci_status": "diagnostic_only",
        }
    score = sum(totals) / (2.0 * pairs)
    z = 1.959963984540054
    z2 = z * z
    center = (score + z2 / (2.0 * pairs)) / (1.0 + z2 / pairs)
    half = z * math.sqrt(
        score * (1.0 - score) / pairs + z2 / (4.0 * pairs * pairs)
    ) / (1.0 + z2 / pairs)
    epsilon = 0.5 / (2.0 * pairs + 1.0)
    low_score = max(epsilon, min(1.0 - epsilon, center - half))
    high_score = max(epsilon, min(1.0 - epsilon, center + half))
    point_score = max(epsilon, min(1.0 - epsilon, score))
    return {
        "pairs": pairs,
        "candidate_score": score,
        "candidate_elo": score_to_elo(point_score),
        "candidate_elo_ci95": [
            score_to_elo(low_score),
            score_to_elo(high_score),
        ],
        "ci_method": "approximate_pair_wilson",
        "ci_status": "diagnostic_only",
    }


def load_openings(path: Path) -> list[Opening]:
    openings: list[Opening] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 1)
        if len(fields) != 2:
            raise TournamentError(f"{path}:{line_number}: expected id and moves")
        identifier, move_text = fields
        if identifier in seen:
            raise TournamentError(f"{path}:{line_number}: duplicate opening {identifier}")
        moves = tuple(move_text.split())
        board = chess.Board()
        try:
            for move_uci in moves:
                if board.outcome(claim_draw=True) is not None:
                    raise ValueError("game ended before opening line finished")
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move {move_uci}")
                board.push(move)
        except (ValueError, chess.IllegalMoveError) as exc:
            raise TournamentError(f"{path}:{line_number}: {exc}") from exc
        openings.append(Opening(identifier, moves))
        seen.add(identifier)
    if len(openings) != 32:
        raise TournamentError(f"expected exactly 32 openings, found {len(openings)}")
    return openings


def build_schedule(openings: list[Opening], games: int, seed: int) -> list[ScheduledGame]:
    if games < 2 or games % 2 != 0:
        raise TournamentError("games must be an even number of at least 2")
    rng = random.Random(seed)
    schedule: list[ScheduledGame] = []
    pair_number = 1
    while len(schedule) < games:
        order = list(range(len(openings)))
        rng.shuffle(order)
        for opening_index in order:
            if len(schedule) >= games:
                break
            opening = openings[opening_index]
            schedule.append(
                ScheduledGame(len(schedule) + 1, pair_number, opening, True)
            )
            schedule.append(
                ScheduledGame(len(schedule) + 1, pair_number, opening, False)
            )
            pair_number += 1
    return schedule


def pgn_for_record(record: GameRecord, baseline_label: str, candidate_label: str) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessEngineDemo E1 tournament"
    game.headers["Round"] = str(record.pair)
    game.headers["Opening"] = record.opening
    game.headers["White"] = baseline_label if record.engine_a_white else candidate_label
    game.headers["Black"] = candidate_label if record.engine_a_white else baseline_label
    game.headers["Result"] = record.result
    game.headers["Pair"] = str(record.pair)
    game.headers["Game"] = str(record.number)
    game.headers["BaselineWhite"] = "yes" if record.engine_a_white else "no"
    game.headers["Reason"] = record.reason
    game.headers["Plies"] = str(record.plies)
    if record.error:
        game.headers["Error"] = record.error
    node = game
    board = game.board()
    for move_uci in record.moves:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            node.comment = f"record stopped before illegal/unreplayed move {move_uci}"
            break
        node = node.add_variation(move)
        board.push(move)
    if record.timeout_engine:
        node.comment = f"timeout: {record.timeout_engine} ({record.timeout_color})"
    return str(game.accept(chess.pgn.StringExporter(headers=True, variations=False)))


def _terminal_result(board: chess.Board) -> tuple[Optional[str], str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, ""
    if outcome.winner is None:
        return "1/2-1/2", outcome.termination.name.lower()
    return ("1-0" if outcome.winner == chess.WHITE else "0-1"), outcome.termination.name.lower()


def validate_bestmove(board: chess.Board, move_uci: str) -> Optional[chess.Move]:
    if move_uci == "0000":
        result, _reason = _terminal_result(board)
        if result is None:
            raise EngineProtocolError("bestmove 0000 with legal moves remaining")
        return None
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise TournamentError(f"malformed bestmove {move_uci!r}") from exc
    if move not in board.legal_moves:
        raise TournamentError(f"illegal bestmove {move_uci}")
    return move


def _safe_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in label)


def _engine_color(engine_side: str, scheduled: ScheduledGame) -> str:
    engine_a = engine_side == "a"
    is_white = scheduled.engine_a_white if engine_a else not scheduled.engine_a_white
    return "white" if is_white else "black"


def _engine_command(path_or_command: str | list[str]) -> list[str]:
    return list(path_or_command) if isinstance(path_or_command, list) else [path_or_command]


def play_game(
    scheduled: ScheduledGame,
    engine_a_path: str | list[str],
    engine_b_path: str | list[str],
    baseline_label: str,
    candidate_label: str,
    hash_mb: int,
    movetime_ms: int,
    move_grace_ms: int,
    max_plies: int,
    diagnostics_dir: Path,
) -> GameRecord:
    board = chess.Board()
    for move_uci in scheduled.opening.moves:
        board.push_uci(move_uci)
    moves = list(scheduled.opening.moves)
    elapsed_ms = {"white": 0.0, "black": 0.0}
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    stderr_paths = {
        "a": diagnostics_dir
        / f"game-{scheduled.number:04d}-a-{_safe_label(baseline_label)}.stderr.log",
        "b": diagnostics_dir
        / f"game-{scheduled.number:04d}-b-{_safe_label(candidate_label)}.stderr.log",
    }
    sessions = {
        "a": EngineSession(
            _engine_command(engine_a_path), baseline_label, hash_mb, stderr_paths["a"]
        ),
        "b": EngineSession(
            _engine_command(engine_b_path), candidate_label, hash_mb, stderr_paths["b"]
        ),
    }
    result: Optional[str] = None
    reason = ""
    active_side: Optional[str] = None
    active_color: Optional[str] = None
    error_side: Optional[str] = None
    timeout_engine: Optional[str] = None
    timeout_color: Optional[str] = None
    timeout_elapsed_ms: Optional[float] = None
    timeout_side: Optional[str] = None
    error: Optional[str] = None
    error_engine: Optional[str] = None
    error_color: Optional[str] = None

    try:
        error_side = "a"
        sessions["a"].start()
        error_side = "b"
        sessions["b"].start()
        result, reason = _terminal_result(board)
        while result is None and len(moves) < max_plies:
            active_side = "a" if board.turn == (chess.WHITE if scheduled.engine_a_white else chess.BLACK) else "b"
            active_color = "white" if board.turn == chess.WHITE else "black"
            session = sessions[active_side]
            try:
                session.position(moves)
                move_uci, elapsed = session.go_movetime(movetime_ms, move_grace_ms)
                elapsed_ms[active_color] += elapsed
            except EngineMoveTimeout as exc:
                timeout_engine = session.label
                timeout_color = active_color
                timeout_elapsed_ms = exc.elapsed_ms
                timeout_side = active_side
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = "timeout"
                break
            except TournamentError as exc:
                error = str(exc)
                error_engine = session.label
                error_color = active_color
                error_side = active_side
                reason = "protocol-error"
                result = "*"
                break

            try:
                move = validate_bestmove(board, move_uci)
            except TournamentError as exc:
                error = f"{session.label}: {exc}"
                error_engine = session.label
                error_color = active_color
                error_side = active_side
                reason = "illegal-move"
                result = "*"
                break
            if move is None:
                result, reason = _terminal_result(board)
                break
            board.push(move)
            moves.append(move_uci)
            result, reason = _terminal_result(board)
        if result is None and len(moves) >= max_plies:
            result = "1/2-1/2"
            reason = "max-plies"
    except EngineMoveTimeout as exc:
        failed_side = error_side or active_side or "a"
        failed_session = sessions[failed_side]
        timeout_engine = failed_session.label
        timeout_color = active_color or _engine_color(failed_side, scheduled)
        timeout_elapsed_ms = exc.elapsed_ms
        timeout_side = failed_side
        result = "0-1" if timeout_color == "white" else "1-0"
        reason = "timeout"
    except EngineStartupTimeout as exc:
        error = str(exc)
        failed_side = error_side or active_side or "a"
        failed_session = sessions[failed_side]
        error_engine = failed_session.label
        error_color = active_color or _engine_color(failed_side, scheduled)
        reason = "startup-timeout"
        result = "*"
    except TournamentError as exc:
        error = str(exc)
        failed_side = error_side or active_side or "a"
        failed_session = sessions[failed_side]
        error_engine = failed_session.label
        error_color = active_color or _engine_color(failed_side, scheduled)
        reason = "protocol-error"
        result = "*"
    finally:
        for session in sessions.values():
            session.close()

    diagnostic_side = error_side if error else timeout_side
    diagnostic_session = sessions.get(diagnostic_side) if diagnostic_side else None
    has_diagnostic = bool(error or timeout_engine)
    stderr_tail = diagnostic_session.stderr_tail if has_diagnostic and diagnostic_session else None
    stderr_log = str(diagnostic_session.stderr_path) if has_diagnostic and diagnostic_session else None
    exit_code = diagnostic_session.returncode if has_diagnostic and diagnostic_session else None
    return GameRecord(
        number=scheduled.number,
        pair=scheduled.pair,
        opening=scheduled.opening.identifier,
        engine_a_white=scheduled.engine_a_white,
        result=result or "*",
        reason=reason or "unknown",
        plies=len(moves),
        moves=moves,
        elapsed_ms=elapsed_ms,
        timeout_engine=timeout_engine,
        timeout_color=timeout_color,
        timeout_elapsed_ms=timeout_elapsed_ms,
        error=error,
        error_engine=error_engine,
        error_color=error_color,
        exit_code=exit_code,
        stderr_log=stderr_log,
        stderr_tail=stderr_tail,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path_or_command: str | list[str]) -> dict[str, Any]:
    executable = path_or_command[0] if isinstance(path_or_command, list) else path_or_command
    path = Path(executable).resolve()
    try:
        return {
            "resolved_path": str(path),
            "sha256": sha256_file(path),
            "file_size": path.stat().st_size,
        }
    except OSError:
        return {"resolved_path": str(path), "sha256": "unknown", "file_size": None}


def detect_git_sha(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def probe_engine(command: list[str], label: str, hash_mb: int, stderr_path: Path) -> dict[str, Any]:
    session = EngineSession(command, label, hash_mb, stderr_path)
    error = None
    try:
        session.start()
    except TournamentError as exc:
        error = str(exc)
    finally:
        session.close()
    return {
        "uci_id_name": session.uci_name,
        "uci_id_author": session.uci_author,
        "return_code": session.returncode,
        "stderr_log": str(stderr_path),
        "stderr_tail": list(session.stderr_tail),
        "probe_error": error,
    }


def build_manifest(
    args: argparse.Namespace,
    openings: list[Opening],
    output_dir: Path,
    engine_a_probe: dict[str, Any],
    engine_b_probe: dict[str, Any],
    started_utc: str,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    command_a = _engine_command(args.engine_a)
    command_b = _engine_command(args.engine_b)
    return {
        "engine_a_baseline": {
            **file_metadata(args.engine_a),
            "command": command_a,
            "label": args.label_a,
            "uci_options": {"Hash": args.hash_mb},
            "user_git_sha": args.sha_a or "unknown",
            **engine_a_probe,
        },
        "engine_b_candidate": {
            **file_metadata(args.engine_b),
            "command": command_b,
            "label": args.label_b,
            "uci_options": {"Hash": args.hash_mb},
            "user_git_sha": args.sha_b or "unknown",
            **engine_b_probe,
        },
        "runner_git_sha": detect_git_sha(repo_root),
        "runner_script_sha256": sha256_file(Path(__file__)),
        "openings_path": str(args.openings.resolve()),
        "openings_sha256": sha256_file(args.openings.resolve()),
        "opening_count": len(openings),
        "python_version": sys.version,
        "python_chess_version": chess.__version__,
        "platform": platform.platform(),
        "cli_args": list(args.cli_args),
        "started_utc": started_utc,
        "ended_utc": None,
        "games_requested": args.games,
        "games_completed": 0,
        "pairs_completed": 0,
        "movetime_ms": args.movetime_ms,
        "move_grace_ms": args.move_grace_ms,
        "hash_mb": args.hash_mb,
        "max_plies": args.max_plies,
        "seed": args.seed,
        "statistical_acceptance": "external_fastchess_openbench_fishtest",
        "diagnostic_model": {
            "model": "diagnostic_pentanomial_fixed_draw_rate",
            "diagnostic_only": True,
            "perspective": STATISTICAL_PERSPECTIVE,
            "elo0": DiagnosticPentanomialState().elo0,
            "elo1": DiagnosticPentanomialState().elo1,
            "draw_rate": args.draw_rate,
            "assumptions": [
                "fixed_draw_rate",
                "independent_game_likelihood_convolution",
            ],
        },
        "output_dir": str(output_dir.resolve()),
    }


def write_summary(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[GameRecord],
    pair_totals: list[float],
    diagnostic: DiagnosticPentanomialState,
    integrity_failed: bool,
) -> dict[str, Any]:
    wins = draws = losses = 0
    for record in records:
        perspective = candidate_result_for_game(record.result, record.engine_a_white)
        if perspective == "win":
            wins += 1
        elif perspective == "draw":
            draws += 1
        elif perspective == "loss":
            losses += 1
    statistics = score_statistics(pair_totals)
    if integrity_failed:
        status = "INTEGRITY_FAIL"
    elif len(records) == manifest["games_requested"]:
        status = "COMPLETED"
    else:
        status = "INCONCLUSIVE"
    summary = {
        "status": status,
        "integrity_status": "FAIL" if integrity_failed else "PASS",
        "statistical_status": "diagnostic_only",
        "elo_status": "diagnostic_only",
        "wins_for_candidate": wins,
        "draws": draws,
        "losses_for_candidate": losses,
        "games_completed": len(records),
        "pairs_completed": len(pair_totals),
        **statistics,
        "timeouts": [
            {
                "game": record.number,
                "engine": record.timeout_engine,
                "color": record.timeout_color,
                "elapsed_ms": record.timeout_elapsed_ms,
                "exit_code": record.exit_code,
                "stderr_log": record.stderr_log,
                "stderr_tail": record.stderr_tail,
                "reason": record.reason,
            }
            for record in records
            if record.timeout_engine
        ],
        "protocol_errors": [
            {
                "game": record.number,
                "engine": record.error_engine,
                "color": record.error_color,
                "exit_code": record.exit_code,
                "stderr_log": record.stderr_log,
                "stderr_tail": record.stderr_tail,
                "reason": record.reason,
                "error": record.error,
            }
            for record in records
            if record.error
        ],
        "diagnostic_model": diagnostic.as_json(),
        "manifest": manifest,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_tournament(args: argparse.Namespace) -> dict[str, Any]:
    if args.hash_mb < 1:
        raise TournamentError("hash-mb must be at least 1")
    if args.movetime_ms < 1:
        raise TournamentError("movetime-ms must be at least 1")
    if args.move_grace_ms < 0:
        raise TournamentError("move-grace-ms must be non-negative")
    if args.max_plies < 1:
        raise TournamentError("max-plies must be at least 1")
    openings = load_openings(args.openings)
    schedule = build_schedule(openings, args.games, args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    started_utc = datetime.now(timezone.utc).isoformat()

    engine_a_command = _engine_command(args.engine_a)
    engine_b_command = _engine_command(args.engine_b)
    probe_a = probe_engine(
        engine_a_command, args.label_a, args.hash_mb, diagnostics_dir / "probe-baseline.stderr.log"
    )
    probe_b = probe_engine(
        engine_b_command, args.label_b, args.hash_mb, diagnostics_dir / "probe-candidate.stderr.log"
    )
    if probe_a["probe_error"] or probe_b["probe_error"]:
        raise TournamentError(
            f"engine probe failed: baseline={probe_a['probe_error']!r}, "
            f"candidate={probe_b['probe_error']!r}"
        )

    manifest = build_manifest(args, openings, output_dir, probe_a, probe_b, started_utc)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostic = DiagnosticPentanomialState(draw_rate=args.draw_rate)
    records: list[GameRecord] = []
    pair_totals: list[float] = []
    integrity_failed = False

    with (output_dir / "games.jsonl").open("w", encoding="utf-8") as jsonl, (
        output_dir / "games.pgn"
    ).open("w", encoding="utf-8") as pgn_file:
        for pair_start in range(0, len(schedule), 2):
            pair_records: list[GameRecord] = []
            pair_schedule = schedule[pair_start : pair_start + 2]
            for scheduled in pair_schedule:
                print(
                    f"game {scheduled.number}/{len(schedule)} pair={scheduled.pair} "
                    f"opening={scheduled.opening.identifier} "
                    f"A={'white' if scheduled.engine_a_white else 'black'}",
                    flush=True,
                )
                record = play_game(
                    scheduled,
                    engine_a_command,
                    engine_b_command,
                    args.label_a,
                    args.label_b,
                    args.hash_mb,
                    args.movetime_ms,
                    args.move_grace_ms,
                    args.max_plies,
                    diagnostics_dir,
                )
                records.append(record)
                pair_records.append(record)
                jsonl.write(json.dumps(record.as_json(), sort_keys=True) + "\n")
                jsonl.flush()
                pgn_file.write(pgn_for_record(record, args.label_a, args.label_b) + "\n\n")
                pgn_file.flush()
                if record.error:
                    integrity_failed = True
                    print(f"protocol-error: {record.error}", file=sys.stderr)
                    break

            if integrity_failed or len(pair_records) != 2:
                break
            if (
                pair_records[0].pair != pair_records[1].pair
                or pair_records[0].opening != pair_records[1].opening
                or pair_records[0].engine_a_white == pair_records[1].engine_a_white
            ):
                integrity_failed = True
                break
            first = candidate_result_for_game(
                pair_records[0].result, pair_records[0].engine_a_white
            )
            second = candidate_result_for_game(
                pair_records[1].result, pair_records[1].engine_a_white
            )
            if first is None or second is None:
                integrity_failed = True
                break
            pair_total = pair_score_from_results(first, second)
            pair_totals.append(pair_total)
            diagnostic.update_pair(pair_total)

    manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["games_completed"] = len(records)
    manifest["pairs_completed"] = len(pair_totals)
    manifest["stop_reason"] = (
        "integrity-failure"
        if integrity_failed
        else "completed" if len(records) == args.games else "inconclusive"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = write_summary(
        output_dir, manifest, records, pair_totals, diagnostic, integrity_failed
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--engine-a", required=True, help="approved baseline executable")
    result.add_argument("--engine-b", required=True, help="candidate executable")
    result.add_argument("--label-a", default="baseline")
    result.add_argument("--label-b", default="candidate")
    result.add_argument("--sha-a", default=None, help="optional baseline git SHA")
    result.add_argument("--sha-b", default=None, help="optional candidate git SHA")
    result.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    result.add_argument("--games", type=int, default=DEFAULT_GAMES)
    result.add_argument("--movetime-ms", type=int, default=DEFAULT_MOVETIME_MS)
    result.add_argument("--move-grace-ms", type=int, default=DEFAULT_MOVE_GRACE_MS)
    result.add_argument("--hash-mb", type=int, default=DEFAULT_HASH_MB)
    result.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument(
        "--draw-rate",
        type=float,
        default=DEFAULT_DRAW_RATE,
        help="diagnostic-only fixed draw rate; never controls tournament status",
    )
    result.add_argument("--output-dir", type=Path, default=Path("tournament-results"))
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
    args.cli_args = list(sys.argv[1:] if argv is None else argv)
    try:
        summary = run_tournament(args)
    except (TournamentError, OSError, ValueError) as exc:
        print(f"tournament_error {exc}", file=sys.stderr)
        return 2
    if summary["integrity_status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
