#!/usr/bin/env python3
"""Deterministic UCI tournament runner for E1 engine comparisons.

The runner deliberately lives outside the Rust production crate.  It uses
python-chess for legality and game adjudication, drives two UCI processes,
and writes enough metadata to reproduce a result without turning a fixed
fixture benchmark into an Elo claim.

Example:

    python tools/tournament.py \
        --engine-a target/release/chess-engine-demo.exe \
        --engine-b target/release/chess-engine-demo.exe \
        --games 64 --output-dir tournament-results/smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
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
DEFAULT_HASH_MB = 16
DEFAULT_MAX_PLIES = 512
DEFAULT_SEED = 0
DEFAULT_OPENINGS = Path(__file__).with_name("openings.txt")


class TournamentError(RuntimeError):
    """A protocol, legality, or configuration failure."""


class EngineTimeout(TournamentError):
    """The side to move did not produce bestmove before the deadline."""


class EngineProtocolError(TournamentError):
    """The UCI child exited or emitted an unusable response."""


@dataclasses.dataclass(frozen=True)
class Opening:
    identifier: str
    moves: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ScheduledGame:
    number: int
    opening: Opening
    engine_a_white: bool


@dataclasses.dataclass
class GameRecord:
    number: int
    opening: str
    engine_a_white: bool
    result: str
    reason: str
    plies: int
    moves: list[str]
    elapsed_ms: dict[str, float]
    timeout_side: Optional[str] = None
    error: Optional[str] = None

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SprtState:
    """Score-based SPRT state for engine A versus engine B.

    A draw contributes half a win and half a loss to the log likelihood.
    This is the standard score-SPRT approximation and is intentionally
    recorded in the output so the result is not mistaken for a full
    three-outcome model.
    """

    elo0: float = 0.0
    elo1: float = 5.0
    alpha: float = 0.05
    beta: float = 0.05
    wins: int = 0
    draws: int = 0
    losses: int = 0
    llr: float = 0.0
    decision: Optional[str] = None

    @property
    def upper_bound(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower_bound(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    def update(self, result_for_a: str) -> Optional[str]:
        if result_for_a not in {"win", "draw", "loss"}:
            raise ValueError(f"invalid SPRT result: {result_for_a}")
        if result_for_a == "win":
            self.wins += 1
            fraction = 1.0
        elif result_for_a == "draw":
            self.draws += 1
            fraction = 0.5
        else:
            self.losses += 1
            fraction = 0.0

        p0 = elo_to_score(self.elo0)
        p1 = elo_to_score(self.elo1)
        win_term = math.log(p1 / p0)
        loss_term = math.log((1.0 - p1) / (1.0 - p0))
        self.llr += fraction * win_term + (1.0 - fraction) * loss_term

        if self.llr >= self.upper_bound:
            self.decision = "PASS"
        elif self.llr <= self.lower_bound:
            self.decision = "REJECTED"
        return self.decision

    def as_json(self) -> dict[str, Any]:
        return {
            "elo0": self.elo0,
            "elo1": self.elo1,
            "alpha": self.alpha,
            "beta": self.beta,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "llr": self.llr,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "decision": self.decision,
        }


class EngineSession:
    """One short-lived UCI process with line-oriented response handling."""

    def __init__(
        self,
        command: list[str],
        label: str,
        hash_mb: int,
        response_timeout: Optional[float] = None,
    ):
        if not command:
            raise ValueError("engine command must not be empty")
        self.command = list(command)
        self.executable = self.command[0]
        self.label = label
        self.hash_mb = hash_mb
        self.response_timeout = response_timeout
        self.process: Optional[subprocess.Popen[str]] = None
        self._lines: queue.Queue[Optional[str]] = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self.last_info: Optional[str] = None

    def __enter__(self) -> "EngineSession":
        self.start()
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
                stderr=subprocess.DEVNULL,
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
        self._reader = threading.Thread(
            target=self._pump_stdout,
            args=(self.process.stdout,),
            name=f"uci-reader-{self.label}",
            daemon=True,
        )
        self._reader.start()
        self.send("uci")
        self.wait_for_prefix("uciok", timeout=5.0)
        self.send(f"setoption name Hash value {self.hash_mb}")
        self.send("isready")
        self.wait_for_prefix("readyok", timeout=5.0)
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for_prefix("readyok", timeout=5.0)

    def _pump_stdout(self, stdout: Any) -> None:
        try:
            for raw in stdout:
                self._lines.put(raw.rstrip("\r\n"))
        finally:
            self._lines.put(None)

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
            raise EngineTimeout(f"{self.label}: no UCI response in {timeout:.3f}s") from exc
        if line is None:
            code = self.process.returncode if self.process else None
            raise EngineProtocolError(f"{self.label}: stdout closed (exit={code})")
        return line

    def wait_for_prefix(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeout(f"{self.label}: waiting for {prefix}")
            line = self._next_line(remaining)
            if line.startswith(prefix):
                return line

    def position(self, moves: Iterable[str]) -> None:
        move_list = list(moves)
        command = "position startpos"
        if move_list:
            command += " moves " + " ".join(move_list)
        self.send(command)

    def go_movetime(
        self, movetime_ms: int, response_timeout: Optional[float] = None
    ) -> tuple[str, float]:
        self.last_info = None
        started = time.monotonic()
        self.send(f"go movetime {movetime_ms}")
        timeout = response_timeout or self.response_timeout
        if timeout is None:
            timeout = max(2.0, movetime_ms / 1000.0 + 2.0)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineTimeout(f"{self.label}: bestmove timeout")
            line = self._next_line(remaining)
            if line.startswith("info "):
                self.last_info = line
            elif line.startswith("bestmove"):
                return parse_bestmove_line(line), (time.monotonic() - started) * 1000.0

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
            if self._reader is not None:
                self._reader.join(timeout=1.0)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.process = None


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> Optional[float]:
    if not 0.0 < score < 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def parse_bestmove_line(line: str) -> str:
    """Parse the UCI bestmove token without deciding board legality."""
    fields = line.split()
    if len(fields) < 2 or fields[0] != "bestmove" or not fields[1]:
        raise EngineProtocolError(f"malformed bestmove line: {line!r}")
    return fields[1]


def score_statistics(wins: int, draws: int, losses: int) -> dict[str, Any]:
    games = wins + draws + losses
    if games == 0:
        return {"games": 0, "score": None, "elo": None, "elo_ci95": None}
    score = (wins + 0.5 * draws) / games
    elo = score_to_elo(score)
    values = [1.0] * wins + [0.5] * draws + [0.0] * losses
    observed_variance = sum((value - score) ** 2 for value in values) / games
    # A pure draw match has zero observed score variance, but it does not
    # prove Elo with zero uncertainty. Use the Bernoulli score variance as a
    # conservative floor for the score-SPRT approximation.
    variance = max(observed_variance, score * (1.0 - score))
    se = math.sqrt(variance / games)
    low_score = max(1e-9, score - 1.96 * se)
    high_score = min(1.0 - 1e-9, score + 1.96 * se)
    low_elo = score_to_elo(low_score)
    high_elo = score_to_elo(high_score)
    return {
        "games": games,
        "score": score,
        "elo": elo,
        "elo_ci95": [low_elo, high_elo],
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
    if games < 1:
        raise TournamentError("games must be at least 1")
    rng = random.Random(seed)
    schedule: list[ScheduledGame] = []
    while len(schedule) < games:
        order = list(range(len(openings)))
        rng.shuffle(order)
        for opening_index in order:
            if len(schedule) >= games:
                break
            opening = openings[opening_index]
            schedule.append(
                ScheduledGame(
                    number=len(schedule) + 1,
                    opening=opening,
                    engine_a_white=True,
                )
            )
            if len(schedule) >= games:
                break
            schedule.append(
                ScheduledGame(
                    number=len(schedule) + 1,
                    opening=opening,
                    engine_a_white=False,
                )
            )
    return schedule


def result_for_a(result: str, engine_a_white: bool) -> Optional[str]:
    if result == "1/2-1/2":
        return "draw"
    if result not in {"1-0", "0-1"}:
        return None
    a_won = (result == "1-0") == engine_a_white
    return "win" if a_won else "loss"


def pgn_for_record(record: GameRecord, engine_a_label: str, engine_b_label: str) -> str:
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessEngineDemo E1 tournament"
    game.headers["Round"] = str(record.number)
    game.headers["Opening"] = record.opening
    game.headers["White"] = engine_a_label if record.engine_a_white else engine_b_label
    game.headers["Black"] = engine_b_label if record.engine_a_white else engine_a_label
    game.headers["Result"] = record.result
    game.headers["EngineAWhite"] = "yes" if record.engine_a_white else "no"
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
    if record.timeout_side:
        node.comment = f"timeout: {record.timeout_side}"
    return str(game.accept(chess.pgn.StringExporter(headers=True, variations=False)))


def _terminal_result(board: chess.Board) -> tuple[Optional[str], str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None, ""
    if outcome.winner is None:
        return "1/2-1/2", outcome.termination.name.lower()
    return ("1-0" if outcome.winner == chess.WHITE else "0-1"), outcome.termination.name.lower()


def validate_bestmove(board: chess.Board, move_uci: str) -> Optional[chess.Move]:
    """Return a legal move, or None for terminal ``bestmove 0000``."""
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


def play_game(
    scheduled: ScheduledGame,
    engine_a_path: str,
    engine_b_path: str,
    engine_a_label: str,
    engine_b_label: str,
    hash_mb: int,
    movetime_ms: int,
    max_plies: int,
) -> GameRecord:
    board = chess.Board()
    for move_uci in scheduled.opening.moves:
        board.push_uci(move_uci)
    moves = list(scheduled.opening.moves)
    elapsed_ms = {"white": 0.0, "black": 0.0}
    timeout_side: Optional[str] = None
    error: Optional[str] = None
    reason = ""

    sessions: dict[str, EngineSession] = {}
    try:
        sessions["a"] = EngineSession([engine_a_path], engine_a_label, hash_mb)
        sessions["b"] = EngineSession([engine_b_path], engine_b_label, hash_mb)
        sessions["a"].start()
        sessions["b"].start()

        result, reason = _terminal_result(board)
        while result is None and len(moves) < max_plies:
            a_to_move = board.turn == (chess.WHITE if scheduled.engine_a_white else chess.BLACK)
            side = "a" if a_to_move else "b"
            color_name = "white" if board.turn == chess.WHITE else "black"
            session = sessions[side]
            try:
                session.position(moves)
                move_uci, elapsed = session.go_movetime(movetime_ms)
                elapsed_ms[color_name] += elapsed
            except EngineTimeout:
                timeout_side = color_name
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                reason = "timeout"
                break
            except TournamentError as exc:
                error = str(exc)
                reason = "protocol-error"
                break

            try:
                move = validate_bestmove(board, move_uci)
            except TournamentError as exc:
                error = f"{side}: {exc}"
                reason = (
                    "protocol-error"
                    if isinstance(exc, EngineProtocolError)
                    else "illegal-move"
                )
                break
            if move is None:
                result, reason = _terminal_result(board)
                break
            board.push(move)
            moves.append(move_uci)
            result, reason = _terminal_result(board)

        if result is None and error is None and len(moves) >= max_plies:
            result = "1/2-1/2"
            reason = "max-plies"
        if result is None and error is not None:
            result = "*"
        if result is None:
            result = "1/2-1/2"
            reason = reason or "draw"
    finally:
        for session in sessions.values():
            session.close()

    return GameRecord(
        number=scheduled.number,
        opening=scheduled.opening.identifier,
        engine_a_white=scheduled.engine_a_white,
        result=result,
        reason=reason or "unknown",
        plies=len(moves),
        moves=moves,
        elapsed_ms=elapsed_ms,
        timeout_side=timeout_side,
        error=error,
    )


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


def write_summary(
    output_dir: Path,
    manifest: dict[str, Any],
    records: list[GameRecord],
    sprt: SprtState,
    integrity_failed: bool,
) -> dict[str, Any]:
    wins = draws = losses = 0
    for record in records:
        perspective = result_for_a(record.result, record.engine_a_white)
        if perspective == "win":
            wins += 1
        elif perspective == "draw":
            draws += 1
        elif perspective == "loss":
            losses += 1
    statistics = score_statistics(wins, draws, losses)
    decision = sprt.decision or "INCONCLUSIVE"
    summary = {
        "status": decision,
        "integrity_status": "FAIL" if integrity_failed else "PASS",
        "manifest": manifest,
        "wins_for_a": wins,
        "draws": draws,
        "losses_for_a": losses,
        **statistics,
        "timeouts": [
            {"game": record.number, "side": record.timeout_side}
            for record in records
            if record.timeout_side
        ],
        "protocol_errors": [
            {"game": record.number, "error": record.error}
            for record in records
            if record.error
        ],
        "sprt": sprt.as_json(),
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
    if args.max_plies < 1:
        raise TournamentError("max-plies must be at least 1")
    openings = load_openings(args.openings)
    schedule = build_schedule(openings, args.games, args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "engine_a": {
            "label": args.label_a,
            "command": [args.engine_a],
            "path": str(Path(args.engine_a).resolve()),
            "sha": args.sha_a,
        },
        "engine_b": {
            "label": args.label_b,
            "command": [args.engine_b],
            "path": str(Path(args.engine_b).resolve()),
            "sha": args.sha_b,
        },
        "openings": str(args.openings.resolve()),
        "opening_count": len(openings),
        "games_requested": args.games,
        "movetime_ms": args.movetime_ms,
        "hash_mb": args.hash_mb,
        "max_plies": args.max_plies,
        "seed": args.seed,
        "sprt": {
            "elo0": args.elo0,
            "elo1": args.elo1,
            "alpha": args.alpha,
            "beta": args.beta,
            "model": "score-SPRT; draws contribute half a win and half a loss",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sprt = SprtState(args.elo0, args.elo1, args.alpha, args.beta)
    records: list[GameRecord] = []
    integrity_failed = False
    with (output_dir / "games.jsonl").open("w", encoding="utf-8") as jsonl, (
        output_dir / "games.pgn"
    ).open("w", encoding="utf-8") as pgn_file:
        for scheduled in schedule:
            print(
                f"game {scheduled.number}/{len(schedule)} "
                f"opening={scheduled.opening.identifier} "
                f"A={'white' if scheduled.engine_a_white else 'black'}",
                flush=True,
            )
            record = play_game(
                scheduled,
                args.engine_a,
                args.engine_b,
                args.label_a,
                args.label_b,
                args.hash_mb,
                args.movetime_ms,
                args.max_plies,
            )
            records.append(record)
            jsonl.write(json.dumps(record.as_json(), sort_keys=True) + "\n")
            jsonl.flush()
            pgn_file.write(
                pgn_for_record(record, args.label_a, args.label_b) + "\n\n"
            )
            pgn_file.flush()

            perspective = result_for_a(record.result, record.engine_a_white)
            if perspective is not None:
                sprt.update(perspective)
            if record.error:
                integrity_failed = True
                print(f"protocol-error: {record.error}", file=sys.stderr)
                break
            if sprt.decision:
                print(f"SPRT {sprt.decision} after {len(records)} games", flush=True)
                break

    summary = write_summary(output_dir, manifest, records, sprt, integrity_failed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_sha = detect_git_sha(repo_root)
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--engine-a", required=True, help="path to engine A executable")
    result.add_argument("--engine-b", required=True, help="path to engine B executable")
    result.add_argument("--label-a", default="baseline")
    result.add_argument("--label-b", default="candidate")
    result.add_argument("--sha-a", default=default_sha)
    result.add_argument("--sha-b", default=default_sha)
    result.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    result.add_argument("--games", type=int, default=DEFAULT_GAMES)
    result.add_argument("--movetime-ms", type=int, default=DEFAULT_MOVETIME_MS)
    result.add_argument("--hash-mb", type=int, default=DEFAULT_HASH_MB)
    result.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--output-dir", type=Path, default=Path("tournament-results"))
    result.add_argument("--elo0", type=float, default=0.0)
    result.add_argument("--elo1", type=float, default=5.0)
    result.add_argument("--alpha", type=float, default=0.05)
    result.add_argument("--beta", type=float, default=0.05)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = parser().parse_args(argv)
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
