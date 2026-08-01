"""Run the pinned D1.11 external validation corpus against two UCI profiles.

This is a safety runner, not an Elo tool. It uses the same FEN, depth, Hash
value, and absolute wall-clock search budget for Current and the candidate.
Every case produces a structured record so a single failure remains
diagnosable instead of being reduced to an aggregate pass rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import chess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "tests" / "data" / "external_validation_v1.epd"
DEFAULT_PROJECT_CORPUS = ROOT / "tests" / "data" / "search_validation.epd"
DEFAULT_SOURCES = ROOT / "tests" / "data" / "external_validation_v1.sources.json"
DEFAULT_BOOKS_MANIFEST = ROOT / "books" / "manifest.json"
READER_QUEUE_SIZE = 256
RECENT_OUTPUT_LINES = 20


class CorpusIntegrityError(RuntimeError):
    """The committed corpus or its provenance metadata is not trustworthy."""


class EngineFailure(RuntimeError):
    """A UCI child failed a protocol, legality, or deadline gate."""


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    fen: str
    source_book_id: str
    source_line: int
    depth: int
    allowed_moves: tuple[str, ...] = ()
    allowed_wildcard: bool = False
    score_class: Optional[str] = None


@dataclass(frozen=True)
class Score:
    kind: str
    value: int


@dataclass(frozen=True)
class SearchResult:
    bestmove: str
    completed_depth: int
    score: Score
    pv: tuple[str, ...]


@dataclass(frozen=True)
class TimedSearchResult:
    bestmove: str
    completed_depth: int
    score: Score
    pv: tuple[str, ...]
    nodes: Optional[int]
    qsearch_nodes: Optional[int]
    elapsed_ms: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusIntegrityError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusIntegrityError(f"metadata root must be an object: {path}")
    return value


def _parse_cases(corpus_path: Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorpusIntegrityError(f"cannot read corpus {corpus_path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 6:
            raise CorpusIntegrityError(
                f"{corpus_path}:{line_number}: expected 6 fields, got {len(fields)}"
            )
        case_id, group, fen, source_book_id, source_line_text, depth_text = fields
        if not case_id or case_id in seen:
            raise CorpusIntegrityError(f"duplicate or empty case id at line {line_number}")
        seen.add(case_id)
        try:
            source_line = int(source_line_text)
            depth = int(depth_text)
            board = chess.Board(fen)
        except (ValueError, TypeError) as exc:
            raise CorpusIntegrityError(
                f"{corpus_path}:{line_number}: invalid FEN or integer field: {exc}"
            ) from exc
        if not board.is_valid():
            raise CorpusIntegrityError(f"{case_id}: FEN is not a valid chess position")
        if not group or source_line <= 0 or depth <= 0:
            raise CorpusIntegrityError(f"{case_id}: group/source line/depth is invalid")
        cases.append(Case(case_id, group, fen, source_book_id, source_line, depth))
    return cases


def load_project_corpus(corpus_path: Path = DEFAULT_PROJECT_CORPUS) -> tuple[list[Case], dict[str, Any]]:
    """Load the pinned D1.10 project-curated corpus for deep differential runs."""

    cases: list[Case] = []
    seen: set[str] = set()
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CorpusIntegrityError(f"cannot read project corpus {corpus_path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) not in (8, 9):
            raise CorpusIntegrityError(
                f"{corpus_path}:{line_number}: expected 8 or 9 fields, got {len(fields)}"
            )
        case_id, group, fen, allowed_text, score_class, depth_text, source, license_text = fields[:8]
        forbidden_text = fields[8] if len(fields) == 9 else "-"
        if not case_id or case_id in seen:
            raise CorpusIntegrityError(f"duplicate or empty project case id at line {line_number}")
        if source != "project-curated" or license_text != "CC0-1.0":
            raise CorpusIntegrityError(f"{case_id}: project corpus source/license changed")
        try:
            board = chess.Board(fen)
            depth = int(depth_text)
        except (ValueError, TypeError) as exc:
            raise CorpusIntegrityError(f"{case_id}: invalid FEN or depth: {exc}") from exc
        # D1.10 is the existing project-curated corpus.  Preserve its
        # historical FEN acceptance boundary here; the Rust harness owns its
        # established corpus semantics, while this loader only needs a
        # parseable position and a legal move set for differential probing.
        if depth <= 0 or not group or not score_class:
            raise CorpusIntegrityError(f"{case_id}: invalid project corpus fields")
        allowed_wildcard = allowed_text == "*"
        allowed_moves = () if allowed_wildcard else tuple(
            move.strip() for move in allowed_text.split(",") if move.strip()
        )
        if not allowed_wildcard and not allowed_moves:
            raise CorpusIntegrityError(f"{case_id}: allowed move set is empty")
        legal = {move.uci() for move in board.legal_moves}
        terminal = board.is_checkmate() or board.is_stalemate()
        if any(move != "0000" and move not in legal for move in allowed_moves):
            raise CorpusIntegrityError(f"{case_id}: allowed move is not legal")
        if "0000" in allowed_moves and not terminal:
            raise CorpusIntegrityError(f"{case_id}: 0000 is only valid for a terminal case")
        forbidden_moves = () if forbidden_text in {"", "-"} else tuple(
            move.strip() for move in forbidden_text.split(",") if move.strip()
        )
        if any(move not in legal for move in forbidden_moves):
            raise CorpusIntegrityError(f"{case_id}: forbidden move is not legal")
        if set(allowed_moves) & set(forbidden_moves):
            raise CorpusIntegrityError(f"{case_id}: allowed and forbidden moves overlap")
        seen.add(case_id)
        cases.append(
            Case(
                case_id,
                group,
                fen,
                source,
                line_number,
                depth,
                allowed_moves,
                allowed_wildcard,
                score_class,
            )
        )
    if len(cases) != 23:
        raise CorpusIntegrityError(f"D1.10 case count changed: expected 23, got {len(cases)}")
    return cases, {
        "corpus_id": "d1.10-project-curated-v2",
        "case_count": len(cases),
        "group_counts": dict(Counter(case.group for case in cases)),
        "snapshot_sha256": sha256_file(corpus_path),
    }


def load_corpus(
    corpus_path: Path = DEFAULT_CORPUS,
    sources_path: Path = DEFAULT_SOURCES,
    books_manifest_path: Path = DEFAULT_BOOKS_MANIFEST,
) -> tuple[list[Case], dict[str, Any]]:
    """Validate the pinned snapshot and its upstream provenance."""

    metadata = _load_json(sources_path)
    if metadata.get("schema_version") != 1:
        raise CorpusIntegrityError("unsupported external corpus metadata schema")
    if metadata.get("corpus_id") != "d1.11-official-stockfish-books-slice-v1":
        raise CorpusIntegrityError("unexpected D1.11 corpus id")
    if metadata.get("license") != "CC0-1.0":
        raise CorpusIntegrityError("D1.11 corpus license must be CC0-1.0")
    if metadata.get("snapshot_file") != "tests/data/external_validation_v1.epd":
        raise CorpusIntegrityError("snapshot_file does not match the committed layout")
    expected_hash = metadata.get("snapshot_sha256")
    actual_hash = sha256_file(corpus_path)
    if expected_hash != actual_hash:
        raise CorpusIntegrityError(
            f"snapshot SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    books_manifest = _load_json(books_manifest_path)
    books = books_manifest.get("books")
    if not isinstance(books, dict):
        raise CorpusIntegrityError("books manifest has no books object")
    sources = metadata.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise CorpusIntegrityError("D1.11 must pin exactly four upstream source files")
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise CorpusIntegrityError("D1.11 source entry must be an object")
        book_id = source.get("book_id")
        upstream = books.get(book_id)
        if not isinstance(book_id, str) or not isinstance(upstream, dict):
            raise CorpusIntegrityError(f"source book is missing from books manifest: {book_id}")
        if book_id in source_by_id:
            raise CorpusIntegrityError(f"duplicate D1.11 source book: {book_id}")
        for field in (
            "source_repository",
            "source_ref",
            "license",
            "raw_content_sha384_base64",
            "archive_url",
        ):
            if source.get(field) != upstream.get(field):
                raise CorpusIntegrityError(f"source metadata mismatch for {book_id}: {field}")
        source_by_id[book_id] = source
    if len(source_by_id) != len(sources):
        raise CorpusIntegrityError("D1.11 source book IDs are not unique")

    cases = _parse_cases(corpus_path)
    if len(cases) != 32:
        raise CorpusIntegrityError(f"D1.11 case count changed: expected 32, got {len(cases)}")
    expected_groups = {
        "closedpos": 8,
        "stalemate-stress": 8,
        "endgames-a": 8,
        "endgames-cdb": 8,
    }
    actual_groups = Counter(case.group for case in cases)
    if dict(actual_groups) != expected_groups:
        raise CorpusIntegrityError(
            f"D1.11 group counts changed: expected {expected_groups}, got {dict(actual_groups)}"
        )
    seen_sources: set[tuple[str, int]] = set()
    for case in cases:
        if case.source_book_id not in source_by_id:
            raise CorpusIntegrityError(f"{case.case_id}: source book is not pinned")
        key = (case.source_book_id, case.source_line)
        if key in seen_sources:
            raise CorpusIntegrityError(f"duplicate upstream source line: {key}")
        seen_sources.add(key)

    return cases, {
        "corpus_id": metadata["corpus_id"],
        "snapshot_sha256": actual_hash,
        "license": metadata["license"],
        "source_refs": {
            book_id: {
                "source_repository": source["source_repository"],
                "source_ref": source["source_ref"],
                "archive_url": source["archive_url"],
                "license": source["license"],
                "raw_content_sha384_base64": source["raw_content_sha384_base64"],
            }
            for book_id, source in source_by_id.items()
        },
        "case_count": len(cases),
        "group_counts": dict(actual_groups),
    }


def parse_score(tokens: list[str]) -> Optional[Score]:
    try:
        score_index = tokens.index("score")
    except ValueError:
        return None
    if score_index + 2 >= len(tokens):
        return None
    kind = tokens[score_index + 1]
    if kind not in {"cp", "mate"}:
        return None
    try:
        value = int(tokens[score_index + 2])
    except ValueError:
        return None
    return Score(kind, value)


def parse_info(line: str) -> Optional[tuple[int, Score, tuple[str, ...]]]:
    tokens = line.split()
    try:
        depth_index = tokens.index("depth")
        depth = int(tokens[depth_index + 1])
    except (ValueError, IndexError):
        return None
    score = parse_score(tokens)
    if score is None:
        return None
    try:
        pv_index = tokens.index("pv")
    except ValueError:
        pv: tuple[str, ...] = ()
    else:
        pv = tuple(tokens[pv_index + 1 :])
    return depth, score, pv


def score_rank(score: Score) -> int:
    if score.kind == "mate":
        return 4 if score.value > 0 else 0
    if score.value >= 300:
        return 3
    if score.value >= -150:
        return 2
    return 1


def _score_class_matches(board: chess.Board, score: Score, score_class: str) -> bool:
    if score_class == "terminal-mate":
        return board.is_checkmate()
    if score_class == "terminal-draw":
        return board.is_stalemate()
    if score_class == "mate":
        return score.kind == "mate" and score.value > 0
    if score_class == "winning":
        return (score.kind == "mate" and score.value > 0) or (
            score.kind == "cp" and score.value >= 300
        )
    if score_class == "nonlosing":
        return score.kind == "mate" and score.value > 0 or (
            score.kind == "cp" and score.value >= -150
        )
    if score_class == "losing":
        return (score.kind == "mate" and score.value < 0) or (
            score.kind == "cp" and score.value < -150
        )
    return False


class EngineSession:
    def __init__(self, program: Path, profile: str, hash_mb: int, timeout_s: float):
        self.profile = profile
        self.hash_mb = hash_mb
        self.timeout_s = timeout_s
        try:
            resolved_program = program.expanduser().resolve()
            command = (
                [sys.executable, "-u", str(resolved_program), "--profile", profile]
                if resolved_program.suffix.lower() == ".py"
                else [str(resolved_program), "--profile", profile]
            )
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise EngineFailure(f"{profile}: failed to start engine: {exc}") from exc
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stdin = self.process.stdin
        self._stdout_queue: queue.Queue[Optional[str]] = queue.Queue(READER_QUEUE_SIZE)
        self._stderr_queue: queue.Queue[Optional[str]] = queue.Queue(READER_QUEUE_SIZE)
        self._recent_stdout: deque[str] = deque(maxlen=RECENT_OUTPUT_LINES)
        self._recent_stderr: deque[str] = deque(maxlen=RECENT_OUTPUT_LINES)
        self._threads = [
            threading.Thread(
                target=self._read_pipe,
                args=(self.process.stdout, self._stdout_queue),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_pipe,
                args=(self.process.stderr, self._stderr_queue),
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _read_pipe(stream: Any, output: queue.Queue[Optional[str]]) -> None:
        for raw in iter(stream.readline, ""):
            line = raw.rstrip("\r\n")
            try:
                output.put_nowait(line)
            except queue.Full:
                try:
                    output.get_nowait()
                except queue.Empty:
                    pass
                try:
                    output.put_nowait(line)
                except queue.Full:
                    pass
        try:
            output.put_nowait(None)
        except queue.Full:
            pass

    def _drain_stderr(self) -> None:
        while True:
            try:
                line = self._stderr_queue.get_nowait()
            except queue.Empty:
                return
            if line is not None:
                self._recent_stderr.append(line)

    def _context(self) -> str:
        self._drain_stderr()
        return (
            f"profile={self.profile} stdout={list(self._recent_stdout)!r} "
            f"stderr={list(self._recent_stderr)!r}"
        )

    def _send(self, line: str) -> None:
        if self.process.poll() is not None:
            raise EngineFailure(f"{self.profile}: process exited before {line!r}; {self._context()}")
        try:
            self._stdin.write(line + "\n")
            self._stdin.flush()
        except OSError as exc:
            raise EngineFailure(f"{self.profile}: send {line!r} failed: {exc}; {self._context()}") from exc

    def _readline(self, deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineFailure(f"{self.profile}: total deadline expired; {self._context()}")
        try:
            line = self._stdout_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise EngineFailure(f"{self.profile}: total deadline expired; {self._context()}") from exc
        if line is None:
            raise EngineFailure(f"{self.profile}: stdout closed; {self._context()}")
        self._recent_stdout.append(line)
        self._drain_stderr()
        return line

    def _read_until(self, prefix: str, timeout_s: float) -> list[str]:
        deadline = time.monotonic() + timeout_s
        lines: list[str] = []
        while True:
            line = self._readline(deadline)
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def handshake(self) -> None:
        self._send("uci")
        lines = self._read_until("uciok", self.timeout_s)
        identity = f"info string search profile {self.profile}"
        if identity not in lines:
            raise EngineFailure(f"{self.profile}: missing profile identity; {self._context()}")
        self._send(f"setoption name Hash value {self.hash_mb}")
        self._send("isready")
        self._read_until("readyok", self.timeout_s)

    def search(self, case: Case) -> SearchResult:
        board = chess.Board(case.fen)
        self._send(f"position fen {case.fen}")
        self._send(f"go depth {case.depth}")
        deadline = time.monotonic() + self.timeout_s
        highest: Optional[tuple[int, Score, tuple[str, ...]]] = None
        while True:
            line = self._readline(deadline)
            if line.startswith("info "):
                info = parse_info(line)
                if info is not None:
                    if highest is None or info[0] > highest[0] or (
                        info[0] == highest[0] and info[1] is not None
                    ):
                        highest = info
            if line.startswith("bestmove "):
                fields = line.split()
                if len(fields) < 2:
                    raise EngineFailure(f"{self.profile}: malformed bestmove; {self._context()}")
                bestmove = fields[1]
                break
        if highest is None:
            if bestmove != "0000" or not board.is_game_over(claim_draw=False):
                raise EngineFailure(f"{self.profile}: no scored info line; {self._context()}")
            terminal_score = Score("mate", 1) if board.is_checkmate() else Score("cp", 0)
            result = SearchResult(bestmove, 0, terminal_score, ())
        else:
            result = SearchResult(bestmove, highest[0], highest[1], highest[2])
        if result.completed_depth < case.depth and result.bestmove != "0000":
            raise EngineFailure(
                f"{self.profile}: completed depth {result.completed_depth} < {case.depth}; {self._context()}"
            )
        self._validate_result(board, result)
        self._validate_case_contract(board, case, result)
        return result

    def search_movetime(self, case: Case, movetime_ms: int) -> TimedSearchResult:
        """Run a fixed-time search and return observable UCI progress metrics.

        The engine's current UCI protocol exposes total nodes, but not the
        diagnostic qsearch counter.  The latter is therefore intentionally
        returned as ``None`` rather than inferred or estimated.
        """

        if movetime_ms <= 0:
            raise ValueError("movetime_ms must be positive")
        board = chess.Board(case.fen)
        self._send(f"position fen {case.fen}")
        started = time.perf_counter()
        self._send(f"go movetime {movetime_ms}")
        deadline = time.monotonic() + self.timeout_s
        highest: Optional[tuple[int, Score, tuple[str, ...]]] = None
        latest_nodes: Optional[int] = None
        while True:
            line = self._readline(deadline)
            if line.startswith("info "):
                info = parse_info(line)
                if info is not None:
                    if highest is None or info[0] >= highest[0]:
                        highest = info
                    tokens = line.split()
                    try:
                        node_index = tokens.index("nodes")
                        parsed_nodes = int(tokens[node_index + 1])
                    except (ValueError, IndexError):
                        parsed_nodes = None
                    if parsed_nodes is not None:
                        latest_nodes = max(latest_nodes or 0, parsed_nodes)
            if line.startswith("bestmove "):
                fields = line.split()
                if len(fields) < 2:
                    raise EngineFailure(f"{self.profile}: malformed bestmove; {self._context()}")
                bestmove = fields[1]
                break
        if highest is None:
            if bestmove != "0000" or not board.is_game_over(claim_draw=False):
                raise EngineFailure(f"{self.profile}: no scored info line; {self._context()}")
            terminal_score = Score("mate", 1) if board.is_checkmate() else Score("cp", 0)
            result = SearchResult(bestmove, 0, terminal_score, ())
        else:
            result = SearchResult(bestmove, highest[0], highest[1], highest[2])
        self._validate_result(board, result)
        self._validate_case_contract(board, case, result)
        return TimedSearchResult(
            bestmove=result.bestmove,
            completed_depth=result.completed_depth,
            score=result.score,
            pv=result.pv,
            nodes=latest_nodes,
            qsearch_nodes=None,
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    def _validate_result(self, board: chess.Board, result: SearchResult) -> None:
        if result.bestmove == "0000":
            if any(board.legal_moves):
                raise EngineFailure(f"{self.profile}: bestmove 0000 with legal moves; {self._context()}")
            if not board.is_checkmate() and not board.is_stalemate():
                raise EngineFailure(f"{self.profile}: 0000 without terminal board; {self._context()}")
            return
        try:
            move = board.parse_uci(result.bestmove)
        except ValueError as exc:
            raise EngineFailure(f"{self.profile}: illegal bestmove {result.bestmove}; {self._context()}") from exc
        if not result.pv or result.pv[0] != result.bestmove:
            raise EngineFailure(f"{self.profile}: PV does not start with bestmove; {self._context()}")
        replay = board.copy(stack=False)
        for move_text in result.pv:
            try:
                replay.push_uci(move_text)
            except ValueError as exc:
                raise EngineFailure(f"{self.profile}: illegal PV move {move_text}; {self._context()}") from exc
        if move not in board.legal_moves:
            raise EngineFailure(f"{self.profile}: bestmove is not legal; {self._context()}")
        if result.score.kind == "mate" and result.score.value == 0:
            raise EngineFailure(f"{self.profile}: mate score has zero distance; {self._context()}")

    def _validate_case_contract(self, board: chess.Board, case: Case, result: SearchResult) -> None:
        if not case.allowed_wildcard and case.allowed_moves and result.bestmove not in case.allowed_moves:
            raise EngineFailure(
                f"{self.profile}: bestmove {result.bestmove} is not allowed for {case.case_id}; "
                f"{self._context()}"
            )
        if case.score_class and not _score_class_matches(board, result.score, case.score_class):
            raise EngineFailure(
                f"{self.profile}: score class {case.score_class} failed for {case.case_id}; "
                f"{self._context()}"
            )

    def close(self, require_success: bool = True) -> None:
        cleanup_error: Optional[BaseException] = None
        if self.process.poll() is None:
            try:
                self._stdin.write("stop\nquit\n")
                self._stdin.flush()
            except OSError as exc:
                cleanup_error = exc
            try:
                self.process.wait(timeout=min(self.timeout_s, 5.0))
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    self.process.wait()
                except (OSError, ValueError) as exc:
                    cleanup_error = cleanup_error or exc
        else:
            try:
                self.process.wait(timeout=0)
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_error = cleanup_error or exc
        self._drain_stderr()
        for thread in self._threads:
            thread.join(timeout=1.0)
        for stream in (self._stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError) as exc:
                    cleanup_error = cleanup_error or exc
        if require_success:
            returncode = self.process.returncode
            if returncode != 0:
                raise EngineFailure(
                    f"{self.profile}: engine exited with code {returncode}; {self._context()}"
                )
            if cleanup_error is not None:
                raise EngineFailure(
                    f"{self.profile}: engine cleanup failed: {cleanup_error}; {self._context()}"
                )

    def __enter__(self) -> "EngineSession":
        return self

    def __exit__(self, exc_type: Any, _value: Any, _traceback: Any) -> None:
        if exc_type is None:
            self.close(require_success=True)
        else:
            try:
                self.close(require_success=False)
            except BaseException:
                pass


def run_corpus(
    engine: Path,
    corpus_path: Path = DEFAULT_CORPUS,
    sources_path: Path = DEFAULT_SOURCES,
    books_manifest_path: Path = DEFAULT_BOOKS_MANIFEST,
    report_path: Optional[Path] = None,
    hash_mb: int = 16,
    timeout_s: float = 30.0,
    baseline_profile: str = "current",
    candidate_profile: str = "current-qsearch-pruning",
    corpus_kind: str = "external",
    depth_override: Optional[int] = None,
) -> dict[str, Any]:
    started = time.time()
    report: dict[str, Any] = {
        "decision": "FAIL",
        "corpus": None,
        "profiles": {"baseline": baseline_profile, "candidate": candidate_profile},
        "resources": {
            "hash_mb": hash_mb,
            "timeout_s": timeout_s,
            "corpus_kind": corpus_kind,
            "depth_override": depth_override,
        },
        "cases": [],
        "group_stats": {},
        "searches_expected": None,
        "searches_completed": 0,
        "integrity_errors": [],
        "elapsed_s": None,
    }
    try:
        if corpus_kind == "external":
            cases, corpus_meta = load_corpus(corpus_path, sources_path, books_manifest_path)
        elif corpus_kind == "project":
            cases, corpus_meta = load_project_corpus(corpus_path)
        else:
            raise CorpusIntegrityError(f"unknown corpus kind: {corpus_kind}")
        source_cases = cases
        if depth_override is not None:
            if depth_override <= 0:
                raise CorpusIntegrityError("depth_override must be positive")
            cases = [replace(case, depth=depth_override) for case in cases]
            corpus_meta = dict(corpus_meta)
            corpus_meta["source_depths"] = sorted({case.depth for case in source_cases})
            corpus_meta["effective_depth"] = depth_override
        report["corpus"] = corpus_meta
        report["searches_expected"] = len(cases) * 2
    except CorpusIntegrityError as exc:
        report["integrity_errors"].append(str(exc))
        report["elapsed_s"] = round(time.time() - started, 3)
        if report_path:
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    for case in cases:
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "group": case.group,
            "source_depth": next(
                source_case.depth for source_case in source_cases if source_case.case_id == case.case_id
            ),
            "depth": case.depth,
            "source_book_id": case.source_book_id,
            "source_line": case.source_line,
            "status": "FAIL",
            "baseline": None,
            "candidate": None,
            "failure": None,
        }
        results: dict[str, SearchResult] = {}
        for role, profile in (("baseline", baseline_profile), ("candidate", candidate_profile)):
            try:
                with EngineSession(engine, profile, hash_mb, timeout_s) as session:
                    session.handshake()
                    results[role] = session.search(case)
                row[role] = asdict(results[role])
                report["searches_completed"] += 1
            except (EngineFailure, OSError, ValueError) as exc:
                row["failure"] = f"{role}: {exc}"
                break
        if row["failure"] is None:
            baseline = results["baseline"]
            candidate = results["candidate"]
            if score_rank(candidate.score) < score_rank(baseline.score):
                row["failure"] = (
                    f"candidate score rank {score_rank(candidate.score)} below baseline "
                    f"{score_rank(baseline.score)}"
                )
            else:
                row["status"] = "PASS"
        report["cases"].append(row)

    for group in sorted({case.group for case in cases}):
        group_rows = [row for row in report["cases"] if row["group"] == group]
        report["group_stats"][group] = {
            "cases": len(group_rows),
            "passed": sum(row["status"] == "PASS" for row in group_rows),
            "failed": sum(row["status"] == "FAIL" for row in group_rows),
        }
    report["decision"] = "PASS" if all(row["status"] == "PASS" for row in report["cases"]) else "FAIL"
    report["elapsed_s"] = round(time.time() - started, 3)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--books-manifest", type=Path, default=DEFAULT_BOOKS_MANIFEST)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--baseline-profile", default="current")
    parser.add_argument("--candidate-profile", default="current-qsearch-pruning")
    parser.add_argument("--corpus-kind", choices=("external", "project"), default="external")
    parser.add_argument("--depth-override", type=int, default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hash_mb < 1 or args.timeout_s <= 0:
        print("FAIL: hash and timeout must be positive")
        return 1
    report = run_corpus(
        engine=args.engine,
        corpus_path=args.corpus,
        sources_path=args.sources,
        books_manifest_path=args.books_manifest,
        report_path=args.report,
        hash_mb=args.hash_mb,
        timeout_s=args.timeout_s,
        baseline_profile=args.baseline_profile,
        candidate_profile=args.candidate_profile,
        corpus_kind=args.corpus_kind,
        depth_override=args.depth_override,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
