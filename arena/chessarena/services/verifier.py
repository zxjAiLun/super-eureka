"""Strict paired-match verifier (section 14).

The final score never relies on cutechess's stdout alone.  This module parses
the pair's match.pgn, replays every move for legality, enforces strict color
swapping and identical openings, cross-checks the cutechess score line, checks
stdout/stderr, and verifies engine/opening provenance.

On any failure the pair is NOT scored and the whole tournament is marked
FAILED; all artifacts are retained.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import chess
import chess.pgn

from ..config import ENGINE_A_NAME, ENGINE_B_NAME, Settings
from . import artifacts
from . import cutechess as cc

# Substrings that are never acceptable in cutechess stdout (section 14.12).
FORBIDDEN_STDOUT = ("illegal", "crash", "timeout", "forfeit", "fatal", "error")

# Engine display names may contain spaces (e.g. "ChessEngine Production"), so
# the two sides are matched lazily up to "vs" / ":".
_SCORE_LINE_RE = re.compile(
    r"Score of\s+(.+?)\s+vs\s+(.+?):\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
)


class VerificationFailure(Exception):
    """A pair failed verification; the tournament must be failed too."""


# ---------------------------------------------------------------------------
# PGN helpers
# ---------------------------------------------------------------------------
def parse_pgn(path: Path) -> List[chess.pgn.Game]:
    games: List[chess.pgn.Game] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception as exc:
                raise VerificationFailure(f"PGN parse error: {exc}") from exc
            if game is None:
                break
            # read_game is forgiving: illegal/ambiguous tokens are recorded in
            # game.errors.  A strict verifier must reject any of them
            # (section 14.5: every move must be legal).
            if getattr(game, "errors", None):
                first = str(game.errors[0])
                raise VerificationFailure(f"PGN contains invalid move: {first}")
            games.append(game)
    return games


def replay_legal(game: chess.pgn.Game) -> None:
    """Replay all moves from the initial position; raise on any illegal move."""
    fen = game.headers.get("FEN")
    board = chess.Board(fen) if fen else chess.Board()
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise VerificationFailure(
                f"illegal move in PGN: {board.fen()} -> {move.uci()}"
            )
        board.push(move)


def position_key(game: chess.pgn.Game) -> str:
    """Normalized FEN used to compare openings across the two games."""
    fen = game.headers.get("FEN")
    board = chess.Board(fen) if fen else chess.Board()
    return board.fen()


def parse_score_line(stdout_lines: List[str]) -> Dict[str, int] | None:
    """Last 'Score of A vs B: W - L - D ...' line, if present."""
    wins = losses = draws = None
    for line in stdout_lines:
        m = _SCORE_LINE_RE.search(line)
        if m:
            wins, losses, draws = int(m.group(3)), int(m.group(4)), int(m.group(5))
    if wins is None:
        return None
    return {"wins": wins, "losses": losses, "draws": draws}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _side_display_name(snapshot, key: str) -> str:
    """PGN-facing engine name for one side.

    New tournaments freeze the preset's display_name into the snapshot;
    legacy rows fall back to the historical EngineA/EngineB constants.
    """
    side = (snapshot or {}).get(key) or {}
    display = side.get("display_name")
    if display:
        return display
    from ..config import ENGINE_A_NAME, ENGINE_B_NAME

    return ENGINE_A_NAME if key == "engine_a" else ENGINE_B_NAME


def verify_pair(
    settings: Settings,
    *,
    tournament,
    pair_job,
    run_dir: Path,
    engine_a_build,
    engine_b_build,
    opening_set,
) -> Dict[str, Any]:
    """Verify a pair's artifacts.  Returns the verification dict on success.

    Raises VerificationFailure with the reason on any check that fails.
    """
    snapshot = tournament.config_snapshot
    tc_preset = snapshot["time_control"]
    from ..config import TIME_CONTROLS

    cutechess_tc = TIME_CONTROLS[tc_preset]["cutechess_tc"]

    match_pgn = run_dir / "match.pgn"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    command_json = run_dir / "command.json"
    opening_epd = run_dir / "opening.epd"

    if not match_pgn.exists():
        raise VerificationFailure("match.pgn missing")
    if match_pgn.stat().st_size == 0:
        raise VerificationFailure("match.pgn is empty")

    games = parse_pgn(match_pgn)
    if len(games) != 2:
        raise VerificationFailure(f"expected 2 games, found {len(games)}")

    # Strict color swap and identity (section 14.6, 14.8).  PGN engine names
    # come from the preset display names (or the legacy EngineA/EngineB
    # constants for pre-preset tournaments).
    a_name = _side_display_name(snapshot, "engine_a")
    b_name = _side_display_name(snapshot, "engine_b")
    if games[0].headers.get("White") != a_name or games[0].headers.get("Black") != b_name:
        raise VerificationFailure(
            f"game 1 color assignment wrong: White={games[0].headers.get('White')} "
            f"Black={games[0].headers.get('Black')} (expected "
            f"{a_name}/{b_name})"
        )
    if games[1].headers.get("White") != b_name or games[1].headers.get("Black") != a_name:
        raise VerificationFailure(
            f"game 2 color assignment wrong: White={games[1].headers.get('White')} "
            f"Black={games[1].headers.get('Black')} (expected "
            f"{b_name}/{a_name})"
        )

    # Opening position key identical across both games (section 14.7)
    key1 = position_key(games[0])
    key2 = position_key(games[1])
    if key1 != key2:
        raise VerificationFailure("the two games used different opening positions")

    # The position must match the registered opening line for this pair index
    expected_fen = _expected_opening_fen(opening_set, pair_job.opening_index)
    if key1 != expected_fen:
        raise VerificationFailure(
            f"opening position mismatch: pair used {key1}, registered line is "
            f"{expected_fen}"
        )

    # Replay every move for legality (section 14.4-14.5)
    for i, game in enumerate(games, start=1):
        try:
            replay_legal(game)
        except VerificationFailure:
            raise
        except Exception as exc:
            raise VerificationFailure(f"game {i} replay error: {exc}") from exc

    # Time control header (section 14.9)
    for i, game in enumerate(games, start=1):
        if game.headers.get("TimeControl") != cutechess_tc:
            raise VerificationFailure(
                f"game {i} TimeControl '{game.headers.get('TimeControl')}' "
                f"does not match preset '{cutechess_tc}'"
            )

    # Results + termination
    results = [game.headers.get("Result") for game in games]
    for result in results:
        if result not in ("1-0", "0-1", "1/2-1/2"):
            raise VerificationFailure(f"unrecognized result '{result}'")
    terminations = [game.headers.get("Termination") for game in games]

    # Recompute A-perspective W/D/L (section 14.10)
    # Game 0 has A as White (result from White's perspective); game 1 has A as
    # Black (result from Black's perspective).
    computed = {"wins": 0, "losses": 0, "draws": 0}
    for idx, result in enumerate(results):
        if result == "1/2-1/2":
            computed["draws"] += 1
        elif idx == 0:  # A is White
            if result == "1-0":
                computed["wins"] += 1
            else:
                computed["losses"] += 1
        else:  # A is Black
            if result == "0-1":
                computed["wins"] += 1
            else:
                computed["losses"] += 1

    # Compare with cutechess score line (section 14.11)
    stdout_lines = cc.read_output_lines(stdout_log)
    score_line = parse_score_line(stdout_lines)
    if score_line is None:
        raise VerificationFailure("no 'Score of' line found in cutechess stdout")
    if score_line != computed:
        raise VerificationFailure(
            f"cutechess score {score_line} disagrees with recomputed {computed}"
        )

    # stdout forbidden words (section 14.12)
    for line in stdout_lines:
        lower = line.lower()
        if any(word in lower for word in FORBIDDEN_STDOUT):
            raise VerificationFailure(
                f"cutechess stdout contains forbidden term in: {line!r}"
            )

    # stderr whitelist (section 14.13)
    stderr_lines = cc.read_output_lines(stderr_log)
    for line in stderr_lines:
        if not line.strip():
            continue
        if not any(token.lower() in line.lower() for token in settings.stderr_whitelist):
            raise VerificationFailure(
                f"unexpected stderr line: {line!r}"
            )

    # Provenance (section 14.14, P2.3)
    _check_command_provenance(settings, command_json, snapshot, run_dir,
                              engine_a_build, engine_b_build)
    _check_engine_provenance(engine_a_build, snapshot["engine_a"])
    _check_engine_provenance(engine_b_build, snapshot["engine_b"])
    _check_opening_provenance(opening_set, snapshot["opening_set"])
    if not opening_epd.exists():
        raise VerificationFailure("opening.epd missing from pair directory")

    verification = {
        "verified": True,
        "pgn_game_count": len(games),
        "colors": [
            {"white": a_name, "black": b_name},
            {"white": b_name, "black": a_name},
        ],
        "opening_position_key": key1,
        "moves_legal": True,
        "results": results,
        "terminations": terminations,
        "candidate_perspective": computed,
        "cutechess_score_line": score_line,
        "engine_a_binary_sha256": engine_a_build.binary_sha256,
        "engine_b_binary_sha256": engine_b_build.binary_sha256,
        "opening_set_sha256": opening_set.sha256,
        "pair_opening_epd_sha256": artifacts.sha256_file(opening_epd),
        "stdout_sha256": artifacts.sha256_file(stdout_log),
        "stderr_sha256": artifacts.sha256_file(stderr_log),
        "pgn_sha256": artifacts.sha256_file(match_pgn),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return verification


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------
def _expected_opening_fen(opening_set, opening_index: int) -> str:
    path = Path(opening_set.file_path)
    if not path.exists():
        raise VerificationFailure(f"opening set file missing: {path}")
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if opening_index >= len(lines):
        raise VerificationFailure(
            f"opening_index {opening_index} out of range ({len(lines)} lines)"
        )
    return chess.Board(lines[opening_index].split(";")[0].strip()).fen()


def _check_command_provenance(settings, command_json: Path, snapshot, run_dir,
                              engine_a_build, engine_b_build) -> None:
    if not command_json.exists():
        raise VerificationFailure("command.json missing from pair directory")
    try:
        cmd = json.loads(command_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationFailure(f"command.json unreadable: {exc}") from exc

    recorded = cmd.get("argv")
    if not isinstance(recorded, list) or not recorded:
        raise VerificationFailure("command.json has no argv")
    if cmd.get("shell", False):
        raise VerificationFailure("command.json has shell=true (forbidden)")

    # Rebuild the expected argv from the recorded snapshot and the registered
    # engine builds, then compare structurally (P2.3).  This pins the engine
    # binaries, both presets (command_args / uci_options frozen in the
    # snapshot), the time control, Hash/Threads, rounds/repeat, concurrency
    # and the opening/pgn paths exactly.
    from .cutechess import build_pair_command

    from ..config import TIME_CONTROLS

    def _engine_cfg(build, snap) -> dict:
        return {
            "build_id": build.build_id,
            "binary_path": build.binary_path,
            "display_name": snap.get("display_name"),
            "command_args": list(
                snap.get("command_args")
                or ["--profile", snap["profile"]]
            ),
            "uci_options": dict(snap.get("uci_options") or {}),
        }

    expected = build_pair_command(
        settings,
        engine_a=_engine_cfg(engine_a_build, snapshot["engine_a"]),
        engine_b=_engine_cfg(engine_b_build, snapshot["engine_b"]),
        time_control=TIME_CONTROLS[snapshot["time_control"]]["cutechess_tc"],
        hash_mb=snapshot.get("hash_mb", settings.hash_mb),
        opening_epd=run_dir / "opening.epd",
        pgn_out=run_dir / "match.pgn",
        threads=snapshot.get("threads", settings.threads),
    )
    if recorded != expected:
        raise VerificationFailure(
            "recorded cutechess argv does not match the expected command "
            f"(recorded {len(recorded)} args vs expected {len(expected)})"
        )


def _check_engine_provenance(build, snapshot_engine) -> None:
    if build.build_id != snapshot_engine["build_id"]:
        raise VerificationFailure("engine build id mismatch in provenance")
    if build.binary_sha256 != snapshot_engine["binary_sha256"]:
        raise VerificationFailure("engine binary SHA mismatch in provenance")
    if build.git_sha != snapshot_engine["git_sha"]:
        raise VerificationFailure("engine git SHA mismatch in provenance")
    # P2.3: re-hash the actual engine binary now, after the match.
    binary = Path(build.binary_path)
    if not binary.exists():
        raise VerificationFailure(
            f"engine binary missing at verification time: {binary}"
        )
    if artifacts.sha256_file(binary) != build.binary_sha256:
        raise VerificationFailure(
            f"engine binary changed since registration: {binary}"
        )


def _check_opening_provenance(opening_set, snapshot_opening) -> None:
    if opening_set.opening_set_id != snapshot_opening["opening_set_id"]:
        raise VerificationFailure("opening set id mismatch in provenance")
    if opening_set.sha256 != snapshot_opening["sha256"]:
        raise VerificationFailure("opening set SHA mismatch in provenance")
