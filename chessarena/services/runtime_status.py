"""Live pair/game runtime status derived from the cutechess run artifacts.

Authoritative ``Game`` rows are only created once a whole pair finishes and
the verifier passes (pair-atomic scoring).  While cutechess is still
running, the admin UI shows live *unverified* progress by parsing the run's
``stdout.log``:

    Started game 1 of 2 (A vs B)
    Finished game 1 (A vs B): 1-0 {White mates}
    Started game 2 of 2 (B vs A)
    Finished game 2 (B vs A): 0-1 {Black mates}

This module never writes Game rows or tournament scores; it only reads
artifacts for display.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

_STARTED_RE = re.compile(r"^Started game (\d+) of (\d+)")
_FINISHED_RE = re.compile(r"^Finished game (\d+) \((.+?)\): (\S+)")


def derive_runtime_status(
    run_dir: Path, *, total_games: int = 2
) -> dict:
    """Parse a pair run directory's stdout.log into a runtime status.

    Returns a dict with keys:
      game_in_pair         1-based game currently started (or None)
      total_games          games per pair
      finished_games_in_pair  number of Finished game lines
      state                "pending" | "game_running" | "pair_done"
      last_result          result of the most recently finished game
      updated_at           mtime of stdout.log (or None if missing)
    """
    status = {
        "game_in_pair": None,
        "total_games": total_games,
        "finished_games_in_pair": 0,
        "state": "pending",
        "last_result": None,
        "updated_at": None,
    }
    stdout = run_dir / "stdout.log"
    if not stdout.is_file():
        return status

    try:
        mtime = stdout.stat().st_mtime
        status["updated_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc)
    except OSError:
        pass

    finished: dict[int, str] = {}
    game_in_pair: int | None = None
    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _STARTED_RE.match(line)
        if m:
            game_in_pair = int(m.group(1))
            total_games = int(m.group(2))
            continue
        m = _FINISHED_RE.match(line)
        if m:
            finished[int(m.group(1))] = m.group(3)

    status["game_in_pair"] = game_in_pair
    status["total_games"] = total_games
    status["finished_games_in_pair"] = len(finished)
    if finished:
        status["last_result"] = finished[max(finished)]
    if len(finished) >= total_games:
        status["state"] = "pair_done"
    elif game_in_pair is not None:
        status["state"] = "game_running"
    return status
