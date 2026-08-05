"""Runtime configuration loaded from environment variables.

All settings are read from the environment (``/etc/chessarena/chessarena.env``
on the server) so the API process, the worker process and management scripts
share one source of truth.  Every setting has a local default so the test
suite can run without any server-side files.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Fixed time-control presets (section 6 of the spec).
# Only these values may be passed to cutechess; arbitrary TC strings are never
# accepted from the API.
# ---------------------------------------------------------------------------
TIME_CONTROLS: Dict[str, Dict[str, str]] = {
    "bullet_1_0": {
        "label": "Bullet 1+0",
        "cutechess_tc": "60",
    },
    "blitz_3_2": {
        "label": "Blitz 3+2",
        "cutechess_tc": "180+2",
    },
    "rapid_5_3": {
        "label": "Rapid 5+3",
        "cutechess_tc": "300+3",
    },
}

# Player name labels used on the cutechess command line and in PGN headers.
# Engine A is the "candidate"; Engine B is the "baseline".
ENGINE_A_NAME = "EngineA"
ENGINE_B_NAME = "EngineB"


@dataclass(frozen=True)
class Settings:
    # Database
    db_url: str = os.environ.get(
        "ARENA_DB_URL", "sqlite:////var/lib/chessarena/arena.db"
    )

    # Directory layout
    run_root: Path = Path(
        os.environ.get("ARENA_RUN_ROOT", "/var/lib/chessarena/runs")
    )
    build_root: Path = Path(
        os.environ.get("ARENA_BUILD_ROOT", "/opt/chessarena/builds")
    )
    opening_root: Path = Path(
        os.environ.get("ARENA_OPENING_ROOT", "/opt/chessarena/openings")
    )

    # Cutechess binary
    cutechess: Path = Path(
        os.environ.get("ARENA_CUTECHESS", "/usr/bin/cutechess-cli")
    )

    # Fixed production constraints (section 2.3) - not user configurable via API
    max_concurrency: int = int(os.environ.get("ARENA_MAX_CONCURRENCY", "1"))
    hash_mb: int = int(os.environ.get("ARENA_HASH_MB", "32"))
    threads: int = int(os.environ.get("ARENA_THREADS", "1"))

    # URL layout behind the reverse proxy
    base_path: str = os.environ.get("ARENA_BASE_PATH", "/chessarena")
    public_url: str = os.environ.get(
        "ARENA_PUBLIC_URL", "https://pearllover.site/chessarena"
    )

    # Worker behavior
    worker_poll_seconds: float = float(
        os.environ.get("ARENA_WORKER_POLL_SECONDS", "2.0")
    )
    worker_heartbeat_seconds: float = float(
        os.environ.get("ARENA_WORKER_HEARTBEAT_SECONDS", "2.0")
    )
    # A heartbeat older than this makes /health report the worker as stale.
    worker_stale_seconds: float = float(
        os.environ.get("ARENA_WORKER_STALE_SECONDS", "15.0")
    )
    # How long the worker waits for cutechess to exit after SIGTERM before
    # sending SIGKILL (section 19).
    shutdown_grace_seconds: float = float(
        os.environ.get("ARENA_SHUTDOWN_GRACE_SECONDS", "15.0")
    )

    # Logging
    log_level: str = os.environ.get("ARENA_LOG_LEVEL", "INFO")

    # P2.4: CSRF protection secret.  Provide ARENA_CSRF_SECRET in production
    # so the token survives API restarts; the random default is fine for a
    # single-process local instance.
    csrf_secret: str = field(
        default_factory=lambda: os.environ.get("ARENA_CSRF_SECRET")
        or secrets.token_hex(32)
    )

    @property
    def csrf_token(self) -> str:
        return hmac.new(
            self.csrf_secret.encode(), b"chessarena-csrf", hashlib.sha256
        ).hexdigest()

    # Stderr whitelist: substrings that are acceptable in a cutechess child's
    # stderr during a pair run.  Anything else fails verification.
    stderr_whitelist: tuple = field(
        default=(
            "info string",
            "Searching:",
            "Time control:",
            "Playing standard chess",
            "game",  # e.g. "Finished game 1"
        )
    )

    @property
    def api_host(self) -> str:
        return "127.0.0.1"

    @property
    def api_port(self) -> int:
        return int(os.environ.get("ARENA_API_PORT", "8787"))


def get_settings() -> Settings:
    return Settings()
