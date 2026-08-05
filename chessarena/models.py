"""Database models for the arena (section 9 of the spec).

Design notes:
- UUIDs are stored as TEXT (portable across SQLite).
- ``engine_builds`` and ``opening_sets`` are immutable registries; nothing in
  the runtime mutates the files they point at.
- ``events`` is written from day one so a v2 live-spectating layer can stream
  from it without schema changes.
- ``worker_state`` is an internal single-row heartbeat used by /health.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def coerce_utc(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; normalize them to UTC-aware."""
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def default_uuid() -> str:
    import uuid

    return str(uuid.uuid4())


# Tournament lifecycle (section 10.1)
DRAFT = "DRAFT"
QUEUED = "QUEUED"
RUNNING = "RUNNING"
PAUSING = "PAUSING"
PAUSED = "PAUSED"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

TOURNAMENT_STATUSES = frozenset(
    {DRAFT, QUEUED, RUNNING, PAUSING, PAUSED, COMPLETED, FAILED, CANCELLED}
)

# Allowed transitions (section 10.1)
TOURNAMENT_TRANSITIONS = {
    DRAFT: {QUEUED},
    QUEUED: {RUNNING, CANCELLED},
    RUNNING: {PAUSING, COMPLETED, FAILED, CANCELLED},
    PAUSING: {PAUSED, COMPLETED, FAILED, CANCELLED},
    PAUSED: {QUEUED, CANCELLED},
    COMPLETED: set(),
    FAILED: set(),
    CANCELLED: set(),
}

# Pair job lifecycle (section 10.2)
PENDING = "PENDING"
RUNNING = "RUNNING"
VERIFYING = "VERIFYING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
INTERRUPTED = "INTERRUPTED"

PAIR_STATUSES = frozenset(
    {PENDING, RUNNING, VERIFYING, COMPLETED, FAILED, INTERRUPTED}
)


class EngineBuild(Base):
    __tablename__ = "engine_builds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    build_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    engine_name: Mapped[str] = mapped_column(String, nullable=False)
    git_sha: Mapped[str] = mapped_column(String, nullable=False)
    binary_path: Mapped[str] = mapped_column(Text, nullable=False)
    binary_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    platform: Mapped[str] = mapped_column(String, nullable=False)
    supported_profiles: Mapped[list] = mapped_column(JSON, nullable=False)
    manifest: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )


class OpeningSet(Base):
    __tablename__ = "opening_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    opening_set_id: Mapped[str] = mapped_column(
        String, unique=True, nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    position_count: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )


class Tournament(Base):
    __tablename__ = "tournaments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=default_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default=DRAFT, nullable=False)

    engine_a_build_id: Mapped[str] = mapped_column(String, nullable=False)
    engine_a_profile: Mapped[str] = mapped_column(String, nullable=False)
    engine_b_build_id: Mapped[str] = mapped_column(String, nullable=False)
    engine_b_profile: Mapped[str] = mapped_column(String, nullable=False)
    opening_set_id: Mapped[str] = mapped_column(String, nullable=False)
    time_control: Mapped[str] = mapped_column(String, nullable=False)

    requested_pairs: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_pairs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    candidate_wins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    candidate_losses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    draws: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    pause_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # P1.3: cross-process force-cancel flag.  The API sets it in the database;
    # the worker polls it, kills the process group, and only then marks the
    # tournament CANCELLED.  In-process shared memory is NOT used because the
    # API and worker are separate systemd processes.
    force_cancel_requested: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)

    pair_jobs: Mapped[list["PairJob"]] = relationship(
        back_populates="tournament",
        order_by="PairJob.pair_index",
        cascade="all, delete-orphan",
    )
    games: Mapped[list["Game"]] = relationship(back_populates="tournament")
    events: Mapped[list["Event"]] = relationship(back_populates="tournament")


class PairJob(Base):
    __tablename__ = "pair_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=default_uuid)
    tournament_id: Mapped[str] = mapped_column(
        ForeignKey("tournaments.id"), index=True, nullable=False
    )
    pair_index: Mapped[int] = mapped_column(Integer, nullable=False)
    opening_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, default=PENDING, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    engine_a_white_game_id: Mapped[str | None] = mapped_column(String, nullable=True)
    engine_a_black_game_id: Mapped[str | None] = mapped_column(String, nullable=True)

    run_directory: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # P1.5: the cutechess process exit code for this attempt (None while
    # pending/running).  A non-zero exit code fails the pair and the
    # tournament even when the artifacts look complete.
    return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)

    tournament: Mapped["Tournament"] = relationship(back_populates="pair_jobs")

    # Keep pair jobs ordered per tournament.
    __table_args__ = (
        # A pair can be retried via attempt, but only one live attempt at a
        # time for a given pair index.  Enforced by the worker, not the DB.
        None,
    )


class Game(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=default_uuid)
    tournament_id: Mapped[str] = mapped_column(
        ForeignKey("tournaments.id"), index=True, nullable=False
    )
    pair_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("pair_jobs.id"), nullable=True
    )
    game_number: Mapped[int] = mapped_column(Integer, nullable=False)
    white_engine: Mapped[str] = mapped_column(String, nullable=False)
    black_engine: Mapped[str] = mapped_column(String, nullable=False)
    opening_index: Mapped[int] = mapped_column(Integer, nullable=False)
    result: Mapped[str | None] = mapped_column(String, nullable=True)
    termination: Mapped[str | None] = mapped_column(String, nullable=True)
    pgn_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    tournament: Mapped["Tournament"] = relationship(back_populates="games")
    pair_job: Mapped["PairJob | None"] = relationship()


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_id: Mapped[str] = mapped_column(
        ForeignKey("tournaments.id"), index=True, nullable=False
    )
    pair_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    game_id: Mapped[str | None] = mapped_column(String, nullable=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    tournament: Mapped["Tournament"] = relationship(back_populates="events")


class WorkerState(Base):
    """Single-row heartbeat written by the worker process (internal).

    ``pid`` / ``pid_start_marker`` / ``pid_cmdline`` record the identity of the
    currently supervised cutechess process.  The identity is written at launch
    time (not just on heartbeat) so recovery can safely terminate an orphaned
    process after an abnormal worker death without risking PID reuse.
    """

    __tablename__ = "worker_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String, default="idle", nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tournament_id: Mapped[str | None] = mapped_column(String, nullable=True)
    pair_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    pid_start_marker: Mapped[str | None] = mapped_column(String, nullable=True)
    pid_cmdline: Mapped[str | None] = mapped_column(Text, nullable=True)
