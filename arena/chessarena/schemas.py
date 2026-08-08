"""Pydantic request / response schemas for the arena API (section 16)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import TIME_CONTROLS

ALLOWED_TIME_CONTROLS = set(TIME_CONTROLS.keys())


# ---------------------------------------------------------------------------
# Builds / opening sets
# ---------------------------------------------------------------------------
class BuildOut(BaseModel):
    build_id: str
    engine_name: str
    git_sha: str
    binary_path: str
    binary_sha256: str
    platform: str
    supported_profiles: List[str]
    created_at: datetime
    enabled: bool

    model_config = {"from_attributes": True}


class OpeningSetOut(BaseModel):
    opening_set_id: str
    file_path: str
    sha256: str
    position_count: int
    created_at: datetime
    enabled: bool

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Tournaments
# ---------------------------------------------------------------------------
class EngineRef(BaseModel):
    preset_id: str = Field(min_length=1)


class TournamentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    engine_a: EngineRef
    engine_b: EngineRef
    opening_set_id: str = Field(min_length=1)
    time_control: str
    pairs: int = Field(ge=1)
    allow_intentional_self_play: bool = False


class GameOut(BaseModel):
    id: str
    game_number: int
    white_engine: str
    black_engine: str
    opening_index: int
    result: Optional[str] = None
    termination: Optional[str] = None
    verified: bool
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PairJobOut(BaseModel):
    id: str
    tournament_id: str
    pair_index: int
    opening_index: int
    status: str
    attempt: int
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    failure_reason: Optional[str] = None

    model_config = {"from_attributes": True}


class EventOut(BaseModel):
    id: int
    tournament_id: str
    pair_job_id: Optional[str] = None
    game_id: Optional[str] = None
    event_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = {"from_attributes": True}


class TournamentOut(BaseModel):
    id: str
    name: str
    status: str
    engine_a_build_id: str
    engine_a_profile: str
    engine_b_build_id: str
    engine_b_profile: str
    engine_a_preset_id: Optional[str] = None
    engine_b_preset_id: Optional[str] = None
    opening_set_id: str
    time_control: str
    requested_pairs: int
    completed_pairs: int
    candidate_wins: int
    candidate_losses: int
    draws: int
    score_percent: Optional[float] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    force_cancel_requested: bool = False

    model_config = {"from_attributes": True}


class TournamentDetailOut(TournamentOut):
    pause_requested: bool
    cancel_requested: bool
    pairs: List[PairJobOut] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str
    database: str
    worker_heartbeat: str
    cutechess: str
    active_tournament_id: Optional[str] = None
    # Deployment gate: enabled builds whose capability schema is still NULL
    # (migration 0006 backfill pending).  >0 means the worker must not start
    # new tournaments.
    uci_capability_gap: int = 0


# ---------------------------------------------------------------------------
# Public replay (anonymous, read-only; whitelist fields only)
# ---------------------------------------------------------------------------
class PublicGameOut(BaseModel):
    id: str
    game_number: int
    white_engine: str
    black_engine: str
    opening_index: int
    result: Optional[str] = None
    termination: Optional[str] = None
    finished_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PublicMatchOut(BaseModel):
    id: str
    name: str
    status: str
    time_control: str
    requested_pairs: int
    completed_pairs: int
    candidate_wins: int
    candidate_losses: int
    draws: int
    score_percent: Optional[float] = None
    finished_at: Optional[datetime] = None
    engine_a_label: str
    engine_b_label: str
    opening_set_id: str


class PublicMatchDetailOut(PublicMatchOut):
    games: List[PublicGameOut] = Field(default_factory=list)
