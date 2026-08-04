"""initial arena schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-05

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "engine_builds",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("build_id", sa.String(), unique=True, nullable=False),
        sa.Column("engine_name", sa.String(), nullable=False),
        sa.Column("git_sha", sa.String(), nullable=False),
        sa.Column("binary_path", sa.Text(), nullable=False),
        sa.Column("binary_sha256", sa.String(length=64), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("supported_profiles", sa.JSON(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
    )
    op.create_table(
        "opening_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("opening_set_id", sa.String(), unique=True, nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("position_count", sa.Integer(), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
    )
    op.create_table(
        "tournaments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("engine_a_build_id", sa.String(), nullable=False),
        sa.Column("engine_a_profile", sa.String(), nullable=False),
        sa.Column("engine_b_build_id", sa.String(), nullable=False),
        sa.Column("engine_b_profile", sa.String(), nullable=False),
        sa.Column("opening_set_id", sa.String(), nullable=False),
        sa.Column("time_control", sa.String(), nullable=False),
        sa.Column("requested_pairs", sa.Integer(), nullable=False),
        sa.Column("completed_pairs", sa.Integer(), nullable=False),
        sa.Column("candidate_wins", sa.Integer(), nullable=False),
        sa.Column("candidate_losses", sa.Integer(), nullable=False),
        sa.Column("draws", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pause_requested", sa.Boolean(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
    )
    op.create_table(
        "pair_jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "tournament_id",
            sa.String(length=36),
            sa.ForeignKey("tournaments.id"),
            nullable=False,
        ),
        sa.Column("pair_index", sa.Integer(), nullable=False),
        sa.Column("opening_index", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("engine_a_white_game_id", sa.String(), nullable=True),
        sa.Column("engine_a_black_game_id", sa.String(), nullable=True),
        sa.Column("run_directory", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("verification", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_pair_jobs_tournament_id", "pair_jobs", ["tournament_id"]
    )
    op.create_table(
        "games",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "tournament_id",
            sa.String(length=36),
            sa.ForeignKey("tournaments.id"),
            nullable=False,
        ),
        sa.Column(
            "pair_job_id",
            sa.String(length=36),
            sa.ForeignKey("pair_jobs.id"),
            nullable=True,
        ),
        sa.Column("game_number", sa.Integer(), nullable=False),
        sa.Column("white_engine", sa.String(), nullable=False),
        sa.Column("black_engine", sa.String(), nullable=False),
        sa.Column("opening_index", sa.Integer(), nullable=False),
        sa.Column("result", sa.String(), nullable=True),
        sa.Column("termination", sa.String(), nullable=True),
        sa.Column("pgn_path", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_games_tournament_id", "games", ["tournament_id"])
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tournament_id",
            sa.String(length=36),
            sa.ForeignKey("tournaments.id"),
            nullable=False,
        ),
        sa.Column("pair_job_id", sa.String(), nullable=True),
        sa.Column("game_id", sa.String(), nullable=True),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_events_tournament_id", "events", ["tournament_id"])
    op.create_table(
        "worker_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("tournament_id", sa.String(), nullable=True),
        sa.Column("pair_job_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("worker_state")
    op.drop_index("ix_events_tournament_id", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_games_tournament_id", table_name="games")
    op.drop_table("games")
    op.drop_index("ix_pair_jobs_tournament_id", table_name="pair_jobs")
    op.drop_table("pair_jobs")
    op.drop_table("tournaments")
    op.drop_table("opening_sets")
    op.drop_table("engine_builds")
