"""engine_presets + tournament preset ids

Revision ID: 0005_engine_presets
Revises: 0004_return_code_attempt
Create Date: 2026-08-06
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op

revision = "0005_engine_presets"
down_revision = "0004_return_code_attempt"
branch_labels = None
depends_on = None

PRODUCTION_PRESET = "chessengine-production"
LEGACY_PRESET = "chessengine-legacy-current"


def _profile_preset(profile: str) -> str | None:
    if profile == "current-final":
        return PRODUCTION_PRESET
    if profile == "current":
        return LEGACY_PRESET
    return None


def upgrade() -> None:
    op.create_table(
        "engine_presets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("preset_id", sa.String(), nullable=False, unique=True),
        sa.Column("build_id", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("command_args", sa.JSON(), nullable=False),
        sa.Column("uci_options", sa.JSON(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("public_visible", sa.Boolean(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.add_column(
        "tournaments",
        sa.Column("engine_a_preset_id", sa.String(), nullable=True),
    )
    op.add_column(
        "tournaments",
        sa.Column("engine_b_preset_id", sa.String(), nullable=True),
    )

    bind = op.get_bind()
    now = datetime.now(timezone.utc).isoformat()

    # Pick the newest enabled build as the target of the stock presets.
    rows = bind.execute(
        sa.text(
            "SELECT build_id, supported_profiles FROM engine_builds "
            "WHERE enabled = 1 ORDER BY created_at DESC, id DESC"
        )
    ).fetchall()
    if rows:
        build_id = rows[0][0]
        profiles = rows[0][1]
        candidates = [
            (
                PRODUCTION_PRESET,
                build_id,
                "ChessEngine Production",
                ["--profile", "current-final"],
                "production",
            ),
            (
                LEGACY_PRESET,
                build_id,
                "ChessEngine Legacy Baseline",
                ["--profile", "current"],
                "legacy",
            ),
        ]
        for preset_id, bid, display, args, category in candidates:
            profile = args[1]
            if profile in (profiles or []):
                bind.execute(
                    sa.text(
                        "INSERT INTO engine_presets "
                        "(preset_id, build_id, display_name, command_args, "
                        "uci_options, category, public_visible, enabled, "
                        "created_at) VALUES (:pid, :bid, :dn, :args, :opts, "
                        ":cat, 1, 1, :ts)"
                    ),
                    {
                        "pid": preset_id,
                        "bid": bid,
                        "dn": display,
                        "args": json.dumps(args),
                        "opts": json.dumps({}),
                        "cat": category,
                        "ts": now,
                    },
                )

    # Backfill historical tournaments from their audit profile columns.
    for col, profile_col in (
        ("engine_a_preset_id", "engine_a_profile"),
        ("engine_b_preset_id", "engine_b_profile"),
    ):
        rows = bind.execute(
            sa.text(
                f"SELECT id, {profile_col} FROM tournaments "
                f"WHERE {col} IS NULL"
            )
        ).fetchall()
        for tid, profile in rows:
            preset = _profile_preset(profile)
            if preset:
                bind.execute(
                    sa.text(
                        f"UPDATE tournaments SET {col} = :pid WHERE id = :tid"
                    ),
                    {"pid": preset, "tid": tid},
                )


def downgrade() -> None:
    op.drop_column("tournaments", "engine_b_preset_id")
    op.drop_column("tournaments", "engine_a_preset_id")
    op.drop_table("engine_presets")
