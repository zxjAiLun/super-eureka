"""engine_builds.uci_options_schema

Revision ID: 0006_uci_options_schema
Revises: 0005_engine_presets
Create Date: 2026-08-08

Adds the UCI capability schema column to engine_builds.  Existing builds are
left NULL (they predate capability capture; the registration layer can
re-probe and backfill when needed).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_uci_options_schema"
down_revision = "0005_engine_presets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "engine_builds",
        sa.Column("uci_options_schema", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("engine_builds", "uci_options_schema")
