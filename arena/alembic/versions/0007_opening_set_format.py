"""opening_sets.format/source

Revision ID: 0007_opening_set_format
Revises: 0006_uci_options_schema
Create Date: 2026-08-08

Generalizes OpeningSet to express book format (pgn|epd) and source
provenance.  Existing rows default to the legacy "epd" format.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_opening_set_format"
down_revision = "0006_uci_options_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "opening_sets",
        sa.Column("format", sa.String(), nullable=False, server_default="epd"),
    )
    op.add_column(
        "opening_sets",
        sa.Column("source", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("opening_sets", "source")
    op.drop_column("opening_sets", "format")
