"""arena control fields: force-cancel flag and pair return code

Revision ID: 0002_arena_control_fields
Revises: 0001_initial
Create Date: 2026-08-05

Adds:
- tournaments.force_cancel_requested (P1.3 cross-process force-cancel flag)
- pair_jobs.return_code (P1.5 cutechess exit code gate)

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_arena_control_fields"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tournaments",
        sa.Column("force_cancel_requested", sa.Boolean(), nullable=False,
                  server_default="0"),
    )
    op.add_column(
        "pair_jobs",
        sa.Column("return_code", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pair_jobs", "return_code")
    op.drop_column("tournaments", "force_cancel_requested")
