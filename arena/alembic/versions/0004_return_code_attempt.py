"""pair_jobs.return_code_attempt

Revision ID: 0004_return_code_attempt
Revises: 0003_worker_process_identity
Create Date: 2026-08-05

Adds pair_jobs.return_code_attempt so manager exit evidence can be verified
as belonging to the CURRENT attempt; both return_code and return_code_attempt
are cleared on every retry (P1: no stale exit evidence across attempts).

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_return_code_attempt"
down_revision = "0003_worker_process_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pair_jobs",
        sa.Column("return_code_attempt", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pair_jobs", "return_code_attempt")
