"""worker_state process identity fields

Revision ID: 0003_worker_process_identity
Revises: 0002_arena_control_fields
Create Date: 2026-08-05

Adds worker_state.pid_start_marker and worker_state.pid_cmdline so recovery
can positively identify (and safely terminate) an orphaned cutechess process
after an abnormal worker death, guarded against PID reuse (P1).

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_worker_process_identity"
down_revision = "0002_arena_control_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "worker_state",
        sa.Column("pid_start_marker", sa.String(), nullable=True),
    )
    op.add_column(
        "worker_state",
        sa.Column("pid_cmdline", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("worker_state", "pid_cmdline")
    op.drop_column("worker_state", "pid_start_marker")
