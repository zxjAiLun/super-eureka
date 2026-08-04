"""Alembic environment.

The database URL comes from ARENA_DB_URL (mirroring chessarena.config.Settings)
so migrations and the app always use the same database.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

# Make the ``arena`` package importable regardless of the CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessarena.db import Base  # noqa: E402
from chessarena import models  # noqa: E402,F401  (register tables)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    return os.environ.get("ARENA_DB_URL", "sqlite:////var/lib/chessarena/arena.db")


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_db_url(), connect_args={"check_same_thread": False})
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
