"""SQLAlchemy engine / session wiring for SQLite with WAL mode.

SQLite is shared by the API process and the worker process, so we enable WAL
(a reader never blocks the writer), a busy timeout, and foreign keys.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from fastapi import Depends
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL, foreign keys and a busy timeout on every SQLite connect."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def make_engine(db_url: str):
    kwargs = {"pool_pre_ping": True}
    if db_url.startswith("sqlite"):
        # SQLite connections are created per-thread by default; FastAPI runs
        # handlers on a thread pool, so allow cross-thread use.  Each request
        # still gets its own Session via get_db().
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(db_url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker) -> Generator[Session, None, None]:
    """Transactional session context that rolls back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db(
    session_factory: sessionmaker = Depends(lambda: _current_session_factory),
) -> Generator[Session, None, None]:
    """FastAPI dependency that yields a request-scoped Session."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_current_session_factory: sessionmaker | None = None


def bind_session_factory(factory: sessionmaker) -> None:
    """Called at app startup so FastAPI dependencies can reach the factory."""
    global _current_session_factory
    _current_session_factory = factory
