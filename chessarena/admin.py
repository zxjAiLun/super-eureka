"""Management commands (section 21).

Usage:
    python -m chessarena.admin disk-usage
    python -m chessarena.admin archive-tournament <tournament_id>

These are the only supported disk administration operations in v1; the worker
never deletes failed or interrupted attempt artifacts.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tarfile
from pathlib import Path

from .config import get_settings
from .db import make_engine, make_session_factory
from .models import Tournament
from .services import artifacts

logger = logging.getLogger("chessarena.admin")


def _disk_usage() -> int:
    run_root = artifacts.get_run_root()
    total_bytes = 0
    if run_root.exists():
        for path in run_root.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size
    mb = total_bytes / (1024 * 1024)
    print(f"run_root: {run_root}")
    print(f"total bytes: {total_bytes} ({mb:.1f} MiB)")
    if run_root.exists():
        tournaments = sorted(p.name for p in run_root.iterdir() if p.is_dir())
        for name in tournaments:
            size = sum(
                f.stat().st_size for f in (run_root / name).rglob("*") if f.is_file()
            )
            print(f"  {name}: {size / (1024 * 1024):.1f} MiB")
    return 0


def _archive_tournament(tournament_id: str, session_factory) -> int:
    with session_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        if tournament is None:
            print(f"tournament not found: {tournament_id}", file=sys.stderr)
            return 2
        status = tournament.status
    run_dir = artifacts.tournament_run_dir(tournament_id)
    if not run_dir.exists():
        print(f"run directory not found: {run_dir}", file=sys.stderr)
        return 2
    archive = run_dir.parent / f"{tournament_id}.tar.zst"
    print(f"archiving {run_dir} -> {archive}")
    # tar with zstd compression; fall back to gzip when zstd is unavailable.
    compressor = (
        tarfile.ZSTD_FILE_FORMAT
        if hasattr(tarfile, "ZSTD_FILE_FORMAT")
        else tarfile.GZIP_COMPRESSED
    )
    with tarfile.open(archive, "w:zst" if compressor == tarfile.ZSTD_FILE_FORMAT else "w:gz") as tf:
        tf.add(run_dir, arcname=tournament_id, recursive=True)
    size = archive.stat().st_size
    print(f"created {archive} ({size / (1024 * 1024):.1f} MiB)")
    print(f"tournament status: {status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    settings = get_settings()
    artifacts.configure_artifact_service(settings)
    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)

    if not argv:
        print(__doc__)
        return 1
    command = argv[0]
    if command == "disk-usage":
        return _disk_usage()
    if command == "archive-tournament":
        if len(argv) < 2:
            print("usage: archive-tournament <tournament_id>", file=sys.stderr)
            return 1
        return _archive_tournament(argv[1], session_factory)
    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
