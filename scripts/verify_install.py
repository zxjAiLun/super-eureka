#!/usr/bin/env python3
"""Verify a deployed arena installation (section 20.1).

Checks (exit 0 on success, 1 on any failure):
- cutechess-cli present and answers -version,
- database reachable and migrated,
- at least one enabled build with an existing, SHA-matching binary,
- at least one enabled opening set with an existing, SHA-matching file,
- run directory exists and is writable.

Usage:
    python scripts/verify_install.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessarena.config import get_settings  # noqa: E402
from chessarena.db import make_engine, make_session_factory  # noqa: E402
from chessarena.models import EngineBuild, OpeningSet  # noqa: E402

failures: list[str] = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cutechess(settings) -> None:
    if not settings.cutechess.exists():
        failures.append(f"cutechess-cli missing: {settings.cutechess}")
        return
    try:
        result = subprocess.run(
            [str(settings.cutechess), "-version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            failures.append("cutechess-cli -version returned non-zero")
        else:
            print(f"cutechess-cli: {(result.stdout or result.stderr).splitlines()[0]}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        failures.append(f"cutechess-cli check failed: {exc}")


def check_database(session_factory) -> None:
    try:
        with session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        failures.append(f"database unreachable: {exc}")
        return
    builds = (
        session_factory()
        .query(EngineBuild)
        .filter(EngineBuild.enabled.is_(True))
        .count()
    )
    openings = (
        session_factory()
        .query(OpeningSet)
        .filter(OpeningSet.enabled.is_(True))
        .count()
    )
    print(f"database ok: {builds} enabled builds, {openings} enabled opening sets")
    if builds == 0:
        failures.append("no enabled engine builds registered")
    if openings == 0:
        failures.append("no enabled opening sets registered")


def check_artifacts(session_factory) -> None:
    with session_factory() as session:
        for build in session.query(EngineBuild).filter(EngineBuild.enabled.is_(True)):
            path = Path(build.binary_path)
            if not path.exists():
                failures.append(f"build {build.build_id}: binary missing {path}")
                continue
            if sha256_file(path) != build.binary_sha256:
                failures.append(f"build {build.build_id}: binary SHA mismatch")
            if not os.access(path, os.X_OK):
                failures.append(f"build {build.build_id}: binary not executable")
        for opening in session.query(OpeningSet).filter(OpeningSet.enabled.is_(True)):
            path = Path(opening.file_path)
            if not path.exists():
                failures.append(
                    f"opening set {opening.opening_set_id}: file missing {path}"
                )
                continue
            if sha256_file(path) != opening.sha256:
                failures.append(
                    f"opening set {opening.opening_set_id}: SHA mismatch"
                )


def check_dirs(settings) -> None:
    for label, path in (
        ("run_root", settings.run_root),
        ("build_root", settings.build_root),
        ("opening_root", settings.opening_root),
    ):
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            failures.append(f"{label} not writable: {path}")
        else:
            print(f"{label} ok: {path}")


def main() -> int:
    settings = get_settings()
    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)

    check_cutechess(settings)
    check_database(session_factory)
    check_artifacts(session_factory)
    check_dirs(settings)

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\ninstall OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
