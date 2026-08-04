#!/usr/bin/env python3
"""Register an immutable engine build directory into the arena database.

Usage:
    python scripts/install_build.py /opt/chessarena/builds/<build_id> [--probe]

Requirements for the build directory (section 7):
    <build_id>/
        engine            (executable, read-only after install)
        manifest.json     (read-only after install)

The script validates the manifest, verifies the binary SHA-256, optionally
probes UCI identity, makes the files read-only, and upserts the build record.
An existing build_id is never silently overwritten; pass --overwrite to force
a re-registration.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

# Make ``arena`` importable when run from a source checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessarena.config import get_settings  # noqa: E402
from chessarena.db import make_engine, make_session_factory  # noqa: E402
from chessarena.models import EngineBuild  # noqa: E402

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "build_id",
    "engine_name",
    "git_sha",
    "binary_sha256",
    "platform",
    "rustc_version",
    "cargo_lock_sha256",
    "supported_profiles",
    "uci_id_name",
    "uci_id_author",
    "created_utc",
}


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_manifest(build_dir: Path) -> dict:
    manifest_path = build_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"error: {manifest_path} not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        sys.exit(f"error: manifest missing keys: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        sys.exit(f"error: unsupported schema_version {manifest['schema_version']}")
    if manifest["build_id"] != build_dir.name:
        sys.exit(
            f"error: manifest build_id {manifest['build_id']!r} does not match "
            f"directory name {build_dir.name!r}"
        )
    if not isinstance(manifest["supported_profiles"], list) or not manifest[
        "supported_profiles"
    ]:
        sys.exit("error: supported_profiles must be a non-empty list")
    return manifest


def probe_uci_identity(binary: Path, expected_name: str, timeout: float = 15.0) -> dict:
    try:
        proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate("uci\nquit\n", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.exit(f"error: UCI probe failed: {exc}")
    id_name = None
    for line in stdout.splitlines():
        if line.startswith("id name "):
            id_name = line[len("id name "):]
    if id_name is None:
        sys.exit(f"error: UCI probe returned no 'id name' (rc={proc.returncode})")
    if expected_name and id_name != expected_name:
        sys.exit(
            f"error: probed id name {id_name!r} does not match manifest "
            f"{expected_name!r}"
        )
    return {"id_name": id_name, "returncode": proc.returncode}


def make_readonly(path: Path) -> None:
    current = stat.S_IMODE(path.stat().st_mode)
    os.chmod(path, current & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--probe", action="store_true", help="run UCI identity probe")
    parser.add_argument(
        "--overwrite", action="store_true", help="allow re-registering an existing build"
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        sys.exit(f"error: build directory not found: {build_dir}")

    manifest = load_and_validate_manifest(build_dir)
    binary = build_dir / "engine"
    if not binary.exists():
        sys.exit(f"error: engine binary not found: {binary}")
    if not os.access(binary, os.X_OK):
        sys.exit(f"error: engine binary is not executable: {binary}")

    actual_sha = sha256_file(binary)
    if actual_sha != manifest["binary_sha256"]:
        sys.exit(
            f"error: binary SHA mismatch: manifest {manifest['binary_sha256']} "
            f"actual {actual_sha}"
        )

    if args.probe:
        probe_uci_identity(binary, manifest["uci_id_name"])

    make_readonly(binary)
    make_readonly(build_dir / "manifest.json")

    settings = get_settings()
    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        existing = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == manifest["build_id"])
            .first()
        )
        if existing is not None and not args.overwrite:
            sys.exit(
                f"error: build {manifest['build_id']} already registered "
                "(use --overwrite to force)"
            )
        if existing is None:
            existing = EngineBuild(
                build_id=manifest["build_id"],
                engine_name=manifest["engine_name"],
                git_sha=manifest["git_sha"],
                binary_path=str(binary),
                binary_sha256=manifest["binary_sha256"],
                platform=manifest["platform"],
                supported_profiles=manifest["supported_profiles"],
                manifest=manifest,
                enabled=True,
            )
            session.add(existing)
        else:
            existing.engine_name = manifest["engine_name"]
            existing.git_sha = manifest["git_sha"]
            existing.binary_path = str(binary)
            existing.binary_sha256 = manifest["binary_sha256"]
            existing.platform = manifest["platform"]
            existing.supported_profiles = manifest["supported_profiles"]
            existing.manifest = manifest
            existing.enabled = True
        session.commit()
    print(f"registered build {manifest['build_id']} -> {binary}")
    print(f"  profiles: {manifest['supported_profiles']}")
    print(f"  binary sha256: {actual_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
