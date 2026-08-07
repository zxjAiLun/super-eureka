#!/usr/bin/env python3
"""Install and register a generic UCI engine build (P4.2 Phase B).

External engines (e.g. Stockfish) do not take the project's ``--profile``
argument; they are configured through UCI options.  This script:

1. verifies the binary SHA-256 against the caller-provided value,
2. runs a real UCI handshake (uci -> id name/option lines -> uciok ->
   isready -> readyok -> quit),
3. requires the listed ``Name:type`` UCI options (e.g.
   UCI_LimitStrength:check UCI_Elo:spin Hash:spin Threads:spin Ponder:check),
4. verifies the requested UCI_Elo values fall within the engine's declared
   min/max,
5. registers an immutable EngineBuild row (idempotent: an existing build_id
   with the same binary SHA is reported as already present; a differing SHA
   is an error).

The binary must already be staged on the server (local-led deployment); this
script never downloads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make ``arena`` importable when run from a source checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessarena.config import get_settings  # noqa: E402
from chessarena.db import make_engine, make_session_factory  # noqa: E402
from chessarena.models import EngineBuild  # noqa: E402
from chessarena.services.uci_probe import (  # noqa: E402
    UciProbeError,
    probe_uci,
    require_option,
)

DEFAULT_REQUIRED_OPTIONS = [
    ("UCI_LimitStrength", "check"),
    ("UCI_Elo", "spin"),
    ("Hash", "spin"),
    ("Threads", "spin"),
    ("Ponder", "check"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--engine-name", required=True)
    parser.add_argument("--binary-name", default="engine")
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--uci-elos", default="1800,2000,2200,2400")
    parser.add_argument(
        "--required-options",
        nargs="+",
        default=[f"{n}:{t}" for n, t in DEFAULT_REQUIRED_OPTIONS],
        help="Name:type pairs to require from the UCI handshake",
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    binary = build_dir / args.binary_name
    if not binary.is_file():
        sys.exit(f"error: engine binary not found: {binary}")
    if not os.access(binary, os.X_OK):
        sys.exit(f"error: engine binary not executable: {binary}")

    actual_sha = sha256_file(binary)
    if actual_sha != args.binary_sha256:
        sys.exit(
            f"error: binary SHA mismatch: manifest {args.binary_sha256} "
            f"actual {actual_sha}"
        )

    engine = make_engine(get_settings().db_url)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        existing = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == args.build_id)
            .first()
        )
        if existing is not None:
            # Idempotent path: read-only verification ONLY.  No writes —
            # the registered build's path identity must match exactly, or
            # this run is pointing at a different build directory.
            if Path(existing.binary_path).resolve() != binary.resolve():
                sys.exit(
                    f"error: build_id {args.build_id} already registered at "
                    f"{existing.binary_path}, not {binary}"
                )
            if existing.binary_sha256 != args.binary_sha256:
                sys.exit(
                    f"error: build_id {args.build_id} already registered with "
                    f"a different binary SHA"
                )
            if sha256_file(Path(existing.binary_path)) != args.binary_sha256:
                sys.exit(
                    f"error: registered binary SHA mismatch on disk for "
                    f"{existing.binary_path}"
                )
            manifest_path = Path(existing.binary_path).parent / "manifest.json"
            if not manifest_path.is_file():
                sys.exit(
                    f"error: manifest.json missing for registered build "
                    f"{args.build_id}"
                )
            print(
                f"build {args.build_id} already registered "
                f"(path, binary SHA and manifest verified; no writes)"
            )
            return 0

        try:
            probe = probe_uci(binary)
        except UciProbeError as exc:
            sys.exit(f"error: UCI probe failed: {exc}")

        for spec in args.required_options:
            name, _, typ = spec.partition(":")
            if not typ:
                sys.exit(
                    f"error: required-options must be Name:type, got {spec!r}"
                )
            require_option(probe, name, typ)

        elo_opt = probe.options.get("UCI_Elo")
        elo_values = [int(v) for v in args.uci_elos.split(",")]
        for value in elo_values:
            if elo_opt is not None:
                if elo_opt.min is not None and value < elo_opt.min:
                    sys.exit(
                        f"error: UCI_Elo {value} below engine minimum "
                        f"{elo_opt.min}"
                    )
                if elo_opt.max is not None and value > elo_opt.max:
                    sys.exit(
                        f"error: UCI_Elo {value} above engine maximum "
                        f"{elo_opt.max}"
                    )

        manifest = {
            "schema_version": 1,
            "build_id": args.build_id,
            "engine_name": args.engine_name,
            "git_sha": "external",
            "binary_sha256": args.binary_sha256,
            "platform": args.platform,
            "rustc_version": "",
            "cargo_lock_sha256": "",
            "supported_profiles": [],
            "uci_id_name": probe.id_name,
            "uci_id_author": "",
            "created_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
        }

        # Filesystem first, DB last (P4.2 repair): a failed DB commit leaves
        # a complete-but-unregistered filesystem, which a re-run can safely
        # register.  The reverse (DB row present but filesystem half-baked)
        # is the unsafe state and is rejected.
        manifest_path = build_dir / "manifest.json"
        manifest_tmp = build_dir / "manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        try:
            with open(manifest_tmp, "rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass
        if manifest_path.exists():
            # Windows refuses to replace a read-only target.
            manifest_path.chmod(0o644)
        os.replace(manifest_tmp, manifest_path)

        binary.chmod(
            binary.stat().st_mode
            & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        )
        manifest_path.chmod(0o444)

        # Post-write re-verification.
        if sha256_file(binary) != args.binary_sha256:
            sys.exit("error: binary SHA changed after install (unexpected)")
        try:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            sys.exit(f"error: manifest.json unreadable after write: {exc}")
        if (
            stored.get("binary_sha256") != args.binary_sha256
            or stored.get("build_id") != args.build_id
        ):
            sys.exit("error: manifest.json content mismatch after write")

        session.add(
            EngineBuild(
                build_id=args.build_id,
                engine_name=args.engine_name,
                git_sha=manifest["git_sha"],
                binary_path=str(binary),
                binary_sha256=args.binary_sha256,
                platform=args.platform,
                supported_profiles=[],
                manifest=manifest,
                enabled=True,
            )
        )
        session.commit()

    print(f"registered external build {args.build_id}")
    print(f"  engine_name: {args.engine_name}")
    print(f"  uci id name: {probe.id_name}")
    print(f"  binary sha256: {args.binary_sha256}")
    print(f"  UCI_Elo range: {elo_opt.min if elo_opt else '?'}..{elo_opt.max if elo_opt else '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
