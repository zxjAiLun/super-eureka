#!/usr/bin/env python3
"""Re-probe UCI capabilities for an existing EngineBuild and backfill its
``uci_options_schema`` (P4.F1 B3b).

Migration 0006 leaves pre-existing builds with ``uci_options_schema=NULL``;
without backfill, the capability-aware runtime would silently omit
Hash/Threads/etc. even though the frozen snapshot expects them.  This script
is the explicit, auditable backfill path:

1. loads the build's recorded binary_path,
2. recomputes the binary SHA-256 and requires it to equal the immutable
   ``binary_sha256`` (fail closed on mismatch),
3. runs a real UCI handshake,
4. writes ``uci_options_schema`` (identity columns are never modified).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Make ``arena`` importable when run from a source checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chessarena.config import get_settings  # noqa: E402
from chessarena.db import make_engine, make_session_factory  # noqa: E402
from chessarena.models import EngineBuild  # noqa: E402
from chessarena.services.uci_probe import probe_uci  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_id")
    args = parser.parse_args()

    settings = get_settings()
    engine = make_engine(settings.db_url)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        build = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == args.build_id)
            .first()
        )
        if build is None:
            sys.exit(f"error: no build with id {args.build_id!r}")

        binary = Path(build.binary_path)
        if not binary.is_file():
            sys.exit(f"error: binary not found: {binary}")

        actual = sha256_file(binary)
        if actual != build.binary_sha256:
            sys.exit(
                "error: binary SHA mismatch (fail closed) — "
                f"recorded {build.binary_sha256}, actual {actual}"
            )

        probe = probe_uci(binary)
        schema = {
            name: {
                "name": opt.name,
                "type": opt.type,
                "default": opt.default,
                "min": opt.min,
                "max": opt.max,
                "vars": list(opt.vars),
            }
            for name, opt in probe.options.items()
        }
        build.uci_options_schema = schema
        session.commit()
        print(f"backfilled {build.build_id}")
        print(f"  uci id name: {probe.id_name}")
        print(f"  uci options: {len(probe.options)}")
        print(f"  binary sha256: {build.binary_sha256} (unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
