#!/usr/bin/env python3
"""Register limited-strength Stockfish presets (P4.2 Phase B).

Creates or updates the four Stockfish Limited presets:

    stockfish-limited-1800 / 2000 / 2200 / 2400

Each preset carries only its engine-specific UCI options:

    {"UCI_LimitStrength": true, "UCI_Elo": <n>}

Runtime-reserved options (Hash/Threads/Ponder) are rejected.  The script
re-probes the installed binary to confirm UCI_LimitStrength is a check
option and UCI_Elo is a spin option whose min/max covers all four values.
Idempotent: re-running updates nothing if the presets are unchanged.
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
from chessarena.models import EngineBuild, EnginePreset  # noqa: E402
from chessarena.services.cutechess import validate_preset_options  # noqa: E402
from chessarena.services.uci_probe import (  # noqa: E402
    UciProbeError,
    probe_uci,
    require_option,
)

STOCKFISH_ELOS = [1800, 2000, 2200, 2400]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _preset_for_elo(build_id: str, elo: int) -> dict:
    return {
        "preset_id": f"stockfish-limited-{elo}",
        "build_id": build_id,
        "display_name": f"Stockfish Limited {elo}",
        "command_args": [],
        "uci_options": {"UCI_LimitStrength": True, "UCI_Elo": elo},
        "category": "external",
        "public_visible": True,
        "enabled": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-id", required=True)
    args = parser.parse_args()

    engine = make_engine(get_settings().db_url)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        build = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == args.build_id)
            .first()
        )
        if build is None:
            sys.exit(
                f"error: build {args.build_id} not registered "
                f"(run install_external_build.py first)"
            )

        # The probe must target the EXACT binary this build row points at,
        # re-verified against the registered SHA — never an arbitrary path.
        binary = Path(build.binary_path)
        if not binary.is_file():
            sys.exit(f"error: registered binary missing: {binary}")
        if sha256_file(binary) != build.binary_sha256:
            sys.exit(
                f"error: binary SHA mismatch for registered build "
                f"{args.build_id}"
            )
        try:
            probe = probe_uci(binary)
        except UciProbeError as exc:
            sys.exit(f"error: UCI probe failed: {exc}")

        require_option(probe, "UCI_LimitStrength", "check")
        elo_opt = require_option(probe, "UCI_Elo", "spin")
        for value in STOCKFISH_ELOS:
            if elo_opt.min is not None and value < elo_opt.min:
                sys.exit(f"error: UCI_Elo {value} below minimum {elo_opt.min}")
            if elo_opt.max is not None and value > elo_opt.max:
                sys.exit(f"error: UCI_Elo {value} above maximum {elo_opt.max}")

        for elo in STOCKFISH_ELOS:
            preset_cfg = _preset_for_elo(args.build_id, elo)
            validate_preset_options(preset_cfg["uci_options"])
            existing = (
                session.query(EnginePreset)
                .filter(
                    EnginePreset.preset_id == preset_cfg["preset_id"]
                )
                .first()
            )
            if existing is not None:
                existing.build_id = preset_cfg["build_id"]
                existing.display_name = preset_cfg["display_name"]
                existing.command_args = preset_cfg["command_args"]
                existing.uci_options = preset_cfg["uci_options"]
                existing.category = preset_cfg["category"]
                existing.public_visible = preset_cfg["public_visible"]
                existing.enabled = preset_cfg["enabled"]
                action = "updated"
            else:
                session.add(EnginePreset(**preset_cfg))
                action = "created"
            print(f"{action} preset {preset_cfg['preset_id']} "
                  f"(UCI_Elo {elo})")
        session.commit()

    print(f"stockfish presets registered for build {args.build_id}")
    print(f"  UCI id name: {probe.id_name}")
    print(f"  UCI_Elo range: {elo_opt.min}..{elo_opt.max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
