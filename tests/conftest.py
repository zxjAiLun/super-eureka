"""Shared pytest fixtures for the arena test suite.

Every test gets an isolated environment:
- fresh SQLite database in a tmp dir,
- fresh run/build/opening roots,
- the fake cutechess-cli shim as ARENA_CUTECHESS,
- one registered engine build (two profiles) and one registered opening set
  (20 unique positions).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import chess
import pytest

from chessarena.config import Settings
from chessarena.db import make_engine, make_session_factory
from chessarena.models import EngineBuild, EnginePreset, OpeningSet
from chessarena.services import artifacts

FIXTURES = Path(__file__).parent / "fixtures"
if sys.platform == "win32":
    FAKE_CUTECHESS = FIXTURES / "fake_cutechess.cmd"
else:
    FAKE_CUTECHESS = FIXTURES / "fake_cutechess.py"
    # On POSIX the fake is a plain script invoked directly by subprocess; make
    # sure it is executable regardless of how the checkout preserved modes.
    FAKE_CUTECHESS.chmod(0o755)

TEST_OPENING_SET_ID = "test-openings-v1"
BUILD_A_ID = "20260805-bde9085-linux-x86_64"
BUILD_B_ID = "20260805-bde9085-linux-x86_64"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_positions(count: int = 20) -> list[str]:
    base = chess.Board()
    moves = list(base.legal_moves)[:count]
    fens = []
    for move in moves:
        board = chess.Board()
        board.push(move)
        fens.append(board.fen())
    return fens


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_url=f"sqlite:///{tmp_path / 'arena.db'}",
        run_root=tmp_path / "runs",
        build_root=tmp_path / "builds",
        opening_root=tmp_path / "openings",
        cutechess=FAKE_CUTECHESS,
        max_concurrency=1,
        hash_mb=32,
        threads=1,
        base_path="/chessarena",
        public_url="http://testserver/chessarena",
        worker_poll_seconds=0.05,
        worker_heartbeat_seconds=0.05,
        worker_stale_seconds=15.0,
        shutdown_grace_seconds=2.0,
    )


@pytest.fixture()
def engine_factory(settings: Settings):
    engine = make_engine(settings.db_url)
    from chessarena.models import Base

    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    artifacts.configure_artifact_service(settings)
    return session_factory


@pytest.fixture()
def build_dir(tmp_path: Path, build_id: str = BUILD_A_ID) -> Path:
    path = tmp_path / "builds" / build_id
    path.mkdir(parents=True)
    content = f"dummy engine binary for {build_id}".encode()
    engine_path = path / "engine"
    engine_path.write_bytes(content)
    # install_build.py checks os.access(X_OK); on POSIX a freshly written file
    # is not executable, so make the dummy engine runnable there.
    if sys.platform != "win32":
        engine_path.chmod(0o755)
    manifest = {
        "schema_version": 1,
        "build_id": build_id,
        "engine_name": "ChessEngineDemo",
        "git_sha": "bde9085e1347d5a1c2d8503baf65478c3b49db0d",
        "binary_sha256": _sha(content),
        "platform": "linux-x86_64",
        "rustc_version": "1.85.0",
        "cargo_lock_sha256": "0" * 64,
        "supported_profiles": ["current-final", "current"],
        "uci_id_name": "ChessEngineDemo",
        "uci_id_author": "Rust-learner",
        "created_utc": "2026-08-05T00:00:00+00:00",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture()
def opening_dir(tmp_path: Path) -> Path:
    path = tmp_path / "openings" / TEST_OPENING_SET_ID
    path.mkdir(parents=True)
    fens = build_positions(20)
    epd = "\n".join(fens) + "\n"
    (path / "openings.epd").write_text(epd, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "opening_set_id": TEST_OPENING_SET_ID,
        "format": "epd",
        "count": len(fens),
        "sha256": _sha(epd.encode()),
        "unique_position_keys": True,
        "non_terminal": True,
        "created_utc": "2026-08-05T00:00:00+00:00",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture()
def registered(engine_factory, build_dir: Path, opening_dir: Path):
    """Register one build (both profiles) and one opening set in the DB."""
    manifest = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))
    with engine_factory() as session:
        session.add(
            EngineBuild(
                build_id=manifest["build_id"],
                engine_name=manifest["engine_name"],
                git_sha=manifest["git_sha"],
                binary_path=str(build_dir / "engine"),
                binary_sha256=manifest["binary_sha256"],
                platform=manifest["platform"],
                supported_profiles=manifest["supported_profiles"],
                manifest=manifest,
                enabled=True,
            )
        )
        session.add(
            EnginePreset(
                preset_id="chessengine-production",
                build_id=manifest["build_id"],
                display_name="ChessEngine Production",
                command_args=["--profile", "current-final"],
                uci_options={},
                category="production",
                public_visible=True,
                enabled=True,
            )
        )
        session.add(
            EnginePreset(
                preset_id="chessengine-legacy-current",
                build_id=manifest["build_id"],
                display_name="ChessEngine Legacy Baseline",
                command_args=["--profile", "current"],
                uci_options={},
                category="legacy",
                public_visible=True,
                enabled=True,
            )
        )
        opening_manifest = json.loads(
            (opening_dir / "manifest.json").read_text(encoding="utf-8")
        )
        session.add(
            OpeningSet(
                opening_set_id=opening_manifest["opening_set_id"],
                file_path=str(opening_dir / "openings.epd"),
                sha256=opening_manifest["sha256"],
                position_count=opening_manifest["count"],
                manifest=opening_manifest,
                enabled=True,
            )
        )
        session.commit()
    return {"build_dir": build_dir, "opening_dir": opening_dir}


@pytest.fixture()
def app_client(settings: Settings, registered):
    from fastapi.testclient import TestClient

    from chessarena.main import create_app

    os.environ["ARENA_DB_URL"] = settings.db_url
    os.environ["ARENA_RUN_ROOT"] = str(settings.run_root)
    os.environ["ARENA_BUILD_ROOT"] = str(settings.build_root)
    os.environ["ARENA_OPENING_ROOT"] = str(settings.opening_root)
    os.environ["ARENA_CUTECHESS"] = str(settings.cutechess)
    os.environ["ARENA_BASE_PATH"] = settings.base_path

    app = create_app(settings)
    app.dependency_overrides.pop("chessarena.config.get_settings", None)
    return TestClient(app)


@pytest.fixture()
def scheduler(settings: Settings, registered, engine_factory):
    """A Scheduler bound to the test DB (no worker loop)."""
    from chessarena.services.scheduler import Scheduler

    return Scheduler(settings, engine_factory)


@pytest.fixture()
def tournament_factory(engine_factory, registered):
    """Create Tournament + PairJob rows directly in the test DB."""
    manifest = json.loads((registered["build_dir"] / "manifest.json").read_text(encoding="utf-8"))
    opening_manifest = json.loads(
        (registered["opening_dir"] / "manifest.json").read_text(encoding="utf-8")
    )

    def make(
        name: str = "test match",
        pairs: int = 10,
        time_control: str = "blitz_3_2",
        status: str = "DRAFT",
        engine_a_profile: str = "current-final",
        engine_b_profile: str = "current",
    ):
        from chessarena.models import PairJob, Tournament, utcnow

        config_snapshot = {
            "engine_a": {
                "build_id": manifest["build_id"],
                "profile": engine_a_profile,
                "git_sha": manifest["git_sha"],
                "binary_sha256": manifest["binary_sha256"],
            },
            "engine_b": {
                "build_id": manifest["build_id"],
                "profile": engine_b_profile,
                "git_sha": manifest["git_sha"],
                "binary_sha256": manifest["binary_sha256"],
            },
            "opening_set": {
                "opening_set_id": opening_manifest["opening_set_id"],
                "sha256": opening_manifest["sha256"],
            },
            "time_control": time_control,
            "hash_mb": 32,
            "concurrency": 1,
            "requested_pairs": pairs,
        }
        with engine_factory() as session:
            tournament = Tournament(
                name=name,
                status=status,
                engine_a_build_id=manifest["build_id"],
                engine_a_profile=engine_a_profile,
                engine_b_build_id=manifest["build_id"],
                engine_b_profile=engine_b_profile,
                opening_set_id=opening_manifest["opening_set_id"],
                time_control=time_control,
                requested_pairs=pairs,
                config_snapshot=config_snapshot,
            )
            session.add(tournament)
            session.flush()
            for pair_index in range(pairs):
                session.add(
                    PairJob(
                        tournament_id=tournament.id,
                        pair_index=pair_index,
                        opening_index=pair_index,
                        status="PENDING",
                        attempt=1,
                    )
                )
            session.commit()
            return tournament.id

    return make


@pytest.fixture()
def pair_context(engine_factory, registered):
    """A single tournament row + its first pair job, loaded from the DB."""
    from chessarena.models import EngineBuild, OpeningSet, PairJob, Tournament

    opening_manifest = json.loads(
        (registered["opening_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    with engine_factory() as session:
        tournament = Tournament(
            name="pair context",
            status="DRAFT",
            engine_a_build_id=BUILD_A_ID,
            engine_a_profile="current-final",
            engine_b_build_id=BUILD_B_ID,
            engine_b_profile="current",
            opening_set_id=TEST_OPENING_SET_ID,
            time_control="blitz_3_2",
            requested_pairs=1,
            config_snapshot={
                "engine_a": {
                    "build_id": BUILD_A_ID,
                    "profile": "current-final",
                    "git_sha": "bde9085e1347d5a1c2d8503baf65478c3b49db0d",
                    "binary_sha256": _sha(f"dummy engine binary for {BUILD_A_ID}".encode()),
                },
                "engine_b": {
                    "build_id": BUILD_B_ID,
                    "profile": "current",
                    "git_sha": "bde9085e1347d5a1c2d8503baf65478c3b49db0d",
                    "binary_sha256": _sha(f"dummy engine binary for {BUILD_B_ID}".encode()),
                },
                "opening_set": {
                    "opening_set_id": TEST_OPENING_SET_ID,
                    "sha256": opening_manifest["sha256"],
                },
                "time_control": "blitz_3_2",
                "hash_mb": 32,
                "concurrency": 1,
                "requested_pairs": 1,
            },
        )
        session.add(tournament)
        session.flush()
        pair = PairJob(
            tournament_id=tournament.id,
            pair_index=0,
            opening_index=0,
            status="PENDING",
            attempt=1,
        )
        session.add(pair)
        session.flush()
        pair_id = pair.id
        tournament_id = tournament.id
        session.commit()

    with engine_factory() as session:
        tournament = session.get(Tournament, tournament_id)
        pair = session.get(PairJob, pair_id)
        engine_a = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == tournament.engine_a_build_id)
            .first()
        )
        engine_b = (
            session.query(EngineBuild)
            .filter(EngineBuild.build_id == tournament.engine_b_build_id)
            .first()
        )
        opening_set = (
            session.query(OpeningSet)
            .filter(OpeningSet.opening_set_id == tournament.opening_set_id)
            .first()
        )
        return {
            "tournament": tournament,
            "pair": pair,
            "engine_a": engine_a,
            "engine_b": engine_b,
            "opening_set": opening_set,
        }
