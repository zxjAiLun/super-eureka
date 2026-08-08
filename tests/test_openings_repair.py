"""Phase C Repair 1 — PGN opening book execution and provenance closure
(P4.F1 P1-1..P1-4)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import pytest

from chessarena.models import (
    CANCELLED,
    COMPLETED,
    FAILED,
    Game,
    OpeningSet,
    Tournament,
)
from chessarena.services import openings
from chessarena.services.cutechess import CutechessLaunchError


def _write_pgn_fixture(path) -> None:
    games = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
        "7. Bb3 d6 8. c3 O-O",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7",
        "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5",
    ]
    blocks = []
    for i, moves in enumerate(games):
        blocks.append(
            f'[Event "test {i}"]\n[White "W"]\n[Black "B"]\n\n{moves}\n\n'
        )
    path.write_text("".join(blocks), encoding="utf-8")


def _register_pgn_set(engine_factory, path: Path, *, default_plies=12,
                      opening_set_id="pgn-book"):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with engine_factory() as session:
        session.add(
            OpeningSet(
                opening_set_id=opening_set_id,
                file_path=str(path),
                sha256=sha,
                position_count=3,
                format="pgn",
                source="test",
                manifest={"format": "pgn", "default_plies": default_plies},
                enabled=True,
            )
        )
        session.commit()
    return sha


def _run_tournament(scheduler, engine_factory, tid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with engine_factory() as session:
            t = session.get(Tournament, tid)
            if t.status == COMPLETED:
                return t
            if t.status in (FAILED, CANCELLED):
                raise AssertionError(f"tournament ended {t.status}")
        scheduler.tick()
        time.sleep(0.05)
    raise AssertionError("tournament did not complete in time")


def _create_tournament(app_client, opening_set_id, **extra):
    payload = {
        "name": f"repair-{uuid.uuid4().hex[:6]}",
        "engine_a": {"preset_id": "chessengine-production"},
        "engine_b": {"preset_id": "chessengine-legacy-current"},
        "opening_set_id": opening_set_id,
        "time_control": "blitz_3_2",
        "pairs": 2,
        "opening_seed": 5,
    }
    payload.update(extra)
    resp = app_client.post(
        "/chessarena/api/v1/tournaments",
        headers={"Origin": "http://testserver"},
        json=payload,
    )
    return resp


# --- P1-1: PGN book end-to-end (scheduler -> cutechess -> verifier) ----------

def test_pgn_book_pair_end_to_end(app_client, settings, engine_factory,
                                  registered, scheduler, tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    _register_pgn_set(engine_factory, p, default_plies=12)

    resp = _create_tournament(app_client, "pgn-book")  # omitted opening_plies
    assert resp.status_code == 201, resp.text
    tid = resp.json()["id"]
    started = app_client.post(
        f"/chessarena/api/v1/tournaments/{tid}/start",
        headers={"Origin": "http://testserver"},
    )
    assert started.status_code == 200, started.text

    with engine_factory() as session:
        t = session.get(Tournament, tid)
        snap = t.config_snapshot["opening_set"]
        assert snap["format"] == "pgn"
        assert snap["plies"] == 12  # resolved from the book default
        assert len(snap["indices"]) == 2

    t = _run_tournament(scheduler, engine_factory, tid)
    assert t.status == COMPLETED
    with engine_factory() as session:
        t = session.get(Tournament, tid)
        for pair in t.pair_jobs:
            assert pair.status == COMPLETED
            assert pair.verification is not None
            assert pair.verification["verified"] is True
            assert pair.verification["moves_legal"] is True
        games = (
            session.query(Game).filter(Game.tournament_id == tid).all()
        )
        assert len(games) == 4  # 2 pairs x 2 games
        # Strict color swap across each pair is enforced by the verifier,
        # which already passed above.


def test_pgn_book_disk_tamper_rejected_before_launch(
    settings, scheduler, engine_factory, tournament_factory, tmp_path
):
    """P1-3: tampering the opening file on disk (DB row untouched) must be
    rejected before Popen."""
    from chessarena.services import artifacts

    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    _register_pgn_set(engine_factory, p, default_plies=12)

    tid = tournament_factory(name="tamper", pairs=1, status="QUEUED")
    with engine_factory() as session:
        t = session.get(Tournament, tid)
        # Simulate a creation snapshot with PGN plies/seed/indices frozen.
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        t.config_snapshot["opening_set"] = {
            "opening_set_id": "pgn-book",
            "sha256": sha,
            "format": "pgn",
            "plies": 12,
            "seed": 5,
            "indices": [0],
        }
        t.opening_set_id = "pgn-book"
        for pair in t.pair_jobs:
            pair.opening_index = 0
        session.commit()

        # Mutate the file on disk; DB row is untouched.
        p.write_text("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1\n",
                     encoding="utf-8")

        pair = t.pair_jobs[0]
        run_dir = artifacts.pair_run_dir(t.id, pair.pair_index, pair.attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(CutechessLaunchError, match="SHA"):
            scheduler._prepare_and_launch(session, t, pair, run_dir)
        assert scheduler.active_proc is None


def test_pgn_book_disk_tamper_rejected_by_verifier(
    settings, engine_factory, tmp_path
):
    """P1-3: tampering the file between launch and verification must make the
    verifier reject the pair."""
    from chessarena.models import PairJob
    from chessarena.services import artifacts
    from chessarena.services.verifier import VerificationFailure, verify_pair

    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    sha = _register_pgn_set(engine_factory, p, default_plies=12)

    with engine_factory() as session:
        from chessarena.models import EngineBuild

        builds = {}
        for bid in ("b1", "b2"):
            bp = tmp_path / f"engine-{bid}"
            bp.write_bytes(b"dummy engine")
            binary_sha = hashlib.sha256(bp.read_bytes()).hexdigest()
            build = EngineBuild(
                build_id=bid,
                engine_name="ChessEngine",
                git_sha="abc123",
                binary_path=str(bp),
                binary_sha256=binary_sha,
                platform="linux-x86_64",
                supported_profiles=[],
                manifest={},
                uci_options_schema={"Hash": {"type": "spin", "min": 1, "max": 1024}},
                enabled=True,
            )
            session.add(build)
            builds[bid] = build
        t = Tournament(
            id=str(uuid.uuid4()),
            name="tamper-verify",
            status=COMPLETED,
            engine_a_build_id="b1",
            engine_a_profile="current-final",
            engine_b_build_id="b2",
            engine_b_profile="current",
            opening_set_id="pgn-book",
            time_control="blitz_3_2",
            requested_pairs=1,
            completed_pairs=0,
            config_snapshot={
                "engine_a": {
                    "build_id": "b1",
                    "profile": "current-final",
                    "command_args": ["--profile", "current-final"],
                    "uci_options": {},
                    "uci_options_schema": {"Hash": {"type": "spin", "min": 1, "max": 1024}},
                    "git_sha": "abc123",
                    "binary_sha256": builds["b1"].binary_sha256,
                },
                "engine_b": {
                    "build_id": "b2",
                    "profile": "current",
                    "command_args": ["--profile", "current"],
                    "uci_options": {},
                    "uci_options_schema": {"Hash": {"type": "spin", "min": 1, "max": 1024}},
                    "git_sha": "abc123",
                    "binary_sha256": builds["b2"].binary_sha256,
                },
                "opening_set": {
                    "opening_set_id": "pgn-book",
                    "sha256": sha,
                    "format": "pgn",
                    "plies": 12,
                    "seed": 1,
                    "indices": [0],
                },
                "time_control": "blitz_3_2",
                "hash_mb": 32,
                "threads": 1,
            },
        )
        session.add(t)
        session.flush()
        pair = PairJob(
            id=str(uuid.uuid4()),
            tournament_id=t.id,
            pair_index=0,
            opening_index=0,
            status="COMPLETED",
            attempt=1,
        )
        session.add(pair)
        session.flush()
        run_dir = artifacts.pair_run_dir(t.id, 0, 1)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Realistic artifacts: cutechess-style PGN + stdout + command.
        from chessarena.services.cutechess import (
            build_pair_command,
            write_command_artifacts,
        )
        from tests.fixtures.fake_cutechess import _game_text, _write_stdout

        opening_fen = openings.opening_fen_for_index(
            session.query(OpeningSet).first(), 0, 12
        )
        (run_dir / "opening.epd").write_text(
            opening_fen + "\n", encoding="utf-8"
        )
        argv = build_pair_command(
            settings,
            engine_a={
                "binary_path": builds["b1"].binary_path,
                "command_args": ["--profile", "current-final"],
                "uci_options": {},
                "uci_options_schema": {"Hash": {"type": "spin", "min": 1, "max": 1024}},
            },
            engine_b={
                "binary_path": builds["b2"].binary_path,
                "command_args": ["--profile", "current"],
                "uci_options": {},
                "uci_options_schema": {"Hash": {"type": "spin", "min": 1, "max": 1024}},
            },
            time_control="180+2",
            hash_mb=32,
            opening_epd=run_dir / "opening.epd",
            pgn_out=run_dir / "match.pgn",
            threads=1,
        )
        write_command_artifacts(run_dir, argv, extra={})
        game1 = _game_text(1, "EngineA", "EngineB", opening_fen, "180+2", "1-0", False)
        game2 = _game_text(2, "EngineB", "EngineA", opening_fen, "180+2", "0-1", False)
        (run_dir / "match.pgn").write_text(game1 + game2, encoding="utf-8")
        _write_stdout(run_dir / "stdout.log", "EngineA", "EngineB", ["1-0", "0-1"])
        (run_dir / "stderr.log").write_text("", encoding="utf-8")
        session.commit()

        # Tamper the opening file AFTER launch artifacts exist: same 16-ply
        # depth but a different (legal) move -> different bytes -> the
        # verifier must reject on the actual-file SHA.
        p.write_text(
            '[Event "test 0"]\n[White "W"]\n[Black "B"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
            "7. Bb3 d6 8. Nbd2 O-O\n\n",
            encoding="utf-8",
        )

        opening_set = session.query(OpeningSet).first()
        with pytest.raises(VerificationFailure, match="SHA"):
            verify_pair(
                settings,
                tournament=t,
                pair_job=pair,
                run_dir=run_dir,
                engine_a_build=builds["b1"],
                engine_b_build=builds["b2"],
                opening_set=opening_set,
            )


# --- P1-2: catalog -> Arena registration --------------------------------------

def test_catalog_registration(settings, engine_factory, tmp_path):
    import base64
    import os
    import subprocess
    import sys

    p = tmp_path / "book.pgn"
    _write_pgn_fixture(p)
    sha384 = base64.b64encode(
        hashlib.sha384(p.read_bytes()).digest()
    ).decode()

    catalog = tmp_path / "manifest.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default": "test-book",
                "books": {
                    "test-book": {
                        "format": "pgn",
                        "expected_positions": 3,
                        "expected_plies": 12,
                        "content_sha384_base64": sha384,
                        "source_repository": "https://github.com/example/books",
                        "source_ref": "abc123",
                        "license": "CC0-1.0",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["ARENA_DB_URL"] = settings.db_url
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts"
                / "register_openings.py"),
            str(p),
            "--catalog", str(catalog),
            "--book-id", "test-book",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with engine_factory() as session:
        os = session.query(OpeningSet).filter(
            OpeningSet.opening_set_id == "test-book"
        ).one()
        assert os.format == "pgn"
        assert os.position_count == 3
        assert os.manifest["default_plies"] == 12
        assert os.manifest["source_repository"].endswith("example/books")
        assert os.manifest["license"] == "CC0-1.0"
        assert "abc123" in (os.source or "")


# --- P1-4: opening_plies contract --------------------------------------------

def test_pgn_plies_default_resolved(app_client, settings, engine_factory,
                                    registered, tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    _register_pgn_set(engine_factory, p, default_plies=16)
    # With 16 plies only game 0 is eligible, so use a single pair.
    resp = _create_tournament(app_client, "pgn-book", pairs=1)
    assert resp.status_code == 201, resp.text
    with engine_factory() as session:
        t = session.get(Tournament, resp.json()["id"])
        assert t.config_snapshot["opening_set"]["plies"] == 16


def test_pgn_no_default_plies_rejected(app_client, settings, engine_factory,
                                       registered, tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    _register_pgn_set(engine_factory, p, default_plies=None)
    resp = _create_tournament(app_client, "pgn-book")
    assert resp.status_code == 422
    assert "opening_plies" in resp.json()["detail"]


def test_pgn_explicit_plies_used(app_client, settings, engine_factory,
                                 registered, tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    _register_pgn_set(engine_factory, p, default_plies=16)
    resp = _create_tournament(app_client, "pgn-book", opening_plies=12)
    assert resp.status_code == 201, resp.text
    with engine_factory() as session:
        t = session.get(Tournament, resp.json()["id"])
        assert t.config_snapshot["opening_set"]["plies"] == 12


def test_epd_explicit_plies_rejected(app_client, settings, engine_factory,
                                     registered, tmp_path):
    epd = tmp_path / "o.epd"
    epd.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\n"
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2\n",
        encoding="utf-8",
    )
    sha = hashlib.sha256(epd.read_bytes()).hexdigest()
    with engine_factory() as session:
        session.add(
            OpeningSet(
                opening_set_id="epd-book",
                file_path=str(epd),
                sha256=sha,
                position_count=3,
                format="epd",
                manifest={"format": "epd"},
                enabled=True,
            )
        )
        session.commit()
    resp = _create_tournament(app_client, "epd-book", opening_plies=8)
    assert resp.status_code == 422
    assert "opening_plies" in resp.json()["detail"]
