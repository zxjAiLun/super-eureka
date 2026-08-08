"""Official opening-book integration tests (P4.F1 Phase C): PGN book FEN
resolution, plies handling, deterministic seeded selection, and snapshot
freezing."""

from __future__ import annotations

import hashlib

import chess
import chess.pgn
import pytest

from chessarena.models import OpeningSet
from chessarena.services import openings
from chessarena.services.cutechess import CutechessLaunchError


def _write_pgn_fixture(path) -> None:
    games = [
        # 16 plies (8 full moves)
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
        "7. Bb3 d6 8. c3 O-O",
        # 12 plies
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7",
        # 10 plies
        "1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5",
    ]
    blocks = []
    for i, moves in enumerate(games):
        blocks.append(
            f'[Event "test {i}"]\n[White "W"]\n[Black "B"]\n\n{moves}\n\n'
        )
    path.write_text("".join(blocks), encoding="utf-8")


class _FakeOpeningSet:
    def __init__(self, path, fmt="pgn"):
        self.file_path = str(path)
        self.format = fmt
        self.manifest = {"format": fmt}
        self.position_count = 3


def _expected_fen(moves_san: str, plies: int) -> str:
    board = chess.Board()
    sans = [t for t in moves_san.split() if not t.rstrip(".").isdigit()][:plies]
    for san in sans:
        board.push_san(san)
    return board.fen()


def test_openings_epd_legacy(tmp_path):
    epd = tmp_path / "o.epd"
    epd.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;c0 x\n",
        encoding="utf-8",
    )
    os = _FakeOpeningSet(epd, "epd")
    assert (
        openings.opening_fen_for_index(os, 0)
        == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )


def test_openings_pgn_fen_16_ply(tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    os = _FakeOpeningSet(p)
    fen = openings.opening_fen_for_index(os, 0, 16)
    assert fen == _expected_fen(
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 "
        "7. Bb3 d6 8. c3 O-O",
        16,
    )


def test_openings_pgn_fen_12_ply(tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    os = _FakeOpeningSet(p)
    fen = openings.opening_fen_for_index(os, 1, 12)
    assert fen == _expected_fen(
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7", 12
    )


def test_openings_pgn_short_entry_excluded(tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    os = _FakeOpeningSet(p)
    # Game 2 has only 10 plies; requesting 12 must fail closed.
    with pytest.raises(CutechessLaunchError, match="shorter"):
        openings.opening_fen_for_index(os, 2, 12)
    # eligible pool excludes the short game.
    assert openings.eligible_openings(os, 12) == [0, 1]


def test_openings_select_deterministic_no_repeat(tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    os = _FakeOpeningSet(p)
    a = openings.select_opening_indices(os, 2, 12, seed=42)
    b = openings.select_opening_indices(os, 2, 12, seed=42)
    assert a == b
    c = openings.select_opening_indices(os, 2, 12, seed=7)
    assert a != c
    assert len(set(a)) == len(a)  # no duplicates


def test_openings_select_insufficient_eligible(tmp_path):
    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    os = _FakeOpeningSet(p)
    with pytest.raises(CutechessLaunchError, match="only"):
        openings.select_opening_indices(os, 5, 12, seed=1)


def test_create_tournament_freezes_opening_contract(
    app_client, settings, engine_factory, registered, tmp_path
):
    import json
    import uuid

    p = tmp_path / "o.pgn"
    _write_pgn_fixture(p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with engine_factory() as session:
        session.add(
            OpeningSet(
                opening_set_id="pgn-book",
                file_path=str(p),
                sha256=sha,
                position_count=3,
                format="pgn",
                source="official-stockfish/books (test)",
                manifest={"format": "pgn", "source": "test"},
                enabled=True,
            )
        )
        session.commit()

    resp = app_client.post(
        "/chessarena/api/v1/tournaments",
        headers={"Origin": "http://testserver"},
        json={
            "name": "pgn-book-run",
            "engine_a": {"preset_id": "chessengine-production"},
            "engine_b": {"preset_id": "chessengine-legacy-current"},
            "opening_set_id": "pgn-book",
            "time_control": "blitz_3_2",
            "pairs": 2,
            "opening_plies": 12,
            "opening_seed": 42,
        },
    )
    assert resp.status_code == 201, resp.text
    tid = resp.json()["id"]

    with engine_factory() as session:
        from chessarena.models import Tournament

        t = session.query(Tournament).filter(Tournament.id == tid).one()
        snap = t.config_snapshot["opening_set"]
        assert snap["format"] == "pgn"
        assert snap["plies"] == 12
        assert snap["seed"] == 42
        assert snap["sha256"] == sha
        assert len(snap["indices"]) == 2
        # indices come from the eligible pool (only games 0 and 1 are >=12
        # plies) and pair jobs carry exactly those frozen indices.
        assert set(snap["indices"]) == {0, 1}
        pair_indices = [p.opening_index for p in t.pair_jobs]
        assert pair_indices == snap["indices"]
