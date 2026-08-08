"""Browser E2E for the modern React replay demo (P4.UI-1).

Starts a real uvicorn server, opens /chessarena/demo/games/{id} in Chromium
and asserts the React island actually works: board mounted, initial
position correct, PGN move count, clicking a ply sets the right FEN,
first/prev/next/last, keyboard navigation, metadata badges, mobile
responsive layout, and zero console/page errors.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

pytest.importorskip("playwright")

from chessarena.models import COMPLETED, Game, Tournament  # noqa: E402

ARENA_ROOT = Path(__file__).resolve().parents[1]

# Contains captures (2.exd5, 2...Qxd5) to exercise move application from
# verbose from/to squares (a regression where SAN re-parse failed on real
# tournament PGNs with "Invalid move: Bxc6").
SAMPLE_PGN = "\n".join(
    [
        '[Event "E2E"]',
        '[Site "?"]',
        '[Date "2026.08.07"]',
        '[Round "1"]',
        '[White "EngineA"]',
        '[Black "EngineB"]',
        '[Result "1-0"]',
        "",
        "1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8 1-0",
        "",
    ]
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PLY3_FEN = "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
PLY4_FEN = "rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
PLY5_FEN = "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 3"
PLY6_FEN = "rnbqkbnr/ppp1pppp/8/8/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4"


FEN_HEADER_PGN = "\n".join(
    [
        '[Event "E2E"]',
        '[Site "?"]',
        '[Date "2026.08.07"]',
        '[Round "3"]',
        '[White "EngineA"]',
        '[Black "EngineB"]',
        '[Result "*"]',
        '[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]',
        "",
        "3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *",
        "",
    ]
)
FEN_START = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
FEN_PLY2 = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"

# Deterministic 80-ply legal game (LCG seed 42) long enough to scroll the
# move list; used to verify the active move auto-scrolls into view.
LONG_PGN = "\n".join(
    [
        '[Event "E2E"]',
        '[Site "?"]',
        '[Date "2026.08.07"]',
        '[Round "1"]',
        '[White "EngineA"]',
        '[Black "EngineB"]',
        '[Result "*"]',
        "",
        "1. f4 d6 2. Nf3 e6 3. h3 c6 4. c4 Qf6 5. f5 Nh6 6. Ng1 Nxf5 7. Qc2 Qg5 "
        "8. Qd3 a5 9. Qd5 Qg3+ 10. Kd1 Qf2 11. h4 b5 12. Kc2 Nd7 13. g3 Qe1 "
        "14. Nc3 Ba6 15. cxb5 a4 16. bxc6 Nc5 17. Nd1 Ne3+ 18. Kb1 Nb7 19. c7 e5 "
        "20. h5 Nxd5 21. Nf2 Qxd2 22. Nf3 Na5 23. Bg2 Nc6 24. Rf1 Rg8 25. Bxd2 g6 "
        "26. c8=N Bxe2 27. Nxd6+ Ke7 28. Re1 Rg7 29. Nh2 Nc7 30. Bh1 Nd8 "
        "31. Rd1 h6 32. Nf3 Ra7 33. Nxe5 Nb7 34. Bf3 Bxd1 35. Nexf7 a3 "
        "36. Nde4 Nb5 37. Nxh6 Ra8 38. hxg6 Na5 39. b4 Bxf3 40. g4 Rh7",
        "",
    ]
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until_up(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise AssertionError(f"server did not come up at {url}")


def test_browser_demo_replay(settings, engine_factory, registered):
    import json

    manifest = json.loads(
        (registered["build_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    opening_manifest = json.loads(
        (registered["opening_dir"] / "manifest.json").read_text(encoding="utf-8")
    )

    with engine_factory() as session:
        tournament = Tournament(
            id=str(uuid.uuid4()),
            name="e2e-demo",
            status=COMPLETED,
            engine_a_build_id=manifest["build_id"],
            engine_a_profile="current-final",
            engine_b_build_id=manifest["build_id"],
            engine_b_profile="current",
            opening_set_id=opening_manifest["opening_set_id"],
            time_control="blitz_3_2",
            requested_pairs=2,
            completed_pairs=2,
            config_snapshot={
                "engine_a": {
                    "build_id": manifest["build_id"],
                    "profile": "current-final",
                    "command_args": ["--profile", "current-final"],
                    "uci_options": {},
                },
                "engine_b": {
                    "build_id": manifest["build_id"],
                    "profile": "current",
                    "command_args": ["--profile", "current"],
                    "uci_options": {},
                },
                "time_control": "blitz_3_2",
                "hash_mb": 32,
                "threads": 1,
            },
        )
        session.add(tournament)
        session.flush()
        from chessarena.models import PairJob

        pair = PairJob(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
            pair_index=0,
            opening_index=0,
            status="COMPLETED",
        )
        session.add(pair)
        session.flush()

        pgn_path = settings.run_root / "demo-match.pgn"
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        pgn_path.write_text(SAMPLE_PGN, encoding="utf-8")
        game = Game(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
            pair_job_id=pair.id,
            game_number=1,
            white_engine="EngineA",
            black_engine="EngineB",
            opening_index=0,
            result="1-0",
            pgn_path=str(pgn_path),
            verified=True,
        )
        session.add(game)
        session.commit()
        gid = game.id

    os.environ["ARENA_DB_URL"] = settings.db_url
    os.environ["ARENA_RUN_ROOT"] = str(settings.run_root)
    os.environ["ARENA_BUILD_ROOT"] = str(settings.build_root)
    os.environ["ARENA_OPENING_ROOT"] = str(settings.opening_root)
    os.environ["ARENA_CUTECHESS"] = str(settings.cutechess)
    os.environ["ARENA_BASE_PATH"] = settings.base_path

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "chessarena.main:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ARENA_ROOT),
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/chessarena"
    try:
        _wait_until_up(f"{base}/demo/games/{gid}")

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            console_errors: list[str] = []

            def _on_console(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)

            page.on("console", _on_console)
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))

            resp = page.goto(
                f"{base}/demo/games/{gid}", wait_until="networkidle"
            )
            assert resp.status == 200
            page.wait_for_timeout(600)

            board = page.locator(".board-wrap")
            assert board.count() == 1, "board not mounted"
            assert board.get_attribute("data-fen") == START_FEN, (
                "initial position wrong"
            )

            # Move count = 6 plies.
            moves = page.locator("button.move")
            assert moves.count() == 6, f"move count wrong: {moves.count()}"

            # Clicking ply 3 (2.exd5, the 3rd move button) sets the FEN.
            moves.nth(2).click()
            page.wait_for_timeout(300)
            assert board.get_attribute("data-fen") == PLY3_FEN

            # Next button -> ply 4 (2...Qxd5 capture).
            page.get_by_label("Next move").click()
            page.wait_for_timeout(300)
            assert board.get_attribute("data-fen") == PLY4_FEN

            # Last -> final position.
            page.locator("button", has_text="last").click()
            page.wait_for_timeout(300)
            assert board.get_attribute("data-fen") == PLY6_FEN

            # Previous -> ply 5.
            page.get_by_label("Previous move").click()
            page.wait_for_timeout(300)
            assert board.get_attribute("data-fen") == PLY5_FEN

            # First -> start.
            page.locator("button", has_text="first").click()
            page.wait_for_timeout(300)
            assert board.get_attribute("data-fen") == START_FEN

            # Keyboard right/left navigation.
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(200)
            assert board.get_attribute("data-fen") != START_FEN
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(200)
            assert board.get_attribute("data-fen") == START_FEN

            # Metadata badges + player names.  White sits BELOW the board
            # (react-chessboard default white orientation), Black ABOVE.
            assert (
                page.locator(".player-card.top .player-name").inner_text()
                == "EngineB"
            )
            assert (
                page.locator(".player-card.bottom .player-name").inner_text()
                == "EngineA"
            )
            body_text = page.locator("body").inner_text()
            assert "Game 1" in body_text
            assert "Pair 1" in body_text
            assert "3+2" in body_text
            assert "blitz_3_2" not in body_text
            assert "1-0" in body_text

            # Mobile viewport: single column layout (moves below board).
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(300)
            cols = page.evaluate(
                "getComputedStyle(document.querySelector('.replay'))"
                ".gridTemplateColumns"
            )
            assert " " not in cols.strip(), (
                f"expected single column on mobile, got {cols}"
            )

            assert not console_errors, f"browser console errors: {console_errors}"
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_browser_demo_replay_from_fen_header(settings, engine_factory, registered):
    """PGNs from opening-book games carry a [FEN] header; the demo must
    start from that position, not the standard array (regression: 'Invalid
    move' during navigation on real tournament PGNs)."""
    import json

    from chessarena.models import PairJob

    manifest = json.loads(
        (registered["build_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    opening_manifest = json.loads(
        (registered["opening_dir"] / "manifest.json").read_text(encoding="utf-8")
    )

    with engine_factory() as session:
        tournament = Tournament(
            id=str(uuid.uuid4()),
            name="e2e-demo-fen",
            status=COMPLETED,
            engine_a_build_id=manifest["build_id"],
            engine_a_profile="current-final",
            engine_b_build_id=manifest["build_id"],
            engine_b_profile="current",
            opening_set_id=opening_manifest["opening_set_id"],
            time_control="blitz_3_2",
            requested_pairs=2,
            completed_pairs=2,
            config_snapshot={},
        )
        session.add(tournament)
        session.flush()
        pair = PairJob(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
            pair_index=0,
            opening_index=0,
            status="COMPLETED",
        )
        session.add(pair)
        session.flush()
        pgn_path = settings.run_root / "demo-fen-match.pgn"
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        pgn_path.write_text(FEN_HEADER_PGN, encoding="utf-8")
        game = Game(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
            pair_job_id=pair.id,
            game_number=1,
            white_engine="EngineA",
            black_engine="EngineB",
            opening_index=0,
            result="*",
            pgn_path=str(pgn_path),
            verified=True,
        )
        session.add(game)
        session.commit()
        gid = game.id

    os.environ["ARENA_DB_URL"] = settings.db_url
    os.environ["ARENA_RUN_ROOT"] = str(settings.run_root)
    os.environ["ARENA_BUILD_ROOT"] = str(settings.build_root)
    os.environ["ARENA_OPENING_ROOT"] = str(settings.opening_root)
    os.environ["ARENA_CUTECHESS"] = str(settings.cutechess)
    os.environ["ARENA_BASE_PATH"] = settings.base_path

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "chessarena.main:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ARENA_ROOT),
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/chessarena"
    try:
        _wait_until_up(f"{base}/demo/games/{gid}")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            console_errors: list[str] = []

            def _on_console(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)

            page.on("console", _on_console)
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))

            resp = page.goto(
                f"{base}/demo/games/{gid}", wait_until="networkidle"
            )
            assert resp.status == 200
            page.wait_for_timeout(600)

            board = page.locator(".board-wrap")
            # Initial position = the [FEN] header, NOT the standard array.
            assert board.get_attribute("data-fen") == FEN_START
            moves = page.locator("button.move")
            assert moves.count() == 6

            # Click ply 2 (1...a6) -> position after that ply.
            moves.nth(1).click()
            page.wait_for_timeout(300)
            assert board.get_attribute("data-fen") == FEN_PLY2

            assert not console_errors, f"browser console errors: {console_errors}"
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_browser_demo_long_game_active_move_scrolls_into_view(
    settings, engine_factory, registered
):
    """P2 polish: navigating a long game keeps the active move visible inside
    the scrollable move list."""
    import json

    from chessarena.models import PairJob

    manifest = json.loads(
        (registered["build_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    opening_manifest = json.loads(
        (registered["opening_dir"] / "manifest.json").read_text(encoding="utf-8")
    )

    with engine_factory() as session:
        tournament = Tournament(
            id=str(uuid.uuid4()),
            name="e2e-demo-long",
            status=COMPLETED,
            engine_a_build_id=manifest["build_id"],
            engine_a_profile="current-final",
            engine_b_build_id=manifest["build_id"],
            engine_b_profile="current",
            opening_set_id=opening_manifest["opening_set_id"],
            time_control="blitz_3_2",
            requested_pairs=2,
            completed_pairs=2,
            config_snapshot={},
        )
        session.add(tournament)
        session.flush()
        pair = PairJob(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
            pair_index=0,
            opening_index=0,
            status="COMPLETED",
        )
        session.add(pair)
        session.flush()
        pgn_path = settings.run_root / "demo-long-match.pgn"
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        pgn_path.write_text(LONG_PGN, encoding="utf-8")
        game = Game(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
            pair_job_id=pair.id,
            game_number=1,
            white_engine="EngineA",
            black_engine="EngineB",
            opening_index=0,
            result="*",
            pgn_path=str(pgn_path),
            verified=True,
        )
        session.add(game)
        session.commit()
        gid = game.id

    os.environ["ARENA_DB_URL"] = settings.db_url
    os.environ["ARENA_RUN_ROOT"] = str(settings.run_root)
    os.environ["ARENA_BUILD_ROOT"] = str(settings.build_root)
    os.environ["ARENA_OPENING_ROOT"] = str(settings.opening_root)
    os.environ["ARENA_CUTECHESS"] = str(settings.cutechess)
    os.environ["ARENA_BASE_PATH"] = settings.base_path

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "chessarena.main:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ARENA_ROOT),
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/chessarena"
    try:
        _wait_until_up(f"{base}/demo/games/{gid}")
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            console_errors: list[str] = []

            def _on_console(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)

            page.on("console", _on_console)
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))

            resp = page.goto(
                f"{base}/demo/games/{gid}", wait_until="networkidle"
            )
            assert resp.status == 200
            page.wait_for_timeout(600)

            moves = page.locator("button.move")
            assert moves.count() == 80

            # Jump to the final ply; the active move must be visible inside
            # the move-list viewport.
            page.locator("button", has_text="last").click()
            page.wait_for_timeout(800)  # allow smooth scroll to settle

            active_visible = page.evaluate(
                """() => {
                    const list = document.querySelector('.moves-list');
                    const active = document.querySelector('.move.active');
                    if (!list || !active) return false;
                    const lr = list.getBoundingClientRect();
                    const ar = active.getBoundingClientRect();
                    return ar.top >= lr.top - 1 && ar.bottom <= lr.bottom + 1;
                }"""
            )
            assert active_visible, "active move not visible in move-list viewport"

            assert not console_errors, f"browser console errors: {console_errors}"
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
