"""Browser E2E for the Lichess PGN viewer (P4.1 Replay Repair).

HTTP 200 alone is not a frontend check: this test starts a real uvicorn
server, opens a verified game page in Chromium, and asserts the viewer
actually mounted (chessground board + move list), the move list is
non-empty, pressing "next" advances the position, and the browser console
logged no errors.

Skipped when Playwright or its browser is unavailable.
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
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0",
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


def test_browser_pgn_replay_renders(settings, engine_factory, registered):
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
            name="e2e-replay",
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

        pgn_path = settings.run_root / "e2e-match.pgn"
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        pgn_path.write_text(SAMPLE_PGN, encoding="utf-8")
        game = Game(
            id=str(uuid.uuid4()),
            tournament_id=tournament.id,
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
        _wait_until_up(f"{base}/games/{gid}")

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

            page.goto(f"{base}/games/{gid}", wait_until="networkidle")
            page.wait_for_timeout(500)

            board = page.locator("#board")
            # The Lichess viewer rewrites the container (#board id is removed
            # and replaced with the lpv structure), so assert on the actual
            # mounted DOM instead: chessground board + move buttons.
            assert page.locator(".lpv__board").count() > 0, (
                "lpv viewer not mounted (init failed)"
            )
            assert page.locator(".lpv__board .cg-wrap").count() > 0, (
                "chessground board not mounted"
            )
            moves = page.locator("button.move")
            assert moves.count() >= 4, f"move list too short: {moves.count()}"

            # Pressing next must advance the position: the prev button loses
            # its disabled state after stepping past the start.
            next_btn = page.locator(".lpv__controls__goto--next")
            prev_btn = page.locator(".lpv__controls__goto--prev")
            assert next_btn.count() > 0, "next button not found"
            assert "disabled" in (prev_btn.get_attribute("class") or ""), (
                "expected prev disabled at start position"
            )
            next_btn.first.click()
            page.wait_for_timeout(300)
            assert "disabled" not in (
                prev_btn.get_attribute("class") or ""
            ), "position did not advance after pressing next"

            assert not console_errors, (
                f"browser console errors: {console_errors}"
            )
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
