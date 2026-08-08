"""Browser E2E for the dynamic preset editor (P4.F1 B4): selecting an
engine build renders its probed UCI options; creating a preset makes it
available in the New Match engine selector."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright")

from chessarena.models import EngineBuild  # noqa: E402

ARENA_ROOT = Path(__file__).resolve().parents[1]


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


def test_browser_preset_editor_flow(settings, engine_factory, registered):
    import json

    with engine_factory() as session:
        build = session.query(EngineBuild).first()
        build.uci_options_schema = {
            "UCI_LimitStrength": {"type": "check"},
            "UCI_Elo": {"type": "spin", "min": 1350, "max": 2850},
            "Style": {"type": "combo", "vars": ["Solid", "Normal", "Very Risky"]},
            "Hash": {"type": "spin", "min": 1, "max": 1024},
        }
        session.commit()
        bid = build.build_id

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
        _wait_until_up(f"{base}/admin/presets/new")
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

            page.goto(f"{base}/admin/presets/new", wait_until="networkidle")
            page.select_option("#build-select", bid)
            page.wait_for_timeout(600)

            # Dynamic controls appear for probed, non-runtime-owned options.
            assert page.locator('input[name="option_UCI_Elo"]').count() == 1
            assert page.locator('input[name="option_UCI_LimitStrength"]').count() == 1
            assert page.locator('select[name="option_Style"]').count() == 1
            # Arena-owned Hash is not rendered.
            assert page.locator('input[name="option_Hash"]').count() == 0

            page.fill('input[name="display_name"]', "Browser Strength")
            page.fill('input[name="option_UCI_Elo"]', "2300")
            page.check('input[name="option_UCI_LimitStrength"]')
            page.select_option('select[name="option_Style"]', "Very Risky")
            page.click('button[type="submit"]')
            page.wait_for_load_state("networkidle")

            # Redirect lands on the New Match page; the preset is selectable.
            assert "/admin/tournaments/new" in page.url
            assert page.locator('select[name="engine_a_preset"] option', has_text="Browser Strength").count() == 1

            assert not console_errors, f"browser console errors: {console_errors}"
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
