"""Live pair/game runtime status (P4.F1 A3).

Game rows stay pair-atomic (created only after the verifier passes); while
cutechess is still running the runtime status is derived from stdout.log so
the admin UI can show "Game 1/2 finished -> Game 2/2 running" before the
pair process exits.
"""

from __future__ import annotations

import os
import time

from chessarena.models import RUNNING, Tournament
from chessarena.services import artifacts
from chessarena.services.runtime_status import derive_runtime_status


def test_derive_runtime_status_stages(tmp_path):
    run = tmp_path / "pair"
    run.mkdir()

    s = derive_runtime_status(run)
    assert s["state"] == "pending"
    assert s["finished_games_in_pair"] == 0

    (run / "stdout.log").write_text(
        "Started game 1 of 2 (A vs B)\n", encoding="utf-8"
    )
    s = derive_runtime_status(run)
    assert s["state"] == "game_running"
    assert s["game_in_pair"] == 1
    assert s["finished_games_in_pair"] == 0

    (run / "stdout.log").write_text(
        "Started game 1 of 2 (A vs B)\n"
        "Finished game 1 (A vs B): 1-0 {White mates}\n",
        encoding="utf-8",
    )
    s = derive_runtime_status(run)
    assert s["finished_games_in_pair"] == 1
    assert s["last_result"] == "1-0"

    (run / "stdout.log").write_text(
        "Started game 1 of 2 (A vs B)\n"
        "Finished game 1 (A vs B): 1-0 {White mates}\n"
        "Started game 2 of 2 (B vs A)\n",
        encoding="utf-8",
    )
    s = derive_runtime_status(run)
    assert s["state"] == "game_running"
    assert s["game_in_pair"] == 2
    assert s["finished_games_in_pair"] == 1

    (run / "stdout.log").write_text(
        "Started game 1 of 2 (A vs B)\n"
        "Finished game 1 (A vs B): 1-0 {White mates}\n"
        "Started game 2 of 2 (B vs A)\n"
        "Finished game 2 (B vs A): 0-1 {Black mates}\n",
        encoding="utf-8",
    )
    s = derive_runtime_status(run)
    assert s["state"] == "pair_done"
    assert s["finished_games_in_pair"] == 2
    assert s["last_result"] == "0-1"


def test_runtime_status_game1_to_game2_while_process_alive(
    settings, scheduler, engine_factory, tournament_factory
):
    """The streaming fake cutechess emits game boundaries with delays; the
    runtime status must show Game 1 finished -> Game 2 running while the
    process is still alive."""
    os.environ["FAKE_CUTECHESS_STREAM"] = "1"
    os.environ["FAKE_CUTECHESS_STREAM_DELAY_MS"] = "300"

    tid = tournament_factory(name="streaming", pairs=1, status=RUNNING)
    try:
        with engine_factory() as session:
            t = session.query(Tournament).filter(Tournament.id == tid).one()
            pair = t.pair_jobs[0]
            pair.status = RUNNING
            session.commit()
            run_dir = artifacts.pair_run_dir(
                t.id, pair.pair_index, pair.attempt
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            scheduler._prepare_and_launch(session, t, pair, run_dir)
            proc = scheduler.active_proc
            assert proc is not None
            try:
                deadline = time.time() + 20
                progressed = False
                while time.time() < deadline:
                    s = derive_runtime_status(run_dir)
                    if (
                        s["finished_games_in_pair"] >= 1
                        and s["game_in_pair"] == 2
                        and proc.poll() is None
                    ):
                        progressed = True
                        break
                    time.sleep(0.15)
                assert progressed, (
                    "never observed Game 2 running while process alive"
                )
                s = derive_runtime_status(run_dir)
                assert s["state"] == "game_running"
                assert s["game_in_pair"] == 2
                assert s["finished_games_in_pair"] == 1
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=10)
    finally:
        os.environ.pop("FAKE_CUTECHESS_STREAM", None)
        os.environ.pop("FAKE_CUTECHESS_STREAM_DELAY_MS", None)


def test_admin_pairs_fragment_shows_runtime_game(
    app_client, settings, engine_factory, tournament_factory
):
    from chessarena.models import PairJob, Tournament as T
    from chessarena.services import artifacts

    tid = tournament_factory(name="frag", pairs=1, status=RUNNING)
    with engine_factory() as session:
        t = session.query(T).filter(T.id == tid).one()
        pair: PairJob = t.pair_jobs[0]
        pair.status = RUNNING
        run_dir = artifacts.pair_run_dir(t.id, pair.pair_index, pair.attempt)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stdout.log").write_text(
            "Started game 1 of 2 (A vs B)\n"
            "Finished game 1 (A vs B): 1-0 {White mates}\n"
            "Started game 2 of 2 (B vs A)\n",
            encoding="utf-8",
        )
        pair.run_directory = str(run_dir)
        session.commit()

    r = app_client.get(f"/chessarena/admin/tournaments/{tid}/pairs")
    assert r.status_code == 200
    assert "Pair progress" in r.text
    assert "Game 1/2 finished · Game 2/2 running" in r.text

    detail = app_client.get(f"/chessarena/admin/tournaments/{tid}")
    assert detail.status_code == 200
    assert 'id="pair-progress"' in detail.text
    assert "Game 1/2 finished · Game 2/2 running" in detail.text


def test_admin_pairs_fragment_idle_has_no_runtime(app_client, tournament_factory):
    tid = tournament_factory(name="frag2", pairs=1, status="QUEUED")
    r = app_client.get(f"/chessarena/admin/tournaments/{tid}/pairs")
    assert r.status_code == 200
    assert "Game" not in r.text or "running" not in r.text
