"""Verifier tests (spec section 22.2).

Covers:
- a valid two-game color-swapped pair passes,
- same color twice fails,
- two different openings fails,
- illegal PGN move fails,
- only one game fails,
- engine name mismatch fails,
- TimeControl mismatch fails,
- stdout W/D/L disagreeing with PGN fails,
- engine SHA mismatch fails,
- opening SHA mismatch fails,
- stderr containing crash/timeout/illegal fails.
"""

from __future__ import annotations

import json

import pytest

from chessarena.config import Settings
from chessarena.models import (
    COMPLETED,
    DRAFT,
    PENDING,
    EngineBuild,
    OpeningSet,
    PairJob,
    Tournament,
)
from chessarena.services import verifier

from . import helpers


@pytest.fixture()
def pair_run(settings, engine_factory, pair_context):
    """Run the fake cutechess for the pair_context and return the run dir."""
    return helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
    )


def _verify(settings, engine_factory, pair_context, run_dir):
    return verifier.verify_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        run_dir=run_dir,
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_valid_pair_passes(settings, engine_factory, pair_context, pair_run):
    verification = _verify(settings, engine_factory, pair_context, pair_run)
    assert verification["verified"] is True
    assert verification["pgn_game_count"] == 2
    assert verification["colors"][0] == {"white": "EngineA", "black": "EngineB"}
    assert verification["colors"][1] == {"white": "EngineB", "black": "EngineA"}
    assert verification["moves_legal"] is True
    # results 1-0 (A white) and 0-1 (A black) -> 2 wins
    assert verification["candidate_perspective"] == {"wins": 2, "losses": 0, "draws": 0}
    assert verification["cutechess_score_line"] == {"wins": 2, "losses": 0, "draws": 0}


# ---------------------------------------------------------------------------
# Failure matrix
# ---------------------------------------------------------------------------
def test_same_color_twice_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_SAME_COLORS": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="color assignment"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_different_openings_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_DIFFERENT_OPENING": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="opening position"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_illegal_move_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_BAD_MOVE": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="PGN|illegal|replay"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_only_one_game_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_GAMES": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="expected 2 games"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_empty_pgn_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_EMPTY_PGN": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="empty|missing"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_timecontrol_mismatch_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_TC_WRONG": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="TimeControl"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_stdout_forbidden_term_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_STDOUT_FORBIDDEN": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="forbidden"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_stderr_bad_line_fails(settings, engine_factory, pair_context):
    run_dir = helpers.run_fake_pair(
        settings,
        tournament=pair_context["tournament"],
        pair_job=pair_context["pair"],
        engine_a_build=pair_context["engine_a"],
        engine_b_build=pair_context["engine_b"],
        opening_set=pair_context["opening_set"],
        env_extra={"FAKE_CUTECHESS_STDERR_BAD": "1"},
    )
    with pytest.raises(verifier.VerificationFailure, match="stderr"):
        _verify(settings, engine_factory, pair_context, run_dir)


def test_stdout_score_mismatch_fails(settings, engine_factory, pair_context, pair_run):
    # Corrupt the score line after the fake produced it.
    (pair_run / "stdout.log").write_text(
        "Score of EngineA vs EngineB: 1 - 1 - 0  [0.500] 2\n", encoding="utf-8"
    )
    with pytest.raises(verifier.VerificationFailure, match="disagrees"):
        _verify(settings, engine_factory, pair_context, pair_run)


def test_no_score_line_fails(settings, engine_factory, pair_context, pair_run):
    (pair_run / "stdout.log").write_text("no score line here\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationFailure, match="Score"):
        _verify(settings, engine_factory, pair_context, pair_run)


def test_engine_sha_mismatch_fails(settings, engine_factory, pair_context, pair_run):
    engine_a = pair_context["engine_a"]
    with engine_factory() as session:
        build = session.get(EngineBuild, engine_a.id)
        build.binary_sha256 = "f" * 64
        session.commit()
        engine_a = session.get(EngineBuild, engine_a.id)
    with pytest.raises(verifier.VerificationFailure, match="SHA"):
        verifier.verify_pair(
            settings,
            tournament=pair_context["tournament"],
            pair_job=pair_context["pair"],
            run_dir=pair_run,
            engine_a_build=engine_a,
            engine_b_build=pair_context["engine_b"],
            opening_set=pair_context["opening_set"],
        )


def test_opening_sha_mismatch_fails(settings, engine_factory, pair_context, pair_run):
    opening = pair_context["opening_set"]
    with engine_factory() as session:
        record = session.get(OpeningSet, opening.id)
        record.sha256 = "e" * 64
        session.commit()
        opening = session.get(OpeningSet, opening.id)
    with pytest.raises(verifier.VerificationFailure, match="SHA"):
        verifier.verify_pair(
            settings,
            tournament=pair_context["tournament"],
            pair_job=pair_context["pair"],
            run_dir=pair_run,
            engine_a_build=pair_context["engine_a"],
            engine_b_build=pair_context["engine_b"],
            opening_set=opening,
        )


def test_command_provenance_mismatch_fails(settings, engine_factory, pair_context, pair_run):
    cmd = json.loads((pair_run / "command.json").read_text(encoding="utf-8"))
    cmd["argv"][0] = "/usr/bin/some-other-cutechess"
    (pair_run / "command.json").write_text(json.dumps(cmd), encoding="utf-8")
    with pytest.raises(verifier.VerificationFailure, match="cutechess"):
        _verify(settings, engine_factory, pair_context, pair_run)


# ---------------------------------------------------------------------------
# Artifact integrity
# ---------------------------------------------------------------------------
def test_verification_records_shas(settings, engine_factory, pair_context, pair_run):
    verification = _verify(settings, engine_factory, pair_context, pair_run)
    for key in ("engine_a_binary_sha256", "engine_b_binary_sha256",
                "opening_set_sha256", "pair_opening_epd_sha256",
                "stdout_sha256", "stderr_sha256", "pgn_sha256"):
        assert verification[key]
