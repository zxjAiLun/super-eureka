"""UCI handshake probe tests (P4.2 Phase B).

Covers parsing, required-option enforcement, malformed lines and the
deterministic option rendering / reserved-option contracts.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from chessarena.services.cutechess import (
    RESERVED_OPTIONS,
    engine_option_args,
    validate_preset_options,
)
from chessarena.services.uci_probe import (
    UciProbeError,
    parse_option_line,
    probe_uci,
    require_option,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_UCI_ENGINE = FIXTURES / "fake_uci_engine.py"
FAKE_UCI_HANG = FIXTURES / "fake_uci_hang.py"
FAKE_UCI_PARTIAL = FIXTURES / "fake_uci_partial.py"


def _probe(env_extra: dict | None = None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return probe_uci(FAKE_UCI_ENGINE, timeout=15)


def test_parse_option_lines():
    check = parse_option_line("option name UCI_LimitStrength type check default false")
    assert check is not None
    assert check.name == "UCI_LimitStrength"
    assert check.type == "check"
    assert check.default == "false"

    spin = parse_option_line(
        "option name UCI_Elo type spin default 1350 min -200 max 2850"
    )
    assert spin is not None
    assert spin.type == "spin"
    assert spin.min == -200
    assert spin.max == 2850

    assert parse_option_line("id name FakeStockfish 17.1") is None
    assert parse_option_line("uciok") is None
    # Sanity check that the parsed options are queryable via require_option.


def test_malformed_option_line_rejected():
    with pytest.raises(UciProbeError):
        parse_option_line("option name bad\x00name type spin default 1")


def test_probe_uci_handshake_ok():
    result = _probe()
    assert result.id_name == "FakeStockfish 17.1"
    assert set(result.options) == {
        "UCI_LimitStrength",
        "UCI_Elo",
        "Hash",
        "Threads",
        "Ponder",
        "Style",
        "Clear Hash",
        "SyzygyPath",
        "My Custom Option",
        "Move Overhead",
    }


def test_require_option_passes_for_known_options():
    result = _probe()
    elo = require_option(result, "UCI_Elo", "spin")
    assert elo.min == -200
    assert elo.max == 2850


def test_require_option_missing_rejected():
    result = _probe()
    with pytest.raises(UciProbeError, match="missing required UCI option"):
        require_option(result, "Skill Level", "spin")


def test_require_option_wrong_type_rejected():
    result = _probe()
    with pytest.raises(UciProbeError, match="expected 'check'"):
        require_option(result, "UCI_Elo", "check")


def test_probe_missing_binary(tmp_path):
    with pytest.raises(UciProbeError, match="binary not found"):
        probe_uci(tmp_path / "nope")


def test_uci_option_args_are_sorted_and_bool_lowercase():
    args = engine_option_args(
        {"uci_options": {"UCI_Elo": 2000, "UCI_LimitStrength": True, "Ponder": False}}
    )
    # Sorted by name: Ponder, UCI_Elo, UCI_LimitStrength
    names = [a.split("=")[0] for a in args]
    assert names == sorted(names)
    assert "option.UCI_LimitStrength=true" in args
    assert "option.Ponder=false" in args
    assert "option.UCI_Elo=2000" in args


def test_runtime_options_sent_per_capability():
    """B5: Hash/Threads are sent per-engine only when the engine's probed
    schema declares them."""
    engine = {
        "uci_options": {"UCI_Elo": 2000},
        "uci_options_schema": {"Hash": {"type": "spin"}, "Threads": {"type": "spin"}},
    }
    args = engine_option_args(engine, hash_mb=32, threads=1)
    assert "option.UCI_Elo=2000" in args
    assert "option.Hash=32" in args
    assert "option.Threads=1" in args

    # An engine that does not declare Hash/Threads gets neither.
    plain = {"uci_options": {}, "uci_options_schema": {}}
    args2 = engine_option_args(plain, hash_mb=32, threads=1)
    assert args2 == []


def test_reserved_options_rejected():
    with pytest.raises(Exception, match="reserved options"):
        validate_preset_options({"UCI_LimitStrength": True, "Hash": 64})
    assert RESERVED_OPTIONS == frozenset(
        {"Hash", "Threads", "Ponder", "OwnBook", "UCI_Chess960"}
    )
    # Non-conflicting preset options pass.
    validate_preset_options({"UCI_LimitStrength": True, "UCI_Elo": 2000})


def test_probe_times_out_when_engine_hangs():
    """P1 regression: an engine that receives uci and never outputs must
    fail at the real deadline and be reaped immediately (no extra grace
    wait)."""
    start = time.monotonic()
    with pytest.raises(UciProbeError, match="timed out"):
        probe_uci(FAKE_UCI_HANG, timeout=2)
    elapsed = time.monotonic() - start
    assert elapsed < 4, f"deadline not enforced: {elapsed:.1f}s"


def test_probe_times_out_on_partial_line_without_newline():
    """P1 regression: a partial line without a newline must also hit the
    real deadline (readline alone would block forever)."""
    start = time.monotonic()
    with pytest.raises(UciProbeError, match="timed out"):
        probe_uci(FAKE_UCI_PARTIAL, timeout=2)
    elapsed = time.monotonic() - start
    assert elapsed < 4, f"deadline not enforced: {elapsed:.1f}s"


def test_parse_option_combo_with_vars():
    opt = parse_option_line(
        "option name Style type combo default Normal var Solid var Normal var Risky"
    )
    assert opt is not None
    assert opt.name == "Style"
    assert opt.type == "combo"
    assert opt.default == "Normal"
    assert opt.vars == ["Solid", "Normal", "Risky"]


def test_parse_option_string_default_with_spaces():
    opt = parse_option_line(
        "option name SyzygyPath type string default <empty>"
    )
    assert opt is not None
    assert opt.type == "string"
    assert opt.default == "<empty>"
    opt2 = parse_option_line(
        "option name My Custom Option type string default some default value"
    )
    assert opt2 is not None
    assert opt2.name == "My Custom Option"
    assert opt2.default == "some default value"


def test_parse_option_button_has_no_default():
    opt = parse_option_line("option name Clear Hash type button")
    assert opt is not None
    assert opt.type == "button"
    assert opt.default is None
    assert opt.min is None and opt.max is None


def test_parse_option_name_with_spaces_and_minmax():
    opt = parse_option_line(
        "option name Move Overhead type spin default 10 min 0 max 5000"
    )
    assert opt is not None
    assert opt.name == "Move Overhead"
    assert opt.default == "10"
    assert opt.min == 0
    assert opt.max == 5000


def test_probe_captures_full_option_schema():
    result = probe_uci(FAKE_UCI_ENGINE, timeout=10)
    opts = result.options
    assert opts["Style"].vars == ["Solid", "Normal", "Risky"]
    assert opts["Style"].type == "combo"
    assert opts["Clear Hash"].type == "button"
    assert opts["My Custom Option"].default == "some default value"
    assert opts["Move Overhead"].min == 0
    assert opts["Move Overhead"].max == 5000


def test_parse_option_combo_var_with_spaces():
    opt = parse_option_line(
        "option name Style type combo default Normal var Very Solid var Normal var Very Risky"
    )
    assert opt is not None
    assert opt.type == "combo"
    assert opt.default == "Normal"
    assert opt.vars == ["Very Solid", "Normal", "Very Risky"]


def test_parse_option_spin_default_still_single_token():
    opt = parse_option_line(
        "option name Move Overhead type spin default 10 min 0 max 5000"
    )
    assert opt.default == "10"
    assert opt.min == 0 and opt.max == 5000


def test_parse_option_string_default_with_embedded_keyword():
    opt = parse_option_line(
        "option name Notes type string default default value here"
    )
    assert opt.type == "string"
    assert opt.default == "default value here"


def test_runtime_policy_options_sent_only_when_declared():
    engine = {
        "uci_options": {},
        "uci_options_schema": {
            "Ponder": {"type": "check"},
            "OwnBook": {"type": "check"},
            "UCI_Chess960": {"type": "check"},
        },
    }
    args = engine_option_args(
        engine, ponder=False, ownbook=False, chess960=False
    )
    assert "option.Ponder=false" in args
    assert "option.OwnBook=false" in args
    assert "option.UCI_Chess960=false" in args


def test_runtime_policy_options_omitted_when_not_declared():
    engine = {"uci_options": {}, "uci_options_schema": {}}
    args = engine_option_args(
        engine, hash_mb=32, threads=1, ponder=False, ownbook=False,
        chess960=False,
    )
    assert args == []


def test_hash_out_of_range_rejected_before_launch():
    engine = {
        "uci_options": {},
        "uci_options_schema": {"Hash": {"type": "spin", "min": 1, "max": 1024}},
    }
    with pytest.raises(Exception, match="above engine maximum"):
        engine_option_args(engine, hash_mb=2048)


def test_threads_under_min_rejected():
    engine = {
        "uci_options": {},
        "uci_options_schema": {"Threads": {"type": "spin", "min": 1, "max": 8}},
    }
    with pytest.raises(Exception, match="below engine minimum"):
        engine_option_args(engine, threads=0)


def test_wrong_reserved_option_type_rejected():
    engine = {
        "uci_options": {},
        "uci_options_schema": {"Hash": {"type": "check"}},
    }
    with pytest.raises(Exception, match="declares type"):
        engine_option_args(engine, hash_mb=32)
