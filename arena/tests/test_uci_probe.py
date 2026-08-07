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
    each_option_args,
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

    common = each_option_args(hash_mb=32, threads=1)
    assert common == ["option.Hash=32", "option.Threads=1"]


def test_reserved_options_rejected():
    with pytest.raises(Exception, match="reserved options"):
        validate_preset_options({"UCI_LimitStrength": True, "Hash": 64})
    assert RESERVED_OPTIONS == frozenset({"Hash", "Threads", "Ponder"})
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
