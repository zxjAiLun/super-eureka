"""UCI handshake probe tests (P4.2 Phase B).

Covers parsing, required-option enforcement, malformed lines and the
deterministic option rendering / reserved-option contracts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from chessarena.services.cutechess import (
    RESERVED_OPTIONS,
    validate_preset_options,
    uci_option_args,
)
from chessarena.services.uci_probe import (
    UciProbeError,
    parse_option_line,
    probe_uci,
    require_option,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_UCI_ENGINE = FIXTURES / "fake_uci_engine.py"


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
    args = uci_option_args(
        {"uci_options": {"UCI_Elo": 2000, "UCI_LimitStrength": True, "Ponder": False}},
        hash_mb=32,
        threads=1,
    )
    # Sorted by name: Hash, Ponder, Threads, UCI_Elo, UCI_LimitStrength
    names = [a.split("=")[0] for a in args]
    assert names == sorted(names)
    assert "option.UCI_LimitStrength=true" in args
    assert "option.Ponder=false" in args
    assert "option.Hash=32" in args
    assert "option.Threads=1" in args


def test_reserved_options_rejected():
    with pytest.raises(Exception, match="reserved options"):
        validate_preset_options({"UCI_LimitStrength": True, "Hash": 64})
    assert RESERVED_OPTIONS == frozenset({"Hash", "Threads", "Ponder"})
    # Non-conflicting preset options pass.
    validate_preset_options({"UCI_LimitStrength": True, "UCI_Elo": 2000})
