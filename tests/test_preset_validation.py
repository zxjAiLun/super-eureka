"""Server-side preset validation against the probed capability schema
(P4.F1 B4)."""

from __future__ import annotations

import pytest

from chessarena.services.cutechess import CutechessLaunchError
from chessarena.services.preset_validation import validate_preset_uci_options

SCHEMA = {
    "UCI_LimitStrength": {"type": "check"},
    "UCI_Elo": {"type": "spin", "min": 1350, "max": 2850},
    "Style": {"type": "combo", "vars": ["Solid", "Normal", "Very Risky"]},
    "Clear Hash": {"type": "button"},
    "SyzygyPath": {"type": "string"},
    "Hash": {"type": "spin", "min": 1, "max": 1024},
}


def test_valid_check_spin_combo():
    out = validate_preset_uci_options(
        SCHEMA,
        {
            "UCI_LimitStrength": "on",
            "UCI_Elo": "2000",
            "Style": "Very Risky",
        },
    )
    assert out["UCI_LimitStrength"] is True
    assert out["UCI_Elo"] == 2000
    assert out["Style"] == "Very Risky"


def test_check_unchecked_is_false():
    out = validate_preset_uci_options(SCHEMA, {"UCI_LimitStrength": "off"})
    assert out["UCI_LimitStrength"] is False


def test_spin_below_min_rejected():
    with pytest.raises(CutechessLaunchError, match="below engine minimum"):
        validate_preset_uci_options(SCHEMA, {"UCI_Elo": "100"})


def test_spin_above_max_rejected():
    with pytest.raises(CutechessLaunchError, match="above engine maximum"):
        validate_preset_uci_options(SCHEMA, {"UCI_Elo": "9999"})


def test_spin_non_integer_rejected():
    with pytest.raises(CutechessLaunchError, match="must be an integer"):
        validate_preset_uci_options(SCHEMA, {"UCI_Elo": "fast"})


def test_combo_invalid_rejected():
    with pytest.raises(CutechessLaunchError, match="not one of"):
        validate_preset_uci_options(SCHEMA, {"Style": "Bogus"})


def test_combo_value_with_spaces_accepted():
    out = validate_preset_uci_options(SCHEMA, {"Style": "Very Risky"})
    assert out["Style"] == "Very Risky"


def test_string_empty_is_dropped():
    out = validate_preset_uci_options(SCHEMA, {"SyzygyPath": ""})
    assert "SyzygyPath" not in out


def test_string_nonempty_rejected_by_policy():
    with pytest.raises(CutechessLaunchError, match="not editable"):
        validate_preset_uci_options(SCHEMA, {"SyzygyPath": "/data/tablebase"})


def test_button_option_rejected():
    with pytest.raises(CutechessLaunchError, match="cannot be persisted"):
        validate_preset_uci_options(SCHEMA, {"Clear Hash": "on"})


def test_unsupported_option_rejected():
    with pytest.raises(CutechessLaunchError, match="unsupported option"):
        validate_preset_uci_options(SCHEMA, {"Nope": "1"})


def test_runtime_owned_option_rejected():
    with pytest.raises(CutechessLaunchError, match="reserved"):
        validate_preset_uci_options(SCHEMA, {"Hash": "64"})
