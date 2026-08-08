"""Server-side EnginePreset UCI-option validation against the probed
capability schema (P4.F1 B4).

Never trusts HTML input: every submitted preset option is re-validated
against the exact EngineBuild capability schema.  Arena-owned runtime
options (Hash/Threads/Ponder/OwnBook/UCI_Chess960) and button options are
rejected; combo values must be exact schema vars; spin values must be
in-range integers; string options are not editable through the web surface
(empty submissions are dropped so the engine default applies).
"""

from __future__ import annotations

from typing import Any

from .cutechess import CutechessLaunchError, RESERVED_OPTIONS

# String options are display-only on the web surface.  Extending this set
# must be deliberate (paths/network addresses are sensitive); an empty
# allowlist means no string option value can be persisted via the editor.
ALLOWED_STRING_VALUES: dict[str, set[str]] = {}


def _as_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "on", "1"):
            return True
        if lowered in ("false", "off", "0", ""):
            return False
    raise CutechessLaunchError(f"check option must be a boolean, got {raw!r}")


def validate_preset_uci_options(schema: dict, submitted: dict) -> dict:
    """Validate ``submitted`` ({option name: raw form value}) against the
    build's UCI capability schema and return the sanitized uci_options dict
    (only non-reserved, supported options with validated values).
    """
    result: dict[str, Any] = {}
    for name, raw in submitted.items():
        if name in RESERVED_OPTIONS:
            raise CutechessLaunchError(
                f"preset must not set reserved option: {name}"
            )
        decl = schema.get(name)
        if decl is None:
            raise CutechessLaunchError(
                f"unsupported option for this build: {name}"
            )
        typ = decl.get("type")
        if typ == "button":
            raise CutechessLaunchError(
                f"button option {name} cannot be persisted as a preset value"
            )
        if typ == "check":
            result[name] = _as_bool(raw)
        elif typ == "spin":
            try:
                value = int(raw)
            except (TypeError, ValueError):
                raise CutechessLaunchError(
                    f"spin option {name} must be an integer, got {raw!r}"
                ) from None
            lo, hi = decl.get("min"), decl.get("max")
            if lo is not None and value < lo:
                raise CutechessLaunchError(
                    f"{name}={value} below engine minimum {lo}"
                )
            if hi is not None and value > hi:
                raise CutechessLaunchError(
                    f"{name}={value} above engine maximum {hi}"
                )
            result[name] = value
        elif typ == "combo":
            vars_ = decl.get("vars") or []
            if raw not in vars_:
                raise CutechessLaunchError(
                    f"combo option {name}: {raw!r} not one of {vars_}"
                )
            result[name] = raw
        elif typ == "string":
            allowed = ALLOWED_STRING_VALUES.get(name)
            if raw and (allowed is None or raw not in allowed):
                raise CutechessLaunchError(
                    f"string option {name} is not editable via the web "
                    "surface; leave empty to use the engine default"
                )
            if raw:
                result[name] = raw
        else:
            raise CutechessLaunchError(
                f"unsupported UCI option type {typ!r} for {name}"
            )
    return result
