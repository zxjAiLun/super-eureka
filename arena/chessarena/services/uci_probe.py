"""UCI handshake probing for generic UCI engines (P4.2 Phase B).

Unlike the project engine, external engines (e.g. Stockfish) do not take a
``--profile`` argument; their behavior is configured through UCI options
(``setoption``).  This module drives a real UCI handshake:

- sends ``uci`` and parses ``option name ... type ...`` lines plus ``id name``,
- requires ``uciok``,
- sends ``isready`` and requires ``readyok``,
- sends ``quit`` and requires the process to exit promptly.
"""

from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Full UCI option line: name may contain spaces; type is check|spin|combo|
# button|string; after the type come optional default/min/max/var clauses.
_OPTION_LINE = re.compile(
    r"^option name (?P<name>.+?)\s+type\s+(?P<type>\w+)\s*(?P<attrs>.*)$"
)

# A clause value runs to the next clause keyword (var/default/min/max) or the
# end of the line — so combo values and string defaults may contain spaces.
_NEXT_CLAUSE = r"(?=\s+var\s|\s+default\s|\s+min\s|\s+max\s|$)"
_DEFAULT_RE = re.compile(r"\bdefault\s+(.+?)" + _NEXT_CLAUSE)
_VAR_RE = re.compile(r"\bvar\s+(.+?)" + _NEXT_CLAUSE)
_MIN_RE = re.compile(r"\bmin\s+(-?\d+)")
_MAX_RE = re.compile(r"\bmax\s+(-?\d+)")


def _parse_option_attrs(typ: str, attrs: str):
    """Extract default/min/max/var from the tail of an option line.

    combo ``var`` values may contain spaces and run until the next
    ``var``/``default``/``min``/``max`` keyword; ``string`` defaults may
    contain spaces up to the end of the line; spin uses a single token
    followed by min/max.
    """
    attrs = attrs.strip()
    default: Optional[str] = None
    if typ == "string":
        m = re.search(r"\bdefault\s+(.*?)\s*$", attrs)
        if m:
            default = m.group(1).strip() or None
    else:
        m = _DEFAULT_RE.search(attrs)
        if m:
            default = m.group(1).strip()
    vars_ = [v.strip() for v in _VAR_RE.findall(attrs)]
    lo = hi = None
    m = _MIN_RE.search(attrs)
    if m:
        lo = int(m.group(1))
    m = _MAX_RE.search(attrs)
    if m:
        hi = int(m.group(1))
    return default, lo, hi, vars_


class UciProbeError(RuntimeError):
    pass


@dataclass
class UciOption:
    name: str
    type: str
    default: Optional[str] = None
    min: Optional[int] = None
    max: Optional[int] = None
    vars: list[str] = field(default_factory=list)


@dataclass
class UciProbeResult:
    id_name: str
    options: dict[str, UciOption] = field(default_factory=dict)


def parse_option_line(line: str) -> Optional[UciOption]:
    """Parse a ``option name ... type ...`` UCI line, or None if not one."""
    m = _OPTION_LINE.match(line.strip())
    if not m:
        return None
    name = m.group("name")
    typ = m.group("type")
    if name is None or "\n" in name or "\x00" in name:
        raise UciProbeError(f"malformed option name: {line!r}")
    default, lo, hi, vars_ = _parse_option_attrs(typ, m.group("attrs"))
    return UciOption(
        name=name, type=typ, default=default, min=lo, max=hi, vars=vars_
    )


def _is_python_script(binary: Path) -> bool:
    try:
        with open(binary, "rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return head.startswith(b"#!") and b"python" in head


def probe_uci(binary: Path, timeout: float = 15.0) -> UciProbeResult:
    """Run a UCI handshake against ``binary`` and return the probed identity
    and options.  Raises UciProbeError on any contract violation.

    A reader thread drains stdout into a queue so ``timeout`` is a real
    deadline: even if the engine hangs without output (or writes a partial
    line without a newline), the main thread wakes via queue.get(timeout)
    and the process is killed.  This works on Windows (no select on pipes).
    """
    if not binary.is_file():
        raise UciProbeError(f"binary not found: {binary}")
    # Test fixtures may be plain python engines that Windows cannot exec
    # directly; launch them through the current interpreter.
    cmd = (
        [sys.executable, str(binary)]
        if _is_python_script(binary)
        else [str(binary)]
    )
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError as exc:
        raise UciProbeError(f"cannot launch {binary}: {exc}") from exc

    assert proc.stdin is not None and proc.stdout is not None

    lines: "queue.Queue[str | None]" = queue.Queue()

    def _reader() -> None:
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)  # EOF sentinel

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    deadline = _deadline(timeout)

    def _next_line() -> str | None:
        """Blocking read bounded by the real deadline."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise UciProbeError("UCI handshake timed out")
        try:
            item = lines.get(timeout=remaining)
        except queue.Empty:
            raise UciProbeError("UCI handshake timed out")
        return item  # None on EOF

    id_name: str | None = None
    options: dict[str, UciOption] = {}
    saw_uciok = False
    saw_readyok = False
    handshake_completed = False
    try:
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        while not saw_uciok:
            line = _next_line()
            if line is None:
                raise UciProbeError("engine closed stdout before uciok")
            line = line.strip()
            if line.startswith("id name "):
                id_name = line[len("id name "):].strip()
            elif line.startswith("option name"):
                opt = parse_option_line(line)
                if opt is not None:
                    options[opt.name] = opt
            elif line == "uciok":
                saw_uciok = True

        proc.stdin.write("isready\n")
        proc.stdin.flush()
        while not saw_readyok:
            line = _next_line()
            if line is None:
                raise UciProbeError("engine closed stdout before readyok")
            if line.strip() == "readyok":
                saw_readyok = True

        proc.stdin.write("quit\n")
        proc.stdin.flush()
        handshake_completed = True
    except BrokenPipeError as exc:
        raise UciProbeError("engine closed its input early") from exc
    finally:
        if handshake_completed:
            # Normal path: quit was accepted; give it a short grace period.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        else:
            # Error/timeout path: reap immediately — no extra grace wait.
            proc.kill()
            proc.wait(timeout=5)

    if not saw_uciok:
        raise UciProbeError("uciok not received")
    if not saw_readyok:
        raise UciProbeError("readyok not received after isready")
    if id_name is None:
        raise UciProbeError("no 'id name' in UCI output")
    return UciProbeResult(id_name=id_name, options=options)


def require_option(
    result: UciProbeResult, name: str, expected_type: str
) -> UciOption:
    opt = result.options.get(name)
    if opt is None:
        raise UciProbeError(f"engine missing required UCI option {name!r}")
    if opt.type != expected_type:
        raise UciProbeError(
            f"UCI option {name!r} has type {opt.type!r}, expected {expected_type!r}"
        )
    return opt


def _deadline(timeout: float) -> float:
    return time.monotonic() + timeout
