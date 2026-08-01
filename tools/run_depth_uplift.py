"""Compare Current and D1.3 at equal fixed time budgets.

This is a measurement tool, not an Elo runner.  Every search uses a fresh UCI
process, the same FEN and Hash setting, and a deterministic interleaving:
``A B / B A / A B / B A / A B``.  The current UCI protocol exposes total
nodes, completed depth, score, bestmove and PV.  It does not expose the
diagnostic qsearch counter, so ``qsearch_nodes`` is reported as null rather
than guessed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from run_external_validation import Case, EngineFailure, EngineSession, SearchResult, parse_info


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT / "target" / "release" / "chess-engine-demo.exe"
DEFAULT_HASH_MB = 16
DEFAULT_MOVETIME_MS = 3000
DEFAULT_REPEATS = 5
DEFAULT_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    fen: str


FIXTURES: tuple[Fixture, ...] = (
    Fixture("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    Fixture("queen-win", "7k/8/8/8/q3Q2p/8/8/4K3 w - - 0 1"),
    Fixture(
        "open-tactical",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
    ),
    Fixture(
        "closed-quiet",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    ),
    Fixture(
        "exposed-king",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 6 5",
    ),
    Fixture(
        "high-branch",
        "r3k2r/pppb1ppp/2np1n2/2q1p3/3pP3/2NP1N2/PPPQBPPP/R3K2R w KQkq - 0 1",
    ),
    Fixture("rook-pawn", "8/8/8/8/8/5k2/5P1K/6R1 w - - 0 1"),
    Fixture("kqk", "7k/8/8/8/8/8/3QK3/8 w - - 0 1"),
    Fixture("krk", "7k/8/8/8/8/8/3RK3/8 w - - 0 1"),
    Fixture(
        "halfmove-ctx",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 80 5",
    ),
)


def _case(fixture: Fixture) -> Case:
    return Case(fixture.fixture_id, "depth-uplift", fixture.fen, "bench", 0, 1)


def _search_movetime(session: EngineSession, case: Case, movetime_ms: int) -> dict[str, Any]:
    """Use the existing D1.11 session lifecycle for a fixed-time search."""

    import chess

    board = chess.Board(case.fen)
    session._send(f"position fen {case.fen}")
    started = time.perf_counter()
    session._send(f"go movetime {movetime_ms}")
    deadline = time.monotonic() + session.timeout_s
    highest = None
    latest_nodes: Optional[int] = None
    time_to_depth_ms: Optional[int] = None
    while True:
        line = session._readline(deadline)
        if line.startswith("info "):
            info = parse_info(line)
            if info is not None:
                if highest is None or info[0] >= highest[0]:
                    highest = info
                tokens = line.split()
                try:
                    time_index = tokens.index("time")
                    parsed_time = int(tokens[time_index + 1])
                except (ValueError, IndexError):
                    parsed_time = None
                if parsed_time is not None:
                    time_to_depth_ms = parsed_time
                try:
                    node_index = tokens.index("nodes")
                    parsed_nodes = int(tokens[node_index + 1])
                except (ValueError, IndexError):
                    parsed_nodes = None
                if parsed_nodes is not None:
                    latest_nodes = max(latest_nodes or 0, parsed_nodes)
        if line.startswith("bestmove "):
            fields = line.split()
            if len(fields) < 2:
                raise EngineFailure(f"{session.profile}: malformed bestmove; {session._context()}")
            bestmove = fields[1]
            break
    if highest is None:
        raise EngineFailure(f"{session.profile}: no scored info line; {session._context()}")
    result = SearchResult(bestmove, highest[0], highest[1], highest[2])
    session._validate_result(board, result)
    return {
        "bestmove": result.bestmove,
        "completed_depth": result.completed_depth,
        "score": {"kind": result.score.kind, "value": result.score.value},
        "pv": list(result.pv),
        "nodes": latest_nodes,
        "qsearch_nodes": None,
        "time_to_depth_ms": time_to_depth_ms,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _run_one(
    engine: Path,
    fixture: Fixture,
    role: str,
    profile: str,
    hash_mb: int,
    movetime_ms: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.time()
    try:
        with EngineSession(engine, profile, hash_mb, timeout_s) as session:
            session.handshake()
            row = _search_movetime(session, _case(fixture), movetime_ms)
        row.update(
            {
                "fixture": fixture.fixture_id,
                "role": role,
                "profile": profile,
                "status": "PASS",
                "host_elapsed_s": round(time.time() - started, 3),
            }
        )
        return row
    except (EngineFailure, OSError, ValueError) as exc:
        return {
            "fixture": fixture.fixture_id,
            "role": role,
            "profile": profile,
            "status": "FAIL",
            "error": str(exc),
            "host_elapsed_s": round(time.time() - started, 3),
        }


def _median(values: list[float]) -> Optional[float]:
    return round(statistics.median(values), 3) if values else None


def _aggregate_fixture(rows: list[dict[str, Any]], fixture_id: str) -> dict[str, Any]:
    rows = [row for row in rows if row.get("fixture") == fixture_id]
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(row["pair"], {})[row["role"]] = row
    deltas: list[float] = []
    time_ratios: list[float] = []
    baseline_depths: list[int] = []
    candidate_depths: list[int] = []
    for pair in by_pair.values():
        baseline = pair.get("baseline")
        candidate = pair.get("candidate")
        if not baseline or not candidate or baseline.get("status") != "PASS" or candidate.get("status") != "PASS":
            continue
        baseline_depths.append(baseline["completed_depth"])
        candidate_depths.append(candidate["completed_depth"])
        deltas.append(candidate["completed_depth"] - baseline["completed_depth"])
        baseline_time = baseline.get("time_to_depth_ms")
        candidate_time = candidate.get("time_to_depth_ms")
        if (
            candidate["completed_depth"] == baseline["completed_depth"]
            and baseline_time is not None
            and candidate_time is not None
            and baseline_time > 0
        ):
            time_ratios.append(candidate_time / baseline_time)
    return {
        "fixture": fixture_id,
        "paired_samples": len(deltas),
        "baseline_depth_median": _median([float(value) for value in baseline_depths]),
        "candidate_depth_median": _median([float(value) for value in candidate_depths]),
        "candidate_minus_baseline_depth_median": _median(deltas),
        "equal_depth_candidate_over_baseline_time_median": _median(time_ratios),
        "qsearch_nodes": None,
    }


def run_depth_uplift(
    engine: Path,
    movetime_ms: int = DEFAULT_MOVETIME_MS,
    repeats: int = DEFAULT_REPEATS,
    hash_mb: int = DEFAULT_HASH_MB,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    fixture_ids: Optional[set[str]] = None,
    report_path: Optional[Path] = None,
    gate_decision: str = "UNDECIDED",
) -> dict[str, Any]:
    if movetime_ms <= 0 or repeats <= 0 or hash_mb <= 0 or timeout_s <= 0:
        raise ValueError("movetime, repeats, hash, and timeout must be positive")
    if gate_decision not in {"PASS", "FAIL", "UNDECIDED"}:
        raise ValueError("gate_decision must be PASS, FAIL, or UNDECIDED")
    fixtures = tuple(f for f in FIXTURES if fixture_ids is None or f.fixture_id in fixture_ids)
    if not fixtures:
        raise ValueError("fixture filter selected no known fixtures")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    profiles = {"baseline": "current", "candidate": "current-qsearch-pruning"}
    for fixture in fixtures:
        for pair in range(1, repeats + 1):
            order = ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
            for role in order:
                row = _run_one(
                    engine,
                    fixture,
                    role,
                    profiles[role],
                    hash_mb,
                    movetime_ms,
                    timeout_s,
                )
                row["pair"] = pair
                row["order"] = " ".join(order)
                rows.append(row)
                if row["status"] != "PASS":
                    errors.append(f"{fixture.fixture_id} pair {pair} {role}: {row['error']}")
    report: dict[str, Any] = {
        "tool": "d1.12-depth-uplift",
        "measurement_status": "FAIL" if errors else "PASS",
        "gate_decision": gate_decision,
        "profiles": profiles,
        "resources": {
            "hash_mb": hash_mb,
            "movetime_ms": movetime_ms,
            "repeats_per_fixture": repeats,
            "schedule": "A B / B A / A B / B A / A B" if repeats == 5 else "alternating A/B",
            "fresh_process_per_search": True,
        },
        "fixtures": [asdict(fixture) for fixture in fixtures],
        "searches_expected": len(fixtures) * repeats * 2,
        "searches_completed": sum(row["status"] == "PASS" for row in rows),
        "qsearch_nodes_observability": "not exposed by current UCI info",
        "rows": rows,
        "fixture_summary": [_aggregate_fixture(rows, fixture.fixture_id) for fixture in fixtures],
        "errors": errors,
        "notes": [
            "This is a fixed-time depth measurement, not an Elo or promotion decision.",
            "qsearch_nodes is null because the current UCI protocol exposes total nodes only.",
            "Equal-depth timing uses the UCI info time at the highest completed depth, not only bestmove wall time.",
        ],
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--movetime-ms", type=int, default=DEFAULT_MOVETIME_MS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--hash-mb", type=int, default=DEFAULT_HASH_MB)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--fixture", action="append", dest="fixtures")
    parser.add_argument("--gate-decision", choices=("PASS", "FAIL", "UNDECIDED"), default="UNDECIDED")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_depth_uplift(
            engine=args.engine,
            movetime_ms=args.movetime_ms,
            repeats=args.repeats,
            hash_mb=args.hash_mb,
            timeout_s=args.timeout_s,
            fixture_ids=set(args.fixtures) if args.fixtures else None,
            report_path=args.report,
            gate_decision=args.gate_decision,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"measurement_status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["measurement_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
