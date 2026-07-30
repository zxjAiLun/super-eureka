#!/usr/bin/env python3
"""Filter existing Fastchess postmortem JSONL for non-mate promotion races.

This is a secondary report over ``analyze_fastchess_pgn.py`` output.  It does
not re-analyze games, change a decision, or label a move as an objective
blunder.  In particular, old records may contain the historical ``latency_ms``
key; output always uses the explicit ``fastchess_latency_delta_ms`` name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"record on line {line_number} is not an object")
            yield record


def _selected(record: dict[str, Any], minimum_loss_cp: int) -> bool:
    flags = record.get("flags") or {}
    return bool(
        flags.get("promotion_race")
        and not flags.get("mate_transition")
        and int(record.get("eval_loss_cp", 0)) >= minimum_loss_cp
    )


def _output_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    if "fastchess_latency_delta_ms" not in result:
        result["fastchess_latency_delta_ms"] = result.pop("latency_ms", None)
    else:
        result.pop("latency_ms", None)
    return result


def select_records(
    input_path: Path, minimum_loss_cp: int = 150
) -> tuple[list[dict[str, Any]], int]:
    all_records = list(_records(input_path))
    selected = [
        _output_record(record)
        for record in all_records
        if _selected(record, minimum_loss_cp)
    ]
    selected.sort(key=lambda record: int(record["eval_loss_cp"]), reverse=True)
    return selected, len(all_records)


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _promotion_summary(records: list[dict[str, Any]], input_records: int, minimum_loss_cp: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input_records": input_records,
        "selected_records": len(records),
        "minimum_loss_cp": minimum_loss_cp,
        "filters": {
            "promotion_race": True,
            "mate_transition": False,
            "eval_loss_cp_gte": minimum_loss_cp,
        },
        "time_pressure_status": "unknown when source records have no time_left_ms",
    }


def _markdown(records: list[dict[str, Any]], minimum_loss_cp: int, top: int) -> str:
    lines = [
        "# Promotion-race candidates",
        "",
        "Diagnostic screening of existing analyzer JSONL; not objective blunder truth.",
        "",
        f"Filter: `promotion_race=true`, `mate_transition=false`, `eval_loss_cp >= {minimum_loss_cp}`.",
        "",
        "| Game | Ply | Mover | Move | Loss cp | Eval before → after | Depth | Think s | PV | Passed pawns before → after | Promotion distance before → after | FEN before |",
        "| ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for record in records[:top]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    record.get("game"),
                    record.get("ply"),
                    record.get("mover"),
                    f"{record.get('san')} ({record.get('uci')})",
                    record.get("eval_loss_cp"),
                    f"{record.get('eval_before_raw')} → {record.get('eval_after_raw')}",
                    record.get("depth"),
                    record.get("think_time_s"),
                    record.get("pv"),
                    f"{record.get('passed_pawns_before')} → {record.get('passed_pawns_after')}",
                    f"{record.get('promotion_distance_before')} → {record.get('promotion_distance_after')}",
                    record.get("fen_before"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The JSONL output retains full source records, including clock fields, "
            "`fastchess_latency_delta_ms`, PV, FEN, and passed-pawn positions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(
    input_path: Path,
    output_dir: Path,
    minimum_loss_cp: int = 150,
    top: int = 20,
) -> dict[str, Any]:
    records, input_count = select_records(input_path, minimum_loss_cp)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _promotion_summary(records, input_count, minimum_loss_cp)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "promotion-races.jsonl").open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(
        _markdown(records, minimum_loss_cp, top), encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--min-loss-cp", type=int, default=150)
    result.add_argument("--top", type=int, default=20)
    return result


def main() -> int:
    args = parser().parse_args()
    summary = write_report(args.input, args.output_dir, args.min_loss_cp, args.top)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
