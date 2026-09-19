"""Freeze three already-inspected cloud cases from a local read-only snapshot.

This does not contact the server or run any chess engine. python-chess is used
only to verify histories, forced branches, legal root moves and checkmate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import chess

CASES = [
    (127, 49, [
        ("after_Nxe5", ["Nxe5"]),
        ("after_Bxe5", ["Nxe5", "Bxe5"]),
        ("after_Qxe5", ["Nxe5", "Bxe5", "Qxe5"]),
        ("after_Qf3_check", ["Nxe5", "Bxe5", "Qxe5", "Qf3+"]),
        ("pv_endpoint_Kg1", ["Nxe5", "Bxe5", "Qxe5", "Qf3+", "Kg1"]),
        ("alternative_evasion_Kh2", ["Nxe5", "Bxe5", "Qxe5", "Qf3+", "Kh2"]),
        ("terminal_Qg2_mate", ["Nxe5", "Bxe5", "Qxe5", "Qf3+", "Kg1", "Qg2#"]),
        ("actual_Kh2", ["Nxe5", "Bxe5", "Kh2"]),
        ("diagnostic_choice_Qe3", ["Qe3"]),
    ]),
    (175, 55, [
        ("after_Qc1", ["Qc1"]),
        ("after_Qxf3", ["Qc1", "Qxf3"]),
        ("diagnostic_choice_Ne1", ["Ne1"]),
    ]),
    (187, 7, [
        ("after_Nd5", ["Nd5"]),
        ("after_exd5_black", ["Nd5", "exd5"]),
        ("after_exd5_white", ["Nd5", "exd5", "exd5"]),
        ("diagnostic_choice_Rhd1", ["Rhd1"]),
    ]),
]


def position_record(board):
    return {
        "fen": board.fen(), "white_to_move": board.turn == chess.WHITE,
        "legal_moves": sorted(m.uci() for m in board.legal_moves),
        "in_check": board.is_check(), "checkmate": board.is_checkmate(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    searches = json.loads((args.snapshot_dir / "selected-searches.json").read_text())
    rows = json.loads((args.snapshot_dir / "move-scan.json").read_text())
    snap_path = args.snapshot_dir / "s14-cloud-snapshot.json"
    snap = json.loads(snap_path.read_text())
    cases = []
    for number, ply, branches in CASES:
        row = next(r for r in rows if r["game"] == number and r["ply"] == ply)
        record = next(r for r in snap["records"] if r["game"]["game_number"] == number)
        log_path = record["game"]["pgn_path"].replace("match.pgn", "stdout.log")
        matched = [s for s in searches if s["fen"] == row["fen"] and s["path"] == log_path]
        assert len(matched) == 1
        s = matched[0]
        board = chess.Board(s["initial_fen"])
        for uci in s["history"]:
            board.push_uci(uci)
        assert board.fen() == row["fen"] and len(s["history"]) == ply - 1
        assert s["bestmove"] == row["uci"]
        forced = []
        for name, sans in branches:
            b = board.copy(stack=True)
            moves = []
            for san in sans:
                move = b.parse_san(san)
                moves.append(move.uci())
                b.push(move)
            forced.append({"name": name, "forced_san": sans, "forced_uci": moves,
                           **position_record(b)})
        cases.append({
            "game_number": number, "game_id": row["id"], "ply": ply,
            "initial_fen": s["initial_fen"], "history_uci": s["history"],
            "root": position_record(board), "historical_go": s["go"],
            "historical_bestmove": s["bestmove"],
            "historical_elapsed_ms": s["elapsed_ms"], "historical_infos": s["infos"],
            "diagnostic_before_white_pawns": row["sf_before"],
            "diagnostic_after_white_pawns": row["sf_after"],
            "diagnostic_sha256": record["analysis_sha256"],
            "branches": forced,
        })
    plan = {
        "schema_version": 1,
        "tournament_id": snap["tournament_id"],
        "source_commit": "315534991628ce084c34392d6abb98d71f3f591a",
        "binary_sha256": "c2428a8453d871c9a6f2ff14780f077b2c2503bdb321419d4334361c54733986",
        "model_sha256": "329b717066082798d2f42d9e4f137385a1482a8cda03ae7849ddc654e097e4b0",
        "snapshot_sha256": hashlib.sha256(snap_path.read_bytes()).hexdigest(),
        "hash_mb": 16, "threads": 1,
        "tt_policy": "fresh process per search; full position history; NOT original warm TT",
        "clock_policy": "local WSL CPU, not a cloud wall-time replication",
        "root_time_ms": [300, 1000, 3000, 10000],
        "root_depths": [4, 5, 6, 7, 8],
        "depth_time_cap_ms": 15000,
        "branch_depths": [1, 2, 4, 6],
        "branch_time_cap_ms": 5000,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    with args.out.open("x", encoding="utf-8") as f:
        f.write(text)
    print("cases", [(c["game_number"], len(c["history_uci"]), len(c["branches"])) for c in cases])
    print("manifest sha256", hashlib.sha256(args.out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
