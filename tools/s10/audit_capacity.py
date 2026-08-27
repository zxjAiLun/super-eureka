import json
from pathlib import Path
import chess.pgn

def audit_pgn_capacity():
    sources_dir = Path("data/s6/sources")
    # Arena sources in data/s6/sources
    source_manifest = json.loads((sources_dir / "source_manifest.json").read_text())
    
    arena_games = 0
    arena_positions = 0
    for name, info in source_manifest.items():
        pgn_file = sources_dir / f"{name}.pgn"
        if not pgn_file.exists():
            continue
        with open(pgn_file, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                arena_games += 1
                # rough count of moves
                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    board.push(move)
                    ply += 1
                    if 12 <= ply <= 160:
                        arena_positions += 1

    print(f"Arena Sources: {arena_games} games, ~{arena_positions} eligible plies")

    # Lichess sources
    for lichess_dir in [sources_dir / "lichess-standard-rated-v1", sources_dir / "lichess-standard-rated-confirm-v1-g1400"]:
        if not lichess_dir.exists():
            continue
        l_manifest = json.loads((lichess_dir / "source-manifest.json").read_text())
        pgn_file = lichess_dir / f"{l_manifest['source_id']}.pgn"
        l_games = 0
        l_positions = 0
        with open(pgn_file, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                l_games += 1
                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    board.push(move)
                    ply += 1
                    if 12 <= ply <= 160:
                        l_positions += 1
        print(f"Lichess Source ({l_manifest['source_id']}): {l_games} games, ~{l_positions} eligible plies")

if __name__ == "__main__":
    audit_pgn_capacity()
