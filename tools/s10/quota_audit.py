import json
from pathlib import Path
import chess
import chess.pgn

def detailed_quota_audit():
    # Load all 3 source dirs
    sources = [
        Path("data/s6/sources"),
        Path("data/s6/sources/lichess-standard-rated-v1"),
        Path("data/s6/sources/lichess-standard-rated-confirm-v1-g1400")
    ]
    # Max per game = 8
    max_per_game = 8
    total_games = 1572 + 2000 + 1400
    max_possible_positions = total_games * max_per_game
    print(f"Total games across available local sources: {total_games}")
    print(f"Max possible sampled positions (at 8/game): {max_possible_positions}")
    print(f"Required target for s10-eval-v1-300k01: 300,000")
    print(f"Deficit: {300000 - max_possible_positions} positions")

if __name__ == "__main__":
    detailed_quota_audit()
