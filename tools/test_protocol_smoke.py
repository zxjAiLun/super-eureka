import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import chess

from run_protocol_smoke import (
    DEFAULT_OPENINGS,
    DiagnosticPentanomialState,
    EngineMoveTimeout,
    EngineProtocolError,
    EngineSession,
    EngineStartupTimeout,
    Opening,
    PAIR_CATEGORIES,
    ScheduledGame,
    TournamentError,
    _terminal_result,
    build_schedule,
    candidate_result_for_game,
    load_openings,
    pair_category,
    pair_score_from_results,
    parse_bestmove_line,
    parser,
    play_game,
    run_tournament,
    score_statistics,
    sha256_file,
    validate_bestmove,
)


class TournamentUnitTests(unittest.TestCase):
    @staticmethod
    def fake_engine(
        mode: str, delay: float = 0.0, reported_profile: str | None = None
    ) -> list[str]:
        script = f"""
import sys
import time
mode = {mode!r}
delay = {delay!r}
reported_profile = {reported_profile!r}
if reported_profile is None:
    reported_profile = sys.argv[sys.argv.index('--profile') + 1] if '--profile' in sys.argv else 'current'
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name E1Fake', flush=True)
        print('id author E1Tests', flush=True)
        print('info string search profile ' + reported_profile, flush=True)
        if mode != 'no-uciok':
            print('uciok', flush=True)
    elif line == 'isready':
        if mode != 'no-readyok':
            print('readyok', flush=True)
    elif line.startswith('go'):
        if mode == 'slow':
            time.sleep(delay)
        print('bestmove a2a3', flush=True)
    elif line == 'quit':
        break
"""
        return [sys.executable, "-u", "-c", script]

    @staticmethod
    def legal_move_engine(delay: float = 0.0, illegal: bool = False) -> list[str]:
        script = f"""
import chess
import sys
import time
delay = {delay!r}
illegal = {illegal!r}
reported_profile = sys.argv[sys.argv.index('--profile') + 1] if '--profile' in sys.argv else 'current'
board = chess.Board()
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name E1LegalFake', flush=True)
        print('id author E1Tests', flush=True)
        print('info string search profile ' + reported_profile, flush=True)
        print('uciok', flush=True)
    elif line == 'isready':
        print('readyok', flush=True)
    elif line.startswith('position'):
        board = chess.Board()
        fields = line.split()
        if 'moves' in fields:
            for move_uci in fields[fields.index('moves') + 1:]:
                board.push_uci(move_uci)
    elif line.startswith('go'):
        time.sleep(delay)
        move = 'a1a8' if illegal else next(iter(board.legal_moves)).uci()
        print('bestmove ' + move, flush=True)
    elif line == 'quit':
        break
"""
        return [sys.executable, "-u", "-c", script]

    @staticmethod
    def startup_failure_engine(mode: str, marker: Path) -> list[str]:
        script = f"""
import chess
from pathlib import Path
import sys
marker = Path({str(marker)!r})
try:
    count = int(marker.read_text()) + 1
except FileNotFoundError:
    count = 1
marker.write_text(str(count))
fail = count >= 2
reported_profile = sys.argv[sys.argv.index('--profile') + 1] if '--profile' in sys.argv else 'current'
board = chess.Board()
for raw in sys.stdin:
    line = raw.strip()
    if line == 'uci':
        print('id name E1StartupFake', flush=True)
        print('id author E1Tests', flush=True)
        print('info string search profile ' + reported_profile, flush=True)
        if not (fail and {mode!r} == 'uciok'):
            print('uciok', flush=True)
    elif line == 'isready':
        if not (fail and {mode!r} == 'readyok'):
            print('readyok', flush=True)
    elif line.startswith('position'):
        board = chess.Board()
        fields = line.split()
        if 'moves' in fields:
            for move_uci in fields[fields.index('moves') + 1:]:
                board.push_uci(move_uci)
    elif line.startswith('go'):
        print('bestmove ' + next(iter(board.legal_moves)).uci(), flush=True)
    elif line == 'quit':
        break
"""
        return [sys.executable, "-u", "-c", script]

    @staticmethod
    def tournament_args(
        engine_a: list[str], engine_b: list[str], output_dir: Path, games: int = 4
    ):
        args = parser().parse_args(
            [
                "--engine-a",
                "baseline-test",
                "--engine-b",
                "candidate-test",
                "--profile-b",
                "current-aspiration",
            ]
        )
        args.engine_a = engine_a
        args.engine_b = engine_b
        args.output_dir = output_dir
        args.games = games
        args.movetime_ms = 5
        args.move_grace_ms = 5
        args.max_plies = 40
        args.cli_args = ["unit-test"]
        return args

    def test_uci_handshake_and_id_and_bestmove(self):
        with tempfile.TemporaryDirectory() as temp:
            stderr_path = Path(temp) / "fake.stderr.log"
            with EngineSession(
                self.fake_engine("normal"), "fake", 16, stderr_path
            ) as engine:
                engine.position([])
                move, elapsed_ms = engine.go_movetime(20, 20)
            self.assertEqual(move, "a2a3")
            self.assertGreaterEqual(elapsed_ms, 0.0)
            self.assertEqual(engine.uci_name, "E1Fake")
            self.assertEqual(engine.uci_author, "E1Tests")
            self.assertEqual(engine.uci_search_profile, "current")
            self.assertTrue(stderr_path.exists())

    def test_host_deadline_accepts_on_time_and_rejects_late_legal_move(self):
        with EngineSession(self.fake_engine("slow", 0.01), "fast", 16) as engine:
            engine.position([])
            self.assertEqual(engine.go_movetime(20, 40)[0], "a2a3")

        with self.assertRaises(EngineMoveTimeout) as context:
            with EngineSession(self.fake_engine("normal"), "slow", 16) as engine:
                engine.position([])
                # The child returns a legal bestmove immediately. Advance a
                # deterministic monotonic clock only for the host-side
                # deadline check so this test proves a late response without
                # relying on thread scheduling or wall-clock sleep.
                with patch(
                    "run_protocol_smoke.time.monotonic",
                    side_effect=(100.000, 100.000, 100.041),
                ):
                    engine.go_movetime(20, 20)
        self.assertAlmostEqual(context.exception.elapsed_ms, 41.0, places=6)

    def test_play_game_records_timeout_loss_for_both_baseline_colours(self):
        opening = Opening("test", ())
        with tempfile.TemporaryDirectory() as temp:
            diagnostics = Path(temp)
            for engine_a_white, expected_result, expected_color in (
                (True, "0-1", "white"),
                (False, "1-0", "black"),
            ):
                record = play_game(
                    ScheduledGame(1, 1, opening, engine_a_white),
                    self.legal_move_engine(0.08),
                    self.legal_move_engine(),
                    "baseline",
                    "candidate",
                    16,
                    5,
                    5,
                    40,
                    diagnostics,
                )
                self.assertEqual(record.result, expected_result)
                self.assertEqual(record.reason, "timeout")
                self.assertEqual(record.timeout_engine, "baseline")
                self.assertEqual(record.timeout_color, expected_color)
                self.assertIsNone(record.error)
                self.assertTrue(record.stderr_log.endswith("-a-baseline.stderr.log"))
                self.assertIsNotNone(record.stderr_tail)

    def test_startup_uciok_timeout_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.tournament_args(
                self.startup_failure_engine("uciok", root / "baseline-count"),
                self.legal_move_engine(),
                root / "uciok",
                games=4,
            )
            with patch.object(EngineSession, "startup_timeout_seconds", 0.5):
                summary = run_tournament(args)
            self.assertEqual(summary["status"], "INTEGRITY_FAIL")
            self.assertEqual(summary["games_completed"], 1)
            self.assertEqual(summary["pairs_completed"], 0)
            self.assertIn("uciok", summary["protocol_errors"][0]["error"])

    def test_startup_readyok_timeout_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.tournament_args(
                self.startup_failure_engine("readyok", root / "baseline-count"),
                self.legal_move_engine(),
                root / "readyok",
                games=4,
            )
            with patch.object(EngineSession, "startup_timeout_seconds", 0.5):
                summary = run_tournament(args)
            self.assertEqual(summary["status"], "INTEGRITY_FAIL")
            self.assertEqual(summary["games_completed"], 1)
            self.assertEqual(summary["pairs_completed"], 0)
            self.assertIn("readyok", summary["protocol_errors"][0]["error"])

    def test_fixed_games_complete_after_move_timeouts_and_keep_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.tournament_args(
                self.legal_move_engine(0.08),
                self.legal_move_engine(),
                root / "timeouts",
                games=4,
            )
            summary = run_tournament(args)
            self.assertEqual(summary["status"], "COMPLETED")
            self.assertEqual(summary["integrity_status"], "PASS")
            self.assertEqual(summary["games_completed"], 4)
            self.assertEqual(summary["pairs_completed"], 2)
            self.assertEqual(len(summary["timeouts"]), 4)
            self.assertTrue(
                all(item["stderr_log"].endswith("-a-baseline.stderr.log") for item in summary["timeouts"])
            )

    def test_manifest_locks_full_commands_binary_identity_and_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.tournament_args(
                self.legal_move_engine(),
                self.legal_move_engine(),
                root / "identity",
                games=2,
            )
            args.sha_a = "baseline-git-sha"
            args.sha_b = "candidate-git-sha"
            summary = run_tournament(args)

            self.assertEqual(summary["status"], "COMPLETED")
            manifest = json.loads((root / "identity" / "manifest.json").read_text())
            expected_executable = str(Path(sys.executable).resolve())
            expected_hash = sha256_file(Path(sys.executable).resolve())
            baseline = manifest["engine_a_baseline"]
            candidate = manifest["engine_b_candidate"]
            self.assertEqual(
                baseline["command"][-2:], ["--profile", "current"]
            )
            self.assertEqual(
                candidate["command"][-2:], ["--profile", "current-aspiration"]
            )
            for entry in (baseline, candidate):
                self.assertEqual(entry["resolved_path"], expected_executable)
                self.assertEqual(entry["sha256"], expected_hash)
                self.assertEqual(entry["file_size"], Path(sys.executable).stat().st_size)
                self.assertEqual(entry["uci_options"], {"Hash": 16})
            self.assertEqual(baseline["git_sha"], "baseline-git-sha")
            self.assertEqual(candidate["git_sha"], "candidate-git-sha")
            self.assertEqual(baseline["expected_search_profile"], "current")
            self.assertEqual(candidate["expected_search_profile"], "current-aspiration")
            self.assertEqual(baseline["uci_search_profile"], "current")
            self.assertEqual(candidate["uci_search_profile"], "current-aspiration")
            self.assertEqual(manifest["profile_integrity"]["status"], "PASS")

    def test_profile_mismatch_fails_before_any_game(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.tournament_args(
                self.legal_move_engine(),
                self.fake_engine("normal", reported_profile="current"),
                root / "mismatch",
                games=2,
            )
            summary = run_tournament(args)

            self.assertEqual(summary["status"], "INTEGRITY_FAIL")
            self.assertEqual(summary["integrity_status"], "FAIL")
            self.assertEqual(summary["games_completed"], 0)
            self.assertEqual(summary["pairs_completed"], 0)
            self.assertEqual(summary["manifest"]["stop_reason"], "profile-integrity-failure")
            self.assertEqual(summary["manifest"]["profile_integrity"]["status"], "FAIL")
            self.assertTrue(summary["manifest"]["profile_integrity"]["errors"])

    def test_same_labels_do_not_collide_in_stderr_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            record = play_game(
                ScheduledGame(1, 1, Opening("test", ()), True),
                self.legal_move_engine(),
                self.legal_move_engine(),
                "same",
                "same",
                16,
                5,
                5,
                1,
                Path(temp),
            )
            paths = sorted(Path(temp).glob("*.stderr.log"))
            self.assertEqual(len(paths), 2)
            self.assertTrue(any("-a-same" in path.name for path in paths))
            self.assertTrue(any("-b-same" in path.name for path in paths))
            self.assertIsNone(record.error)

    def test_protocol_error_does_not_count_incomplete_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = run_tournament(
                self.tournament_args(
                    self.legal_move_engine(),
                    self.legal_move_engine(illegal=True),
                    root / "illegal",
                    games=4,
                )
            )
            self.assertEqual(summary["status"], "INTEGRITY_FAIL")
            self.assertIn(summary["games_completed"], {1, 2})
            self.assertEqual(summary["pairs_completed"], 0)
            self.assertEqual(summary["diagnostic_model"]["pairs_completed"], 0)
            self.assertTrue(Path(summary["protocol_errors"][0]["stderr_log"]).exists())

    def test_bestmove_parser_and_legality_errors(self):
        self.assertEqual(parse_bestmove_line("bestmove a2a3 ponder a7a6"), "a2a3")
        with self.assertRaises(EngineProtocolError):
            parse_bestmove_line("bestmove")
        with self.assertRaises(TournamentError):
            validate_bestmove(chess.Board(), "a1a8")

    def test_bestmove_0000_only_accepts_terminal_position(self):
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_checkmate())
        self.assertIsNone(validate_bestmove(board, "0000"))
        self.assertEqual(_terminal_result(board), ("1-0", "checkmate"))
        with self.assertRaises(EngineProtocolError):
            validate_bestmove(chess.Board(), "0000")

    def test_candidate_perspective_maps_both_colours(self):
        self.assertEqual(candidate_result_for_game("0-1", True), "win")
        self.assertEqual(candidate_result_for_game("1-0", True), "loss")
        self.assertEqual(candidate_result_for_game("1-0", False), "win")
        self.assertEqual(candidate_result_for_game("0-1", False), "loss")
        self.assertEqual(candidate_result_for_game("1/2-1/2", True), "draw")

    def test_opening_suite_has_32_legal_lines(self):
        openings = load_openings(DEFAULT_OPENINGS)
        self.assertEqual(len(openings), 32)
        self.assertEqual(len({opening.identifier for opening in openings}), 32)
        for opening in openings:
            board = chess.Board()
            for move_uci in opening.moves:
                move = chess.Move.from_uci(move_uci)
                self.assertIn(move, board.legal_moves, opening.identifier)
                board.push(move)

    def test_schedule_requires_even_games_and_pairs_colours(self):
        openings = load_openings(DEFAULT_OPENINGS)
        with self.assertRaises(TournamentError):
            build_schedule(openings, 63, seed=7)
        schedule = build_schedule(openings, 64, seed=7)
        self.assertEqual(sum(game.engine_a_white for game in schedule), 32)
        for index in range(0, 64, 2):
            first, second = schedule[index : index + 2]
            self.assertEqual(first.pair, second.pair)
            self.assertEqual(first.opening, second.opening)
            self.assertTrue(first.engine_a_white)
            self.assertFalse(second.engine_a_white)

    def test_diagnostic_pentanomial_counts_all_categories_without_decision(self):
        state = DiagnosticPentanomialState(elo1=100.0)
        for total in (0.0, 0.5, 1.0, 1.5, 2.0):
            state.update_pair(total)
        self.assertEqual(state.pairs_completed, 5)
        self.assertEqual(state.counts, {category: 1 for category in PAIR_CATEGORIES})
        self.assertEqual(pair_category(pair_score_from_results("win", "loss")), "1.0")
        self.assertEqual(pair_category(pair_score_from_results("loss", "win")), "1.0")
        self.assertTrue(state.as_json()["diagnostic_only"])
        self.assertNotIn("decision", state.as_json())

    def test_diagnostic_counts_are_order_invariant(self):
        first = DiagnosticPentanomialState()
        second = DiagnosticPentanomialState()
        for total in (0.0, 0.5, 1.0, 1.5, 2.0):
            first.update_pair(total)
        for total in (2.0, 1.5, 1.0, 0.5, 0.0):
            second.update_pair(total)
        self.assertEqual(first.counts, second.counts)

    def test_pair_level_ci_is_finite_and_explicitly_diagnostic(self):
        for totals in ([0.0], [2.0], [1.0] * 4, [0.0] * 20, [2.0] * 20):
            stats = score_statistics(totals)
            low, high = stats["candidate_elo_ci95"]
            self.assertTrue(math.isfinite(low))
            self.assertTrue(math.isfinite(high))
            self.assertLessEqual(low, high)
            self.assertEqual(stats["ci_method"], "approximate_pair_wilson")
            self.assertEqual(stats["ci_status"], "diagnostic_only")
        draws = score_statistics([1.0] * 4)
        more_draws = score_statistics([1.0] * 40)
        self.assertLess(
            more_draws["candidate_elo_ci95"][1] - more_draws["candidate_elo_ci95"][0],
            draws["candidate_elo_ci95"][1] - draws["candidate_elo_ci95"][0],
        )
        self.assertLessEqual(score_statistics([1.0] * 20)["candidate_elo_ci95"][0], 0.0)
        self.assertGreaterEqual(score_statistics([1.0] * 20)["candidate_elo_ci95"][1], 0.0)

    def test_startup_timeout_types_are_distinct_from_move_timeout(self):
        with self.assertRaises(EngineStartupTimeout):
            with patch.object(EngineSession, "startup_timeout_seconds", 0.05):
                with EngineSession(
                    self.fake_engine("no-uciok"), "startup", 16
                ):
                    pass
        self.assertTrue(issubclass(EngineMoveTimeout, TournamentError))

    def test_sha256_file_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample"
            path.write_bytes(b"e1")
            self.assertEqual(
                sha256_file(path),
                "8b5cc4df7eec7d32a7814eca4af047ae33b2d52342667715682e19c25b0b9faa",
            )


if __name__ == "__main__":
    unittest.main()
