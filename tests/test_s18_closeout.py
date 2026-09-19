"""Offline closeout checks. No engine, large PGN or model dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCH = load("s18_match", "tools/s18/s18_ab_match.py")
B = load("b_probe", "tools/diagnostics/diagnose_b_class_detailed.py")
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"


def game(round_id, white, black, result, t1, t2):
    return (f'[Event "test"]\n[Round "{round_id}"]\n[White "{white}"]\n'
            f'[Black "{black}"]\n[Result "{result}"]\n[SetUp "1"]\n[FEN "{FEN}"]\n\n'
            f'1... e5 {{0.00/1 {t1}s}} 2. e4 {{0.00/1 {t2}s}} {result}\n\n')


class CloseoutEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "match.pgn"
        self.games = [game(1, "A", "B", "1-0", 0.2, 0.4),
                      game(2, "B", "A", "0-1", 0.8, 0.6)]

    def test_pairs_and_black_to_move_time_attribution(self):
        self.path.write_text("".join(reversed(self.games)))
        # format=epd resets half/fullmove counters in the actual game FEN.
        r = MATCH.parse_pgn_and_stats(self.path, "A", "B", [FEN.rsplit(" ", 2)[0] + " 7 9"])
        self.assertEqual(r["candidate_W"], 2)
        self.assertEqual(r["pentanomial"]["counts"], [0, 0, 0, 0, 1])
        self.assertEqual(r["move_time"]["A"]["total_seconds"], 1.2)
        self.assertEqual(r["move_time"]["B"]["total_seconds"], 0.8)
        self.assertNotIn("elo_ci95", r)

    def test_incomplete_unpaired_and_same_color_refused(self):
        cases = [self.games[0], self.games[0].replace("1-0", "*") + self.games[1],
                 self.games[0] + self.games[0].replace('[Round "1"]', '[Round "2"]'),
                 self.games[0] + self.games[0]]
        for text in cases:
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    MATCH.parse_pgn_and_stats(self.path, "A", "B", [FEN])

    def test_wrong_opening_set_refused(self):
        self.path.write_text("".join(self.games))
        with self.assertRaises(ValueError):
            MATCH.parse_pgn_and_stats(self.path, "A", "B", [])

    def test_elo_only_from_cutechess(self):
        self.path.write_text("Elo difference: -21.7 +/- 39.1, LOS: 13.7 %, DrawRatio: 16.4 %\n")
        s = MATCH.read_cutechess_elo(self.path)
        self.assertEqual(s["reported_ci95_half_width"], 39.1)
        self.assertEqual(s["elo"], -21.7)
        self.path.write_text("no summary")
        with self.assertRaises(ValueError):
            MATCH.read_cutechess_elo(self.path)

    def test_frozen_b_probe_has_no_measured_cutoff_claim(self):
        rows = json.loads(B.EVIDENCE.read_text())
        r = B.summarize(rows)
        self.assertEqual(r["labeled_rows"], 40)
        self.assertEqual(r["proxy_positive_rows"], 0)
        self.assertEqual(r["rules"][0]["FP"], 16)
        self.assertEqual(r["rules"][1]["FP"], 14)
        self.assertTrue(all(x["recall"] is None for x in r["rules"]))
        self.assertEqual(len(r["gap_ge_500"]), 3)
        self.assertEqual(sum(x["mate_sentinel"] for x in r["gap_ge_500"]), 2)

    def test_saved_qc1_pair_evidence_recomputes_result(self):
        base = ROOT / "results/s18-ab"
        report = json.loads((base / "qc1-match-report.json").read_text(encoding="utf-8"))
        evidence = json.loads((base / "qc1-pairs.json").read_text(encoding="utf-8"))
        self.assertEqual(report["pgn_sha256"], evidence["pgn_sha256"])
        penta, points = [0] * 5, []
        self.assertEqual(len(evidence["pairs"]), 128)
        for pair in evidence["pairs"]:
            self.assertEqual({g["candidate_color"] for g in pair["games"]}, {"white", "black"})
            pair_points = [g["candidate_points"] for g in pair["games"]]
            points.extend(pair_points)
            penta[round(sum(pair_points) * 2)] += 1
        self.assertEqual(penta, [20, 23, 53, 17, 15])
        self.assertEqual([points.count(1), points.count(0.5), points.count(0)], [99, 42, 115])
        self.assertEqual(report["elo_statistics"]["reported_ci95_half_width"], 39.1)
        self.assertNotIn("elo_ci95_paired", report["result"])

    def test_saved_cost_summary_recomputes_ratios(self):
        data = json.loads((ROOT / "results/s18-ab/qc1-cost-evidence.json").read_text(encoding="utf-8"))
        rows = data["rows"]
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({r["id"] for r in rows}), 32)
        for metric, denominator, numerator, floor in [
                ("nodes", "base_nodes", "qc1_nodes", 1),
                ("wall", "base_wall_ms", "qc1_wall_ms", 0.1)]:
            ratios = sorted(r[numerator] / max(r[denominator], floor) for r in rows)
            self.assertEqual(data[f"{metric}_ratio_median"], ratios[len(ratios)//2])
            self.assertEqual(data[f"{metric}_ratio_p90"], ratios[int(len(ratios)*0.9)])

    def test_s14_compact_ladder_has_legal_final_pvs(self):
        data = json.loads((ROOT / "docs/diagnostics/s14-knight-ladder-20260919.result.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["results"]), 97)
        for row in data["results"]:
            self.assertNotIn("iterations", row)
            board = MATCH.chess.Board(row["fen"])
            for move in (row.get("last") or {}).get("pv", []):
                board.push_uci(move)  # python-chess rejects illegal continuations

    def test_corrupt_b_labels_refused(self):
        rows = json.loads(B.EVIDENCE.read_text())
        rows[0]["is_false_optimism"] = not rows[0]["is_false_optimism"]
        with self.assertRaises(ValueError):
            B.summarize(rows)


if __name__ == "__main__":
    unittest.main()
