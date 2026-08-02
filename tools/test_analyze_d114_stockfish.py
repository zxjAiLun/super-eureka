import unittest

from analyze_d114_stockfish import (
    ParsedScore,
    centipawn_loss,
    classify_cpl,
    mate_category,
    parse_comment_score,
    score_metrics,
)


class D114StockfishAnalysisTests(unittest.TestCase):
    def test_comment_score_is_cp_only_and_keeps_depth(self):
        self.assertEqual(parse_comment_score("+1.25/7 0.20s").cp, 125)
        self.assertEqual(parse_comment_score("+1.25/7 0.20s").depth, 7)

    def test_cpl_thresholds_are_fixed(self):
        self.assertEqual(classify_cpl(0), "normal")
        self.assertEqual(classify_cpl(29), "normal")
        self.assertEqual(classify_cpl(30), "inaccuracy")
        self.assertEqual(classify_cpl(80), "mistake")
        self.assertEqual(classify_cpl(180), "blunder")

    def test_mate_scores_are_not_converted_to_cp(self):
        best = ParsedScore("mate", 4)
        played = ParsedScore("cp", 850)
        self.assertIsNone(centipawn_loss(best, played))
        self.assertEqual(mate_category(best, played), "missed_mate")
        self.assertTrue(score_metrics(
            type("Search", (), {"best_move": "e2e4", "score": best, "depth": 10, "pv": ()})(),
            type("Search", (), {"best_move": "e2e4", "score": played, "depth": 10, "pv": ()})(),
            "e2e4",
        )["mate_swing"])

    def test_allowed_mate_is_not_counted_as_a_mate_loss(self):
        score = ParsedScore("mate", 3)
        metrics = score_metrics(
            type("Search", (), {"best_move": "e2e4", "score": score, "depth": 10, "pv": ()})(),
            type("Search", (), {"best_move": "e2e4", "score": score, "depth": 10, "pv": ()})(),
            "e2e4",
        )
        self.assertTrue(metrics["mate_outcome"])
        self.assertFalse(metrics["mate_swing"])
        self.assertEqual(metrics["classification"], "mate-outcome")


if __name__ == "__main__":
    unittest.main()
