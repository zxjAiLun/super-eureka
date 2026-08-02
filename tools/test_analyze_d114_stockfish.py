import unittest

from analyze_d114_stockfish import (
    ParsedScore,
    SearchResult,
    centipawn_loss,
    classify_cpl,
    common_position_comparison,
    mate_category,
    parse_comment_score,
    score_metrics,
)


class D114StockfishAnalysisTests(unittest.TestCase):
    @staticmethod
    def search(score, best_move="e2e4"):
        return SearchResult(best_move, score, 10, ())

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
        self.assertEqual(mate_category(best, played), "winning_mate_delayed")
        self.assertTrue(score_metrics(
            self.search(best),
            self.search(played),
            "e2e4",
        )["mate_swing"])

    def test_negative_mate_direction_is_from_side_to_move_perspective(self):
        best = ParsedScore("mate", -7)
        accelerated_loss = ParsedScore("mate", -3)
        delayed_loss = ParsedScore("mate", -11)

        self.assertEqual(mate_category(best, accelerated_loss), "losing_mate_accelerated")
        self.assertEqual(mate_category(best, delayed_loss), "losing_mate_delayed")
        self.assertTrue(score_metrics(
            self.search(best), self.search(accelerated_loss), "e2e4"
        )["mate_swing"])
        self.assertFalse(score_metrics(
            self.search(best), self.search(delayed_loss), "e2e4"
        )["mate_swing"])

    def test_positive_mate_direction_has_explicit_names(self):
        best = ParsedScore("mate", 5)
        delayed = ParsedScore("mate", 8)
        accelerated = ParsedScore("mate", 3)

        self.assertEqual(mate_category(best, delayed), "winning_mate_delayed")
        self.assertEqual(mate_category(best, accelerated), "winning_mate_accelerated")

    def test_common_position_comparison_separates_same_and_different_moves(self):
        records = [
            {"pair": 1, "fen": "fen-a", "profile": "Current", "common_in_pair": True,
             "played_move": "e2e4", "centipawn_loss": 20},
            {"pair": 1, "fen": "fen-a", "profile": "CurrentLmr", "common_in_pair": True,
             "played_move": "e2e4", "centipawn_loss": 10},
            {"pair": 2, "fen": "fen-b", "profile": "Current", "common_in_pair": True,
             "played_move": "d2d4", "centipawn_loss": 30},
            {"pair": 2, "fen": "fen-b", "profile": "CurrentLmr", "common_in_pair": True,
             "played_move": "c2c4", "centipawn_loss": 10},
        ]

        comparison = common_position_comparison(records)
        self.assertEqual(comparison["same_move_groups"], 1)
        self.assertEqual(comparison["different_move_groups"], 1)
        self.assertEqual(comparison["different_move_candidate_lower_cpl"], 1)
        self.assertEqual(comparison["different_move_baseline_lower_cpl"], 0)
        self.assertEqual(comparison["different_move_comparable_cp_groups"], 1)

    def test_allowed_mate_is_not_counted_as_a_mate_loss(self):
        score = ParsedScore("mate", 3)
        metrics = score_metrics(
            self.search(score),
            self.search(score),
            "e2e4",
        )
        self.assertTrue(metrics["mate_outcome"])
        self.assertFalse(metrics["mate_swing"])
        self.assertEqual(metrics["classification"], "mate-outcome")


if __name__ == "__main__":
    unittest.main()
