import unittest

from verify_d114_partial import complete_pair_counts, verify_partial


class PartialVerifierContractTests(unittest.TestCase):
    def test_827_games_use_only_826_games_for_413_pairs(self):
        self.assertEqual(complete_pair_counts(827), (827, 826, 413))

    def test_partial_verifier_is_exposed(self):
        self.assertTrue(callable(verify_partial))


if __name__ == "__main__":
    unittest.main()
