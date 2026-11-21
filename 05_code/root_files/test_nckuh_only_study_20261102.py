"""Focused checks for fixed-score uncertainty and patient-only endpoint safeguards."""
import unittest

import numpy as np

from nckuh_only_study_20261102 import metric, weighted_metrics_batch, clean_name


class WeightedMetricTests(unittest.TestCase):
    def test_ties_zero_weight_and_unequal_clusters(self):
        rng = np.random.default_rng(19)
        y = np.r_[np.zeros(21), np.ones(32)]
        score = np.round(rng.random(len(y)), 1)
        weights = rng.integers(0, 5, size=(150, len(y))).astype(float)
        actual = weighted_metrics_batch(y, score, weights)
        expected = np.array([[metric(y, score, row)[key] for key in ["auroc", "ap", "auprc", "brier"]] for row in weights])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_remove_only_annotation_marker(self):
        self.assertEqual(clean_name("01234567_20200101_red_1_3_8_RL.png"), "01234567_20200101_1_3_8_RL.png")


if __name__ == "__main__":
    unittest.main()
