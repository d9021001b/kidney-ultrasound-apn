import unittest

import numpy as np

from run_graph_dense_vote_development import recover_graph_score


class GraphScoreRecoveryTests(unittest.TestCase):
    def test_recovers_probability_from_pair_mean(self):
        rad = np.array([0.2, 0.8])
        graph = np.array([0.6, 0.4])
        pair_graph = (rad + graph) / 2.0
        np.testing.assert_allclose(recover_graph_score(rad, pair_graph), graph)


if __name__ == "__main__":
    unittest.main()
