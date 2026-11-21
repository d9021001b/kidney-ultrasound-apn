import unittest

import torch

from run_graph_attention_mil_development import normalized_rbf_adjacency


class GraphAdjacencyTests(unittest.TestCase):
    def test_adjacency_is_symmetric_and_padding_is_zero(self):
        values = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [9.0, 9.0]]])
        valid = torch.tensor([[True, True, False]])
        adjacency = normalized_rbf_adjacency(values, valid)
        self.assertTrue(torch.allclose(adjacency, adjacency.transpose(1, 2)))
        self.assertTrue(torch.all(adjacency[0, 2] == 0))
        self.assertTrue(torch.all(adjacency[0, :, 2] == 0))

    def test_singleton_has_unit_self_loop(self):
        values = torch.zeros((1, 2, 3))
        valid = torch.tensor([[True, False]])
        adjacency = normalized_rbf_adjacency(values, valid)
        self.assertEqual(float(adjacency[0, 0, 0]), 1.0)
        self.assertEqual(float(adjacency[0, 1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
