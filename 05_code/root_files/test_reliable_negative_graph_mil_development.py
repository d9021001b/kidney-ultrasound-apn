import unittest

import torch

from run_reliable_negative_graph_mil_development import reliable_negative_instance_loss


class ReliableNegativeInstanceLossTests(unittest.TestCase):
    def test_uses_only_valid_instances_from_negative_bags(self):
        logits = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        valid = torch.tensor([[True, False], [True, True]])
        labels = torch.tensor([0.0, 1.0])
        loss = reliable_negative_instance_loss(logits, valid, labels)
        self.assertAlmostEqual(float(loss), 0.693147, places=5)

    def test_all_positive_batch_has_differentiable_zero_loss(self):
        logits = torch.ones((1, 2), requires_grad=True)
        valid = torch.tensor([[True, True]])
        labels = torch.tensor([1.0])
        loss = reliable_negative_instance_loss(logits, valid, labels)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.all(logits.grad == 0))


if __name__ == "__main__":
    unittest.main()
