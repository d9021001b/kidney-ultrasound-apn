import unittest

import numpy as np
import torch

from nckuh_dmsa_pair_ranking_20261102 import pairs_within_patient, ranking_loss


class SamePatientRankingTests(unittest.TestCase):
    def test_only_source_discordant_pairs(self):
        aliases = ['R','R','L','L','B','B','N','N','single']
        labels = [0,1,1,0,1,1,0,0,1]
        positive, negative = pairs_within_patient(aliases, labels)
        np.testing.assert_equal(positive, [1,2])
        np.testing.assert_equal(negative, [0,3])
        self.assertTrue(all(aliases[a]==aliases[b] for a,b in zip(positive,negative)))

    def test_gradient_improves_positive_minus_negative(self):
        logits = torch.tensor([0.,0.], requires_grad=True)
        loss = ranking_loss(logits, [1], [0])
        loss.backward()
        self.assertGreater(logits.grad[0].item(), 0)
        self.assertLess(logits.grad[1].item(), 0)
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)

    def test_no_positive_patch_or_bilateral_order_is_invented(self):
        positive, negative = pairs_within_patient(['B','B','N','N'], [1,1,0,0])
        self.assertEqual(len(positive), 0)
        self.assertEqual(ranking_loss(torch.zeros(4), positive, negative).item(), 0.)


if __name__ == '__main__':
    unittest.main()
