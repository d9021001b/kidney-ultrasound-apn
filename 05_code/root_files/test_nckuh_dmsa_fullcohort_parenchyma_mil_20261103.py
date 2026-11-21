import unittest

import numpy as np

import nckuh_dmsa_fullcohort_parenchyma_mil_20261103 as study


class ParenchymaMILTest(unittest.TestCase):
    def test_cyan_rule_excludes_cyan_but_preserves_blue_and_red(self):
        rgb = np.array([[[0, 255, 255], [0, 0, 255], [255, 0, 0], [128, 128, 128]]], dtype=np.uint8)
        found = study.cyan_overlay_mask(rgb)
        np.testing.assert_array_equal(found, [[True, False, False, False]])

    def test_grid_coverage(self):
        mask = np.zeros((224, 224), bool)
        mask[:32, :32] = True
        coverage = study.grid_coverage(mask)
        self.assertEqual(coverage.shape, (7, 7))
        self.assertEqual(float(coverage[0, 0]), 1.0)
        self.assertEqual(float(coverage[1, 1]), 0.0)

    def test_model_returns_normalized_attention(self):
        import torch
        model = study.ParenchymaMIL("gated")
        values = torch.randn(2, 5, 32)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        logits, attention, pooled = model(values, mask)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(pooled.shape), (2, 32))
        torch.testing.assert_close(attention.sum(1), torch.ones(2))
        self.assertTrue(torch.all(attention[~mask] == 0))


if __name__ == "__main__":
    unittest.main()
