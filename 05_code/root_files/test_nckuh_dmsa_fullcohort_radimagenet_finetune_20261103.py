import unittest

import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 as study


class FullCohortFineTuneTest(unittest.TestCase):
    def test_feature_augmentation_is_deterministic(self):
        values = torch.arange(2*3*7*7, dtype=torch.float32).reshape(2, 3, 7, 7)
        first = study.feature_augment(values, torch.Generator().manual_seed(4))
        second = study.feature_augment(values, torch.Generator().manual_seed(4))
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, values))

    def test_patient_balanced_weights(self):
        images = pd.DataFrame({
            "patient_alias": ["P1", "P1", "P2", "P3"],
            "kidney_APN": [0, 0, 1, 1],
        })
        weights = study.image_weights(images, np.arange(4))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:].sum()))
        self.assertAlmostEqual(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights[2]), float(weights[3]))


if __name__ == "__main__":
    unittest.main()
