import numpy as np
import torch

import nckuh_dmsa_radiodino_frozen_20261106 as study


def test_patch_tokens_drop_cls_and_restore_grid():
    class Model(torch.nn.Module):
        def forward_features(self, images):
            return torch.ones(len(images), 197, 384)

    result = study.patch_tokens(Model(), torch.zeros(2, 3, 224, 224))
    assert result.shape == (2, 14, 14, 384)


def test_official_normalization_and_weight_are_registered():
    np.testing.assert_allclose(study.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(study.STD, [0.229, 0.224, 0.225])
    assert len(study.WEIGHT_SHA256) == 64
