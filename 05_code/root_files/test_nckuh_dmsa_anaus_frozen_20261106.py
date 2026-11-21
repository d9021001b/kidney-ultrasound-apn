import numpy as np
import torch

import nckuh_dmsa_anaus_frozen_20261106 as study


def test_official_checkpoint_loads_strictly():
    model = study.load_model()
    assert model.fc.__class__.__name__ == "Identity"
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_176_512


def test_forward_feature_shapes():
    model = study.load_model()
    with torch.inference_mode():
        global_feature, layer3, layer4 = study.forward_features(
            model, torch.zeros(2, 3, 224, 224)
        )
    assert global_feature.shape == (2, 512)
    assert layer3.shape == (2, 256, 14, 14)
    assert layer4.shape == (2, 512, 7, 7)


def test_masked_summary_ignores_outside_values():
    feature = torch.ones(1, 2, 4, 4)
    feature[:, :, 2:, :] = 100
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :] = 1
    summary = study.masked_summary(feature, mask)
    assert summary.shape == (1, 8)
    np.testing.assert_allclose(summary.numpy(), [[1, 1, 0, 0, 1, 1, 1, 1]])
