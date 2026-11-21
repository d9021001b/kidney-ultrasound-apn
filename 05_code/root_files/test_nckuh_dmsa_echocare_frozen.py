import numpy as np
import torch

import nckuh_dmsa_echocare_frozen as study


def test_summarize_final_gap_and_multiscale_masked_moments():
    features = [
        torch.zeros(2, channels, size, size)
        for channels, size in zip((128, 256, 512, 1024, 2048), (128, 64, 32, 16, 8))
    ]
    for level, feature in enumerate(features, start=1):
        feature[0, :, : feature.shape[-2] // 2] = level
        feature[1] = level + 1
    masks = torch.zeros(2, 256, 256)
    masks[0, :128] = 1
    masks[1] = 1

    gap, multiscale, sums = study.summarize(features, masks)

    assert gap.shape == (2, 2048)
    assert multiscale.shape == (2, 7936)
    assert sums.shape == (10,)
    np.testing.assert_allclose(gap[0], 2.5)
    np.testing.assert_allclose(gap[1], 6)
    np.testing.assert_allclose(multiscale[0, :128], 1)
    np.testing.assert_allclose(multiscale[0, 128:256], 0)
    assert np.isfinite(multiscale).all()


def test_checkpoint_identity_is_registered():
    assert study.CHECKPOINT_BYTES == 481_812_900
    assert study.CHECKPOINT_SHA256 == "f9039c62fcab62cfbedcdd9fe849b9acd7edaa740c0acac903573931680c5883"
