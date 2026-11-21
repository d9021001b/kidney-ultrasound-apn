import sys

import numpy as np
import torch

import nckuh_dmsa_tinyusfm_frozen_20261105 as experiment


def test_official_checkpoint_loads_strictly_and_returns_expected_tokens():
    model = experiment.load_model(torch.device("cpu"))
    with torch.inference_mode():
        tokens = experiment.forward_tokens(model, torch.zeros(1, 3, 224, 224))
    assert tuple(tokens.shape) == (1, 197, 192)
    assert torch.isfinite(tokens).all()


def test_masked_token_summary_has_expected_shape_and_fallback():
    rng = np.random.default_rng(20261105)
    tokens = rng.normal(size=(2, 197, 192)).astype(np.float32)
    masks = np.zeros((2, 224, 224), np.float32)
    masks[0, 32:192, 48:176] = 1
    masks[1, 100:108, 100:108] = 1
    cls, mean, std = experiment.summarize(tokens, masks)
    assert cls.shape == mean.shape == std.shape == (2, 192)
    assert np.isfinite(cls).all() and np.isfinite(mean).all() and np.isfinite(std).all()
    np.testing.assert_array_equal(cls, tokens[:, 0])
