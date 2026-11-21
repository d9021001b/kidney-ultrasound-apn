import numpy as np
import pytest
import torch

import nckuh_dmsa_sonichu_frozen_20261106 as study


def test_token_summary_uses_only_covered_patches():
    tokens = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
    coverage = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])
    global_feature, masked_feature = study.summarize_tokens(tokens, coverage)
    selected = torch.stack([tokens[0, 0, 0], tokens[0, 1, 0]])
    torch.testing.assert_close(global_feature[0], tokens[0].mean((0, 1)))
    torch.testing.assert_close(masked_feature[0, :3], selected.mean(0))
    assert masked_feature.shape == (1, 12)


def test_token_summary_falls_back_to_maximum_coverage_patch():
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    coverage = torch.tensor([[[0.01, 0.02], [0.03, 0.09]]])
    _, masked = study.summarize_tokens(tokens, coverage)
    expected = tokens[0, 1, 1]
    torch.testing.assert_close(masked[0, :4], expected)
    torch.testing.assert_close(masked[0, 8:12], expected)


def test_official_normalization_is_registered():
    np.testing.assert_allclose(study.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(study.STD, [0.229, 0.224, 0.225])
    assert len(study.WEIGHT_SHA256) == 64
