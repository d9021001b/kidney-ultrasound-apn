import numpy as np
import torch

import nckuh_dmsa_biomedclip_frozen_20261106 as study


def test_token_summary_uses_only_covered_patches():
    tokens = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
    coverage = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])
    summary, counts = study.summarize_tokens(tokens, coverage)
    selected = torch.stack([tokens[0, 0, 0], tokens[0, 1, 0]])
    torch.testing.assert_close(summary[0, :3], selected.mean(0))
    assert summary.shape == (1, 12)
    assert counts.tolist() == [2]


def test_token_summary_falls_back_to_maximum_coverage_patch():
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    coverage = torch.tensor([[[0.01, 0.02], [0.03, 0.09]]])
    summary, counts = study.summarize_tokens(tokens, coverage)
    expected = tokens[0, 1, 1]
    torch.testing.assert_close(summary[0, :4], expected)
    torch.testing.assert_close(summary[0, 8:12], expected)
    assert counts.tolist() == [1]


def test_official_normalization_and_weight_are_registered():
    np.testing.assert_allclose(
        study.MEAN, [0.48145466, 0.4578275, 0.40821073]
    )
    np.testing.assert_allclose(
        study.STD, [0.26862954, 0.26130258, 0.27577711]
    )
    assert len(study.WEIGHT_SHA256) == 64


def test_projected_patch_tokens_apply_visual_projection():
    class Trunk(torch.nn.Module):
        def forward_features(self, images):
            return torch.ones(len(images), 197, 768)

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = Trunk()
            self.head = torch.nn.Linear(768, 512, bias=False)

    result = study.projected_patch_tokens(Visual(), torch.zeros(2, 3, 224, 224))
    assert result.shape == (2, 14, 14, 512)
