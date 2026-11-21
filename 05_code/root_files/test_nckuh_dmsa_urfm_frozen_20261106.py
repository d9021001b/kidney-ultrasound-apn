import numpy as np
import torch

import nckuh_dmsa_urfm_frozen_20261106 as study


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


def test_checkpoint_state_prefers_state_dict_and_strips_module_prefix():
    checkpoint = {
        "state_dict": {"module.patch_embed.weight": torch.ones(1)},
        "model": {"wrong": torch.zeros(1)},
    }
    result = study.checkpoint_state(checkpoint)
    assert list(result) == ["patch_embed.weight"]


def test_official_normalization_and_weight_size_are_registered():
    np.testing.assert_allclose(study.MEAN, [0.485, 0.456, 0.406])
    np.testing.assert_allclose(study.STD, [0.229, 0.224, 0.225])
    assert study.EXPECTED_WEIGHT_BYTES == 2_299_362_921


def test_urfm_feature_shapes_and_global_pooling():
    class PatchEmbed(torch.nn.Module):
        def forward(self, images):
            return torch.ones(len(images), 196, 4)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, 197, 4))
            self.pos_drop = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.fc_norm = torch.nn.Identity()

    global_embedding, patches = study.urfm_features(
        Model(), torch.zeros(2, 3, 224, 224)
    )
    assert global_embedding.shape == (2, 4)
    assert patches.shape == (2, 14, 14, 4)
    torch.testing.assert_close(global_embedding, torch.ones(2, 4))
