import torch

import nckuh_dmsa_urfm_huge_frozen_20261108 as huge


def test_huge_configuration_matches_official_architecture_and_weight():
    assert huge.MODEL_FACTORY_NAME == "vit_huge_patch16"
    assert huge.PATCH_SIZE == 16
    assert huge.PATCH_GRID == 14
    assert huge.EMBED_DIM == 1280
    assert huge.EXPECTED_WEIGHT_BYTES == 11_027_164_461
    assert huge.WEIGHT_SHA256 == (
        "0693b0ac0b335a095904362af9f2625e473a1040e2c1b2a0e5fed9ef9066dca6"
    )


def test_huge_feature_grid_uses_all_196_patch_tokens():
    class PatchEmbed(torch.nn.Module):
        def forward(self, images):
            return torch.ones(len(images), 196, 8)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, 8))
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, 197, 8))
            self.pos_drop = torch.nn.Identity()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])
            self.fc_norm = torch.nn.Identity()

    global_embedding, patches = huge.urfm_features(
        Model(), torch.zeros(2, 3, 224, 224)
    )
    assert global_embedding.shape == (2, 8)
    assert patches.shape == (2, 14, 14, 8)


def test_huge_builder_reconstructs_checkpoint_patch16_architecture():
    class FakeModels:
        @staticmethod
        def VisionTransformer(**kwargs):
            return kwargs

    with huge.configured():
        kwargs = huge.study.build_encoder(FakeModels)
    assert kwargs["patch_size"] == 16
    assert kwargs["embed_dim"] == 1280
    assert kwargs["depth"] == 32
    assert kwargs["num_heads"] == 16
    assert kwargs["global_pool"] is True
