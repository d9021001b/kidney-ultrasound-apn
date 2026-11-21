import torch

from nckuh_dmsa_uricads_official_frozen_20261105 import OfficialGlobalEncoder


def test_official_checkpoint_maps_to_modern_resnet():
    from nckuh_dmsa_uricads_official_frozen_20261105 import MODEL_PATHS

    model = OfficialGlobalEncoder(MODEL_PATHS[0]).eval()
    with torch.inference_mode():
        embedding, scores = model(torch.zeros(1, 3, 224, 224))
    assert embedding.shape == (1, 2048)
    assert scores.shape == (1, 7)
    assert torch.isfinite(embedding).all() and torch.isfinite(scores).all()
