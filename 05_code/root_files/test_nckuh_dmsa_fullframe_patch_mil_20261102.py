import numpy as np

import nckuh_dmsa_fullframe_patch_mil_20261102 as study


def test_clean_patch_groups_cover_every_kidney_after_cache():
    if not (study.OUT / "patch_cache_audit.json").exists():
        return
    _, kidneys, images = study.F.matched_reference()
    values, aliases, groups = study.patch_data(images, kidneys)
    assert len(groups) == len(kidneys) and all(len(group) > 0 for group in groups)
    assert np.isfinite(values).all()


def test_mean_and_gated_model_outputs_are_masked():
    import torch
    for mode in ("mean", "gated"):
        model = study.PatchBagModel(mode)
        x = torch.randn(3, 7, 32)
        mask = torch.tensor([[1,1,0,0,0,0,0],[1,1,1,0,0,0,0],[1,1,1,1,1,0,0]], dtype=torch.bool)
        score, weights = model(x, mask)
        assert score.shape == (3,) and weights.shape == (3,7)
        assert not weights[~mask].any()
        np.testing.assert_allclose(weights.sum(1).detach().numpy(), 1, atol=1e-6)
