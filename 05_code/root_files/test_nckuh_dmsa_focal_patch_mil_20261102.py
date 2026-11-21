import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_focal_patch_mil_20261102 as study


def test_pooling_ignores_padding():
    valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    original = torch.randn(1, 5, study.PCA_WIDTH)
    changed = original.clone(); changed[:, 3:] = 10000
    for mode in study.MODES:
        torch.manual_seed(3)
        model = study.FocalMIL(mode).eval()
        with torch.inference_mode():
            first = model(original, valid)[0]
            second = model(changed, valid)[0]
        assert torch.allclose(first, second, atol=1e-6, rtol=0)


def test_unilateral_pair_uses_positive_then_negative():
    kidneys = pd.DataFrame({
        "patient_alias": ["A", "A", "B", "B", "C"],
        "kidney_APN": [1, 0, 1, 1, 0],
    })
    pairs = study.unilateral_pairs(np.array([0, 1, 2, 3, 4]), kidneys)
    assert pairs == [(0, 1)]


def test_all_pooling_modes_return_finite_bag_scores():
    values = torch.randn(4, 8, study.PCA_WIDTH)
    valid = torch.ones(4, 8, dtype=torch.bool); valid[0, 3:] = False
    for mode in study.MODES:
        bag, instance = study.FocalMIL(mode)(values, valid)
        assert bag.shape == (4,) and instance.shape == (4, 8)
        assert torch.isfinite(bag).all()

