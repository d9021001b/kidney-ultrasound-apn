import cv2
import numpy as np
import torch

import nckuh_dmsa_uri_patch_mil_20261106 as study


def test_patch_geometry_is_finite_and_reflection_invariant():
    mask = np.zeros((100, 160), np.uint8)
    cv2.ellipse(mask, (80, 50), (55, 28), 20, 0, 360, 1, -1)
    first_index, _, first = study.patch_geometry(study.tight_mask(mask, 800))
    second_index, _, second = study.patch_geometry(study.tight_mask(np.fliplr(mask), 800))
    assert len(first_index) == len(second_index)
    assert first.shape[1] == 4
    assert np.isfinite(first).all() and np.isfinite(second).all()
    np.testing.assert_allclose(np.sort(first[:, 0]), np.sort(second[:, 0]), atol=0.08)
    np.testing.assert_allclose(np.sort(first[:, 1]), np.sort(second[:, 1]), atol=0.08)


def test_attention_is_normalized_over_valid_patches():
    model = study.PatchMIL(5, "gated").eval()
    values = torch.randn(2, 4, 5)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    with torch.no_grad():
        bag, patch, attention = model(values, mask)
    assert bag.shape == (2,)
    assert patch.shape == (2, 4)
    torch.testing.assert_close(attention.sum(1), torch.ones(2))
    assert torch.equal(attention[~mask], torch.zeros_like(attention[~mask]))


def test_mean_pool_ignores_padding():
    model = study.PatchMIL(3, "mean").eval()
    values = torch.randn(1, 4, 3)
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        _, _, attention = model(values, mask)
    torch.testing.assert_close(attention, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
