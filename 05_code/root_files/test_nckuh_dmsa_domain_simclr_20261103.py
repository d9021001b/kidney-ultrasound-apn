import torch

from nckuh_dmsa_domain_simclr_20261103 import color_augment, nt_xent


def test_augmentation_preserves_tensor_shape_and_bounds():
    x = torch.zeros(4, 3, 16, 16)
    y = color_augment(x)
    assert y.shape == x.shape
    assert float(y.min()) >= -1 and float(y.max()) <= 1


def test_nt_xent_prefers_matching_views():
    first = torch.eye(4)
    assert nt_xent(first, first) < nt_xent(first, first[torch.tensor([1, 0, 3, 2])])
