import torch

from nckuh_dmsa_view_supervised_mil_20261104 import ViewSupervisedMIL


def test_view_queries_only_attend_assigned_plane() -> None:
    model = ViewSupervisedMIL((0, 1))
    values = torch.randn(1, 5, 32)
    mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)
    views = torch.tensor([[0, 0, 1, 1, -1]])
    logits, weights, evidence = model(values, mask, views)
    assert logits.shape == (1,)
    assert weights.shape == (1, 2, 5)
    assert evidence.shape == (1, 2, 32)
    assert torch.allclose(weights[0, 0, 2:], torch.zeros_like(weights[0, 0, 2:]))
    assert torch.allclose(weights[0, 1, :2], torch.zeros_like(weights[0, 1, :2]))
    assert torch.allclose(weights[0, :, 4], torch.zeros_like(weights[0, :, 4]))


def test_missing_plane_falls_back_to_valid_patches() -> None:
    model = ViewSupervisedMIL((0, 1))
    values = torch.randn(1, 4, 32)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    views = torch.tensor([[0, 0, -1, -1]])
    _, weights, _ = model(values, mask, views)
    assert torch.allclose(weights.sum(dim=2), torch.ones(1, 2))
    assert torch.allclose(weights[0, :, 2:], torch.zeros_like(weights[0, :, 2:]))

