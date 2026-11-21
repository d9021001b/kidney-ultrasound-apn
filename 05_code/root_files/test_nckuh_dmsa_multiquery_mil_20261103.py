import torch

from nckuh_dmsa_multiquery_mil_20261103 import MultiQueryMIL, attention_diversity


def test_multiquery_shape_and_mask() -> None:
    model = MultiQueryMIL(4)
    values = torch.randn(2, 6, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    views = torch.tensor([[0, 0, 1, -1, -1, -1], [0, 1, 1, 0, 1, -1]])
    logits, weights, evidence = model(values, mask, views)
    assert logits.shape == (2,)
    assert weights.shape == (2, 4, 6)
    assert evidence.shape == (2, 4, 32)
    assert torch.allclose(weights[0, :, 3:], torch.zeros_like(weights[0, :, 3:]))


def test_diversity_penalizes_identical_queries() -> None:
    identical = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    separate = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert attention_diversity(separate) < attention_diversity(identical)
