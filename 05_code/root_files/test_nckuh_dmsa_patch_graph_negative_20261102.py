import torch

import nckuh_dmsa_patch_graph_negative_20261102 as study


def test_compaction_is_bounded_and_nonempty():
    bags = torch.arange(2*100*3, dtype=torch.float32).reshape(2, 100, 3)
    valid = torch.zeros(2, 100, dtype=torch.bool)
    valid[0, :100] = True; valid[1, :17] = True
    compact, mask, selected = study.compact_bags(bags, valid, max_nodes=64)
    assert compact.shape == (2, 64, 3)
    assert mask.sum(1).tolist() == [64, 17]
    assert selected[0][0] == 0 and selected[0][-1] == 99


def test_reliable_negative_loss_has_no_positive_bag_gradient():
    logits = torch.zeros(2, 3, requires_grad=True)
    valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    labels = torch.tensor([0.0, 1.0])
    loss = study.reliable_negative_loss(logits, valid, labels)
    loss.backward()
    assert logits.grad[0, :2].abs().sum() > 0
    assert logits.grad[0, 2] == 0
    assert logits.grad[1].abs().sum() == 0


def test_graph_masks_padding_and_model_shapes():
    values = torch.randn(2, 5, 32)
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    adjacency = study.normalized_rbf_adjacency(values, valid)
    assert torch.allclose(adjacency, adjacency.transpose(1, 2), atol=1e-6)
    assert adjacency[0, 3:].abs().sum() == 0 and adjacency[0, :, 3:].abs().sum() == 0
    for mode in study.MODES:
        model = study.PatchGraphModel(mode).eval()
        bag, instance, attention = model(values, valid)
        assert bag.shape == (2,) and instance.shape == (2, 5) and attention.shape == (2, 5)
        assert torch.allclose((attention*valid).sum(1), torch.ones(2), atol=1e-6)

