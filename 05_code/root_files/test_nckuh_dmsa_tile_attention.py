import torch

from nckuh_dmsa_tile_attention import TileHead


def test_pooling_ignores_padding_and_is_permutation_invariant():
    x = torch.randn(2, 4, 32)
    mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    for learned in (False, True):
        model = TileHead(learned).eval()
        score, weights = model(x, mask)
        assert (weights[~mask] == 0).all()
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        torch.testing.assert_close(model(x.flip((1,)), mask.flip((1,)))[0], score)
