import torch

from nckuh_dmsa_paired_tail_finetune_20261105 import probability_from_logits


def test_hierarchical_probabilities_are_normalized_and_ordered_n_l_r_b():
    any_logit = torch.tensor([0.0])
    bilateral_logit = torch.tensor([0.0])
    left_logit = torch.tensor([0.0])
    probability = probability_from_logits(any_logit, bilateral_logit, left_logit)
    torch.testing.assert_close(probability, torch.tensor([[0.5, 0.125, 0.125, 0.25]]))
    torch.testing.assert_close(probability.sum(1), torch.ones(1))
