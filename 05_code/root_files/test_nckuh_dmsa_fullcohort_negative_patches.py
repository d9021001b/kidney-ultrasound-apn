import torch

from nckuh_dmsa_fullcohort_negative_patches import negative_loss


def test_auxiliary_gradient_excludes_other_kidneys_and_padding():
    logits = torch.zeros(3, 2, requires_grad=True)
    valid = torch.tensor([[True, False], [True, True], [True, True]])
    negative_loss(logits, valid, torch.tensor([True, False, True])).backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[.25, 0.], [0., 0.], [.125, .125]]))
