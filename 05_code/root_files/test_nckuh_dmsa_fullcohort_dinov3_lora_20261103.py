import numpy as np
import pandas as pd
import torch
from torch import nn

import nckuh_dmsa_fullcohort_dinov3_lora_20261103 as M


def test_lora_starts_as_exact_base_layer():
    torch.manual_seed(7)
    base = nn.Linear(6, 4)
    values = torch.randn(3, 6)
    expected = base(values).detach()
    wrapped = M.LoRALinear(base, rank=2, alpha=4, dropout=0)
    assert torch.equal(expected, wrapped(values).detach())


def test_kidney_head_ignores_unselected_patch_values():
    torch.manual_seed(8)
    head = M.KidneyHead(width=4).eval()
    hidden = torch.randn(2, 9, 4)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    changed = hidden.clone(); changed[:, 5:][~mask] += 1000
    assert torch.allclose(head(hidden, mask), head(changed, mask))


def test_softmax_pooling_returns_one_value_per_kidney():
    images = pd.DataFrame({"kidney_alias": ["A", "A", "B"]})
    score = M.aggregate_softmax2(np.asarray([.2, .8, .6]), images, pd.Series(["A", "B"]))
    assert score.shape == (2,)
    assert np.all((score > 0) & (score < 1))
    assert score[0] > .5
