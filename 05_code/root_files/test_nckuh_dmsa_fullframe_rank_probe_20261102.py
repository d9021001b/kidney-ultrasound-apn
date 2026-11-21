import numpy as np
import torch

import nckuh_dmsa_fullframe_rank_probe_20261102 as study


def test_rank_loss_gradient_orders_positive_above_negative():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    study.rank_loss(logits, labels).backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0


def test_unilateral_rank_loss_gradient():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    study.unilateral_rank_loss(logits, [(0, 1)]).backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0


def test_unilateral_pairs_exclude_concordant_patients():
    _, kidneys, _ = study.F.matched_reference()
    labels = kidneys.kidney_APN.to_numpy(int)
    indices = np.arange(len(kidneys))
    pairs = study.unilateral_pairs(indices, kidneys, labels)
    assert pairs
    for positive, negative in pairs:
        assert labels[positive] == 1 and labels[negative] == 0
        assert kidneys.patient_alias.iloc[positive] == kidneys.patient_alias.iloc[negative]
