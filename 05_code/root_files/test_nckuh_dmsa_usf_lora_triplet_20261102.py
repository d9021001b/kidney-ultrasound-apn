import numpy as np
import torch

import nckuh_dmsa_usf_lora_triplet_20261102 as study


def test_arm_flags_keep_noaug_unaugmented():
    assert [a for a in study.ARMS if study.arm_flags(a)["augmented"]] == study.ARMS[:4]
    assert not study.arm_flags("frozen_bce_noaug")["augmented"]
    assert not study.arm_flags("lora_bce_noaug")["augmented"]


def test_triplet_requires_different_patient_positive():
    embedding = torch.nn.functional.normalize(torch.randn(2, 2, 8), dim=-1)
    labels = torch.tensor([[0, 0], [1, 1]])
    present = torch.ones(2, 2, dtype=torch.bool)
    loss, anchors = study.batch_triplet_loss(torch, embedding, labels, present, torch.arange(2))
    assert anchors == 0
    assert float(loss) == 0.0

    embedding = torch.nn.functional.normalize(torch.randn(3, 2, 8), dim=-1)
    labels = torch.tensor([[0, 1], [0, 0], [1, 1]])
    present = torch.ones(3, 2, dtype=torch.bool)
    loss, anchors = study.batch_triplet_loss(torch, embedding, labels, present, torch.arange(3))
    assert anchors == 6
    assert torch.isfinite(loss)


def test_shared_head_is_side_exchange_equivariant():
    torch.manual_seed(7)
    head = study.make_head(torch).eval()
    values = torch.randn(3, 4, 1536, device="cuda")
    sides = torch.tensor([[0, 0, 1, 1]] * 3, device="cuda")
    with torch.inference_mode():
        original, _, _ = head(values, sides)
        exchanged, _, _ = head(values, 1-sides)
    assert torch.allclose(original[:, [1, 0]], exchanged, atol=1e-6, rtol=0)

