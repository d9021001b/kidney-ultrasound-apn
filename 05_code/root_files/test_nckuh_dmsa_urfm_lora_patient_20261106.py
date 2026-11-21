import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_urfm_lora_patient_20261106 as method


def test_augmentation_preserves_masked_support_and_shape():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 10:22] = True
    image[mask] = 180
    for variant in (0, 1, 2):
        augmented, augmented_mask = method.augmented_canvas(image, mask, variant)
        assert augmented.shape == image.shape
        assert augmented_mask.shape == mask.shape
        assert augmented.dtype == np.uint8
        assert augmented_mask.dtype == bool
        assert not augmented[~augmented_mask].any()
        assert augmented_mask.sum() > 0


def test_patient_design_keeps_side_labels_and_missing_kidney():
    patients = pd.DataFrame({"patient_alias": ["P1", "P2"]})
    kidneys = pd.DataFrame({
        "patient_alias": ["P1", "P1", "P2"],
        "side": ["left", "right", "left"],
        "kidney_APN": [1, 0, 1],
    })
    images = pd.DataFrame({
        "patient_alias": ["P1", "P1", "P2"],
        "side": ["left", "right", "left"],
    })
    image_index, image_side, labels, present = method.patient_design(patients, kidneys, images)
    assert image_index.tolist() == [[0, 1], [2, -1]]
    assert image_side.tolist() == [[0, 1], [0, -1]]
    assert labels.tolist() == [[1, 0], [1, -1]]
    assert present.tolist() == [[True, True], [True, False]]


def test_pairwise_rank_loss_rewards_correct_ordering():
    target = torch.tensor([1, 1, 0, 0])
    good = method.pairwise_rank_loss(torch.tensor([3.0, 2.0, -2.0, -3.0]), target)
    bad = method.pairwise_rank_loss(torch.tensor([-3.0, -2.0, 2.0, 3.0]), target)
    assert float(good) < float(bad)
    assert float(method.pairwise_rank_loss(torch.tensor([1.0, 2.0]), torch.tensor([1, 1]))) == 0.0


def test_patient_bag_head_has_fixed_two_kidney_output():
    torch.manual_seed(1)
    head = method.PatientBagHead()
    values = torch.randn(3, 5, 768)
    sides = torch.tensor([[0, 0, 1, 1, -1], [0, 1, -1, -1, -1], [1, 1, 1, -1, -1]])
    logits, present = head(values, sides)
    assert logits.shape == (3, 2)
    assert present.tolist() == [[True, True], [True, True], [False, True]]
    assert torch.isfinite(logits).all()
