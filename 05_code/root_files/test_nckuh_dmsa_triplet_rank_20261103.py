import numpy as np
import pandas as pd
import torch

from nckuh_dmsa_triplet_rank_20261103 import (
    batch_hard_triplet,
    kidney_summary_features,
    patient_pairs,
)


def test_kidney_summary_preserves_view_difference() -> None:
    images = pd.DataFrame({
        "kidney_alias": ["K1", "K1"],
        "view": ["LL", "LT"],
    })
    kidneys = pd.DataFrame({"kidney_alias": ["K1"]})
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    output = kidney_summary_features(values, images, kidneys)
    assert output.shape == (1, 12)
    np.testing.assert_allclose(output[0, 8:10], [-2.0, -2.0])
    np.testing.assert_allclose(output[0, -2:], [1.0, 1.0])


def test_patient_pairs_encode_affected_side_direction() -> None:
    kidneys = pd.DataFrame({
        "patient_alias": ["P1", "P1", "P2", "P2", "P3", "P3"],
        "side": ["left", "right"] * 3,
        "DMSA_APN": ["L", "L", "R", "R", "B", "B"],
    })
    assert patient_pairs(kidneys, np.arange(6)) == [(0, 1, 1), (2, 3, -1)]


def test_triplet_loss_rewards_separation() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0]])
    mixed = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.1, 0.0], [2.1, 0.0]])
    assert batch_hard_triplet(separated, labels) < batch_hard_triplet(mixed, labels)
