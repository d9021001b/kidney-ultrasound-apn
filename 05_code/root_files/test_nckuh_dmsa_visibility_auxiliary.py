import numpy as np
import pandas as pd
import pytest
import torch

from nckuh_dmsa_visibility_auxiliary import fit_head, validate_review


def test_review_refuses_unreadable_as_negative():
    frame = pd.DataFrame(dict(case_id=["V001"], side=["left"], view_type=["L"],
                              page_file=["V001.png"], image_slot=["left_L"],
                              readability=["unreadable"], visible_finding=["absent"],
                              finding_type=[""], region_on_display=[""], confidence_1_to_5=[3], notes=[""]))
    with pytest.raises(ValueError, match="Unreadable"):
        validate_review(frame, frame)


def test_both_training_arms_on_synthetic_bags():
    torch.manual_seed(17)
    bags = torch.randn(8, 3, 32)
    masks = torch.ones(8, 3, dtype=torch.bool)
    y = torch.tensor([0., 1., 0., 1.])
    predictions = []
    for strength in [0., .3]:
        model = fit_head(bags, masks, np.arange(4), y, np.arange(4, 8), y, 42, strength, epochs=3)
        scores = model(bags[:4], masks[:4])[0].detach()
        assert torch.isfinite(scores).all()
        predictions.append(scores)
    assert not torch.allclose(*predictions)
