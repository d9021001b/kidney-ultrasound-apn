import numpy as np
import torch

import nckuh_dmsa_uricads_full_detector_20261105 as study
import nckuh_dmsa_uricads_official_frozen_20261105 as uri


def test_official_local_detector_maps_every_relevant_checkpoint_tensor():
    payload = torch.load(uri.MODEL_PATHS[0], map_location="cpu", weights_only=True)["state_dict"]
    mapped = study.map_official_detector_state(payload)
    expected = [
        key
        for key in payload
        if key.startswith(("backbone.body.", "backbone.fpn2.", "rpn2.", "roi_heads2."))
    ]
    assert len(mapped) == len(expected)
    assert "rpn.head.conv.0.0.weight" in mapped
    assert "backbone.fpn.inner_blocks.0.0.weight" in mapped
    assert "roi_heads.box_predictor.cls_score.weight" in mapped


def test_detection_aggregation_has_fixed_finite_geometry():
    detection = {
        "labels": torch.tensor([1, 1, 3]),
        "scores": torch.tensor([0.8, 0.4, 0.7]),
        "boxes": torch.tensor([[80, 80, 240, 240], [400, 400, 480, 520], [200, 100, 600, 500]]),
    }
    local, features = study.aggregate_detection(detection)
    assert local.shape == (7,)
    assert features.shape == (6 * len(study.DETECTION_FEATURES),)
    assert np.isfinite(local).all() and np.isfinite(features).all()
    assert np.isclose(local[1], 0.8) and np.isclose(local[3], 0.7)
    assert 0 <= local[0] <= 1


def test_official_fusion_is_bounded_convex_combination():
    model = object.__new__(study.OfficialLocalDetector)
    torch.nn.Module.__init__(model)
    model.alpha_weight = torch.zeros(7, 12)
    model.alpha_bias = torch.ones(7)
    model.beta_weight = torch.zeros(7, 12)
    model.beta_bias = torch.ones(7)
    global_scores = torch.full((2, 7), 0.8)
    local_scores = torch.full((2, 7), 0.2)
    final, alpha, beta = model.official_fusion(global_scores, local_scores)
    torch.testing.assert_close(final, torch.full((2, 7), 0.5))
    torch.testing.assert_close(alpha + beta, torch.ones_like(alpha))
