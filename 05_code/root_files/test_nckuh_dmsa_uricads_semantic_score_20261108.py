import numpy as np
import pandas as pd

import nckuh_dmsa_uricads_semantic_score_20261108 as study


def test_semantic_scores_are_fixed_finite_probabilities():
    values = np.zeros((2, 5, 7), dtype=np.float32)
    values[0, :, study.CLASS_INDICES] = 0.5
    values[1, :, study.CLASS_INDICES] = 1.0
    scores = study.semantic_image_scores(values)
    assert set(scores) == {"semantic_mean", "semantic_max", "semantic_union"}
    assert np.allclose(scores["semantic_mean"], [0.5, 1.0])
    assert np.allclose(scores["semantic_max"], [0.5, 1.0])
    assert np.allclose(scores["semantic_union"], [0.9375, 1.0])


def test_kidney_aggregation_supports_mean_and_focal_max():
    images = pd.DataFrame({"kidney_alias": ["K1", "K1", "K2"]})
    kidneys = pd.DataFrame({"kidney_alias": ["K2", "K1"]})
    values = np.array([0.2, 0.8, 0.5])
    assert np.allclose(study.aggregate_kidneys(values, images, kidneys, "mean"), [0.5, 0.5])
    assert np.allclose(study.aggregate_kidneys(values, images, kidneys, "max"), [0.5, 0.8])
