import numpy as np
import pandas as pd

import nckuh_dmsa_uri_image_mil_20261106 as study


def test_poolers_are_finite_and_ordered():
    values = np.array([0.1, 0.4, 0.9])
    pooled = {name: study.pool_values(values, name) for name in study.POOLERS}
    assert all(0 < value < 1 for value in pooled.values())
    assert pooled["max_probability"] == 0.9
    assert pooled["mean_probability"] == np.mean(values)
    assert pooled["softmax_logit"] > pooled["mean_logit"]


def test_pool_kidneys_respects_bag_membership():
    images = pd.DataFrame(
        {"kidney_alias": ["A", "A", "B"], "patient_alias": ["P1", "P1", "P2"]}
    )
    kidneys = pd.DataFrame({"kidney_alias": ["B", "A"]})
    result = study.pool_kidneys(np.array([0.2, 0.6, 0.9]), images, kidneys, "mean_probability")
    np.testing.assert_allclose(result, [0.9, 0.4])


def test_image_weights_equalize_patient_contribution():
    images = pd.DataFrame({"patient_alias": ["P1", "P1", "P1", "P2"]})
    weights = study.image_weights(images, {"P1", "P2"})
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1])
    np.testing.assert_allclose(
        pd.Series(weights).groupby(images.patient_alias).sum().to_numpy(), [1, 1]
    )
