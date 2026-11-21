import numpy as np

import nckuh_dmsa_uri_lightgbm_factorized_20261107 as method


def test_compose_probability_is_normalized_and_ordered_nlrb():
    probability = method.compose_probability(
        np.array([0.8]), np.array([0.25]), np.array([0.75]),
    )
    np.testing.assert_allclose(probability, [[0.2, 0.45, 0.15, 0.2]])
    np.testing.assert_allclose(probability.sum(1), 1.0)


def test_compose_probability_extreme_states():
    probability = method.compose_probability(
        np.array([0.0, 1.0, 1.0]),
        np.array([0.5, 1.0, 0.0]),
        np.array([0.5, 0.5, 1.0]),
    )
    np.testing.assert_allclose(probability, [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
