import numpy as np

import nckuh_dmsa_uri_lightgbm_state_20261107 as method


def test_state_kidney_rows_matches_nlrb_definition():
    states = np.array([0, 1, 2, 3])
    assert method.state_kidney_rows(states).tolist() == [
        [0, 0], [1, 0], [0, 1], [1, 1],
    ]


def test_probability_mapping_accepts_arbitrary_class_order():
    probability = np.array([[0.10, 0.20, 0.30, 0.40]])
    classes = np.array([3, 0, 2, 1])
    score = method.probability_kidney_rows(probability, classes)
    np.testing.assert_allclose(score, [[0.50, 0.40]])


def test_metric_rewards_correct_state_ranking():
    states = np.array([0, 1, 2, 3])
    good = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    bad = 1.0 - good
    assert method.metric_from_states(states, good)["auroc"] == 1.0
    assert method.metric_from_states(states, good)["ap"] == 1.0
    assert method.metric_from_states(states, bad)["auroc"] == 0.0
