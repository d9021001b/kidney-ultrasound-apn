import numpy as np

import nckuh_dmsa_fullframe_extreme_specialist_20261102 as study


def test_extreme_replacement_preserves_probability_and_unilateral_states():
    states = np.asarray([[.4, .1, .2, .3], [.15, .35, .4, .1]])
    revised = study.replace_extreme_probability(states, np.asarray([.8, .2]))
    np.testing.assert_allclose(revised.sum(1), 1)
    np.testing.assert_allclose(revised[:, 1:3], states[:, 1:3])
    np.testing.assert_allclose(revised[:, 0] + revised[:, 3], states[:, 0] + states[:, 3])
    np.testing.assert_allclose(revised[:, 3] / (revised[:, 0] + revised[:, 3]), [.8, .2])


def test_halfblend_uses_original_and_specialist_conditionals_equally():
    states = np.asarray([[.6, .1, .1, .2]])
    revised = study.replace_extreme_probability(states, np.asarray([.75]), True)
    expected = ((.2 / .8) + .75) / 2
    assert np.isclose(revised[0, 3] / (revised[0, 0] + revised[0, 3]), expected)


def test_patient_weights_equalize_each_class_and_patient():
    image_patients = np.asarray([0, 0, 1, 2, 2, 2, 3])
    codes = np.asarray([0, 0, 3, 3])
    weights = study.image_training_weights(image_patients, codes)
    totals = np.asarray([weights[image_patients == patient].sum() for patient in range(4)])
    np.testing.assert_allclose(totals, totals[0])
    assert np.isclose(totals[:2].sum(), totals[2:].sum())


def test_kidney_marginals_match_four_state_definition():
    states = np.asarray([[.1, .2, .3, .4]])
    left, right = study.kidney_marginals(states)
    np.testing.assert_allclose(left, [.6])
    np.testing.assert_allclose(right, [.7])
