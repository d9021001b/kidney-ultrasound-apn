import numpy as np

import nckuh_dmsa_uri_normal_reference_20261107 as module


def test_normal_reference_scores_shifted_samples_higher():
    rng = np.random.default_rng(41)
    normal = rng.normal(0.0, 1.0, size=(90, 24))
    shifted = rng.normal(2.0, 1.0, size=(30, 24))
    values = np.vstack([normal, shifted])
    model = module.fit_normal_model(values, np.arange(len(normal)), components=8, seed=7)
    score = module.normal_anomaly_score(model, values)
    assert np.isfinite(score).all()
    assert np.median(score[len(normal):]) > np.median(score[:len(normal)])


def test_aggregate_images_preserves_kidney_order():
    image_score = np.asarray([1.0, 3.0, 2.0, 6.0])
    image_kidney_index = np.asarray([0, 0, 1, 1])
    mean = module.aggregate_images(image_score, image_kidney_index, 2, "mean")
    maximum = module.aggregate_images(image_score, image_kidney_index, 2, "max")
    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(maximum, [3.0, 6.0])


def test_configuration_grid_is_small_and_unique():
    names = [configuration.name for configuration in module.CONFIGURATIONS]
    assert len(names) == 16
    assert len(set(names)) == len(names)
