import cv2
import numpy as np

import nckuh_dmsa_pathology_regional_features_20261105 as experiment


def synthetic_roi():
    height, width = 96, 128
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (46, 28), 12, 0, 360, 1, -1)
    yy, xx = np.indices(mask.shape)
    gray = np.rint(40 + 130 * xx / width + 30 * np.sin(yy / 7.0)).clip(0, 255).astype(np.uint8)
    return gray, mask.astype(bool)


def test_features_are_finite_and_named():
    gray, mask = synthetic_roi()
    values, names = experiment.image_features(gray, mask)
    assert len(values) == len(names)
    assert len(values) > 300
    assert len(set(names)) == len(names)
    assert np.isfinite(values).all()


def test_pixels_outside_contour_do_not_change_features():
    gray, mask = synthetic_roi()
    altered = gray.copy()
    rng = np.random.default_rng(20261105)
    altered[~mask] = rng.integers(0, 256, size=(~mask).sum(), dtype=np.uint8)
    first, names = experiment.image_features(gray, mask)
    second, second_names = experiment.image_features(altered, mask)
    assert names == second_names
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_centered_laterality_preserves_patient_mean_shift():
    reference = np.array([0.4, 0.4, 0.7, 0.7])
    candidate = np.array([0.2, 0.8, 0.6, 0.4])
    aliases = __import__("pandas").Series(["P1", "P1", "P2", "P2"])
    result = experiment.centered_laterality(reference, candidate, aliases, 0.25)
    np.testing.assert_allclose(result, [0.325, 0.475, 0.725, 0.675])
    np.testing.assert_allclose([result[:2].mean(), result[2:].mean()], [0.4, 0.7])
