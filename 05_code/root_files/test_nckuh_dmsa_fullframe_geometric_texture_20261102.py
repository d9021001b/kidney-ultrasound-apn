import cv2
import numpy as np

import nckuh_dmsa_fullframe_geometric_texture_20261102 as study


def synthetic_kidney():
    mask = np.zeros((96, 144), np.uint8)
    cv2.ellipse(mask, (72, 48), (48, 23), 17, 0, 360, 1, -1)
    yy, xx = np.indices(mask.shape)
    gray = np.clip(45 + .8 * xx + .35 * yy + 18 * np.sin(xx / 7), 0, 255).astype(np.uint8)
    return gray, mask


def test_descriptor_is_finite_and_named():
    gray, mask = synthetic_kidney()
    values, shape = study.descriptor(gray, mask)
    assert values.shape == (284,)
    assert shape.shape == (8,)
    assert len(study.feature_names()) == len(set(study.feature_names())) == 284
    assert np.isfinite(values).all()


def test_descriptor_ignores_pixels_outside_mask():
    gray, mask = synthetic_kidney()
    rng = np.random.default_rng(7)
    altered = gray.copy()
    altered[mask == 0] = rng.integers(0, 256, size=(mask == 0).sum(), dtype=np.uint8)
    first, _ = study.descriptor(gray, mask)
    second, _ = study.descriptor(altered, mask)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_long_axis_summary_is_stable_to_180_degree_rotation():
    gray, mask = synthetic_kidney()
    first, _ = study.descriptor(gray, mask)
    second, _ = study.descriptor(np.rot90(gray, 2), np.rot90(mask, 2))
    # Each sign-invariant long-axis summary starts with 24 intensity features.
    # Spatial filters can differ at discrete nearest-neighbor tie points.
    for start in (170, 208, 246):
        np.testing.assert_allclose(first[start:start + 24], second[start:start + 24], rtol=0, atol=0)
