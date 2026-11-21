import numpy as np

import nckuh_dmsa_registered_asymmetry_20261105 as study


def ellipse(shape=(96, 128), center=(64, 48), axes=(42, 24)):
    import cv2

    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    y, x = np.indices(shape)
    image = ((x + 2 * y) % 255).astype(np.uint8)
    return image, mask


def test_canonical_canvas_is_masked_and_fixed_size():
    image, mask = ellipse()
    canvas, binary = study.canonical_canvas(image, mask)
    assert canvas.shape == binary.shape == (study.SIZE, study.SIZE)
    assert np.count_nonzero(canvas[binary == 0]) == 0
    assert 0 < canvas.max() <= 1


def test_contour_registration_is_finite_and_improves_shifted_iou():
    image, mask = ellipse(shape=(study.SIZE, study.SIZE), center=(96, 96), axes=(54, 30))
    shifted_image = np.roll(image.astype(np.float32) / 255, 7, axis=1)
    shifted_mask = np.roll(mask, 7, axis=1)
    before = np.logical_and(mask, shifted_mask).sum() / np.logical_or(mask, shifted_mask).sum()
    registered, registered_mask, after = study.register_by_contour(mask, shifted_image, shifted_mask)
    assert np.isfinite(registered).all()
    assert set(np.unique(registered_mask)) <= {0, 1}
    assert after >= before


def test_registered_pair_feature_geometry_and_missing_flag():
    image, mask = ellipse(shape=(study.SIZE, study.SIZE), center=(96, 96), axes=(54, 30))
    item = (image.astype(np.float32) / 255, mask)
    values, quality = study.registered_pair_features(item, item)
    missing_left, _ = study.registered_pair_features(None, item)
    missing_right, _ = study.registered_pair_features(item, None)
    expected = 4 * (4 * study.GRID * study.GRID + 13) + 1
    assert values.shape == missing_left.shape == missing_right.shape == (expected,)
    assert np.isfinite(values).all()
    assert min(quality) >= 0
    assert missing_left[-1] == -1 and missing_right[-1] == 1
