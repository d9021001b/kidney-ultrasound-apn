import cv2
import numpy as np

import nckuh_dmsa_fullframe_relative_echo_probe_20261103 as study


def synthetic_frame():
    yy, xx = np.indices((180, 240))
    frame = 55 + .08 * xx + .05 * yy + 4 * np.sin(xx / 11)
    mask = np.zeros(frame.shape, np.uint8)
    cv2.ellipse(mask, (120, 95), (58, 30), 13, 0, 360, 1, -1)
    frame[mask > 0] += 12 + 5 * np.sin(xx[mask > 0] / 7)
    sector = np.ones(frame.shape, bool)
    return frame.astype(np.float32), mask, sector


def test_relative_echo_is_affine_intensity_invariant():
    frame, mask, sector = synthetic_frame()
    first, _ = study.relative_echo_features(frame, mask, sector)
    second, _ = study.relative_echo_features(frame * 1.2 + 10, mask, sector)
    np.testing.assert_allclose(first, second, rtol=5e-5, atol=5e-5)


def test_relative_echo_ignores_pixels_outside_sector():
    frame, mask, sector = synthetic_frame()
    sector[:, :30] = False
    first, _ = study.relative_echo_features(frame, mask, sector)
    altered = frame.copy(); altered[:, :30] = 240
    second, _ = study.relative_echo_features(altered, mask, sector)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


def test_relative_echo_shape_and_audit():
    frame, mask, sector = synthetic_frame()
    values, audit = study.relative_echo_features(frame, mask, sector)
    assert values.shape == (50,)
    assert np.isfinite(values).all()
    assert audit["reference_pixels"] >= 1024
