import cv2
import numpy as np

from nckuh_dmsa_fullcohort_directional import matched_canvases


def test_contour_axis_alignment_rotates_slanted_mask():
    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.ellipse(mask, (40, 40), (24, 7), 35, 0, 360, 1, -1)
    (_, native), (gray, aligned), angle = matched_canvases(mask * 200, mask)
    assert native.shape == aligned.shape
    assert 30 < angle < 40
    y, x = np.nonzero(aligned)
    assert np.var(x) > 4 * np.var(y)
    assert abs(np.corrcoef(x, y)[0, 1]) < .1
    assert not gray[aligned == 0].any()
    (native_gray, native_roi), (control_gray, control_roi), control_angle = matched_canvases(
        mask * 200, mask, prescribed_angle=0.)
    assert control_angle == 0.
    np.testing.assert_array_equal(native_roi, control_roi)
    np.testing.assert_array_equal(native_gray, control_gray)
