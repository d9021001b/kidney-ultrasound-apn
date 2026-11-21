import cv2
import numpy as np

from nckuh_dmsa_overlay_inpaint import overlay_masks


def test_overlay_detector_keeps_interior_doppler_and_removes_ui_colours():
    mask = np.zeros((40, 60), np.uint8)
    cv2.ellipse(mask, (30, 20), (24, 14), 0, 0, 360, 1, -1)
    bgr = np.full((40, 60, 3), 80, np.uint8)
    bgr[20, 30] = (0, 0, 255)  # Interior red may be Doppler.
    bgr[20, 6] = (0, 0, 255)  # Red at the supplied contour boundary.
    bgr[17:24, 25] = (255, 255, 0)  # Cyan caliper/UI mark.

    artifact, cyan, red = overlay_masks(bgr, mask)

    assert not artifact[20, 30]
    assert red[20, 6]
    assert cyan[18, 25]
    assert artifact[18, 25]
    assert not artifact[10, 30]
