import cv2
import numpy as np

import nckuh_dmsa_fullcohort_tick_geometry_20261103 as M


def test_tick_detector_finds_periodic_scale():
    frame = np.zeros((400, 600, 3), np.uint8)
    for y in range(50, 351, 30):
        cv2.line(frame, (20, y), (24, y), (255, 255, 255), 2)
    spacing = M.T.depth_tick_spacing(frame)
    assert np.isfinite(spacing)
    assert abs(spacing - 30) <= 1


def test_tick_geometry_has_explicit_missingness():
    frame = np.zeros((200, 300, 3), np.uint8)
    mask = np.zeros((200, 300), np.uint8)
    cv2.ellipse(mask, (150, 100), (60, 35), 0, 0, 360, 1, -1)
    values = M.T.tick_geometry(frame, mask)
    assert values.shape == (9,)
    assert values[-1] == 0
    assert np.isnan(values[:8]).all()


def test_reference_cohort_is_complete():
    patients, kidneys, images = M.C.load_reference()
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
