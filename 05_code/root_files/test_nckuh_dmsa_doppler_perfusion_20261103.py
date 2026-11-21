import numpy as np

from nckuh_dmsa_doppler_perfusion_20261103 import doppler_masks, image_features


def test_red_blue_doppler_excludes_cyan_contour():
    image = np.zeros((31, 31, 3), dtype=np.uint8)
    mask = np.zeros((31, 31), dtype=np.uint8)
    mask[3:28, 3:28] = 1
    image[10:14, 10:14] = (0, 0, 255)       # red BGR
    image[16:20, 16:20] = (255, 0, 0)       # blue BGR
    image[6:9, 6:9] = (255, 255, 0)         # cyan BGR contour-like UI colour
    masks = doppler_masks(image, mask)
    assert masks["red_orange"].sum() == 16
    assert masks["blue"].sum() == 16
    assert masks["flow"].sum() == 32
    assert masks["cyan_green"].sum() == 9


def test_feature_summary_has_flow_and_location():
    image = np.zeros((31, 31, 3), dtype=np.uint8)
    mask = np.zeros((31, 31), dtype=np.uint8)
    mask[3:28, 3:28] = 1
    image[10:14, 10:14] = (0, 0, 255)
    values = image_features(image, mask)
    assert values["flow_present"] == 1.0
    assert 0 < values["flow_fraction"] < 1
    assert values["blue_fraction"] == 0
    assert values["flow_component_count"] == 1
