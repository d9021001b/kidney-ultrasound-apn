import cv2
import numpy as np
import pandas as pd

import nckuh_dmsa_geometric_texture_20261102 as study


def test_descriptor_is_fixed_and_finite():
    gray = cv2.imread(str(study.FEATURES / "percentile/I0001.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(study.FEATURES / "masks/I0001.png"), cv2.IMREAD_GRAYSCALE) > 0
    values, shape = study.descriptor(gray, mask)
    assert values.shape == (284,) and shape.shape == (8,)
    assert np.isfinite(values).all()


def test_view_preserving_pair_and_kidney_orientation():
    patients = pd.DataFrame({"patient_alias": ["P001"]})
    images = pd.DataFrame({
        "patient_alias": ["P001"] * 4,
        "side": ["left", "left", "right", "right"],
        "view": ["LL", "LT", "RL", "RT"],
    })
    values = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    pair = study.view_preserving_pairs(values, images, patients)
    assert pair.shape == (1, 26)
    kidneys = pd.DataFrame({"patient_alias": ["P001", "P001"], "side": ["left", "right"]})
    oriented = study.oriented_kidneys(pair, patients, kidneys)
    assert np.array_equal(oriented[0], pair[0])
    assert np.array_equal(oriented[1, :13], pair[0, 13:])
    assert np.array_equal(oriented[1, 13:], pair[0, :13])

