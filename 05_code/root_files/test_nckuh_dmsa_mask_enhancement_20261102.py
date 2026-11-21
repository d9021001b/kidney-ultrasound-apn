import cv2
import numpy as np

import nckuh_dmsa_mask_enhancement_20261102 as study


def test_variants_are_deterministic_and_masked():
    gray = np.arange(400, dtype=np.uint8).reshape(20, 20)
    mask = np.zeros_like(gray, dtype=bool); mask[3:17, 4:16] = True
    first = study.enhancement_variants(gray, mask)
    second = study.enhancement_variants(gray, mask)
    assert set(first) == set(study.VARIANTS)
    for name in study.VARIANTS:
        np.testing.assert_array_equal(first[name], second[name])
        assert not first[name][~mask].any()


def test_registered_base_matches_existing_image():
    ident = "I0001"
    gray = cv2.imread(str(study.FEATURES / "native" / f"{ident}.png"), 0)
    mask = cv2.imread(str(study.FEATURES / "masks" / f"{ident}.png"), 0) > 0
    expected = cv2.imread(str(study.FEATURES / "percentile" / f"{ident}.png"), 0)
    np.testing.assert_array_equal(study.enhancement_variants(gray, mask)["base"], expected)


def test_pair_swap_is_an_involution():
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    np.testing.assert_array_equal(study.swap_pair(study.swap_pair(values)), values)
