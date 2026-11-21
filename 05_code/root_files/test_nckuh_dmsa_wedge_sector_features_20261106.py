import cv2
import numpy as np

import nckuh_dmsa_wedge_sector_features_20261106 as study


def synthetic_kidney() -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((160, 220), np.uint8)
    cv2.ellipse(mask, (110, 80), (80, 42), 12, 0, 360, 1, -1)
    image = np.full(mask.shape, 90, np.uint8)
    image[mask > 0] = 110
    wedge = np.array([[40, 58], [85, 70], [41, 86]], np.int32)
    cv2.fillPoly(image, [wedge], 190)
    return image, mask.astype(bool)


def test_wedge_features_are_finite_and_named():
    image, mask = synthetic_kidney()
    values, names = study.image_features(image, mask)
    assert values.shape == (len(names),)
    assert len(names) > 150
    assert np.isfinite(values).all()
    assert "bright_component_depth_span" in names
    assert "width_positive_bulge" in names


def test_sector_summaries_are_invariant_to_half_turn():
    image, mask = synthetic_kidney()
    values, names = study.image_features(image, mask)
    rotated, rotated_names = study.image_features(
        np.rot90(image, 2).copy(), np.rot90(mask, 2).copy()
    )
    assert names == rotated_names
    np.testing.assert_allclose(values, rotated, rtol=2e-3, atol=2e-3)


def test_focal_wedge_changes_sector_features():
    image, mask = synthetic_kidney()
    uniform = image.copy()
    uniform[mask] = 110
    wedge_values, names = study.image_features(image, mask)
    uniform_values, _ = study.image_features(uniform, mask)
    indices = [index for index, name in enumerate(names) if "intensity" in name or "bright" in name]
    assert np.linalg.norm(wedge_values[indices] - uniform_values[indices]) > 1.0


def test_circular_run_wraps_across_sector_zero():
    flags = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    assert study.circular_run(flags) == 4 / 12
