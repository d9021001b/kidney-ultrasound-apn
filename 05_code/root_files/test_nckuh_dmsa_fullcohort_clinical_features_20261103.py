import cv2
import json
import numpy as np

import nckuh_dmsa_fullcohort_clinical_features_20261103 as C


def synthetic_kidney() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:180, :240]
    mask = (((xx - 120) / 70) ** 2 + ((yy - 90) / 42) ** 2 <= 1).astype(np.uint8)
    gray = np.clip(45 + .25 * xx + .1 * yy, 0, 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    image[(mask > 0) & (yy < 60)] = (210, 210, 210)
    return image, mask


def test_focal_features_are_finite_and_label_free():
    image, mask = synthetic_kidney()
    first = C.focal_features(image, mask)
    second = C.focal_features(image, mask)
    assert first.ndim == 1
    assert len(first) > 100
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_focal_features_respond_to_polar_change():
    image, mask = synthetic_kidney()
    changed = image.copy()
    yy = np.mgrid[:180, :240][0]
    changed[(mask > 0) & (yy > 120)] = (5, 5, 5)
    assert not np.allclose(C.focal_features(image, mask), C.focal_features(changed, mask))


def test_small_mask_echo_support_is_finite():
    image, mask = synthetic_kidney()
    small = cv2.resize(mask, (60, 45), interpolation=cv2.INTER_NEAREST)
    small_image = cv2.resize(image, (60, 45), interpolation=cv2.INTER_LINEAR)
    values, audit, resized = C.relative_echo_with_small_mask_support(small_image, small)
    assert resized
    assert np.isfinite(values).all()
    assert audit["reference_pixels"] >= 1024


def test_outline_polygon_is_filled(tmp_path):
    image = np.zeros((100, 120, 3), np.uint8)
    image_path = tmp_path / "image.png"
    annotation_path = tmp_path / "image.json"
    cv2.imwrite(str(image_path), image)
    points = [[20, 50], [30, 25], [75, 20], [100, 50], [75, 80], [30, 75]]
    annotation_path.write_text(json.dumps({"shapes": [{"label": "kidney", "shape_type": "polygon",
                                                        "points": points}]}), encoding="utf-8")
    row = type("Row", (), {"source_type": "labelme_polygon", "source_path_private": str(image_path),
                            "annotation_path_private": str(annotation_path)})()
    _, mask = C._load_source(row)
    assert row.outline_repaired is False or mask.sum() > 1000
    assert mask[50, 60] == 1


def test_reference_cohort_is_complete():
    patients, kidneys, images = C.load_reference()
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    assert set(patients.DMSA_APN) <= {"N", "L", "R", "B"}
