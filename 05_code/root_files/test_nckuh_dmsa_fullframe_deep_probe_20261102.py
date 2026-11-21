import cv2
import numpy as np

import nckuh_dmsa_fullframe_deep_probe_20261102 as study


def test_tight_canvas_masks_color_and_gray():
    mask = np.zeros((30, 50), np.uint8); mask[5:25, 10:40] = 1
    gray = np.full((30, 50), 100, np.uint8)
    color = np.dstack([gray, gray // 2, gray // 4])
    gray_canvas, gray_mask = study.tight_canvas(gray, mask)
    color_canvas, color_mask = study.tight_canvas(color, mask)
    assert gray_canvas.shape == color_canvas.shape == (224, 224, 3)
    np.testing.assert_array_equal(gray_mask, color_mask)
    assert not gray_canvas[gray_mask == 0].any() and not color_canvas[color_mask == 0].any()


def test_actual_clean_roi_contract():
    manifest = np.genfromtxt(study.F.OUT / "deidentified_image_manifest.csv", delimiter=",", dtype=str, skip_header=1)
    ident = manifest[0, 0]
    color = cv2.imread(str(study.F.OUT / "clean_rgb" / f"{ident}.png"))
    mask = cv2.imread(str(study.F.OUT / "masks" / f"{ident}.png"), 0) > 0
    canvas, canvas_mask = study.tight_canvas(color, mask)
    assert canvas.shape == (224, 224, 3) and canvas_mask.any()
