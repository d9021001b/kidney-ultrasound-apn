import numpy as np

import kidneyus_internal_region_probe as K


def test_tight_roi_canvas_roundtrip_preserves_mask_shape_and_support():
    gray = np.arange(80 * 120, dtype=np.uint8).reshape(80, 120)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[10:70, 20:100] = 1
    canvas, canvas_mask, transform = K.embed_on_training_canvas(gray, mask, 2.0)
    assert canvas.shape == K.CANVAS_SHAPE
    assert canvas_mask.shape == K.CANVAS_SHAPE
    restored = K.invert_canvas_labels(canvas_mask, transform)
    assert restored.shape == mask.shape
    assert np.array_equal(restored, mask)
