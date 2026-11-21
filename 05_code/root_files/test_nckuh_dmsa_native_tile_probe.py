import numpy as np

from nckuh_dmsa_native_tile_probe import tiles


def test_tiles_preserve_shape_and_have_whole_fallback():
    rgb = np.full((36, 48, 3), 127, dtype=np.uint8)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    mask[2:-2, 2:-2] = True
    crops, fallback = tiles(rgb, mask)
    assert not fallback and len(crops) == 4
    assert all(c.shape == (224, 224, 3) for c in crops)
    mask[:] = False
    mask[2:8, 2:8] = True
    crops, fallback = tiles(rgb, mask)
    assert fallback and len(crops) == 1
