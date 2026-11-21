import numpy as np

from nckuh_dmsa_uricads_official_tiles_20261105 import tile_canvases


def test_tile_canvases_are_masked_and_fixed_size():
    image = np.full((90, 120, 3), 255, np.uint8)
    mask = np.zeros((90, 120), np.uint8)
    mask[10:80, 20:100] = 1
    tiles, fallback = tile_canvases(image, mask, size=128)
    assert len(tiles) == 4 and not fallback
    assert all(tile.shape == (128, 128, 3) for tile in tiles)
    assert all(np.any(tile == 0) for tile in tiles)
