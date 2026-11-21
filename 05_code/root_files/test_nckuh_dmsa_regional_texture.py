import numpy as np

from nckuh_dmsa_regional_texture import regions


def test_regions_partition_mask_without_gaps_or_overlap():
    mask = np.zeros((13, 19), dtype=np.uint8)
    mask[2:9, 3:16] = 1
    mask[4, 8] = 0
    a, middle, b = regions(mask)
    np.testing.assert_array_equal(a + middle + b, mask)
    assert np.nonzero(a)[1].max() < np.nonzero(middle)[1].min()
    assert np.nonzero(middle)[1].max() < np.nonzero(b)[1].min()
