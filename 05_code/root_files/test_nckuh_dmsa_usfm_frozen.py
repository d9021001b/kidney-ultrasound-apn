import numpy as np

from nckuh_dmsa_usfm_frozen import summarize


def test_usfm_roi_summary_uses_mask_covered_tokens():
    tokens = np.zeros((1, 197, 768), np.float32)
    tokens[0, 1] = 2
    tokens[0, 2] = 6
    mask = np.zeros((1, 224, 224), np.float32)
    mask[0, :16, :32] = 1

    gap, roi, counts = summarize(tokens, mask)

    assert gap.shape == (1, 768)
    assert roi.shape == (1, 1536)
    assert counts.tolist() == [2]
    np.testing.assert_allclose(roi[0, :768], 4)
    np.testing.assert_allclose(roi[0, 768:], 2)
