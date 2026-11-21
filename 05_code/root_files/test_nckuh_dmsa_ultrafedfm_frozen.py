import numpy as np

import nckuh_dmsa_ultrafedfm_frozen as study


def test_summarize_uses_masked_patches_and_cls_token():
    tokens = np.zeros((2, 197, 768), dtype=np.float32)
    tokens[:, 0] = np.arange(768, dtype=np.float32)
    tokens[0, np.asarray([0, 1, 14, 15]) + 1] = 2
    tokens[1, 1:] = 3
    masks = np.zeros((2, 224, 224), dtype=np.float32)
    masks[0, :32, :32] = 1
    masks[1] = 1

    cls, roi, counts = study.summarize(tokens, masks)

    assert cls.shape == (2, 768)
    assert roi.shape == (2, 1536)
    np.testing.assert_array_equal(cls[0], np.arange(768, dtype=np.float32))
    assert counts.tolist() == [4, 196]
    np.testing.assert_allclose(roi[0, :768], 2)
    np.testing.assert_allclose(roi[0, 768:], 0)
    np.testing.assert_allclose(roi[1, :768], 3)


def test_checkpoint_identity_is_registered():
    assert study.CHECKPOINT_BYTES == 1_341_123_093
    assert study.CHECKPOINT_SHA256 == "837fe48d140cf62fa2a9efbf056939d80d1714311e3c1b5fa26da7d65018e549"
