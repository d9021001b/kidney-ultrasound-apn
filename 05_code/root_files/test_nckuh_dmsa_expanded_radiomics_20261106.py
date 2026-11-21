import numpy as np

import nckuh_dmsa_expanded_radiomics_20261106 as study


def test_nearest_extension_preserves_roi_and_removes_black_boundary():
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    extended = study.nearest_extension(gray, mask)
    np.testing.assert_array_equal(extended[mask], gray[mask])
    assert extended.shape == gray.shape
    assert np.all(extended[~mask] > 0)


def test_registered_filter_bank_is_expanded_and_fixed():
    assert tuple(study.IMAGE_TYPES) == (
        "Original",
        "Wavelet",
        "LoG",
        "Gradient",
        "LBP2D",
        "Square",
        "SquareRoot",
        "Logarithm",
        "Exponential",
    )
    assert study.IMAGE_TYPES["LoG"]["sigma"] == [1.0, 2.0, 3.0]
    assert "shape2D" in study.FEATURE_CLASSES


def test_state_probabilities_sum_to_one(monkeypatch):
    class Head:
        def __init__(self, value):
            self.value = value

        def predict_proba(self, values):
            return np.c_[np.full(len(values), 1 - self.value), np.full(len(values), self.value)]

    heads = iter((Head(0.8), Head(0.25), Head(0.6)))
    monkeypatch.setattr(study, "fit_head", lambda *args, **kwargs: next(heads))
    states = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    probability, models = study.state_probabilities(
        np.zeros((8, 3)), np.zeros((8, 2)), states, np.arange(8), 1
    )
    np.testing.assert_allclose(probability.sum(1), 1.0)
    np.testing.assert_allclose(probability[0], [0.2, 0.36, 0.24, 0.2])
    assert set(models) == {"any", "bilateral_given_any", "left_given_unilateral"}
