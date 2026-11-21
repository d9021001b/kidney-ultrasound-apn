import numpy as np

from nckuh_dmsa_hierarchical_rbf_svm_20261104 import state_probabilities


def test_state_probabilities_are_normalized(monkeypatch) -> None:
    class Head:
        best_score_ = 0.5
        best_params_ = {}

        def __init__(self, value):
            self.value = value

        def predict_proba(self, values):
            positive = np.full(len(values), self.value)
            return np.c_[1 - positive, positive]

    values = iter((Head(0.8), Head(0.25), Head(0.6)))
    monkeypatch.setattr(
        "nckuh_dmsa_hierarchical_rbf_svm_20261104.fit_head",
        lambda *args, **kwargs: next(values),
    )
    symmetric = np.ones((8, 4))
    antisymmetric = np.ones((8, 2))
    states = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    probability, _ = state_probabilities(
        symmetric, antisymmetric, states, np.arange(8), 7
    )
    np.testing.assert_allclose(probability.sum(1), 1.0)
    np.testing.assert_allclose(probability[0], [0.2, 0.36, 0.24, 0.2])

