import numpy as np

from nckuh_dmsa_transductive_label_spreading_20261104 import transductive_predict


def test_query_labels_are_not_required() -> None:
    fit = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    labels = np.array([0, 0, 1, 1])
    query = np.array([[0.05, 0.02], [0.95, 0.98]])
    score, _ = transductive_predict(fit, labels, query, k=2, alpha=0.2)
    assert score.shape == (2,)
    assert np.isfinite(score).all()
    assert score[0] < score[1]

