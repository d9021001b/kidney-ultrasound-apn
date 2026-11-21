import numpy as np

import nckuh_dmsa_fullframe_rank_refine_20261102 as study


def test_torch_linear_reproduces_logistic_probabilities():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 4))
    y = np.array([0, 1] * 20)
    model = study.fit_logistic(x, y, 7)
    head = study.torch_linear(model)
    import torch
    with torch.inference_mode():
        score = torch.sigmoid(head(torch.from_numpy(x.astype(np.float32))).squeeze(1)).numpy()
    np.testing.assert_allclose(score, model.predict_proba(x)[:, 1], atol=1e-6)


def test_epoch_zero_is_recorded():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.array([0, 1] * 10)
    model = study.fit_logistic(x, y, 9)
    _, history = study.refine(study.ARMS[0], model, x, y, [], study.EPOCHS, x[:5])
    assert set(history) == {0} and history[0].shape == (5,)
