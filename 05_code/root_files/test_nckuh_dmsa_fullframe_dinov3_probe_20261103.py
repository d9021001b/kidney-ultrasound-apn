import numpy as np

import nckuh_dmsa_fullframe_dinov3_probe_20261103 as study


def test_declared_dinov3_bundle_and_groups():
    assert study.BUNDLE.name == "dinov3_vith16plus_weights_20260913"
    assert len(study.GROUPS) == len(set(study.GROUPS)) == 9
    assert "dinov3_extreme_replace_plus_deep_radiomics" in study.GROUPS


def test_extreme_integration_preserves_unilateral_states():
    states = np.asarray([[.2, .25, .15, .4]])
    revised = study.E.replace_extreme_probability(states, np.asarray([.8]))
    np.testing.assert_allclose(revised[:, 1:3], states[:, 1:3])
    np.testing.assert_allclose(revised[:, 0] + revised[:, 3], states[:, 0] + states[:, 3])
