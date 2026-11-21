import json

import numpy as np
import pandas as pd

import nckuh_dmsa_uri_contralateral_local_20261105 as study
from audit_nckuh_dmsa_uri_contralateral_local_20261105 import OUT, paired_values


def test_side_features_are_finite_and_mark_opposite_presence():
    rng = np.random.default_rng(7)
    own = rng.normal(size=(9, 6)).astype(np.float32)
    opposite = rng.normal(size=(8, 6)).astype(np.float32)
    weight = rng.normal(size=(7, 6)).astype(np.float32)
    bias = rng.normal(size=7).astype(np.float32)
    present = study.side_features(own, opposite, weight, bias, 0.75)
    absent = study.side_features(own, None, weight, bias, 0.75)
    assert present.shape == absent.shape == (43,)
    assert np.isfinite(present).all() and np.isfinite(absent).all()
    assert present[-1] == 1 and absent[-1] == 0


def test_completed_encoding_is_label_free_and_has_registered_geometry():
    features = np.load(OUT / "uri_contralateral_local_features.npy", mmap_mode="r")
    dictionary = pd.read_csv(OUT / "feature_dictionary.csv")
    audit = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    assert features.shape == (820, 215)
    assert len(dictionary) == 215
    assert np.isfinite(features).all()
    assert audit["labels_used_in_encoding"] == []


def test_paired_values_preserve_affected_side_sign():
    frame = pd.DataFrame(
        {
            "patient_alias": ["P1", "P1", "P2", "P2"],
            "label": [1, 0, 0, 1],
            "score": [0.8, 0.2, 0.7, 0.4],
        }
    )
    gaps, positive, negative = paired_values(frame)
    np.testing.assert_allclose(gaps, [0.6, -0.3])
    np.testing.assert_allclose(positive, [0.8, 0.4])
    np.testing.assert_allclose(negative, [0.2, 0.7])
