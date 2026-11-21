import hashlib
import json

import numpy as np
import pandas as pd

import nckuh_dmsa_openus_encode_wsl_20261105 as encoder
import nckuh_dmsa_openus_frozen_20261105 as experiment


def test_masked_openus_summary_shapes_and_fallbacks():
    rng = np.random.default_rng(20261105)
    tokens = rng.normal(size=(2, 50, 768)).astype(np.float32)
    masks = np.zeros((2, 224, 224), np.float32)
    masks[0, 32:192, 48:176] = 1
    masks[1, 100:108, 100:108] = 1
    global_mean, roi_mean, roi_std, contrasts = encoder.summarize(tokens, masks)
    assert global_mean.shape == roi_mean.shape == roi_std.shape == (2, 768)
    assert contrasts.shape == (2, 2304)
    assert all(np.isfinite(value).all() for value in (global_mean, roi_mean, roi_std, contrasts))
    np.testing.assert_array_equal(global_mean, tokens[:, 0])


def test_completed_openus_artifacts_are_locked_and_test_unopened():
    lock = json.loads((experiment.OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert experiment.digest(experiment.OUT / "protocol.json") == lock["protocol_sha256"]
    assert experiment.digest(experiment.OUT / "encoding_manifest.csv") == lock["manifest_sha256"]
    audit = json.loads((experiment.OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    assert audit["strict_weight_load"] is True
    assert audit["teacher_state_keys"] == 412
    assert audit["labels_used_in_encoding"] == []
    assert audit["embedding_sha256"] == experiment.digest(
        experiment.OUT / "openus_embeddings.npz"
    )
    predictions = pd.read_csv(experiment.OUT / "development_predictions.csv")
    assert len(predictions) == 2706
    assert predictions.group.nunique() == 11
    assert predictions.score.between(0, 1).all()
    integrity = json.loads((experiment.OUT / "integrity_audit.json").read_text(encoding="utf-8"))
    assert integrity["exploratory_test_rows_scored"] == 0

