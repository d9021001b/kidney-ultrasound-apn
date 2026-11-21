import io
from pathlib import Path
import pickle

import numpy as np
import pytest

import kidneyus_segmentation_probe as K


def test_largest_component_keeps_largest_region():
    mask = np.zeros((20, 30), np.uint8)
    mask[2:5, 2:5] = 1
    mask[8:18, 10:28] = 1
    result = K.largest_component(mask)
    assert int(result.sum()) == 180
    assert not result[3, 3]


def test_overlap_metrics_are_exact():
    reference = np.array([[1, 1], [0, 0]], np.uint8)
    prediction = np.array([[1, 0], [1, 0]], np.uint8)
    metrics = K.overlap_metrics(prediction, reference)
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["dice"] == pytest.approx(1 / 2)


def test_restricted_unpickler_denies_arbitrary_global():
    payload = pickle.dumps(Path("unsafe"))
    with pytest.raises(pickle.UnpicklingError):
        K.RestrictedNumpyUnpickler(io.BytesIO(payload)).load()


def test_manifest_reader_drops_diagnosis_label():
    rows = K.read_manifest_without_diagnosis()
    assert len(rows) == 349
    assert all("label" not in row for row in rows)
    assert len({row["image_id"] for row in rows}) == 349


def test_model_family_is_complete():
    for fold in K.FOLDS:
        checkpoint = K.MODEL_FOLDER / f"fold_{fold}" / K.CHECKPOINT_NAME
        assert checkpoint.is_file()
        assert Path(str(checkpoint) + ".pkl").is_file()

