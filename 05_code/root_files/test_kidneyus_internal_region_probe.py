import numpy as np
import pytest

from kidneyus_internal_region_probe import (
    CLASS_NAMES,
    dice,
    embed_on_training_canvas,
    invert_canvas_labels,
    selected_packet_image_ids,
    validate_model_family,
)


def test_embed_and_invert_round_trip_for_labels():
    gray = np.arange(24, dtype=np.uint8).reshape(4, 6)
    outer = np.zeros((4, 6), dtype=np.uint8)
    outer[1:3, 1:5] = 1
    canvas, canvas_outer, transform = embed_on_training_canvas(
        gray, outer, 2.0, canvas_shape=(16, 20)
    )
    assert canvas.shape == (16, 20)
    assert canvas_outer.sum() == outer.sum() * 4
    restored = invert_canvas_labels(canvas_outer, transform)
    np.testing.assert_array_equal(restored, outer)


def test_embed_rejects_invalid_scale():
    with pytest.raises(ValueError):
        embed_on_training_canvas(
            np.zeros((4, 6), dtype=np.uint8),
            np.ones((4, 6), dtype=np.uint8),
            0,
        )


def test_dice_handles_empty_and_disjoint_masks():
    empty = np.zeros((3, 3), dtype=np.uint8)
    assert dice(empty, empty) == 1.0
    a = empty.copy()
    b = empty.copy()
    a[0, 0] = 1
    b[2, 2] = 1
    assert dice(a, b) == 0.0


def test_official_task002_family_is_complete_and_multiclass():
    records = validate_model_family()
    assert len(records) == 5
    assert [record["fold"] for record in records] == [0, 1, 2, 3, 4]
    assert CLASS_NAMES == {1: "cec", 2: "medulla", 3: "cortex"}


def test_blinded_packet_selection_is_unique():
    image_ids = selected_packet_image_ids()
    assert len(image_ids) == 32
    assert len(set(image_ids)) == 32
