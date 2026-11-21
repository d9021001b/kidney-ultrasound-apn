import numpy as np
import pandas as pd
import pytest

import ingest_evaluate_nckuh_fullcohort_visibility_20261107 as module


def completed_frame():
    rows = []
    for case in range(1, 211):
        for side in ("left", "right"):
            rows.append({
                "case_id": f"V{case:03d}",
                "kidney_side": side,
                "available_views": "L+T",
                "image_readable": "yes",
                "laterality_confirmed": "yes",
                "apn_compatible_finding": "no",
                "primary_echo_pattern": "none",
                "abnormal_region": "none",
                "wedge_shaped_focus": "no",
                "renal_enlargement": "no",
                "poor_corticomedullary_diff": "no",
                "pelvic_dilation": "no",
                "perfusion_defect": "not shown",
                "artifact_obscuration": "no",
                "confidence_1_to_5": 4,
                "reviewer_notes": "",
            })
    return pd.DataFrame(rows, columns=module.EXPECTED_COLUMNS)


def test_completed_review_validates_without_private_key():
    frame = module.validate_completed(completed_frame(), "reader-1", "2026-11-07")
    assert len(frame) == 420
    assert module.completion(frame)["remaining_required_rows"] == 0


def test_visible_finding_rejects_none_pattern():
    frame = completed_frame()
    frame.loc[0, "apn_compatible_finding"] = "yes"
    with pytest.raises(ValueError, match="cannot use none"):
        module.validate_completed(frame, "reader-1", "2026-11-07")


def test_unavailable_view_requires_not_imaged_state():
    frame = completed_frame()
    frame.loc[0, "available_views"] = "none"
    with pytest.raises(ValueError, match="not imaged"):
        module.validate_completed(frame, "reader-1", "2026-11-07")


def test_unavailable_view_does_not_require_confidence():
    frame = completed_frame()
    frame["confidence_1_to_5"] = frame["confidence_1_to_5"].astype(object)
    frame.loc[0, "available_views"] = "none"
    frame.loc[0, "image_readable"] = "not imaged"
    frame.loc[0, "laterality_confirmed"] = "not imaged"
    frame.loc[0, "apn_compatible_finding"] = "not assessable"
    frame.loc[0, "primary_echo_pattern"] = "not assessable"
    frame.loc[0, "abnormal_region"] = "not assessable"
    for column in (
        "wedge_shaped_focus", "renal_enlargement", "poor_corticomedullary_diff",
        "pelvic_dilation", "perfusion_defect", "artifact_obscuration",
    ):
        frame.loc[0, column] = "not assessable"
    frame.loc[0, "confidence_1_to_5"] = ""
    validated = module.validate_completed(frame, "reader-1", "2026-11-07")
    assert module.completion(validated)["remaining_required_rows"] == 0


def test_direct_visibility_score_uses_confidence():
    frame = pd.DataFrame({
        "apn_compatible_finding": ["yes", "no", "uncertain"],
        "confidence_1_to_5": [5, 5, 5],
    })
    assert module.direct_visibility_score(frame).tolist() == [1.0, 0.0, 0.5]


def test_inner_selection_runs_at_patient_level():
    states = np.tile(np.arange(4), 6)
    patients = pd.DataFrame({
        "patient_alias": [f"P{index:03d}" for index in range(len(states))],
    })
    rows = []
    for patient_index, (alias, state) in enumerate(zip(patients.patient_alias, states)):
        for side_index, side in enumerate(("left", "right")):
            label = int(state == 3 or state == side_index + 1)
            rows.append({
                "patient_alias": alias,
                "kidney_APN": label,
                "available_views": "L+T",
                "image_readable": "yes",
                "laterality_confirmed": "yes",
                "apn_compatible_finding": "yes" if label else "no",
                "primary_echo_pattern": "hypoechoic" if label else "none",
                "abnormal_region": "upper pole" if label else "none",
                "wedge_shaped_focus": "yes" if label else "no",
                "renal_enlargement": "yes" if label else "no",
                "poor_corticomedullary_diff": "yes" if label else "no",
                "pelvic_dilation": "no",
                "perfusion_defect": "not shown",
                "artifact_obscuration": "no",
                "confidence_1_to_5": 4,
            })
    frame = pd.DataFrame(rows)
    model, selection, records = module.select_regularization(
        frame, patients, np.arange(len(patients)), states, 17,
    )
    score = model.predict_proba(frame)[:, 1]
    assert len(records) == len(module.REGULARIZATION)
    assert selection["C"] in module.REGULARIZATION
    assert np.isfinite(score).all()
