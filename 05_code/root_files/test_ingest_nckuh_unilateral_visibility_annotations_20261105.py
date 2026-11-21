import pandas as pd
import pytest

import ingest_nckuh_unilateral_visibility_annotations_20261105 as module


def completed_frame():
    rows = []
    for index in range(1, 84):
        rows.append(
            {
                "case_id": f"U{index:03d}",
                "left_laterality_confirmed": "yes",
                "right_laterality_confirmed": "yes",
                "left_visible_abnormality": "no",
                "right_visible_abnormality": "no",
                "left_finding_type": "none",
                "right_finding_type": "none",
                "left_region": "none",
                "right_region": "none",
                "left_confidence_1_to_5": 4,
                "right_confidence_1_to_5": 4,
                "image_quality_exclusion": "no",
                "reviewer_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=module.EXPECTED_COLUMNS)


def test_completed_review_validates_without_dmsa():
    result = module.validate_completed(completed_frame(), "reviewer-1", "2026-11-05")
    assert len(result) == 83
    assert module.completion(result)["remaining_required_cases"] == 0


def test_visible_abnormality_rejects_none_detail():
    frame = completed_frame()
    frame.loc[0, "left_visible_abnormality"] = "yes"
    with pytest.raises(ValueError, match="Visible abnormality"):
        module.validate_completed(frame, "reviewer-1", "2026-11-05")
