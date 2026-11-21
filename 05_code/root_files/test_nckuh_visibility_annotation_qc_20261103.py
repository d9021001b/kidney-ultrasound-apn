import pandas as pd
import pytest

from nckuh_visibility_annotation_qc_20261103 import REQUIRED, agreement, normalize


def review(value=("yes", "no")):
    return pd.DataFrame([
        {column: (f"V{index:03d}" if column == "case_id" else "note" if column == "reviewer_notes" else item)
         for column in REQUIRED}
        for index, item in enumerate(value, start=1)
    ])


def test_agreement_and_adjudication_row():
    first, second = normalize(review()), normalize(review())
    second.loc[0, "left_visible_abnormality"] = "no"
    metrics, adjudication, consensus = agreement(first, second)
    assert len(metrics) == len(REQUIRED) - 2
    assert len(adjudication) == 1
    assert "adjudicated_left_visible_abnormality" in adjudication
    assert consensus.loc[0, "left_visible_abnormality"] == ""


def test_blank_required_review_field_is_rejected():
    data = review(("", ""))
    with pytest.raises(ValueError, match="Incomplete"):
        normalize(data)
