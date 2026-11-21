import pandas as pd
import pytest

from nckuh_dmsa_recovery_intake import FIELDS, validate


def test_only_linked_report_with_consistent_code_is_eligible():
    form = pd.DataFrame([{"patient_alias": "synthetic", **dict.fromkeys(FIELDS, "")}])
    assert validate(form, form) == []
    with pytest.raises(ValueError, match="cannot supply"):
        validate(form.assign(report_status="not_performed", dmsa_apn_r_l_b_n="N"), form)
    report = form.assign(report_status="report_available", dmsa_scan_date_iso="2020-01-02",
                         source_reference="synthetic", episode_link_confirmed="yes",
                         reference_interpretation="acute_apn", dmsa_apn_r_l_b_n="L")
    assert validate(report, form) == ["synthetic"]
    assert validate(report.assign(episode_link_confirmed="uncertain"), form) == []
    with pytest.raises(ValueError, match="disagree"):
        validate(report.assign(dmsa_apn_r_l_b_n="N"), form)
