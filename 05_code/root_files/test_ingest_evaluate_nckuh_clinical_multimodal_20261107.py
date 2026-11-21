import numpy as np
import pandas as pd
import pytest

import ingest_evaluate_nckuh_clinical_multimodal_20261107 as module


def completed_frame():
    rows = []
    for index in range(210):
        row = {column: "" for column in module.COLUMNS}
        row.update({
            "病歷號": f"M{index:04d}", "研究代碼": f"P{index:03d}",
            "腎臟超音波日期": "2026-01-01", "回填狀態": "完成",
            "最高體溫_C": 39.0, "CRP_mg_L": 80.0, "白血球酯酶": "2+",
            "亞硝酸鹽": "陽性", "尿液培養陽性": "是", "血液培養陽性": "否",
            "左腎_報告疑似APN": "否", "右腎_報告疑似APN": "否",
            "左腎_局部實質異常": "否", "右腎_局部實質異常": "否",
            "左腎_腎盂擴張": "否", "右腎_腎盂擴張": "否",
            "左腎_Doppler灌流缺損": "未做", "右腎_Doppler灌流缺損": "未做",
            "左側VUR分級": "未知", "右側VUR分級": "未知",
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=module.COLUMNS)


def test_completed_clinical_return_validates_without_dmsa():
    result = module.validate_completed(completed_frame())
    assert len(result) == 210
    assert module.completion(result)["remaining_status_rows"] == 0


def test_unknown_values_remain_blank():
    frame = completed_frame()
    frame["CRP_mg_L"] = frame["CRP_mg_L"].astype(object)
    frame.loc[0, "CRP_mg_L"] = ""
    result = module.validate_completed(frame)
    assert pd.isna(result.loc[0, "CRP_mg_L"])


def test_out_of_range_value_is_rejected():
    frame = completed_frame()
    frame.loc[0, "最高體溫_C"] = 70
    with pytest.raises(ValueError, match="Out-of-range"):
        module.validate_completed(frame)


def test_direct_report_score_preserves_neutral_missing():
    frame = pd.DataFrame({
        "report_apn": ["是", "否", "missing"],
        "parenchymal_abnormality": ["是", "否", "missing"],
        "doppler_defect": ["是", "否", "missing"],
    })
    np.testing.assert_allclose(module.direct_report_score(frame), [1.0, 0.0, 0.5])


def test_model_pipeline_handles_missing_values():
    rows = []
    for index in range(40):
        label = index % 2
        row = {column: np.nan for column in module.NUMERIC_SOURCE}
        row.update({column: "missing" for column in module.PATIENT_CATEGORICAL_SOURCE + module.SIDE_FEATURES})
        row["kidney_APN"] = label
        row["report_apn"] = "是" if label else "否"
        row["parenchymal_abnormality"] = "是" if label else "否"
        rows.append(row)
    frame = pd.DataFrame(rows)
    model = module.make_model("combined", 0.1, "l2", 7)
    model.fit(frame, frame.kidney_APN)
    score = model.predict_proba(frame)[:, 1]
    assert np.isfinite(score).all()
