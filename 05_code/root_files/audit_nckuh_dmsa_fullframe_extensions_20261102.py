"""Integrity audit for the registered NCKUH clean-full-frame extension experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "hierarchical": "NCKUH_DMSA_fullframe_hierarchical_probe_20261102",
    "image_instance": "NCKUH_DMSA_fullframe_image_instance_probe_20261102",
    "radimagenet_mil": "NCKUH_DMSA_fullframe_radimagenet_mil_20261102",
    "directional": "NCKUH_DMSA_fullframe_directional_probe_20261102",
    "relative_geometry": "NCKUH_DMSA_fullframe_relative_geometry_probe_20261102",
    "tick_geometry": "NCKUH_DMSA_fullframe_tick_geometry_probe_20261102",
}
OUT = ROOT / "04_quality_archive/NCKUH_fullframe_extension_integrity_20261102.json"


def audit_one(directory: Path) -> dict:
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    decision = json.loads((directory / "development_decision.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(directory / "development_metrics.csv")
    predictions = pd.read_csv(directory / "development_predictions.csv")
    summary = pd.read_csv(directory / "development_summary.csv").set_index("group")
    intervals = pd.read_csv(directory / "development_confidence_intervals.csv")
    assert set(metrics.role) == {"development"}
    assert not decision["test_scored"]
    assert set(metrics.group) == set(protocol["groups"]) == set(summary.index)
    assert set(intervals.metric) >= {"auroc", "ap"}
    assert not {"patient_id", "mrn", "medical_record_number"} & set(predictions.columns.str.lower())
    assert not predictions.duplicated(["seed", "group", "kidney_alias"]).any()
    assert predictions.patient_alias.nunique() == 46
    for (seed, group), subset in predictions.groupby(["seed", "group"]):
        row = metrics[metrics.seed.eq(seed) & metrics.group.eq(group)].iloc[0]
        assert abs(roc_auc_score(subset.label, subset.score) - row.auroc) < 1e-12
        assert abs(average_precision_score(subset.label, subset.score) - row.ap) < 1e-12
    rebuilt = metrics.groupby("group")[["auroc", "ap"]].mean().sort_index()
    assert np.allclose(rebuilt.to_numpy(), summary.loc[rebuilt.index, ["auroc", "ap"]].to_numpy(), atol=1e-12)
    best = summary.sort_values(["auroc", "ap"], ascending=False).iloc[0]
    return {
        "groups": len(summary), "draws": int(metrics.seed.nunique()),
        "prediction_rows": len(predictions), "distinct_development_patients": 46,
        "best_group": str(summary.sort_values(["auroc", "ap"], ascending=False).index[0]),
        "best_mean_auroc": float(best.auroc), "best_mean_ap": float(best.ap),
        "test_scored": False, "recomputed_exactly": True,
    }


def main() -> None:
    result = {"completed_utc": datetime.now(timezone.utc).isoformat(), "experiments": {}}
    for name, folder in EXPERIMENTS.items():
        result["experiments"][name] = audit_one(ROOT / "02_experiments" / folder)
    result["all_passed"] = True
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
