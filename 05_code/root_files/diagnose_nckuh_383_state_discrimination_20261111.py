"""Separate patient disease detection from within-patient laterality ranking."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
OUT = ROOT / "02_experiments/NCKUH_revised_383_clinical_audit_20261111"
PREDICTIONS = [
    ROOT / "02_experiments/NCKUH_revised_383_baseline_20261111/development_predictions.csv",
    ROOT / "02_experiments/NCKUH_revised_383_uricads_20261111/development_predictions.csv",
    OUT / "development_predictions.csv",
]


def main():
    assignments = pd.read_csv(BASE / "patient_assignments.csv").set_index("patient_alias")
    all_scores = pd.concat((pd.read_csv(path) for path in PREDICTIONS), ignore_index=True)
    all_scores["side"] = all_scores.kidney_alias.str.rsplit("_", n=1).str[-1].map(
        {"L": "left", "R": "right"})
    assert all_scores.side.notna().all()
    rows = []
    for name, part in all_scores.groupby("group"):
        wide = part.pivot(index="patient_alias", columns="side", values="score")
        codes = assignments.loc[wide.index, "DMSA_APN"]
        assert wide.notna().all().all() and len(wide) == 70
        overall = wide.mean(axis=1)
        bn = codes.isin(["B", "N"])
        any_label = codes.ne("N").astype(int)
        unilateral = codes.isin(["L", "R"])
        side_delta = np.where(codes[unilateral].eq("L"),
                              wide.loc[unilateral, "left"] - wide.loc[unilateral, "right"],
                              wide.loc[unilateral, "right"] - wide.loc[unilateral, "left"])
        rows.append({
            "group": name,
            "patients": len(wide),
            "any_apn_patients": int(any_label.sum()),
            "any_apn_prevalence": float(any_label.mean()),
            "any_apn_auroc": roc_auc_score(any_label, overall),
            "any_apn_ap": average_precision_score(any_label, overall),
            "bilateral_vs_none_patients": int(bn.sum()),
            "bilateral_vs_none_prevalence": float(codes[bn].eq("B").mean()),
            "bilateral_vs_none_auroc": roc_auc_score(codes[bn].eq("B"), overall[bn]),
            "bilateral_vs_none_ap": average_precision_score(codes[bn].eq("B"), overall[bn]),
            "unilateral_patients": int(unilateral.sum()),
            "unilateral_side_correct": int((side_delta > 0).sum()),
            "unilateral_side_ties": int((side_delta == 0).sum()),
            "unilateral_side_accuracy": float((side_delta > 0).mean()),
            "unilateral_signed_score_margin": float(np.mean(side_delta)),
        })
    result = pd.DataFrame(rows).sort_values("group")
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "state_discrimination_diagnostic.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
