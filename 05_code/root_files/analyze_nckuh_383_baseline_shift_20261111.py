"""Deidentified development failure analysis by cohort overlap and source."""

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
BASELINE = ROOT / "02_experiments/NCKUH_revised_383_baseline_20261111"


def main():
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype=str)
    source = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    predictions = pd.read_csv(BASELINE / "development_predictions.csv")
    with (ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv").open(
            encoding="utf-8-sig", newline="") as stream:
        old = {str(int(row["patient_id"])) if row["patient_id"].isdigit() else row["patient_id"]
               for row in csv.DictReader(stream) if row["DMSA_APN"]}
    linkage["old_210"] = linkage.patient_id.isin(old)
    image_sources = images[["image_id", "kidney_alias"]].merge(
        source[["image_id", "source_path"]], on="image_id", validate="one_to_one")
    image_sources["folder"] = image_sources.source_path.map(lambda p: Path(p).parent.name)
    kidney_folders = image_sources.groupby("kidney_alias").folder.agg(
        lambda values: "+".join(sorted(set(values)))).rename("source_folder")
    dev = predictions.merge(linkage[["patient_alias", "old_210"]],
                            on="patient_alias", validate="many_to_one")
    dev = dev.merge(kidney_folders, on="kidney_alias", validate="many_to_one")
    rows = []
    for group, group_frame in dev.groupby("group"):
        for factor in ("old_210", "source_folder"):
            for level, frame in group_frame.groupby(factor):
                labels = frame.label.to_numpy(int)
                rows.append({"model": group, "factor": factor, "level": str(level),
                             "patients": frame.patient_alias.nunique(), "kidneys": len(frame),
                             "prevalence": float(labels.mean()),
                             "auroc": float(roc_auc_score(labels, frame.score)) if len(set(labels)) == 2 else None,
                             "ap": float(average_precision_score(labels, frame.score)) if labels.sum() else None})
    result = pd.DataFrame(rows).sort_values(["model", "factor", "level"])
    result.to_csv(BASELINE / "development_shift_analysis.csv", index=False)
    (BASELINE / "shift_analysis_scope.json").write_text(json.dumps({
        "scope": "Post-hoc development-only diagnosis; subgroup results cannot select a model or claim validation",
        "strata": ["previous 210-person source-labeled cohort overlap", "selected contour-panel folder"],
        "test_scored": False,
    }, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
