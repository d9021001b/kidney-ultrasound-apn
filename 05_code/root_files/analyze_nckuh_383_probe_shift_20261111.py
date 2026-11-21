"""Post-hoc development-only probe-stratified URI-CADS failure analysis."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from nckuh_probe_overlay_ocr_20261111 import source_images


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
EXPERIMENT = ROOT / "02_experiments/NCKUH_revised_383_uricads_20261111"


def main():
    ocr = json.loads((REGISTRY / "probe_overlay_ocr_full_test_predict.json").read_text())
    paths = source_images()["test_predict"]
    assert len(paths) == ocr["total_examined"] == 826
    probe = {str(paths[row["file_ordinal"]].resolve()).lower(): row["probe"]
             for row in ocr["deidentified_records"]}
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    sources = pd.read_csv(REGISTRY / "source_linkage_private.csv", dtype=str)
    joined = images[["image_id", "kidney_alias"]].merge(
        sources[["image_id", "source_path"]], on="image_id", validate="one_to_one")
    joined["probe"] = joined.source_path.map(lambda path: probe.get(str(Path(path).resolve()).lower(), "unknown"))

    def category(values):
        kinds = set(values)
        if kinds == {"convex"}:
            return "explicit_convex_only"
        if kinds == {"linear"}:
            return "explicit_linear_only"
        if kinds == {"convex", "linear"}:
            return "both_explicit_probes"
        return "unknown_present"

    kidney_probe = joined.groupby("kidney_alias").probe.agg(category).rename("probe_category")
    predictions = pd.read_csv(EXPERIMENT / "development_predictions.csv")
    predictions = predictions.merge(kidney_probe, on="kidney_alias", validate="many_to_one")
    rows = []
    for (group, probe_category), frame in predictions.groupby(["group", "probe_category"]):
        labels = frame.label.to_numpy(int)
        rows.append({"model": group, "probe_category": probe_category,
                     "patients": frame.patient_alias.nunique(), "kidneys": len(frame),
                     "prevalence": float(labels.mean()),
                     "auroc": float(roc_auc_score(labels, frame.score))
                     if len(frame) >= 10 and len(set(labels)) == 2 else None,
                     "ap": float(average_precision_score(labels, frame.score))
                     if len(frame) >= 10 and labels.sum() else None})
    output = pd.DataFrame(rows).sort_values(["model", "probe_category"])
    output.to_csv(EXPERIMENT / "development_probe_shift.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
