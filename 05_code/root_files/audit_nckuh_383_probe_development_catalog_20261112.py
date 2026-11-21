"""Re-score archived revised-cohort development predictions by probe route.

This is a diagnostic catalog, not a new model-selection or test procedure.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, ROOT, digest


EXP = ROOT / "02_experiments"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
OUT = ROUTER / "archived_development_probe_catalog"
OTHER_SCORES = (
    "old_fixed_chain", "old_fixed_plus_hierarchy", "old_full_final", "new_revised_stack",
    "old_labeled_train_only_score", "added_train_only_score",
)


def score_columns(frame):
    return [column for column in ("score", *OTHER_SCORES, *[c for c in frame if c.startswith("base_")])
            if column in frame and pd.api.types.is_numeric_dtype(frame[column])]


def main():
    if (OUT / "audit.json").exists():
        raise FileExistsError("Preserve archived prediction catalog")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    people = pd.read_csv(REGISTRY / "patient_assignments.csv")
    route = pd.read_csv(ROUTER / "image_routes.csv")
    routed = images[["image_id", "kidney_alias"]].merge(
        route[["image_id", "route"]], on="image_id", validate="one_to_one"
    )
    allowed = {probe: set(routed.loc[routed.route.eq(probe), "kidney_alias"])
               for probe in ("convex", "linear")}
    reference = kidneys[["kidney_alias", "patient_alias", "kidney_APN"]].merge(
        people[["patient_alias", "role"]], on="patient_alias", validate="many_to_one"
    ).set_index("kidney_alias")
    paths = sorted(EXP.glob("NCKUH_revised_383_*/**/*development_predictions.csv"))
    assert len(paths) >= 20
    records, skipped = [], []
    for path in paths:
        if "historical_210_route_audit" in str(path):
            continue
        frame = pd.read_csv(path)
        if "kidney_alias" not in frame or "patient_alias" not in frame:
            skipped.append({"file": str(path.relative_to(ROOT)), "reason": "no kidney and patient aliases"})
            continue
        if not set(frame.kidney_alias) <= set(reference.index):
            skipped.append({"file": str(path.relative_to(ROOT)), "reason": "not registered revised-cohort kidneys"})
            continue
        indexed = reference.loc[frame.kidney_alias]
        assert frame.patient_alias.tolist() == indexed.patient_alias.tolist(), path
        assert indexed.role.eq("development").all(), path
        for label_column in ("label", "kidney_APN"):
            if label_column in frame:
                assert frame[label_column].astype(int).tolist() == indexed.kidney_APN.tolist(), path
        candidates = score_columns(frame)
        if not candidates:
            skipped.append({"file": str(path.relative_to(ROOT)), "reason": "no numeric score column"})
            continue
        grouping = [column for column in ("group", "seed") if column in frame]
        partitions = frame.groupby(grouping, dropna=False) if grouping else [((), frame)]
        for key, part in partitions:
            descriptor = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
            if part.kidney_alias.duplicated().any():
                skipped.append({"file": str(path.relative_to(ROOT)), "reason": "duplicate kidney within group"})
                continue
            for probe in ("convex", "linear"):
                subset = part[part.kidney_alias.isin(allowed[probe])]
                if len(subset) < 20:
                    continue
                labels = reference.loc[subset.kidney_alias, "kidney_APN"].to_numpy(int)
                if len(np.unique(labels)) != 2:
                    continue
                for column in candidates:
                    values = subset[column].to_numpy(float)
                    if not np.isfinite(values).all():
                        continue
                    records.append({"file": str(path.relative_to(ROOT)),
                                    "file_sha256": digest(path), "group": str(descriptor.get("group", "")),
                                    "seed": str(descriptor.get("seed", "")), "score_column": column,
                                    "probe": probe, "patients": subset.patient_alias.nunique(),
                                    "kidneys": len(subset), "positive_fraction": float(labels.mean()),
                                    "auroc": float(roc_auc_score(labels, values)),
                                    "ap": float(average_precision_score(labels, values))})
    catalog = pd.DataFrame(records).sort_values(["probe", "auroc", "ap"], ascending=[True, False, False])
    assert len(catalog) >= 40
    OUT.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT / "archived_probe_development_scores.csv", index=False)
    audit = {"source_routes_sha256": digest(ROUTER / "image_routes.csv"),
             "archived_prediction_files": len(set(catalog.file)), "probe_scores": len(catalog),
             "skipped": skipped, "catalog_sha256": digest(OUT / "archived_probe_development_scores.csv"),
             "apn_test_scored": False,
             "interpretation": "Post-hoc catalog of repeatedly explored development predictions. Highest values are not selected or validated models. Require training-only selection and new independent confirmation."}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for probe in ("convex", "linear"):
        print(probe)
        print(catalog[catalog.probe.eq(probe)][
            ["file", "group", "score_column", "patients", "kidneys", "auroc", "ap"]
        ].head(12).to_string(index=False))
    print(json.dumps({key: value for key, value in audit.items() if key != "skipped"}, indent=2))


if __name__ == "__main__":
    main()
