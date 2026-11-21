"""Route contour-ready ultrasound by probe field shape without APN labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from classify_nckuh_383_probe_geometry_20261112 import model
from nckuh_probe_blind_sector import image_frame, sector_features
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
AUDIT = EXP / "NCKUH_revised_383_audit_20261111"
PROBE = EXP / "NCKUH_revised_383_probe_geometry_20261112"
OUT = EXP / "NCKUH_revised_383_probe_router_20261112"
SEED = 20261112
WORKBOOK = ROOT / "2004-2019-NCKUH-亞大(1).xlsx"


def classify_probability(value, lower=.10, upper=.90):
    if value <= lower:
        return "linear"
    if value >= upper:
        return "convex"
    return "unknown"


def main():
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    for name in ("contour_image_manifest.csv", "patient_assignments.csv"):
        assert digest(REGISTRY / name) == lock["files_sha256"][name]
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    people = pd.read_csv(REGISTRY / "patient_assignments.csv")
    qc = pd.read_csv(EXP / "NCKUH_revised_383_contour_cache_20261111/extraction_qc.csv")
    registered = pd.read_csv(PROBE / "registered_image_probe_labels.csv")
    private = pd.read_csv(AUDIT / "source_linkage_private.csv", dtype=str)
    sheet = load_workbook(WORKBOOK, read_only=True, data_only=True)["完整資料彙整"]
    rows = sheet.iter_rows(values_only=True)
    headers = list(next(rows))
    id_index, probe_index = headers.index("id"), headers.index("Probe")
    workbook = pd.DataFrame(
        [(str(row[id_index]).strip().lstrip("0"), str(row[probe_index]).strip().lower())
         for row in rows if row[id_index] is not None],
        columns=["normalized_id", "workbook_probe"],
    )
    assert len(workbook) == 383 and workbook.normalized_id.is_unique
    assert workbook.workbook_probe.value_counts().to_dict() == {
        "convex": 244, "linear": 124, "both": 15,
    }
    private["normalized_id"] = private.patient_id.str.lstrip("0")
    joined = images[["image_id", "patient_alias", "source_type"]].merge(
        people[["patient_alias", "role"]], on="patient_alias", validate="many_to_one"
    ).merge(qc[["image_id", "height", "width"]], on="image_id", validate="one_to_one"
    ).merge(registered[["image_id", "probe", "evidence"]], on="image_id", how="left",
            validate="one_to_one"
    ).merge(private[["image_id", "source_path", "normalized_id"]], on="image_id", validate="one_to_one"
    ).merge(workbook, on="normalized_id", validate="many_to_one")
    assert len(joined) == 1362 and joined.image_id.is_unique
    assert joined.groupby("patient_alias").workbook_probe.nunique().eq(1).all()
    conflict = joined.evidence.eq("OCR") & joined.workbook_probe.isin(("convex", "linear")) & joined.probe.ne(joined.workbook_probe)
    assert not conflict.any(), "Workbook probe conflicts with explicit image OCR"
    assert all(Path(path).is_file() for path in joined.source_path)

    x = np.stack([sector_features(image_frame(Path(row.source_path),
                            panel=row.source_type == "panel_with_mask"))
                  for row in joined.itertuples(index=False)])
    workbook_single = joined.workbook_probe.isin(("convex", "linear"))
    ocr_both = joined.workbook_probe.eq("both") & joined.evidence.eq("OCR") & joined.probe.isin(("convex", "linear"))
    labeled = workbook_single | ocr_both
    reference = np.where(workbook_single, joined.workbook_probe, joined.probe)
    train = np.flatnonzero(labeled & joined.role.eq("train"))
    dev = np.flatnonzero(labeled & joined.role.eq("development"))
    test = np.flatnonzero(labeled & joined.role.eq("exploratory_test"))
    y = (reference == "convex").astype(int)
    groups = joined.patient_alias.to_numpy()
    assert set(groups[train]).isdisjoint(groups[dev])
    assert len(train) >= 100 and len(dev) >= 20

    oof = np.full(len(joined), np.nan)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fit, hold in splitter.split(x[train], y[train], groups[train]):
        oof[train[hold]] = model().fit(x[train[fit]], y[train[fit]]).predict_proba(x[train[hold]])[:, 1]
    fitted = model().fit(x[train], y[train])
    dev_probability = fitted.predict_proba(x[dev])[:, 1]
    test_probability = fitted.predict_proba(x[test])[:, 1]
    probability = fitted.predict_proba(x)[:, 1]

    def validation(indices, values):
        prediction = (values >= .5).astype(int)
        selective = np.asarray([classify_probability(v) for v in values])
        kept = selective != "unknown"
        actual = y[indices]
        return {"images": len(indices), "patients": len(set(groups[indices])),
                "convex_images": int(actual.sum()), "linear_images": int((1 - actual).sum()),
                "auroc": float(roc_auc_score(actual, values)),
                "balanced_accuracy_at_0_5": float(balanced_accuracy_score(actual, prediction)),
                "confusion_linear_convex_at_0_5": confusion_matrix(actual, prediction, labels=[0, 1]).tolist(),
                "selective_coverage_at_0_1_0_9": float(kept.mean()),
                "selective_accuracy": float((selective[kept] == reference[indices][kept]).mean()) if kept.any() else None}

    report = {"source_lock_sha256": digest(REGISTRY / "cohort_lock.json"),
              "workbook_sha256": digest(WORKBOOK),
              "workbook_patient_probe_counts": workbook.workbook_probe.value_counts().to_dict(),
              "source_probe_sha256": digest(PROBE / "registered_image_probe_labels.csv"),
              "training_oof": validation(train, oof[train]),
              "development_probe_validation": validation(dev, dev_probability),
              "heldout_probe_only_validation": validation(test, test_probability),
              "geometry": "Sector widths by depth and upper-edge shape from source pixels only: fan-like convex versus near-rectangular linear field; no kidney mask, APN label, or patient-registry membership input.",
              "router": "Single-probe workbook label first; for both-probe patients, explicit image OCR first, then geometry classifier only at p<=0.1 or p>=0.9; otherwise abstain.",
              "caveats": "Workbook Probe is patient-level and both needs image-level adjudication. OCR is not manufacturer metadata. Probe frequency is not inferred. Geometry model fits training-role probe labels only; the geometry extraction was refined after development-image inspection, so development probe validation is exploratory. Held-out test Probe labels are used for acquisition QA only; APN test labels and scores are never used."}
    joined["convex_probability"] = probability
    joined["geometry_route"] = [classify_probability(p) for p in probability]
    joined["route"] = np.where(labeled, reference, joined.geometry_route)
    joined["route_evidence"] = np.where(workbook_single, "workbook", np.where(ocr_both, "OCR", "sector_geometry"))
    joined.loc[joined.route.eq("unknown"), "route_evidence"] = "abstain"
    route = joined.groupby("patient_alias").route.agg(
        lambda labels: labels.iloc[0] if labels.nunique() == 1 and labels.iloc[0] != "unknown" else "unknown_or_mixed"
    )
    patients = people[people.patient_alias.isin(joined.patient_alias)].copy()
    patients["probe_route"] = patients.patient_alias.map(route)
    patients.loc[patients.patient_alias.isin(joined.loc[joined.workbook_probe.eq("both"), "patient_alias"]), "probe_route"] = "both_image_level"
    report["image_routes"] = joined.route.value_counts().to_dict()
    report["patient_routes"] = patients.probe_route.value_counts().to_dict()
    report["patient_routes_by_role"] = pd.crosstab(patients.role, patients.probe_route).to_dict()
    report["test_APN_scored"] = False

    OUT.mkdir(parents=True, exist_ok=True)
    joined[["image_id", "patient_alias", "role", "height", "width", "workbook_probe", "probe", "evidence",
            "convex_probability", "geometry_route", "route", "route_evidence"]].to_csv(
                OUT / "image_routes.csv", index=False
            )
    patients[["patient_alias", "role", "probe_route"]].to_csv(OUT / "patient_routes.csv", index=False)
    (OUT / "router_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
