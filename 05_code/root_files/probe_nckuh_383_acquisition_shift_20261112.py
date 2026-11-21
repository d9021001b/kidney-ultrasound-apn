"""Training-only, label-blind probe of old-versus-added image-domain shift."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
FEATURES = EXP / "NCKUH_revised_383_legacy_features_20261112"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def oof_auc(values, labels, pca=False):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if pca:
        steps.append(PCA(32, random_state=20261112))
    steps.append(LogisticRegression(C=.1, max_iter=2000))
    model = make_pipeline(*steps)
    scores = cross_val_predict(
        model, values, labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=20261112),
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(labels, scores))


def patient_image_features(values, images, patients):
    paired = base.patient_pair_features(values, images, patients)
    half = paired.shape[1] // 2
    return np.c_[
        (paired[:, :half] + paired[:, half:]) / 2,
        np.abs(paired[:, :half] - paired[:, half:]),
    ]


def main():
    lock = json.loads((REGISTRY / "cohort_lock.json").read_text(encoding="utf-8"))
    source = REGISTRY / "contour_image_manifest.csv"
    assert digest(source) == lock["files_sha256"][source.name]
    manifest = pd.read_csv(source)
    assignments = pd.read_csv(REGISTRY / "patient_assignments.csv")
    patients = assignments[
        assignments.role.eq("train") & assignments.patient_alias.isin(manifest.patient_alias)
    ].reset_index(drop=True)
    assert len(patients) == 204
    selected = np.flatnonzero(manifest.patient_alias.isin(patients.patient_alias))
    images = manifest.iloc[selected].reset_index(drop=True)

    old = pd.read_csv(
        EXP / "NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv",
        dtype={"patient_id": str},
    )
    old_ids = set(old.loc[old.DMSA_APN.notna(), "patient_id"])
    linkage = pd.read_csv(REGISTRY / "patient_linkage_private.csv", dtype={"patient_id": str})
    source_ids = patients.patient_alias.map(linkage.set_index("patient_alias").patient_id)
    added = (~source_ids.isin(old_ids)).to_numpy(int)
    assert (int(added.sum()), int((1 - added).sum())) == (90, 114)

    tile = np.load(FEATURES / "tile_mean.npy")[selected]
    usfm = np.load(FEATURES / "usfm_gap.npy")[selected]
    source_auc = {}
    for name, values in (("tile", tile), ("usfm", usfm), ("tile_plus_usfm", np.c_[tile, usfm])):
        source_auc[name] = oof_auc(patient_image_features(values, images, patients), added, pca=True)

    qc = pd.read_csv(CACHE / "extraction_qc.csv")
    qc["roi_aspect"] = qc.roi_width / qc.roi_height
    qc["mask_fill"] = qc.mask_pixels / (qc.roi_height * qc.roi_width)
    columns = ["height", "width", "roi_height", "roi_width", "mask_pixels",
               "mask_fraction_in_source_frame", "roi_aspect", "mask_fill"]
    geometry = qc.groupby("patient_alias")[columns].agg(["mean", "std"])
    geometry.columns = ["_".join(pair) for pair in geometry.columns]
    source_auc["image_geometry_only"] = oof_auc(
        geometry.reindex(patients.patient_alias).to_numpy(), added
    )
    qc = qc[qc.patient_alias.isin(patients.patient_alias)].copy()
    group = dict(zip(patients.patient_alias, np.where(added, "added", "old_labeled")))
    qc["source_group"] = qc.patient_alias.map(group)
    medians = qc.groupby("source_group")[columns].median().to_dict(orient="index")

    probe = pd.read_csv(
        EXP / "NCKUH_revised_383_probe_geometry_20261112/registered_image_probe_labels.csv"
    ).merge(qc[["image_id", "patient_alias", "source_group"]], on="image_id", how="inner")
    probe_summary = {}
    for name, rows in probe.groupby("source_group"):
        probe_summary[name] = {
            "covered_patients": int(rows.patient_alias.nunique()),
            "images": len(rows),
            "probe_images": {key: int(value) for key, value in Counter(rows.probe).items()},
        }

    result = {
        "scope": "Training-role patients only; source-group prediction, not APN prediction.",
        "patients": {"old_labeled": 114, "added": 90},
        "source_group_fivefold_oof_auroc": source_auc,
        "source_frame_and_roi_medians": medians,
        "partial_probe_evidence": probe_summary,
        "caveat": "Covariate shift is observed. Probe, resolution, source archive, and clinical spectrum are confounded; this does not prove which factor causes APN performance loss.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acquisition_shift.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
