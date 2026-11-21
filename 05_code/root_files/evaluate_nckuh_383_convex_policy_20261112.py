"""Same-evaluation-set comparison of all-probe and high-confidence convex-only training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_nckuh_383_oof_stacking_20261112 import dino_model, geometry_model, make_geometry
from evaluate_nckuh_383_view_paired_20261112 import paired_features


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "02_experiments/NCKUH_revised_383_audit_20261111"
DINO = ROOT / "02_experiments/NCKUH_revised_383_dinov3_20261111"
GEOMETRY = ROOT / "02_experiments/NCKUH_revised_383_relative_geometry_20261112"
PROBE = ROOT / "02_experiments/NCKUH_revised_383_probe_geometry_20261112"
OUT = ROOT / "02_experiments/NCKUH_revised_383_convex_policy_20261112"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve convex policy comparison")
    images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    kidneys = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv")
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    labels = pd.read_csv(PROBE / "registered_image_probe_labels.csv")
    image_probe = images[["image_id", "patient_alias"]].merge(labels, on="image_id", how="left")
    image_probe["probe"] = image_probe.probe.fillna("unclassified")
    patient_probe = image_probe.groupby("patient_alias").probe.agg(
        lambda x: "convex_only" if set(x) == {"convex"} else
        "linear_only" if set(x) == {"linear"} else "unknown_or_mixed")
    kidneys = kidneys.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    kidneys["probe_policy"] = kidneys.patient_alias.map(patient_probe)
    y = kidneys.kidney_APN.to_numpy(int)
    all_train = np.flatnonzero(kidneys.role.eq("train"))
    convex_train = np.flatnonzero(kidneys.role.eq("train") & kidneys.probe_policy.eq("convex_only"))
    convex_dev = np.flatnonzero(kidneys.role.eq("development") & kidneys.probe_policy.eq("convex_only"))
    assert kidneys.patient_alias.iloc[convex_train].nunique() == 55
    assert kidneys.patient_alias.iloc[convex_dev].nunique() == 14 and len(convex_dev) == 28
    dino_values = paired_features(np.load(DINO / "dinov3_class.npy"), images, kidneys)
    geometry_values = make_geometry(np.load(GEOMETRY / "relative_geometry_image.npy"), images, kidneys)
    target = kidneys.iloc[convex_dev].reset_index(drop=True)
    clusters = [np.flatnonzero(target.patient_alias.eq(pid)) for pid in sorted(target.patient_alias.unique())]
    rng = np.random.default_rng(20261112)
    draws = [np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
             for _ in range(5000)]
    draws = [idx for idx in draws if np.unique(y[convex_dev][idx]).size == 2]
    results, predictions = [], []
    for policy, train in (("all_probe_train", all_train), ("convex_only_train", convex_train)):
        dm = dino_model().fit(dino_values[train], y[train])
        gm = geometry_model().fit(geometry_values[train], y[train])
        scores = {"dino": dm.predict_proba(dino_values[convex_dev])[:, 1],
                  "geometry": gm.predict_proba(geometry_values[convex_dev])[:, 1]}
        scores["fixed_dino_geometry_mean"] = (scores["dino"] + scores["geometry"]) / 2
        for source, score in scores.items():
            auc = np.asarray([roc_auc_score(y[convex_dev][idx], score[idx]) for idx in draws])
            ap = np.asarray([average_precision_score(y[convex_dev][idx], score[idx]) for idx in draws])
            group = f"{policy}_{source}"
            results.append({"group": group, "training_patients": kidneys.patient_alias.iloc[train].nunique(),
                            "development_patients": target.patient_alias.nunique(), "kidneys": len(convex_dev),
                            "prevalence": y[convex_dev].mean(), "auroc": roc_auc_score(y[convex_dev], score),
                            "ap": average_precision_score(y[convex_dev], score),
                            "auroc_ci_low": np.percentile(auc, 2.5), "auroc_ci_high": np.percentile(auc, 97.5),
                            "ap_ci_low": np.percentile(ap, 2.5), "ap_ci_high": np.percentile(ap, 97.5)})
            predictions.extend({"group": group, "patient_alias": row.patient_alias,
                                "kidney_alias": row.kidney_alias, "label": int(label), "score": float(value)}
                               for row, label, value in zip(target.itertuples(index=False), y[convex_dev], score))
    pd.DataFrame(results).to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "probe_gate": "OCR plus sector-geometry classifier with patient-grouped OOF AUROC and balanced accuracy >=.90",
        "comparison": "all-probe versus convex-only training evaluated on identical 14-patient convex-only development subset",
        "models": "fixed DINOv3 paired-view and relative-geometry heads selected previously using training only",
        "test_scored": False,
        "warning": "Small exploratory subset; not a main-result estimate and not evidence that probe type causally changes performance"}, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
