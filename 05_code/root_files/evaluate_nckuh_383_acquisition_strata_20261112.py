"""Select an image-only head separately within each frame-size dataset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, digest


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
STRATA = EXP / "NCKUH_revised_383_acquisition_strata_20261112"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
LEGACY = EXP / "NCKUH_revised_383_legacy_features_20261112"
DINO = EXP / "NCKUH_revised_383_dinov3_20261111"
OUT = STRATA / "method_screen"
GROUPS = ("frame_720x960", "frame_450x560")
CANDIDATES = ("rad_color", "tile", "dinov3_class", "dinov3_patch", "uri_mean", "radiomics")
SEED = 20261112


def metric(labels, scores):
    return {"auroc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores))}


def aggregate(values, images, kidneys):
    grouped = images.groupby("kidney_alias", sort=False).indices
    return np.asarray([
        np.r_[values[grouped[key]].mean(0), values[grouped[key]].std(0)]
        for key in kidneys.kidney_alias
    ], dtype=np.float64)


def fit_predict(name, image_values, images, kidneys, fit_aliases, query):
    train_images = np.flatnonzero(images.patient_alias.isin(fit_aliases))
    train_kidneys = np.flatnonzero(kidneys.patient_alias.isin(fit_aliases))
    labels = kidneys.kidney_APN.to_numpy(int)
    if name == "radiomics":
        kidney_values = aggregate(image_values, images, kidneys)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(), SelectKBest(f_classif, k=64),
            LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=SEED),
        )
    else:
        scaler = StandardScaler().fit(image_values[train_images])
        scaled = scaler.transform(image_values)
        pca = PCA(32, random_state=SEED).fit(scaled[train_images])
        kidney_values = aggregate(pca.transform(scaled), images, kidneys)
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                               max_iter=3000, random_state=SEED),
        )
    model.fit(kidney_values[train_kidneys], labels[train_kidneys])
    return model.predict_proba(kidney_values[query])[:, 1]


def threshold_for_sensitivity(labels, scores, minimum=.85):
    candidates = np.r_[0.0, np.sort(np.unique(scores)), 1.0 + 1e-9]
    feasible = []
    for value in candidates:
        predicted = scores >= value
        sensitivity = predicted[labels == 1].mean()
        specificity = (~predicted[labels == 0]).mean()
        if sensitivity >= minimum:
            feasible.append((specificity, value, sensitivity))
    _, threshold, sensitivity = max(feasible, key=lambda item: (item[0], item[1]))
    return float(threshold), float(sensitivity)


def bootstrap(target, scores, repetitions=2000):
    groups = [target.index[target.patient_alias.eq(alias)].to_numpy()
              for alias in sorted(target.patient_alias.unique())]
    labels = target.kidney_APN.to_numpy(int)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(repetitions):
        selected = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if np.unique(labels[selected]).size < 2:
            continue
        result = metric(labels[selected], scores[selected])
        samples.append((result["auroc"], result["ap"]))
    interval = np.quantile(np.asarray(samples), [0.025, 0.975], axis=0)
    return {"valid_bootstraps": len(samples),
            "auroc_95pct": interval[:, 0].tolist(),
            "ap_95pct": interval[:, 1].tolist()}


def main():
    protocol = json.loads((STRATA / "dataset_protocol.json").read_text(encoding="utf-8"))
    assert digest(REGISTRY / "cohort_lock.json") == protocol["cohort_lock_sha256"]
    full_images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    index = {key: i for i, key in enumerate(full_images.image_id)}
    raw_radiomics = pd.read_csv(CACHE / "radiomics.csv").set_index("image_id").reindex(full_images.image_id)
    columns = [column for column in raw_radiomics if not column.endswith(("_Energy", "_TotalEnergy"))]
    radiomics = raw_radiomics[columns].to_numpy(float)
    radiomics[~np.isfinite(radiomics)] = np.nan
    sources = {
        "rad_color": np.load(LEGACY / "rad_color.npy", mmap_mode="r"),
        "tile": np.load(LEGACY / "tile_mean.npy", mmap_mode="r"),
        "dinov3_class": np.load(DINO / "dinov3_class.npy", mmap_mode="r"),
        "dinov3_patch": np.load(DINO / "dinov3_patch.npy", mmap_mode="r"),
        "uri_mean": np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r").mean(1),
        "radiomics": radiomics,
    }
    assert all(len(values) == len(full_images) for values in sources.values())
    report = {"source_protocol_sha256": digest(STRATA / "dataset_protocol.json"),
              "candidates": list(CANDIDATES), "selection": "Highest minimum of patient-OOF AUROC and AP; AUROC tie-break. No development score in selection.",
              "threshold": "Training-OOF highest specificity subject to sensitivity >=0.85; never tuned on development.",
              "test_scored": False, "datasets": {}}
    OUT.mkdir(parents=True, exist_ok=True)

    for group in GROUPS:
        folder = STRATA / group
        assert digest(folder / "patient_manifest.csv") == protocol["datasets"][group]["patient_manifest_sha256"]
        people = pd.read_csv(folder / "patient_manifest.csv")
        people = people[people.role.isin(("train", "development"))].reset_index(drop=True)
        images = pd.read_csv(folder / "image_manifest.csv")
        images = images[images.patient_alias.isin(people.patient_alias)].reset_index(drop=True)
        kidneys = pd.read_csv(folder / "kidney_manifest.csv")
        kidneys = kidneys[kidneys.patient_alias.isin(people.patient_alias)].reset_index(drop=True)
        rows = np.asarray([index[key] for key in images.image_id])
        representations = {name: np.asarray(values[rows], dtype=np.float64)
                           for name, values in sources.items()}
        train_people = people[people.role.eq("train")].reset_index(drop=True)
        dev_aliases = set(people.loc[people.role.eq("development"), "patient_alias"])
        dev = np.flatnonzero(kidneys.patient_alias.isin(dev_aliases))
        target = kidneys.iloc[dev].reset_index(drop=True)
        splitter = StratifiedKFold(4, shuffle=True, random_state=SEED)
        train_kidneys = np.flatnonzero(kidneys.patient_alias.isin(train_people.patient_alias))
        candidate_results = {}
        for name in CANDIDATES:
            oof = np.full(len(kidneys), np.nan)
            for fit, hold in splitter.split(train_people, train_people.DMSA_APN):
                fit_aliases = set(train_people.patient_alias.iloc[fit])
                hold_aliases = set(train_people.patient_alias.iloc[hold])
                query = np.flatnonzero(kidneys.patient_alias.isin(hold_aliases))
                oof[query] = fit_predict(name, representations[name], images, kidneys,
                                         fit_aliases, query)
            assert np.isfinite(oof[train_kidneys]).all()
            candidate_results[name] = metric(kidneys.kidney_APN.iloc[train_kidneys],
                                             oof[train_kidneys])
            candidate_results[name]["oof_scores"] = oof[train_kidneys]
        winner = max(CANDIDATES, key=lambda name: (
            min(candidate_results[name]["auroc"], candidate_results[name]["ap"]),
            candidate_results[name]["auroc"],
        ))
        oof_scores = candidate_results[winner].pop("oof_scores")
        for name in CANDIDATES:
            candidate_results[name].pop("oof_scores", None)
        oof_labels = kidneys.kidney_APN.iloc[train_kidneys].to_numpy(int)
        threshold, oof_sensitivity = threshold_for_sensitivity(oof_labels, oof_scores)
        score = fit_predict(winner, representations[winner], images, kidneys,
                            set(train_people.patient_alias), dev)
        labels = target.kidney_APN.to_numpy(int)
        predicted = score >= threshold
        result = {"train_patients": len(train_people), "development_patients": target.patient_alias.nunique(),
                  "development_kidneys": len(target), "positive_fraction": float(labels.mean()),
                  "training_oof": candidate_results, "selected_method": winner,
                  "selected_training_oof_threshold": threshold,
                  "training_oof_sensitivity_at_threshold": oof_sensitivity,
                  "development": metric(labels, score) | bootstrap(target, score) | {
                      "sensitivity_at_training_threshold": float(predicted[labels == 1].mean()),
                      "specificity_at_training_threshold": float((~predicted[labels == 0]).mean()),
                  }}
        report["datasets"][group] = result
        target[["patient_alias", "kidney_alias", "side", "kidney_APN"]].assign(
            selected_method=winner, score=score, prediction=predicted.astype(int)
        ).to_csv(OUT / f"{group}_development_predictions.csv", index=False)
        print(group, winner, result["development"], flush=True)
    (OUT / "separate_development_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
