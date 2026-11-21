"""Side-specific negative-kidney patch memory under corrected NCKUH DMSA labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_clinical_features_20261103 as C
import nckuh_dmsa_fullcohort_fullframe_20261103 as F
from nckuh_only_study_20261102 import digest, write_json


ROOT = F.ROOT
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_negative_memory_20261103"
CACHE = ROOT / "02_experiments/NCKUH_normal_patch_probe_20261102"
META = ROOT / "02_experiments/NCKUH_only_study_20261102/image_manifest_private.csv"
REFERENCE = "fullcohort_rad_color_image_plus_patient_direct"
GROUPS = ("dinov3_negative_memory_lr", "dinov3_negative_memory_rbf",
          "dinov3_negative_memory_lr_plus_instance_reference")


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered negative-memory protocol")
    F.require_lock()
    inputs = [CACHE / "dinov3_roi_patches.npy", CACHE / "patch_offsets.npy", META,
              F.OUT / "patient_assignments.csv", F.OUT / "instance_development_predictions.csv", Path(__file__)]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "Same corrected 210-patient, 414-kidney, 820-image DMSA cohort and fixed patient roles",
        "hypothesis": "A focal APN appearance may be better represented as deviation from same-view DMSA-negative renal patches than as a global positive-class embedding",
        "patches": "Existing frozen DINOv3 in-mask patch cache, matched by deidentified image_id; no new image label enters encoding",
        "memory": "For each outer draw, fit PCA32 and 128 MiniBatchKMeans prototypes per LL/LT/RL/RT only from training-role kidneys with side-specific DMSA label0; robustly standardize nearest-prototype distance within each view",
        "kidney_features": "Mean, standard deviation, 90th/95th/99th percentile, maximum, top10-percent mean and positive-distance fraction across all patches from one kidney",
        "head": "Training-kidney calibrated balanced logistic regression or fixed RBF-SVM; one fixed equal-probability fusion with the unchanged image-instance reference",
        "evaluation": "Kidney-level development roles only; patient keeps both kidneys and every view in one role; exploratory test unscored",
        "prohibited": "No development memory, no test memory, no positive patch label, no lesion claim and no development-selected fusion weight",
        "target": "Development AUROC>0.8 and AP>0.8; locked confirmation remains required",
        "groups": list(GROUPS), "inputs": {str(path): digest(path) for path in inputs},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {"source_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "inputs": {str(path): digest(path) for path in inputs}})
    print("Registered corrected side-specific negative-memory experiment", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(Path(__file__)) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["inputs"].items():
        assert digest(Path(name)) == expected, name


def summarize_distances(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    count = max(1, int(np.ceil(.10 * len(values))))
    return np.asarray([values.mean(), values.std(), *np.percentile(values, [90, 95, 99]),
                       values.max(), np.partition(values, len(values) - count)[-count:].mean(),
                       (values > 0).mean()], dtype=np.float32)


def fit_memory(source, offsets: np.ndarray, metadata: pd.DataFrame, images: pd.DataFrame,
               training_images: np.ndarray, negative_images: np.ndarray, seed: int):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed + 901)
    image_lookup = metadata.reset_index().set_index("image_id")["index"]
    old_rows = image_lookup.reindex(images.image_id).to_numpy(int)

    def patches(rows: np.ndarray) -> np.ndarray:
        blocks = [np.asarray(source[offsets[old_rows[index]]:offsets[old_rows[index] + 1]]) for index in rows]
        return np.concatenate(blocks)

    negative = patches(negative_images)
    negative /= np.linalg.norm(negative, axis=1, keepdims=True).clip(1e-8)
    fit = negative if len(negative) <= 20000 else negative[rng.choice(len(negative), 20000, replace=False)]
    pca = PCA(32, whiten=True, random_state=seed).fit(fit)
    memory, scales = {}, {}
    for view in ("LL", "LT", "RL", "RT"):
        rows = negative_images[images.view.iloc[negative_images].eq(view).to_numpy()]
        values = patches(rows)
        values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
        reduced = pca.transform(values).astype(np.float32)
        count = min(128, len(reduced))
        model = MiniBatchKMeans(n_clusters=count, batch_size=2048, n_init=3,
                                max_iter=200, random_state=seed).fit(reduced)
        centers = model.cluster_centers_.astype(np.float32)
        distance = np.sqrt(((reduced[:, None] - centers[None]) ** 2).sum(2)).min(1)
        center = float(np.median(distance)); scale = float(np.median(np.abs(distance - center)) / .6745)
        memory[view] = centers
        scales[view] = (center, max(scale, 1e-4))
    return pca, memory, scales, old_rows


def kidney_anomaly_features(source, offsets, metadata, images, kidneys, pca, memory, scales, old_rows):
    output = []
    for kidney in kidneys.itertuples(index=False):
        rows = np.flatnonzero(images.kidney_alias.eq(kidney.kidney_alias))
        all_distances = []
        for row in rows:
            old = old_rows[row]
            values = np.array(source[offsets[old]:offsets[old + 1]], dtype=np.float32, copy=True)
            values /= np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
            reduced = pca.transform(values).astype(np.float32)
            centers = memory[images.view.iloc[row]]
            distance = np.sqrt(((reduced[:, None] - centers[None]) ** 2).sum(2)).min(1)
            center, scale = scales[images.view.iloc[row]]
            all_distances.extend(((distance - center) / scale).tolist())
        output.append(summarize_distances(np.asarray(all_distances)))
    return np.asarray(output)


def fit_head(values, labels, train, development, seed, kind):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    classifier = (LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                     max_iter=3000, random_state=seed) if kind == "lr" else
                  SVC(C=.1, kernel="rbf", gamma="scale", class_weight="balanced",
                      probability=True, random_state=seed))
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), classifier)
    model.fit(values[train], labels[train])
    return model.predict_proba(values[development])[:, 1], model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed negative-memory predictions")
    patients, kidneys, images = C.load_reference()
    metadata = pd.read_csv(META, dtype={"patient_id": str})
    assert set(images.image_id).issubset(set(metadata.image_id))
    source = np.load(CACHE / "dinov3_roi_patches.npy", mmap_mode="r")
    offsets = np.load(CACHE / "patch_offsets.npy")
    assignments = pd.read_csv(F.OUT / "patient_assignments.csv")
    reference_frame = pd.read_csv(F.OUT / "instance_development_predictions.csv")
    reference = reference_frame[reference_frame.group.eq(REFERENCE)].set_index(["seed", "kidney_alias"]).score
    labels = kidneys.kidney_APN.to_numpy(int)
    metrics, predictions, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in F.SEEDS:
            mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
            image_roles = images.patient_alias.map(mapping)
            kidney_roles = kidneys.patient_alias.map(mapping)
            training_images = np.flatnonzero(image_roles.eq("train"))
            negative_images = training_images[images.kidney_APN.iloc[training_images].eq(0).to_numpy()]
            train = np.flatnonzero(kidney_roles.eq("train"))
            development = np.flatnonzero(kidney_roles.eq("development"))
            pca, memory, scales, old_rows = fit_memory(
                source, offsets, metadata, images, training_images, negative_images, seed)
            values = kidney_anomaly_features(
                source, offsets, metadata, images, kidneys, pca, memory, scales, old_rows)
            lr_score, lr_model = fit_head(values, labels, train, development, seed, "lr")
            rbf_score, rbf_model = fit_head(values, labels, train, development, seed, "rbf")
            aliases = kidneys.kidney_alias.iloc[development]
            reference_score = np.asarray([reference.loc[(seed, alias)] for alias in aliases])
            scores = {GROUPS[0]: lr_score, GROUPS[1]: rbf_score,
                      GROUPS[2]: (lr_score + reference_score) / 2}
            joblib.dump({"pca": pca, "memory": memory, "scales": scales,
                         "lr": lr_model, "rbf": rbf_model},
                        OUT / "models" / f"{seed}_negative_memory.joblib", compress=3)
            audits.append({"seed": seed,
                "training_negative_kidneys": int(kidneys.iloc[train].kidney_APN.eq(0).sum()),
                "training_negative_images": len(negative_images),
                "memory_sizes": {view: len(value) for view, value in memory.items()},
                "labels_used_in_patch_encoder": [], "exploratory_test_scored": False})
            for group, score in scores.items():
                result = F.metric(labels[development], score)
                metrics.append({"seed": seed, "role": "development", "group": group,
                                "patients": 42, "kidneys": len(development), **result})
                for local, index in enumerate(development):
                    predictions.append({"seed": seed, "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index], "side": kidneys.side.iloc[index],
                        "label": int(labels[index]), "score": float(score[local])})
            print(f"Completed negative-memory draw {seed}", flush=True)
    frame = pd.DataFrame(metrics); frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    write_json(OUT / "memory_audit.json", {"draws": audits})
    summary = frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False})
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args(); {"prepare": prepare, "run": run}[args.stage]()
