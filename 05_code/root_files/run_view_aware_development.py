"""Development-only fixed-view fusion for segmented kidney ultrasound images."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import renal_random_validation as R
from renal_dinov3_native_probe import indices_for
from run_uniusnet_frozen_development import load_reference, metric_row, read_csv
from run_usf_mae_development import sha256, write_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_view_aware_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
RANDOM_ROOT = ROOT / "CYCH_random_validation_20260913"
DENSE_PATH = FEATURE_ROOT / "features" / "dense_tight_image.npy"
RADIOMICS_PATH = RANDOM_ROOT / "features" / "radiomics_model_features.csv"
PAIR_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
VIEWS = ("LL", "LT", "RL", "RT")


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))


def prepare() -> None:
    for name in ("models", "source"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source = FEATURE_ROOT / name
        destination = OUT / name
        if destination.exists():
            assert sha256(destination) == sha256(source)
        else:
            shutil.copy2(source, destination)
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    assert len(patients) == 98 and int(patients.label.sum()) == 52
    assert len(manifest) == 349 and manifest.image_id.nunique() == 349
    assert set(manifest.view) == set(VIEWS)
    assert int(manifest.groupby(["patient_alias", "view"]).size().max()) == 1
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only fixed-view fusion; no exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "method_sources": [
            {
                "name": "Multi-instance Deep Learning with Graph Convolutional Neural Networks for Diagnosis of Kidney Diseases Using Ultrasound Imaging",
                "url": "https://doi.org/10.1007/978-3-030-32689-0_15",
                "relevance": "Treats multiple heterogeneous two-dimensional kidney ultrasound views as a patient bag",
            },
            {
                "name": "Multi-instance deep learning of ultrasound imaging data for pattern classification of congenital abnormalities of the kidney and urinary tract in children",
                "url": "https://doi.org/10.1016/j.urology.2020.05.019",
                "relevance": "Reports an advantage for multi-view patient-level kidney ultrasound classification",
            },
        ],
        "views": {
            "order": VIEWS,
            "images_per_patient_view": "at most one",
            "missing_slots": int(len(patients) * len(VIEWS) - len(manifest)),
        },
        "image_features": {
            "dense": {
                "description": "Frozen RadImageNet DenseNet-121 tight segmented-kidney embedding",
                "sha256": sha256(DENSE_PATH),
            },
            "radiomics": {
                "description": "Existing 457 PyRadiomics image descriptors",
                "sha256": sha256(RADIOMICS_PATH),
            },
        },
        "training_only_preprocessing": "Image-level median imputation when needed, standardization, and PCA(32), all fitted only to images from the 58 training patients in each draw",
        "patient_representation": "Concatenate the 32-dimensional image vector in fixed LL/LT/RL/RT slots; a missing slot is the training-centered zero vector; append four presence indicators",
        "heads": "Balanced L2 logistic regression C=0.1 for view-aware DenseNet and radiomics; their probability mean is VIEW-PAIR",
        "candidate_gate": "Relative to fixed pair: mean AUROC gain at least 0.03, mean AP gain at least 0.02, and at least 6/10 draws improve both",
        "directional_vote": "2/3 VIEW-PAIR + 1/3 existing I3, assessed against the existing pair+I3 reference",
        "vote_gate": "Both mean metrics improve and at least 6/10 draws improve jointly",
        "reference_hashes": {"pair": sha256(PAIR_PREDICTIONS), "i3": sha256(I3_PREDICTIONS)},
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        write_json(protocol_path, protocol)
    lock_path = OUT / "protocol_locked.json"
    if not lock_path.exists():
        write_json(
            lock_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(protocol_path),
                "source_sha256": sha256(Path(__file__)),
            },
        )
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    print("Fixed-view protocol locked", flush=True)


def build_view_slots(
    image_values: np.ndarray,
    manifest: pd.DataFrame,
    patients: pd.DataFrame,
) -> np.ndarray:
    values = np.asarray(image_values, dtype=np.float32)
    if values.shape[0] != len(manifest):
        raise ValueError("Image feature and manifest rows differ")
    width = values.shape[1]
    result = np.zeros((len(patients), len(VIEWS) * width + len(VIEWS)), dtype=np.float32)
    patient_row = {alias: index for index, alias in enumerate(patients.patient_alias)}
    view_row = {view: index for index, view in enumerate(VIEWS)}
    seen: set[tuple[str, str]] = set()
    for image_index, row in enumerate(manifest.itertuples(index=False)):
        key = (row.patient_alias, row.view)
        if key in seen:
            raise ValueError(f"Duplicate patient-view slot: {key}")
        seen.add(key)
        p = patient_row[row.patient_alias]
        v = view_row[row.view]
        result[p, v * width : (v + 1) * width] = values[image_index]
        result[p, len(VIEWS) * width + v] = 1.0
    return result


def fit_image_projection(values: np.ndarray, train_images: np.ndarray, seed: int):
    imputer = SimpleImputer(strategy="median").fit(values[train_images])
    imputed = imputer.transform(values)
    scaler = StandardScaler().fit(imputed[train_images])
    standardized = scaler.transform(imputed)
    components = min(32, int(train_images.sum()) - 1, standardized.shape[1])
    pca = PCA(n_components=components, random_state=seed).fit(standardized[train_images])
    projected = pca.transform(standardized).astype(np.float32)
    assert projected.shape == (len(values), components) and np.isfinite(projected).all()
    return projected, {"imputer": imputer, "scaler": scaler, "pca": pca}


def make_patient_head(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


def evaluate() -> None:
    require_lock()
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    dense = np.load(DENSE_PATH)
    radiomics_frame = pd.read_csv(RADIOMICS_PATH).set_index("image_id")
    radiomics = radiomics_frame.reindex(manifest.image_id).to_numpy(float)
    assert dense.shape == (349, 1024)
    assert radiomics.shape == (349, 457)
    assert np.isfinite(dense).all()
    pair_reference = load_reference(PAIR_PREDICTIONS, "PAIR", aliases)
    i3_reference = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)

    metrics = []
    predictions = []
    scores_by_seed = {}
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            indices = indices_for(patients, assignments, seed)
            train_aliases = set(patients.patient_alias.iloc[indices["train"]])
            train_images = manifest.patient_alias.isin(train_aliases).to_numpy()
            dense_image, dense_projection = fit_image_projection(dense, train_images, seed)
            rad_image, rad_projection = fit_image_projection(radiomics, train_images, seed)
            dense_patient = build_view_slots(dense_image, manifest, patients)
            rad_patient = build_view_slots(rad_image, manifest, patients)
            dense_head = make_patient_head(seed)
            rad_head = make_patient_head(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense_head.fit(dense_patient[indices["train"]], labels[indices["train"]])
                rad_head.fit(rad_patient[indices["train"]], labels[indices["train"]])
            dense_score = dense_head.predict_proba(dense_patient)[:, 1]
            rad_score = rad_head.predict_proba(rad_patient)[:, 1]
            view_pair = (dense_score + rad_score) / 2.0
            view_vote = (2.0 * view_pair + i3_reference[seed]) / 3.0
            incumbent = (2.0 * pair_reference[seed] + i3_reference[seed]) / 3.0
            scores_by_seed[seed] = {
                "VIEW-DENSE": dense_score,
                "VIEW-RAD": rad_score,
                "VIEW-PAIR": view_pair,
                "VIEW-PAIR+I3": view_vote,
                "PAIR": pair_reference[seed],
                "PAIR+I3-REFERENCE": incumbent,
            }
            development = indices["validation"]
            for group, score in scores_by_seed[seed].items():
                metrics.append(metric_row(seed, group, labels[development], score[development]))
                for patient_index in development:
                    predictions.append(
                        {
                            "seed": seed,
                            "group": group,
                            "role": "development_validation",
                            "patient_alias": aliases[patient_index],
                            "label": int(labels[patient_index]),
                            "score": float(score[patient_index]),
                        }
                    )
            joblib.dump(
                {
                    "dense_projection": dense_projection,
                    "radiomics_projection": rad_projection,
                    "dense_head": dense_head,
                    "radiomics_head": rad_head,
                    "views": VIEWS,
                },
                OUT / "models" / f"fixed_view_{seed}.joblib",
                compress=3,
            )
            print(f"Fixed-view development: {seed}", flush=True)

    frame = pd.DataFrame(metrics)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(
            (
                (frame.loc[frame.group.eq(group), "auroc"] > 0.8)
                & (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)
            ).sum()
        )
        for group in summary.group
    ]
    wide = frame.pivot(index="seed", columns="group", values=["auroc", "average_precision"])
    pair_auc_gain = wide["auroc"]["VIEW-PAIR"] - wide["auroc"]["PAIR"]
    pair_ap_gain = wide["average_precision"]["VIEW-PAIR"] - wide["average_precision"]["PAIR"]
    pair_wins = int(((pair_auc_gain > 0) & (pair_ap_gain > 0)).sum())
    vote_auc_gain = wide["auroc"]["VIEW-PAIR+I3"] - wide["auroc"]["PAIR+I3-REFERENCE"]
    vote_ap_gain = (
        wide["average_precision"]["VIEW-PAIR+I3"]
        - wide["average_precision"]["PAIR+I3-REFERENCE"]
    )
    vote_wins = int(((vote_auc_gain > 0) & (vote_ap_gain > 0)).sum())
    decision = {
        "selection_role": "development_validation",
        "view_pair": {
            "delta_auroc": float(pair_auc_gain.mean()),
            "delta_ap": float(pair_ap_gain.mean()),
            "joint_wins": pair_wins,
            "passed": bool(pair_auc_gain.mean() >= 0.03 and pair_ap_gain.mean() >= 0.02 and pair_wins >= 6),
        },
        "view_pair_plus_i3": {
            "delta_auroc": float(vote_auc_gain.mean()),
            "delta_ap": float(vote_ap_gain.mean()),
            "joint_wins": vote_wins,
            "promoted": bool(vote_auc_gain.mean() > 0 and vote_ap_gain.mean() > 0 and vote_wins >= 6),
        },
        "exploratory_test_access": "prohibited",
    }
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    write_json(OUT / "development_decision.json", decision)
    lines = [
        "# Fixed-view kidney-ultrasound development screen",
        "",
        "No exploratory-test result was computed.",
        "",
        "Each patient has at most one image in each locked LL, LT, RL, and RT slot. This preserves side and plane rather than averaging all images together.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |"
        )
    lines.extend(["", "## Decision", "", f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`"])
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate", "all"))
    args = parser.parse_args()
    stages = ("prepare", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()


if __name__ == "__main__":
    main()
