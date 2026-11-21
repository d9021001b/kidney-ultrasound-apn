"""Development-only multi-instance pooling for CYCH kidney ultrasound images."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_voting_validation as V
from renal_dinov3_native_probe import indices_for


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_image_score_mil_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
PAIR_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
BRANCHES = {
    "DENSE": FEATURE_ROOT / "features" / "dense_tight_image.npy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str})


def make_head(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


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
    assert manifest.groupby("patient_id").label.nunique().max() == 1
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only multi-instance learning screen; no new exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "method_source": {
            "paper": "Multi-instance deep learning of ultrasound imaging data for pattern classification of congenital abnormalities of the kidney and urinary tract in children",
            "doi": "https://doi.org/10.1016/j.urology.2020.05.019",
            "code": "https://github.com/YS181818/CAKUT_diagnosis",
        },
        "adaptation": "Fixed image-level logistic heads replace the published fine-tuned CNN so the already extracted segmented-ROI representations can be screened without target encoder fine-tuning.",
        "branches": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in BRANCHES.items()
        },
        "input": "Mandatory segmented kidney ROI; frozen RadImageNet DenseNet-121 tight-ROI image representation plus the existing patient-level PyRadiomics branch",
        "head": "Fit only images from the 58 training patients; patient-balanced image weights; train-only median imputation, constant removal, standardization, balanced L2 logistic regression C=0.1",
        "bag_pooling": ["mean of DenseNet image scores", "maximum of DenseNet image scores"],
        "pair_adaptation": "Average the pooled DenseNet bag score with the unchanged patient-level PyRadiomics probability",
        "candidate_selection": "Choose the pooling rule with highest mean of development AUROC and AP; ties prefer mean pooling",
        "candidate_gate": "Mean development AUROC and AP are each no more than 0.02 below the existing patient-level fixed pair",
        "ensemble": "Only an eligible bag candidate may form fixed 2/3 bag score + 1/3 directional I3 score",
        "ensemble_gate": "Compared with the existing 2/3 fixed pair + 1/3 I3: both mean metrics improve and at least 6/10 draws improve jointly",
        "reference_hashes": {
            "pair": sha256(PAIR_PREDICTIONS),
            "i3": sha256(I3_PREDICTIONS),
        },
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
    print("Image-score MIL protocol locked", flush=True)


def load_reference(path: Path, group: str, aliases: list[str]) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame.role.eq("development_validation") & frame.group.eq(group),
        ["seed", "patient_alias", "label", "score"],
    ]
    result = {}
    for seed in R.SEEDS:
        subset = frame.loc[frame.seed.eq(seed)].set_index("patient_alias")
        assert len(subset) == 20 and subset.index.is_unique
        result[seed] = subset.reindex(aliases).score.to_numpy(float)
    return result


def patient_pool(scores: np.ndarray, manifest: pd.DataFrame, patients: pd.DataFrame, mode: str) -> np.ndarray:
    output = []
    for patient_id in patients.patient_id:
        values = scores[manifest.patient_id.eq(patient_id).to_numpy()]
        assert 1 <= len(values) <= 10 and np.isfinite(values).all()
        output.append(values.mean() if mode == "mean" else values.max())
    return np.asarray(output)


def metric_row(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"seed": seed, "group": group, **V.score_metrics(labels, scores)}


def evaluate() -> None:
    require_lock()
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    patient_labels = patients.label.to_numpy(int)
    image_labels = manifest.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    features = {name: np.load(path) for name, path in BRANCHES.items()}
    assert all(value.shape[0] == 349 and np.isfinite(value).all() for value in features.values())
    pair_reference = load_reference(PAIR_PREDICTIONS, "PAIR", aliases)
    rad_reference = load_reference(PAIR_PREDICTIONS, "RAD", aliases)
    i3_reference = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)

    metrics = []
    predictions = []
    bag_scores: dict[int, dict[str, np.ndarray]] = {}
    model_records = []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            patient_indices = indices_for(patients, assignments, seed)
            train_ids = set(patients.patient_id.iloc[patient_indices["train"]])
            train_images = manifest.patient_id.isin(train_ids).to_numpy()
            train_counts = manifest.loc[train_images].groupby("patient_id").size()
            sample_weight = manifest.loc[train_images, "patient_id"].map(lambda value: 1.0 / train_counts[value]).to_numpy()
            branch_scores = []
            for branch, values in features.items():
                head = make_head(seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    head.fit(values[train_images], image_labels[train_images], logisticregression__sample_weight=sample_weight)
                scores = head.predict_proba(values)[:, 1]
                assert np.isfinite(scores).all()
                model_path = OUT / "models" / f"{branch}_{seed}.joblib"
                joblib.dump(head, model_path, compress=3)
                assert np.allclose(joblib.load(model_path).predict_proba(values)[:, 1], scores, atol=1e-12)
                branch_scores.append(scores)
                model_records.append({"seed": seed, "branch": branch, "sha256": sha256(model_path), "train_images": int(train_images.sum())})
            assert len(branch_scores) == 1
            dense_image_scores = branch_scores[0]
            bag_scores[seed] = {}
            development = patient_indices["validation"]
            for mode in ("mean", "max"):
                group = f"PAIR-IMAGE-{mode.upper()}"
                dense_bag = patient_pool(dense_image_scores, manifest, patients, mode)
                scores = (rad_reference[seed] + dense_bag) / 2.0
                bag_scores[seed][group] = scores
                metrics.append(metric_row(seed, group, patient_labels[development], scores[development]))
                for index in development:
                    predictions.append(
                        {
                            "seed": seed,
                            "group": group,
                            "role": "development_validation",
                            "patient_alias": aliases[index],
                            "label": int(patient_labels[index]),
                            "score": float(scores[index]),
                        }
                    )
            pair = pair_reference[seed]
            i3 = i3_reference[seed]
            incumbent = (2.0 * pair + i3) / 3.0
            metrics.append(metric_row(seed, "PAIR", patient_labels[development], pair[development]))
            metrics.append(metric_row(seed, "PAIR+I3-REFERENCE", patient_labels[development], incumbent[development]))
            print(f"Image-score MIL complete: {seed}", flush=True)

    frame = pd.DataFrame(metrics)
    means = frame.groupby("group")[["auroc", "average_precision"]].mean()
    pair_mean = means.loc["PAIR"]
    candidates = []
    for group in ("PAIR-IMAGE-MEAN", "PAIR-IMAGE-MAX"):
        delta = means.loc[group] - pair_mean
        candidates.append(
            {
                "group": group,
                "delta_auroc_vs_pair": float(delta.auroc),
                "delta_ap_vs_pair": float(delta.average_precision),
                "eligible": bool((delta >= -0.02 - 1e-12).all()),
                "mean_score": float(means.loc[group].mean()),
            }
        )
    eligible = sorted(
        (item for item in candidates if item["eligible"]),
        key=lambda item: (-item["mean_score"], item["group"] != "PAIR-IMAGE-MEAN"),
    )
    selected = eligible[0]["group"] if eligible else None
    decision = {"selection_role": "development_validation", "candidates": candidates, "selected": selected}

    if selected:
        ensemble_rows = []
        for seed in R.SEEDS:
            development = indices_for(patients, assignments, seed)["validation"]
            i3 = i3_reference[seed]
            candidate = bag_scores[seed][selected]
            scores = (2.0 * candidate + i3) / 3.0
            row = metric_row(seed, "BAG+I3", patient_labels[development], scores[development])
            metrics.append(row)
            ensemble_rows.append(row)
            for index in development:
                predictions.append(
                    {
                        "seed": seed,
                        "group": "BAG+I3",
                        "role": "development_validation",
                        "patient_alias": aliases[index],
                        "label": int(patient_labels[index]),
                        "score": float(scores[index]),
                    }
                )
        frame = pd.DataFrame(metrics)
        compare = frame.loc[frame.group.isin(["PAIR+I3-REFERENCE", "BAG+I3"])]
        wide = compare.pivot(index="seed", columns="group", values=["auroc", "average_precision"])
        delta_auc = wide["auroc"]["BAG+I3"] - wide["auroc"]["PAIR+I3-REFERENCE"]
        delta_ap = wide["average_precision"]["BAG+I3"] - wide["average_precision"]["PAIR+I3-REFERENCE"]
        joint_wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        decision["ensemble"] = {
            "delta_auroc": float(delta_auc.mean()),
            "delta_ap": float(delta_ap.mean()),
            "joint_wins": joint_wins,
            "promoted": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and joint_wins >= 6),
        }

    frame = pd.DataFrame(metrics)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(((frame.loc[frame.group.eq(group), "auroc"] > 0.8) &
             (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)).sum())
        for group in summary.group
    ]
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    write_json(OUT / "model_provenance.json", model_records)
    write_json(OUT / "development_decision.json", decision)

    lines = [
        "# Image-score multi-instance development screen",
        "",
        "This development-only experiment adapts published kidney-ultrasound patient-bag pooling to the fixed CYCH cohort.",
        "No new exploratory-test result was computed.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Selected eligible bag rule: {selected or 'none'}.",
        f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`",
        "",
        "Primary method reference: https://doi.org/10.1016/j.urology.2020.05.019",
        "Official implementation: https://github.com/YS181818/CAKUT_diagnosis",
    ])
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
