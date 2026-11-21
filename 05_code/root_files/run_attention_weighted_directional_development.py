"""Development-only attention weighting of image-level directional I3 features."""
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
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_voting_validation as V
from renal_dinov3_native_probe import indices_for
from run_gated_attention_mil_development import build_bags, train_one


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_attention_weighted_directional_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
DENSE_IMAGE = FEATURE_ROOT / "features" / "dense_tight_image.npy"
I3_IMAGE = ROOT / "CYCH_interpretable_features_20260913" / "features" / "I3-direction-original_image.csv"
PAIR_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
CANDIDATES = ("GATED-DENSE+I3", "PAIR+ATTN-I3", "GATED-DENSE+ATTN-I3")


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
    for name in ("models", "source", "features"):
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
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only attention-weighted directional-feature screen; no new exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "clinical_rationale": "Acute focal nephritis commonly appears as a focal echogenic, often upper-pole wedge-shaped abnormality; uniform image averaging may dilute a positive instance.",
        "clinical_sources": [
            "https://pubmed.ncbi.nlm.nih.gov/17987762/",
            "https://pubmed.ncbi.nlm.nih.gov/12069464/",
        ],
        "method_sources": [
            "https://proceedings.mlr.press/v80/ilse18a.html",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6938161/",
        ],
        "dense_image": {"path": str(DENSE_IMAGE), "sha256": sha256(DENSE_IMAGE)},
        "i3_image": {"path": str(I3_IMAGE), "sha256": sha256(I3_IMAGE), "features": 25},
        "attention": "Same prespecified gated-attention DenseNet patient-bag model: train-only scaler and PCA(32), hidden 32, attention 16, 300 epochs",
        "directional_pooling": "Use DenseNet-derived attention weights to compute weighted mean and weighted standard deviation of the 25 I3 image features; fit the same fixed patient-level L2 logistic head on 58 training patients",
        "ablations": {
            "GATED-DENSE+I3": "RAD + gated-attention DenseNet + unchanged I3, equal probabilities",
            "PAIR+ATTN-I3": "unchanged RAD/DenseNet pair plus attention-weighted I3, equal three-branch probabilities",
            "GATED-DENSE+ATTN-I3": "RAD + gated-attention DenseNet + attention-weighted I3, equal probabilities",
        },
        "selection_gate": "Relative to the unchanged pair+I3 reference, both mean development metrics improve and at least 6/10 draws improve jointly; choose highest mean AUROC/AP only among passing candidates",
        "interpretation": "Attention weights rank images within a patient and are not lesion masks or causal explanations.",
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
    print("Attention-weighted I3 protocol locked", flush=True)


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


def weighted_directional_features(
    image_features: np.ndarray,
    attention: np.ndarray,
    manifest: pd.DataFrame,
    patients: pd.DataFrame,
) -> np.ndarray:
    output = []
    for row, patient_id in enumerate(patients.patient_id):
        indices = np.flatnonzero(manifest.patient_id.eq(patient_id).to_numpy())
        values = image_features[indices]
        weights = attention[row, : len(indices)]
        weights = weights / weights.sum()
        mean = np.sum(weights[:, None] * values, axis=0)
        variance = np.sum(weights[:, None] * np.square(values - mean), axis=0)
        output.append(np.concatenate([mean, np.sqrt(np.maximum(variance, 0.0))]))
    result = np.asarray(output, dtype=np.float32)
    assert result.shape == (98, 50) and np.isfinite(result).all()
    return result


def metric_row(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"seed": seed, "group": group, **V.score_metrics(labels, scores)}


def evaluate() -> None:
    require_lock()
    import torch

    torch.set_num_threads(4)
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    dense = np.load(DENSE_IMAGE)
    i3_frame = pd.read_csv(I3_IMAGE)
    assert i3_frame.image_id.tolist() == manifest.image_id.tolist()
    i3_image = i3_frame.drop(columns="image_id").to_numpy(np.float32)
    assert dense.shape[0] == i3_image.shape[0] == 349 and i3_image.shape[1] == 25
    assert np.isfinite(dense).all() and np.isfinite(i3_image).all()
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    pair_reference = load_reference(PAIR_PREDICTIONS, "PAIR", aliases)
    rad_reference = load_reference(PAIR_PREDICTIONS, "RAD", aliases)
    i3_reference = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)

    metrics = []
    predictions = []
    attention_rows = []
    provenance = []
    for seed in R.SEEDS:
        indices = indices_for(patients, assignments, seed)
        train_ids = set(patients.patient_id.iloc[indices["train"]])
        train_images = manifest.patient_id.isin(train_ids).to_numpy()
        scaler = StandardScaler().fit(dense[train_images])
        standardized = scaler.transform(dense)
        pca = PCA(n_components=32, random_state=seed).fit(standardized[train_images])
        reduced = pca.transform(standardized).astype(np.float32)
        bags, masks, image_ids = build_bags(reduced, manifest, patients)
        model, dense_scores, attention, loss_record = train_one(seed, indices["train"], labels, bags, masks)

        directional = weighted_directional_features(i3_image, attention, manifest, patients)
        head = make_head(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            head.fit(directional[indices["train"]], labels[indices["train"]])
        directional_scores = head.predict_proba(directional)[:, 1]
        assert np.isfinite(directional_scores).all()

        model_path = OUT / "models" / f"gated_attention_{seed}.pt"
        torch.save(model.state_dict(), model_path)
        head_path = OUT / "models" / f"attention_i3_{seed}.joblib"
        joblib.dump(head, head_path, compress=3)
        preprocessing_path = OUT / "models" / f"dense_preprocessing_{seed}.joblib"
        joblib.dump({"scaler": scaler, "pca": pca}, preprocessing_path, compress=3)
        np.save(OUT / "features" / f"attention_i3_{seed}.npy", directional)
        provenance.append(
            {
                "seed": seed,
                "attention_model_sha256": sha256(model_path),
                "i3_head_sha256": sha256(head_path),
                "preprocessing_sha256": sha256(preprocessing_path),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                **loss_record,
            }
        )

        pair = pair_reference[seed]
        rad = rad_reference[seed]
        i3 = i3_reference[seed]
        scores = {
            "PAIR+I3-REFERENCE": (2.0 * pair + i3) / 3.0,
            "GATED-DENSE+I3": (rad + dense_scores + i3) / 3.0,
            "PAIR+ATTN-I3": (2.0 * pair + directional_scores) / 3.0,
            "GATED-DENSE+ATTN-I3": (rad + dense_scores + directional_scores) / 3.0,
        }
        development = indices["validation"]
        for group, score in scores.items():
            metrics.append(metric_row(seed, group, labels[development], score[development]))
            for index in development:
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "role": "development_validation",
                        "patient_alias": aliases[index],
                        "label": int(labels[index]),
                        "score": float(score[index]),
                    }
                )
        for index in development:
            for position, image_id in enumerate(image_ids[index]):
                attention_rows.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "patient_alias": aliases[index],
                        "image_id": image_id,
                        "attention_weight": float(attention[index, position]),
                    }
                )
        print(f"Attention-weighted directional features complete: {seed}", flush=True)

    frame = pd.DataFrame(metrics)
    reference = frame.loc[frame.group.eq("PAIR+I3-REFERENCE")].set_index("seed")
    decisions = []
    for group in CANDIDATES:
        candidate = frame.loc[frame.group.eq(group)].set_index("seed")
        delta_auc = candidate.auroc - reference.auroc
        delta_ap = candidate.average_precision - reference.average_precision
        joint_wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        decisions.append(
            {
                "group": group,
                "delta_auroc": float(delta_auc.mean()),
                "delta_ap": float(delta_ap.mean()),
                "joint_wins": joint_wins,
                "passed": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and joint_wins >= 6),
                "mean_score": float(candidate[["auroc", "average_precision"]].to_numpy().mean()),
            }
        )
    passed = sorted((item for item in decisions if item["passed"]), key=lambda item: (-item["mean_score"], item["group"]))
    decision = {
        "selection_role": "development_validation",
        "candidates": decisions,
        "selected": passed[0]["group"] if passed else None,
    }

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
    pd.DataFrame(attention_rows).to_csv(OUT / "development_image_attention.csv", index=False)
    write_json(OUT / "training_provenance.json", provenance)
    write_json(OUT / "development_decision.json", decision)

    lines = [
        "# Attention-weighted directional development screen",
        "",
        "DenseNet-derived patient-bag attention was used to weight the existing image-level I3 features.",
        "No new exploratory-test result was computed. Attention is not lesion truth.",
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
        f"Selected candidate: {decision['selected'] or 'none'}.",
        f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`",
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
