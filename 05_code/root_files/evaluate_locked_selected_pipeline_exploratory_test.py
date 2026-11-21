"""One-time exploratory-test evaluation of the locked CYCH selected pipeline.

The selected model, preprocessing objects, fixed equal-vote rule, and threshold
were established before this script reads any exploratory-test score or label.
The ten test draws overlap in patients and therefore characterize random-split
transport only; they are not an independent external validation cohort.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

import renal_random_validation as R
from renal_dinov3_native_probe import indices_for
from run_gated_attention_mil_development import build_bags
from run_graph_attention_mil_development import normalized_rbf_adjacency


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "CYCH_reliable_negative_graph_mil_development_20260914"
FEATURE_ROOT = ROOT / "CYCH_dinov3_native_probe_20260913"
I3_ROOT = ROOT / "CYCH_interpretable_features_20260913"
OUT = ROOT / "CYCH_locked_selected_exploratory_test_20260915"

FEATURES = FEATURE_ROOT / "features" / "dense_tight_image.npy"
PATIENTS = MODEL_ROOT / "patients.csv"
MANIFEST = MODEL_ROOT / "image_manifest.csv"
ASSIGNMENTS = MODEL_ROOT / "random_patient_assignments.csv"
BASE_PREDICTIONS = FEATURE_ROOT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = I3_ROOT / "patient_predictions_by_role.csv"
DEVELOPMENT_PREDICTIONS = MODEL_ROOT / "development_predictions.csv"
CANDIDATE_LOCK = MODEL_ROOT / "candidate_lock.json"
SEEDS = list(range(20260913, 20260923))
GROUPS = (
    "RELIABLE-NEGATIVE-GCN-MIL",
    "PYRADIOMICS",
    "DIRECTIONAL-WAVELET-GLRLM-I3",
    "FINAL-EQUAL-VOTE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def input_paths() -> list[Path]:
    fixed = [
        FEATURES,
        PATIENTS,
        MANIFEST,
        ASSIGNMENTS,
        BASE_PREDICTIONS,
        I3_PREDICTIONS,
        DEVELOPMENT_PREDICTIONS,
        CANDIDATE_LOCK,
        MODEL_ROOT / "protocol.json",
        MODEL_ROOT / "protocol_locked.json",
        MODEL_ROOT / "development_decision.json",
    ]
    for seed in SEEDS:
        fixed.extend(
            [
                MODEL_ROOT / "models" / f"preprocess_{seed}.joblib",
                MODEL_ROOT / "models" / f"reliable_negative_graph_{seed}.pth",
            ]
        )
    missing = [str(path) for path in fixed if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing locked inputs: " + ", ".join(missing))
    return fixed


def prepare_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source").mkdir(exist_ok=True)
    selected = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if selected.get("selected_candidate") != "RELIABLE-NEG-GRAPH+I3":
        raise RuntimeError("Unexpected development-selected candidate")
    protocol = {
        "scope": "One-time exploratory internal-test evaluation after development selection",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "fixed_draws": {"seeds": SEEDS, "train": 58, "development": 20, "exploratory_test": 20},
        "fitted_objects": "Load the ten saved reliable-negative GCN-MIL state dictionaries and their training-only StandardScaler/PCA objects; no refitting",
        "score_rule": "Arithmetic mean of reliable-negative GCN-MIL, PyRadiomics, and directional wavelet/GLRLM I3 probabilities",
        "operating_threshold": 0.5,
        "test_use": "No model, weight, threshold, feature, or calibration selection is permitted after test access",
        "interpretation": "Overlapping random-split exploratory internal tests, not independent external validation",
        "development_reconstruction": "Before accepting test output, reproduce the archived development final scores from the loaded models within numerical tolerance",
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing test protocol differs from the fixed protocol")
    else:
        write_json(protocol_path, protocol)

    lock = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(protocol_path),
        "development_candidate_lock_sha256": sha256(CANDIDATE_LOCK),
        "inputs_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in input_paths()
        },
        "status": "locked_before_exploratory_test_evaluation",
    }
    lock_path = OUT / "evaluation_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in lock.items() if key != "utc"}
        previous_comparable = {key: value for key, value in previous.items() if key != "utc"}
        if previous_comparable != comparable:
            raise RuntimeError("Existing evaluation lock does not match current inputs")
    else:
        write_json(lock_path, lock)
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    print("Locked selected-pipeline exploratory-test evaluation before score access", flush=True)


def require_lock() -> dict:
    lock_path = OUT / "evaluation_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("Run prepare-lock before exploratory-test evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["source_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("Evaluation source changed after lock")
    if lock["protocol_sha256"] != sha256(OUT / "protocol.json"):
        raise RuntimeError("Evaluation protocol changed after lock")
    if lock["development_candidate_lock_sha256"] != sha256(CANDIDATE_LOCK):
        raise RuntimeError("Development candidate lock changed after test lock")
    current = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in input_paths()
    }
    if current != lock["inputs_sha256"]:
        raise RuntimeError("A locked evaluation input changed")
    return lock


def scores_for(
    frame: pd.DataFrame,
    seed: int,
    role: str,
    group: str,
    aliases: list[str],
) -> np.ndarray:
    subset = frame.loc[
        frame.seed.eq(seed) & frame.role.eq(role) & frame.group.eq(group),
        ["patient_alias", "label", "score"],
    ].set_index("patient_alias")
    if len(subset) != 20 or not subset.index.is_unique or set(subset.index) != set(aliases):
        raise RuntimeError(f"Invalid {group} predictions for {seed}/{role}")
    values = subset.reindex(aliases).score.to_numpy(float)
    if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1:
        raise RuntimeError(f"Invalid probability values for {group}/{seed}/{role}")
    return values


def development_reference(
    frame: pd.DataFrame, seed: int, aliases: list[str]
) -> np.ndarray:
    subset = frame.loc[
        frame.seed.eq(seed)
        & frame.role.eq("development_validation")
        & frame.group.eq("RELIABLE-NEG-GRAPH+I3"),
        ["patient_alias", "score"],
    ].set_index("patient_alias")
    if len(subset) != 20 or set(subset.index) != set(aliases):
        raise RuntimeError(f"Invalid archived development prediction for {seed}")
    return subset.reindex(aliases).score.to_numpy(float)


def metric_row(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    precision, recall, _ = precision_recall_curve(labels, scores)
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "seed": seed,
        "role": "exploratory_internal_test",
        "group": group,
        "patients": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "auprc_trapezoid": float(abs(np.trapezoid(precision, recall))),
        "sensitivity_at_0_5": float(tp / (tp + fn)),
        "specificity_at_0_5": float(tn / (tn + fp)),
        "accuracy_at_0_5": float(accuracy_score(labels, predicted)),
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, np.clip(scores, 1e-8, 1 - 1e-8))),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        subset = metrics.loc[metrics.group.eq(group)]
        rows.append(
            {
                "group": group,
                "draws": len(subset),
                "mean_auroc": subset.auroc.mean(),
                "std_auroc": subset.auroc.std(ddof=1),
                "range_auroc_low": subset.auroc.min(),
                "range_auroc_high": subset.auroc.max(),
                "mean_average_precision": subset.average_precision.mean(),
                "std_average_precision": subset.average_precision.std(ddof=1),
                "range_average_precision_low": subset.average_precision.min(),
                "range_average_precision_high": subset.average_precision.max(),
                "mean_auprc_trapezoid": subset.auprc_trapezoid.mean(),
                "std_auprc_trapezoid": subset.auprc_trapezoid.std(ddof=1),
                "mean_sensitivity_at_0_5": subset.sensitivity_at_0_5.mean(),
                "mean_specificity_at_0_5": subset.specificity_at_0_5.mean(),
                "mean_accuracy_at_0_5": subset.accuracy_at_0_5.mean(),
                "mean_brier": subset.brier.mean(),
                "std_brier": subset.brier.std(ddof=1),
                "joint_auroc_ap_over_0_8_draws": int(
                    ((subset.auroc > 0.8) & (subset.average_precision > 0.8)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate() -> None:
    lock = require_lock()
    import torch
    from torch import nn

    class ReliableNegativeGraphMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn1 = nn.Linear(32, 64, bias=False)
            self.gcn2 = nn.Linear(64, 32, bias=False)
            self.dropout = nn.Dropout(0.25)
            self.attention_v = nn.Linear(32, 16)
            self.attention_u = nn.Linear(32, 16)
            self.attention_w = nn.Linear(16, 1, bias=False)
            self.bag_classifier = nn.Linear(32, 1)
            self.instance_classifier = nn.Linear(32, 1)

        def forward(self, values, valid):
            adjacency = normalized_rbf_adjacency(values, valid)
            hidden = torch.relu(self.gcn1(torch.bmm(adjacency, values)))
            hidden = self.dropout(hidden)
            hidden = torch.relu(self.gcn2(torch.bmm(adjacency, hidden)))
            raw = self.attention_w(
                torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))
            ).squeeze(-1)
            raw = raw.masked_fill(~valid, -1e9)
            attention = torch.softmax(raw, dim=1)
            pooled = torch.sum(attention.unsqueeze(-1) * hidden, dim=1)
            bag_logits = self.bag_classifier(pooled).squeeze(-1)
            instance_logits = self.instance_classifier(hidden).squeeze(-1)
            return bag_logits, attention, instance_logits

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    patients = pd.read_csv(PATIENTS)
    manifest = pd.read_csv(MANIFEST)
    assignments = pd.read_csv(ASSIGNMENTS)
    features = np.load(FEATURES)
    base = pd.read_csv(BASE_PREDICTIONS)
    i3 = pd.read_csv(I3_PREDICTIONS)
    development = pd.read_csv(DEVELOPMENT_PREDICTIONS)
    labels = patients.label.to_numpy(int)
    patient_aliases = patients.patient_alias.tolist()
    if len(patients) != 98 or labels.sum() != 52 or features.shape != (349, 1024):
        raise RuntimeError("Unexpected cohort or feature dimensions")

    metric_rows = []
    prediction_rows = []
    instance_rows = []
    reconstruction_errors = []
    for seed in SEEDS:
        index = indices_for(patients, assignments, seed)
        preprocess = joblib.load(MODEL_ROOT / "models" / f"preprocess_{seed}.joblib")
        reduced = preprocess["pca"].transform(preprocess["scaler"].transform(features)).astype(np.float32)
        bags, masks, image_ids = build_bags(reduced, manifest, patients)
        model = ReliableNegativeGraphMIL()
        model.load_state_dict(
            torch.load(
                MODEL_ROOT / "models" / f"reliable_negative_graph_{seed}.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        with torch.inference_mode():
            logits, attention, instance_scores = model(bags, masks)
            graph_all = torch.sigmoid(logits).numpy()
            attention = attention.numpy()
            instance_scores = torch.sigmoid(instance_scores).numpy()

        dev = index["validation"]
        dev_aliases = [patient_aliases[value] for value in dev]
        dev_rad = scores_for(base, seed, "development_validation", "RAD", dev_aliases)
        dev_i3 = scores_for(
            i3, seed, "development_validation", "I3-direction-original", dev_aliases
        )
        reconstructed = (graph_all[dev] + dev_rad + dev_i3) / 3.0
        archived = development_reference(development, seed, dev_aliases)
        reconstruction_errors.append(float(np.max(np.abs(reconstructed - archived))))

        test = index["test"]
        test_aliases = [patient_aliases[value] for value in test]
        test_labels = labels[test]
        graph = graph_all[test]
        rad = scores_for(base, seed, "exploratory_test", "RAD", test_aliases)
        directional = scores_for(
            i3, seed, "exploratory_test", "I3-direction-original", test_aliases
        )
        final = (graph + rad + directional) / 3.0
        scores_by_group = {
            "RELIABLE-NEGATIVE-GCN-MIL": graph,
            "PYRADIOMICS": rad,
            "DIRECTIONAL-WAVELET-GLRLM-I3": directional,
            "FINAL-EQUAL-VOTE": final,
        }
        for group, scores in scores_by_group.items():
            metric_rows.append(metric_row(seed, group, test_labels, scores))
            for alias, label, score in zip(test_aliases, test_labels, scores):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "role": "exploratory_internal_test",
                        "group": group,
                        "patient_alias": alias,
                        "label": int(label),
                        "score": float(score),
                    }
                )
        for patient_index in test:
            alias = patient_aliases[patient_index]
            for position, image_id in enumerate(image_ids[patient_index]):
                instance_rows.append(
                    {
                        "seed": seed,
                        "role": "exploratory_internal_test",
                        "patient_alias": alias,
                        "image_id": image_id,
                        "attention": float(attention[patient_index, position]),
                        "instance_score": float(instance_scores[patient_index, position]),
                    }
                )

    maximum_error = max(reconstruction_errors)
    if maximum_error > 2e-6:
        raise RuntimeError(f"Archived development reconstruction failed: {maximum_error}")
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    instances = pd.DataFrame(instance_rows)
    summary = summarize(metrics)
    metrics.to_csv(OUT / "exploratory_test_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "exploratory_test_summary.csv", index=False)
    predictions.to_csv(OUT / "exploratory_test_predictions_deidentified.csv", index=False)
    instances.to_csv(OUT / "exploratory_test_instance_outputs_deidentified.csv", index=False)
    write_json(
        OUT / "evaluation_provenance.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_lock_sha256": sha256(OUT / "evaluation_lock.json"),
            "maximum_development_score_reconstruction_error": maximum_error,
            "test_prediction_rows": len(predictions),
            "test_instance_rows": len(instances),
            "test_draws": len(SEEDS),
            "patients_per_draw": 20,
            "post_test_model_changes": "prohibited",
            "interpretation": lock["status"],
        },
    )
    final = summary.set_index("group").loc["FINAL-EQUAL-VOTE"]
    lines = [
        "# Locked selected-pipeline exploratory internal test",
        "",
        "The model, preprocessing objects, fixed equal-vote rule, and 0.50 threshold were locked before these rows were read. No test-dependent refitting, weighting, calibration, or threshold selection was performed.",
        "",
        "The ten test draws contain 20 patients each and overlap in patients across draws. Their mean and standard deviation describe random-split transport within this previously explored 98-patient cohort; they are not independent-cohort confidence intervals.",
        "",
        "| Configuration | Mean AUROC | Mean AP | Mean trapezoidal AUPRC | Mean Brier | Both AUROC/AP >0.8 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.group} | {row.mean_auroc:.3f} ({row.std_auroc:.3f}) | "
            f"{row.mean_average_precision:.3f} ({row.std_average_precision:.3f}) | "
            f"{row.mean_auprc_trapezoid:.3f} ({row.std_auprc_trapezoid:.3f}) | "
            f"{row.mean_brier:.3f} ({row.std_brier:.3f}) | "
            f"{row.joint_auroc_ap_over_0_8_draws}/10 |"
        )
    lines.extend(
        [
            "",
            "## Fixed operating point",
            "",
            f"For FINAL-EQUAL-VOTE at 0.50, mean sensitivity was {final.mean_sensitivity_at_0_5:.3f}, mean specificity was {final.mean_specificity_at_0_5:.3f}, and mean accuracy was {final.mean_accuracy_at_0_5:.3f}.",
            "",
            "Development-score reconstruction from the saved model and preprocessing objects passed before output acceptance.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"maximum_development_reconstruction_error={maximum_error:.3g}", flush=True)


def self_test() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    row = metric_row(1, "x", labels, scores)
    assert row["auroc"] == 1.0 and row["average_precision"] == 1.0
    assert row["sensitivity_at_0_5"] == 1.0 and row["specificity_at_0_5"] == 1.0
    assert set(SEEDS) == set(R.SEEDS)
    print("Self-test passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("self-test", "prepare-lock", "evaluate", "all"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "prepare-lock":
        prepare_lock()
    elif args.stage == "evaluate":
        evaluate()
    else:
        self_test()
        prepare_lock()
        evaluate()


if __name__ == "__main__":
    main()
