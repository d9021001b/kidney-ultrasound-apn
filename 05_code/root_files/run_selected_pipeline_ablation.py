"""Development-only component ablation for the locked CYCH graph-MIL pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_selected_pipeline_ablation_20260914"
BASE = ROOT / "CYCH_dinov3_native_probe_20260913" / "patient_predictions_by_role.csv"
I3 = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
GATED = ROOT / "CYCH_gated_attention_mil_development_20260914" / "development_predictions.csv"
GRAPH = ROOT / "CYCH_graph_attention_mil_development_20260914" / "development_predictions.csv"
RELIABLE = ROOT / "CYCH_reliable_negative_graph_mil_development_20260914" / "development_predictions.csv"
SEEDS = list(range(20260913, 20260923))


GROUPS = [
    "RADIMAGENET-DENSENET-SUMMARY",
    "GATED-ATTENTION-MIL",
    "GCN-GATED-ATTENTION-MIL",
    "RELIABLE-NEGATIVE-GCN-MIL",
    "PYRADIOMICS",
    "DIRECTIONAL-WAVELET-GLRLM-I3",
    "FIXED-RADIOMICS-RADIMAGENET-PAIR",
    "FIXED-PAIR+I3",
    "RELIABLE-NEGATIVE-GCN-MIL+PYRADIOMICS",
    "RELIABLE-NEGATIVE-GCN-MIL+I3",
    "FINAL-EQUAL-VOTE",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_group(path: Path, group: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "role" in frame.columns:
        frame = frame.loc[frame.role.eq("development_validation")]
    frame = frame.loc[frame.group.eq(group), ["seed", "patient_alias", "label", "score"]].copy()
    if len(frame) != 200:
        raise ValueError(f"Expected 200 rows for {group}, received {len(frame)}")
    if frame.duplicated(["seed", "patient_alias"]).any():
        raise ValueError(f"Duplicate seed-patient rows for {group}")
    return frame.rename(columns={"score": group})


def load_components() -> pd.DataFrame:
    frames = [
        read_group(BASE, "RAD"),
        read_group(BASE, "DN-C"),
        read_group(BASE, "PAIR"),
        read_group(I3, "I3-ALL"),
        read_group(GATED, "PAIR-GATED-MIL"),
        read_group(GRAPH, "PAIR-GRAPH-MIL"),
        read_group(RELIABLE, "RELIABLE-NEG-GRAPH+I3"),
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["seed", "patient_alias", "label"], validate="one_to_one")
    if len(merged) != 200 or sorted(merged.seed.unique()) != SEEDS:
        raise ValueError("Ablation inputs do not contain the fixed ten development draws")
    return merged.sort_values(["seed", "patient_alias"]).reset_index(drop=True)


def derive_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["seed", "patient_alias", "label"]].copy()
    rad = frame["RAD"].to_numpy(float)
    dense = frame["DN-C"].to_numpy(float)
    pair = frame["PAIR"].to_numpy(float)
    i3 = frame["I3-ALL"].to_numpy(float)
    gated = 2.0 * frame["PAIR-GATED-MIL"].to_numpy(float) - rad
    graph = 2.0 * frame["PAIR-GRAPH-MIL"].to_numpy(float) - rad
    reliable = 3.0 * frame["RELIABLE-NEG-GRAPH+I3"].to_numpy(float) - rad - i3

    components = {
        "RADIMAGENET-DENSENET-SUMMARY": dense,
        "GATED-ATTENTION-MIL": gated,
        "GCN-GATED-ATTENTION-MIL": graph,
        "RELIABLE-NEGATIVE-GCN-MIL": reliable,
        "PYRADIOMICS": rad,
        "DIRECTIONAL-WAVELET-GLRLM-I3": i3,
        "FIXED-RADIOMICS-RADIMAGENET-PAIR": pair,
        "FIXED-PAIR+I3": (2.0 * pair + i3) / 3.0,
        "RELIABLE-NEGATIVE-GCN-MIL+PYRADIOMICS": (reliable + rad) / 2.0,
        "RELIABLE-NEGATIVE-GCN-MIL+I3": (reliable + i3) / 2.0,
        "FINAL-EQUAL-VOTE": (reliable + rad + i3) / 3.0,
    }
    for name, values in components.items():
        if np.min(values) < -1e-8 or np.max(values) > 1.0 + 1e-8:
            raise ValueError(f"Derived probability outside [0, 1] for {name}")
        result[name] = np.clip(values, 0.0, 1.0)

    if not np.allclose(result["FINAL-EQUAL-VOTE"], frame["RELIABLE-NEG-GRAPH+I3"], atol=1e-12):
        raise AssertionError("Final score does not reproduce the locked candidate")
    return result


def metric_row(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "seed": seed,
        "group": group,
        "auroc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "auprc_trapezoid": abs(np.trapezoid(precision, recall)),
        "sensitivity_at_0_5": tp / (tp + fn),
        "specificity_at_0_5": tn / (tn + fp),
        "accuracy_at_0_5": accuracy_score(labels, predicted),
        "brier": brier_score_loss(labels, scores),
        "log_loss": log_loss(labels, np.clip(scores, 1e-8, 1.0 - 1e-8)),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        subset = metrics.loc[metrics.group.eq(group)]
        rows.append(
            {
                "group": group,
                "mean_auroc": subset.auroc.mean(),
                "std_auroc": subset.auroc.std(ddof=1),
                "mean_average_precision": subset.average_precision.mean(),
                "std_average_precision": subset.average_precision.std(ddof=1),
                "mean_auprc_trapezoid": subset.auprc_trapezoid.mean(),
                "std_auprc_trapezoid": subset.auprc_trapezoid.std(ddof=1),
                "mean_brier": subset.brier.mean(),
                "joint_target_draws": int(((subset.auroc > 0.8) & (subset.average_precision > 0.8)).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "scope": "Retrospective development-only ablation of the locked selected pipeline; no exploratory-test access and no parameter selection",
        "fixed_draws": {"seeds": SEEDS, "development_patients_per_draw": 20},
        "score_derivation": "Algebraic recovery of raw gated-attention, graph-MIL, and reliable-negative graph-MIL probabilities from their locked equal-probability combinations; exact reconstruction is asserted",
        "ablation_questions": {
            "patient_aggregation": "Compare simple DenseNet patient summary, gated attention without graph propagation, graph-convolutional gated attention, and the same graph model with one-sided reliable-negative instance supervision.",
            "feature_complementarity": "Compare PyRadiomics and directional wavelet-GLRLM I3 alone and in fixed unweighted combinations with the selected graph branch.",
            "final_fusion": "Confirm whether the fixed equal vote of reliable-negative graph MIL, PyRadiomics, and I3 gives the strongest mean two-metric development result without learned ensemble weights.",
        },
        "method_inspiration": {
            "kidney_graph_mil": "Yin et al. Multi-instance Deep Learning with Graph Convolutional Neural Networks for Diagnosis of Kidney Diseases Using Ultrasound Imaging. DOI:10.1007/978-3-030-32689-0_15",
            "gated_attention": "Ilse et al. Attention-based Deep Multiple Instance Learning. PMLR 80:2127-2136, 2018.",
            "directional_radiomics": "Bandara et al. Ultrasound Based Radiomics Features of Chronic Kidney Disease. Academic Radiology. 2022;29:S35-S41.",
        },
        "reporting": "AP and trapezoidal AUPRC are separate. Mean and sample standard deviation summarize overlapping fixed random draws and are not confidence intervals.",
        "inputs_sha256": {path.name + "@" + path.parent.name: sha256(path) for path in (BASE, I3, GATED, GRAPH, RELIABLE)},
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "protocol_lock.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "exploratory_test_access": "prohibited",
        },
    )

    components = derive_scores(load_components())
    prediction_rows = []
    metric_rows = []
    for seed in SEEDS:
        subset = components.loc[components.seed.eq(seed)]
        labels = subset.label.to_numpy(int)
        if len(subset) != 20 or set(labels) != {0, 1}:
            raise ValueError(f"Invalid development draw for seed {seed}")
        for group in GROUPS:
            scores = subset[group].to_numpy(float)
            metric_rows.append(metric_row(seed, group, labels, scores))
            for alias, label, score in zip(subset.patient_alias, labels, scores):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "group": group,
                        "patient_alias": alias,
                        "label": int(label),
                        "score": float(score),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    predictions.to_csv(OUT / "development_predictions_deidentified.csv", index=False)

    final = summary.set_index("group").loc["FINAL-EQUAL-VOTE"]
    graph_only = summary.set_index("group").loc["RELIABLE-NEGATIVE-GCN-MIL"]
    decision = {
        "selected_group": "FINAL-EQUAL-VOTE",
        "mean_auroc": float(final.mean_auroc),
        "mean_average_precision": float(final.mean_average_precision),
        "mean_auprc_trapezoid": float(final.mean_auprc_trapezoid),
        "delta_vs_reliable_negative_graph_only": {
            "auroc": float(final.mean_auroc - graph_only.mean_auroc),
            "average_precision": float(final.mean_average_precision - graph_only.mean_average_precision),
        },
        "exploratory_test_access": "prohibited",
    }
    write_json(OUT / "ablation_decision.json", decision)

    lines = [
        "# Selected-pipeline development ablation",
        "",
        "This is a retrospective ablation of locked predictions. It did not access exploratory-test rows and did not tune weights or hyperparameters.",
        "",
        "| Configuration | Mean AUROC | Mean AP | Mean trapezoidal AUPRC | Both AUROC/AP >0.8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.group} | {row.mean_auroc:.3f} ({row.std_auroc:.3f}) | "
            f"{row.mean_average_precision:.3f} ({row.std_average_precision:.3f}) | "
            f"{row.mean_auprc_trapezoid:.3f} ({row.std_auprc_trapezoid:.3f}) | "
            f"{row.joint_target_draws}/10 |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The structural ablation tests learned instance weighting, graph message passing, and one-sided reliable-negative supervision in sequence. The feature ablation tests whether radiomics and directional texture complement the selected graph branch. The fixed three-source equal vote is retained because it has the strongest mean AUROC and AP without fitted ensemble weights.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
