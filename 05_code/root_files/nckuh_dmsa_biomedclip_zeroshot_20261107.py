"""Literature-anchored BioMedCLIP zero-shot semantics for NCKUH renal APN."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

import nckuh_dmsa_biomedclip_frozen_20261106 as biomed
import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_biomedclip_zeroshot_20261107"
WORK = OUT.with_name(OUT.name + "_working")
TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract"
    / "snapshots/d673b8835373c6fa116d6d8006b33d48734e305d"
)
REFERENCE_GROUP = "official_uri_embedding_folds_quarter_plus_current"
POSITIVE_PROMPTS = (
    "renal ultrasound showing acute pyelonephritis",
    "renal ultrasound with focal cortical hypoechogenicity",
    "renal ultrasound with focal cortical hyperechogenicity",
    "renal ultrasound with renal enlargement and loss of corticomedullary differentiation",
    "renal ultrasound with a focal wedge-shaped parenchymal abnormality",
    "renal ultrasound with decreased renal parenchymal perfusion",
)
NEGATIVE_PROMPTS = (
    "normal renal ultrasound",
    "normal kidney with preserved corticomedullary differentiation",
    "normal renal parenchymal echogenicity without a focal lesion",
    "normal kidney without enlargement or parenchymal abnormality",
)
REPRESENTATIONS = ("global", "patch_mean")
POOLING = ("mean", "max")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def zero_shot_probability(
    image_embedding: np.ndarray,
    positive_text: np.ndarray,
    negative_text: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    image = functional.normalize(torch.as_tensor(image_embedding, dtype=torch.float32), dim=1)
    positive = functional.normalize(torch.as_tensor(positive_text, dtype=torch.float32), dim=1)
    negative = functional.normalize(torch.as_tensor(negative_text, dtype=torch.float32), dim=1)
    positive_logit = torch.logsumexp(logit_scale * (image @ positive.T), dim=1) - np.log(
        len(positive_text)
    )
    negative_logit = torch.logsumexp(logit_scale * (image @ negative.T), dim=1) - np.log(
        len(negative_text)
    )
    return torch.sigmoid(positive_logit - negative_logit).numpy()


def pool_kidneys(
    image_score: np.ndarray,
    images: pd.DataFrame,
    kidneys: pd.DataFrame,
    method: str,
) -> np.ndarray:
    if method not in POOLING:
        raise ValueError(f"Unknown pooling method: {method}")
    score = pd.Series(np.asarray(image_score, dtype=float), index=images.index)
    grouped = score.groupby(images.kidney_alias)
    lookup = (grouped.mean() if method == "mean" else grouped.max()).to_dict()
    if not set(kidneys.kidney_alias).issubset(lookup):
        raise ValueError("Every scored kidney must have a zero-shot image score")
    return np.asarray([lookup[alias] for alias in kidneys.kidney_alias], dtype=float)


def encode_text() -> tuple[np.ndarray, np.ndarray, float]:
    if not TOKENIZER.is_dir():
        raise FileNotFoundError(f"Missing local BioMedBERT tokenizer: {TOKENIZER}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import open_clip
    from transformers import AutoTokenizer

    model, _ = open_clip.create_model_from_pretrained(
        biomed.MODEL_ID, cache_dir=str(biomed.CACHE_DIR)
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    prompts = list(POSITIVE_PROMPTS + NEGATIVE_PROMPTS)
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )["input_ids"]
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        text = functional.normalize(model.encode_text(tokens), dim=1).cpu().numpy()
        logit_scale = float(model.logit_scale.exp().clamp(max=100).cpu())
    positive = text[: len(POSITIVE_PROMPTS)]
    negative = text[len(POSITIVE_PROMPTS) :]
    if positive.shape != (len(POSITIVE_PROMPTS), 512) or negative.shape != (
        len(NEGATIVE_PROMPTS),
        512,
    ):
        raise ValueError("Unexpected text embedding geometry")
    return positive, negative, logit_scale


def protocol() -> dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A fixed literature-derived APN semantic direction may add information not recovered by a supervised whole-kidney classifier",
        "references": [
            "https://pubmed.ncbi.nlm.nih.gov/1651644/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7475307/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC13086778/",
        ],
        "model": "Official Microsoft BioMedCLIP image and text towers; local-only inference",
        "positive_prompts": list(POSITIVE_PROMPTS),
        "negative_prompts": list(NEGATIVE_PROMPTS),
        "representations": list(REPRESENTATIONS),
        "pooling": list(POOLING),
        "primary_candidate": "patch_mean_max_20pct_plus_uri",
        "fusion": "Fixed 20% zero-shot probability plus 80% unchanged official URI-CADS reference",
        "selection": "No labels, development values, identifiers, DMSA codes, or prompt search enter text/image encoding or aggregation",
        "evaluation": "Same three fixed 42-patient development roles; kidney scoring; exploratory-test roles remain unopened",
        "target": "Development mean AUROC>0.8 and AP>0.8 followed by locked confirmation",
    }


def run() -> None:
    if OUT.exists() or WORK.exists():
        raise FileExistsError("Preserve existing BioMedCLIP zero-shot experiment")
    if digest(biomed.WEIGHT) != biomed.WEIGHT_SHA256:
        raise ValueError("Official BioMedCLIP weight hash mismatch")
    WORK.mkdir(parents=True)
    write_json(WORK / "protocol.json", protocol())
    patients, kidneys, images = cohort.load_reference()
    global_embedding = np.load(biomed.OUT / "biomedclip_global.npy")
    patch_summary = np.load(biomed.OUT / "biomedclip_teacher_normalized.npy")
    if global_embedding.shape != (820, 512) or patch_summary.shape != (820, 2048):
        raise ValueError("Unexpected BioMedCLIP visual cache geometry")
    positive, negative, logit_scale = encode_text()
    np.save(WORK / "positive_text_embeddings.npy", positive)
    np.save(WORK / "negative_text_embeddings.npy", negative)
    write_json(
        WORK / "text_encoder_audit.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "official_weight_sha256": digest(biomed.WEIGHT),
            "tokenizer_revision": TOKENIZER.name,
            "offline": True,
            "logit_scale": logit_scale,
            "positive_shape": list(positive.shape),
            "negative_shape": list(negative.shape),
            "labels_used": [],
        },
    )
    image_probabilities = {
        "global": zero_shot_probability(global_embedding, positive, negative, logit_scale),
        "patch_mean": zero_shot_probability(
            patch_summary[:, :512], positive, negative, logit_scale
        ),
    }
    kidney_probabilities = {
        f"{representation}_{method}": pool_kidneys(
            image_probabilities[representation], images, kidneys, method
        )
        for representation in REPRESENTATIONS
        for method in POOLING
    }
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    reference = pd.read_csv(uri.OUT / "development_predictions.csv")
    reference = reference[reference.group.eq(REFERENCE_GROUP)].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    labels = kidneys.kidney_APN.to_numpy(int)
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        kidney_roles = kidneys.patient_alias.map(mapping)
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        fixed_uri = reference.loc[
            [(seed, alias) for alias in target.kidney_alias], "score"
        ].to_numpy(float)
        draw_scores: dict[str, np.ndarray] = {"official_uri_fixed_reference": fixed_uri}
        for name, all_scores in kidney_probabilities.items():
            current = all_scores[development]
            draw_scores[f"biomedclip_zeroshot_{name}"] = current
            draw_scores[f"biomedclip_zeroshot_{name}_20pct_plus_uri"] = (
                0.20 * current + 0.80 * fixed_uri
            )
        for group, score in draw_scores.items():
            result = base.metric(labels[development], score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **result,
                }
            )
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local]),
                    }
                )
        print(f"BioMedCLIP zero-shot draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(WORK / "development_metrics.csv", index=False)
    prediction_frame.to_csv(WORK / "development_predictions.csv", index=False)
    summary.to_csv(WORK / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "official_uri_fixed_reference"
    )
    ci.to_csv(WORK / "development_ci.csv", index=False)
    delta.to_csv(WORK / "paired_delta_ci.csv", index=False)
    np.save(WORK / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        WORK / "decision.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "groups_above_both_development_targets": achieved,
            "promotion": bool(achieved),
            "exploratory_test_rows_scored": 0,
            "primary_candidate": "biomedclip_zeroshot_patch_mean_max_20pct_plus_uri",
            "inputs": {
                str((biomed.OUT / "biomedclip_global.npy").relative_to(ROOT)): digest(
                    biomed.OUT / "biomedclip_global.npy"
                ),
                str(
                    (biomed.OUT / "biomedclip_teacher_normalized.npy").relative_to(ROOT)
                ): digest(biomed.OUT / "biomedclip_teacher_normalized.npy"),
            },
        },
    )
    primary = summary.loc["biomedclip_zeroshot_patch_mean_max_20pct_plus_uri"]
    reference_row = summary.loc["official_uri_fixed_reference"]
    readme = f"""# NCKUH literature-anchored BioMedCLIP zero-shot experiment

The official BioMedCLIP text tower encoded a fixed prompt ensemble describing published
APN ultrasound findings and normal renal controls. All inference was local and offline.
No patient label, identifier, development result, or DMSA code entered prompt creation,
image encoding, or kidney aggregation. Global and mask-covered patch-mean image
embeddings used fixed mean or maximum aggregation across each kidney's views.

The preregistered primary patch-mean/max 20% fusion produced mean AUROC
{primary.auroc_mean:.4f} and AP {primary.ap_mean:.4f}. The unchanged URI-CADS reference
was AUROC {reference_row.auroc_mean:.4f} and AP {reference_row.ap_mean:.4f}. Patient-
cluster intervals and paired changes are in `development_ci.csv` and
`paired_delta_ci.csv`. No exploratory-test row was scored.
"""
    (WORK / "README.md").write_text(readme, encoding="utf-8")
    WORK.rename(OUT)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    run()
