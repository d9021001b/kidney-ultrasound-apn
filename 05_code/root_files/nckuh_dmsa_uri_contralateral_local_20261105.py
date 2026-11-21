"""Contralateral local URI-CADS asymmetry for kidney-level NCKUH APN."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_uricads_official_frozen_20261105 as uri
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_uri_contralateral_local_20261105"
REFERENCE = uri.OUT / "development_predictions.csv"
SEEDS = (20261102, 20261103, 20261104)
CURRENT_GROUP = "current_tile_usfm_reference"
URI_GROUP = "official_uri_embedding_folds_quarter_plus_current"
PLANES = (("LL", "RL"), ("LT", "RT"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference as load

    patients, kidneys, images = load()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    if (len(patients), len(kidneys), len(images)) != (210, 414, 820):
        raise ValueError("Unexpected cohort")
    return patients, kidneys, images


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve completed or partial experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    _, _, images = load_reference()
    images[["image_id", "patient_alias", "kidney_alias", "side", "view"]].to_csv(
        OUT / "encoding_manifest.csv", index=False
    )
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Within-patient contralateral token matching can expose focal unilateral cortical abnormalities that global kidney pooling suppresses",
        "clinical_motivation": "APN-compatible B-mode findings may be focal and either hyper- or hypoechoic; the unaffected contralateral kidney provides a personalized appearance reference",
        "method_inspiration": [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7566852/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10247636/",
        ],
        "encoder": "Five official URI-CADS ResNet-50 folds, frozen",
        "input": "Registered contour-masked tight 800x800 RGB kidney canvas with public URI-CADS normalization",
        "local_map": "Final 25x25x2048 convolution map; tokens retained where downsampled kidney-mask coverage is at least 0.25",
        "matching": "Same-patient same-plane left/right cosine nearest-neighbor distances; no spatial registration or assumed anatomical correspondence",
        "compact_features": "Per fold and image: global, distance-weighted, and top-decile URI-CADS pathology scores; token-score P90/max; match-distance moments/quantiles; mask coverage and opposite-view availability",
        "state_models": "Global all-head control; local all-head control; global any/bilateral plus local laterality; global any/bilateral plus concatenated global/local laterality",
        "selection": "All head PCA/LR settings unchanged and selected only inside each 126-patient training role",
        "fusion": "Candidate alone and fixed 20% or 25% contribution to the unchanged current chain; unchanged official URI-CADS fusion retained as reference",
        "cohort": "210 patients, 414 kidneys, 820 images; three fixed 126/42/42 assignments",
        "exploratory_test": "Unopened",
        "inputs": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                Path(__file__),
                base.OUT / "patient_assignments.csv",
                REFERENCE,
                *uri.MODEL_PATHS,
            )
        },
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "manifest_sha256": digest(OUT / "encoding_manifest.csv"),
    })
    print(f"Prepared URI-CADS contralateral local experiment: {OUT}", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Protocol changed after registration")
    if digest(OUT / "encoding_manifest.csv") != lock["manifest_sha256"]:
        raise ValueError("Manifest changed after registration")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def side_features(
    tokens: np.ndarray,
    opposite: np.ndarray | None,
    classifier_weight: np.ndarray,
    classifier_bias: np.ndarray,
    coverage: float,
) -> np.ndarray:
    """Summarize focal unmatched tokens without diagnosis labels."""
    logits = tokens @ classifier_weight.T + classifier_bias
    token_probability = sigmoid(logits)
    global_probability = sigmoid(logits.mean(0))
    if opposite is None or len(opposite) == 0:
        distance = np.zeros(len(tokens), np.float32)
        opposite_present = 0.0
    else:
        left = tokens / np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-8)
        right = opposite / np.maximum(np.linalg.norm(opposite, axis=1, keepdims=True), 1e-8)
        distance = 1.0 - np.max(left @ right.T, axis=1)
        distance = np.clip(distance, 0.0, 2.0).astype(np.float32)
        opposite_present = 1.0
    weights = distance + 1e-4
    weighted_probability = sigmoid((logits * weights[:, None]).sum(0) / weights.sum())
    count = max(1, int(np.ceil(0.10 * len(tokens))))
    top = np.argpartition(distance, -count)[-count:]
    top_probability = sigmoid(logits[top].mean(0))
    distance_summary = np.asarray([
        distance.mean(), distance.std(), *np.quantile(distance, (0.75, 0.90, 0.95)), distance.max()
    ], np.float32)
    return np.concatenate([
        global_probability,
        weighted_probability,
        top_probability,
        np.quantile(token_probability, 0.90, axis=0),
        token_probability.max(0),
        distance_summary,
        np.asarray([coverage, opposite_present], np.float32),
    ]).astype(np.float32)


def paired_local_features(
    maps: np.ndarray,
    coverages: np.ndarray,
    images: pd.DataFrame,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    if maps.shape != (820, 2048, 25, 25) or coverages.shape != (820, 25, 25):
        raise ValueError("Unexpected URI-CADS map geometry")
    result = np.zeros((len(images), 43), np.float32)
    for _, group in images.groupby("patient_alias", sort=False):
        view_lookup = {view: int(index) for index, view in zip(group.index, group.view)}
        for left_view, right_view in PLANES:
            left_index = view_lookup.get(left_view)
            right_index = view_lookup.get(right_view)
            for own_index, opposite_index in ((left_index, right_index), (right_index, left_index)):
                if own_index is None:
                    continue
                own_mask = coverages[own_index] >= 0.25
                if not own_mask.any():
                    own_mask.flat[np.argmax(coverages[own_index])] = True
                own = maps[own_index].transpose(1, 2, 0)[own_mask].astype(np.float32)
                opposite = None
                if opposite_index is not None:
                    opposite_mask = coverages[opposite_index] >= 0.25
                    if not opposite_mask.any():
                        opposite_mask.flat[np.argmax(coverages[opposite_index])] = True
                    opposite = maps[opposite_index].transpose(1, 2, 0)[opposite_mask].astype(np.float32)
                result[own_index] = side_features(
                    own, opposite, weight, bias, float(own_mask.mean())
                )
    if not np.isfinite(result).all() or (result == 0).all(axis=1).any():
        raise ValueError("Invalid contralateral local features")
    return result


def encode() -> None:
    require_lock()
    output = OUT / "uri_contralateral_local_features.npy"
    if output.exists():
        raise FileExistsError(f"Preserve existing local features: {output}")
    _, _, images = load_reference()
    manifest = pd.read_csv(OUT / "encoding_manifest.csv")
    if manifest.image_id.tolist() != images.image_id.tolist():
        raise ValueError("Manifest order mismatch")
    mean, std = uri.public_normalization()
    mean_tensor = torch.from_numpy(mean).cuda().view(1, 3, 1, 1)
    std_tensor = torch.from_numpy(std).cuda().view(1, 3, 1, 1)
    fold_features, fold_audit = [], []
    coverage_reference = None
    for fold, checkpoint in enumerate(uri.MODEL_PATHS, start=1):
        model = uri.OfficialGlobalEncoder(checkpoint).cuda().eval().requires_grad_(False)
        temporary = OUT / f"fold{fold}_feature_maps.tmp.npy"
        maps = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(820, 2048, 25, 25)
        )
        coverages = np.zeros((820, 25, 25), np.float32)
        with torch.inference_mode():
            for start in range(0, len(images), 8):
                canvases, batch_coverage = [], []
                for identifier in images.image_id.iloc[start:start + 8]:
                    bgr = cv2.imread(str(base.OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
                    mask_image = cv2.imread(str(base.OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
                    if bgr is None or mask_image is None:
                        raise FileNotFoundError(identifier)
                    mask = mask_image > 0
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    canvas, canvas_mask = base.tight_canvas(rgb * mask[..., None], mask, size=800)
                    canvases.append(canvas)
                    batch_coverage.append(
                        cv2.resize(canvas_mask.astype(np.float32), (25, 25), interpolation=cv2.INTER_AREA)
                    )
                tensor = torch.from_numpy(
                    np.stack(canvases).transpose(0, 3, 1, 2).copy()
                ).cuda().float() / 255.0
                feature_map = model.features((tensor - mean_tensor) / std_tensor)
                maps[start:start + len(feature_map)] = feature_map.cpu().numpy().astype(np.float16)
                coverages[start:start + len(feature_map)] = np.stack(batch_coverage)
                if start % 80 == 0:
                    print(f"URI local fold {fold}/5 encoded {min(start + 8, len(images))}/820", flush=True)
        maps.flush()
        if coverage_reference is None:
            coverage_reference = coverages.copy()
        else:
            np.testing.assert_allclose(coverage_reference, coverages, atol=0, rtol=0)
        weight = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
        bias = model.classifier.bias.detach().cpu().numpy().astype(np.float32)
        current = paired_local_features(maps, coverages, images, weight, bias)
        fold_features.append(current)
        fold_audit.append({
            "fold": fold,
            "checkpoint": checkpoint.name,
            "sha256": digest(checkpoint),
            "feature_min": float(current.min()),
            "feature_max": float(current.max()),
        })
        del maps, model
        temporary.unlink()
        torch.cuda.empty_cache()
        print(f"URI local fold {fold}/5 paired", flush=True)
    values = np.concatenate(fold_features, axis=1).astype(np.float32)
    if values.shape != (820, 215) or not np.isfinite(values).all():
        raise ValueError("Unexpected URI local feature matrix")
    np.save(output, values)
    names = []
    stem = (
        [f"global_class_{i}" for i in range(7)]
        + [f"distance_weighted_class_{i}" for i in range(7)]
        + [f"top_decile_class_{i}" for i in range(7)]
        + [f"token_p90_class_{i}" for i in range(7)]
        + [f"token_max_class_{i}" for i in range(7)]
        + ["match_mean", "match_std", "match_p75", "match_p90", "match_p95", "match_max"]
        + ["mask_coverage", "opposite_present"]
    )
    for fold in range(1, 6):
        names.extend([f"fold{fold}_{name}" for name in stem])
    pd.DataFrame({"column": np.arange(len(names)), "feature": names}).to_csv(
        OUT / "feature_dictionary.csv", index=False
    )
    write_json(OUT / "encoder_audit.json", {
        "images": 820,
        "folds": fold_audit,
        "feature_shape": list(values.shape),
        "labels_used_in_encoding": [],
        "feature_sha256": digest(output),
        "temporary_maps_retained": False,
    })


def probability_from_heads(
    symmetric: np.ndarray,
    antisymmetric: np.ndarray,
    states: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    any_label = (states != hierarchy.STATE_CODE["N"]).astype(int)
    any_model = hierarchy.fit_head(symmetric, any_label, train, seed)
    affected = train[states[train] != hierarchy.STATE_CODE["N"]]
    bilateral_label = (states == hierarchy.STATE_CODE["B"]).astype(int)
    bilateral_model = hierarchy.fit_head(symmetric, bilateral_label, affected, seed + 101)
    unilateral = train[np.isin(states[train], [hierarchy.STATE_CODE["L"], hierarchy.STATE_CODE["R"]])]
    left_label = (states == hierarchy.STATE_CODE["L"]).astype(int)
    laterality_model = hierarchy.fit_head(antisymmetric, left_label, unilateral, seed + 202)
    p_any = any_model.predict_proba(symmetric)[:, 1]
    p_bilateral = bilateral_model.predict_proba(symmetric)[:, 1]
    p_left = laterality_model.predict_proba(antisymmetric)[:, 1]
    probability = np.c_[
        1 - p_any,
        p_any * (1 - p_bilateral) * p_left,
        p_any * (1 - p_bilateral) * (1 - p_left),
        p_any * p_bilateral,
    ]
    probability /= probability.sum(1, keepdims=True)
    return probability, {
        "any": any_model,
        "bilateral_given_any": bilateral_model,
        "left_given_unilateral": laterality_model,
    }


def hybrid_probability(
    global_views: tuple[np.ndarray, np.ndarray],
    local_antisymmetric: np.ndarray,
    states: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    global_symmetric, global_antisymmetric = global_views
    any_label = (states != hierarchy.STATE_CODE["N"]).astype(int)
    any_model = hierarchy.fit_head(global_symmetric, any_label, train, seed)
    affected = train[states[train] != hierarchy.STATE_CODE["N"]]
    bilateral_label = (states == hierarchy.STATE_CODE["B"]).astype(int)
    bilateral_model = hierarchy.fit_head(global_symmetric, bilateral_label, affected, seed + 101)
    unilateral = train[np.isin(states[train], [hierarchy.STATE_CODE["L"], hierarchy.STATE_CODE["R"]])]
    left_label = (states == hierarchy.STATE_CODE["L"]).astype(int)
    laterality_model = hierarchy.fit_head(local_antisymmetric, left_label, unilateral, seed + 202)
    p_any = any_model.predict_proba(global_symmetric)[:, 1]
    p_bilateral = bilateral_model.predict_proba(global_symmetric)[:, 1]
    p_left = laterality_model.predict_proba(local_antisymmetric)[:, 1]
    probability = np.c_[
        1 - p_any,
        p_any * (1 - p_bilateral) * p_left,
        p_any * (1 - p_bilateral) * (1 - p_left),
        p_any * p_bilateral,
    ]
    probability /= probability.sum(1, keepdims=True)
    return probability, {
        "any": any_model,
        "bilateral_given_any": bilateral_model,
        "left_given_unilateral": laterality_model,
    }


def evaluate() -> None:
    require_lock()
    if (OUT / "development_metrics.csv").exists():
        raise FileExistsError("Preserve completed local evaluation")
    audit = json.loads((OUT / "encoder_audit.json").read_text(encoding="utf-8"))
    feature_path = OUT / "uri_contralateral_local_features.npy"
    if audit["feature_sha256"] != digest(feature_path):
        raise ValueError("Local feature hash mismatch")
    local = np.load(feature_path)
    global_embedding = np.load(uri.OUT / "official_fold_embeddings.npy").reshape(820, -1)
    patients, kidneys, images = load_reference()
    if pd.read_csv(OUT / "encoding_manifest.csv").image_id.tolist() != images.image_id.tolist():
        raise ValueError("Feature order mismatch")
    global_views = hierarchy.structured_views(
        base.patient_pair_features(global_embedding, images, patients)
    )
    local_views = hierarchy.structured_views(
        base.patient_pair_features(local, images, patients)
    )
    combined_antisymmetric = np.c_[global_views[1], local_views[1]]
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    references = pd.read_csv(REFERENCE)
    references = references[references.group.isin([CURRENT_GROUP, URI_GROUP])].set_index(
        ["seed", "group", "kidney_alias"], verify_integrity=True
    )

    metrics, predictions, selections = [], [], []
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        train = np.flatnonzero(patients.patient_alias.map(mapping).eq("train"))
        development = np.flatnonzero(kidneys.patient_alias.map(mapping).eq("development"))
        target = kidneys.iloc[development]
        current = references.loc[
            [(seed, CURRENT_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        fixed_uri = references.loc[
            [(seed, URI_GROUP, alias) for alias in target.kidney_alias]
        ].score.to_numpy(float)
        probability_sets = {}
        probability_sets["uri_global_state"], global_models = probability_from_heads(
            global_views[0], global_views[1], states, train, seed
        )
        probability_sets["uri_local_state"], local_models = probability_from_heads(
            local_views[0], local_views[1], states, train, seed
        )
        probability_sets["uri_global_any_bilateral_local_laterality"], hybrid_models = hybrid_probability(
            global_views, local_views[1], states, train, seed
        )
        probability_sets["uri_global_any_bilateral_combined_laterality"], combined_models = hybrid_probability(
            global_views, combined_antisymmetric, states, train, seed
        )
        all_models = {
            "global": global_models,
            "local": local_models,
            "hybrid_local": hybrid_models,
            "hybrid_combined": combined_models,
        }
        scores = {
            "current_tile_usfm_reference": current,
            "official_uri_fixed_quarter_reference": fixed_uri,
        }
        for name, probability in probability_sets.items():
            candidate = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            scores[name] = candidate
            scores[f"{name}_quarter_plus_current"] = 0.25 * candidate + 0.75 * current
            scores[f"{name}_fifth_plus_current"] = 0.20 * candidate + 0.80 * current
        joblib.dump(all_models, OUT / "models" / f"{seed}_uri_contralateral_local.joblib", compress=3)
        for source, models in all_models.items():
            for head, model in models.items():
                selections.append({
                    "seed": seed,
                    "source": source,
                    "head": head,
                    "inner_cv_auroc": float(model.best_score_),
                    **model.best_params_,
                })
        for group, score in scores.items():
            metrics.append({
                "seed": seed,
                "role": "development",
                "group": group,
                "patients": 42,
                "kidneys": len(development),
                **hierarchy.metric(labels[development], score),
            })
            for local_index, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(score[local_index]),
                })
        print(f"Completed URI contralateral local draw {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, SEEDS, "official_uri_fixed_quarter_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(OUT / "decision.json", {
        "groups_above_both_development_targets": achieved,
        "promotion": bool(achieved),
        "reference": "official_uri_fixed_quarter_reference",
        "exploratory_test_rows_scored": 0,
    })
    write_json(OUT / "integrity_audit.json", {
        "patients": len(patients),
        "kidneys": len(kidneys),
        "images": len(images),
        "prediction_rows": len(prediction_frame),
        "development_groups": prediction_frame.group.nunique(),
        "finite_scores": bool(np.isfinite(prediction_frame.score).all()),
        "unit_interval_scores": bool(prediction_frame.score.between(0, 1).all()),
        "exploratory_test_rows_scored": 0,
    })
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "encode", "evaluate"))
    action = parser.parse_args().action
    globals()[action]()


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
