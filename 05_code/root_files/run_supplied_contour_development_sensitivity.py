"""Development-only sensitivity analysis using supplied kidney contours for every image.

This analysis isolates dependence on the pre-existing single-class segmentation
checkpoint. It preserves the cohort, patient assignments, feature definitions,
classifier heads, graph-MIL architecture, and equal-vote rule used by the selected
pipeline. Exploratory-test labels and outcomes are not evaluated or written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import renal_dinov3_native_probe as N
import renal_enhancement_validation as E
import renal_random_validation as R
import renal_stability_region_probe as S
import renal_voting_validation as V
from recrop_with_yolo import bbox_of, polygon_mask
from run_gated_attention_mil_development import build_bags, load_reference
from run_reliable_negative_graph_mil_development import train_one


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "CYCH_random_validation_20260913"
REFERENCE = ROOT / "CYCH_reliable_negative_graph_mil_development_20260914" / "development_predictions.csv"
MASK_POLICY = os.environ.get("CYCH_MASK_POLICY", "supplied")
if MASK_POLICY not in {"supplied", "intersection"}:
    raise ValueError("CYCH_MASK_POLICY must be 'supplied' or 'intersection'")
OUT = Path(
    os.environ.get(
        "CYCH_MASK_SENSITIVITY_DIR",
        str(
            ROOT
            / (
                "CYCH_supplied_contour_development_sensitivity_20260915"
                if MASK_POLICY == "supplied"
                else "CYCH_constrained_intersection_development_sensitivity_20260915"
            )
        ),
    )
)
GROUP_PREFIX = "SUPPLIED-CONTOUR" if MASK_POLICY == "supplied" else "CONSTRAINED-INTERSECTION"
POLICY_LABEL = "Supplied contours only" if MASK_POLICY == "supplied" else "Model-manual intersections"
SEEDS = tuple(R.SEEDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def group(name: str) -> str:
    return f"{GROUP_PREFIX}-{name}"


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))


def prepare() -> None:
    for name in (
        "native",
        "masks",
        "images/percentile",
        "features",
        "models",
        "qa",
        "source",
        "figures_600dpi",
    ):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name in ("patients.csv", "random_patient_assignments.csv"):
        shutil.copy2(BASE / name, OUT / name)

    original = pd.read_csv(BASE / "image_manifest.csv", dtype={"patient_id": str})
    records = []
    for row in original.itertuples():
        bgr = cv2.imread(str(row.clean_path))
        if bgr is None:
            raise FileNotFoundError(row.clean_path)
        h, w = bgr.shape[:2]
        supplied_mask = polygon_mask(ROOT / "cych_image_label" / Path(row.filename).with_suffix(".txt"), h, w)
        full_mask = supplied_mask
        mask_source = "supplied_contour_only"
        if MASK_POLICY == "intersection" and row.mask_source == "existing_yolo":
            predicted = (cv2.imread(str(BASE / "predicted_masks" / f"{row.image_id}.png"), 0) > 0).astype(np.uint8)
            if predicted.shape != supplied_mask.shape:
                raise ValueError(f"Incompatible predicted mask: {row.image_id}")
            candidate = (predicted & supplied_mask).astype(np.uint8)
            if int(candidate.sum()) >= 64:
                full_mask = candidate
                mask_source = "model_supplied_intersection"
            else:
                mask_source = "supplied_contour_intersection_fallback"
        elif MASK_POLICY == "intersection":
            mask_source = "supplied_contour_existing_fallback"
        if int(full_mask.sum()) < 64:
            raise ValueError(f"Unusable supplied contour: {row.image_id}")
        x0, y0, x1, y1 = bbox_of(full_mask)
        x0, y0, x1, y1 = max(0, x0 - 4), max(0, y0 - 4), min(w, x1 + 4), min(h, y1 + 4)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        mask = full_mask[y0:y1, x0:x1].astype(np.uint8)
        normalized = E.enhance(gray, mask.astype(bool), "percentile")
        paths = {
            "native": OUT / "native" / f"{row.image_id}.png",
            "mask": OUT / "masks" / f"{row.image_id}.png",
            "percentile": OUT / "images" / "percentile" / f"{row.image_id}.png",
        }
        assert cv2.imwrite(str(paths["native"]), gray)
        assert cv2.imwrite(str(paths["mask"]), mask * 255)
        assert cv2.imwrite(str(paths["percentile"]), normalized)
        assert not normalized[mask == 0].any()
        record = row._asdict()
        record.update(
            mask_source=mask_source,
            manual_iou=1.0,
            width=x1 - x0,
            height=y1 - y0,
            mask_pixels=int(mask.sum()),
            mask_sha256=sha256(paths["mask"]),
        )
        records.append(record)
    manifest = pd.DataFrame(records)
    assert len(manifest) == 349 and manifest.patient_id.nunique() == 98
    manifest.to_csv(OUT / "image_manifest.csv", index=False)

    patients = read("patients.csv")
    assignments = read("random_patient_assignments.csv")
    for seed in SEEDS:
        N.indices_for(patients, assignments, seed)
    patients.drop(columns=["patient_id"]).to_csv(OUT / "patients_deidentified.csv", index=False)
    assignments.drop(columns=["patient_id"]).to_csv(
        OUT / "random_patient_assignments_deidentified.csv", index=False
    )
    safe_manifest = manifest[
        ["image_id", "patient_alias", "label", "view", "mask_source", "mask_pixels", "width", "height", "mask_sha256"]
    ]
    safe_manifest.to_csv(OUT / "image_manifest_deidentified.csv", index=False)

    inherited = original.mask_source.value_counts().to_dict()
    protocol = {
        "scope": "Development-only mask-source sensitivity analysis; no exploratory-test outcome evaluation",
        "mask_policy": MASK_POLICY,
        "motivation": (
            "The archived single-class YOLOv12x-seg checkpoint has conflicting local task provenance. Replacing every candidate mask with the supplied kidney contour tests whether the selected classifier depends on that checkpoint."
            if MASK_POLICY == "supplied"
            else "Some accepted candidate masks extend beyond the supplied kidney contour. Intersecting every accepted candidate with that contour tests whether performance can be retained while guaranteeing that all analyzed pixels lie inside the supplied renal boundary."
        ),
        "design": "One-factor input ablation. Cohort, ten patient assignments, image labels, feature definitions, train-only fitting, graph-MIL architecture, and equal-vote rule remain fixed; only the renal contour policy changes.",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": list(SEEDS), "train": 58, "development": 20, "exploratory_test": "not evaluated"},
        "baseline_mask_sources": inherited,
        "sensitivity_mask_source": (
            "Existing supplied kidney polygon for all 349 images"
            if MASK_POLICY == "supplied"
            else "Intersection of accepted model candidate and supplied kidney polygon; supplied polygon retained for all previous fallbacks"
        ),
        "features": {
            "RAD": "ROI P1-P99 normalized PyRadiomics descriptors, with scale-dependent energy features removed; patient image mean and standard deviation",
            "DenseNet": "Frozen RadImageNet DenseNet-121 tight renal ROI image embeddings",
            "I3": "Original plus coif1 level-1 wavelet entropy, variance, and directional GLRLM descriptors; patient image mean and standard deviation",
        },
        "heads": "Fixed train-only heads from the selected pipeline: top-32 L2 logistic regression for RAD/DenseNet and full constant-filtered L2 logistic regression for I3",
        "graph_mil": "Training-image StandardScaler and PCA(32), two GCN layers, gated attention, class-balanced bag BCE plus negative-bag reliable-instance BCE, 300 deterministic epochs",
        "vote": "Unweighted arithmetic mean of RAD, graph-MIL, and I3 probabilities",
        "interpretation": "Paired per-draw differences and prediction rank correlation are descriptive sensitivity evidence. No model selection, noninferiority claim, or independent-validation claim is made.",
        "reference_predictions_sha256": sha256(REFERENCE),
        "cohort_hashes": {
            name: sha256(OUT / name)
            for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")
        },
    }
    write_json(OUT / "protocol.json", protocol)
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    write_json(
        OUT / "protocol_locked.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "source_sha256": sha256(Path(__file__)),
        },
    )
    print(f"Prepared 349 {MASK_POLICY} kidney ROIs; protocol locked", flush=True)


def _extract_rad(item: tuple[str, str, str]) -> dict:
    image_id, gray_path, mask_path = item
    return R.radiomics_one({"image_id": image_id, "gray_path": gray_path, "mask_path": mask_path})


def _extract_i3(item: tuple[str, str, str]) -> dict:
    image_id, raw_path, mask_path = item
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    if raw is None or mask is None or raw.shape != mask.shape:
        raise ValueError(f"Missing or incompatible source image/mask: {image_id}")
    normalized = S.percentile_normalized(raw, mask)
    selected, erosion = S.support_mask(mask, mask)
    image, selected = S.centered_canvas(normalized, mask, selected)
    return {"image_id": image_id, "erosion_kernel": erosion, **S.radiomics_vector(image, selected)}


def extract() -> None:
    require_lock()
    manifest, patients = read("image_manifest.csv"), read("patients.csv")
    rad_items = [
        (
            row.image_id,
            str(OUT / "images" / "percentile" / f"{row.image_id}.png"),
            str(OUT / "masks" / f"{row.image_id}.png"),
        )
        for row in manifest.itertuples()
    ]
    i3_items = [
        (
            row.image_id,
            str(OUT / "native" / f"{row.image_id}.png"),
            str(OUT / "masks" / f"{row.image_id}.png"),
        )
        for row in manifest.itertuples()
    ]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rad_rows = []
        for i, row in enumerate(pool.map(_extract_rad, rad_items), 1):
            rad_rows.append(row)
            if i % 50 == 0 or i == len(rad_items):
                print(f"Supplied-contour PyRadiomics: {i}/{len(rad_items)}", flush=True)
    rad = pd.DataFrame(rad_rows).set_index("image_id").loc[manifest.image_id]
    rad.to_csv(OUT / "features" / "radiomics_percentile_image.csv")
    rad_model = rad.loc[:, ~rad.columns.str.endswith(("_Energy", "_TotalEnergy"))]
    assert rad_model.shape == (349, 457) and np.isfinite(rad_model.to_numpy()).all()
    np.save(OUT / "features" / "radiomics_percentile.npy", R.aggregate(rad_model.to_numpy(), manifest, patients))

    with ProcessPoolExecutor(max_workers=6) as pool:
        i3_rows = []
        for i, row in enumerate(pool.map(_extract_i3, i3_items), 1):
            i3_rows.append(row)
            if i % 50 == 0 or i == len(i3_items):
                print(f"Supplied-contour directional features: {i}/{len(i3_items)}", flush=True)
    i3 = pd.DataFrame(i3_rows).set_index("image_id").loc[manifest.image_id]
    i3[["erosion_kernel"]].to_csv(OUT / "qa" / "i3_support_audit.csv")
    i3 = i3.drop(columns="erosion_kernel")
    assert i3.shape == (349, 25) and np.isfinite(i3.to_numpy()).all()
    i3.to_csv(OUT / "features" / "I3_direction_original_image.csv")
    np.save(OUT / "features" / "I3_direction_original.npy", R.aggregate(i3.to_numpy(), manifest, patients))
    write_json(OUT / "features" / "I3_feature_names.json", list(i3.columns))
    print(f"Saved {MASK_POLICY} RAD and I3 patient matrices", flush=True)


def encode() -> None:
    require_lock()
    import torch

    from radimagenet_backbones import RADIMAGENET_DIR, radimagenet_densenet121

    torch.set_num_threads(4)
    torch.manual_seed(SEEDS[0])
    manifest, patients = read("image_manifest.csv"), read("patients.csv")
    canvas, tight = [], []
    for row in manifest.itertuples():
        gray = cv2.imread(str(OUT / "images" / "percentile" / f"{row.image_id}.png"), 0)
        mask = (cv2.imread(str(OUT / "masks" / f"{row.image_id}.png"), 0) > 0).astype(np.uint8)
        canvas.append(R.canvas_image(gray, mask))
        tight.append(N.tight_canvas(gray, mask)[0])
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    before = sha256(RADIMAGENET_DIR / "DenseNet121.pt")
    for name, arrays in (("dense_canvas", canvas), ("dense_tight", tight)):
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(arrays), 32):
                x = torch.from_numpy(np.stack(arrays[start : start + 32]).transpose(0, 3, 1, 2).copy()).cuda().float()
                chunks.append(model(x / 127.5 - 1).cpu().numpy())
        values = np.concatenate(chunks).astype(np.float32)
        assert values.shape == (349, 1024) and np.isfinite(values).all()
        np.save(OUT / "features" / f"{name}_image.npy", values)
        np.save(OUT / "features" / f"{name}.npy", R.aggregate(values, manifest, patients))
        print(f"Supplied-contour RadImageNet {name}: {values.shape}", flush=True)
    write_json(
        OUT / "features" / "dense_provenance.json",
        {
            "model": "RadImageNet DenseNet-121",
            "checkpoint": str(RADIMAGENET_DIR / "DenseNet121.pt"),
            "checkpoint_sha256": before,
            "trainable_parameters": 0,
            "normalization": "x/127.5-1",
            "image_dimension": 1024,
        },
    )


def metric(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"seed": seed, "group": group, **V.score_metrics(labels, scores)}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, current in frame.groupby("group", sort=False):
        row = {"group": group, "draws": len(current)}
        for field in ("auroc", "average_precision", "auprc_trapezoid", "brier"):
            row[f"mean_{field}"] = float(current[field].mean())
            row[f"std_{field}"] = float(current[field].std(ddof=1))
        row["both_over_0_8_draws"] = int(
            ((current.auroc > 0.8) & (current.average_precision > 0.8)).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate() -> None:
    require_lock()
    import torch

    torch.set_num_threads(4)
    manifest, patients, assignments = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    rad = np.load(OUT / "features" / "radiomics_percentile.npy")
    dense_patient = np.load(OUT / "features" / "dense_canvas.npy")
    dense_image = np.load(OUT / "features" / "dense_tight_image.npy")
    i3 = np.load(OUT / "features" / "I3_direction_original.npy")
    assert rad.shape == (98, 914) and dense_patient.shape == (98, 2048)
    assert dense_image.shape == (349, 1024) and i3.shape == (98, 50)
    reference = load_reference(REFERENCE, "RELIABLE-NEG-GRAPH+I3", aliases)
    metrics, predictions, training, agreement = [], [], [], []
    with threadpool_limits(limits=4):
        for seed in SEEDS:
            indices = N.indices_for(patients, assignments, seed)
            train_indices = indices["train"]
            development = indices["validation"]
            scores = {}
            for group_name, values, head in (
                (group("RAD"), rad, V.make_head("lr", seed, rad.shape[1])),
                (group("DENSENET"), dense_patient, V.make_head("lr", seed, dense_patient.shape[1])),
                (group("I3"), i3, N.full_head(seed)),
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    head.fit(values[train_indices], labels[train_indices])
                scores[group_name] = head.predict_proba(values)[:, 1]
                joblib.dump(head, OUT / "models" / f"{group_name}_{seed}.joblib", compress=3)

            train_ids = set(patients.patient_id.iloc[train_indices])
            train_images = manifest.patient_id.isin(train_ids).to_numpy()
            scaler = StandardScaler().fit(dense_image[train_images])
            standardized = scaler.transform(dense_image)
            pca = PCA(n_components=32, random_state=seed).fit(standardized[train_images])
            reduced = pca.transform(standardized).astype(np.float32)
            bags, masks, _ = build_bags(reduced, manifest, patients)
            graph, graph_scores, _, _, loss = train_one(seed, train_indices, labels, bags, masks)
            scores[group("GRAPH")] = graph_scores
            scores[group("PAIR")] = (
                scores[group("RAD")] + scores[group("DENSENET")]
            ) / 2.0
            scores[group("FINAL")] = (
                scores[group("RAD")]
                + scores[group("GRAPH")]
                + scores[group("I3")]
            ) / 3.0
            reference_scores = reference[seed][development]
            metrics.append(metric(seed, "CURRENT-HYBRID-FINAL", labels[development], reference_scores))
            for group_name, score in scores.items():
                metrics.append(metric(seed, group_name, labels[development], score[development]))
            correlation = spearmanr(reference_scores, scores[group("FINAL")][development]).statistic
            agreement.append(
                {
                    "seed": seed,
                    "spearman": float(correlation),
                    "mean_absolute_probability_change": float(
                        np.mean(np.abs(reference_scores - scores[group("FINAL")][development]))
                    ),
                }
            )
            for patient_index, current_score, reference_score in zip(
                development,
                scores[group("FINAL")][development],
                reference_scores,
            ):
                predictions.append(
                    {
                        "seed": seed,
                        "role": "development_validation",
                        "patient_alias": aliases[patient_index],
                        "label": int(labels[patient_index]),
                        "current_hybrid_score": float(reference_score),
                        "sensitivity_score": float(current_score),
                    }
                )
            joblib.dump({"scaler": scaler, "pca": pca}, OUT / "models" / f"graph_preprocess_{seed}.joblib", compress=3)
            torch.save(graph.state_dict(), OUT / "models" / f"{MASK_POLICY}_graph_{seed}.pth")
            training.append(
                {
                    "seed": seed,
                    **loss,
                    "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                    "reliable_negative_instances": int(
                        masks[train_indices][labels[train_indices] == 0].sum().item()
                    ),
                }
            )
            print(f"{POLICY_LABEL} development sensitivity: {seed}", flush=True)

    frame = pd.DataFrame(metrics)
    summary = summarize(frame)
    current = frame.loc[frame.group.eq("CURRENT-HYBRID-FINAL")].set_index("seed")
    supplied = frame.loc[frame.group.eq(group("FINAL"))].set_index("seed")
    delta = pd.DataFrame(
        {
            "seed": list(SEEDS),
            "delta_auroc": supplied.auroc - current.auroc,
            "delta_average_precision": supplied.average_precision - current.average_precision,
            "delta_auprc_trapezoid": supplied.auprc_trapezoid - current.auprc_trapezoid,
            "delta_brier": supplied.brier - current.brier,
        }
    ).reset_index(drop=True)
    decision = {
        "selection_role": "development_validation",
        "mean_delta_auroc": float(delta.delta_auroc.mean()),
        "mean_delta_average_precision": float(delta.delta_average_precision.mean()),
        "mean_delta_auprc_trapezoid": float(delta.delta_auprc_trapezoid.mean()),
        "mean_delta_brier": float(delta.delta_brier.mean()),
        "both_metrics_improved_draws": int(
            ((delta.delta_auroc > 0) & (delta.delta_average_precision > 0)).sum()
        ),
        "both_metrics_decreased_draws": int(
            ((delta.delta_auroc < 0) & (delta.delta_average_precision < 0)).sum()
        ),
        "mean_spearman": float(pd.DataFrame(agreement).spearman.mean()),
        "interpretation": "Descriptive mask-source sensitivity only; no promotion, noninferiority, or independent-validation decision.",
        "exploratory_test_access": "prohibited",
    }
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    delta.to_csv(OUT / "paired_mask_source_deltas.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions_deidentified.csv", index=False)
    pd.DataFrame(agreement).to_csv(OUT / "prediction_agreement.csv", index=False)
    write_json(OUT / "training_provenance.json", training)
    write_json(OUT / "sensitivity_interpretation.json", decision)
    make_figure(frame)
    write_report(summary, decision)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


def make_figure(frame: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = frame.loc[frame.group.isin(("CURRENT-HYBRID-FINAL", group("FINAL")))].copy()
    colors = {"CURRENT-HYBRID-FINAL": "#4C78A8", group("FINAL"): "#E45756"}
    labels = {"CURRENT-HYBRID-FINAL": "Hybrid accepted/fallback masks", group("FINAL"): POLICY_LABEL}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True)
    for axis, metric_name, title in zip(axes, ("auroc", "average_precision"), ("AUROC", "Average precision")):
        for seed in SEEDS:
            pair = selected.loc[selected.seed.eq(seed)].set_index("group")[metric_name]
            axis.plot([0, 1], [pair["CURRENT-HYBRID-FINAL"], pair[group("FINAL")]], color="#B7B7B7", linewidth=0.8, zorder=1)
        for x, group_name in enumerate(("CURRENT-HYBRID-FINAL", group("FINAL"))):
            values = selected.loc[selected.group.eq(group_name), metric_name]
            axis.scatter(np.full(len(values), x), values, color=colors[group_name], s=24, label=labels[group_name], zorder=2)
            axis.scatter([x], [values.mean()], marker="D", s=52, color="black", zorder=3)
        axis.axhline(0.8, color="#666666", linestyle="--", linewidth=0.8)
        axis.set_xticks([0, 1], ["Hybrid", "Supplied" if MASK_POLICY == "supplied" else "Intersection"])
        axis.set_ylim(0.45, 1.01)
        axis.set_title(title)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[0].set_ylabel("Development metric")
    fig.suptitle("Renal contour source sensitivity across ten fixed patient draws", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "figures_600dpi" / f"Figure_S1_{MASK_POLICY}_contour_sensitivity.png"
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)


def write_report(summary: pd.DataFrame, decision: dict) -> None:
    indexed = summary.set_index("group")
    current = indexed.loc["CURRENT-HYBRID-FINAL"]
    supplied = indexed.loc[group("FINAL")]
    lines = [
        f"# {POLICY_LABEL} development sensitivity",
        "",
        "## Motivation",
        "",
        (
            "The reused single-class YOLOv12x-seg checkpoint has conflicting local task provenance. This one-factor ablation therefore replaces every checkpoint-proposed mask with the existing supplied kidney contour while keeping the cohort, ten patient draws, feature definitions, training rules, graph-MIL model, and equal vote fixed."
            if MASK_POLICY == "supplied"
            else "Visual quality control showed that some accepted model masks extended beyond the supplied kidney contour. This one-factor ablation intersects accepted candidates with the supplied contour and retains the supplied contour for prior fallbacks, guaranteeing that every analyzed pixel lies within the supplied renal boundary."
        ),
        "",
        "## Scope",
        "",
        "Only the development role was evaluated. The exploratory internal test was not inspected, selected on, or reported in this sensitivity analysis.",
        "",
        "## Results",
        "",
        "| Mask input and model | AUROC mean (SD) | AP mean (SD) | AUPRC mean (SD) | Brier mean (SD) | Both AUROC/AP >0.8 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Current hybrid final | {current.mean_auroc:.3f} ({current.std_auroc:.3f}) | {current.mean_average_precision:.3f} ({current.std_average_precision:.3f}) | {current.mean_auprc_trapezoid:.3f} ({current.std_auprc_trapezoid:.3f}) | {current.mean_brier:.3f} ({current.std_brier:.3f}) | {int(current.both_over_0_8_draws)}/10 |",
        f"| {POLICY_LABEL} final | {supplied.mean_auroc:.3f} ({supplied.std_auroc:.3f}) | {supplied.mean_average_precision:.3f} ({supplied.std_average_precision:.3f}) | {supplied.mean_auprc_trapezoid:.3f} ({supplied.std_auprc_trapezoid:.3f}) | {supplied.mean_brier:.3f} ({supplied.std_brier:.3f}) | {int(supplied.both_over_0_8_draws)}/10 |",
        "",
        f"Mean paired changes for {POLICY_LABEL.lower()} minus current hybrid masks were AUROC {decision['mean_delta_auroc']:+.3f}, AP {decision['mean_delta_average_precision']:+.3f}, trapezoidal AUPRC {decision['mean_delta_auprc_trapezoid']:+.3f}, and Brier {decision['mean_delta_brier']:+.3f}. The mean within-draw Spearman correlation between final probabilities was {decision['mean_spearman']:.3f}.",
        "",
        "## Interpretation",
        "",
        "These results isolate mask-source dependence but do not establish segmentation accuracy, noninferiority, or external validity. The supplied contours also require author confirmation of annotation provenance and blinding. Because the same 98 patients were previously explored, all results remain internal and exploratory.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract", "encode", "evaluate", "all"))
    args = parser.parse_args()
    stages = ("prepare", "extract", "encode", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()


if __name__ == "__main__":
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    main()
