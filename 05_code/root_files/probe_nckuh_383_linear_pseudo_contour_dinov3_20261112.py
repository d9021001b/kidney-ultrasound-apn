"""Exploratory linear APN probe: add train-only unreviewed predicted contours."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluate_nckuh_383_baseline_20261111 import aggregate
from nckuh_dmsa_fullcohort_fullframe_20261103 import crop_masked, percentile_gray, tight_canvas
from prepare_nckuh_383_contour_cohort_20261111 import OUT as REGISTRY, ROOT, digest


EXP = ROOT / "02_experiments"
ROUTER = EXP / "NCKUH_revised_383_probe_router_20261112"
COMP = ROUTER / "linear_segmentation_finetune/missing_linear_candidates/component_review"
SOURCE = COMP.parent
OLD_CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
OLD_DINO = EXP / "NCKUH_revised_383_dinov3_20261111"
WEIGHTS = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
OUT = ROUTER / "linear_pseudo_contour_dinov3_probe"
SEED = 20261112


def train_components():
    audit = json.loads((COMP / "audit.json").read_text(encoding="utf-8"))
    assert digest(COMP / "component_manifest_deidentified.csv") == audit["component_manifest_sha256"]
    components = pd.read_csv(COMP / "component_manifest_deidentified.csv")
    components = components[components.role.eq("train")].reset_index(drop=True)
    assert len(components) == 180 and components.patient_alias.nunique() == 26
    return components


def prepare():
    if (OUT / "roi_audit.json").exists():
        raise FileExistsError("Preserve extracted provisional ROI")
    components = train_components()
    linkage = pd.read_csv(SOURCE / "source_linkage_private.csv")
    paths = linkage.set_index("candidate_id").source_path
    assert components.candidate_id.isin(paths.index).all()
    (OUT / "percentile").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(exist_ok=True)
    rows = []
    for item in components.itertuples(index=False):
        image = cv2.imread(str(paths[item.candidate_id]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(COMP / "masks" / f"{item.instance_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(item.instance_id)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        assert mask.shape == image.shape[:2]
        roi, roi_mask = crop_masked(image, (mask > 0).astype(np.uint8))
        gray = percentile_gray(roi, roi_mask)
        assert cv2.imwrite(str(OUT / "percentile" / f"{item.instance_id}.png"), gray)
        assert cv2.imwrite(str(OUT / "masks" / f"{item.instance_id}.png"), roi_mask * 255)
        rows.append({"instance_id": item.instance_id, "candidate_id": item.candidate_id,
                     "patient_alias": item.patient_alias,
                     "kidney_alias": item.patient_alias + "_" + item.side,
                     "source_type": "unreviewed_predicted_component", "mask_pixels": int(roi_mask.sum())})
    manifest = pd.DataFrame(rows)
    assert len(manifest) == 180 and manifest.instance_id.is_unique
    manifest.to_csv(OUT / "train_instance_manifest_deidentified.csv", index=False)
    report = {"components_sha256": digest(COMP / "component_manifest_deidentified.csv"),
              "instances": len(manifest), "patients": manifest.patient_alias.nunique(),
              "kidneys": manifest.kidney_alias.nunique(),
              "manifest_sha256": digest(OUT / "train_instance_manifest_deidentified.csv"),
              "apn_labels_used": False, "test_scored": False,
              "warning": "Unreviewed predicted masks; exploratory train-only input, not accepted contour ground truth."}
    (OUT / "roi_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def encode():
    import torch
    from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel

    if (OUT / "dinov3_class.npy").exists():
        raise FileExistsError("Preserve encoded provisional ROI")
    audit = json.loads((OUT / "roi_audit.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(OUT / "train_instance_manifest_deidentified.csv")
    assert digest(OUT / "train_instance_manifest_deidentified.csv") == audit["manifest_sha256"]
    source = json.loads((WEIGHTS / "WEIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    assert digest(WEIGHTS / "model.safetensors") == source["weight_sha256"]
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = DINOv3ViTConfig.from_pretrained(WEIGHTS, local_files_only=True)
    model, info = DINOv3ViTModel.from_pretrained(
        WEIGHTS, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True, attn_implementation="sdpa")
    assert not any(info.get(key, []) for key in
                   ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    model = model.cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(WEIGHTS, local_files_only=True, trust_remote_code=False)
    classes = []
    with torch.inference_mode():
        for start in range(0, len(manifest), 4):
            canvases = []
            for ident in manifest.instance_id.iloc[start:start + 4]:
                image = cv2.imread(str(OUT / "percentile" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(OUT / "masks" / f"{ident}.png"), cv2.IMREAD_GRAYSCALE)
                if image.ndim == 3 and image.shape[2] == 1:
                    image = image[:, :, 0]
                if mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask[:, :, 0]
                assert image.shape == mask.shape and np.count_nonzero(mask)
                canvas, _ = tight_canvas(image, mask > 0)
                canvases.append(canvas)
            batch = processor(images=canvases, return_tensors="pt", do_resize=False)
            tokens = model(pixel_values=batch["pixel_values"].cuda()).last_hidden_state
            classes.extend(tokens[:, 0].cpu().numpy())
            if start % 40 == 0:
                print(f"Encoded {min(start+4,len(manifest))}/{len(manifest)}", flush=True)
    matrix = np.asarray(classes, dtype=np.float32)
    assert matrix.shape == (180, 1280) and np.isfinite(matrix).all()
    np.save(OUT / "dinov3_class.npy", matrix)
    report = {"images": len(matrix), "features": matrix.shape[1],
              "official_weight_sha256": source["weight_sha256"],
              "roi_manifest_sha256": audit["manifest_sha256"],
              "features_sha256": digest(OUT / "dinov3_class.npy"),
              "frozen": True, "apn_labels_used": False, "test_scored": False}
    (OUT / "encoding_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def model_scores(image_features, images, kidneys, train_patients, query_kidneys):
    train_images = np.flatnonzero(images.patient_alias.isin(train_patients))
    scaler = StandardScaler().fit(image_features[train_images])
    transformed = scaler.transform(image_features)
    pca = PCA(32, random_state=SEED).fit(transformed[train_images])
    kidney_features = aggregate(pca.transform(transformed), images, kidneys.kidney_alias)
    fit = np.flatnonzero(kidneys.patient_alias.isin(train_patients))
    labels = kidneys.kidney_APN.to_numpy(int)
    head_scaler = StandardScaler().fit(kidney_features[fit])
    head = LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                              max_iter=3000, random_state=SEED).fit(
                                  head_scaler.transform(kidney_features[fit]), labels[fit])
    return head.predict_proba(head_scaler.transform(kidney_features[query_kidneys]))[:, 1]


def evaluate():
    if (OUT / "development_comparison.json").exists():
        raise FileExistsError("Preserve provisional training comparison")
    audit = json.loads((OUT / "encoding_audit.json").read_text(encoding="utf-8"))
    assert digest(OUT / "dinov3_class.npy") == audit["features_sha256"]
    old_images = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    route = pd.read_csv(ROUTER / "image_routes.csv")
    old_images = old_images[old_images.image_id.isin(route.loc[route.route.eq("linear"), "image_id"])].copy()
    roles = pd.read_csv(REGISTRY / "patient_assignments.csv")
    old_images = old_images.merge(roles[["patient_alias", "role"]], on="patient_alias", validate="many_to_one")
    old_images = old_images[old_images.role.isin(("train", "development"))].reset_index(drop=True)
    original_all = pd.read_csv(REGISTRY / "contour_image_manifest.csv")
    original_matrix = np.load(OLD_DINO / "dinov3_class.npy", mmap_mode="r")
    index = {value: i for i, value in enumerate(original_all.image_id)}
    old_features = np.asarray(original_matrix[[index[value] for value in old_images.image_id]], dtype=np.float32)
    new_images = pd.read_csv(OUT / "train_instance_manifest_deidentified.csv").rename(columns={"instance_id": "image_id"})
    new_images["role"] = "train"
    new_features = np.load(OUT / "dinov3_class.npy")
    assert len(new_images) == len(new_features) == 180
    assert set(new_images.patient_alias).isdisjoint(old_images.patient_alias)
    people = roles.set_index("patient_alias")
    assert people.loc[new_images.patient_alias, "role"].eq("train").all()
    old_train = set(old_images.loc[old_images.role.eq("train"), "patient_alias"])
    dev_patients = set(old_images.loc[old_images.role.eq("development"), "patient_alias"])
    assert len(old_train) == 59 and len(dev_patients) == 20
    output = {"protocol": "Frozen official DINOv3 class token; train-image scaler/PCA32; kidney image mean/std; balanced LR C=0.1. Fixed old linear train versus old plus 26 pseudo-contour patients; unchanged 20-patient supplied-contour development.",
              "source_pseudo_features_sha256": audit["features_sha256"],
              "source_routes_sha256": digest(ROUTER / "image_routes.csv"),
              "test_scored": False, "arms": {}}
    predictions = []
    for name, use_new in (("supplied_contours_only", False), ("plus_unreviewed_pseudo_train", True)):
        images = pd.concat([old_images, new_images], ignore_index=True) if use_new else old_images.copy()
        values = np.concatenate([old_features, new_features], axis=0) if use_new else old_features
        kidneys = images[["kidney_alias", "patient_alias", "role"]].drop_duplicates("kidney_alias").reset_index(drop=True)
        assert kidneys.groupby("patient_alias").role.nunique().eq(1).all()
        code = people.loc[kidneys.patient_alias, "DMSA_APN"].to_numpy(str)
        side = kidneys.kidney_alias.str[-1].to_numpy(str)
        kidneys["kidney_APN"] = ((code == "B") | (code == side)).astype(int)
        old_check = pd.read_csv(REGISTRY / "contour_kidney_manifest.csv").set_index("kidney_alias")
        both = kidneys[kidneys.kidney_alias.isin(old_check.index)]
        assert both.kidney_APN.tolist() == old_check.loc[both.kidney_alias, "kidney_APN"].tolist()
        train_patients = set(kidneys.loc[kidneys.role.eq("train"), "patient_alias"])
        assert len(train_patients) == len(old_train) + (26 if use_new else 0)
        dev = np.flatnonzero(kidneys.role.eq("development"))
        assert set(kidneys.patient_alias.iloc[dev]) == dev_patients and len(dev) == 40
        score = model_scores(values, images, kidneys, train_patients, dev)
        y = kidneys.kidney_APN.iloc[dev].to_numpy(int)
        output["arms"][name] = {"train_patients": len(train_patients),
                                "train_kidneys": int(kidneys.role.eq("train").sum()),
                                "train_images": int(images.role.eq("train").sum()),
                                "development_patients": len(dev_patients), "development_kidneys": len(dev),
                                "development_auroc": float(roc_auc_score(y, score)),
                                "development_ap": float(average_precision_score(y, score))}
        predictions.extend({"arm": name, "patient_alias": kidneys.patient_alias.iloc[i],
                            "kidney_alias": kidneys.kidney_alias.iloc[i], "label": int(y[j]),
                            "score": float(score[j])} for j, i in enumerate(dev))
    output["warning"] = "Added contours have not passed anatomical review. This is an exploratory feasibility contrast on repeatedly studied development patients, not a final model or independent validation."
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    output["development_predictions_sha256"] = digest(OUT / "development_predictions.csv")
    (OUT / "development_comparison.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def uncertainty():
    target = OUT / "paired_uncertainty.json"
    if target.exists():
        raise FileExistsError("Preserve paired uncertainty")
    report = json.loads((OUT / "development_comparison.json").read_text(encoding="utf-8"))
    assert digest(OUT / "development_predictions.csv") == report["development_predictions_sha256"]
    frame = pd.read_csv(OUT / "development_predictions.csv")
    baseline = frame[frame.arm.eq("supplied_contours_only")].set_index("kidney_alias")
    added = frame[frame.arm.eq("plus_unreviewed_pseudo_train")].set_index("kidney_alias")
    assert set(baseline.index) == set(added.index) and len(baseline) == 40
    baseline = baseline.sort_index()
    added = added.loc[baseline.index]
    assert baseline.patient_alias.tolist() == added.patient_alias.tolist()
    assert baseline.label.tolist() == added.label.tolist()
    patients = sorted(baseline.patient_alias.unique())
    groups = [np.flatnonzero(baseline.patient_alias.eq(patient)) for patient in patients]
    labels = baseline.label.to_numpy(int)
    old = baseline.score.to_numpy(float)
    new = added.score.to_numpy(float)
    random = np.random.default_rng(SEED)
    draws = []
    for _ in range(10000):
        sampled = np.concatenate([groups[j] for j in random.integers(0, len(groups), len(groups))])
        y = labels[sampled]
        if np.unique(y).size != 2:
            continue
        draws.append((roc_auc_score(y, old[sampled]), average_precision_score(y, old[sampled]),
                      roc_auc_score(y, new[sampled]), average_precision_score(y, new[sampled])))
    values = np.asarray(draws)
    delta = values[:, 2:] - values[:, :2]
    result = {"patients": len(patients), "kidneys": len(labels), "valid_patient_cluster_draws": len(values),
              "baseline_auroc_95ci": np.percentile(values[:, 0], [2.5, 97.5]).tolist(),
              "baseline_ap_95ci": np.percentile(values[:, 1], [2.5, 97.5]).tolist(),
              "pseudo_augmented_auroc_95ci": np.percentile(values[:, 2], [2.5, 97.5]).tolist(),
              "pseudo_augmented_ap_95ci": np.percentile(values[:, 3], [2.5, 97.5]).tolist(),
              "paired_auroc_delta": float(roc_auc_score(labels, new) - roc_auc_score(labels, old)),
              "paired_ap_delta": float(average_precision_score(labels, new) - average_precision_score(labels, old)),
              "paired_auroc_delta_95ci": np.percentile(delta[:, 0], [2.5, 97.5]).tolist(),
              "paired_ap_delta_95ci": np.percentile(delta[:, 1], [2.5, 97.5]).tolist(),
              "scope": "Conditional exploratory interval on the same 20 repeatedly studied development patients; pseudo masks have not been reviewed; no test APN scores."}
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate", "uncertainty"))
    {"prepare": prepare, "encode": encode, "evaluate": evaluate,
     "uncertainty": uncertainty}[parser.parse_args().stage]()
