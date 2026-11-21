"""Build a contour-first full-frame path for all source-labeled NCKUH kidneys."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_fullframe_20261103"
LABELS = ROOT / "02_experiments/NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv"
ASSIGNMENTS = ROOT / "02_experiments/NCKUH_DMSA_kidney_baseline_20261102/patient_assignments.csv"
PANEL_ROOTS = (
    ROOT / "03_data_models/ki-data/test_predict",
    ROOT / "03_data_models/ki-data/NCKUH3",
)
POLYGON_ROOT = ROOT / "03_data_models/ki-data/before-labeled"
LEGACY_NATIVE = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH/native"
LEGACY_MASKS = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH/masks"
DINO_BUNDLE = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
SEEDS = (20261102, 20261103, 20261104)
GROUPS = (
    "fullcohort_fullframe_radiomics_anova64_lr",
    "fullcohort_fullframe_radiomics_anova128_lr",
    "fullcohort_fullframe_radiomics_pca64_rbf_svm",
    "fullcohort_fullframe_radiomics_xgboost",
)
DEEP_GROUPS = (
    "fullcohort_radimagenet_gray_pca32_lr",
    "fullcohort_radimagenet_color_pca32_lr",
    "fullcohort_dinov3_class_pca32_lr",
    "fullcohort_dinov3_patch_pca32_lr",
    "fullcohort_deep_concat_pca64_lr",
    "fullcohort_radimagenet_color_plus_radiomics",
    "fullcohort_dinov3_class_plus_radiomics",
    "fullcohort_deep_concat_plus_radiomics",
    "fullcohort_fixed_rad_color_dinov3_radiomics",
)
STATE_GROUPS = (
    "fullcohort_rad_color_four_state_powerset",
    "fullcohort_rad_color_four_state_plus_direct",
    "fullcohort_deep_four_state_powerset",
    "fullcohort_deep_four_state_plus_direct",
    "fullcohort_rad_color_extreme_reallocated",
    "fullcohort_rad_color_extreme_reallocated_plus_direct",
)
STATE_CODE = {"N": 0, "L": 1, "R": 2, "B": 3}
INSTANCE_GROUPS = (
    "fullcohort_rad_color_image_logit_mean",
    "fullcohort_rad_color_view_specific_logit_mean",
    "fullcohort_rad_gray_image_logit_mean",
    "fullcohort_dinov3_image_logit_mean",
    "fullcohort_rad_color_image_plus_patient_direct",
    "fullcohort_rad_color_view_specific_plus_patient_direct",
)
DOMAIN_GROUPS = (
    "fullcohort_domain_specific_rad_color_image",
    "fullcohort_domain_specific_plus_patient_direct",
    "fullcohort_domain_specific_plus_state_direct",
)
VIEW_RE = re.compile(r"_(LL|LT|RL|RT)(?:$|[- (])", re.IGNORECASE)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_patient_id(value: str) -> str:
    value = str(value)
    return str(int(value)) if value.isdigit() else value


def parse_source_key(path: Path) -> tuple[str, str | None, str] | None:
    patient = re.match(r"([^_]+)", path.name)
    view = VIEW_RE.search(path.stem)
    if patient is None or view is None:
        return None
    date = re.match(r"[^_]+_([0-9]{8,9})_", path.name)
    return normalize_patient_id(patient.group(1)), date.group(1) if date else None, view.group(1).upper()


def panel_index() -> tuple[dict[tuple[str, str, str], list[Path]], dict[tuple[str, str], list[Path]]]:
    exact: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    dateless: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for root in PANEL_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parsed = parse_source_key(path)
            if parsed is None:
                continue
            patient, date, view = parsed
            if date is None:
                dateless[(patient, view)].append(path)
            else:
                exact[(patient, date, view)].append(path)
    for values in exact.values():
        values.sort(key=lambda path: ("test_predict" not in path.parts, "-1" in path.stem, str(path)))
    for values in dateless.values():
        values.sort(key=str)
    return exact, dateless


def polygon_index() -> dict[tuple[str, str, str], tuple[Path, Path]]:
    output = {}
    for json_path in POLYGON_ROOT.glob("*.json"):
        parsed = parse_source_key(json_path)
        if parsed is None or parsed[1] is None:
            continue
        image_path = json_path.with_suffix(".tif")
        if image_path.exists():
            output[(parsed[0], parsed[1], parsed[2])] = (image_path, json_path)
    return output


def split_panel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[1] % 2:
        raise ValueError(f"Invalid image-mask panel: {path}")
    half = image.shape[1] // 2
    frame = image[:, :half]
    right = cv2.cvtColor(image[:, half:], cv2.COLOR_BGR2GRAY)
    binary = (right > 127).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError(f"Panel has no kidney component: {path}")
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == selected).astype(np.uint8)
    if int(mask.sum()) < 256:
        raise ValueError(f"Panel kidney mask is too small: {path}")
    return frame, mask


def load_polygon(image_path: Path, json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable polygon image: {image_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    mask = np.zeros(image.shape[:2], np.uint8)
    polygons = []
    for shape in payload.get("shapes", []):
        if str(shape.get("label", "")).lower() != "kidney":
            continue
        points = np.rint(np.asarray(shape.get("points", []), dtype=float)).astype(np.int32)
        if len(points) >= 3:
            polygons.append(points)
    if not polygons:
        raise ValueError(f"No kidney polygon: {json_path}")
    cv2.fillPoly(mask, polygons, 1)
    return image, mask


def crop_masked(image: np.ndarray, mask: np.ndarray, margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        raise ValueError("Empty kidney mask")
    x0, x1 = max(0, int(xx.min()) - margin), min(mask.shape[1], int(xx.max()) + margin + 1)
    y0, y1 = max(0, int(yy.min()) - margin), min(mask.shape[0], int(yy.max()) + margin + 1)
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def percentile_gray(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = mask.astype(bool)
    indices = distance_transform_edt(~binary, return_distances=False, return_indices=True)
    extended = gray[tuple(indices)]
    lo, hi = np.percentile(gray[binary], [1, 99])
    scaled = extended if hi <= lo else np.rint(np.clip((extended.astype(float) - lo) / (hi - lo), 0, 1) * 255)
    return np.where(binary, scaled, 0).astype(np.uint8)


def labeled_images() -> pd.DataFrame:
    frame = pd.read_csv(LABELS, dtype={"patient_id": str, "exam_date_raw": str})
    frame = frame[frame.kidney_APN.notna()].copy().reset_index(drop=True)
    frame["normalized_patient_id"] = frame.patient_id.map(normalize_patient_id)
    assert len(frame) == 820 and frame.patient_alias.nunique() == 210
    return frame


def source_plan() -> pd.DataFrame:
    images = labeled_images()
    exact, dateless = panel_index()
    polygons = polygon_index()
    records = []
    for row in images.itertuples(index=False):
        key = (row.normalized_patient_id, row.exam_date_raw, row.view)
        candidates = exact.get(key, [])
        if candidates:
            source_type, source_path, annotation_path = "aligned_panel", candidates[0], None
        elif len(dateless.get((row.normalized_patient_id, row.view), [])) == 1:
            source_type = "aligned_panel_dateless"
            source_path = dateless[(row.normalized_patient_id, row.view)][0]
            annotation_path = None
        elif key in polygons:
            source_type = "labelme_polygon"
            source_path, annotation_path = polygons[key]
        else:
            source_path = LEGACY_NATIVE / f"{row.image_id}.png"
            annotation_path = LEGACY_MASKS / f"{row.image_id}.png"
            if not source_path.exists() or not annotation_path.exists():
                raise FileNotFoundError(f"No unique contour-first source for {row.image_id}")
            source_type = "legacy_contour_crop"
        records.append({
            "image_id": row.image_id,
            "patient_alias": row.patient_alias,
            "kidney_alias": row.kidney_alias,
            "side": row.side,
            "view": row.view,
            "source_type": source_type,
            "source_path_private": str(source_path),
            "annotation_path_private": "" if annotation_path is None else str(annotation_path),
            "candidate_count": len(candidates),
        })
    return pd.DataFrame(records)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered full-cohort protocol")
    plan = source_plan()
    counts = plan.source_type.value_counts().to_dict()
    public = plan.drop(columns=["source_path_private", "annotation_path_private"])
    public.to_csv(OUT / "deidentified_source_manifest.csv", index=False)
    assignments = pd.read_csv(ASSIGNMENTS)
    assignments[assignments.seed.isin(SEEDS)].to_csv(OUT / "patient_assignments.csv", index=False)
    write_json(OUT / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Extend contour-first clean full-frame processing from the availability subset to all210 source-labeled NCKUH patients",
        "cohort": "210 patients,414 imaged kidneys,820 images with source R/L/B/N DMSA codes",
        "source_priority": "Aligned full-frame plus supplied mask panel; LabelMe kidney polygon; explicit legacy-contour fallback only when neither full-frame source exists",
        "source_counts": counts,
        "processing": "Kidney bounding box plus8 pixels; nearest-kidney extension; within-mask P1-P99 normalization; outside-mask zero",
        "roles": "Reuse exact first three126/42/42 patient assignments from the corrected-label baseline; exploratory-test roles remain unscored",
        "prohibited": "No filename diagnosis suffix, MRN, DMSA code, test score, or development-derived source choice enters feature extraction",
        "target": "Development AUROC>0.8 and AP>0.8 followed by locked confirmation",
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in (LABELS, ASSIGNMENTS, Path(__file__))},
    })
    write_json(OUT / "run_lock.json", {
        "protocol_sha256": digest(OUT / "protocol.json"),
        "manifest_sha256": digest(OUT / "deidentified_source_manifest.csv"),
        "assignments_sha256": digest(OUT / "patient_assignments.csv"),
        "source_sha256": digest(Path(__file__)),
    })
    print(f"Registered full-cohort contour path: {counts}", flush=True)


def require_lock() -> None:
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    assert digest(OUT / "deidentified_source_manifest.csv") == lock["manifest_sha256"]
    assert digest(OUT / "patient_assignments.csv") == lock["assignments_sha256"]
    assert digest(Path(__file__)) == lock["source_sha256"]


def extract() -> None:
    require_lock()
    if (OUT / "extraction_audit.json").exists():
        raise FileExistsError("Preserve completed full-cohort extraction")
    for name in ("clean_rgb", "masks", "percentile"):
        (OUT / name).mkdir(exist_ok=True)
    plan = source_plan().set_index("image_id")
    audit = []
    for row in labeled_images().itertuples(index=False):
        source = plan.loc[row.image_id]
        image_path = Path(source.source_path_private)
        if source.source_type.startswith("aligned_panel"):
            image, mask = split_panel(image_path)
        elif source.source_type == "labelme_polygon":
            image, mask = load_polygon(image_path, Path(source.annotation_path_private))
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(source.annotation_path_private), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise ValueError(f"Unreadable legacy fallback: {row.image_id}")
            mask = (mask > 0).astype(np.uint8)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image-mask shape mismatch: {row.image_id}")
        roi, roi_mask = crop_masked(image, mask)
        gray = percentile_gray(roi, roi_mask)
        cv2.imwrite(str(OUT / "clean_rgb" / f"{row.image_id}.png"), roi)
        cv2.imwrite(str(OUT / "masks" / f"{row.image_id}.png"), roi_mask * 255)
        cv2.imwrite(str(OUT / "percentile" / f"{row.image_id}.png"), gray)
        audit.append({
            "image_id": row.image_id,
            "patient_alias": row.patient_alias,
            "source_type": source.source_type,
            "frame_height": image.shape[0],
            "frame_width": image.shape[1],
            "roi_height": roi.shape[0],
            "roi_width": roi.shape[1],
            "mask_pixels": int(mask.sum()),
            "outside_mask_nonzero": int(np.count_nonzero(gray[~roi_mask.astype(bool)])),
        })
    frame = pd.DataFrame(audit)
    frame.to_csv(OUT / "deidentified_extraction_qc.csv", index=False)
    assert len(frame) == 820 and frame.image_id.nunique() == 820
    assert frame.outside_mask_nonzero.sum() == 0
    write_json(OUT / "extraction_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "patients": int(frame.patient_alias.nunique()),
        "images": len(frame),
        "source_counts": frame.source_type.value_counts().to_dict(),
        "saved_clean_rgb": len(list((OUT / "clean_rgb").glob("*.png"))),
        "saved_masks": len(list((OUT / "masks").glob("*.png"))),
        "saved_percentile": len(list((OUT / "percentile").glob("*.png"))),
        "outside_mask_nonzero_pixels": int(frame.outside_mask_nonzero.sum()),
        "labels_used_in_extraction": [],
    })
    print("Extracted 820 contour-first images", flush=True)


def radiomics() -> None:
    require_lock()
    if (OUT / "radiomics_fullcohort.csv").exists():
        raise FileExistsError("Preserve completed full-cohort radiomics")
    images = labeled_images()
    sys.path.insert(0, str(ROOT / "02_experiments/CYCH_honest_reevaluation_20261003"))
    from extract_site import _radiomics_worker, run_pool

    jobs = [(ident, OUT / "percentile" / f"{ident}.png", OUT / "masks" / f"{ident}.png") for ident in images.image_id]
    frame = run_pool(_radiomics_worker, jobs, processes=8)
    frame.to_csv(OUT / "radiomics_fullcohort.csv")
    write_json(OUT / "radiomics_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(frame),
        "columns": len(frame.columns),
        "labels_used_in_feature_extraction": [],
        "sha256": digest(OUT / "radiomics_fullcohort.csv"),
    })
    print(f"Extracted radiomics for {len(frame)} images", flush=True)


def tight_canvas(image: np.ndarray, mask: np.ndarray, size: int = 224) -> tuple[np.ndarray, np.ndarray]:
    binary = mask.astype(bool)
    yy, xx = np.nonzero(binary)
    region = image[yy.min():yy.max()+1, xx.min():xx.max()+1]
    selected = binary[yy.min():yy.max()+1, xx.min():xx.max()+1].astype(np.uint8)
    scale = size / max(region.shape[:2])
    height, width = max(1, round(region.shape[0] * scale)), max(1, round(region.shape[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    output = np.zeros(shape, np.uint8)
    output_mask = np.zeros((size, size), np.uint8)
    top, left = (size-height)//2, (size-width)//2
    output[top:top+height, left:left+width] = cv2.resize(region, (width, height), interpolation=interpolation)
    output_mask[top:top+height, left:left+width] = cv2.resize(selected, (width, height), interpolation=cv2.INTER_NEAREST)
    output *= output_mask[..., None] if output.ndim == 3 else output_mask
    if output.ndim == 2:
        output = np.repeat(output[..., None], 3, axis=2)
    return output, output_mask


def encode_deep() -> None:
    require_lock()
    if (OUT / "dinov3_class.npy").exists():
        raise FileExistsError("Preserve completed full-cohort deep embeddings")
    import torch
    from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel

    write_json(OUT / "deep_protocol.json", {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "encoders": "Frozen RadImageNet DenseNet-121 grayscale/color and official local DINOv3 ViT-H+/16 class plus within-mask patch summaries",
        "groups": list(DEEP_GROUPS),
        "aggregation": "Longitudinal/transverse mean, standard deviation, difference, ordered target/opposite kidney; training-only PCA and balanced LR",
        "roles": "Exact first three126/42/42 patient assignments; exploratory test remains unscored",
        "labels_used_in_encoding": [],
    })
    torch.set_num_threads(4)
    torch.manual_seed(20261103)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    images = labeled_images()
    gray_canvases, color_canvases, masks = [], [], []
    for identifier in images.image_id:
        color = cv2.imread(str(OUT / "clean_rgb" / f"{identifier}.png"), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(str(OUT / "percentile" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(OUT / "masks" / f"{identifier}.png"), cv2.IMREAD_GRAYSCALE) > 0
        gray_canvas, canvas_mask = tight_canvas(gray, mask)
        color_canvas, color_mask = tight_canvas(color, mask)
        assert np.array_equal(canvas_mask, color_mask)
        gray_canvases.append(gray_canvas)
        color_canvases.append(color_canvas)
        masks.append(canvas_mask)

    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    rad = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    rad_gray, rad_color = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            for canvases, output in ((gray_canvases, rad_gray), (color_canvases, rad_color)):
                tensor = torch.from_numpy(np.stack(canvases[start:start+32]).transpose(0, 3, 1, 2).copy()).cuda().float()
                output.append(rad(tensor / 127.5 - 1).cpu().numpy())
    rad_gray = np.concatenate(rad_gray).astype(np.float32)
    rad_color = np.concatenate(rad_color).astype(np.float32)
    del rad
    torch.cuda.empty_cache()

    config = DINOv3ViTConfig.from_pretrained(DINO_BUNDLE, local_files_only=True)
    dino, info = DINOv3ViTModel.from_pretrained(
        DINO_BUNDLE, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, output_loading_info=True, attn_implementation="sdpa")
    assert not any(info.get(key, []) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    dino = dino.cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(DINO_BUNDLE, local_files_only=True, trust_remote_code=False)
    class_rows, patch_rows, counts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), 4):
            canvases = gray_canvases[start:start+4]
            batch_masks = masks[start:start+4]
            batch = processor(images=canvases, return_tensors="pt", do_resize=False)
            tokens = dino(pixel_values=batch["pixel_values"].cuda()).last_hidden_state.cpu().numpy()
            class_rows.extend(tokens[:, 0])
            for local, binary in enumerate(batch_masks):
                coverage = binary.reshape(14, 16, 14, 16).mean((1, 3)).reshape(-1)
                selected = coverage >= .8
                if not selected.any():
                    selected[np.argmax(coverage)] = True
                patches = tokens[local, 5:][selected]
                patch_rows.append(np.r_[patches.mean(0), patches.std(0)])
                counts.append(int(selected.sum()))
            if start % 80 == 0:
                print(f"Frozen full-cohort DINOv3 {min(start+4, len(images))}/{len(images)}", flush=True)
    class_rows = np.asarray(class_rows, dtype=np.float32)
    patch_rows = np.asarray(patch_rows, dtype=np.float32)
    assert rad_gray.shape == rad_color.shape == (820, 1024)
    assert class_rows.shape == (820, 1280) and patch_rows.shape == (820, 2560)
    np.save(OUT / "radimagenet_gray.npy", rad_gray)
    np.save(OUT / "radimagenet_color.npy", rad_color)
    np.save(OUT / "dinov3_class.npy", class_rows)
    np.save(OUT / "dinov3_patch_summary.npy", patch_rows)
    write_json(OUT / "deep_encoder_audit.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": 820,
        "shapes": {"rad_gray": list(rad_gray.shape), "rad_color": list(rad_color.shape),
                   "dino_class": list(class_rows.shape), "dino_patch": list(patch_rows.shape)},
        "selected_patch_min_median_max": [min(counts), float(np.median(counts)), max(counts)],
        "frozen": True,
        "labels_used_in_encoding": [],
        "dinov3_weight_sha256": digest(DINO_BUNDLE / "model.safetensors"),
    })
    print("Encoded 820 contour-first images with RadImageNet and DINOv3", flush=True)


def patient_pair_features(image_values: np.ndarray, images: pd.DataFrame, patients: pd.DataFrame) -> np.ndarray:
    width = image_values.shape[1]
    output = []
    for alias in patients.patient_alias:
        sides = []
        for side in ("left", "right"):
            subset = images[(images.patient_alias == alias) & (images.side == side)]
            rows = subset.index.to_numpy()
            mean = image_values[rows].mean(0) if len(rows) else np.full(width, np.nan)
            std = image_values[rows].std(0) if len(rows) else np.full(width, np.nan)
            views, flags = [], [float(len(rows) > 0)]
            for suffix in ("L", "T"):
                selected = subset.index[subset.view.str.endswith(suffix)].to_numpy()
                views.append(image_values[selected].mean(0) if len(selected) else np.full(width, np.nan))
                flags.append(float(len(selected) > 0))
            sides.append(np.concatenate([mean, std, views[0], views[1], views[0] - views[1], flags]))
        output.append(np.concatenate(sides))
    return np.asarray(output, dtype=np.float32)


def orient_kidneys(patient_pairs: np.ndarray, patients: pd.DataFrame, kidneys: pd.DataFrame) -> np.ndarray:
    half = patient_pairs.shape[1] // 2
    swapped = np.c_[patient_pairs[:, half:], patient_pairs[:, :half]]
    lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    return np.stack([
        patient_pairs[lookup[row.patient_alias]] if row.side == "left" else swapped[lookup[row.patient_alias]]
        for row in kidneys.itertuples(index=False)
    ])


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def fit_predict(values: np.ndarray, train: np.ndarray, development: np.ndarray,
                labels: np.ndarray, seed: int, group: str):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if "pca64" in group:
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                  VarianceThreshold(), PCA(n_components=64, whiten=True, random_state=seed))
        model = SVC(C=0.1, kernel="rbf", gamma="scale", class_weight="balanced", probability=True,
                    random_state=seed)
    else:
        selected = 64 if "anova64" in group else 128
        transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                  VarianceThreshold(), SelectKBest(f_classif, k=selected))
        if "xgboost" in group:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=.03, subsample=.8,
                                  colsample_bytree=.5, min_child_weight=5, reg_lambda=10,
                                  objective="binary:logistic", eval_metric="logloss", n_jobs=4,
                                  random_state=seed)
        else:
            model = LogisticRegression(C=.01, class_weight="balanced", solver="liblinear",
                                       max_iter=3000, random_state=seed)
    x_train = transform.fit_transform(values[train], labels[train])
    x_development = transform.transform(values[development])
    if "xgboost" in group:
        target = labels[train]
        weights = np.where(target == 1, len(target)/(2*target.sum()), len(target)/(2*(len(target)-target.sum())))
        model.fit(x_train, target, sample_weight=weights)
    else:
        model.fit(x_train, labels[train])
    return model.predict_proba(x_development)[:, 1], transform, model


def run() -> None:
    require_lock()
    if (OUT / "development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-cohort predictions")
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    assert (len(patients), len(kidneys), len(images)) == (210, 414, 820)
    radiomic = pd.read_csv(OUT / "radiomics_fullcohort.csv").set_index("image_id").reindex(images.image_id)
    columns = [column for column in radiomic if not column.endswith(("_Energy", "_TotalEnergy"))]
    image_values = radiomic[columns].to_numpy(float)
    values = orient_kidneys(patient_pair_features(image_values, images, patients), patients, kidneys)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    predictions, metrics = [], []
    (OUT / "models").mkdir(exist_ok=True)
    for seed in SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        development = np.flatnonzero(roles.eq("development"))
        assert kidneys.patient_alias.iloc[train].nunique() == 126
        assert kidneys.patient_alias.iloc[development].nunique() == 42
        assert not roles.iloc[development].eq("exploratory_test").any()
        for group in GROUPS:
            scores, transform, model = fit_predict(values, train, development, labels, seed, group)
            joblib.dump({"transform": transform, "model": model}, OUT / "models" / f"{seed}_{group}.joblib", compress=3)
            result = metric(labels[development], scores)
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development), **result})
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(scores[local]),
                })
        print(f"Completed full-cohort draw {seed}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.sort_values(["auroc_mean", "ap_mean"], ascending=False).to_csv(OUT / "development_summary.csv")
    achieved = summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist()
    write_json(OUT / "development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
        "selection_unit": "kidney",
        "split_unit": "patient",
    })
    print(summary.sort_values(["auroc_mean", "ap_mean"], ascending=False).to_string(), flush=True)


def fit_deep(values: np.ndarray, train: np.ndarray, development: np.ndarray,
             labels: np.ndarray, seed: int, components: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(components, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(values[train], labels[train])
    return model.predict_proba(values[development])[:, 1], model


def run_deep() -> None:
    require_lock()
    if (OUT / "deep_development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-cohort deep predictions")
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    labels = kidneys.kidney_APN.to_numpy(int)
    image_arrays = {
        "rad_gray": np.load(OUT / "radimagenet_gray.npy"),
        "rad_color": np.load(OUT / "radimagenet_color.npy"),
        "dino_class": np.load(OUT / "dinov3_class.npy"),
        "dino_patch": np.load(OUT / "dinov3_patch_summary.npy"),
    }
    oriented = {
        name: orient_kidneys(patient_pair_features(values, images, patients), patients, kidneys)
        for name, values in image_arrays.items()
    }
    oriented["concat"] = np.concatenate(list(oriented.values()), axis=1)
    radiomic_predictions = pd.read_csv(OUT / "development_predictions.csv")
    radiomic_reference = radiomic_predictions[
        radiomic_predictions.group.eq("fullcohort_fullframe_radiomics_anova64_lr")
    ].set_index(["seed", "kidney_alias"]).score
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    predictions, metrics = [], []
    (OUT / "deep_models").mkdir(exist_ok=True)
    for seed in SEEDS:
        roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(roles.eq("train"))
        development = np.flatnonzero(roles.eq("development"))
        specs = {
            DEEP_GROUPS[0]: (oriented["rad_gray"], 32),
            DEEP_GROUPS[1]: (oriented["rad_color"], 32),
            DEEP_GROUPS[2]: (oriented["dino_class"], 32),
            DEEP_GROUPS[3]: (oriented["dino_patch"], 32),
            DEEP_GROUPS[4]: (oriented["concat"], 64),
        }
        scores = {}
        for group, (values, components) in specs.items():
            scores[group], model = fit_deep(values, train, development, labels, seed, components)
            joblib.dump(model, OUT / "deep_models" / f"{seed}_{group}.joblib", compress=3)
        radiomic_score = np.asarray([
            radiomic_reference.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]
        ])
        scores[DEEP_GROUPS[5]] = (scores[DEEP_GROUPS[1]] + radiomic_score) / 2
        scores[DEEP_GROUPS[6]] = (scores[DEEP_GROUPS[2]] + radiomic_score) / 2
        scores[DEEP_GROUPS[7]] = (scores[DEEP_GROUPS[4]] + radiomic_score) / 2
        scores[DEEP_GROUPS[8]] = (scores[DEEP_GROUPS[1]] + scores[DEEP_GROUPS[2]] + radiomic_score) / 3
        for group in DEEP_GROUPS:
            result = metric(labels[development], scores[group])
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development), **result})
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(scores[group][local]),
                })
        print(f"Completed full-cohort deep draw {seed}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "deep_development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "deep_development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "deep_development_summary.csv")
    achieved = summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist()
    write_json(OUT / "deep_development_decision.json", {
        "groups_above_both_development_targets": achieved,
        "exploratory_test_scored": False,
        "selection_unit": "kidney",
        "split_unit": "patient",
    })
    print(summary.to_string(), flush=True)


def fit_state_model(values: np.ndarray, train: np.ndarray, states: np.ndarray, seed: int, components: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(components, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, class_weight="balanced", solver="lbfgs",
                                             max_iter=3000, random_state=seed))
    model.fit(values[train], states[train])
    probability = model.predict_proba(values)
    ordered = np.zeros((len(values), 4), dtype=float)
    ordered[:, model[-1].classes_.astype(int)] = probability
    return ordered, model


def fit_extreme_model(values: np.ndarray, train: np.ndarray, states: np.ndarray, seed: int):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    half = values.shape[1] // 2
    symmetric = np.c_[values[:, :half] + values[:, half:], np.abs(values[:, :half] - values[:, half:])]
    extreme = train[np.isin(states[train], [0, 3])]
    target = (states[extreme] == 3).astype(int)
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(32, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, class_weight="balanced", solver="liblinear",
                                             max_iter=3000, random_state=seed))
    model.fit(symmetric[extreme], target)
    return model.predict_proba(symmetric)[:, 1], model


def kidney_marginals(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return probability[:, 1] + probability[:, 3], probability[:, 2] + probability[:, 3]


def reallocate_extremes(probability: np.ndarray, bilateral_probability: np.ndarray) -> np.ndarray:
    revised = probability.copy()
    unilateral_mass = revised[:, 1] + revised[:, 2]
    extreme_mass = np.maximum(0.0, 1.0 - unilateral_mass)
    revised[:, 0] = extreme_mass * (1.0 - bilateral_probability)
    revised[:, 3] = extreme_mass * bilateral_probability
    return revised / revised.sum(axis=1, keepdims=True)


def run_states() -> None:
    require_lock()
    if (OUT / "state_development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-cohort state predictions")
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    states = patients.DMSA_APN.map(STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    rad_image = np.load(OUT / "radimagenet_color.npy")
    deep_image = np.c_[rad_image, np.load(OUT / "dinov3_class.npy")]
    rad_pair = patient_pair_features(rad_image, images, patients)
    deep_pair = patient_pair_features(deep_image, images, patients)
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    direct = pd.read_csv(OUT / "deep_development_predictions.csv")
    direct = direct[direct.group.eq("fullcohort_radimagenet_color_pca32_lr")].set_index(["seed", "kidney_alias"]).score
    patient_lookup = {alias: index for index, alias in enumerate(patients.patient_alias)}
    predictions, metrics = [], []
    (OUT / "state_models").mkdir(exist_ok=True)
    for seed in SEEDS:
        patient_roles = patients.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        train = np.flatnonzero(patient_roles.eq("train"))
        rad_probability, rad_model = fit_state_model(rad_pair, train, states, seed, 32)
        deep_probability, deep_model = fit_state_model(deep_pair, train, states, seed, 64)
        extreme_q, extreme_model = fit_extreme_model(rad_pair, train, states, seed)
        revised_probability = reallocate_extremes(rad_probability, extreme_q)
        joblib.dump(rad_model, OUT / "state_models" / f"{seed}_rad_four_state.joblib", compress=3)
        joblib.dump(deep_model, OUT / "state_models" / f"{seed}_deep_four_state.joblib", compress=3)
        joblib.dump(extreme_model, OUT / "state_models" / f"{seed}_rad_extreme.joblib", compress=3)
        kidney_roles = kidneys.patient_alias.map(assignments[assignments.seed.eq(seed)].set_index("patient_alias").role)
        development = np.flatnonzero(kidney_roles.eq("development"))

        def as_kidney(probability: np.ndarray) -> np.ndarray:
            left, right = kidney_marginals(probability)
            return np.asarray([
                left[patient_lookup[kidneys.patient_alias.iloc[index]]]
                if kidneys.side.iloc[index] == "left"
                else right[patient_lookup[kidneys.patient_alias.iloc[index]]]
                for index in development
            ])

        rad_score = as_kidney(rad_probability)
        deep_score = as_kidney(deep_probability)
        revised_score = as_kidney(revised_probability)
        direct_score = np.asarray([direct.loc[(seed, alias)] for alias in kidneys.kidney_alias.iloc[development]])
        scores = {
            STATE_GROUPS[0]: rad_score,
            STATE_GROUPS[1]: (rad_score + direct_score) / 2,
            STATE_GROUPS[2]: deep_score,
            STATE_GROUPS[3]: (deep_score + direct_score) / 2,
            STATE_GROUPS[4]: revised_score,
            STATE_GROUPS[5]: (revised_score + direct_score) / 2,
        }
        for group, score in scores.items():
            result = metric(labels[development], score)
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development), **result})
            for local, index in enumerate(development):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed full-cohort state draw {seed}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "state_development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "state_development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "state_development_summary.csv")
    write_json(OUT / "state_development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False,
        "state_definition": STATE_CODE,
    })
    print(summary.to_string(), flush=True)


def image_weights(images: pd.DataFrame, indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    subset = images.iloc[indices]
    patient_counts = subset.patient_alias.value_counts()
    weights = 1.0 / subset.patient_alias.map(patient_counts).to_numpy(float)
    target = labels[indices]
    for value in (0, 1):
        selected = target == value
        if selected.any():
            weights[selected] *= 0.5 / weights[selected].sum()
    return weights / weights.mean()


def fit_image_head(values: np.ndarray, train: np.ndarray, labels: np.ndarray,
                   weights: np.ndarray, seed: int, components: int = 32):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          PCA(components, whiten=True, random_state=seed),
                          LogisticRegression(C=.1, solver="liblinear", max_iter=3000, random_state=seed))
    model.fit(values[train], labels[train], logisticregression__sample_weight=weights)
    return model


def aggregate_image_logits(scores: np.ndarray, images: pd.DataFrame, kidney_aliases: pd.Series) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1-1e-6)
    logits = np.log(clipped / (1-clipped))
    means = pd.Series(logits, index=images.kidney_alias).groupby(level=0).mean()
    values = means.reindex(kidney_aliases).to_numpy(float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def run_instances() -> None:
    require_lock()
    if (OUT / "instance_development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-cohort instance predictions")
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    image_labels = images.kidney_APN.to_numpy(int)
    kidney_labels = kidneys.kidney_APN.to_numpy(int)
    arrays = {
        "rad_color": np.load(OUT / "radimagenet_color.npy"),
        "rad_gray": np.load(OUT / "radimagenet_gray.npy"),
        "dino": np.load(OUT / "dinov3_class.npy"),
    }
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    direct = pd.read_csv(OUT / "deep_development_predictions.csv")
    direct = direct[direct.group.eq("fullcohort_radimagenet_color_pca32_lr")].set_index(["seed", "kidney_alias"]).score
    predictions, metrics = [], []
    (OUT / "instance_models").mkdir(exist_ok=True)
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        image_roles = images.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train_images = np.flatnonzero(image_roles.eq("train"))
        development_kidneys = np.flatnonzero(kidney_roles.eq("development"))
        all_scores = {}
        for name, values in arrays.items():
            weights = image_weights(images, train_images, image_labels)
            model = fit_image_head(values, train_images, image_labels, weights, seed)
            all_scores[name] = model.predict_proba(values)[:, 1]
            joblib.dump(model, OUT / "instance_models" / f"{seed}_{name}.joblib", compress=3)

        view_scores = np.full(len(images), np.nan)
        for suffix in ("L", "T"):
            selected_train = train_images[images.view.iloc[train_images].str.endswith(suffix).to_numpy()]
            weights = image_weights(images, selected_train, image_labels)
            model = fit_image_head(arrays["rad_color"], selected_train, image_labels, weights, seed)
            selected_all = np.flatnonzero(images.view.str.endswith(suffix))
            view_scores[selected_all] = model.predict_proba(arrays["rad_color"][selected_all])[:, 1]
            joblib.dump(model, OUT / "instance_models" / f"{seed}_rad_color_view_{suffix}.joblib", compress=3)
        assert np.isfinite(view_scores).all()

        aliases = kidneys.kidney_alias.iloc[development_kidneys]
        rad_color_score = aggregate_image_logits(all_scores["rad_color"], images, aliases)
        view_score = aggregate_image_logits(view_scores, images, aliases)
        rad_gray_score = aggregate_image_logits(all_scores["rad_gray"], images, aliases)
        dino_score = aggregate_image_logits(all_scores["dino"], images, aliases)
        direct_score = np.asarray([direct.loc[(seed, alias)] for alias in aliases])
        scores = {
            INSTANCE_GROUPS[0]: rad_color_score,
            INSTANCE_GROUPS[1]: view_score,
            INSTANCE_GROUPS[2]: rad_gray_score,
            INSTANCE_GROUPS[3]: dino_score,
            INSTANCE_GROUPS[4]: (rad_color_score + direct_score) / 2,
            INSTANCE_GROUPS[5]: (view_score + direct_score) / 2,
        }
        for group, score in scores.items():
            result = metric(kidney_labels[development_kidneys], score)
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development_kidneys), **result})
            for local, index in enumerate(development_kidneys):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(kidney_labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed full-cohort instance draw {seed}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "instance_development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "instance_development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "instance_development_summary.csv")
    write_json(OUT / "instance_development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False,
        "image_labels": "Source-verified kidney APN label inherited only by images of that same kidney",
    })
    print(summary.to_string(), flush=True)


def run_domains() -> None:
    require_lock()
    if (OUT / "domain_development_predictions.csv").exists():
        raise FileExistsError("Preserve completed full-cohort domain predictions")
    sys.path.insert(0, str(Path(__file__).parent))
    from nckuh_dmsa_kidney_baseline_20261102 import load_reference

    patients, kidneys, images = load_reference()
    images = images[images.kidney_APN.notna()].copy().reset_index(drop=True)
    kidneys = kidneys[kidneys.kidney_APN.notna()].copy().reset_index(drop=True)
    patients = patients[patients.patient_alias.isin(images.patient_alias)].copy().reset_index(drop=True)
    image_labels = images.kidney_APN.to_numpy(int)
    kidney_labels = kidneys.kidney_APN.to_numpy(int)
    values = np.load(OUT / "radimagenet_color.npy")
    qc = pd.read_csv(OUT / "deidentified_extraction_qc.csv").set_index("image_id").reindex(images.image_id)
    domains = np.where(qc.frame_width.to_numpy(int) <= 700, "small_frame", "large_frame")
    assignments = pd.read_csv(OUT / "patient_assignments.csv")
    direct = pd.read_csv(OUT / "deep_development_predictions.csv")
    direct = direct[direct.group.eq("fullcohort_radimagenet_color_pca32_lr")].set_index(["seed", "kidney_alias"]).score
    state = pd.read_csv(OUT / "state_development_predictions.csv")
    state = state[state.group.eq("fullcohort_deep_four_state_plus_direct")].set_index(["seed", "kidney_alias"]).score
    predictions, metrics = [], []
    (OUT / "domain_models").mkdir(exist_ok=True)
    for seed in SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        image_roles = images.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train_images = np.flatnonzero(image_roles.eq("train"))
        development_kidneys = np.flatnonzero(kidney_roles.eq("development"))
        weights = image_weights(images, train_images, image_labels)
        global_model = fit_image_head(values, train_images, image_labels, weights, seed)
        scores = global_model.predict_proba(values)[:, 1]
        joblib.dump(global_model, OUT / "domain_models" / f"{seed}_global.joblib", compress=3)
        for domain in ("small_frame", "large_frame"):
            selected_train = train_images[domains[train_images] == domain]
            if len(selected_train) < 40 or np.unique(image_labels[selected_train]).size < 2:
                continue
            selected_weights = image_weights(images, selected_train, image_labels)
            model = fit_image_head(values, selected_train, image_labels, selected_weights, seed)
            selected_all = np.flatnonzero(domains == domain)
            scores[selected_all] = model.predict_proba(values[selected_all])[:, 1]
            joblib.dump(model, OUT / "domain_models" / f"{seed}_{domain}.joblib", compress=3)
        aliases = kidneys.kidney_alias.iloc[development_kidneys]
        domain_score = aggregate_image_logits(scores, images, aliases)
        direct_score = np.asarray([direct.loc[(seed, alias)] for alias in aliases])
        state_score = np.asarray([state.loc[(seed, alias)] for alias in aliases])
        groups = {
            DOMAIN_GROUPS[0]: domain_score,
            DOMAIN_GROUPS[1]: (domain_score + direct_score) / 2,
            DOMAIN_GROUPS[2]: (domain_score + state_score) / 2,
        }
        for group, score in groups.items():
            result = metric(kidney_labels[development_kidneys], score)
            metrics.append({"seed": seed, "role": "development", "group": group,
                            "patients": 42, "kidneys": len(development_kidneys), **result})
            for local, index in enumerate(development_kidneys):
                predictions.append({
                    "seed": seed,
                    "group": group,
                    "patient_alias": kidneys.patient_alias.iloc[index],
                    "kidney_alias": kidneys.kidney_alias.iloc[index],
                    "side": kidneys.side.iloc[index],
                    "label": int(kidney_labels[index]),
                    "score": float(score[local]),
                })
        print(f"Completed full-cohort domain draw {seed}", flush=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "domain_development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "domain_development_predictions.csv", index=False)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    summary.to_csv(OUT / "domain_development_summary.csv")
    write_json(OUT / "domain_development_decision.json", {
        "groups_above_both_development_targets": summary[(summary.auroc_mean > .8) & (summary.ap_mean > .8)].index.tolist(),
        "exploratory_test_scored": False,
        "domain_rule": "frame width<=700 versus >700, fixed before scoring; rare domains use the global training-only head",
    })
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "extract", "radiomics", "encode_deep", "run", "run_deep", "run_states", "run_instances", "run_domains"))
    args = parser.parse_args()
    {"prepare": prepare, "extract": extract, "radiomics": radiomics, "encode_deep": encode_deep,
     "run": run, "run_deep": run_deep, "run_states": run_states,
     "run_instances": run_instances, "run_domains": run_domains}[args.stage]()
