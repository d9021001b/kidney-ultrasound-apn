"""Bounded, literature-guided NCKUH representation screen using development only."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import ROOT, OUT as BASE, FEATURES, SEEDS, digest, write_json, metric

OUT = ROOT / "02_experiments/NCKUH_representation_screen_20261102"
DINO = ROOT / "03_data_models/dinov3_vith16plus_weights_20260913"
FEATURE_GROUPS = ["radiomics_view", "radiomics_bilateral", "radiomics_sidepool",
                  "dino_class_summary", "dino_patch_summary", "dino_class_view", "dino_class_bilateral"]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    objective="Find useful patient-level APN label representations in NCKUH, aiming for AUROC and AP > .8 without fabricated side labels",
                    source_cohort_sha256=digest(BASE / "cohort_audit.json"),
                    patient_assignment_sha256=digest(BASE / "random_patient_assignments.csv"),
                    features=FEATURE_GROUPS,
                    diagnostic_only="view-availability and mask-acceptance representation is a confounding control, not an eligible disease model",
                    hypotheses=["Same-plane left-right differences can retain information lost when averaging both kidneys",
                                "View-stratified texture avoids mixing longitudinal and transverse acquisitions",
                                "Frozen DINOv3 class and kidney-internal patch tokens may encode nonredundant local appearance"],
                    references=[dict(url="https://doi.org/10.2214/ajr.146.4.775", reason="Clinical evidence for bilateral size changes and asymmetry. No physical size or swelling will be inferred from uncalibrated pixels"),
                                dict(url="https://pubmed.ncbi.nlm.nih.gov/17987762/", reason="Focal renal changes can be locally echogenic and do not occupy every view. No pseudo-lesion truth is created"),
                                dict(url="https://arxiv.org/abs/2508.10104", reason="Dense frozen self-supervised representations; exact local ViT-H+/16 variant"),
                                dict(url="https://github.com/facebookresearch/dinov3", reason="Official model source and variant documentation")],
                    encoders="Native installed transformers DINOv3ViTModel with strict loading, safetensors, local_files_only and trust_remote_code=False; no helper-code execution or fine-tuning",
                    heads="Two fixed heads per representation: training-only median imputation, standardization and ANOVA32 followed by balanced LR C=.1; or the same selector with balanced HistGradientBoosting max_iter100, max_leaf_nodes7, min_samples_leaf15, l2_regularization1",
                    maximum_candidates=14,
                    selection="Mean of development AUROC and AP across same ten draws, with fixed name ordering as tie-break; test scoring prohibited until explicit candidate lock",
                    promotion="At least +.03 mean development AUROC and +.02 mean development AP versus original fixed fusion, with both improving in >=6/10 draws. Passing this gate alone does not establish goal completion",
                    interpretation="Historical exploration and overlapping random roles remain; no independent clinical validation, no image-level or kidney-level ground truth",
                    target="Report all candidates; >.8 on selected development scores alone is not confirmation")
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve an existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json")))
    print("Bounded representation hypotheses and promotion gate fixed", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"]
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]


def load_metadata():
    return (pd.read_csv(BASE / "image_manifest_private.csv", dtype={"patient_id": str}),
            pd.read_csv(BASE / "patients_private.csv", dtype={"patient_id": str}))


def tight_canvas(gray, mask):
    yy, xx = np.nonzero(mask)
    region = gray[yy.min():yy.max()+1, xx.min():xx.max()+1]
    selected = mask[yy.min():yy.max()+1, xx.min():xx.max()+1].astype(np.uint8)
    scale = 224 / max(region.shape)
    h, w = max(1, round(region.shape[0] * scale)), max(1, round(region.shape[1] * scale))
    result = np.zeros((224, 224), np.uint8)
    binary = np.zeros_like(result)
    y, x = (224-h)//2, (224-w)//2
    result[y:y+h, x:x+w] = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    binary[y:y+h, x:x+w] = cv2.resize(selected, (w, h), interpolation=cv2.INTER_NEAREST)
    result *= binary
    return np.repeat(result[..., None], 3, axis=2), binary


def encode():
    require_lock()
    import torch
    from transformers import DINOv3ViTModel, DINOv3ViTConfig, AutoImageProcessor
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261102)
    bundle = json.loads((DINO / "WEIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    for item in bundle["files"]:
        assert (DINO / item["path"]).stat().st_size == item["bytes"]
        assert digest(DINO / item["path"]) == item["sha256"]
    manifest, _ = load_metadata()
    config = DINOv3ViTConfig.from_pretrained(DINO, local_files_only=True)
    model, info = DINOv3ViTModel.from_pretrained(DINO, config=config, local_files_only=True,
                                               trust_remote_code=False, dtype=torch.float32,
                                               output_loading_info=True, attn_implementation="sdpa")
    assert not any(info.get(k, []) for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
    model = model.cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(DINO, local_files_only=True, trust_remote_code=False)
    class_rows, patch_rows, coverage_rows = [], [], []
    with torch.inference_mode():
        for start in range(0, len(manifest), 4):
            images, masks = [], []
            for ident in manifest.image_id.iloc[start:start+4]:
                gray = cv2.imread(str(FEATURES / "percentile" / f"{ident}.png"), 0)
                mask = cv2.imread(str(FEATURES / "masks" / f"{ident}.png"), 0) > 0
                image, binary = tight_canvas(gray, mask)
                images.append(image); masks.append(binary)
            batch = processor(images=images, return_tensors="pt", do_resize=False)
            tokens = model(pixel_values=batch["pixel_values"].cuda()).last_hidden_state.cpu().numpy()
            assert tokens.shape == (len(images), 201, 1280) and np.isfinite(tokens).all()
            class_rows.extend(tokens[:, 0])
            for j, binary in enumerate(masks):
                coverage = binary.reshape(14, 16, 14, 16).mean((1, 3)).reshape(-1)
                selected = coverage >= .8
                if not selected.any():
                    selected[np.argmax(coverage)] = True
                patches = tokens[j, 5:][selected]
                patch_rows.append(np.r_[patches.mean(0), patches.std(0)])
                coverage_rows.append(dict(image_id=manifest.image_id.iloc[start+j], selected_patches=int(selected.sum())))
            if start % 64 == 0:
                print(f"Frozen DINOv3 NCKUH: {min(start+4, len(manifest))}/{len(manifest)}", flush=True)
    np.save(OUT / "dino_class_image.npy", np.asarray(class_rows, dtype=np.float32))
    np.save(OUT / "dino_patch_image.npy", np.asarray(patch_rows, dtype=np.float32))
    pd.DataFrame(coverage_rows).to_csv(OUT / "patch_coverage.csv", index=False)
    assert digest(DINO / "model.safetensors") == bundle["weight_sha256"]
    write_json(OUT / "encoder_provenance.json", dict(model_id=bundle["model_id"], weight_sha256=bundle["weight_sha256"],
                                                     images=len(manifest), frozen=True, strict_load=True, input_size=224,
                                                     remote_code=False, torch=torch.__version__,
                                                     transformers=__import__("transformers").__version__,
                                                     class_shape=list(np.asarray(class_rows).shape),
                                                     patch_shape=list(np.asarray(patch_rows).shape)))
    print("Frozen NCKUH DINOv3 features saved; weights unchanged", flush=True)


def aggregate(values, manifest, patients, mode):
    width = values.shape[1]
    rows = []
    for pid in patients.patient_id:
        sub = manifest.patient_id.eq(pid).to_numpy()
        if mode == "summary":
            rows.append(np.r_[values[sub].mean(0), values[sub].std(0)])
            continue
        slots = []
        for view in ["LL", "LT", "RL", "RT"]:
            chosen = sub & manifest["view"].eq(view).to_numpy()
            slots.append(values[chosen].mean(0) if chosen.any() else np.full(width, np.nan))
        slots = np.asarray(slots)
        if mode == "view":
            row = slots.ravel()
        elif mode == "bilateral":
            difference = np.abs(slots[:2] - slots[2:])
            relative = difference / (np.abs(slots[:2]) + np.abs(slots[2:]) + 1e-6)
            row = np.r_[difference.ravel(), relative.ravel()]
        elif mode == "sidepool":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                left, right = np.nanmean(slots[:2], axis=0), np.nanmean(slots[2:], axis=0)
            row = np.r_[np.minimum(left, right), np.maximum(left, right), np.abs(left-right)]
        else:
            raise ValueError(mode)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def build():
    require_lock()
    manifest, patients = load_metadata()
    rad = pd.read_csv(FEATURES / "radiomics_percentile_image.csv").set_index("image_id").reindex(manifest.image_id)
    rad = rad[[c for c in rad if not c.endswith(("_Energy", "_TotalEnergy"))]]
    direction = pd.read_csv(FEATURES / "i3_image.csv").set_index("image_id").reindex(manifest.image_id)
    values = np.c_[rad.to_numpy(), direction.to_numpy()]
    for mode in ["view", "bilateral", "sidepool"]:
        np.save(OUT / f"radiomics_{mode}.npy", aggregate(values, manifest, patients, mode))
    for name, modes in [("class", ["summary", "view", "bilateral"]), ("patch", ["summary"])]:
        path = OUT / f"dino_{name}_image.npy"
        if path.exists():
            for mode in modes:
                np.save(OUT / f"dino_{name}_{mode}.npy", aggregate(np.load(path), manifest, patients, mode))
    controls = []
    for pid in patients.patient_id:
        sub = manifest[manifest.patient_id.eq(pid)]
        controls.append([int(sub["view"].eq(v).any()) for v in ["LL", "LT", "RL", "RT"]] +
                        [len(sub), float(sub.mask_source.eq("existing_yolo").mean())])
    np.save(OUT / "acquisition_control.npy", np.asarray(controls, dtype=np.float32))
    print("View matched and bilateral representations built without diagnostic side labels", flush=True)


def screen():
    require_lock()
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    manifest, patients = load_metadata()
    assignments = pd.read_csv(BASE / "random_patient_assignments.csv")
    groups = [name for name in FEATURE_GROUPS if (OUT / f"{name}.npy").exists()]
    assert groups
    metrics, predictions = [], []
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
        for feature in groups + ["acquisition_control"]:
            values = np.load(OUT / f"{feature}.npy")
            for head_name in ["LR", "HGB"]:
                name = feature + "_" + head_name
                for seed in SEEDS:
                    roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
                    train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
                    classifier = (LogisticRegression(C=.1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed)
                                  if head_name == "LR" else HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                                      min_samples_leaf=15, l2_regularization=1, class_weight="balanced", random_state=seed,
                                      early_stopping=False))
                    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                          SelectKBest(f_classif, k=min(32, values.shape[1])), classifier)
                    model.fit(values[train], patients.label.to_numpy()[train])
                    score = model.predict_proba(values[dev])[:, 1]
                    metrics.append(dict(seed=seed, group=name, role="development", **metric(patients.label.to_numpy()[dev], score)))
                    for index, probability in zip(dev, score):
                        predictions.append(dict(seed=seed, group=name, role="development", patient_alias=patients.patient_alias.iloc[index], label=int(patients.label.iloc[index]), score=float(probability)))
                    (OUT / "models").mkdir(exist_ok=True)
                    joblib.dump(model, OUT / "models" / f"{name}_{seed}.joblib", compress=3)
                print(f"Development only completed {name}", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean()
    summary["mean_score"] = summary.mean(axis=1)
    summary = summary.sort_values(["mean_score", "group"], ascending=[False, True])
    summary.to_csv(OUT / "development_summary.csv")
    baseline = pd.read_csv(BASE / "metrics_per_draw.csv")
    baseline = baseline[baseline.role.eq("development") & baseline.group.eq("Fixed three source fusion")].set_index("seed")
    eligible = []
    for group in summary.index:
        if group.startswith("acquisition_control"):
            continue
        delta = frame[frame.group.eq(group)].set_index("seed")[["auroc", "ap"]] - baseline[["auroc", "ap"]]
        joint = int(delta.gt(0).all(axis=1).sum())
        if delta.auroc.mean() >= .03 and delta.ap.mean() >= .02 and joint >= 6:
            eligible.append(group)
    decision = dict(selected=eligible[0] if eligible else None, selection_role="development only", eligible=eligible,
                    no_test_scoring_performed=True, tested_disease_candidates=len(groups) * 2,
                    goal_met_in_development=bool(eligible and (summary.loc[eligible[0], ["auroc", "ap"]] > .8).all()))
    write_json(OUT / "candidate_decision.json", decision)
    print(summary.to_string(), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "encode", "build", "screen"])
    a = p.parse_args()
    globals()[a.stage]()
