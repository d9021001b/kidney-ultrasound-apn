"""Ultrasound-domain encoder and feature-selection controls; development only."""
import argparse
from datetime import datetime, timezone
import json
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import ROOT, FEATURES, SEEDS, OUT as BASE, digest, write_json, metric
from nckuh_representation_screen_20261102 import OUT as PREVIOUS, aggregate, tight_canvas, load_metadata

OUT = ROOT / "02_experiments/NCKUH_ultrasound_probe_20261102"
WEIGHT = ROOT / "03_data_models/external_models/USF-MAE_official_20260914/USF_MAE_100epoch.pth"
CONFIGS = [("usf_class", h) for h in ["top32", "full", "pca32"]] + [("usf_local", h) for h in ["top32", "full", "pca32"]] + [
    ("dino_class", "full"), ("dino_class", "pca32"), ("dino_patch", "full"), ("dino_patch", "pca32"),
    ("dense", "full"), ("dense", "pca32"), ("bilateral", "full"), ("bilateral", "pca32")]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "02_experiments/CYCH_usf_mae_development_20260914/protocol.json").read_text(encoding="utf-8"))
    assert digest(WEIGHT) == old["checkpoint"]["sha256"]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    hypotheses=["Ultrasound-only MAE pretraining may reduce domain mismatch",
                                "A supervised top32 filter may discard a distributed frozen representation; compare full regularized and unsupervised PCA32 linear probes"],
                    references=["https://arxiv.org/abs/2510.22990", "https://github.com/Yusufii9/USF-MAE"],
                    checkpoint_sha256=digest(WEIGHT), checkpoint_loading="weights_only=True, strict match to all 150 encoder entries of timm vit_base_patch16_224; decoder excluded",
                    input="Same mandatory masked percentile ROI, tight 224 canvas, official ImageNet channel mean/std; class and in-mask patch tokens; frozen model",
                    configurations=CONFIGS, classification="Fixed class-balanced liblinear LR; C=.1 after ANOVA32 or PCA32, C=.01 for full standardized features; no search",
                    candidate_count=len(CONFIGS), sampling="Existing NCKUH-only fixed 158/53/53 patient draws; score development only",
                    selection="Average development AUROC and AP, fixed-name tie-break; same +.03 AUROC / +.02 AP / >=6 joint wins promotion gate versus baseline fusion",
                    prohibited="No test scoring, no mixing hospital disease data, no invented kidney outcomes, no fusion of ineligible encoders",
                    caveat="Cohort and development data have been explored; no confirmatory claim or goal completion from selected development values")
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json")))
    print("Ultrasound encoder and distributed-feature control protocol fixed", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"] and digest(OUT / "protocol.json") == lock["protocol_sha256"]


def encode():
    require_lock()
    import torch
    import timm
    torch.set_num_threads(4)
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    before = digest(WEIGHT)
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    keys = set(model.state_dict())
    encoder = {key: value for key, value in checkpoint.items() if key in keys}
    assert set(encoder) == keys and len(encoder) == 150
    model.load_state_dict(encoder, strict=True)
    model = model.cuda().eval().requires_grad_(False)
    manifest, patients = load_metadata()
    mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    cls, local = [], []
    with torch.inference_mode():
        for start in range(0, len(manifest), 16):
            images, masks = [], []
            for ident in manifest.image_id.iloc[start:start+16]:
                gray = cv2.imread(str(FEATURES / "percentile" / f"{ident}.png"), 0)
                mask = cv2.imread(str(FEATURES / "masks" / f"{ident}.png"), 0) > 0
                image, binary = tight_canvas(gray, mask)
                images.append(image); masks.append(binary)
            tensor = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).cuda().float() / 255.
            tokens = model.forward_features((tensor - mean) / std).cpu().numpy()
            assert tokens.shape == (len(images), 197, 768) and np.isfinite(tokens).all()
            cls.extend(tokens[:, 0])
            for j, binary in enumerate(masks):
                coverage = binary.reshape(14, 16, 14, 16).mean((1, 3)).ravel()
                choose = coverage >= .8
                if not choose.any():
                    choose[np.argmax(coverage)] = True
                patches = tokens[j, 1:][choose]
                local.append(np.r_[tokens[j, 0], patches.mean(0), patches.std(0)])
            if start % 128 == 0:
                print(f"Frozen USF-MAE NCKUH {min(start+16, len(manifest))}/{len(manifest)}", flush=True)
    for name, array in [("usf_class", np.asarray(cls, dtype=np.float32)), ("usf_local", np.asarray(local, dtype=np.float32))]:
        np.save(OUT / f"{name}_image.npy", array)
        np.save(OUT / f"{name}.npy", aggregate(array, manifest, patients, "summary"))
    assert digest(WEIGHT) == before
    write_json(OUT / "encoder_provenance.json", dict(checkpoint_sha256=before, loaded_encoder_tensors=150,
                                                     frozen=True, images=len(manifest), torch=torch.__version__, timm=timm.__version__))
    print("USF-MAE frozen NCKUH features saved", flush=True)


def screen():
    require_lock()
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    manifest, patients = load_metadata()
    assignments = pd.read_csv(BASE / "random_patient_assignments.csv")
    arrays = {name: np.load(OUT / f"{name}.npy") for name in ["usf_class", "usf_local"]}
    arrays.update(dino_class=np.load(PREVIOUS / "dino_class_summary.npy"), dino_patch=np.load(PREVIOUS / "dino_patch_summary.npy"),
                  bilateral=np.load(PREVIOUS / "radiomics_bilateral.npy"))
    dense = np.load(FEATURES / "dense_tight_image.npy")
    arrays["dense"] = aggregate(dense, manifest, patients, "summary")
    records, predictions = [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
        for name, reduction in CONFIGS:
            values = arrays[name]
            group = name + "_" + reduction
            for seed in SEEDS:
                roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
                train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
                transform = SelectKBest(f_classif, k=32) if reduction == "top32" else (PCA(32, random_state=seed) if reduction == "pca32" else VarianceThreshold())
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), transform,
                                      LogisticRegression(C=.01 if reduction == "full" else .1, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed))
                model.fit(values[train], patients.label.to_numpy()[train])
                score = model.predict_proba(values[dev])[:, 1]
                records.append(dict(seed=seed, group=group, role="development", **metric(patients.label.to_numpy()[dev], score)))
                for index, probability in zip(dev, score):
                    predictions.append(dict(seed=seed, group=group, role="development", patient_alias=patients.patient_alias.iloc[index], label=int(patients.label.iloc[index]), score=float(probability)))
                joblib.dump(model, OUT / "models" / f"{group}_{seed}.joblib", compress=3)
            print(f"Development only completed {group}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    summary = frame.groupby("group")[["auroc", "ap"]].mean()
    summary["mean_score"] = summary.mean(axis=1)
    summary = summary.sort_values(["mean_score", "group"], ascending=[False, True])
    summary.to_csv(OUT / "development_summary.csv")
    reference = pd.read_csv(BASE / "metrics_per_draw.csv")
    reference = reference[reference.role.eq("development") & reference.group.eq("Fixed three source fusion")].set_index("seed")
    eligible = []
    for group in summary.index:
        delta = frame[frame.group.eq(group)].set_index("seed")[["auroc", "ap"]] - reference[["auroc", "ap"]]
        if delta.auroc.mean() >= .03 and delta.ap.mean() >= .02 and delta.gt(0).all(axis=1).sum() >= 6:
            eligible.append(group)
    write_json(OUT / "candidate_decision.json", dict(selected=eligible[0] if eligible else None, eligible=eligible,
                                                     selection_role="development only", test_scoring_performed=False,
                                                     tested_candidates=len(CONFIGS), target_met_on_selected_development=bool(eligible and (summary.loc[eligible[0], ["auroc", "ap"]] > .8).all())))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "encode", "screen"])
    args = parser.parse_args()
    globals()[args.stage]()
