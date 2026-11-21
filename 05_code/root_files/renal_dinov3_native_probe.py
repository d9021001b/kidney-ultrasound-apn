"""Bounded native DINOv3 probe. Configuration IDs are not new model names."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shutil
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_local_features as L
import renal_voting_validation as V
from renal_dinov3_single_encoder import verify_bundle, DINO3

OUT = R.ROOT / "CYCH_dinov3_native_probe_20260913"
CONFIGS = {
    "RAD": ("radiomics_percentile", "top32", None),
    "DN-A": ("radimagenet_percentile", "top32", None),
    "DN-B": ("radimagenet_percentile", "full", None),
    "DN-C": ("dense_tight", "full", None),
    "D3-A": ("dino_canvas_cls", "top32", "DN-A"),
    "D3-B": ("dino_canvas_cls", "full", "DN-B"),
    "D3-C": ("dino_tight_cls", "full", "DN-C"),
    "D3-D": ("dino_tight_patch_mean_std", "full", "DN-C"),
    "D3-E": ("dino_tight_cls_patch_mean", "full", "DN-C"),
}


def read(name):
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def require_lock():
    lock = json.loads((OUT / "protocol_locked.json").read_text())
    assert lock["sha256"] == R.sha256(OUT / "protocol.json"), "Protocol changed"
    assert lock["source_sha256"] == R.sha256(Path(__file__)), "Source changed after lock"


def full_head(seed):
    return make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(), StandardScaler(),
        LogisticRegression(C=0.1, solver="liblinear", class_weight="balanced", max_iter=2000,
                           random_state=seed))


def tight_canvas(gray, mask, size=224):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError("Empty kidney mask")
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()+1), int(yy.min()), int(yy.max()+1)
    h, w = y1-y0, x1-x0
    scale = size/max(h, w)
    nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
    py, px = (size-nh)//2, (size-nw)//2
    output, om = np.zeros((size, size), np.uint8), np.zeros((size, size), np.uint8)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    output[py:py+nh, px:px+nw] = cv2.resize(gray[y0:y1, x0:x1], (nw, nh), interpolation=interp)
    om[py:py+nh, px:px+nw] = cv2.resize(mask[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_NEAREST)
    output *= om
    transform = dict(x0=x0, y0=y0, x1=x1, y1=y1, pad_x=px, pad_y=py,
                     scale_x=nw/w, scale_y=nh/h, resized_w=nw, resized_h=nh)
    return np.repeat(output[..., None], 3, axis=2), om, transform


def patch_selection(mask):
    coverage = mask.reshape(14, 16, 14, 16).mean((1, 3)).reshape(-1)
    selected = coverage >= 0.8
    fallback = not bool(selected.any())
    if fallback:
        selected[np.argmax(coverage)] = True
    return coverage, selected, fallback


def indices_for(patients, split, seed):
    per = split[split.seed.eq(seed)]
    assert len(per) == 98 and per.patient_id.nunique() == 98
    assert set(per.patient_id) == set(patients.patient_id)
    indices = {role: np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(role), "patient_id"]))
               for role in ("train", "validation", "test")}
    assert [len(indices[r]) for r in indices] == [58, 20, 20]
    assert len(set(np.concatenate(list(indices.values())))) == 98
    return indices


def prepare():
    OUT.mkdir(exist_ok=True)
    for directory in ("features", "models", "source", "qa", "figures_600dpi", "inputs/tight", "inputs/canvas"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        dst = OUT / name
        if dst.exists():
            assert R.sha256(dst) == R.sha256(L.OUT / name)
        else:
            shutil.copy2(L.OUT / name, dst)
    df, patients, split = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    assert len(df) == 349 and len(patients) == 98 and int(patients.label.sum()) == 52
    assert df.image_id.nunique() == 349
    assert not df.groupby("patient_id").label.nunique().gt(1).any()
    assert df.groupby("patient_id").label.first().to_dict() == patients.set_index("patient_id").label.to_dict()
    assert not df.source_sha256.duplicated().any(), "Exact image duplicates require reconciliation"
    for seed in R.SEEDS:
        indices_for(patients, split, seed)
    inherited = []
    for name in ("radiomics_percentile", "radimagenet_percentile"):
        src, dst = L.OUT / "features" / f"{name}.npy", OUT / "features" / f"{name}.npy"
        shutil.copy2(src, dst)
        inherited.append(dict(source=str(src), sha256=R.sha256(src), shape=list(np.load(src).shape)))
    protocol = dict(
        timestamp_scope="Locked before corrected-native classification; earlier custom-loader results already known.",
        cohort=dict(patients=98, images=349, positive=52, negative=46), seeds=R.SEEDS,
        assignments={name: R.sha256(OUT / name) for name in ("patients.csv", "image_manifest.csv", "random_patient_assignments.csv")},
        roles=dict(train=58, development=20, exploratory_test=20), configs=CONFIGS,
        loader="Explicit transformers.DINOv3ViTModel; local_files_only=True; trust_remote_code=False; eager float32; no custom bundle code.",
        bundle=verify_bundle(), inherited=inherited,
        inputs="Unchanged P1-P99 grayscale kidney masks. Original canvas exactly R.canvas_image. Tight bounding box letterbox at224, nearest mask, area downsampling or linear upsampling. Native resolution not increased in information.",
        patches="14x14; skip1CLS+4registers; >=0.8 in-mask coverage; fallback single maximal-coverage patch, first row-major tie. Preserve all coverage and inverse coordinate mapping.",
        representations="D3-A/B canvas CLS; C tight CLS; D tight in-mask patch mean/std; E tight CLS+patch mean; image features -> patient image mean/std.",
        head="C=.1 L2 balanced liblinear max_iter2000. Train-only imputer/scaler; original top32 head for A, full head includes train-only constant removal. No search.",
        single_gate="Compare each D3 to matched DN control on development. Both mean AUROC/AP deltas >=-.02. Eligible winner by mean(AUROC+AP)/2, ties simpler configuration ID.",
        vote_gate="At most one D3 winner: equal mean of RAD,DN-A,candidate probabilities. Development vs fixed pair: mean AUROC delta>=.03, AP delta>=.02, both improve in>=6/10 draws. No tuning weights or thresholds.",
        leakage="Only58 training patients fit any estimator. Development chooses configuration; test evaluated after decision. All98 previously explored. Ten overlapping draws are not independent cohorts or k-fold validation.",
        tests="Full model state hash before/after; repeat and batch1/batch2 maxabs<=1e-4; frozen gradients; complete key match; DenseNet re-extraction patient cache allclose rtol1e-4 atol1e-4; inherited baseline metric reproduction.",
        exclusions="No clinical covariates, generated images, target encoder pretraining, segmentation retraining, artifact inpainting, pseudo-lesion labels or test-selected models.")
    normalized = json.loads(json.dumps(protocol))
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == normalized, "Choose a new directory for protocol changes"
    else:
        R.write_json(path, normalized)
    if not (OUT / "protocol_locked.json").exists():
        R.write_json(OUT / "protocol_locked.json", dict(utc=datetime.now(timezone.utc).isoformat(),
            sha256=R.sha256(path), source_sha256=R.sha256(Path(__file__))))
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    R.write_json(OUT / "preserved_manuscript.json", dict(path=str(R.ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"),
        sha256=R.sha256(R.ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx")))
    print("Locked native DINOv3 probe; cohort and all ten patient assignments verified", flush=True)


def encode():
    require_lock()
    import torch
    import transformers
    from transformers import AutoImageProcessor, DINOv3ViTConfig, DINOv3ViTModel
    from radimagenet_backbones import radimagenet_densenet121, RADIMAGENET_DIR
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    df, patients = read("image_manifest.csv"), read("patients.csv")
    arrays = {"canvas": [], "tight": []}
    selections, mapping, qa = [], [], []
    for row in df.itertuples():
        gray = cv2.imread(str(L.OUT / "images/percentile" / f"{row.image_id}.png"), 0)
        mp = L.OUT / "masks" / f"{row.image_id}.png"
        assert R.sha256(mp) == row.mask_sha256
        mask = (cv2.imread(str(mp), 0) > 0).astype(np.uint8)
        assert gray.shape == mask.shape and not gray[mask == 0].any()
        canvas = R.canvas_image(gray, mask)
        tight, tm, trans = tight_canvas(gray, mask)
        coverage, selected, fallback = patch_selection(tm)
        arrays["canvas"].append(canvas)
        arrays["tight"].append(tight)
        selections.append(selected)
        for mode, im in (("canvas", canvas), ("tight", tight)):
            assert cv2.imwrite(str(OUT / "inputs" / mode / f"{row.image_id}.png"), im)
        for k, value in enumerate(coverage):
            cx, cy = (k % 14 + .5)*16-.5, (k//14 + .5)*16-.5
            mapping.append(dict(image_id=row.image_id, patch_index=k, coverage=float(value),
                selected=bool(selected[k]), fallback=fallback, canvas_center_x=cx, canvas_center_y=cy,
                native_center_x=(cx-trans["pad_x"]+.5)/trans["scale_x"]-.5+trans["x0"],
                native_center_y=(cy-trans["pad_y"]+.5)/trans["scale_y"]-.5+trans["y0"], **trans))
        qa.append(dict(image_id=row.image_id, width=gray.shape[1], height=gray.shape[0],
                       tight_mask_fraction=float(tm.mean()), selected_patches=int(selected.sum()), fallback=fallback,
                       image_sha256=R.sha256(L.OUT / "images/percentile" / f"{row.image_id}.png")))
    pd.DataFrame(mapping).to_csv(OUT / "patch_coordinate_mapping.csv", index=False)
    pd.DataFrame(qa).to_csv(OUT / "input_audit.csv", index=False)
    config = DINOv3ViTConfig.from_pretrained(DINO3, local_files_only=True)
    model, info = DINOv3ViTModel.from_pretrained(DINO3, config=config, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, output_loading_info=True, attn_implementation="eager")
    unmatched = {key: list(info.get(key, [])) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    assert not any(unmatched.values()), unmatched
    model = model.cuda().eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    processor = AutoImageProcessor.from_pretrained(DINO3, local_files_only=True, trust_remote_code=False)
    before = L.model_digest(model)
    provenance = dict(transformers=transformers.__version__, torch=torch.__version__, device=torch.cuda.get_device_name(),
        class_name=f"{type(model).__module__}.{type(model).__name__}", unmatched_keys=unmatched,
        implementation_sha256=R.sha256(Path(inspect.getfile(DINOv3ViTModel))), state_before=before,
        weight_sha256=R.sha256(DINO3 / "model.safetensors"), processor=processor.to_dict())
    with torch.inference_mode():
        x = processor(images=arrays["canvas"][:2], return_tensors="pt")["pixel_values"].cuda()
        a = model(pixel_values=x).last_hidden_state
        b = model(pixel_values=x).last_hidden_state
        c = model(pixel_values=x[:1]).last_hidden_state
        checks = dict(repeat_maxabs=float((a-b).abs().max()), batch_maxabs=float((a[:1]-c).abs().max()))
        assert max(checks.values()) <= 1e-4, checks
        provenance["inference_checks"] = checks
        for mode in arrays:
            cls, means, stds = [], [], []
            for start in range(0, len(df), 8):
                x = processor(images=arrays[mode][start:start+8], return_tensors="pt")["pixel_values"].cuda()
                tokens = model(pixel_values=x).last_hidden_state.cpu().numpy()
                assert tokens.shape[1:] == (201, 1280) and np.isfinite(tokens).all()
                cls.append(tokens[:, 0])
                if mode == "tight":
                    for j, t in enumerate(tokens):
                        selected = t[5:][selections[start+j]]
                        means.append(selected.mean(0))
                        stds.append(selected.std(0))
                if start % 64 == 0:
                    print(f"Native DINOv3 {mode}: {min(start+8,len(df))}/{len(df)}", flush=True)
            feature = np.concatenate(cls).astype(np.float32)
            save_features(f"dino_{mode}_cls", feature, df, patients)
            if mode == "tight":
                save_features("dino_tight_patch_mean_std", np.concatenate([means, stds], axis=1), df, patients)
                save_features("dino_tight_cls_patch_mean", np.concatenate([feature, means], axis=1), df, patients)
    provenance["state_after"] = L.model_digest(model)
    assert provenance["state_after"] == before
    R.write_json(OUT / "dino_provenance.json", provenance)
    del model
    torch.cuda.empty_cache()
    dense = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    dense_before = L.model_digest(dense)
    with torch.inference_mode():
        for mode in arrays:
            chunks = []
            for start in range(0, len(df), 32):
                x = torch.from_numpy(np.stack(arrays[mode][start:start+32]).transpose(0,3,1,2).copy()).cuda().float()/127.5-1
                chunks.append(dense(x).cpu().numpy())
            im = np.concatenate(chunks)
            save_features(f"dense_{mode}", im, df, patients)
            print(f"DenseNet {mode}: {im.shape}", flush=True)
    dense_after = L.model_digest(dense)
    assert dense_before == dense_after
    extracted = np.load(OUT / "features/dense_canvas.npy")
    inherited = np.load(OUT / "features/radimagenet_percentile.npy")
    delta = float(np.max(np.abs(extracted-inherited)))
    assert np.allclose(extracted, inherited, rtol=1e-4, atol=1e-4), delta
    R.write_json(OUT / "dense_provenance.json", dict(state_before=dense_before, state_after=dense_after,
        checkpoint=str(RADIMAGENET_DIR / "DenseNet121.pt"), weight_sha256=R.sha256(RADIMAGENET_DIR / "DenseNet121.pt"),
        inherited_cache_maxabs=delta, matched_backbone_keys=725, normalization="x/127.5-1"))
    print("Frozen state, input geometry, finite features and inherited DenseNet cache verified", flush=True)


def save_features(name, image_features, df, patients):
    image_features = np.asarray(image_features, np.float32)
    assert len(image_features) == 349 and np.isfinite(image_features).all()
    np.save(OUT / "features" / f"{name}_image.npy", image_features)
    np.save(OUT / "features" / f"{name}.npy", R.aggregate(image_features, df, patients))


def gate_decision(dev, configs=CONFIGS):
    avg = dev.groupby("group")[["auroc", "average_precision"]].mean()
    decisions = []
    for name, (_, _, comparator) in configs.items():
        if comparator is None:
            continue
        delta = avg.loc[name]-avg.loc[comparator]
        decisions.append(dict(group=name, comparator=comparator, delta_auroc=float(delta.auroc),
            delta_ap=float(delta.average_precision), eligible=bool((delta >= -.02-1e-12).all()),
            mean_score=float(avg.loc[name].mean())))
    eligible = sorted([d for d in decisions if d["eligible"]], key=lambda d: (-d["mean_score"], d["group"]))
    return dict(selection_role="development_validation", candidates=decisions,
                selected=eligible[0]["group"] if eligible else None)


def evaluate():
    require_lock()
    patients, split = read("patients.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(int)
    arrays = {g: np.load(OUT / "features" / f"{spec[0]}.npy") for g, spec in CONFIGS.items()}
    scores, metrics, predictions, heads = {}, [], [], []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            idx = indices_for(patients, split, seed)
            scores[seed] = {}
            for name, values in arrays.items():
                head = V.make_head("lr", seed, values.shape[1]) if CONFIGS[name][1] == "top32" else full_head(seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    head.fit(values[idx["train"]], y[idx["train"]])
                score = head.predict_proba(values)[:, list(head.classes_).index(1)]
                assert np.isfinite(score).all()
                path = OUT / "models" / f"{name}_{seed}.joblib"
                joblib.dump(head, path, compress=3)
                assert np.allclose(joblib.load(path).predict_proba(values)[:, 1], score, atol=1e-12)
                scores[seed][name] = score
                heads.append(dict(seed=seed, group=name, sha256=R.sha256(path),
                    train_patients=patients.patient_alias.iloc[idx["train"]].tolist(), dimension=int(values.shape[1])))
            scores[seed]["PAIR"] = (scores[seed]["RAD"]+scores[seed]["DN-A"])/2
            for name, score in scores[seed].items():
                metrics.append(dict(seed=seed, group=name, role="development_validation", **V.score_metrics(y[idx["validation"]], score[idx["validation"]])))
            print(f"Fixed LR heads completed: {seed}", flush=True)
    dev = pd.DataFrame(metrics)
    avg = dev.groupby("group")[["auroc", "average_precision"]].mean()
    assert np.allclose(avg.loc["DN-A"], [.7, .7730878609850607], atol=1e-10)
    assert np.allclose(avg.loc["PAIR"], [.7323232323232325, .792643331626937], atol=1e-8), avg.loc["PAIR"].to_dict()
    decision = gate_decision(dev)
    winner = decision["selected"]
    if winner:
        for seed in R.SEEDS:
            idx = indices_for(patients, split, seed)["validation"]
            scores[seed]["VOTE"] = (scores[seed]["RAD"]+scores[seed]["DN-A"]+scores[seed][winner])/3
            metrics.append(dict(seed=seed, group="VOTE", role="development_validation", **V.score_metrics(y[idx], scores[seed]["VOTE"][idx])))
        md = pd.DataFrame(metrics).set_index(["group", "seed"])
        delta = md.loc["VOTE", ["auroc", "average_precision"]]-md.loc["PAIR", ["auroc", "average_precision"]]
        wins = int((delta.gt(0).all(axis=1)).sum())
        decision["vote"] = dict(delta_auroc=float(delta.auroc.mean()), delta_ap=float(delta.average_precision.mean()),
            joint_wins=wins, promoted=bool(delta.auroc.mean() >= .03-1e-12 and delta.average_precision.mean() >= .02-1e-12 and wins >= 6))
    R.write_json(OUT / "development_decision.json", decision)
    for seed in R.SEEDS:
        idx = indices_for(patients, split, seed)
        for name, score in scores[seed].items():
            metrics.append(dict(seed=seed, group=name, role="exploratory_test", **V.score_metrics(y[idx["test"]], score[idx["test"]])))
            for role, label in (("validation", "development_validation"), ("test", "exploratory_test")):
                for i in idx[role]:
                    predictions.append(dict(seed=seed, group=name, role=label, patient_alias=patients.patient_alias.iloc[i], label=int(y[i]), score=float(score[i])))
    pd.DataFrame(metrics).to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    R.write_json(OUT / "head_provenance.json", heads)
    summary = summarize(pd.DataFrame(metrics))
    summary.to_csv(OUT / "summary_by_role.csv", index=False)
    print(summary[["role", "group", "auroc_mean", "average_precision_mean"]].to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


def summarize(metrics):
    rows = []
    for (role, group), frame in metrics.groupby(["role", "group"], sort=False):
        row = dict(role=role, group=group, draws=len(frame), joint_target_draws=int(((frame.auroc > .8)&(frame.average_precision > .8)).sum()))
        for name in ("auroc", "average_precision", "auprc_trapezoid", "sensitivity", "specificity", "brier", "log_loss"):
            for agg in ("mean", "std", "min", "max"):
                row[f"{name}_{agg}"] = float(getattr(frame[name], agg)())
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate", "all"))
    args = parser.parse_args()
    for stage in (("prepare", "encode", "evaluate") if args.stage == "all" else (args.stage,)):
        globals()[stage]()
