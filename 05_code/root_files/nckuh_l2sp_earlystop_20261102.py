"""Reuse existing RadImageNet L2-SP and patient-triplet code on NCKUH only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"))
import renal_l2sp_earlystop_probe as W
import renal_triplet_probe as T
from radimagenet_backbones import radimagenet_densenet121, RADIMAGENET_DIR
from nckuh_only_study_20261102 import OUT as BASE, FEATURES, SEEDS, digest, write_json, metric
from nckuh_representation_screen_20261102 import tight_canvas, load_metadata

OUT = ROOT / "02_experiments/NCKUH_l2sp_earlystop_20261102"
PILOT_SEEDS = SEEDS[:3]
CONFIGS = W.CONFIGS


def configure():
    W.MAX_EPOCHS, W.MIN_EPOCHS, W.PATIENCE = 80, 10, 10


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    motivation="Fixed-epoch USF-MAE LoRA overfit. Test convolutional medical pretraining with training-only early stopping and a pretrained-weight anchor instead of unrestricted adaptation",
                    sources=["https://proceedings.mlr.press/v80/li18a.html", "https://arxiv.org/abs/1503.03832", "https://doi.org/10.1148/ryai.210315"],
                    pretrained_weight_sha256=digest(RADIMAGENET_DIR / "DenseNet121.pt"),
                    reused_code_sha256={name:digest(Path(__file__).with_name(name)) for name in ["renal_l2sp_earlystop_probe.py", "renal_triplet_probe.py"]},
                    assignments_sha256=digest(BASE / "random_patient_assignments.csv"),
                    cohort="NCKUH264 only, original mandatory kidney masks, no mixed-hospital classifier, no external target disease checkpoint",
                    backbone="Frozen RadImageNet prefix through transition3, cache float32(1027,512,7,7); denseblock4 and norm5 affine adapted; all BatchNorm running buffers frozen",
                    configurations=CONFIGS, pilot_seeds=PILOT_SEEDS,
                    inner_selection="One stratified126/32 split inside each158 training patients, seed+4000; select epoch by inner32 unweighted patient BCE only, max80/min10/patience10; reset and refit all158 for selected epoch count",
                    objectives="Reuse existing patient-balanced CE and optional patient-centroid semi-hard triplet implementation, margin.2. Positive triplets require distinct same-class training patients, never guessed positive kidney patches",
                    optimization="Reuse Adam head lr.001, tail lr.00001. New-head weight decay.0001; unanchored tail decay.0001; anchored tail decay0 plus .0001*sum squared distance from original pretrained tail",
                    outputs="Two fixed readouts per configuration: direct patient-centroid classifier and training-fitted full regularized LR on128D image embeddings summarized by patient mean/SD",
                    evaluation="Outer development53 only; no exploratory-test scoring or outer-development epoch selection",
                    expansion_gate="Mean development AUROC>=.65 and AP>=.78 in3 pilot draws, and both metrics improve against bilateral_full in at least2/3 draws; goal remains AUROC/AP>.8 on properly evaluated data",
                    caveat="This is exploratory method adaptation using existing tested code, not a new named architecture or proof that every positive patient's image shows APN")
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json")))
    print("NCKUH L2-SP/early-stop controlled pilot fixed", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"] and digest(OUT / "protocol.json") == lock["protocol_sha256"]
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for name, value in protocol["reused_code_sha256"].items():
        assert digest(Path(__file__).with_name(name)) == value
    assert digest(BASE / "random_patient_assignments.csv") == protocol["assignments_sha256"]
    return protocol


def cache():
    protocol = require_lock()
    assert digest(RADIMAGENET_DIR / "DenseNet121.pt") == protocol["pretrained_weight_sha256"]
    T.configure_torch()
    manifest, _ = load_metadata()
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    prefix, tail = T.split_encoder(model)
    maps, embeddings = [], []
    source_state = W.L.model_digest(model)
    with torch.inference_mode():
        for start in range(0, len(manifest), 32):
            images = []
            for ident in manifest.image_id.iloc[start:start+32]:
                gray = cv2.imread(str(FEATURES / "percentile" / f"{ident}.png"), 0)
                mask = cv2.imread(str(FEATURES / "masks" / f"{ident}.png"), 0) > 0
                image, _ = tight_canvas(gray, mask)
                images.append(image)
            tensor = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).cuda().float()/127.5-1
            current = prefix(tensor)
            embedded = tail(current)
            if start == 0:
                error = float((model(tensor)-embedded).abs().max())
                assert error <= 1e-6
            maps.append(current.cpu().numpy()); embeddings.append(embedded.cpu().numpy())
    maps, embeddings = np.concatenate(maps), np.concatenate(embeddings)
    assert maps.shape == (1027,512,7,7)
    expected = np.load(FEATURES / "dense_tight_image.npy")
    assert np.allclose(embeddings, expected, rtol=1e-4, atol=1e-4)
    assert W.L.model_digest(model) == source_state
    np.save(OUT / "prefix_maps.npy", maps)
    write_json(OUT / "cache_audit.json", dict(source_checkpoint_unchanged=True, prefix_full_model_maxabs=error,
                                             existing_embedding_maxabs=float(np.abs(embeddings-expected).max()),
                                             prefix_cache_sha256=digest(OUT / "prefix_maps.npy")))
    print("NCKUH-only frozen convolutional prefix matches existing DenseNet embeddings", flush=True)


def pilot():
    require_lock(); configure(); T.configure_torch()
    manifest, patients = load_metadata()
    maps = torch.from_numpy(np.load(OUT / "prefix_maps.npy")).cuda()
    assignments = pd.read_csv(BASE / "random_patient_assignments.csv")
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    _, original_tail = T.split_encoder(model)
    original_hash = W.L.model_digest(original_tail)
    (OUT / "models").mkdir(exist_ok=True)
    metrics, predictions, training = [], [], []
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for group, settings in CONFIGS.items():
            for seed in PILOT_SEEDS:
                saved = OUT / f"result_{group}_{seed}.json"
                if saved.exists():
                    result = json.loads(saved.read_text(encoding="utf-8"))
                    metrics.extend(result["metrics"]); predictions.extend(result["predictions"]); training.append(result["training"])
                    continue
                roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
                train, dev = np.flatnonzero(roles.eq("train")), np.flatnonzero(roles.eq("development"))
                fit_local, inner_local = next(StratifiedShuffleSplit(n_splits=1, test_size=32, random_state=seed+4000).split(train, patients.label.to_numpy()[train]))
                fit, inner = train[fit_local], train[inner_local]
                assert len(fit) == 126 and len(inner) == 32 and not set(inner).intersection(dev)
                epoch, history, audit = W.run_inner_early_stopping(maps, manifest, patients, fit, inner, original_tail, seed, settings["objective"], settings["l2sp"])
                print(f"{group} seed{seed} training-only selection epoch{epoch}, inner run{len(history)} epochs", flush=True)
                head, tail, refit_history, refit_audit = W.refit_full_outer_train(maps, manifest, patients, train, original_tail, seed, settings["objective"], settings["l2sp"], epoch)
                embeddings = []
                with torch.inference_mode():
                    for start in range(0, len(manifest), 64):
                        embeddings.append(head(tail(maps[start:start+64])).cpu().numpy())
                embeddings = np.concatenate(embeddings)
                patient_features = np.stack([np.r_[embeddings[manifest.patient_alias.eq(alias)].mean(0), embeddings[manifest.patient_alias.eq(alias)].std(0)] for alias in patients.patient_alias])
                lr = W.N.full_head(seed)
                lr.fit(patient_features[train], patients.label.to_numpy()[train])
                lr_scores = lr.predict_proba(patient_features[dev])[:, 1]
                with torch.inference_mode():
                    rows, bags, _ = W.make_bags(manifest, patients, dev, maps.device)
                    vectors = torch.nn.functional.normalize(bags @ torch.from_numpy(embeddings[rows]).cuda(), dim=1)
                    direct_scores = torch.sigmoid(head.classifier(vectors)).flatten().cpu().numpy()
                current_metrics, current_predictions = [], []
                for suffix, values in [("direct", direct_scores), ("summary_LR", lr_scores)]:
                    name = group+"_"+suffix
                    current_metrics.append(dict(seed=seed, group=name, role="development", **metric(patients.label.to_numpy()[dev], values)))
                    for index, score in zip(dev, values):
                        current_predictions.append(dict(seed=seed, group=name, role="development", patient_alias=patients.patient_alias.iloc[index], label=int(patients.label.iloc[index]), score=float(score)))
                checkpoint = OUT / "models" / f"{group}_{seed}.pth"
                torch.save(dict(head=head.state_dict(), tail=tail.state_dict(), selected_epoch=epoch), checkpoint)
                joblib.dump(lr, OUT / "models" / f"{group}_{seed}_lr.joblib", compress=3)
                pd.DataFrame(history+refit_history).to_csv(OUT / f"history_{group}_{seed}.csv", index=False)
                train_record = dict(seed=seed, group=group, inner_patient_aliases=patients.patient_alias.iloc[inner].tolist(),
                                    audit=audit, refit_audit=refit_audit, original_tail_unchanged=W.L.model_digest(original_tail)==original_hash)
                assert train_record["original_tail_unchanged"]
                result = dict(metrics=current_metrics, predictions=current_predictions, training=train_record)
                write_json(saved, result)
                metrics.extend(current_metrics); predictions.extend(current_predictions); training.append(train_record)
                print(pd.DataFrame(current_metrics)[["group","auroc","ap"]].to_string(index=False), flush=True)
                del head, tail
                torch.cuda.empty_cache()
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "pilot_development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "pilot_development_predictions.csv", index=False)
    write_json(OUT / "training_audits.json", training)
    summary = frame.groupby("group")[["auroc","ap"]].mean()
    summary["mean_score"] = summary.mean(axis=1)
    summary = summary.sort_values("mean_score", ascending=False)
    summary.to_csv(OUT / "pilot_summary.csv")
    baseline = pd.read_csv(ROOT / "02_experiments/NCKUH_ultrasound_probe_20261102/development_metrics.csv")
    baseline = baseline[baseline.group.eq("bilateral_full") & baseline.seed.isin(PILOT_SEEDS)].set_index("seed")
    eligible = []
    for group in summary.index:
        delta = frame[frame.group.eq(group)].set_index("seed")[["auroc","ap"]]-baseline[["auroc","ap"]]
        if summary.loc[group,"auroc"] >= .65 and summary.loc[group,"ap"] >= .78 and delta.gt(0).all(axis=1).sum() >= 2:
            eligible.append(group)
    write_json(OUT / "pilot_decision.json", dict(selected=eligible[0] if eligible else None, eligible=eligible, test_scoring_performed=False, goal_achieved=False))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare","cache","pilot"])
    args = parser.parse_args()
    configure()
    globals()[args.stage]()
