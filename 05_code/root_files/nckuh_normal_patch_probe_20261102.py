"""PatchCore-inspired APN-negative patch reference, not lesion ground truth."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TQDM_DISABLE", "1")
import cv2
import joblib
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import ROOT, OUT as BASE, FEATURES, SEEDS, digest, write_json
from nckuh_representation_screen_20261102 import DINO, load_metadata, tight_canvas
from nckuh_lora_bilateral_pilot_20261102 import OUT as CACHE, original_encoder, WEIGHT

OUT = ROOT / "02_experiments/NCKUH_normal_patch_probe_20261102"
PATCHCORE = ROOT / "03_data_models/third_party/patchcore-inspection"
ENCODERS = ["dinov3", "usf_mae"]
AGGREGATES = ["maximum", "q99", "top10_mean"]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    hypothesis="Patient and image means can obscure focal appearance. Distances to same-view APN-negative training patches may preserve unusual local texture without inventing positive patch labels",
                    method_source="https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html",
                    source_code="https://github.com/amazon-science/patchcore-inspection",
                    sampler_sha256=digest(PATCHCORE / "src/patchcore/sampler.py"),
                    source_commit="fcaa92f124fb1ad74a7acf56726decd4b27cbcad", source_sampler_git_blob="09f57b50f92fb402838e31e101d52f2e872f48a5",
                    source_checkout_note="Sampler and package sources match Git commit. Optional Git LFS demonstration assets were not downloaded; no source model or data asset from that repository is required",
                    encoders=ENCODERS, patch_selection="Within unchanged kidney mask, at least80 percent coverage of a 16x16 patch. Existing original-transform selectors reused and hash-recorded",
                    patch_selector_sha256=digest(CACHE / "patch_selection.npy"),
                    pretrained_weights=dict(dinov3=digest(DINO / "model.safetensors"), usf_mae=digest(WEIGHT)),
                    adaptation="Use frozen transformer tokens, L2 normalize and fit PCA64 only on patches from APN-negative training patients. The official approximate greedy sampler retains5 percent per LL/LT/RL/RT memory bank. Exact torch.cdist nearest-neighbor distances to same-view reference",
                    reference_semantics="APN-negative does not mean anatomically healthy. No positive patient patch is asserted to contain APN. Heatmaps are model deviations, not scars or clinician-confirmed lesions",
                    deviations_from_original_PatchCore="Medical transformers instead of WRN feature maps, negative-only training PCA64, same-view reference banks, no paper-specific neighbor reweighting, patient pooling. This is a documented adaptation, not an exact reproduction or a newly named model",
                    patient_aggregates=AGGREGATES, candidates=6,
                    evaluation="Only development patients in ten fixed NCKUH-only draws. Raw distance scores are not calibrated APN probabilities. AUROC/AP/AUPRC only, no Brier or arbitrary0.5 operating point",
                    promotion="Both mean metrics improve over bilateral_full by >=.03 AUROC and >=.02 AP, with joint improvement in >=6/10 draws. Goal remains AUROC and AP>.8; no model chosen using test results")
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=digest(__file__), protocol_sha256=digest(OUT / "protocol.json")))
    print("Six normal-reference local patch configurations fixed; no test scoring", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(__file__) == lock["source_sha256"] and digest(OUT / "protocol.json") == lock["protocol_sha256"]
    return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def encode():
    protocol = require_lock()
    import torch
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, AutoImageProcessor
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, _ = load_metadata()
    selection = np.load(CACHE / "patch_selection.npy")[0]
    assert selection.shape == (1027, 196) and selection.any(1).all()
    assert digest(CACHE / "patch_selection.npy") == protocol["patch_selector_sha256"]
    offsets = np.r_[0, selection.sum(1).cumsum()]
    np.save(OUT / "patch_offsets.npy", offsets)
    provenance = {}
    for name in ENCODERS:
        output = OUT / f"{name}_roi_patches.npy"
        if output.exists():
            raise FileExistsError("Preserve an existing patch cache")
        width = 1280 if name == "dinov3" else 768
        values = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(int(offsets[-1]), width))
        if name == "dinov3":
            assert digest(DINO / "model.safetensors") == protocol["pretrained_weights"][name]
            model, info = DINOv3ViTModel.from_pretrained(DINO, config=DINOv3ViTConfig.from_pretrained(DINO, local_files_only=True),
                trust_remote_code=False, local_files_only=True, dtype=torch.float32, output_loading_info=True, attn_implementation="sdpa")
            assert not any(info.get(k, []) for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
            model = model.cuda().eval().requires_grad_(False)
            processor = AutoImageProcessor.from_pretrained(DINO, local_files_only=True, trust_remote_code=False)
            batch_size = 4
        else:
            assert digest(WEIGHT) == protocol["pretrained_weights"][name]
            model = original_encoder()
            lower = np.load(CACHE / "lower8_float32.npy", mmap_mode="r")
            batch_size = 32
        check_mean, check_sd = [], []
        with torch.inference_mode():
            for start in range(0, len(manifest), batch_size):
                stop = min(start+batch_size, len(manifest))
                if name == "dinov3":
                    images = []
                    for ident in manifest.image_id.iloc[start:stop]:
                        gray = cv2.imread(str(FEATURES / "percentile" / f"{ident}.png"), 0)
                        mask = cv2.imread(str(FEATURES / "masks" / f"{ident}.png"), 0) > 0
                        image, binary = tight_canvas(gray, mask)
                        coverage = binary.reshape(14, 16, 14, 16).mean((1, 3)).ravel() >= .8
                        if not coverage.any():
                            coverage = selection[start+len(images)]
                        assert np.array_equal(coverage, selection[start+len(images)])
                        images.append(image)
                    tensor = processor(images=images, return_tensors="pt", do_resize=False)["pixel_values"].cuda()
                    tokens = model(pixel_values=tensor).last_hidden_state[:, 5:].cpu().numpy()
                else:
                    tokens = torch.from_numpy(np.array(lower[0, start:stop])).cuda()
                    for block in model.blocks[8:]:
                        tokens = block(tokens)
                    tokens = model.norm(tokens)[:, 1:].cpu().numpy()
                for j in range(stop-start):
                    index = start+j
                    patches = tokens[j][selection[index]]
                    values[offsets[index]:offsets[index+1]] = patches
                    check_mean.append(patches.mean(0)); check_sd.append(patches.std(0))
                if start % 128 == 0:
                    print(f"{name} kidney covered patch cache {stop}/{len(manifest)}", flush=True)
        values.flush(); del values
        if name == "dinov3":
            expected = np.load(ROOT / "02_experiments/NCKUH_representation_screen_20261102/dino_patch_image.npy")
            actual = np.c_[check_mean, check_sd]
        else:
            expected = np.load(ROOT / "02_experiments/NCKUH_ultrasound_probe_20261102/usf_local_image.npy")[:, 768:]
            actual = np.c_[check_mean, check_sd]
        error = float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), error
        provenance[name] = dict(patches=int(offsets[-1]), dimension=width, prior_aggregation_maxabs=error,
                                frozen=True, patch_cache_sha256=digest(output))
        del model
        torch.cuda.empty_cache()
    write_json(OUT / "encoder_provenance.json", provenance)


def pool_score(distances, method):
    if method == "maximum":
        return float(np.max(distances))
    if method == "q99":
        return float(np.quantile(distances, .99))
    count = max(1, int(np.ceil(.1*len(distances))))
    return float(np.partition(distances, len(distances)-count)[-count:].mean())


def score():
    protocol = require_lock()
    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    from threadpoolctl import threadpool_limits
    assert digest(PATCHCORE / "src/patchcore/sampler.py") == protocol["sampler_sha256"]
    sys.path.insert(0, str(PATCHCORE / "src"))
    from patchcore.sampler import ApproximateGreedyCoresetSampler
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest, patients = load_metadata()
    offsets = np.load(OUT / "patch_offsets.npy")
    assignments = pd.read_csv(BASE / "random_patient_assignments.csv")
    image_index = np.repeat(np.arange(len(manifest)), np.diff(offsets))
    image_views = manifest["view"].to_numpy()
    patch_views = image_views[image_index]
    metrics, predictions, audits = [], [], []
    (OUT / "models").mkdir(exist_ok=True)
    with threadpool_limits(limits=4), torch.inference_mode():
        for name in ENCODERS:
            source = np.load(OUT / f"{name}_roi_patches.npy", mmap_mode="r")
            for seed in SEEDS:
                roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
                negative = patients.loc[roles.eq("train").to_numpy() & patients.label.eq(0).to_numpy(), "patient_id"]
                dev = np.flatnonzero(roles.eq("development"))
                train_images = manifest.patient_id.isin(negative).to_numpy()
                train_patch = train_images[image_index]
                train = np.asarray(source[train_patch])
                train = train/np.linalg.norm(train, axis=1, keepdims=True).clip(1e-8)
                pca = PCA(64, random_state=seed).fit(train)
                reduced_train = pca.transform(train).astype(np.float32)
                train_views = patch_views[train_patch]
                memory = {}
                torch.manual_seed(seed); np.random.seed(seed)
                for view in ["LL", "LT", "RL", "RT"]:
                    selected = reduced_train[train_views == view]
                    assert len(selected) >= 20
                    sampler = ApproximateGreedyCoresetSampler(.05, torch.device("cuda"), dimension_to_project_features_to=64)
                    memory[view] = sampler.run(selected)
                scores = {method: [] for method in AGGREGATES}
                for patient_index in dev:
                    rows = np.flatnonzero(manifest.patient_id.eq(patients.patient_id.iloc[patient_index]))
                    patient_distances = []
                    for row in rows:
                        patches = np.asarray(source[offsets[row]:offsets[row+1]])
                        patches = patches/np.linalg.norm(patches, axis=1, keepdims=True).clip(1e-8)
                        query = torch.from_numpy(pca.transform(patches).astype(np.float32)).cuda()
                        bank = torch.from_numpy(memory[image_views[row]]).cuda()
                        patient_distances.extend(torch.cdist(query, bank).min(1).values.cpu().tolist())
                    for method in AGGREGATES:
                        scores[method].append(pool_score(np.asarray(patient_distances), method))
                y = patients.label.to_numpy()[dev]
                for method in AGGREGATES:
                    values = np.asarray(scores[method])
                    precision, recall, _ = precision_recall_curve(y, values)
                    group = name+"_"+method
                    metrics.append(dict(seed=seed, group=group, role="development", auroc=roc_auc_score(y, values),
                                        ap=average_precision_score(y, values), auprc=abs(np.trapezoid(precision, recall))))
                    for index, value in zip(dev, values):
                        predictions.append(dict(seed=seed, group=group, role="development", patient_alias=patients.patient_alias.iloc[index], label=int(patients.label.iloc[index]), score=float(value)))
                joblib.dump(dict(pca=pca, reference_memory=memory), OUT / "models" / f"{name}_{seed}.joblib", compress=3)
                audits.append(dict(encoder=name, seed=seed, reference_patients=len(negative),
                                    reference_patches=int(train_patch.sum()), memory_sizes={v:len(m) for v,m in memory.items()},
                                    only_negative_training_patients=True, evaluated_role="development"))
                print(f"{name} APN-negative reference completed seed {seed}", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "development_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "development_predictions.csv", index=False)
    write_json(OUT / "reference_audit.json", audits)
    summary = frame.groupby("group")[["auroc", "ap"]].mean()
    summary["mean_score"] = summary.mean(axis=1)
    summary = summary.sort_values("mean_score", ascending=False)
    summary.to_csv(OUT / "development_summary.csv")
    incumbent = pd.read_csv(ROOT / "02_experiments/NCKUH_ultrasound_probe_20261102/development_metrics.csv")
    incumbent = incumbent[incumbent.group.eq("bilateral_full")].set_index("seed")
    eligible = []
    for group in summary.index:
        delta = frame[frame.group.eq(group)].set_index("seed")[["auroc", "ap"]]-incumbent[["auroc", "ap"]]
        if delta.auroc.mean() >= .03 and delta.ap.mean() >= .02 and delta.gt(0).all(axis=1).sum() >= 6:
            eligible.append(group)
    write_json(OUT / "candidate_decision.json", dict(selected=eligible[0] if eligible else None, eligible=eligible,
                                                     test_scoring_performed=False, goal_achieved=False))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "encode", "score"])
    arguments = parser.parse_args()
    globals()[arguments.stage]()
