"""Development-only frozen TinyUSFM feature screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import renal_local_features as L
import renal_random_validation as R
from renal_dinov3_native_probe import indices_for
from run_uniusnet_frozen_development import make_head, metric_row, read_csv
from run_usf_mae_development import sha256, write_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_tinyusfm_frozen_development_20260914"
MODEL_DIR = ROOT / "external_models" / "TinyUSFM_official_20260914"
CHECKPOINT = MODEL_DIR / "TinyUSFM.pth"
DINO_OUT = ROOT / "CYCH_dinov3_native_probe_20260913"
PAIR_PREDICTIONS = DINO_OUT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
CONFIGS = {
    "TINY-GAP": "tinyusfm_gap",
    "TINY-CLS": "tinyusfm_cls",
    "TINY-ROI-HET": "tinyusfm_roi_mean_std",
}


def patch_selections(manifest: pd.DataFrame) -> list[np.ndarray]:
    mapping = pd.read_csv(DINO_OUT / "patch_coordinate_mapping.csv")
    selections = []
    for image_id in manifest.image_id:
        frame = mapping.loc[mapping.image_id.eq(image_id)].sort_values("patch_index")
        if len(frame) != 196:
            raise RuntimeError(f"Missing locked patch mapping for {image_id}")
        selected = frame.selected.astype(bool).to_numpy()
        if not selected.any():
            raise RuntimeError(f"Empty locked patch selection for {image_id}")
        selections.append(selected)
    return selections


def require_lock() -> None:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))


def prepare() -> None:
    for name in ("features", "models", "source"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        source = DINO_OUT / name
        destination = OUT / name
        if destination.exists():
            assert sha256(destination) == sha256(source)
        else:
            shutil.copy2(source, destination)
    patients = read_csv(OUT / "patients.csv")
    manifest = read_csv(OUT / "image_manifest.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    assert len(patients) == 98 and int(patients.label.sum()) == 52
    assert len(manifest) == 349 and manifest.image_id.nunique() == 349
    for seed in R.SEEDS:
        indices_for(patients, assignments, seed)
    protocol = {
        "scope": "development-only frozen TinyUSFM screen; no exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "model": {
            "name": "TinyUSFM",
            "architecture": "official 5.5M-parameter ViT, patch size 16, 192-dimensional tokens",
            "source": "https://github.com/MacDunno/TinyUSFM",
            "paper": "https://doi.org/10.1109/JBHI.2026.3678309",
            "license": "Apache-2.0 repository license",
            "checkpoint_source": "https://drive.google.com/file/d/1P_wDuU-StALLJEBaymioaYu52CScW-l-/view",
            "checkpoint_sha256": sha256(CHECKPOINT),
            "declared_pretraining": "domain-separated masked image modeling and consistency-driven distillation for ultrasound",
        },
        "input": "Existing 224 by 224 tight kidney ROI after mandatory segmentation; RGB; official ImageNet mean and standard deviation",
        "representations": {
            "TINY-GAP": "mean of all 196 final normalized patch tokens, matching the official downstream global-pooling strategy",
            "TINY-CLS": "final normalized class token",
            "TINY-ROI-HET": "mean and standard deviation of final patch tokens selected by the locked kidney-mask map",
        },
        "compatibility": "The official subclass forward is incompatible with the installed newer timm signature; extraction calls the unchanged official patch embedding, blocks, and final norm directly",
        "patient_aggregation": "mean and standard deviation across first-examination images",
        "head": "training-only median imputation, constant removal, standardization, balanced L2 logistic regression C=0.1",
        "single_gate": "Both development means no more than 0.02 below tight-ROI RadImageNet DenseNet-121",
        "vote": "Only an eligible representation may form 0.60 fixed pair + 0.20 I3 + 0.20 TinyUSFM",
        "vote_gate": "Relative to fixed pair+I3: both mean metrics improve and at least 6/10 draws improve jointly",
        "reference_hashes": {
            "pair": sha256(PAIR_PREDICTIONS),
            "i3": sha256(I3_PREDICTIONS),
            "patch_mapping": sha256(DINO_OUT / "patch_coordinate_mapping.csv"),
        },
    }
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding="utf-8")) == protocol
    else:
        write_json(protocol_path, protocol)
    lock_path = OUT / "protocol_locked.json"
    if not lock_path.exists():
        write_json(
            lock_path,
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": sha256(protocol_path),
                "source_sha256": sha256(Path(__file__)),
            },
        )
    shutil.copy2(__file__, OUT / "source" / Path(__file__).name)
    print("TinyUSFM protocol locked before feature extraction", flush=True)


def aggregate_and_save(name: str, values: np.ndarray, manifest: pd.DataFrame, patients: pd.DataFrame) -> None:
    values = np.asarray(values, dtype=np.float32)
    assert values.shape[0] == 349 and np.isfinite(values).all()
    np.save(OUT / "features" / f"{name}_image.npy", values)
    patient_values = R.aggregate(values, manifest, patients).astype(np.float32)
    assert patient_values.shape[0] == 98 and np.isfinite(patient_values).all()
    np.save(OUT / "features" / f"{name}.npy", patient_values)


def all_tokens(model, images):
    import torch

    x = model.patch_embed(images)
    cls = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = model.pos_drop(x + model.pos_embed)
    for block in model.blocks:
        x = block(x)
    return model.norm(x)


def encode() -> None:
    require_lock()
    import torch

    sys.path.insert(0, str(MODEL_DIR))
    from model.tinyusfm import TinyUSFM

    torch.manual_seed(20260914)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    wrapper = TinyUSFM(num_classes=0, global_pool=False)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    loaded = wrapper.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys and len(state) == 150
    wrapper = wrapper.cuda().eval().requires_grad_(False)
    model = wrapper.model
    assert not any(parameter.requires_grad for parameter in wrapper.parameters())

    manifest = read_csv(OUT / "image_manifest.csv")
    patients = read_csv(OUT / "patients.csv")
    selected = patch_selections(manifest)
    images = []
    for image_id in manifest.image_id:
        image = cv2.imread(str(DINO_OUT / "inputs" / "tight" / f"{image_id}.png"), cv2.IMREAD_COLOR)
        assert image is not None and image.shape == (224, 224, 3)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def tensor(batch: list[np.ndarray]):
        value = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2).copy()).cuda().float() / 255.0
        return (value - mean) / std

    before = L.model_digest(wrapper)
    gap_values, cls_values, roi_values = [], [], []
    with torch.inference_mode():
        diagnostic = tensor(images[:2])
        first = all_tokens(model, diagnostic)
        repeat_maxabs = float((first - all_tokens(model, diagnostic)).abs().max())
        batch_maxabs = float((first[:1] - all_tokens(model, diagnostic[:1])).abs().max())
        assert max(repeat_maxabs, batch_maxabs) <= 1e-4
        for start in range(0, len(images), 64):
            tokens = all_tokens(model, tensor(images[start : start + 64])).cpu().numpy()
            assert tokens.shape[1:] == (197, 192) and np.isfinite(tokens).all()
            cls_values.append(tokens[:, 0])
            gap_values.append(tokens[:, 1:].mean(axis=1))
            for offset, token in enumerate(tokens):
                roi = token[1:][selected[start + offset]]
                roi_values.append(np.concatenate([roi.mean(axis=0), roi.std(axis=0)]))
            print(f"TinyUSFM features: {min(start + 64, len(images))}/{len(images)}", flush=True)
    after = L.model_digest(wrapper)
    assert before == after
    aggregate_and_save("tinyusfm_gap", np.concatenate(gap_values), manifest, patients)
    aggregate_and_save("tinyusfm_cls", np.concatenate(cls_values), manifest, patients)
    aggregate_and_save("tinyusfm_roi_mean_std", np.stack(roi_values), manifest, patients)
    write_json(
        OUT / "encoder_provenance.json",
        {
            "architecture": "official TinyUSFM ViT",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "loaded_tensors": len(state),
            "state_before": before,
            "state_after": after,
            "repeat_maxabs": repeat_maxabs,
            "batch_maxabs": batch_maxabs,
        },
    )
    print("Frozen TinyUSFM features saved", flush=True)


def load_reference(path: Path, group: str, aliases: list[str]) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame.role.eq("development_validation") & frame.group.eq(group),
        ["seed", "patient_alias", "score"],
    ]
    result = {}
    for seed in R.SEEDS:
        subset = frame.loc[frame.seed.eq(seed)].set_index("patient_alias")
        assert len(subset) == 20 and subset.index.is_unique
        result[seed] = subset.reindex(aliases).score.to_numpy(float)
    return result


def evaluate() -> None:
    require_lock()
    patients = read_csv(OUT / "patients.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    arrays = {name: np.load(OUT / "features" / f"{feature}.npy") for name, feature in CONFIGS.items()}
    arrays["DN-TIGHT"] = np.load(DINO_OUT / "features" / "dense_tight.npy")
    pair = load_reference(PAIR_PREDICTIONS, "PAIR", aliases)
    i3 = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)
    metrics = []
    scores_by_seed = {}
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            indices = indices_for(patients, assignments, seed)
            scores_by_seed[seed] = {}
            for group, values in arrays.items():
                head = make_head(seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    head.fit(values[indices["train"]], labels[indices["train"]])
                score = head.predict_proba(values)[:, 1]
                scores_by_seed[seed][group] = score
                joblib.dump(head, OUT / "models" / f"{group}_{seed}.joblib", compress=3)
            development = indices["validation"]
            for group, score in scores_by_seed[seed].items():
                metrics.append(metric_row(seed, group, labels[development], score[development]))
            incumbent = (2.0 * pair[seed] + i3[seed]) / 3.0
            metrics.append(metric_row(seed, "PAIR+I3-REFERENCE", labels[development], incumbent[development]))
            print(f"TinyUSFM development heads: {seed}", flush=True)
    frame = pd.DataFrame(metrics)
    means = frame.groupby("group")[["auroc", "average_precision"]].mean()
    decisions = []
    for group in CONFIGS:
        delta = means.loc[group] - means.loc["DN-TIGHT"]
        decisions.append(
            {
                "group": group,
                "delta_auroc_vs_dense": float(delta.auroc),
                "delta_ap_vs_dense": float(delta.average_precision),
                "eligible": bool((delta >= -0.02 - 1e-12).all()),
                "mean_score": float(means.loc[group].mean()),
            }
        )
    eligible = sorted(
        (item for item in decisions if item["eligible"]),
        key=lambda item: (-item["mean_score"], item["group"]),
    )
    selected = eligible[0]["group"] if eligible else None
    decision = {"selection_role": "development_validation", "candidates": decisions, "selected": selected}
    if selected:
        for seed in R.SEEDS:
            development = indices_for(patients, assignments, seed)["validation"]
            score = 0.60 * pair[seed] + 0.20 * i3[seed] + 0.20 * scores_by_seed[seed][selected]
            metrics.append(metric_row(seed, "PAIR60+I320+TINY20", labels[development], score[development]))
        frame = pd.DataFrame(metrics)
        wide = frame.loc[
            frame.group.isin(["PAIR+I3-REFERENCE", "PAIR60+I320+TINY20"])
        ].pivot(index="seed", columns="group", values=["auroc", "average_precision"])
        delta_auc = wide["auroc"]["PAIR60+I320+TINY20"] - wide["auroc"]["PAIR+I3-REFERENCE"]
        delta_ap = (
            wide["average_precision"]["PAIR60+I320+TINY20"]
            - wide["average_precision"]["PAIR+I3-REFERENCE"]
        )
        wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        decision["vote"] = {
            "delta_auroc": float(delta_auc.mean()),
            "delta_ap": float(delta_ap.mean()),
            "joint_wins": wins,
            "promoted": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and wins >= 6),
        }
    frame = pd.DataFrame(metrics)
    summary = frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(
            (
                (frame.loc[frame.group.eq(group), "auroc"] > 0.8)
                & (frame.loc[frame.group.eq(group), "average_precision"] > 0.8)
            ).sum()
        )
        for group in summary.group
    ]
    frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv", index=False)
    write_json(OUT / "development_decision.json", decision)
    lines = [
        "# Frozen TinyUSFM development screen",
        "",
        "No exploratory-test result was computed.",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected eligible representation: {selected or 'none'}.",
            f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "evaluate", "all"))
    args = parser.parse_args()
    stages = ("prepare", "encode", "evaluate") if args.stage == "all" else (args.stage,)
    for stage in stages:
        globals()[stage]()


if __name__ == "__main__":
    main()
