"""Development-only frozen USF-MAE feature screen for the CYCH cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings

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
from renal_dinov3_native_probe import indices_for


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_usf_mae_development_20260914"
CHECKPOINT = ROOT / "external_models" / "USF-MAE_official_20260914" / "USF_MAE_100epoch.pth"
DINO_OUT = ROOT / "CYCH_dinov3_native_probe_20260913"
PAIR_PREDICTIONS = DINO_OUT / "patient_predictions_by_role.csv"
I3_PREDICTIONS = ROOT / "CYCH_directional_robustness_selection_20260914" / "development_predictions.csv"
CONFIGS = {
    "USF-CLS": "usf_tight_cls",
    "USF-LOCAL": "usf_tight_cls_patch_mean_std",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"patient_id": str})


def require_lock() -> dict:
    lock = json.loads((OUT / "protocol_locked.json").read_text(encoding="utf-8"))
    assert lock["protocol_sha256"] == sha256(OUT / "protocol.json")
    assert lock["source_sha256"] == sha256(Path(__file__))
    return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def make_head(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        VarianceThreshold(),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


def prepare() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    for name in ("features", "models", "source", "qa"):
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
        "scope": "development-only frozen-feature screen; no new exploratory-test evaluation",
        "cohort": {"patients": 98, "images": 349, "positive": 52, "negative": 46},
        "draws": {"seeds": R.SEEDS, "train": 58, "development": 20},
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": sha256(CHECKPOINT),
            "bytes": CHECKPOINT.stat().st_size,
            "source": "https://github.com/Yusufii9/USF-MAE",
            "download_id": "1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr",
            "declared_training": "OpenUS-46, 370000 ultrasound images, 100 epochs",
        },
        "encoder": "timm vit_base_patch16_224; all 150 encoder tensors loaded from the official MAE checkpoint; frozen",
        "input": "Existing 224x224 tight kidney ROI after mandatory segmentation; RGB replication; nearest resize already applied; ImageNet mean/std used by the official USF-MAE data loader",
        "representations": {
            "USF-CLS": "final normalized class token",
            "USF-LOCAL": "class token plus mean and standard deviation of final normalized patch tokens whose kidney-mask coverage is at least 0.8",
        },
        "patient_aggregation": "mean and standard deviation across all first-examination images",
        "head": "train-only median imputation, constant removal, standardization, balanced L2 logistic regression C=0.1",
        "single_encoder_gate": "Both mean development AUROC and AP must be no more than 0.02 below the matched tight-ROI RadImageNet DenseNet-121 control",
        "ensemble": "Only an eligible USF representation may form 0.60 fixed pair + 0.20 I3 + 0.20 USF",
        "ensemble_gate": "Compared with the fixed 2/3 pair + 1/3 I3 incumbent: both mean metrics improve and at least 6 of 10 draws improve jointly",
        "selection": "No labels are used for encoder training or image preprocessing. Development labels are used only by the fixed linear head and declared gates.",
        "source_inputs": {
            "pair_predictions_sha256": sha256(PAIR_PREDICTIONS),
            "i3_predictions_sha256": sha256(I3_PREDICTIONS),
            "patch_mapping_sha256": sha256(DINO_OUT / "patch_coordinate_mapping.csv"),
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
    print("USF-MAE protocol locked before feature extraction", flush=True)


def patch_selections(manifest: pd.DataFrame) -> list[np.ndarray]:
    mapping = pd.read_csv(DINO_OUT / "patch_coordinate_mapping.csv")
    selections = []
    for image_id in manifest.image_id:
        frame = mapping.loc[mapping.image_id.eq(image_id)].sort_values("patch_index")
        assert len(frame) == 196
        selected = frame.selected.astype(bool).to_numpy()
        assert selected.any()
        selections.append(selected)
    return selections


def aggregate_and_save(name: str, values: np.ndarray, manifest: pd.DataFrame, patients: pd.DataFrame) -> None:
    values = np.asarray(values, dtype=np.float32)
    assert values.shape[0] == 349 and np.isfinite(values).all()
    np.save(OUT / "features" / f"{name}_image.npy", values)
    patient_values = R.aggregate(values, manifest, patients).astype(np.float32)
    assert patient_values.shape[0] == 98 and np.isfinite(patient_values).all()
    np.save(OUT / "features" / f"{name}.npy", patient_values)


def encode() -> None:
    require_lock()
    import timm
    import torch

    torch.manual_seed(20260914)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model_keys = set(model.state_dict())
    encoder = {key: value for key, value in checkpoint.items() if key in model_keys}
    assert set(encoder) == model_keys and len(encoder) == 150
    load = model.load_state_dict(encoder, strict=True)
    assert not load.missing_keys and not load.unexpected_keys
    model = model.cuda().eval().requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())

    manifest = read_csv(OUT / "image_manifest.csv")
    patients = read_csv(OUT / "patients.csv")
    selected = patch_selections(manifest)
    images = []
    for image_id in manifest.image_id:
        path = DINO_OUT / "inputs" / "tight" / f"{image_id}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None and image.shape == (224, 224, 3)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def batch_tensor(batch: list[np.ndarray]):
        value = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2).copy()).cuda().float() / 255.0
        return (value - mean) / std

    before = L.model_digest(model)
    cls_values, local_values = [], []
    with torch.inference_mode():
        diagnostic = batch_tensor(images[:2])
        first = model.forward_features(diagnostic)
        second = model.forward_features(diagnostic)
        single = model.forward_features(diagnostic[:1])
        repeat_maxabs = float((first - second).abs().max())
        batch_maxabs = float((first[:1] - single).abs().max())
        assert max(repeat_maxabs, batch_maxabs) <= 1e-4

        for start in range(0, len(images), 16):
            tokens = model.forward_features(batch_tensor(images[start : start + 16])).cpu().numpy()
            assert tokens.shape[1:] == (197, 768) and np.isfinite(tokens).all()
            cls_values.append(tokens[:, 0])
            for offset, token in enumerate(tokens):
                patches = token[1:][selected[start + offset]]
                local_values.append(np.concatenate([token[0], patches.mean(0), patches.std(0)]))
            print(f"USF-MAE features: {min(start + 16, len(images))}/{len(images)}", flush=True)

    after = L.model_digest(model)
    assert before == after
    aggregate_and_save("usf_tight_cls", np.concatenate(cls_values), manifest, patients)
    aggregate_and_save("usf_tight_cls_patch_mean_std", np.stack(local_values), manifest, patients)
    write_json(
        OUT / "encoder_provenance.json",
        {
            "architecture": "timm.vit_base_patch16_224",
            "timm_version": timm.__version__,
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "checkpoint_tensors": len(checkpoint),
            "loaded_encoder_tensors": len(encoder),
            "state_before": before,
            "state_after": after,
            "repeat_maxabs": repeat_maxabs,
            "batch_maxabs": batch_maxabs,
        },
    )
    print("Frozen USF-MAE features and provenance saved", flush=True)


def load_reference(path: Path, group: str, aliases: list[str]) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame.role.eq("development_validation") & frame.group.eq(group),
        ["seed", "patient_alias", "label", "score"],
    ]
    result = {}
    for seed in R.SEEDS:
        subset = frame.loc[frame.seed.eq(seed)].set_index("patient_alias")
        assert len(subset) == 20 and subset.index.is_unique
        result[seed] = subset.reindex(aliases).score.to_numpy(float)
    return result


def metric_row(seed: int, group: str, labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"seed": seed, "group": group, **V.score_metrics(labels, scores)}


def evaluate() -> None:
    require_lock()
    patients = read_csv(OUT / "patients.csv")
    assignments = read_csv(OUT / "random_patient_assignments.csv")
    labels = patients.label.to_numpy(int)
    aliases = patients.patient_alias.tolist()
    arrays = {name: np.load(OUT / "features" / f"{feature}.npy") for name, feature in CONFIGS.items()}
    arrays["DN-TIGHT"] = np.load(DINO_OUT / "features" / "dense_tight.npy")
    pair_reference = load_reference(PAIR_PREDICTIONS, "PAIR", aliases)
    i3_reference = load_reference(I3_PREDICTIONS, "I3-ALL", aliases)

    metrics = []
    prediction_rows = []
    candidate_scores: dict[int, dict[str, np.ndarray]] = {}
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            indices = indices_for(patients, assignments, seed)
            development = indices["validation"]
            candidate_scores[seed] = {}
            for group, values in arrays.items():
                head = make_head(seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    head.fit(values[indices["train"]], labels[indices["train"]])
                scores = head.predict_proba(values)[:, 1]
                assert np.isfinite(scores).all()
                model_path = OUT / "models" / f"{group}_{seed}.joblib"
                joblib.dump(head, model_path, compress=3)
                assert np.allclose(joblib.load(model_path).predict_proba(values)[:, 1], scores, atol=1e-12)
                candidate_scores[seed][group] = scores
                metrics.append(metric_row(seed, group, labels[development], scores[development]))
                for index in development:
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "group": group,
                            "role": "development_validation",
                            "patient_alias": aliases[index],
                            "label": int(labels[index]),
                            "score": float(scores[index]),
                        }
                    )
            pair = pair_reference[seed]
            i3 = i3_reference[seed]
            assert np.isfinite(pair[development]).all() and np.isfinite(i3[development]).all()
            incumbent = (2.0 * pair + i3) / 3.0
            metrics.append(metric_row(seed, "PAIR+I3-REFERENCE", labels[development], incumbent[development]))
            print(f"Development heads complete: {seed}", flush=True)

    metric_frame = pd.DataFrame(metrics)
    means = metric_frame.groupby("group")[["auroc", "average_precision"]].mean()
    dense_mean = means.loc["DN-TIGHT"]
    candidates = []
    for group in CONFIGS:
        delta = means.loc[group] - dense_mean
        candidates.append(
            {
                "group": group,
                "delta_auroc_vs_dense": float(delta.auroc),
                "delta_ap_vs_dense": float(delta.average_precision),
                "eligible": bool((delta >= -0.02 - 1e-12).all()),
                "mean_score": float(means.loc[group].mean()),
            }
        )
    eligible = sorted((item for item in candidates if item["eligible"]), key=lambda item: (-item["mean_score"], item["group"]))
    selected = eligible[0]["group"] if eligible else None
    decision = {"selection_role": "development_validation", "single_encoder_candidates": candidates, "selected": selected}

    if selected:
        ensemble_rows = []
        for seed in R.SEEDS:
            indices = indices_for(patients, assignments, seed)["validation"]
            pair = pair_reference[seed]
            i3 = i3_reference[seed]
            usf = candidate_scores[seed][selected]
            ensemble = 0.60 * pair + 0.20 * i3 + 0.20 * usf
            row = metric_row(seed, "PAIR60+I320+USF20", labels[indices], ensemble[indices])
            metrics.append(row)
            ensemble_rows.append(row)
            for index in indices:
                prediction_rows.append(
                    {
                        "seed": seed,
                        "group": "PAIR60+I320+USF20",
                        "role": "development_validation",
                        "patient_alias": aliases[index],
                        "label": int(labels[index]),
                        "score": float(ensemble[index]),
                    }
                )
        metric_frame = pd.DataFrame(metrics)
        paired = metric_frame.loc[metric_frame.group.isin(["PAIR+I3-REFERENCE", "PAIR60+I320+USF20"])]
        wide = paired.pivot(index="seed", columns="group", values=["auroc", "average_precision"])
        delta_auc = wide["auroc"]["PAIR60+I320+USF20"] - wide["auroc"]["PAIR+I3-REFERENCE"]
        delta_ap = wide["average_precision"]["PAIR60+I320+USF20"] - wide["average_precision"]["PAIR+I3-REFERENCE"]
        joint_wins = int(((delta_auc > 0) & (delta_ap > 0)).sum())
        decision["ensemble"] = {
            "delta_auroc": float(delta_auc.mean()),
            "delta_ap": float(delta_ap.mean()),
            "joint_wins": joint_wins,
            "promoted": bool(delta_auc.mean() > 0 and delta_ap.mean() > 0 and joint_wins >= 6),
        }

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(OUT / "development_metrics_per_draw.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "development_predictions.csv", index=False)
    summary = metric_frame.groupby("group", as_index=False).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        mean_average_precision=("average_precision", "mean"),
        std_average_precision=("average_precision", "std"),
    )
    summary["joint_target_draws"] = [
        int(((metric_frame.loc[metric_frame.group.eq(group), "auroc"] > 0.8) &
             (metric_frame.loc[metric_frame.group.eq(group), "average_precision"] > 0.8)).sum())
        for group in summary.group
    ]
    summary.to_csv(OUT / "development_summary.csv", index=False)
    write_json(OUT / "development_decision.json", decision)

    lines = [
        "# Frozen USF-MAE development-only screen",
        "",
        "The official 100-epoch USF-MAE ViT-Base/16 checkpoint was evaluated as a frozen encoder on segmented kidney ROIs.",
        "No new exploratory-test result was computed.",
        "",
        "## Development results",
        "",
        "| Group | Mean AUROC | Mean AP | Both >0.8 draws |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.group} | {row.mean_auroc:.3f} | {row.mean_average_precision:.3f} | {row.joint_target_draws}/10 |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Selected eligible representation: {selected or 'none'}.",
        f"Decision record: `{json.dumps(decision, separators=(',', ':'))}`",
        "",
        "The model source is https://github.com/Yusufii9/USF-MAE and the publication DOI is https://doi.org/10.1016/j.bspc.2026.110313.",
    ])
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
