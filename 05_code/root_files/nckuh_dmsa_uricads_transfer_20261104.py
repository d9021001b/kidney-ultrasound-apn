"""Transfer physician-labeled URI-CADS renal phenotypes to NCKUH APN development."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch import nn

import nckuh_dmsa_fullcohort_fullframe_20261103 as base
import nckuh_dmsa_fullcohort_radimagenet_finetune_20261103 as radft
import nckuh_dmsa_hierarchical_paired_20261103 as hierarchy
import nckuh_dmsa_multiquery_mil_20261103 as cohort
import nckuh_dmsa_native_tile_probe as tile_probe
from nckuh_dmsa_cluster_ci import cluster_intervals


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "third_party/URI-CADS/bbdd"
PUBLIC_ZIP = ROOT / "third_party/URI-CADS/downloads/bbdd_kidney.zip"
PUBLIC_README = ROOT / "third_party/URI-CADS/downloads/uri-cad_2024_v01_readme.txt"
RAD_SOURCE = ROOT / "02_experiments/CYCH_feature_research_results_20260913/source"
PREFIX_CACHE = ROOT / "02_experiments/NCKUH_DMSA_fullcohort_radimagenet_finetune_20261103/prefix_maps_float16.npy"
USFM = ROOT / "02_experiments/NCKUH_DMSA_usfm_frozen_20261104/usfm_gap.npy"
RAD_TILE = ROOT / "02_experiments/NCKUH_DMSA_native_tile_probe_20261103/tile_features.npz"
CURRENT = ROOT / "02_experiments/NCKUH_DMSA_hierarchical_tile_usfm_20261104/development_predictions.csv"
OUT = ROOT / "02_experiments/NCKUH_DMSA_uricads_transfer_20261104"
MODEL_SEEDS = (20261104, 20261105, 20261106)
CHECKPOINT_EPOCHS = (5, 10, 20, 30)
LABEL_NAMES = (
    "pathological",
    "poor_corticomedullary_differentiation",
    "hyperechogenicity",
    "cyst",
    "prominent_pyramid",
    "hydronephrosis",
    "other_local_lesion",
)


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_label(path: Path) -> tuple[np.ndarray, int]:
    text = path.read_text(encoding="utf-8")
    diagnosis = int(re.search(r"Global diagnosis: (\d+)", text).group(1))
    global_pathology = tuple(
        map(int, re.search(r"Global pathologies: (\d+) (\d+)", text).groups())
    )
    view = int(re.search(r"View: (\d+)", text).group(1))
    local = {
        int(value)
        for value in re.findall(
            r"Bounding box \d+: \d+ \d+ \d+ \d+ (\d+)", text
        )
    }
    labels = np.asarray(
        [
            diagnosis,
            global_pathology[0],
            global_pathology[1],
            bool(local.intersection({2, 3})),
            4 in local,
            7 in local,
            bool(local.intersection({1, 5, 6, 8, 9})),
        ],
        dtype=np.float32,
    )
    return labels, view


class PathologyTail(nn.Module):
    def __init__(self, tail: nn.Module):
        super().__init__()
        self.tail = tail
        self.classifier = nn.Linear(1024, len(LABEL_NAMES))

    def forward(self, maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.tail(maps)
        return embedding, self.classifier(embedding)


def construct(seed: int) -> PathologyTail:
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    torch.manual_seed(seed)
    original = radimagenet_densenet121().cuda().eval()
    _, tail = radft.split_encoder(original)
    return PathologyTail(copy.deepcopy(tail)).cuda()


def keep_bn_fixed(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def feature_augment(maps: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    output = maps.clone()
    flips = torch.rand(len(output), generator=generator) < 0.5
    if flips.any():
        selected = flips.to(output.device)
        output[selected] = torch.flip(output[selected], dims=(-1,))
    keep = torch.rand((len(output), 1, 7, 7), generator=generator) >= 0.08
    output *= keep.to(output.device)
    noise = torch.randn(output.shape, generator=generator, dtype=output.dtype)
    return output + 0.015 * noise.to(output.device)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Preserve registered experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": (
            "URI-CADS supplies independent physician-consensus renal ultrasound labels for "
            "poor corticomedullary differentiation, hyperechogenicity, and local lesions; "
            "test whether this domain supervision improves DMSA-defined APN discrimination"
        ),
        "public_source": "URI-CADS DOI 10.21950/0E0CTF, CC BY-NC 4.0",
        "public_cohort": "1985 adult B-mode renal ultrasound clinical cases, one image per case",
        "initialization": (
            "Verified RadImageNet DenseNet-121; prefix through transition3 frozen; "
            "denseblock4 and a seven-label head adapted only on URI-CADS"
        ),
        "public_labels": list(LABEL_NAMES),
        "public_selection": (
            "Three fixed 80/20 public-data splits; select 5/10/20/30 epochs by validation "
            "mean of macro AUROC and AP, then refit each seed on all 1985 public images"
        ),
        "nckuh_inputs": (
            "Existing contour-first 820 NCKUH images; whole-kidney and four overlapping "
            "mask-constrained tiles; frozen transfer without NCKUH gradient updates"
        ),
        "nckuh_evaluation": (
            "Four-state patient model with training-only PCA/logistic heads on the same three "
            "126-patient training and 42-patient development roles"
        ),
        "locked_role": "Exploratory test remains unscored unless mean AUROC and AP both exceed 0.8",
    }
    write_json(OUT / "protocol.json", protocol)
    write_json(
        OUT / "source_audit.json",
        {
            "public_zip_sha256": digest(PUBLIC_ZIP),
            "public_readme_sha256": digest(PUBLIC_README),
            "public_images": len(list((PUBLIC / "images").glob("*.jpg"))),
            "public_masks": len(list((PUBLIC / "masks").glob("*.mat"))),
            "public_labels": len(list((PUBLIC / "labels").glob("*.txt"))),
        },
    )
    print(f"Registered {OUT}", flush=True)


def cache_public() -> None:
    target = OUT / "public_prefix_maps_float16.npy"
    if target.exists():
        raise FileExistsError(target)
    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    identifiers = sorted(int(path.stem) for path in (PUBLIC / "images").glob("*.jpg"))
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    prefix, _ = radft.split_encoder(model)
    maps, labels, manifest = [], [], []
    with torch.inference_mode():
        for start in range(0, len(identifiers), 32):
            canvases = []
            for identifier in identifiers[start : start + 32]:
                bgr = cv2.imread(str(PUBLIC / "images" / f"{identifier}.jpg"))
                mask = scipy.io.loadmat(PUBLIC / "masks" / f"{identifier}.mat")["mask"] > 0
                if bgr is None or bgr.shape[:2] != mask.shape or mask.sum() < 256:
                    raise ValueError(f"Invalid URI-CADS image or mask: {identifier}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                canvas, _ = base.tight_canvas(rgb, mask)
                canvases.append(canvas)
                current_labels, view = parse_label(PUBLIC / "labels" / f"{identifier}.txt")
                labels.append(current_labels)
                manifest.append({"public_id": identifier, "view": view, **dict(zip(LABEL_NAMES, current_labels.astype(int)))})
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float()
            maps.append(prefix(tensor / 127.5 - 1).cpu().numpy().astype(np.float16))
            print(f"URI-CADS cached {min(start + 32, len(identifiers))}/{len(identifiers)}", flush=True)
    maps_array = np.concatenate(maps)
    labels_array = np.stack(labels)
    if maps_array.shape != (1985, 512, 7, 7) or labels_array.shape != (1985, 7):
        raise ValueError("Unexpected URI-CADS cache geometry")
    np.save(target, maps_array)
    np.save(OUT / "public_labels.npy", labels_array)
    pd.DataFrame(manifest).to_csv(OUT / "public_manifest.csv", index=False)
    write_json(
        OUT / "public_cache_audit.json",
        {
            "maps_shape": list(maps_array.shape),
            "labels_shape": list(labels_array.shape),
            "positive_counts": dict(zip(LABEL_NAMES, labels_array.sum(0).astype(int).tolist())),
            "finite": bool(np.isfinite(maps_array).all()),
            "nckuh_labels_used": [],
        },
    )


def _train_model(
    seed: int,
    maps: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    epochs: int,
    validation: np.ndarray | None = None,
) -> tuple[PathologyTail, list[dict]]:
    model = construct(seed)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
            {"params": model.tail.parameters(), "lr": 2e-5, "weight_decay": 0.0},
        ]
    )
    selected_labels = labels[selected]
    positive = np.maximum(selected_labels.sum(0), 1)
    negative = len(selected_labels) - positive
    pos_weight = torch.from_numpy(np.clip(negative / positive, 1, 10).astype(np.float32)).cuda()
    all_labels = torch.from_numpy(labels.astype(np.float32)).cuda()
    generator = torch.Generator(device="cpu").manual_seed(seed + 500)
    rng = np.random.default_rng(seed + 600)
    records: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        keep_bn_fixed(model)
        order = selected[rng.permutation(len(selected))]
        for start in range(0, len(selected), 64):
            index = order[start : start + 64]
            tensor = torch.from_numpy(np.asarray(maps[index], dtype=np.float32)).cuda()
            tensor = feature_augment(tensor, generator)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(tensor)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, all_labels[index], pos_weight=pos_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if validation is not None and epoch in CHECKPOINT_EPOCHS:
            _, scores = _predict(model, maps, validation)
            aucs, aps = [], []
            for column in range(labels.shape[1]):
                truth = labels[validation, column]
                if len(np.unique(truth)) == 2:
                    aucs.append(roc_auc_score(truth, scores[:, column]))
                    aps.append(average_precision_score(truth, scores[:, column]))
            records.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "macro_auroc": float(np.mean(aucs)),
                    "macro_ap": float(np.mean(aps)),
                    "selection_score": float((np.mean(aucs) + np.mean(aps)) / 2),
                }
            )
    return model, records


def _predict(
    model: PathologyTail, maps: np.ndarray, selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, scores = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), 128):
            index = selected[start : start + 128]
            tensor = torch.from_numpy(np.asarray(maps[index], dtype=np.float32)).cuda()
            embedding, logits = model(tensor)
            embeddings.append(embedding.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(scores)


def train_public() -> None:
    maps = np.load(OUT / "public_prefix_maps_float16.npy", mmap_mode="r")
    labels = np.load(OUT / "public_labels.npy").astype(np.float32)
    diagnosis = labels[:, 0].astype(int)
    stratification = diagnosis * 4 + labels[:, 1].astype(int) * 2 + labels[:, 2].astype(int)
    selections = []
    all_indices = np.arange(len(labels))
    for seed in MODEL_SEEDS:
        train, validation = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=seed,
            stratify=stratification,
        )
        _, records = _train_model(seed, maps, labels, train, max(CHECKPOINT_EPOCHS), validation)
        selections.extend(records)
        chosen = max(records, key=lambda row: (row["selection_score"], -row["epoch"]))["epoch"]
        model, _ = _train_model(seed + 1000, maps, labels, all_indices, int(chosen))
        torch.save(
            {"state_dict": model.state_dict(), "seed": seed, "selected_epoch": int(chosen)},
            OUT / "models" / f"uricads_tail_{seed}.pt",
        )
        print(f"URI-CADS model seed={seed} selected_epoch={chosen}", flush=True)
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(selections)
    frame.to_csv(OUT / "public_validation_history.csv", index=False)
    write_json(
        OUT / "public_training_audit.json",
        {
            "model_seeds": list(MODEL_SEEDS),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "selected_epochs": {
                str(seed): int(
                    frame[frame.seed.eq(seed)].sort_values(
                        ["selection_score", "epoch"], ascending=[False, True]
                    ).iloc[0].epoch
                )
                for seed in MODEL_SEEDS
            },
            "nckuh_labels_used": [],
            "refit_public_rows": 1985,
        },
    )


def load_models() -> list[PathologyTail]:
    models = []
    for seed in MODEL_SEEDS:
        blob = torch.load(OUT / "models" / f"uricads_tail_{seed}.pt", map_location="cpu", weights_only=True)
        model = construct(seed + 1000)
        model.load_state_dict(blob["state_dict"])
        models.append(model.eval().requires_grad_(False))
    return models


def encode_nckuh() -> None:
    models = load_models()
    maps = np.load(PREFIX_CACHE, mmap_mode="r")
    full_embeddings, full_scores = [], []
    with torch.inference_mode():
        for start in range(0, len(maps), 64):
            tensor = torch.from_numpy(np.asarray(maps[start : start + 64], dtype=np.float32)).cuda()
            current_embeddings, current_scores = [], []
            for model in models:
                embedding, logits = model(tensor)
                current_embeddings.append(embedding)
                current_scores.append(torch.sigmoid(logits))
            full_embeddings.append(torch.stack(current_embeddings).mean(0).cpu().numpy())
            full_scores.append(torch.stack(current_scores).mean(0).cpu().numpy())
    full_embeddings = np.concatenate(full_embeddings)
    full_scores = np.concatenate(full_scores)

    sys.path.insert(0, str(RAD_SOURCE))
    from radimagenet_backbones import radimagenet_densenet121

    encoder = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    prefix, _ = radft.split_encoder(encoder)
    _, _, images = radft.load_reference()
    tile_embeddings, tile_scores, coverage = [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), 16):
            canvases, lengths = [], []
            for identifier in images.image_id.iloc[start : start + 16]:
                bgr = cv2.imread(str(base.OUT / "clean_rgb" / f"{identifier}.png"))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(base.OUT / "masks" / f"{identifier}.png"), 0) > 0
                current, fallback = tile_probe.tiles(rgb, mask)
                canvases.extend(current)
                lengths.append(len(current))
                coverage.append({"image_id": identifier, "tiles": len(current), "fallback": fallback})
            tensor = torch.from_numpy(np.stack(canvases).transpose(0, 3, 1, 2).copy()).cuda().float()
            prefix_maps = prefix(tensor / 127.5 - 1)
            model_embeddings, model_scores = [], []
            for model in models:
                embedding, logits = model(prefix_maps)
                model_embeddings.append(embedding)
                model_scores.append(torch.sigmoid(logits))
            embedding = torch.stack(model_embeddings).mean(0).cpu().numpy()
            scores = torch.stack(model_scores).mean(0).cpu().numpy()
            for image_embedding, image_score in zip(
                np.split(embedding, np.cumsum(lengths)[:-1]),
                np.split(scores, np.cumsum(lengths)[:-1]),
            ):
                tile_embeddings.append(image_embedding.mean(0))
                tile_scores.append(image_score.mean(0))
            print(f"NCKUH URI-CADS tiles encoded {min(start + 16, len(images))}/{len(images)}", flush=True)
    tile_embeddings = np.asarray(tile_embeddings)
    tile_scores = np.asarray(tile_scores)
    arrays = {
        "nckuh_uricads_full_embedding.npy": full_embeddings,
        "nckuh_uricads_full_scores.npy": full_scores,
        "nckuh_uricads_tile_embedding.npy": tile_embeddings,
        "nckuh_uricads_tile_scores.npy": tile_scores,
    }
    for name, values in arrays.items():
        if values.shape[0] != 820 or not np.isfinite(values).all():
            raise ValueError(f"Invalid transfer array {name}: {values.shape}")
        np.save(OUT / name, values.astype(np.float32))
    pd.DataFrame(coverage).to_csv(OUT / "nckuh_tile_coverage.csv", index=False)
    write_json(
        OUT / "nckuh_encoder_audit.json",
        {
            "images": 820,
            "models": len(models),
            "shapes": {name: list(values.shape) for name, values in arrays.items()},
            "total_tiles": int(sum(row["tiles"] for row in coverage)),
            "whole_kidney_fallbacks": int(sum(row["fallback"] for row in coverage)),
            "nckuh_labels_used_in_encoding": [],
        },
    )


def representations() -> dict[str, np.ndarray]:
    full_embedding = np.load(OUT / "nckuh_uricads_full_embedding.npy")
    tile_embedding = np.load(OUT / "nckuh_uricads_tile_embedding.npy")
    full_scores = np.load(OUT / "nckuh_uricads_full_scores.npy")
    tile_scores = np.load(OUT / "nckuh_uricads_tile_scores.npy")
    usfm = np.load(USFM)
    with np.load(RAD_TILE) as bundle:
        rad_tile = bundle["mean"]
    return {
        "uricads_full_embedding": full_embedding,
        "uricads_tile_embedding": tile_embedding,
        "uricads_full_tile_scores": np.c_[full_scores, tile_scores],
        "uricads_full_tile_embedding": np.c_[full_embedding, tile_embedding],
        "uricads_tile_usfm": np.c_[tile_embedding, usfm],
        "uricads_tile_rad_usfm": np.c_[tile_embedding, rad_tile, usfm],
    }


def evaluate() -> None:
    patients, kidneys, images = cohort.load_reference()
    values = representations()
    structured = {
        name: hierarchy.structured_views(base.patient_pair_features(array, images, patients))
        for name, array in values.items()
    }
    states = patients.DMSA_APN.map(hierarchy.STATE_CODE).to_numpy(int)
    labels = kidneys.kidney_APN.to_numpy(int)
    assignments = pd.read_csv(base.OUT / "patient_assignments.csv")
    current = pd.read_csv(CURRENT)
    current = current[current.group.eq("hierarchical_tile_usfm_quarter_plus_fixed")].set_index(
        ["seed", "kidney_alias"], verify_integrity=True
    )
    metrics, predictions, selections = [], [], []
    model_dir = OUT / "development_models"
    model_dir.mkdir(exist_ok=True)
    for seed in base.SEEDS:
        mapping = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role
        patient_roles = patients.patient_alias.map(mapping)
        kidney_roles = kidneys.patient_alias.map(mapping)
        train = np.flatnonzero(patient_roles.eq("train"))
        development = np.flatnonzero(kidney_roles.eq("development"))
        target = kidneys.iloc[development]
        current_rows = current.loc[[(seed, alias) for alias in target.kidney_alias]]
        np.testing.assert_array_equal(current_rows.label, labels[development])
        reference = current_rows.score.to_numpy()
        draw_scores = {"current_tile_usfm_reference": reference}
        for name, (symmetric, antisymmetric) in structured.items():
            probability, models = hierarchy.state_probabilities(
                symmetric, antisymmetric, states, train, seed
            )
            score = hierarchy.kidney_scores(probability, patients, kidneys)[development]
            draw_scores[name] = score
            draw_scores[f"{name}_quarter_plus_current"] = 0.25 * score + 0.75 * reference
            draw_scores[f"{name}_equal_plus_current"] = 0.50 * score + 0.50 * reference
            joblib.dump(models, model_dir / f"{seed}_{name}.joblib", compress=3)
            for head, model in models.items():
                selections.append(
                    {
                        "seed": seed,
                        "source": name,
                        "head": head,
                        "inner_cv_auroc": float(model.best_score_),
                        **model.best_params_,
                    }
                )
        for group, score in draw_scores.items():
            result = hierarchy.metric(labels[development], score)
            metrics.append(
                {
                    "seed": seed,
                    "role": "development",
                    "group": group,
                    "patients": target.patient_alias.nunique(),
                    "kidneys": len(target),
                    **result,
                }
            )
            for local, index in enumerate(development):
                predictions.append(
                    {
                        "seed": seed,
                        "group": group,
                        "patient_alias": kidneys.patient_alias.iloc[index],
                        "kidney_alias": kidneys.kidney_alias.iloc[index],
                        "DMSA_APN": kidneys.DMSA_APN.iloc[index],
                        "side": kidneys.side.iloc[index],
                        "label": int(labels[index]),
                        "score": float(score[local]),
                    }
                )
        print(f"URI-CADS transfer development draw {seed} complete", flush=True)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    summary = metric_frame.groupby("group")[["auroc", "ap", "brier"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.sort_values(["auroc_mean", "ap_mean"], ascending=False)
    metric_frame.to_csv(OUT / "development_metrics.csv", index=False)
    prediction_frame.to_csv(OUT / "development_predictions.csv", index=False)
    summary.to_csv(OUT / "development_summary.csv")
    pd.DataFrame(selections).to_csv(OUT / "training_inner_cv_selections.csv", index=False)
    groups = prediction_frame.group.drop_duplicates().tolist()
    ci, delta, samples = cluster_intervals(
        prediction_frame, groups, base.SEEDS, "current_tile_usfm_reference"
    )
    ci.to_csv(OUT / "development_ci.csv", index=False)
    delta.to_csv(OUT / "paired_delta_ci.csv", index=False)
    np.save(OUT / "development_patient_cluster_bootstrap.npy", samples)
    achieved = summary[(summary.auroc_mean > 0.8) & (summary.ap_mean > 0.8)].index.tolist()
    write_json(
        OUT / "decision.json",
        {
            "groups_above_both_development_targets": achieved,
            "exploratory_test_rows_scored": 0,
            "promotion": bool(achieved),
            "reference": "hierarchical_tile_usfm_quarter_plus_fixed",
        },
    )
    print(summary.to_string(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "cache_public", "train_public", "encode_nckuh", "evaluate", "all"),
    )
    args = parser.parse_args()
    stages = (prepare, cache_public, train_public, encode_nckuh, evaluate)
    if args.stage == "all":
        for stage in stages:
            stage()
    else:
        dict((stage.__name__, stage) for stage in stages)[args.stage]()


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
