"""Audit paired NCKUH source, image, and feature continuity without exporting IDs."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "02_experiments"
OLD = EXP / "NCKUH_DMSA_fullcohort_fullframe_20261103"
NEW = EXP / "NCKUH_revised_383_audit_20261111"
CACHE = EXP / "NCKUH_revised_383_contour_cache_20261111"
OUT = EXP / "NCKUH_revised_383_root_cause_20261112"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.digest()


def performance(rows):
    labels = np.array([int(row["label"]) for row in rows])
    scores = np.array([float(row["score"]) for row in rows])
    return {
        "patients": len({row["patient_alias"] for row in rows}),
        "kidneys": len(rows),
        "positive_fraction": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
    }


def paired_vectors(old, new, matches):
    distances, similarities = [], []
    for old_index, new_index in matches:
        a = old[old_index].ravel().astype(np.float64)
        b = new[new_index].ravel().astype(np.float64)
        distances.append(float(np.mean(np.abs(a - b))))
        similarities.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return {
        "pairs": len(matches),
        "mean_absolute_difference_quantiles": np.quantile(
            distances, [0, 0.5, 0.9, 0.99, 1]
        ).tolist(),
        "cosine_similarity_quantiles": np.quantile(
            similarities, [0, 0.01, 0.1, 0.5, 1]
        ).tolist(),
    }


def paired_model_delta(old_rows, new_rows, replicates=3000):
    newer = {row["kidney_alias"]: row for row in new_rows}
    assert len(old_rows) == len(newer)
    clusters = defaultdict(list)
    for row in old_rows:
        current = newer[row["kidney_alias"]]
        assert row["patient_alias"] == current["patient_alias"]
        assert row["kidney_APN"] == current["label"]
        clusters[row["patient_alias"]].append((
            int(row["kidney_APN"]), float(row["score"]), float(current["score"])
        ))
    groups = list(clusters.values())
    rng = np.random.default_rng(20261112)
    deltas = []
    for _ in range(replicates):
        sample = np.asarray([
            value for index in rng.integers(0, len(groups), len(groups))
            for value in groups[index]
        ])
        labels = sample[:, 0]
        if np.unique(labels).size < 2:
            continue
        deltas.append((
            roc_auc_score(labels, sample[:, 2]) - roc_auc_score(labels, sample[:, 1]),
            average_precision_score(labels, sample[:, 2])
            - average_precision_score(labels, sample[:, 1]),
        ))
    interval = np.quantile(np.asarray(deltas), [0.025, 0.975], axis=0)
    original = performance([dict(row, label=row["kidney_APN"]) for row in old_rows])
    revised = performance(new_rows)
    return {
        "patients": len(groups),
        "valid_bootstraps": len(deltas),
        "new_minus_old_auroc": revised["auroc"] - original["auroc"],
        "new_minus_old_auroc_95pct": interval[:, 0].tolist(),
        "new_minus_old_ap": revised["ap"] - original["ap"],
        "new_minus_old_ap_95pct": interval[:, 1].tolist(),
        "caveat": "Both models and this development set were explored; interval is conditional, not independent confirmation.",
    }


def main():
    old_patients = read_csv(EXP / "NCKUH_DMSA_label_audit_20261102/patient_label_linkage_private.csv")
    old_by_id = {row["patient_id"]: row for row in old_patients}
    new_link = read_csv(NEW / "patient_linkage_private.csv")
    new_assign = {row["patient_alias"]: row for row in read_csv(NEW / "patient_assignments.csv")}
    assert len(new_assign) == 383 and len(new_link) == 383

    label_counts = Counter()
    old_labeled_aliases = set()
    old_unlabeled_aliases = set()
    new_registry_aliases = set()
    for row in new_link:
        old = old_by_id.get(row["patient_id"])
        current = new_assign[row["patient_alias"]]
        if old is None:
            label_counts["not_in_old_registry"] += 1
            new_registry_aliases.add(row["patient_alias"])
        elif not old["DMSA_APN"]:
            label_counts["old_unlabeled"] += 1
            old_unlabeled_aliases.add(row["patient_alias"])
        elif old["DMSA_APN"] == current["DMSA_APN"]:
            label_counts["labeled_and_same"] += 1
            old_labeled_aliases.add(row["patient_alias"])
        else:
            label_counts["labeled_but_different"] += 1

    old_images = read_csv(EXP / "NCKUH_DMSA_label_audit_20261102/image_labels_dmsa_private.csv")
    old_ids = [row["image_id"] for row in read_csv(OLD / "deidentified_source_manifest.csv")]
    new_images = read_csv(NEW / "contour_image_manifest.csv")
    new_ids = [row["image_id"] for row in new_images]
    new_by_image = {row["image_id"]: row for row in new_images}
    old_pos = {key: index for index, key in enumerate(old_ids)}
    new_pos = {key: index for index, key in enumerate(new_ids)}
    new_index = defaultdict(list)
    for row in read_csv(NEW / "source_linkage_private.csv"):
        image = new_by_image.get(row["image_id"])
        if image is not None:
            key = (row["patient_id"], row["exam_date_raw"], image["view"])
            new_index[key].append(row["image_id"])

    image_counts = Counter()
    matches = []
    for row in old_images:
        if row["image_id"] not in old_pos:
            continue
        key = (row["patient_id"], row["exam_date_raw"], row["view"])
        options = new_index[key]
        if len(options) != 1:
            image_counts["unmatched_or_ambiguous"] += 1
            continue
        old_id, new_id = row["image_id"], options[0]
        matches.append((old_pos[old_id], new_pos[new_id]))
        image_counts["matched"] += 1
        for folder in ("clean_rgb", "masks", "percentile"):
            a = OLD / folder / f"{old_id}.png"
            b = CACHE / folder / f"{new_id}.png"
            if not a.exists() or not b.exists():
                image_counts[f"{folder}_missing"] += 1
            elif digest(a) == digest(b):
                image_counts[f"{folder}_identical"] += 1
            else:
                image_counts[f"{folder}_different"] += 1
    assert len(old_ids) == 820 and sum(image_counts[k] for k in ("matched", "unmatched_or_ambiguous")) == 820

    old_embeddings = np.load(
        EXP / "NCKUH_DMSA_uricads_official_frozen_20261105/official_fold_embeddings.npy",
        mmap_mode="r",
    )
    new_embeddings = np.load(CACHE / "uricads_fold_embeddings.npy", mmap_mode="r")
    assert old_embeddings.shape == (820, 5, 2048)
    assert new_embeddings.shape == (1362, 5, 2048)
    old_tile = np.load(EXP / "NCKUH_DMSA_native_tile_probe_20261103/tile_features.npz")["mean"]
    old_usfm = np.load(EXP / "NCKUH_DMSA_usfm_frozen_20261104/usfm_gap.npy")
    new_tile = np.load(EXP / "NCKUH_revised_383_legacy_features_20261112/tile_mean.npy")
    new_usfm = np.load(EXP / "NCKUH_revised_383_legacy_features_20261112/usfm_gap.npy")

    new_predictions = read_csv(
        EXP / "NCKUH_revised_383_oof_hierarchy_stacking_20261112/development_predictions.csv"
    )
    old_reference = [
        row for row in read_csv(
            EXP / "NCKUH_DMSA_uri_lightgbm_factorized_20261107/development_metrics.csv"
        ) if row["group"] == "official_uri_fixed_reference"
    ]
    result = {
        "label_overlap": dict(label_counts),
        "paired_image_files": dict(image_counts),
        "paired_uri_embeddings": paired_vectors(old_embeddings, new_embeddings, matches),
        "paired_tile_embeddings": paired_vectors(old_tile, new_tile, matches),
        "paired_usfm_embeddings": paired_vectors(old_usfm, new_usfm, matches),
        "old_reference_three_development_draws": [
            {key: row[key] for key in ("seed", "patients", "kidneys", "auroc", "ap")}
            for row in old_reference
        ],
        "new_development": {
            "all": performance(new_predictions),
            "old_210_overlap": performance([
                row for row in new_predictions if row["patient_alias"] in old_labeled_aliases
            ]),
            "outside_old_210": performance([
                row for row in new_predictions if row["patient_alias"] not in old_labeled_aliases
            ]),
        },
        "interpretation_limit": "Different models, training roles, and development patients; not a paired performance comparison.",
    }
    legacy = read_csv(OUT / "legacy_hierarchy_development_predictions.csv")
    result["legacy_tile_usfm_development"] = {
        "all": performance([dict(row, label=row["kidney_APN"]) for row in legacy]),
        "old_210_overlap": performance([
            dict(row, label=row["kidney_APN"]) for row in legacy
            if row["patient_alias"] in old_labeled_aliases
        ]),
        "outside_old_210": performance([
            dict(row, label=row["kidney_APN"]) for row in legacy
            if row["patient_alias"] not in old_labeled_aliases
        ]),
    }
    for name, aliases in (
        ("old_unlabeled", old_unlabeled_aliases),
        ("new_registry", new_registry_aliases),
    ):
        result["new_development"][name] = performance([
            row for row in new_predictions if row["patient_alias"] in aliases
        ])
        result["legacy_tile_usfm_development"][name] = performance([
            dict(row, label=row["kidney_APN"]) for row in legacy
            if row["patient_alias"] in aliases
        ])
    probe_rows = read_csv(
        EXP / "NCKUH_revised_383_probe_geometry_20261112/registered_image_probe_labels.csv"
    )
    probe_counts = {}
    development_aliases = {row["patient_alias"] for row in new_predictions}
    for name, aliases in (
        ("old_210_overlap", development_aliases & old_labeled_aliases),
        ("outside_old_210", development_aliases - old_labeled_aliases),
    ):
        selected = [
            (new_by_image[row["image_id"]]["patient_alias"], row["probe"])
            for row in probe_rows
            if row["image_id"] in new_by_image
            and new_by_image[row["image_id"]]["patient_alias"] in aliases
        ]
        probe_counts[name] = {
            "development_patients": len(aliases),
            "patients_with_probe_evidence": len({alias for alias, _ in selected}),
            "images_with_probe_evidence": len(selected),
            "probe_images": dict(Counter(probe for _, probe in selected)),
        }
    result["partial_probe_evidence"] = probe_counts
    result["paired_new_stack_vs_legacy_branch"] = paired_model_delta(legacy, new_predictions)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
