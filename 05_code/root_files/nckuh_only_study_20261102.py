"""Audit and replicate fixed renal-ultrasound methods using NCKUH data only.

The current binary outcomes are patient-level supplied labels. This script never
creates kidney-level diagnostic truth by copying them to the two sides.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_only_study_20261102"
FEATURES = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH"
SOURCE = ROOT / "03_data_models/yolo_crops/manifest.csv"
EXTERNAL_SEGMENTER = Path("C:/Users/udoo_w2/Desktop/work-kidney-tumor/kidney-tumor/best.pt")
EXPECTED_SEGMENTER = "b0fe64d610d70cb9444cf85cb02f11add6a70f97a7df9c4b8d100d69e7fba18d"
SEEDS = list(range(20261102, 20261112))
GROUPS = ["DenseNet summary LR", "PyRadiomics LR", "Directional LR",
          "Radiomics DenseNet pair", "Gated attention MIL", "Capacity matched no graph MIL",
          "Graph MIL bag loss", "Graph MIL negative loss", "Graph radiomics pair",
          "Graph directional pair", "Fixed three source fusion", "Summary three source fusion"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_name(name):
    return str(name).replace("_red_", "_", 1)


def patient_key(value):
    return str(value).lstrip("0") or "0"


def audit():
    import cv2
    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(SOURCE, dtype=str)
    source = master.loc[master.site.eq("NCKUH")].copy()
    source["exam_key"] = source.exam_date_raw.str.replace(r"\D", "", regex=True).str[:8]
    earliest = source.groupby("patient_id").exam_key.transform("min")
    selected = source.loc[source.exam_key.eq(earliest)].sort_values(["patient_id", "filename"]).reset_index(drop=True)
    existing = pd.read_csv(FEATURES / "image_manifest.csv", dtype=str)
    patients = pd.read_csv(FEATURES / "patients.csv", dtype={"patient_id": str})
    assert len(source) == 1031 and len(selected) == len(existing) == 1027 and len(patients) == 264
    for column in ("patient_id", "filename", "label"):
        assert selected[column].tolist() == existing[column].tolist(), column
    assert (selected.groupby("patient_id").label.nunique() == 1).all()
    assert patients.patient_id.tolist() == sorted(selected.patient_id.unique())
    assert patients.label.tolist() == selected.groupby("patient_id").label.first().astype(int).tolist()
    raw = pd.read_csv(ROOT / "03_data_models/NCKUH/NCKUH.csv", dtype=str)
    raw["clean_name"] = raw.filename.map(clean_name)
    assert raw.groupby("clean_name").label.nunique().max() == 1
    raw_labels = raw.groupby("clean_name").label.first()
    assert selected.filename.map(clean_name).map(raw_labels).tolist() == selected.label.tolist()
    corrected = existing.drop(columns=["view"]).merge(
        selected[["filename", "view", "exam_date_raw", "exam_date_iso"]],
        on="filename", how="left", validate="one_to_one", sort=False)
    corrected["side"] = corrected["view"].str[0].map({"L": "left", "R": "right"})
    assert corrected.side.notna().all() and corrected["view"].isin(["LL", "LT", "RL", "RT"]).all()
    corrected.to_csv(OUT / "image_manifest_private.csv", index=False, encoding="utf-8-sig")
    patients.to_csv(OUT / "patients_private.csv", index=False, encoding="utf-8-sig")
    image_audit = []
    source_hashes = set()
    for row in corrected.itertuples(index=False):
        image = ROOT / "03_data_models/original_png_nckuh" / clean_name(row.filename)
        contour = ROOT / "03_data_models/nckuh_image_label" / Path(row.filename).with_suffix(".txt")
        assert image.is_file() and contour.is_file()
        source_hash = digest(image)
        source_hashes.add(source_hash)
        native = cv2.imread(str(FEATURES / "native" / f"{row.image_id}.png"), 0)
        mask = cv2.imread(str(FEATURES / "masks" / f"{row.image_id}.png"), 0)
        percentile = cv2.imread(str(FEATURES / "percentile" / f"{row.image_id}.png"), 0)
        assert native is not None and native.shape == mask.shape == percentile.shape
        assert set(np.unique(mask)).issubset({0, 255}) and np.count_nonzero(mask) >= 64
        assert np.count_nonzero(percentile[mask == 0]) == 0
        image_audit.append(dict(image_id=row.image_id, source_sha256=source_hash,
                                contour_sha256=digest(contour),
                                native_sha256=digest(FEATURES / "native" / f"{row.image_id}.png"),
                                mask_sha256=digest(FEATURES / "masks" / f"{row.image_id}.png")))
    pd.DataFrame(image_audit).to_csv(OUT / "image_hash_audit.csv", index=False)
    assert len(source_hashes) == len(corrected)
    cych = master.loc[master.site.eq("CYCH")]
    overlap = set(source.patient_id.map(patient_key)) & set(cych.patient_id.map(patient_key))
    assert not overlap
    cych_hashes = {digest(ROOT / "03_data_models/original_png" / clean_name(name)) for name in cych.filename}
    assert not source_hashes.intersection(cych_hashes)
    arrays = {}
    for filename, shape in [("dense_tight_image.npy", (1027, 1024)), ("radiomics_percentile.npy", (264, 914)), ("i3.npy", (264, 50))]:
        array = np.load(FEATURES / filename)
        assert array.shape == shape and np.isfinite(array).all()
        arrays[filename] = dict(shape=list(array.shape), sha256=digest(FEATURES / filename))
    # Reconcile saved patient summaries against their image-level features.
    for csv_name, npy_name, expected_dim in [("radiomics_percentile_image.csv", "radiomics_percentile.npy", 457),
                                            ("i3_image.csv", "i3.npy", 25)]:
        frame = pd.read_csv(FEATURES / csv_name).set_index("image_id").reindex(corrected.image_id)
        if expected_dim == 457:
            frame = frame[[c for c in frame if not c.endswith(("_Energy", "_TotalEnergy"))]]
        assert frame.shape == (1027, expected_dim)
        values = frame.to_numpy(float)
        rebuilt = np.stack([np.r_[values[corrected.patient_id.eq(pid)].mean(0),
                                    values[corrected.patient_id.eq(pid)].std(0)] for pid in patients.patient_id]).astype(np.float32)
        np.testing.assert_allclose(rebuilt, np.load(FEATURES / npy_name), rtol=1e-5, atol=1e-4)
    request = pd.read_csv(ROOT / "03_data_models/kidney_label_request/kidney_label_request.csv", dtype=str)
    request.columns = [c.split(" ")[0] for c in request.columns]
    request = request[request.patient_id.map(patient_key).isin(source.patient_id.map(patient_key))]
    filled = {c: int((request[c].fillna("").str.strip() != "").sum()) for c in ("left_dmsa", "right_dmsa")}
    segmenter_hash = digest(EXTERNAL_SEGMENTER)
    assert segmenter_hash == EXPECTED_SEGMENTER
    accepted = corrected.mask_source.eq("existing_yolo")
    ious = corrected.predicted_iou.astype(float)
    counts = corrected.groupby(["patient_id", "side"]).size()
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), site_tag="NCKUH",
                  source_rows=len(source), retained_images=len(corrected), patients=len(patients),
                  positive_patients=int(patients.label.sum()), negative_patients=int((patients.label == 0).sum()),
                  excluded_later_exam_images=len(source) - len(corrected),
                  imaged_kidneys=len(counts), left_kidneys=int((counts.index.get_level_values("side") == "left").sum()),
                  right_kidneys=int((counts.index.get_level_values("side") == "right").sum()),
                  both_kidneys_patients=int((corrected.groupby("patient_id").side.nunique() == 2).sum()),
                  views=corrected["view"].value_counts().to_dict(),
                  label_request_exams=len(request), populated_side_label_fields=filled,
                  kidney_APN_ground_truth_available=False,
                  outcome="Supplied patient-level binary label, not independently verified side-specific APN",
                  overlapping_CYCH_patients=len(overlap), overlapping_CYCH_image_hashes=0,
                  duplicate_retained_source_hashes=len(corrected) - len(source_hashes),
                  accepted_model_masks=int(accepted.sum()), fallback_masks=int((~accepted).sum()),
                  accepted_iou_quantiles=ious[accepted].quantile([.25, .5, .75]).to_dict(),
                  all_prediction_iou_quantiles=ious.quantile([.25, .5, .75]).to_dict(),
                  corrupted_legacy_view_rows=int((~existing["view"].isin(["LL", "LT", "RL", "RT"])).sum()),
                  corrected_view_source="Exact one-to-one filename join to canonical manifest; source features unmodified",
                  feature_arrays=arrays, external_segmenter_sha256=segmenter_hash,
                  source_sha256=digest(SOURCE), raw_label_csv_sha256=digest(ROOT / "03_data_models/NCKUH/NCKUH.csv"),
                  source_manuscript_sha256=digest(ROOT / "01_submission/CYCH_label_clarification_20261003/01_Main_manuscript_BMC_Medical_Imaging_focused_20261101.docx"),
                  prior_exploration="NCKUH features and repeated cross-validation were already examined in October 2026; new random draws do not form a previously untouched cohort",
                  clinical_metadata="Age, sex, institutional provenance, reference standard and ethics require original clinical documentation")
    write_json(OUT / "cohort_audit.json", report)
    print(json.dumps({k: report[k] for k in ["patients", "retained_images", "positive_patients", "negative_patients", "imaged_kidneys", "populated_side_label_fields", "overlapping_CYCH_patients"]}), flush=True)


def metric(y, score, weight=None):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, brier_score_loss, confusion_matrix
    precision, recall, _ = precision_recall_curve(y, score, sample_weight=weight)
    tn, fp, fn, tp = confusion_matrix(y, score >= .5, labels=[0, 1], sample_weight=weight).ravel()
    return dict(auroc=float(roc_auc_score(y, score, sample_weight=weight)),
                ap=float(average_precision_score(y, score, sample_weight=weight)),
                auprc=float(abs(np.trapezoid(precision, recall))),
                brier=float(brier_score_loss(y, score, sample_weight=weight)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                accuracy=float((tp + tn) / (tp + tn + fp + fn)))


def prepare():
    from sklearn.model_selection import train_test_split
    audit_report = json.loads((OUT / "cohort_audit.json").read_text(encoding="utf-8"))
    assert audit_report["overlapping_CYCH_patients"] == 0
    assignments = []
    patients = pd.read_csv(OUT / "patients_private.csv", dtype={"patient_id": str})
    for seed in SEEDS:
        train, rest = train_test_split(np.arange(len(patients)), train_size=158, random_state=seed, stratify=patients.label)
        development, test = train_test_split(rest, train_size=53, random_state=seed + 1000, stratify=patients.label.to_numpy()[rest])
        assert len(set(train) | set(development) | set(test)) == 264
        for role, index in [("train", train), ("development", development), ("exploratory_test", test)]:
            for i in index:
                assignments.append(dict(seed=seed, patient_alias=patients.patient_alias.iloc[i], role=role))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    purpose="NCKUH-only fixed-method exploratory replication, not kidney-level validation",
                    endpoint="Supplied patient-level 0/1 label; no invented side-specific diagnosis",
                    cohort={k: audit_report[k] for k in ("patients", "retained_images", "positive_patients", "negative_patients")},
                    excluded_data="No CYCH training, validation, test images or labels; no two-site adapted segmentation checkpoint; no disease classifier pretrained on these NCKUH patients",
                    segmentation="Unchanged non-CYCH/non-NCKUH pretrained checkpoint; contour-assisted QC and supplied-contour fallback retained as a hybrid research workflow",
                    pretrained_encoder="Frozen RadImageNet DenseNet-121; only NCKUH classification heads retrained",
                    split="Ten fixed stratified random patient draws, 158 train / 53 development / 53 exploratory test, no k-fold",
                    seeds=SEEDS, groups=GROUPS, feature_selection="Training-only imputation, standardization and F-test top 32 for radiomics and DenseNet summaries; full 50 directional features",
                    graph_preprocessing="Training-images-only StandardScaler and PCA32",
                    neural_training="AdamW lr=.001 weight_decay=.001, 300 fixed full-batch epochs, dropout=.25, gradient norm5, 32/64/32 graph layers, gated attention16",
                    ablation="One-layer gated MIL; capacity-matched two-layer identity adjacency; graph bag-only; graph negative-only auxiliary loss; fixed pairwise and three-way probability averages",
                    control_of_selection="Fixed three-source fusion is the transferred primary candidate irrespective of results. No new hyperparameters, weights, thresholds or best-seed selection",
                    threshold=.5, uncertainty="10000 valid stratified patient-cluster bootstraps of fixed predictions retaining all linked random-draw appearances, paired across configurations; conditional within-cohort intervals",
                    limitation=audit_report["prior_exploration"],
                    kidney_endpoint_gate="Requires independently verified per-side APN labels, diagnostic timing and patient-grouped splitting. Patient labels cannot satisfy this gate.")
    for name, value in [("protocol.json", protocol)]:
        if (OUT / name).exists():
            raise FileExistsError("Preserve existing protocol")
        write_json(OUT / name, value)
    pd.DataFrame(assignments).to_csv(OUT / "random_patient_assignments.csv", index=False)
    lock = dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=digest(__file__),
                protocol_sha256=digest(OUT / "protocol.json"), cohort_audit_sha256=digest(OUT / "cohort_audit.json"),
                assignments_sha256=digest(OUT / "random_patient_assignments.csv"),
                features={n: digest(FEATURES / n) for n in ["dense_tight_image.npy", "radiomics_percentile.npy", "i3.npy"]})
    write_json(OUT / "new_run_lock.json", lock)
    print("Fixed replication protocol recorded before this new experiment; prior cohort exploration disclosed", flush=True)


def train_neural(seed, mode, index, y, bags, valid):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    class BagModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.single = nn.Linear(32, 32) if mode == "gated" else None
            self.layer1 = nn.Linear(32, 64, bias=False) if mode != "gated" else None
            self.layer2 = nn.Linear(64, 32, bias=False) if mode != "gated" else None
            self.dropout = nn.Dropout(.25)
            self.v, self.u, self.w = nn.Linear(32, 16), nn.Linear(32, 16), nn.Linear(16, 1, bias=False)
            self.classifier, self.instance = nn.Linear(32, 1), nn.Linear(32, 1)

        def forward(self, x, mask):
            if mode == "gated":
                hidden = self.dropout(torch.relu(self.single(x)))
            else:
                if mode == "identity":
                    adjacency = torch.diag_embed(mask.float())
                else:
                    differences = x.unsqueeze(2) - x.unsqueeze(1)
                    distance = (differences ** 2).sum(-1)
                    pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
                    eye = torch.eye(x.shape[1], dtype=torch.bool).unsqueeze(0)
                    off_diag = torch.where(pairs & ~eye & (distance > 0), distance, torch.nan)
                    bandwidth = torch.nanmedian(off_diag.flatten(1), dim=1).values
                    bandwidth = torch.where(torch.isfinite(bandwidth), bandwidth, torch.ones_like(bandwidth)).clamp_min(1e-6)
                    adjacency = torch.exp(-distance / bandwidth[:, None, None]) * pairs
                    degree = adjacency.sum(-1).clamp_min(1e-8)
                    adjacency = adjacency / (degree.unsqueeze(-1) * degree.unsqueeze(-2)).sqrt()
                hidden = self.dropout(torch.relu(self.layer1(torch.bmm(adjacency, x))))
                hidden = torch.relu(self.layer2(torch.bmm(adjacency, hidden)))
            weights = self.w(torch.tanh(self.v(hidden)) * torch.sigmoid(self.u(hidden))).squeeze(-1)
            weights = torch.softmax(weights.masked_fill(~mask, -1e9), dim=1)
            pooled = (weights.unsqueeze(-1) * hidden).sum(1)
            return self.classifier(pooled).squeeze(-1), self.instance(hidden).squeeze(-1), weights

    model = BagModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    target = torch.tensor(y[index], dtype=torch.float32)
    class_weight = torch.where(target > 0, len(target) / (2 * target.sum()), len(target) / (2 * (len(target) - target.sum())))
    for _ in range(300):
        model.train()
        optimizer.zero_grad()
        logits, instances, _ = model(bags[index], valid[index])
        loss = (nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * class_weight).mean()
        if mode == "negative":
            selected = valid[index] & (target[:, None] == 0)
            loss = loss + nn.functional.binary_cross_entropy_with_logits(instances[selected], torch.zeros_like(instances[selected]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits, instances, weights = model(bags, valid)
        probabilities = torch.sigmoid(logits).numpy()
    assert np.isfinite(probabilities).all()
    return probabilities, model.state_dict(), weights.numpy(), torch.sigmoid(instances).numpy(), sum(p.numel() for p in model.parameters())


def run():
    import torch
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    torch.set_num_threads(4)
    lock = json.loads((OUT / "new_run_lock.json").read_text(encoding="utf-8"))
    assert lock["source_sha256"] == digest(__file__)
    assert lock["protocol_sha256"] == digest(OUT / "protocol.json")
    assert lock["assignments_sha256"] == digest(OUT / "random_patient_assignments.csv")
    for name, value in lock["features"].items():
        assert digest(FEATURES / name) == value
    if (OUT / "predictions.csv").exists():
        raise FileExistsError("Preserve completed results")
    (OUT / "models").mkdir(exist_ok=True)
    patients = pd.read_csv(OUT / "patients_private.csv", dtype={"patient_id": str})
    manifest = pd.read_csv(OUT / "image_manifest_private.csv", dtype={"patient_id": str})
    y = patients.label.to_numpy(int)
    dense = np.load(FEATURES / "dense_tight_image.npy")
    rad = np.load(FEATURES / "radiomics_percentile.npy")
    directional = np.load(FEATURES / "i3.npy")
    row_groups = [np.flatnonzero(manifest.patient_id.eq(pid)) for pid in patients.patient_id]
    dense_summary = np.stack([np.r_[dense[g].mean(0), dense[g].std(0)] for g in row_groups])
    assignments = pd.read_csv(OUT / "random_patient_assignments.csv")
    outputs, metrics, train_records, instance_rows = [], [], [], []
    for seed in SEEDS:
        roles = assignments[assignments.seed.eq(seed)].set_index("patient_alias").role.reindex(patients.patient_alias)
        train = np.flatnonzero(roles.eq("train"))
        scores = {}
        for name, values, select in [(GROUPS[0], dense_summary, True), (GROUPS[1], rad, True), (GROUPS[2], directional, False)]:
            transform = SelectKBest(f_classif, k=32) if select else VarianceThreshold()
            head = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), transform,
                                 LogisticRegression(C=.1, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=seed))
            head.fit(values[train], y[train])
            scores[name] = head.predict_proba(values)[:, list(head.classes_).index(1)]
            joblib.dump(head, OUT / "models" / f"{seed}_{name.split()[0]}_head.joblib", compress=3)
        scores[GROUPS[3]] = (scores[GROUPS[0]] + scores[GROUPS[1]]) / 2
        train_images = np.concatenate([row_groups[i] for i in train])
        scaler = StandardScaler().fit(dense[train_images])
        standardized = scaler.transform(dense)
        pca = PCA(32, random_state=seed).fit(standardized[train_images])
        reduced = pca.transform(standardized).astype(np.float32)
        width = max(map(len, row_groups))
        arrays = np.zeros((len(y), width, 32), dtype=np.float32)
        mask = np.zeros((len(y), width), dtype=bool)
        for i, rows in enumerate(row_groups):
            arrays[i, :len(rows)] = reduced[rows]
            mask[i, :len(rows)] = True
        bags, valid = torch.from_numpy(arrays), torch.from_numpy(mask)
        joblib.dump(dict(scaler=scaler, pca=pca), OUT / "models" / f"{seed}_preprocess.joblib", compress=3)
        for group, mode in [(GROUPS[4], "gated"), (GROUPS[5], "identity"), (GROUPS[6], "graph"), (GROUPS[7], "negative")]:
            probabilities, state, attention, instance_score, parameters = train_neural(seed, mode, train, y, bags, valid)
            scores[group] = probabilities
            torch.save(state, OUT / "models" / f"{seed}_{mode}.pth")
            train_records.append(dict(seed=seed, group=group, parameters=parameters, train_patients=len(train), train_images=len(train_images)))
            if mode == "negative":
                for i, rows in enumerate(row_groups):
                    if roles.iloc[i] == "train":
                        continue
                    for j, row in enumerate(rows):
                        instance_rows.append(dict(seed=seed, patient_alias=patients.patient_alias.iloc[i], role=roles.iloc[i],
                                                  image_id=manifest.image_id.iloc[row], side=manifest.side.iloc[row],
                                                  attention=float(attention[i, j]), instance_score=float(instance_score[i, j])))
        scores[GROUPS[8]] = (scores[GROUPS[7]] + scores[GROUPS[1]]) / 2
        scores[GROUPS[9]] = (scores[GROUPS[7]] + scores[GROUPS[2]]) / 2
        scores[GROUPS[10]] = (scores[GROUPS[7]] + scores[GROUPS[1]] + scores[GROUPS[2]]) / 3
        scores[GROUPS[11]] = (scores[GROUPS[0]] + scores[GROUPS[1]] + scores[GROUPS[2]]) / 3
        for role in ["development", "exploratory_test"]:
            index = np.flatnonzero(roles.eq(role))
            assert len(index) == 53
            for group in GROUPS:
                prediction = scores[group][index]
                metrics.append(dict(seed=seed, role=role, group=group, patients=len(index), **metric(y[index], prediction)))
                for i in index:
                    outputs.append(dict(seed=seed, role=role, group=group, patient_alias=patients.patient_alias.iloc[i],
                                        label=int(y[i]), score=float(scores[group][i])))
        print(f"Completed fixed draw {seed}", flush=True)
    pd.DataFrame(outputs).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "metrics_per_draw.csv", index=False)
    pd.DataFrame(train_records).to_csv(OUT / "training_records.csv", index=False)
    pd.DataFrame(instance_rows).to_csv(OUT / "instance_outputs.csv", index=False)
    summary = pd.DataFrame(metrics).groupby(["role", "group"])[["auroc", "ap", "auprc", "brier", "sensitivity", "specificity", "accuracy"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(OUT / "summary.csv")
    print(summary.loc[(slice(None), GROUPS[10]), :].to_string(), flush=True)


def bootstrap():
    predictions = pd.read_csv(OUT / "predictions.csv")
    metrics = pd.read_csv(OUT / "metrics_per_draw.csv")
    results, deltas = [], []
    for role in ["development", "exploratory_test"]:
        subset = predictions[predictions.role.eq(role)]
        sample = subset[subset.group.eq(GROUPS[10])]
        aliases = sorted(sample.patient_alias.unique())
        labels_by_alias = sample.groupby("patient_alias").label.first().reindex(aliases).to_numpy(int)
        matrices = []
        for seed in SEEDS:
            draw = subset[subset.seed.eq(seed)]
            pivot = draw.pivot(index="patient_alias", columns="group", values="score").reindex(columns=GROUPS)
            target = draw.groupby("patient_alias").label.first().reindex(pivot.index).to_numpy(int)
            idx = np.array([aliases.index(a) for a in pivot.index])
            matrices.append((idx, target, pivot.to_numpy(float)))
        rng = np.random.default_rng(20261102)
        positive, negative = np.flatnonzero(labels_by_alias == 1), np.flatnonzero(labels_by_alias == 0)
        batches, total = [], 0
        while total < 10000:
            batch_size = min(2000, 10000 - total)
            sampled = np.c_[rng.choice(positive, (batch_size, len(positive)), replace=True),
                            rng.choice(negative, (batch_size, len(negative)), replace=True)]
            counts = np.zeros((batch_size, len(aliases)), dtype=float)
            np.add.at(counts, (np.arange(batch_size)[:, None], sampled), 1)
            keep = np.ones(batch_size, dtype=bool)
            values = np.zeros((batch_size, len(GROUPS), 4))
            for idx, target, scores in matrices:
                weight = counts[:, idx]
                keep &= (weight[:, target == 0].sum(1) > 0) & (weight[:, target == 1].sum(1) > 0)
                for j in range(len(GROUPS)):
                    values[:, j] += weighted_metrics_batch(target, scores[:, j], weight)
            batches.append(values[keep] / len(SEEDS))
            total += int(keep.sum())
            print(f"{role}: {total} valid bootstrap replicates", flush=True)
        array = np.concatenate(batches)[:10000]
        np.save(OUT / f"bootstrap_{role}.npy", array)
        for j, group in enumerate(GROUPS):
            for k, name in enumerate(["auroc", "ap", "auprc", "brier"]):
                point = metrics[(metrics.role.eq(role)) & metrics.group.eq(group)][name].mean()
                low, high = np.quantile(array[:, j, k], [.025, .975])
                results.append(dict(role=role, group=group, metric=name, mean=float(point), low=float(low), high=float(high), distinct_patients=len(aliases)))
        for control in [GROUPS[3], GROUPS[5], GROUPS[6], GROUPS[7], GROUPS[11]]:
            for k, name in enumerate(["auroc", "ap", "auprc", "brier"]):
                difference = array[:, GROUPS.index(GROUPS[10]), k] - array[:, GROUPS.index(control), k]
                point = metrics[metrics.role.eq(role) & metrics.group.eq(GROUPS[10])][name].mean() - metrics[metrics.role.eq(role) & metrics.group.eq(control)][name].mean()
                low, high = np.quantile(difference, [.025, .975])
                deltas.append(dict(role=role, comparison="Fixed fusion minus " + control, metric=name, difference=point, low=low, high=high))
    pd.DataFrame(results).to_csv(OUT / "confidence_intervals.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "paired_differences.csv", index=False)


def weighted_metrics_batch(y, score, weights):
    """Exact weighted ROC, step AP, trapezoidal PR and Brier for fixed scores."""
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
    positives = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    negatives = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    tp, fp = positives.cumsum(1), negatives.cumsum(1)
    total_positive, total_negative = tp[:, -1:], fp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / total_positive
        fpr = fp / total_negative
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
        recall0 = np.c_[np.zeros(len(weights)), recall]
        fpr0 = np.c_[np.zeros(len(weights)), fpr]
        roc = np.sum(np.diff(fpr0) * (recall0[:, 1:] + recall0[:, :-1]) / 2, axis=1)
        ap = np.sum(np.diff(recall0) * precision, axis=1)
        pr = np.sum(np.diff(recall0) * (precision + np.c_[np.ones(len(weights)), precision[:, :-1]]) / 2, axis=1)
        brier = (weights * (score - y) ** 2).sum(1) / weights.sum(1)
    return np.c_[roc, ap, pr, brier]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "prepare", "run", "bootstrap"])
    args = parser.parse_args()
    globals()[args.stage]()
