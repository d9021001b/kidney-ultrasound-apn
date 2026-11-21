"""Fixed, patient-disjoint triplet ablation using the existing RadImageNet encoder."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_local_features as L
import renal_dinov3_native_probe as N
import renal_interpretable_probe as I
import renal_voting_validation as V
from radimagenet_backbones import radimagenet_densenet121, RADIMAGENET_DIR

OUT = R.ROOT / "CYCH_triplet_probe_20260913"
CONFIGS = {f"{scope}-{objective}": (scope, objective)
           for scope in ("P", "F") for objective in ("CE", "T", "CE-T")}
EPOCHS = 200
MARGIN = 0.2
HEAD_LR, TAIL_LR, WEIGHT_DECAY = 1e-3, 1e-5, 1e-4


def read(name):
    return pd.read_csv(OUT / name, dtype={"patient_id": str})


def configure_torch():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def source_files():
    return [Path(__file__), *[R.ROOT / f"{m}.py" for m in (
        "renal_random_validation", "renal_local_features", "renal_dinov3_native_probe",
        "renal_dinov3_single_encoder", "renal_interpretable_probe", "renal_voting_validation",
        "renal_foundation_screening", "renal_roi_enhancement", "radimagenet_backbones")
        if (R.ROOT / f"{m}.py").exists()]]


def require_lock():
    lock = json.loads((OUT / "protocol_locked.json").read_text())
    assert lock["protocol_sha256"] == R.sha256(OUT / "protocol.json")
    for path, digest in lock["sources"].items():
        assert R.sha256(Path(path)) == digest, f"Locked source changed: {path}"
    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, digest in protocol["inputs"].items():
        assert R.sha256(Path(path)) == digest, f"Locked input changed: {path}"


def prepare():
    if (OUT / "protocol_locked.json").exists():
        require_lock()
        print("Existing triplet protocol verified", flush=True)
        return
    for directory in ("source", "cache", "features", "models", "runs", "qa", "figures_600dpi"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv"):
        shutil.copy2(L.OUT / name, OUT / name)
    df, patients, split = read("image_manifest.csv"), read("patients.csv"), read("random_patient_assignments.csv")
    assert len(df) == 349 and len(patients) == 98 and patients.label.sum() == 52
    assert df.image_id.nunique() == len(df) and not df.source_sha256.duplicated().any()
    assert df.groupby("patient_id").label.first().to_dict() == patients.set_index("patient_id").label.to_dict()
    for seed in R.SEEDS:
        N.indices_for(patients, split, seed)
    paths = [OUT / name for name in ("image_manifest.csv", "patients.csv", "random_patient_assignments.csv")]
    paths += [RADIMAGENET_DIR / "DenseNet121.pt", N.OUT / "features/dense_canvas_image.npy",
              N.OUT / "patient_predictions_by_role.csv", I.OUT / "patient_predictions_by_role.csv"]
    protocol = dict(
        cohort=dict(patients=98, images=349, positive=52, negative=46), seeds=R.SEEDS,
        roles=dict(train=58, development=20, exploratory_test=20),
        inputs={str(p): R.sha256(p) for p in paths}, configs=CONFIGS,
        initialization="Existing RadImageNet DenseNet121.pt. Every fit resets the original tail and paired random projection/classifier. No old triplet checkpoint or clinical covariates.",
        image_input="Unchanged P1-P99 masked grayscale; exact original R.canvas_image at224; x/127.5-1. Prefix through transition3 frozen and cached without labels. No new augmentation or mask exclusions.",
        scopes="P: frozen entire backbone, projection only. F: last denseblock4 and norm5 affine parameters plus projection. All BatchNorm running buffers frozen in eval mode.",
        representation="Linear1024->128 bias=False; L2 normalize each image. Training: mean image embedding per unique patient, renormalized. Downstream: image-embedding mean/std ->256features, training-only full LR head.",
        training=dict(epochs=EPOCHS, full_training_image_pass_each_epoch=True, projection_lr=HEAD_LR,
                      tail_lr=TAIL_LR, weight_decay=WEIGHT_DECAY, optimizer="Adam", gradient_clip_norm=5,
                      triplet_margin=MARGIN, embedding_dimension=128, optimizer_replicates_per_draw=1,
                      rng="seed+37 projection; seed+97 dedicated CPU triplet miner"),
        loss="CE is patient-balanced binary cross entropy with logits (2-class CE). T is patient-balanced torch TripletMarginLoss p=2 eps=1e-6 swap=False, non-squared Euclidean distance. CE-T is their unweighted sum. No validation/test loss, early stopping or hyperparameter search.",
        mining="One anchor per training patient; random same-class DIFFERENT patient positive. Random semi-hard negative with d_ap<d_an<d_ap+0.2; random opposite-class fallback. Detached embeddings, dedicated generator. Different patients by construction; full audit at epochs1,50,100,200.",
        head="N.full_head: train-only median imputation, variance filter, scaling, balanced liblinear LR C=.1 max_iter2000. No PCA, clinical variables, SVM or test calibration.",
        gate="Only T/CE-T eligible: development mean AUROC and AP each >= matched CE minus.02 AND DN-B minus.02. Rank mean(AUROC+AP)/2; ties simpler P then pureT then ID. At most one replacement vote RAD+candidate+I3, three equal members, replacing DN-A. Promote vs current directional vote only if mean dev deltaAUROC>=.03, deltaAP>=.02, joint wins>=6/10.",
        caveats="All98 patients previously explored; overlapping random draws, not k-fold or independent confirmation. Test metrics only after writing development decision. Original paper used128features,margin.2,200epochs but image-level triplets, clinical fusion and PCA2/SVM; this is an image-only patient-level methodological ablation, NOT exact reproduction.",
        references=["https://arxiv.org/abs/1503.03832", "https://github.com/BMEII-AI/RadImageNet",
                    "https://doi.org/10.1148/ryai.210315",
                    "https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html"])
    R.write_json(OUT / "protocol.json", protocol)
    sources = {str(p): R.sha256(p) for p in source_files()}
    R.write_json(OUT / "protocol_locked.json", dict(utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=R.sha256(OUT / "protocol.json"), sources=sources))
    for p in source_files():
        shutil.copy2(p, OUT / "source" / p.name)
    manuscript = R.ROOT / "yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"
    R.write_json(OUT / "preserved_manuscript.json", dict(path=str(manuscript), sha256=R.sha256(manuscript)))
    print("Locked six triplet ablation configurations before new outcome evaluation", flush=True)


def split_encoder(model):
    prefix = nn.Sequential(OrderedDict((k, v) for k, v in model.features.named_children()
                                      if k not in ("denseblock4", "norm5")))
    tail = nn.Sequential(OrderedDict(denseblock4=model.features.denseblock4,
        norm5=model.features.norm5, relu=nn.ReLU(), pool=nn.AdaptiveAvgPool2d(1), flatten=nn.Flatten()))
    return prefix, tail


def encode():
    require_lock()
    if (OUT / "cache_audit.json").exists():
        audit = json.loads((OUT / "cache_audit.json").read_text())
        for name, digest in audit["cache_sha256"].items():
            assert R.sha256(OUT / "cache" / name) == digest
        print("Verified existing frozen-prefix cache", flush=True)
        return
    configure_torch()
    torch.manual_seed(R.SEEDS[0])
    df = read("image_manifest.csv")
    arrays, inputs = [], []
    for row in df.itertuples():
        ip, mp = L.OUT / "images/percentile" / f"{row.image_id}.png", L.OUT / "masks" / f"{row.image_id}.png"
        assert R.sha256(mp) == row.mask_sha256
        gray, mask = cv2.imread(str(ip), 0), (cv2.imread(str(mp), 0) > 0).astype(np.uint8)
        assert gray.shape == mask.shape and not gray[mask == 0].any()
        canvas = R.canvas_image(gray, mask)
        assert np.array_equal(canvas, cv2.imread(str(N.OUT / "inputs/canvas" / f"{row.image_id}.png")))
        arrays.append(canvas)
        inputs.append(dict(image_id=row.image_id, image_sha256=R.sha256(ip), mask_sha256=R.sha256(mp)))
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    before = L.model_digest(model)
    prefix, tail = split_encoder(model)
    cached, dense = [], []
    with torch.inference_mode():
        for start in range(0, len(arrays), 32):
            x = torch.from_numpy(np.stack(arrays[start:start+32]).transpose(0, 3, 1, 2)).cuda().float()/127.5-1
            maps = prefix(x)
            features = tail(maps)
            if start == 0:
                direct_error = float((features-model(x)).abs().max())
                assert direct_error <= 1e-4
            cached.append(maps.cpu().numpy())
            dense.append(features.cpu().numpy())
    maps, features = np.concatenate(cached), np.concatenate(dense)
    assert maps.shape == (349, 512, 7, 7) and features.shape == (349, 1024)
    old = np.load(N.OUT / "features/dense_canvas_image.npy")
    assert np.allclose(features, old, rtol=1e-4, atol=1e-4)
    assert L.model_digest(model) == before
    np.save(OUT / "cache/prefix_maps.npy", maps)
    np.save(OUT / "cache/dense_image.npy", features)
    R.write_json(OUT / "cache_audit.json", dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        backbone_sha256=R.sha256(RADIMAGENET_DIR / "DenseNet121.pt"), state_before=before,
        state_after=L.model_digest(model), prefix_state=L.model_digest(prefix), tail_state=L.model_digest(tail),
        cache_sha256={name: R.sha256(OUT / "cache" / name) for name in ("prefix_maps.npy", "dense_image.npy")},
        direct_vs_split_maxabs=direct_error, inherited_cache_maxabs=float(np.abs(features-old).max()), inputs=inputs))
    print("Frozen-prefix cache matched full pretrained DenseNet and inherited features", flush=True)


class Projection(nn.Module):
    def __init__(self, input_dimension=1024):
        super().__init__()
        self.projection = nn.Linear(input_dimension, 128, bias=False)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


def bag_matrix(image_patients, patients):
    matrix = (np.asarray(patients)[:, None] == np.asarray(image_patients)[None, :]).astype(np.float32)
    assert (matrix.sum(0) == 1).all() and (matrix.sum(1) > 0).all()
    return torch.from_numpy(matrix / matrix.sum(1, keepdims=True))


def mine_triplets(embeddings, labels, generator, margin=MARGIN):
    # CPU mining avoids a device synchronization for each patient's candidate list.
    z, y = embeddings.detach().cpu(), labels.detach().cpu()
    distances = torch.cdist(z, z)
    positives, negatives, semi = [], [], []
    for anchor in range(len(y)):
        pos = torch.where((y == y[anchor]) & (torch.arange(len(y)) != anchor))[0]
        neg = torch.where(y != y[anchor])[0]
        if not len(pos) or not len(neg):
            raise ValueError("Triplets require at least two distinct patients in each class")
        p = pos[torch.randint(len(pos), (1,), generator=generator).item()].item()
        d = distances[anchor]
        eligible = neg[(d[neg] > d[p]) & (d[neg] < d[p]+margin)]
        pool = eligible if len(eligible) else neg
        n = pool[torch.randint(len(pool), (1,), generator=generator).item()].item()
        positives.append(p)
        negatives.append(n)
        semi.append(bool(len(eligible)))
    return torch.tensor(positives, device=embeddings.device), torch.tensor(negatives, device=embeddings.device), semi


def train_network(x, bags, labels, original_tail, seed, scope, objective):
    """Accept only training tensors; no validation IDs, labels or outcomes enter this function."""
    torch.manual_seed(seed+37)
    head = Projection(x.shape[1] if scope == "P" else 1024).to(x.device)
    head.classifier.requires_grad_(objective != "T")
    initial_head = L.model_digest(head)
    tail = copy.deepcopy(original_tail).to(x.device).eval().requires_grad_(True) if scope == "F" else nn.Identity()
    initial_tail = L.model_digest(tail)
    buffers = {k: v.clone() for k, v in tail.named_buffers()}
    groups = [dict(params=[p for p in head.parameters() if p.requires_grad], lr=HEAD_LR)]
    if scope == "F":
        groups.append(dict(params=tail.parameters(), lr=TAIL_LR))
    optimizer = torch.optim.Adam(groups, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed+97)
    weights = torch.tensor([len(labels)/(2*int((labels == c).sum())) for c in labels], device=x.device)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2, eps=1e-6, swap=False, reduction="none")
    history, mined = [], []
    start_time = time.monotonic()
    for epoch in range(1, EPOCHS+1):
        optimizer.zero_grad(set_to_none=True)
        zi = head(tail(x))
        zp = F.normalize(bags @ zi, dim=1)
        ce = (F.binary_cross_entropy_with_logits(head.classifier(zp).squeeze(1), labels.float(), reduction="none")*weights).mean() if objective != "T" else zp.new_zeros(())
        triplet, active, semi_fraction = zp.new_zeros(()), 0., 0.
        if objective != "CE":
            pos, neg, semi = mine_triplets(zp, labels, generator)
            losses = criterion(zp, zp[pos], zp[neg])
            triplet = (losses*weights).mean()
            active, semi_fraction = float((losses > 0).float().mean()), float(np.mean(semi))
            if epoch in (1, 50, 100, 200):
                mined.extend(dict(epoch=epoch, anchor=a, positive=p, negative=n, semi_hard=s)
                             for a, (p, n, s) in enumerate(zip(pos.cpu().tolist(), neg.cpu().tolist(), semi)))
        loss = ce+triplet
        assert torch.isfinite(loss)
        loss.backward()
        if epoch == 1:
            assert head.projection.weight.grad is not None and head.projection.weight.grad.abs().sum() > 0
            if scope == "F":
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tail.parameters())
        nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 5)
        optimizer.step()
        history.append(dict(epoch=epoch, loss=float(loss.detach()), ce=float(ce.detach()),
                            triplet=float(triplet.detach()), active_fraction=active, semi_hard_fraction=semi_fraction))
        if epoch in (1, 50, 100, 150, 200):
            print(f"{scope}-{objective} seed={seed} epoch={epoch}/{EPOCHS} loss={loss.item():.5f} elapsed={time.monotonic()-start_time:.1f}s", flush=True)
    assert all(torch.equal(v, buffers[k]) for k, v in tail.named_buffers()), "BatchNorm buffers changed"
    final_tail = L.model_digest(tail)
    if scope == "F":
        assert initial_tail != final_tail, "Fine-tuned tail did not change"
    audit = dict(seed=seed, scope=scope, objective=objective, epochs=EPOCHS, elapsed_seconds=time.monotonic()-start_time,
        initial_head=initial_head, initial_tail=initial_tail, final_head=L.model_digest(head), final_tail=final_tail,
        batchnorm_buffers_unchanged=True, train_patients=len(labels), train_images=len(x),
        trainable_parameters=sum(p.numel() for g in optimizer.param_groups for p in g["params"]))
    return head.eval(), tail.eval(), history, mined, audit


def train():
    require_lock()
    configure_torch()
    encode()
    patients, df, split = read("patients.csv"), read("image_manifest.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(int)
    maps = torch.from_numpy(np.load(OUT / "cache/prefix_maps.npy")).cuda()
    dense = torch.from_numpy(np.load(OUT / "cache/dense_image.npy")).cuda()
    model = radimagenet_densenet121().eval().requires_grad_(False)
    prefix, original_tail = split_encoder(model)
    prefix_before, tail_before = L.model_digest(prefix), L.model_digest(original_tail)
    for seed in R.SEEDS:
        idx = N.indices_for(patients, split, seed)["train"]
        aliases = patients.patient_alias.iloc[idx].to_numpy()
        rows = np.flatnonzero(df.patient_alias.isin(aliases))
        bags = bag_matrix(df.patient_alias.iloc[rows], aliases).cuda()
        labels = torch.from_numpy(y[idx]).cuda()
        for group, (scope, objective) in CONFIGS.items():
            run = OUT / "runs" / f"{group}_{seed}"
            run.mkdir(exist_ok=True)
            if (run / "complete.json").exists():
                saved = json.loads((run / "complete.json").read_text())
                for path, digest in saved["outputs"].items():
                    assert R.sha256(OUT / path) == digest
                print(f"Verified completed {group} {seed}", flush=True)
                continue
            source = dense if scope == "P" else maps
            head, tail, history, mined, audit = train_network(source[rows], bags, labels, original_tail, seed, scope, objective)
            embeddings = []
            with torch.inference_mode():
                for start in range(0, len(df), 32):
                    embeddings.append(head(tail(source[start:start+32])).cpu().numpy())
            image_features = np.concatenate(embeddings)
            assert np.isfinite(image_features).all() and np.allclose(np.linalg.norm(image_features, axis=1), 1, atol=1e-5)
            patient_features = R.aggregate(image_features, df, patients)
            feature_path = OUT / "features" / f"{group}_{seed}.npy"
            image_path = OUT / "features" / f"{group}_{seed}_image.npy"
            np.save(feature_path, patient_features)
            np.save(image_path, image_features)
            checkpoint = OUT / "models" / f"{group}_{seed}.pt"
            torch.save(dict(head={k: v.cpu() for k, v in head.state_dict().items()},
                tail={k: v.cpu() for k, v in tail.state_dict().items()} if scope == "F" else {},
                seed=seed, scope=scope, objective=objective, pretrained_sha256=R.sha256(RADIMAGENET_DIR / "DenseNet121.pt")), checkpoint)
            audit["training_patient_aliases"] = aliases.tolist()
            audit["training_image_ids"] = df.image_id.iloc[rows].tolist()
            audit["checkpoint_sha256"] = R.sha256(checkpoint)
            pd.DataFrame(history).to_csv(run / "loss_history.csv", index=False)
            if mined:
                for triplet in mined:
                    for key in ("anchor", "positive", "negative"):
                        triplet[key] = aliases[triplet[key]]
                pd.DataFrame(mined).to_csv(run / "triplet_audit.csv", index=False)
            assert L.model_digest(prefix) == prefix_before and L.model_digest(original_tail) == tail_before
            audit["original_prefix_and_tail_unchanged"] = True
            R.write_json(run / "fit_audit.json", audit)
            paths = [feature_path, image_path, checkpoint, run / "loss_history.csv", run / "fit_audit.json"]
            if mined:
                paths.append(run / "triplet_audit.csv")
            R.write_json(run / "complete.json", dict(outputs={str(p.relative_to(OUT)): R.sha256(p) for p in paths}))
            del head, tail
            torch.cuda.empty_cache()
    print("All60 fixed-epoch training runs complete", flush=True)


def choose_candidate(metrics):
    means = metrics.groupby("group")[["auroc", "average_precision"]].mean()
    rows = []
    for group, (scope, objective) in CONFIGS.items():
        if objective == "CE":
            continue
        delta_dn = means.loc[group]-means.loc["DN-B"]
        delta_ce = means.loc[group]-means.loc[f"{scope}-CE"]
        rows.append(dict(group=group, eligible=bool((delta_dn >= -.02-1e-12).all() and (delta_ce >= -.02-1e-12).all()),
            rank_score=float(means.loc[group].mean()), delta_dn_auroc=float(delta_dn.auroc),
            delta_dn_ap=float(delta_dn.average_precision), delta_ce_auroc=float(delta_ce.auroc), delta_ce_ap=float(delta_ce.average_precision)))
    eligible = sorted([r for r in rows if r["eligible"]], key=lambda r: (-r["rank_score"],
        CONFIGS[r["group"]][0] != "P", CONFIGS[r["group"]][1] != "T", r["group"]))
    return dict(selected=eligible[0]["group"] if eligible else None, candidates=rows)


def promotion(delta):
    return dict(delta_auroc=float(delta.auroc.mean()), delta_ap=float(delta.average_precision.mean()),
        joint_wins=int(delta.gt(0).all(axis=1).sum()), promoted=bool(delta.auroc.mean() >= .03-1e-12 and
        delta.average_precision.mean() >= .02-1e-12 and delta.gt(0).all(axis=1).sum() >= 6))


def evaluate():
    require_lock()
    patients, split = read("patients.csv"), read("random_patient_assignments.csv")
    y = patients.label.to_numpy(int)
    historical = pd.read_csv(I.OUT / "patient_predictions_by_role.csv")
    native = pd.read_csv(N.OUT / "patient_predictions_by_role.csv")
    scores, records, predictions, fit_audits = {}, [], [], []
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            idx = N.indices_for(patients, split, seed)
            scores[seed] = {}
            for group in CONFIGS:
                run = OUT / "runs" / f"{group}_{seed}"
                assert (run / "complete.json").exists()
                x = np.load(OUT / "features" / f"{group}_{seed}.npy")
                head = N.full_head(seed).fit(x[idx["train"]], y[idx["train"]])
                score = head.predict_proba(x)[:, 1]
                path = OUT / "models" / f"{group}_{seed}_lr.joblib"
                joblib.dump(head, path, compress=3)
                assert np.allclose(joblib.load(path).predict_proba(x)[:, 1], score, atol=1e-12)
                assert np.allclose(head[0].statistics_, np.median(x[idx["train"]], axis=0))
                scores[seed][group] = score
                fit_audits.append(dict(seed=seed, group=group, train_patient_aliases=patients.patient_alias.iloc[idx["train"]].tolist(), sha256=R.sha256(path)))
            for group, source_group, source in (("RAD", "RAD", historical), ("DN-A", "DN-A", historical),
                    ("I3", "I3-direction-original", historical), ("DIRECTION-VOTE", "VOTE", historical), ("DN-B", "DN-B", native)):
                per = source[source.seed.eq(seed) & source.group.eq(source_group)]
                assert len(per) == 40 and per.patient_alias.nunique() == 40
                score = np.full(len(patients), np.nan)
                lookup = per.set_index("patient_alias")
                for role in ("validation", "test"):
                    aliases = patients.patient_alias.iloc[idx[role]]
                    assert np.array_equal(lookup.loc[aliases, "label"], y[idx[role]])
                    score[idx[role]] = lookup.loc[aliases, "score"]
                scores[seed][group] = score
            assert np.allclose(scores[seed]["DIRECTION-VOTE"][idx["validation"]],
                (scores[seed]["RAD"]+scores[seed]["DN-A"]+scores[seed]["I3"])[idx["validation"]]/3)
            for group, score in scores[seed].items():
                records.append(dict(seed=seed, group=group, role="development_validation",
                                    **V.score_metrics(y[idx["validation"]], score[idx["validation"]])))
    decision = choose_candidate(pd.DataFrame(records))
    winner = decision["selected"]
    if winner:
        for seed in R.SEEDS:
            ix = N.indices_for(patients, split, seed)["validation"]
            scores[seed]["REPLACEMENT-VOTE"] = (scores[seed]["RAD"]+scores[seed][winner]+scores[seed]["I3"])/3
            records.append(dict(seed=seed, group="REPLACEMENT-VOTE", role="development_validation",
                                **V.score_metrics(y[ix], scores[seed]["REPLACEMENT-VOTE"][ix])))
        md = pd.DataFrame(records).set_index(["group", "seed"])
        decision["vote"] = promotion(md.loc["REPLACEMENT-VOTE", ["auroc", "average_precision"]]-md.loc["DIRECTION-VOTE", ["auroc", "average_precision"]])
    decision["utc_before_test_metrics"] = datetime.now(timezone.utc).isoformat()
    R.write_json(OUT / "development_decision.json", decision)
    for seed in R.SEEDS:
        idx = N.indices_for(patients, split, seed)
        for group, score in scores[seed].items():
            records.append(dict(seed=seed, group=group, role="exploratory_test", **V.score_metrics(y[idx["test"]], score[idx["test"]])))
            for role, label in (("validation", "development_validation"), ("test", "exploratory_test")):
                for i in idx[role]:
                    predictions.append(dict(seed=seed, group=group, role=label,
                        patient_alias=patients.patient_alias.iloc[i], label=int(y[i]), score=float(score[i])))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "metrics_by_role_per_draw.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "patient_predictions_by_role.csv", index=False)
    summary = N.summarize(metrics)
    summary.to_csv(OUT / "summary_by_role.csv", index=False)
    R.write_json(OUT / "lr_fit_audit.json", fit_audits)
    print(summary[["role", "group", "auroc_mean", "average_precision_mean"]].to_string(index=False), flush=True)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "encode", "train", "evaluate", "all"))
    args = parser.parse_args()
    for stage in (("prepare", "encode", "train", "evaluate") if args.stage == "all" else (args.stage,)):
        globals()[stage]()
