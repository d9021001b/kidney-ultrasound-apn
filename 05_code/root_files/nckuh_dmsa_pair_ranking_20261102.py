"""Train-only early-stopped kidney classifier with within-patient ranking ablation."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from nckuh_dmsa_kidney_baseline_20261102 import (
    AUDIT, FEATURES, OUT as BASE, ROOT, SOURCE, SEEDS, load_reference,
)
from nckuh_only_study_20261102 import digest, metric, weighted_metrics_batch, write_json
from nckuh_l2sp_earlystop_20261102 import T, W, radimagenet_densenet121, RADIMAGENET_DIR

OUT = ROOT / "02_experiments/NCKUH_DMSA_pair_ranking_20261102"
CACHE = ROOT / "02_experiments/NCKUH_l2sp_earlystop_20261102"
PILOT_SEEDS = SEEDS[:3]
CONFIGS = [("frozen_BCE", False, False), ("frozen_BCE_rank", False, True),
           ("adapted_L2SP_BCE", True, False), ("adapted_L2SP_BCE_rank", True, True)]
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 80, 10, 15
REFERENCE = "Summary three source fusion"


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def pairs_within_patient(aliases, labels):
    aliases, labels = np.asarray(aliases), np.asarray(labels, dtype=int)
    positive, negative = [], []
    for alias in dict.fromkeys(aliases.tolist()):
        rows = np.flatnonzero(aliases == alias)
        if len(rows) > 2:
            raise ValueError("More than two anatomical kidneys for a patient")
        if len(rows) == 2 and set(labels[rows]) == {0, 1}:
            positive.append(int(rows[labels[rows] == 1][0]))
            negative.append(int(rows[labels[rows] == 0][0]))
    return np.asarray(positive, dtype=int), np.asarray(negative, dtype=int)


def ranking_loss(logits, positive, negative):
    if len(positive) == 0:
        return logits.new_zeros(())
    return F.softplus(logits[negative] - logits[positive]).mean()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve registered paired-ranking protocol")
    load_reference()
    cache_audit = json.loads((CACHE / "cache_audit.json").read_text(encoding="utf-8"))
    assert cache_audit["source_checkpoint_unchanged"]
    assert cache_audit["prefix_full_model_maxabs"] <= 1e-6
    assert digest(CACHE / "prefix_maps.npy") == cache_audit["prefix_cache_sha256"]
    paths = [Path(__file__), SOURCE, BASE / "patient_assignments.csv", BASE / "development_metrics.csv",
             BASE / "development_predictions.csv", AUDIT / "patient_label_linkage_private.csv",
             AUDIT / "kidney_label_linkage_private.csv", AUDIT / "image_labels_dmsa_private.csv",
             CACHE / "prefix_maps.npy", CACHE / "cache_audit.json", FEATURES / "dense_tight_image.npy",
             RADIMAGENET_DIR / "DenseNet121.pt", Path(T.__file__), Path(W.__file__)]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="Same-patient discordant kidney supervision may discourage patient/acquisition shortcuts and improve APN laterality discrimination",
        references=[dict(url="https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/", title="Burges et al. 2005 RankNet", scope="Pairwise logistic ranking loss, adapted to clinically confirmed unilateral pairs; not a reproduced renal-diagnosis model"),
                    dict(url="https://proceedings.mlr.press/v80/li18a.html", title="Li et al. 2018 L2-SP", scope="Anchor adapted pretrained parameters to original weights"),
                    dict(url="https://doi.org/10.1148/ryai.210315", title="RadImageNet", scope="Existing external medical encoder initialization")],
        cohort="210 source-code-available patients,414 imaged kidneys,820 images; source APN R/L/B/N;54 unknown outcomes excluded",
        configurations=CONFIGS, pilot_seeds=PILOT_SEEDS,
        reused_modules={"Projection_and_split_encoder": str(T.__file__), "l2sp_penalty": str(W.__file__)},
        pretrained="Existing frozen RadImageNet prefix through transition3; exact verified original float32(1027,512,7,7) maps, no new segmentation or CYCH disease checkpoint",
        model="Reuse linear1024-to128 L2-normalized image projection. Mean within kidney and renormalize, then a shared linear binary classifier. No opposite kidney diagnosis as inference input",
        loss="Class-balanced kidney BCE + optional .5*mean softplus(logit_APN_negative-logit_APN_positive) over training same-patient discordant pairs. B/N pairs have no ranking term. Adapted arms add .0001*sum((tail-original_tail)^2)",
        optimization="Adam head lr.001 weight_decay.0001; adapted denseblock4/norm5 affine lr.00001 decay0 with L2-SP. Gradient norm clip5. BatchNorm running buffers frozen in eval mode. Full training-image batch",
        inner_split="One100/26 patient random split inside each126-patient training role, stratified by R/L/B/N, seed+4000. All kidneys/views of a patient remain together",
        early_stopping="Maximize mean(inner26 kidney AUROC,AP), exact ties keep earliest epoch; max80epochs, minimum stopping10, patience15. Log inner BCE as a diagnostic, not selector. Reset original weights and identical seed, refit all126 training patients for selected epoch count",
        evaluation="Development42 patients only; no development epoch selection, no test scoring, no k-fold",
        expansion_gate="Three-draw mean AUROC>=.70, AP>=.78, gain>=.03AUROC and>=.02AP versus corrected summary fusion, with joint improvement>=2/3; target remains both>.8",
        ranking_audit="Report within-patient positive-over-negative accuracy and paired logit gaps separately from kidney AUROC/AP; not a substitute for the main endpoint",
        caveat="Within-patient supervision is an experimental adaptation, not proof of visible lesions or localization. Historical exploration and unresolved exact DMSA timing remain")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(protocol_sha256=digest(OUT / "protocol.json"),
        inputs={str(path): digest(path) for path in paths}))
    print("Registered four matched training-scope/ranking arms and three pilot draws before fitting", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == lock["protocol_sha256"]
    for path, sha in lock["inputs"].items():
        assert digest(Path(path)) == sha, path


def phase_data(images, kidneys, aliases, device):
    selected = kidneys[kidneys.patient_alias.isin(aliases)].reset_index(drop=True)
    rows = np.flatnonzero(images.kidney_alias.isin(selected.kidney_alias))
    bags = T.bag_matrix(images.kidney_alias.iloc[rows].to_numpy(), selected.kidney_alias.to_numpy()).to(device)
    y = torch.tensor(selected.kidney_APN.to_numpy(int), device=device, dtype=torch.float32)
    positive, negative = pairs_within_patient(selected.patient_alias.to_numpy(), y.cpu().numpy())
    return dict(rows=rows, bags=bags, y=y, positive=torch.tensor(positive, device=device),
                negative=torch.tensor(negative, device=device), kidneys=selected)


def fresh_network(original_tail, seed, adapted, device):
    torch.manual_seed(seed+37)
    head = T.Projection(1024).to(device)
    tail = copy.deepcopy(original_tail).to(device).eval().requires_grad_(adapted)
    anchor = {name: value.detach().clone() for name, value in tail.named_parameters()}
    parameters = [dict(params=head.parameters(), lr=.001, weight_decay=.0001)]
    if adapted:
        parameters.append(dict(params=tail.parameters(), lr=.00001, weight_decay=0.))
    return head, tail, anchor, torch.optim.Adam(parameters)


def logits_for(head, tail, maps, data, frozen_dense=None):
    features = frozen_dense[data["rows"]] if frozen_dense is not None else tail(maps[data["rows"]])
    embeddings = F.normalize(data["bags"] @ head(features), dim=1)
    return head.classifier(embeddings).squeeze(1)


def fit_phase(maps, dense, fit, inner, original_tail, seed, adapted, ranked, epochs=None):
    head, tail, anchor, optimizer = fresh_network(original_tail, seed, adapted, maps.device)
    buffers = {name: value.detach().clone() for name, value in tail.named_buffers()}
    y = fit["y"]
    weights = torch.where(y>0, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))
    history, best_epoch, best_score, bad = [], 1, -float('inf'), 0
    limit = epochs if epochs is not None else MAX_EPOCHS
    for epoch in range(1, limit+1):
        optimizer.zero_grad(set_to_none=True)
        logits = logits_for(head, tail, maps, fit, None if adapted else dense)
        bce = (F.binary_cross_entropy_with_logits(logits, y, reduction='none')*weights).mean()
        rank = ranking_loss(logits, fit['positive'], fit['negative']) if ranked else logits.new_zeros(())
        anchor_penalty = W.l2sp_penalty(tail, anchor, adapted)
        loss = bce + .5*rank + .0001*anchor_penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite kidney objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 5.)
        optimizer.step()
        row = dict(phase='inner' if inner is not None else 'refit', epoch=epoch,
                   train_bce=float(bce.detach()), rank_loss=float(rank.detach()), anchor_penalty=float(anchor_penalty.detach()))
        if inner is not None:
            with torch.inference_mode():
                validation_logits = logits_for(head, tail, maps, inner, None if adapted else dense)
                validation = F.binary_cross_entropy_with_logits(validation_logits, inner['y']).item()
                validation_scores = torch.sigmoid(validation_logits).cpu().numpy()
            validation_metrics = metric(inner['y'].cpu().numpy().astype(int), validation_scores)
            selection_score = (validation_metrics['auroc']+validation_metrics['ap'])/2
            row['inner_validation_bce'] = validation
            row['inner_validation_auroc'] = validation_metrics['auroc']
            row['inner_validation_ap'] = validation_metrics['ap']
            if selection_score > best_score + 1e-12:
                best_score, best_epoch, bad = selection_score, epoch, 0
            else:
                bad += 1
        history.append(row)
        if inner is not None and epoch>=MIN_EPOCHS and bad>=PATIENCE:
            break
    assert all(torch.equal(value, buffers[name]) for name, value in tail.named_buffers())
    displacement = float(W.l2sp_penalty(tail, anchor, True).detach())
    if not adapted:
        assert displacement == 0
    return head, tail, best_epoch if inner is not None else epochs, history, displacement


def pilot():
    from sklearn.model_selection import train_test_split
    configure(); require_lock()
    patients, kidneys, images = load_reference()
    assignments = pd.read_csv(BASE / 'patient_assignments.csv')
    maps = torch.from_numpy(np.load(CACHE / 'prefix_maps.npy')).cuda()
    dense = torch.from_numpy(np.load(FEATURES / 'dense_tight_image.npy')).cuda()
    model = radimagenet_densenet121().cuda().eval().requires_grad_(False)
    _, original_tail = T.split_encoder(model)
    original_digest = W.L.model_digest(original_tail)
    with torch.inference_mode():
        error = (original_tail(maps[:16])-dense[:16]).abs().max().item()
    assert error < 1e-4
    metrics, predictions, audits = [], [], []
    (OUT / 'models').mkdir(exist_ok=True)
    for group, adapted, ranked in CONFIGS:
        for seed in PILOT_SEEDS:
            checkpoint_result = OUT / f'result_{group}_{seed}.json'
            if checkpoint_result.exists():
                saved = json.loads(checkpoint_result.read_text(encoding='utf-8'))
                metrics.append(saved['metric']); predictions.extend(saved['predictions']); audits.append(saved['audit'])
                continue
            start = time.monotonic()
            mapping = assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
            outer = patients[patients.patient_alias.map(mapping).eq('train')]
            fit_alias, inner_alias = train_test_split(outer.patient_alias.to_numpy(), train_size=100,
                random_state=seed+4000, stratify=outer.DMSA_APN)
            dev_alias = patients.loc[patients.patient_alias.map(mapping).eq('development'),'patient_alias'].to_numpy()
            assert len(inner_alias)==26 and not set(fit_alias)&set(inner_alias) and not set(outer.patient_alias)&set(dev_alias)
            fit = phase_data(images, kidneys, fit_alias, maps.device)
            inner = phase_data(images, kidneys, inner_alias, maps.device)
            head, tail, epoch, history, _ = fit_phase(maps, dense, fit, inner, original_tail, seed, adapted, ranked)
            del head, tail
            full = phase_data(images, kidneys, outer.patient_alias.to_numpy(), maps.device)
            head, tail, _, refit, displacement = fit_phase(maps, dense, full, None, original_tail, seed, adapted, ranked, epoch)
            dev = phase_data(images, kidneys, dev_alias, maps.device)
            with torch.inference_mode():
                logits = logits_for(head, tail, maps, dev, None if adapted else dense)
                scores = torch.sigmoid(logits).cpu().numpy()
                positive, negative = dev['positive'], dev['negative']
                gap = (logits[positive]-logits[negative]).cpu().numpy()
                pair_accuracy = float(np.mean(gap>0)+.5*np.mean(gap==0)) if len(gap) else None
            record = dict(group=group, seed=seed, role='development', patients=42, kidneys=len(scores),
                selected_epoch=int(epoch), pair_order_accuracy=pair_accuracy, discordant_pairs=len(gap),
                **metric(dev['y'].cpu().numpy().astype(int), scores))
            rows = [dict(group=group, seed=seed, patient_alias=k.patient_alias, kidney_alias=k.kidney_alias,
                         DMSA_APN=k.DMSA_APN, side=k.side, label=int(k.kidney_APN), score=float(s))
                    for k,s in zip(dev['kidneys'].itertuples(index=False), scores)]
            audit = dict(group=group, seed=seed, fit_patients=100, inner_patients=26, full_refit_patients=126,
                selected_epoch=int(epoch), inner_epochs=len(history), training_discordant_pairs=len(full['positive']),
                fitted_tail_squared_displacement=displacement, original_tail_unchanged=W.L.model_digest(original_tail)==original_digest,
                fit_aliases=fit_alias.tolist(), inner_aliases=inner_alias.tolist(), elapsed_seconds=time.monotonic()-start)
            assert audit['original_tail_unchanged']
            torch.save(dict(head=head.state_dict(), tail=tail.state_dict(), adapted=adapted, ranked=ranked,
                            selected_epoch=int(epoch)), OUT/'models'/f'{group}_{seed}.pth')
            pd.DataFrame(history+refit).to_csv(OUT/f'history_{group}_{seed}.csv', index=False)
            write_json(checkpoint_result, dict(metric=record, predictions=rows, audit=audit))
            metrics.append(record); predictions.extend(rows); audits.append(audit)
            print(f"{group} {seed}: inner-selected epoch{epoch}, AUROC{record['auroc']:.6f} AP{record['ap']:.6f}, pairs{pair_accuracy}, seconds{audit['elapsed_seconds']:.1f}", flush=True)
            del head, tail
            torch.cuda.empty_cache()
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT/'pilot_metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT/'pilot_predictions.csv', index=False)
    write_json(OUT/'training_audits.json', audits)
    summary = frame.groupby('group')[['auroc','ap','pair_order_accuracy','selected_epoch']].mean()
    summary['mean_score'] = summary[['auroc','ap']].mean(axis=1)
    summary = summary.sort_values(['mean_score','group'], ascending=[False,True])
    reference = pd.read_csv(BASE/'development_metrics.csv')
    reference = reference[reference.group.eq(REFERENCE)&reference.seed.isin(PILOT_SEEDS)].set_index('seed')[['auroc','ap']]
    eligible = []
    for group in summary.index:
        delta = frame[frame.group.eq(group)].set_index('seed')[['auroc','ap']]-reference
        wins = int(delta.gt(0).all(axis=1).sum())
        summary.loc[group,'delta_auroc'] = delta.auroc.mean()
        summary.loc[group,'delta_ap'] = delta.ap.mean()
        summary.loc[group,'joint_wins'] = wins
        if summary.loc[group,'auroc']>=.70 and summary.loc[group,'ap']>=.78 and delta.auroc.mean()>=.03 and delta.ap.mean()>=.02 and wins>=2:
            eligible.append(group)
    summary.to_csv(OUT/'pilot_summary.csv')
    write_json(OUT/'pilot_decision.json', dict(selected_for_expansion=eligible[0] if eligible else None,
        eligible=eligible, test_scored=False, expansion_run=False, target_achieved=False))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','pilot'])
    globals()[parser.parse_args().stage]()
